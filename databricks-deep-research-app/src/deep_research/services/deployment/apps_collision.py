"""Owner-aware ``AlreadyExists`` resolution for Databricks Apps (Section S3).

When ``apps.create`` raises an ``AlreadyExists``-style error, the caller
needs to know whether it owns the existing app (and may therefore ``update``
it) or whether it collides with another user's app (and must pick a new name).

This module encapsulates that logic so it can be unit-tested in isolation and
reused by any future translator that creates Apps.

All SDK imports are **lazy** (inside the async function body) so tests can run
without a real Databricks SDK installation.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppOwnershipCheck:
    """Verdict returned by :func:`resolve_apps_already_exists`.

    Attributes
    ----------
    deployer_can_redeploy:
        ``True`` when the calling user is the creator or has CAN_MANAGE on the
        existing app.  The caller may safely call ``apps.update``.
    existing_owner:
        Email or principal-id of the app's creator if known; ``None`` if the
        SDK did not expose it or the app was already deleted (race).
    failure_reason:
        ``"race_deleted"`` when ``apps.get`` returned 404 (the app was deleted
        between the ``create`` and the ``get`` — caller should retry
        ``apps.create`` once).
        ``"permission_check_failed"`` when an SDK error prevented the ownership
        check — treat as a collision and ask the user to pick a different name.
        ``None`` when the check completed cleanly.
    """

    deployer_can_redeploy: bool
    existing_owner: str | None
    failure_reason: str | None  # "race_deleted" | "permission_check_failed" | None


# ---------------------------------------------------------------------------
# Helpers — kept private; exposed via the two public functions below.
# ---------------------------------------------------------------------------


def _is_not_found(exc: BaseException) -> bool:
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "notfound" in cls_name
        or "doesnotexist" in cls_name
        or "404" in msg
        or "not found" in msg
    )


def _is_permission_denied(exc: BaseException) -> bool:
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "permissiondenied" in cls_name
        or "forbidden" in cls_name
        or "unauthorized" in cls_name
        or "403" in msg
        or "permission denied" in msg
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def resolve_apps_already_exists(
    *,
    workspace_client: Any,
    app_name: str,
    deployer_email: str,
) -> AppOwnershipCheck:
    """Triage an ``AlreadyExists`` collision on ``apps.create``.

    Tries (in order):

    1. ``apps.get(name).creator == deployer_email`` → can redeploy.
    2. ``apps.permissions.get(object_type="apps", object_id=name)`` — search
       for a CAN_MANAGE / IS_OWNER entry matching the deployer → can redeploy.
    3. ``apps.get`` returned ``NotFound`` (race: deleted between create + get)
       → ``failure_reason="race_deleted"``.
    4. No MANAGE entry found → collision, ``deployer_can_redeploy=False``.

    Any unexpected SDK exception is caught and surfaced as
    ``failure_reason="permission_check_failed"`` so the caller can present a
    safe "pick a different name" message rather than crashing.

    Never raises.
    """
    # Step 1: try apps.get to read the creator field.
    def _do_get() -> Any:
        return workspace_client.apps.get(app_name)

    try:
        existing_app = await asyncio.to_thread(_do_get)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found(exc):
            # Race: deleted between our create attempt and this get.
            return AppOwnershipCheck(
                deployer_can_redeploy=False,
                existing_owner=None,
                failure_reason="race_deleted",
            )
        return AppOwnershipCheck(
            deployer_can_redeploy=False,
            existing_owner=None,
            failure_reason="permission_check_failed",
        )

    creator: str | None = getattr(existing_app, "creator", None)

    if creator and creator == deployer_email:
        return AppOwnershipCheck(
            deployer_can_redeploy=True,
            existing_owner=creator,
            failure_reason=None,
        )

    # Step 2: check permissions API for CAN_MANAGE / IS_OWNER.
    def _do_permissions() -> Any:
        return workspace_client.apps.permissions.get(
            object_type="apps",
            object_id=app_name,
        )

    try:
        perms = await asyncio.to_thread(_do_permissions)
        acl = getattr(perms, "access_control_list", None) or []
        for ace in acl:
            # The ACE structure varies slightly by SDK version; be defensive.
            principal = getattr(ace, "user_name", None) or getattr(ace, "group_name", None)
            if not principal:
                continue
            # Match by email for users; skip groups.
            if principal != deployer_email:
                continue
            for perm in getattr(ace, "all_permissions", []):
                level = str(getattr(perm, "permission_level", "")).upper()
                if level in ("CAN_MANAGE", "IS_OWNER"):
                    return AppOwnershipCheck(
                        deployer_can_redeploy=True,
                        existing_owner=creator,
                        failure_reason=None,
                    )
    except Exception:  # noqa: BLE001
        # Permissions API error → fail-closed (treat as collision).
        return AppOwnershipCheck(
            deployer_can_redeploy=False,
            existing_owner=creator,
            failure_reason="permission_check_failed",
        )

    # No matching MANAGE entry found → different owner.
    return AppOwnershipCheck(
        deployer_can_redeploy=False,
        existing_owner=creator,
        failure_reason=None,
    )


def generate_suggested_name(*, app_name: str, deployer_email: str) -> str:
    """Derive a deterministic collision-free name from the deployer's email.

    Pattern: ``"{app_name}-{slug}"`` truncated to the Apps name limit (30 chars).

    Slug derivation:
    - Take the local part of the email (before ``@``).
    - Lowercase and replace non-alphanumeric runs with ``-``.
    - Strip leading/trailing ``-`` and truncate to 10 chars.
    - If the result is empty (degenerate email), use the last 6 hex chars of
      the SHA-1 of the full email address.

    If the resulting candidate still exceeds 30 chars, ``app_name`` is trimmed
    from the right (preserving the ``dr-shell-`` prefix).  If that still does
    not fit, ``dr-shell-{slug}`` is used as the final fallback.
    """
    local_part = deployer_email.split("@")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", local_part.lower()).strip("-")[:10]

    if not slug:
        slug = hashlib.sha1(deployer_email.encode()).hexdigest()[:6]

    candidate = f"{app_name}-{slug}"
    if len(candidate) <= 30:
        return candidate

    # Trim app_name from the right while preserving the "dr-shell-" prefix.
    prefix = "dr-shell-"
    suffix = f"-{slug}"
    max_base = 30 - len(suffix)

    if app_name.startswith(prefix) and max_base >= len(prefix):
        trimmed_base = app_name[:max_base]
        candidate = f"{trimmed_base}{suffix}"
        if len(candidate) <= 30:
            return candidate

    # Final fallback: just prefix + slug (guaranteed <= 30 for slug <= 10).
    return f"{prefix}{slug}"
