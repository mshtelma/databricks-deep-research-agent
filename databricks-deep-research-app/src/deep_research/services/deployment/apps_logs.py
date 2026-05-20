"""Best-effort app log tail fetcher for the Deploy Here feature (Section S4b).

Used when the reachability probe times out — fetches a redacted snippet of
the Databricks App's deployment/status logs so the frontend can show the user
what went wrong without requiring them to open the Databricks UI.

Returns ``None`` rather than raising so the caller can gracefully omit log
data from the response.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Secret-pattern redaction (applied before returning any log text)
# ---------------------------------------------------------------------------

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdapi[a-f0-9]{28,}\b"),  # Databricks PATs
    re.compile(r"\bxoxb-[A-Za-z0-9-]+\b"),  # Slack bot tokens
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),  # GitHub personal access tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key IDs
    re.compile(r"(?i)(api[-_]?key|password|secret)\s*[:=]\s*\S+"),
)


def _redact(text: str) -> str:
    """Replace all recognized secret patterns with ``***REDACTED***``."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("***REDACTED***", text)
    return text


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppLogTail:
    """A redacted, truncated snippet of a Databricks App's deployment logs.

    Attributes
    ----------
    text:
        The log text (possibly empty).
    truncated:
        ``True`` when ``max_lines`` or ``max_bytes`` forced truncation.
    source:
        Which SDK method was used to obtain the log text — useful for triage.
    """

    text: str
    truncated: bool
    source: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_app_log_tail(
    *,
    workspace_client: Any,
    app_name: str,
    max_lines: int = 50,
    max_bytes: int = 5000,
) -> AppLogTail | None:
    """Fetch a redacted tail of recent deployment logs for ``app_name``.

    Two SDK fallback paths (tried in order):

    1. ``workspace_client.apps.list_app_deployments(app_name)`` → most recent
       deployment → ``apps.get_app_deployment(app_name, deployment_id).status_message``.
       Source tag: ``"app_deployment_status_message"``.
    2. ``workspace_client.apps.get(app_name)`` → concatenate
       ``pending_deployment.status_message`` and ``compute_status.message``.
       Source tag: ``"app_status_messages"``.

    Returns ``None`` when both paths fail or produce no text.  All SDK calls
    are individually wrapped in ``try/except``; partial failures fall through
    to the next path.
    """
    raw: str | None = None
    source: str = "unknown"

    # ------------------------------------------------------------------
    # Path 1: list_app_deployments → get_app_deployment.status_message
    # ------------------------------------------------------------------
    try:
        def _list_deployments() -> Any:
            return workspace_client.apps.list_app_deployments(app_name)

        deployments_iter = await asyncio.to_thread(_list_deployments)
        deployments = list(deployments_iter) if deployments_iter is not None else []

        if deployments:
            # Sort by creation time if available; otherwise take the last item.
            most_recent = deployments[-1]
            for dep in deployments:
                dep_ts = getattr(dep, "create_time", None) or getattr(dep, "created_at", None)
                cur_ts = getattr(most_recent, "create_time", None) or getattr(most_recent, "created_at", None)
                if dep_ts and cur_ts and dep_ts > cur_ts:
                    most_recent = dep

            deployment_id = (
                getattr(most_recent, "deployment_id", None)
                or getattr(most_recent, "id", None)
            )

            if deployment_id is not None:
                def _get_deployment(dep_id: Any = deployment_id) -> Any:
                    return workspace_client.apps.get_app_deployment(app_name, dep_id)

                dep_detail = await asyncio.to_thread(_get_deployment)
                status_obj = getattr(dep_detail, "status", None)
                msg = getattr(status_obj, "message", None) if status_obj else None
                if not msg:
                    # Some SDK versions put message directly on the deployment.
                    msg = getattr(dep_detail, "status_message", None)
                if msg:
                    raw = str(msg)
                    source = "app_deployment_status_message"
    except Exception:  # noqa: BLE001
        pass  # Fall through to path 2.

    # ------------------------------------------------------------------
    # Path 2: apps.get → pending_deployment + compute_status messages
    # ------------------------------------------------------------------
    if raw is None:
        try:
            def _get_app() -> Any:
                return workspace_client.apps.get(app_name)

            app_obj = await asyncio.to_thread(_get_app)
            parts: list[str] = []

            pending_dep = getattr(app_obj, "pending_deployment", None)
            if pending_dep is not None:
                pd_status = getattr(pending_dep, "status", None)
                pd_msg = getattr(pd_status, "message", None) if pd_status else None
                if not pd_msg:
                    pd_msg = getattr(pending_dep, "status_message", None)
                if pd_msg:
                    parts.append(str(pd_msg))

            compute_status = getattr(app_obj, "compute_status", None)
            if compute_status is not None:
                cs_msg = getattr(compute_status, "message", None)
                if cs_msg:
                    parts.append(str(cs_msg))

            if parts:
                raw = "\n".join(parts)
                source = "app_status_messages"
        except Exception:  # noqa: BLE001
            pass

    if raw is None:
        return None

    # ------------------------------------------------------------------
    # Redact secrets and apply line / byte limits
    # ------------------------------------------------------------------
    redacted = _redact(raw)
    lines = redacted.splitlines()
    truncated = False

    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated = True

    text = "\n".join(lines)
    if len(text.encode()) > max_bytes:
        # Trim bytes from the start (oldest lines) until we're under the limit.
        encoded = text.encode()
        encoded = encoded[-max_bytes:]
        # Decode safely, discarding any partial multi-byte char at the boundary.
        text = encoded.decode("utf-8", errors="ignore")
        truncated = True

    if not text.strip():
        return None

    return AppLogTail(text=text, truncated=truncated, source=source)
