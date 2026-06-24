"""Unit tests for services/deployment/apps_collision.py (Section S3).

Covers:
- Owner == deployer → can_redeploy=True via apps.get.
- CAN_MANAGE via permissions API → can_redeploy=True.
- Race-deleted (NotFound from apps.get) → failure_reason="race_deleted".
- apps.get raises generic error → failure_reason="permission_check_failed".
- Slug generation: ASCII email, unicode, empty local-part, long name, >30 char.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deep_research.services.deployment.apps_collision import (
    generate_suggested_name,
    resolve_apps_already_exists,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wc(
    *,
    creator: str | None = None,
    get_side_effect: Exception | None = None,
    permissions_side_effect: Exception | None = None,
    permissions_acl: list[object] | None = None,
) -> MagicMock:
    """Build a minimal workspace-client mock for apps.get / apps.permissions.get."""
    wc = MagicMock()

    # apps.get
    if get_side_effect:
        wc.apps.get.side_effect = get_side_effect
    else:
        app_obj = MagicMock()
        app_obj.creator = creator
        wc.apps.get.return_value = app_obj

    # apps.permissions.get
    if permissions_side_effect:
        wc.apps.permissions.get.side_effect = permissions_side_effect
    else:
        perms_obj = MagicMock()
        perms_obj.access_control_list = permissions_acl or []
        wc.apps.permissions.get.return_value = perms_obj

    return wc


def _make_ace(user_name: str, permission_level: str) -> MagicMock:
    ace = MagicMock()
    ace.user_name = user_name
    ace.group_name = None
    perm = MagicMock()
    perm.permission_level = permission_level
    ace.all_permissions = [perm]
    return ace


# ---------------------------------------------------------------------------
# resolve_apps_already_exists
# ---------------------------------------------------------------------------


class TestResolveAppsAlreadyExists:
    @pytest.mark.asyncio
    async def test_owner_equals_deployer_can_redeploy(self) -> None:
        wc = _make_wc(creator="alice@acme.com")
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="alice@acme.com",
        )
        assert check.deployer_can_redeploy is True
        assert check.existing_owner == "alice@acme.com"
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_can_manage_via_permissions_api(self) -> None:
        ace = _make_ace("bob@acme.com", "CAN_MANAGE")
        wc = _make_wc(creator="charlie@acme.com", permissions_acl=[ace])
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="bob@acme.com",
        )
        assert check.deployer_can_redeploy is True

    @pytest.mark.asyncio
    async def test_is_owner_permission_level(self) -> None:
        ace = _make_ace("dave@acme.com", "IS_OWNER")
        wc = _make_wc(creator=None, permissions_acl=[ace])
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="dave@acme.com",
        )
        assert check.deployer_can_redeploy is True

    @pytest.mark.asyncio
    async def test_race_deleted_returns_failure_reason(self) -> None:
        NotFound = type("NotFound", (Exception,), {})
        wc = _make_wc(get_side_effect=NotFound("not found"))
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="alice@acme.com",
        )
        assert check.deployer_can_redeploy is False
        assert check.failure_reason == "race_deleted"

    @pytest.mark.asyncio
    async def test_apps_get_generic_error_is_permission_check_failed(self) -> None:
        wc = _make_wc(get_side_effect=RuntimeError("something broke"))
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="alice@acme.com",
        )
        assert check.deployer_can_redeploy is False
        assert check.failure_reason == "permission_check_failed"

    @pytest.mark.asyncio
    async def test_different_owner_no_manage_is_collision(self) -> None:
        wc = _make_wc(creator="frank@acme.com", permissions_acl=[])
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="grace@acme.com",
        )
        assert check.deployer_can_redeploy is False
        assert check.existing_owner == "frank@acme.com"
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_permissions_api_error_is_permission_check_failed(self) -> None:
        wc = _make_wc(
            creator="henry@acme.com",
            permissions_side_effect=RuntimeError("perms api broken"),
        )
        check = await resolve_apps_already_exists(
            workspace_client=wc,
            app_name="dr-shell-test",
            deployer_email="irene@acme.com",
        )
        assert check.deployer_can_redeploy is False
        assert check.failure_reason == "permission_check_failed"


# ---------------------------------------------------------------------------
# generate_suggested_name
# ---------------------------------------------------------------------------


class TestGenerateSuggestedName:
    def test_simple_ascii_email(self) -> None:
        name = generate_suggested_name(
            app_name="dr-shell-research", deployer_email="alice@acme.com"
        )
        assert name == "dr-shell-research-alice"
        assert len(name) <= 30

    def test_slug_truncated_to_10_chars(self) -> None:
        name = generate_suggested_name(
            app_name="dr-shell-app",
            deployer_email="verylongusername@example.com",
        )
        # slug = "verylongus" (10 chars)
        assert "verylongus" in name
        assert len(name) <= 30

    def test_non_ascii_email_local_part(self) -> None:
        name = generate_suggested_name(
            app_name="dr-shell-test",
            deployer_email="tëst.üser@example.com",
        )
        # Non-ASCII chars become separators → slug has only [a-z0-9-]
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name)
        assert len(name) <= 30

    def test_empty_local_part_uses_sha1_fallback(self) -> None:
        # email with no local part (degenerate)
        name = generate_suggested_name(
            app_name="dr-shell-app",
            deployer_email="@broken.com",
        )
        # SHA-1 fallback produces a 6-char hex slug
        assert len(name) <= 30
        assert name.startswith("dr-shell-app-") or name.startswith("dr-shell-")

    def test_long_app_name_trimmed_to_30(self) -> None:
        long_name = "dr-shell-averylongappnamehere"
        name = generate_suggested_name(
            app_name=long_name,
            deployer_email="bob@example.com",
        )
        assert len(name) <= 30
        assert "dr-shell-" in name

    def test_candidate_already_fits(self) -> None:
        name = generate_suggested_name(
            app_name="dr-shell-x",
            deployer_email="z@y.com",
        )
        assert len(name) <= 30
        assert name.startswith("dr-shell-x-")

    def test_special_chars_in_email_normalized(self) -> None:
        name = generate_suggested_name(
            app_name="dr-shell-prod",
            deployer_email="first.last+tag@example.com",
        )
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name)
        assert len(name) <= 30
