"""Tests for auth middleware exception handling with Databricks SDK errors."""

from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import DatabricksError, PermissionDenied, Unauthenticated
from fastapi import HTTPException

from deep_research.core.auth import UserIdentity
from deep_research.middleware.auth import get_current_user_identity


def _mock_request(headers=None):
    """Create a mock FastAPI Request."""
    request = MagicMock()
    request.headers = headers or {}
    request.state = MagicMock()
    return request


def _mock_settings(is_production=False):
    """Create mock Settings.

    User-sync is disabled here because these tests target OBO/SP/anon
    auth resolution, not the synchronous user-sync path that runs after
    a successful resolve. With user_sync_enabled=False the new
    ``_ensure_user_synced`` short-circuits before touching any backend.
    """
    settings = MagicMock()
    settings.is_production = is_production
    settings.user_sync_enabled = False
    return settings


def _sp_user():
    return UserIdentity(user_id="sp-123", email="sp@test.com", display_name="SP")


@pytest.mark.asyncio
async def test_obo_unauthenticated_falls_through_to_sp():
    """Unauthenticated from expired OBO token should fall through to SP auth."""
    request = _mock_request(headers={"x-forwarded-access-token": "expired-token"})
    settings = _mock_settings(is_production=False)

    with (
        patch("deep_research.core.auth.extract_obo_token", return_value="expired-token"),
        patch("deep_research.core.auth.get_user_workspace_client") as mock_user_client,
        patch("deep_research.middleware.auth.get_workspace_client") as mock_ws,
        patch("deep_research.middleware.auth.get_current_user", return_value=_sp_user()),
    ):
        # OBO raises Unauthenticated
        client_mock = MagicMock()
        client_mock.current_user.me.side_effect = Unauthenticated("Token expired")
        mock_user_client.return_value = client_mock

        mock_ws.return_value = MagicMock()

        user = await get_current_user_identity(request, settings)

    # Should have fallen through to SP auth
    assert user.user_id == "sp-123"


@pytest.mark.asyncio
async def test_obo_permission_denied_falls_through_to_sp():
    """PermissionDenied during OBO should fall through to SP auth."""
    request = _mock_request(headers={"x-forwarded-access-token": "limited-token"})
    settings = _mock_settings(is_production=False)

    with (
        patch("deep_research.core.auth.extract_obo_token", return_value="limited-token"),
        patch("deep_research.core.auth.get_user_workspace_client") as mock_user_client,
        patch("deep_research.middleware.auth.get_workspace_client") as mock_ws,
        patch("deep_research.middleware.auth.get_current_user", return_value=_sp_user()),
    ):
        client_mock = MagicMock()
        client_mock.current_user.me.side_effect = PermissionDenied("No access")
        mock_user_client.return_value = client_mock

        mock_ws.return_value = MagicMock()

        user = await get_current_user_identity(request, settings)

    assert user.user_id == "sp-123"


@pytest.mark.asyncio
async def test_sp_databricks_error_falls_to_anonymous_in_dev():
    """DatabricksError in SP auth should fall through to anonymous in dev mode."""
    request = _mock_request()
    settings = _mock_settings(is_production=False)

    with (
        patch("deep_research.core.auth.extract_obo_token", return_value=None),
        patch("deep_research.middleware.auth.get_workspace_client", side_effect=DatabricksError("Config error")),
    ):
        user = await get_current_user_identity(request, settings)

    assert user.user_id == "anonymous"


@pytest.mark.asyncio
async def test_sp_databricks_error_raises_401_in_production():
    """DatabricksError in SP auth should raise 401 in production."""
    request = _mock_request()
    settings = _mock_settings(is_production=True)

    with (
        patch("deep_research.core.auth.extract_obo_token", return_value=None),
        patch("deep_research.middleware.auth.get_workspace_client", side_effect=DatabricksError("Config error")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_identity(request, settings)

        assert exc_info.value.status_code == 401
