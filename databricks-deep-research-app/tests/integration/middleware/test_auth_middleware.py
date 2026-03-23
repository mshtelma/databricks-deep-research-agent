"""Integration tests for OBO authentication middleware flow."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

from deep_research.core.auth import UserIdentity
from deep_research.middleware.auth import get_current_user_identity


@pytest.mark.integration
class TestOboAuthenticationMiddleware:
    """Integration tests for OBO token-based authentication in middleware."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create a mock FastAPI request."""
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        return request

    @pytest.fixture
    def mock_settings_production(self) -> MagicMock:
        """Create mock settings for production mode."""
        settings = MagicMock()
        settings.is_production = True
        return settings

    @pytest.fixture
    def mock_settings_development(self) -> MagicMock:
        """Create mock settings for development mode."""
        settings = MagicMock()
        settings.is_production = False
        return settings

    @pytest.mark.asyncio
    @patch("deep_research.core.auth.get_user_workspace_client")
    @patch("deep_research.core.auth.extract_obo_token")
    @patch("deep_research.middleware.auth.get_workspace_client")
    async def test_obo_token_resolves_user_identity(
        self,
        mock_get_sp_client: MagicMock,
        mock_extract_obo: MagicMock,
        mock_get_user_client: MagicMock,
        mock_request: MagicMock,
        mock_settings_production: MagicMock,
    ) -> None:
        """When x-forwarded-access-token present, should use it for user identity."""
        # Setup OBO token extraction
        mock_extract_obo.return_value = "user-oauth-token"

        # Setup user client to return actual user
        mock_user_client = MagicMock()
        mock_workspace_user = MagicMock()
        mock_workspace_user.id = 12345
        mock_workspace_user.user_name = "realuser@company.com"
        mock_workspace_user.display_name = "Real User"
        mock_user_client.current_user.me.return_value = mock_workspace_user
        mock_get_user_client.return_value = mock_user_client

        # Setup SP client for backend operations
        mock_sp_client = MagicMock()
        mock_get_sp_client.return_value = mock_sp_client

        # Execute
        result = await get_current_user_identity(mock_request, mock_settings_production)

        # Verify
        assert result.user_id == "12345"
        assert result.email == "realuser@company.com"
        assert result.display_name == "Real User"
        mock_get_user_client.assert_called_once_with("user-oauth-token")
        # SP client should be stored for backend operations
        assert mock_request.state.workspace_client == mock_sp_client

    @pytest.mark.asyncio
    @patch("deep_research.middleware.auth.get_workspace_client")
    @patch("deep_research.middleware.auth.get_current_user")
    @patch("deep_research.core.auth.extract_obo_token")
    async def test_falls_back_to_sp_when_no_obo_token(
        self,
        mock_extract_obo: MagicMock,
        mock_get_current_user: MagicMock,
        mock_get_sp_client: MagicMock,
        mock_request: MagicMock,
        mock_settings_production: MagicMock,
    ) -> None:
        """When OBO token absent, should fall back to service principal."""
        # No OBO token
        mock_extract_obo.return_value = None

        # SP auth returns service principal identity
        mock_sp_client = MagicMock()
        mock_get_sp_client.return_value = mock_sp_client
        mock_get_current_user.return_value = UserIdentity(
            user_id="sp-12345",
            email="service-principal@databricks.com",
            display_name="Service Principal",
        )

        # Execute
        result = await get_current_user_identity(mock_request, mock_settings_production)

        # Verify falls back to SP
        assert result.user_id == "sp-12345"
        assert result.email == "service-principal@databricks.com"

    @pytest.mark.asyncio
    @patch("deep_research.core.auth.get_user_workspace_client")
    @patch("deep_research.core.auth.extract_obo_token")
    @patch("deep_research.middleware.auth.get_workspace_client")
    @patch("deep_research.middleware.auth.get_current_user")
    async def test_falls_back_to_sp_when_obo_fails(
        self,
        mock_get_current_user: MagicMock,
        mock_get_sp_client: MagicMock,
        mock_extract_obo: MagicMock,
        mock_get_user_client: MagicMock,
        mock_request: MagicMock,
        mock_settings_production: MagicMock,
    ) -> None:
        """When OBO token fails, should fall back to service principal."""
        # OBO token present but fails
        mock_extract_obo.return_value = "invalid-token"
        mock_user_client = MagicMock()
        mock_user_client.current_user.me.side_effect = Exception("Token expired")
        mock_get_user_client.return_value = mock_user_client

        # SP auth succeeds
        mock_sp_client = MagicMock()
        mock_get_sp_client.return_value = mock_sp_client
        mock_get_current_user.return_value = UserIdentity(
            user_id="sp-fallback",
            email="fallback@databricks.com",
            display_name="Fallback SP",
        )

        # Execute
        result = await get_current_user_identity(mock_request, mock_settings_production)

        # Verify falls back to SP
        assert result.user_id == "sp-fallback"

    @pytest.mark.asyncio
    @patch("deep_research.middleware.auth.get_workspace_client")
    @patch("deep_research.core.auth.extract_obo_token")
    async def test_anonymous_fallback_in_development(
        self,
        mock_extract_obo: MagicMock,
        mock_get_sp_client: MagicMock,
        mock_request: MagicMock,
        mock_settings_development: MagicMock,
    ) -> None:
        """In development mode, falls back to anonymous when all auth fails."""
        # No OBO token
        mock_extract_obo.return_value = None
        # SP auth fails
        mock_get_sp_client.side_effect = Exception("No credentials")

        # Execute
        result = await get_current_user_identity(mock_request, mock_settings_development)

        # Verify anonymous fallback
        assert result.user_id == "anonymous"
        assert result.email == "anonymous@local.dev"

    @pytest.mark.asyncio
    @patch("deep_research.middleware.auth.get_workspace_client")
    @patch("deep_research.core.auth.extract_obo_token")
    async def test_raises_401_in_production_when_all_auth_fails(
        self,
        mock_extract_obo: MagicMock,
        mock_get_sp_client: MagicMock,
        mock_request: MagicMock,
        mock_settings_production: MagicMock,
    ) -> None:
        """In production mode, raises 401 when all auth methods fail."""
        from fastapi import HTTPException

        # No OBO token
        mock_extract_obo.return_value = None
        # SP auth fails
        mock_get_sp_client.side_effect = Exception("No credentials")

        # Execute and verify
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_identity(mock_request, mock_settings_production)

        assert exc_info.value.status_code == 401
        assert "Authentication failed" in str(exc_info.value.detail)
