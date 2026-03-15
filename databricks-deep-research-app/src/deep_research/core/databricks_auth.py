"""Centralized Databricks authentication with OAuth token management.

This module provides a single source of truth for Databricks authentication,
eliminating duplication across LLM client, embedder, and database modules.

Usage:
    from deep_research.core.databricks_auth import get_databricks_auth

    auth = get_databricks_auth()
    token = auth.get_token()       # Always fresh OAuth token
    url = auth.get_base_url()      # Serving endpoint URL
    client = auth.get_client()     # WorkspaceClient for special cases
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from databricks.sdk import WorkspaceClient

from deep_research.core.config import get_settings
from deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)

# OAuth token configuration
TOKEN_LIFETIME = timedelta(hours=1)
TOKEN_REFRESH_BUFFER = timedelta(minutes=5)

# Auth mode type
AuthMode = Literal["direct_token", "profile", "automatic"]


@dataclass
class OAuthCredential:
    """OAuth credential with expiration tracking."""

    token: str
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if token is expired or about to expire.

        Returns True when within TOKEN_REFRESH_BUFFER of expiration,
        allowing proactive refresh before actual expiry.
        """
        return datetime.now(UTC) >= (self.expires_at - TOKEN_REFRESH_BUFFER)


class DatabricksAuth:
    """Centralized Databricks authentication.

    Supports three auth modes (priority order):
    1. Direct token: DATABRICKS_TOKEN + DATABRICKS_HOST
    2. Profile OAuth: DATABRICKS_CONFIG_PROFILE from ~/.databrickscfg
    3. Automatic OAuth: Databricks Apps environment (service principal)

    This class manages:
    - WorkspaceClient creation with appropriate auth
    - OAuth token lifecycle with auto-refresh
    - Serving endpoint base URL derivation

    Example:
        auth = get_databricks_auth()
        token = auth.get_token()       # Always returns a fresh token
        url = auth.get_base_url()      # Serving endpoint URL
        client = auth.get_client()     # For Lakebase or other APIs
    """

    def __init__(self) -> None:
        """Initialize DatabricksAuth with appropriate auth mode."""
        settings = get_settings()
        self._settings = settings
        self._client: WorkspaceClient | None = None
        self._credential: OAuthCredential | None = None
        self._base_url: str | None = None

        # Determine auth mode once at init (priority order)
        if settings.databricks_token:
            self._auth_mode: AuthMode = "direct_token"
        elif settings.databricks_config_profile:
            self._auth_mode = "profile"
        elif settings.is_databricks_app:
            self._auth_mode = "automatic"
        else:
            raise ValueError(
                "No Databricks auth configured. Set one of:\n"
                "  - DATABRICKS_TOKEN + DATABRICKS_HOST (direct token)\n"
                "  - DATABRICKS_CONFIG_PROFILE (profile-based OAuth)\n"
                "  - Run as Databricks App (automatic OAuth)"
            )

        logger.info("DATABRICKS_AUTH_INIT", mode=self._auth_mode)

    def get_client(self) -> WorkspaceClient:
        """Get WorkspaceClient instance (creates if needed).

        Returns:
            Configured WorkspaceClient with appropriate authentication.
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> WorkspaceClient:
        """Create WorkspaceClient based on determined auth mode.

        Returns:
            New WorkspaceClient instance.
        """
        if self._auth_mode == "direct_token":
            logger.debug("WORKSPACE_CLIENT_CREATE", mode="direct_token")
            return WorkspaceClient(
                host=self._settings.databricks_host,
                token=self._settings.databricks_token,
            )
        elif self._auth_mode == "profile":
            logger.debug(
                "WORKSPACE_CLIENT_CREATE",
                mode="profile",
                profile=self._settings.databricks_config_profile,
            )
            return WorkspaceClient(
                profile=self._settings.databricks_config_profile
            )
        else:  # automatic
            logger.debug("WORKSPACE_CLIENT_CREATE", mode="automatic")
            return WorkspaceClient()

    def get_token(self, force_refresh: bool = False) -> str:
        """Get valid OAuth token, refreshing if needed.

        For direct token mode, returns the static token.
        For OAuth modes, manages token lifecycle with auto-refresh.

        Args:
            force_refresh: Force credential refresh even if not expired.

        Returns:
            Valid OAuth/access token.
        """
        # Direct token mode: no refresh needed
        if self._auth_mode == "direct_token":
            token = self._settings.databricks_token
            if not token:
                raise ValueError("DATABRICKS_TOKEN is not set")
            return token

        # OAuth modes: check expiration and refresh if needed
        if (
            self._credential is None
            or self._credential.is_expired
            or force_refresh
        ):
            self._credential = self._generate_credential()

        return self._credential.token

    def _generate_credential(self) -> OAuthCredential:
        """Generate credential via WorkspaceClient.

        Uses authenticate() which works for ALL auth types:
        - PAT auth (direct token)
        - OAuth profile auth
        - Databricks Apps automatic auth (service principal)

        Returns:
            Fresh credential with token extracted from Authorization header.

        Raises:
            RuntimeError: If token extraction fails.
        """
        client = self.get_client()
        headers = client.config.authenticate()

        # Extract token from 'Authorization: Bearer <token>' header
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.error(
                "AUTH_HEADER_INVALID",
                mode=self._auth_mode,
                header_prefix=auth_header[:20] if auth_header else "empty",
            )
            raise RuntimeError(
                f"Unexpected auth header format: {auth_header[:20] if auth_header else 'empty'}..."
            )

        token = auth_header.removeprefix("Bearer ")
        if not token:
            logger.error("AUTH_TOKEN_EMPTY", mode=self._auth_mode)
            raise RuntimeError("Empty token in Authorization header")

        expires_at = datetime.now(UTC) + TOKEN_LIFETIME

        logger.debug(
            "CREDENTIAL_GENERATED",
            mode=self._auth_mode,
            expires_at=expires_at.isoformat(),
        )

        return OAuthCredential(token=token, expires_at=expires_at)

    def get_base_url(self) -> str:
        """Get Databricks serving endpoint base URL.

        Returns:
            URL in format: https://<host>/serving-endpoints
        """
        if self._base_url is None:
            if self._auth_mode == "direct_token":
                host = self._settings.databricks_host
                if not host:
                    raise ValueError("DATABRICKS_HOST is not set")
                self._base_url = f"{host}/serving-endpoints"
            else:
                # OAuth modes: get host from WorkspaceClient
                client = self.get_client()
                self._base_url = f"{client.config.host}/serving-endpoints"

            logger.debug("BASE_URL_RESOLVED", url=self._base_url)

        return self._base_url

    @property
    def auth_mode(self) -> AuthMode:
        """Current authentication mode."""
        return self._auth_mode

    @property
    def is_oauth(self) -> bool:
        """Check if using OAuth-based authentication (profile or automatic)."""
        return self._auth_mode in ("profile", "automatic")


# Singleton cache
_auth_instance: DatabricksAuth | None = None


def get_databricks_auth() -> DatabricksAuth:
    """Get singleton DatabricksAuth instance.

    Returns:
        Shared DatabricksAuth instance.
    """
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = DatabricksAuth()
    return _auth_instance


def clear_databricks_auth() -> None:
    """Clear the singleton auth instance.

    Useful for testing or when settings change.
    """
    global _auth_instance
    _auth_instance = None
