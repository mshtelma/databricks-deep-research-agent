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

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from databricks.sdk import WorkspaceClient

from deep_research.core.config import get_settings
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from fastapi import Request

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

    def invalidate(self) -> None:
        """Drop cached credential, WorkspaceClient, and base URL.

        Forces the next get_token() / get_client() / get_base_url() call to
        rebuild everything from scratch. Defends against a poisoned SDK-side
        token cache (e.g., when ``databricks auth login`` ran in another
        shell and the on-disk token cache rotated).

        Concurrent invocation is idempotent: each writer sets fields to None,
        the next reader rebuilds. Worst case is a redundant rebuild.

        Effects on shared consumers of this singleton:
          * LLM client : next request mints a fresh bearer (the point of this).
          * Lakebase   : its credential provider lazily reads get_client();
                         the next read rebuilds the WorkspaceClient (+~50-200ms
                         one-time). No correctness impact.
          * Embedder   : same path as LLM client.

        Distinct from ``clear_databricks_auth()`` (module-level) which drops
        the singleton itself — used by tests when settings change.
        """
        logger.warning("DATABRICKS_AUTH_INVALIDATE", mode=self._auth_mode)
        self._credential = None
        self._client = None
        self._base_url = None


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


def get_user_workspace_client(request: "Request") -> WorkspaceClient:
    """Build a request-scoped WorkspaceClient using the user's OBO token.

    This client is request-scoped and MUST NOT be cached across requests.
    Each call returns a fresh client built from the current request's OBO header.

    The OBO (On-Behalf-Of) token is injected by Databricks Apps as the
    ``X-Forwarded-Access-Token`` header.  This function is intentionally NOT
    cached — each invocation builds a new ``WorkspaceClient`` from the token
    present on the current request.

    Host resolution priority:
      1. ``DATABRICKS_HOST`` environment variable.
      2. Host from the singleton SP client (``get_databricks_auth().get_client()``).
      3. Derived from ``request.url`` — only used for local dev when neither of
         the above is available.

    Args:
        request: The current FastAPI ``Request`` object.

    Returns:
        A fresh ``WorkspaceClient`` authenticated with the user's OBO token.

    Raises:
        HTTPException: 401 with ``error_kind='missing_obo_token'`` when running
            inside Databricks Apps (``settings.is_databricks_app`` is True) and
            the ``X-Forwarded-Access-Token`` header is absent.  This prevents
            silent SP fallback in production.
    """
    from fastapi import HTTPException

    settings = get_settings()

    obo_token: str | None = request.headers.get("X-Forwarded-Access-Token")

    # Diagnostic log: fires at the entry of every call, even before any guards
    # or branches. If this log is absent in production, the function is not
    # being invoked from the route handler — narrows the search to "request
    # is not reaching FastAPI" vs "function is failing inside".
    logger.info(
        "OBO_WC_ENTRY",
        path=str(request.url.path),
        has_obo_header=obo_token is not None,
        obo_header_len=len(obo_token) if obo_token else 0,
        is_databricks_app=settings.is_databricks_app,
    )

    # T2 fix: treat empty-string header the same as a missing header.
    # `request.headers.get(...)` returns "" if the header is present but
    # blank, which would otherwise sail past the `is None` check and reach
    # the SDK with `token=""`, surfacing an opaque downstream error.
    if not obo_token:
        if settings.is_databricks_app:
            raise HTTPException(
                status_code=401,
                detail={
                    "error_kind": "missing_obo_token",
                    "message": (
                        "OBO header X-Forwarded-Access-Token is required"
                        " when running in Databricks Apps."
                    ),
                },
            )
        # Local dev: fall back to SP client and log a warning.
        logger.warning("OBO_HEADER_MISSING_LOCAL_DEV_FALLBACK_SP")
        return get_databricks_auth().get_client()

    # Resolve host.
    host: str | None = os.environ.get("DATABRICKS_HOST")
    if not host:
        try:
            host = get_databricks_auth().get_client().config.host
        except Exception:
            host = None
    if not host:
        host = str(request.url.scheme) + "://" + str(request.url.netloc)

    # Structured per-request observability. `actor` matches the vocabulary
    # used by the CanDeployHereResponse schema so logs and API responses
    # share one terminology.
    logger.info(
        "OBO_WC_CONSTRUCT",
        host=host,
        has_obo=bool(obo_token),
        env_oauth_set=bool(os.environ.get("DATABRICKS_CLIENT_ID")),
        actor="obo",
    )

    # T2 fix: `auth_type="pat"` tells the Databricks SDK to ignore any
    # OAuth-m2m credentials that the Databricks Apps runtime sets via
    # `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` env vars. Without
    # this kwarg, the SDK Config detects BOTH the explicit kwarg `token`
    # (pat) AND the env-OAuth (oauth-m2m) and raises
    # `ValueError: more than one authorization method configured`.
    # Mirrors the existing canonical pattern at core/auth.py:99.
    return WorkspaceClient(host=host, token=obo_token, auth_type="pat")
