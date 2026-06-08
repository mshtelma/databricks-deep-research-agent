"""Databricks authentication utilities."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from databricks.sdk import WorkspaceClient

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from databricks.sdk.service.iam import User

from deep_research.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserIdentity:
    """User identity extracted from Databricks authentication."""

    user_id: str
    email: str
    display_name: str

    @classmethod
    def from_workspace_user(cls, user: "User") -> "UserIdentity":
        """Create from Databricks workspace user object."""
        return cls(
            user_id=str(user.id) if user.id else (user.user_name or "unknown"),
            email=user.user_name or "",
            display_name=user.display_name or user.user_name or "Unknown User",
        )

    @classmethod
    def anonymous(cls) -> "UserIdentity":
        """Create anonymous user for development/testing."""
        return cls(
            user_id="anonymous",
            email="anonymous@local.dev",
            display_name="Anonymous User",
        )


def get_workspace_client() -> WorkspaceClient:
    """Get Databricks WorkspaceClient using automatic auth.

    In Databricks Apps, auto-detects service principal from environment.
    In development, uses profile or token from settings.

    Note: OBO (On-Behalf-Of) authentication is not currently used.
    When needed for user-specific data access (e.g., Vector Search),
    implement a separate function that explicitly disables OAuth env vars.

    Returns:
        Configured WorkspaceClient instance.
    """
    settings = get_settings()

    # Profile-based auth (local development)
    if settings.databricks_config_profile:
        return WorkspaceClient(profile=settings.databricks_config_profile)

    # Direct token auth (local development fallback)
    if settings.databricks_host and settings.databricks_token:
        return WorkspaceClient(
            host=settings.databricks_host,
            token=settings.databricks_token,
        )

    # Automatic auth (Databricks Apps - service principal)
    return WorkspaceClient()


def get_user_workspace_client(token: str) -> WorkspaceClient:
    """Create WorkspaceClient using user's OAuth token for identity resolution.

    Used to resolve the actual end user's identity from the x-forwarded-access-token
    header in Databricks Apps deployments. This client should ONLY be used for
    identity resolution - backend operations use the service principal client.

    Args:
        token: User's OAuth access token from x-forwarded-access-token header.

    Returns:
        WorkspaceClient configured with the user's token.
    """
    settings = get_settings()
    host = settings.databricks_host
    if not host:
        # Derive host from existing service principal client
        sp_client = get_workspace_client()
        host = sp_client.config.host

    # Use auth_type='pat' to explicitly force token authentication.
    # This prevents the SDK from auto-detecting OAuth env vars
    # (DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET) set for the SP.
    return WorkspaceClient(host=host, token=token, auth_type="pat")


async def get_service_principal_workspace_client(
    sp_id: str,
    requesting_user_id: str,
    permissions_check: "Callable[[str, str], Awaitable[bool]] | None" = None,
) -> WorkspaceClient:
    """V1.5: Returns a WorkspaceClient authenticated as the given Service Principal.

    SECURITY: enforces CAN_USE_AS permission check BEFORE any tool factory runs.

    Args:
        sp_id: UUID of the Databricks Service Principal to run as.
        requesting_user_id: The user ID requesting the SP credential.
        permissions_check: Optional async callable that verifies the user has
            CAN_USE_AS permission on the SP. Signature: (user_id, sp_id) -> bool.

    Raises:
        HTTPException: 403 if permissions_check returns False.
        NotImplementedError: Always (production SP-auth path not yet wired).
    """
    from fastapi import HTTPException

    if permissions_check is not None:
        ok = await permissions_check(requesting_user_id, sp_id)
        if not ok:
            raise HTTPException(
                status_code=403,
                detail=f"missing CAN_USE_AS permission for SP {sp_id}",
            )
    # Production implementation: use Databricks SDK's SP-auth path here.
    # For V1.5 the function signature is the contract; downstream callers can mock for tests.
    raise NotImplementedError(
        "SP credential resolution requires the Databricks SDK SP-auth path; "
        "wire to workspace_client.config.authenticate(client_id=sp_id, ...)"
    )


def get_current_user(client: WorkspaceClient) -> UserIdentity:
    """Get current authenticated user from WorkspaceClient.

    Args:
        client: Authenticated WorkspaceClient instance.

    Returns:
        UserIdentity of the current user.
    """
    try:
        current_user = client.current_user.me()
        return UserIdentity.from_workspace_user(current_user)
    except Exception:
        # Fall back to anonymous for local development
        return UserIdentity.anonymous()


def extract_obo_token(headers: dict[str, str]) -> str | None:
    """Extract OBO (On-Behalf-Of) token from request headers.

    In Databricks Apps, the user's OAuth token is forwarded as
    'x-forwarded-access-token' header for user identity resolution.

    Args:
        headers: Request headers dictionary (keys should be lowercase).

    Returns:
        OAuth token if present, None otherwise.
    """
    # Log available headers for debugging (without sensitive values)
    auth_related_headers = [k for k in headers if "forward" in k.lower() or "auth" in k.lower()]
    logger.debug(f"OBO extraction - Auth-related headers present: {auth_related_headers}")

    token = headers.get("x-forwarded-access-token")
    if token:
        logger.info("OBO_TOKEN_FOUND length=%d", len(token))
    else:
        logger.debug("OBO token not found in headers - x-forwarded-access-token header absent")

    return token


_cached_workspace_host: str | None = None
_workspace_host_resolved: bool = False


# ----- V1.5 OBO Token Refresh entry point -----


async def refresh_user_token(workspace_client: WorkspaceClient, current_token: str) -> str | None:
    """V1.5 OBO refresh entry point. Wraps services.obo_client.refresh.

    Returns the new token if rotation succeeded, None if disabled, raises on failure.
    """
    from deep_research.services.obo_client import refresh as _refresh

    return await _refresh(current_token, workspace_client)


def get_workspace_host() -> str | None:
    """Get Databricks workspace base URL for constructing resource links.

    Returns host like 'https://my-workspace.cloud.databricks.com' or None
    when not available (unit tests, offline development).

    Result is cached after first successful resolution.
    """
    global _cached_workspace_host, _workspace_host_resolved
    if _workspace_host_resolved:
        return _cached_workspace_host

    host: str | None = None

    # Priority 1: Explicit host from settings
    settings = get_settings()
    if settings.databricks_host:
        host = settings.databricks_host.rstrip("/")
        if not host.startswith("http"):
            host = f"https://{host}"

    # Priority 2: Derive from workspace client config
    if not host:
        try:
            client = get_workspace_client()
            h = client.config.host
            if h:
                host = h.rstrip("/")
        except Exception:
            pass

    _cached_workspace_host = host
    _workspace_host_resolved = True
    return host
