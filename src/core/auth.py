"""Databricks authentication utilities."""

from dataclasses import dataclass
from typing import Any

from databricks.sdk import WorkspaceClient

from src.core.config import get_settings


@dataclass(frozen=True)
class UserIdentity:
    """User identity extracted from Databricks authentication."""

    user_id: str
    email: str
    display_name: str

    @classmethod
    def from_workspace_user(cls, user: Any) -> "UserIdentity":
        """Create from Databricks workspace user object."""
        return cls(
            user_id=str(user.id) if user.id else user.user_name,
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


def get_workspace_client(token: str | None = None) -> WorkspaceClient:
    """Get Databricks WorkspaceClient.

    Args:
        token: Optional OAuth token for OBO (On-Behalf-Of) authentication.
               If not provided, uses default authentication from environment.

    Returns:
        Configured WorkspaceClient instance.
    """
    settings = get_settings()

    if token:
        # OBO authentication with provided token
        return WorkspaceClient(
            host=settings.databricks_host,
            token=token,
        )

    # Default authentication (profile or environment)
    if settings.databricks_config_profile:
        return WorkspaceClient(profile=settings.databricks_config_profile)

    if settings.databricks_host and settings.databricks_token:
        return WorkspaceClient(
            host=settings.databricks_host,
            token=settings.databricks_token,
        )

    # Fall back to default SDK authentication
    return WorkspaceClient()


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
    'x-forwarded-access-token' header for impersonation.

    Args:
        headers: Request headers dictionary.

    Returns:
        OAuth token if present, None otherwise.
    """
    return headers.get("x-forwarded-access-token")
