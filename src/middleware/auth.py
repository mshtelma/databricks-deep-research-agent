"""Databricks authentication middleware."""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from src.core.auth import (
    UserIdentity,
    get_current_user,
    get_workspace_client,
)
from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


async def get_current_user_identity(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserIdentity:
    """FastAPI dependency to get current user identity.

    In Databricks Apps, uses the app's service principal for all operations.
    OBO (On-Behalf-Of) authentication is not currently used to avoid conflicts
    with the auto-injected OAuth credentials in the Databricks Apps environment.

    In development, falls back to configured credentials or anonymous.

    Args:
        request: FastAPI request object.
        settings: Application settings.

    Returns:
        UserIdentity of the authenticated user.

    Raises:
        HTTPException: If all authentication methods fail.
    """
    # Use service principal auth (WorkspaceClient auto-detects environment)
    try:
        client = get_workspace_client()
        user = get_current_user(client)

        # Store in request state for later use
        request.state.user = user
        request.state.workspace_client = client

        logger.debug(f"Service principal auth successful: user={user.email}")
        return user

    except Exception as e:
        logger.warning(f"Service principal auth failed: {e}")

    # Fallback: In development mode, allow anonymous access
    if not settings.is_production:
        user = UserIdentity.anonymous()
        request.state.user = user
        logger.debug("Using anonymous user (development mode)")
        return user

    # All methods failed in production
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication failed",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Type alias for dependency injection
CurrentUser = Annotated[UserIdentity, Depends(get_current_user_identity)]


def require_authenticated_user(user: CurrentUser) -> UserIdentity:
    """Dependency that requires a non-anonymous user.

    Use this for endpoints that require actual authentication,
    not just identification.
    """
    if user.user_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


AuthenticatedUser = Annotated[UserIdentity, Depends(require_authenticated_user)]
