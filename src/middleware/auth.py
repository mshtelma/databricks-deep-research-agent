"""Databricks authentication middleware."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from src.core.auth import (
    UserIdentity,
    extract_obo_token,
    get_current_user,
    get_workspace_client,
)
from src.core.config import Settings, get_settings


async def get_current_user_identity(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    x_forwarded_access_token: str | None = Header(None),
) -> UserIdentity:
    """FastAPI dependency to get current user identity.

    In production (Databricks Apps), uses the OBO token from headers.
    In development, falls back to configured credentials or anonymous.

    Args:
        request: FastAPI request object.
        settings: Application settings.
        x_forwarded_access_token: OBO token from Databricks Apps proxy.

    Returns:
        UserIdentity of the authenticated user.

    Raises:
        HTTPException: If authentication fails in production mode.
    """
    # Try to get OBO token from header
    token = x_forwarded_access_token or extract_obo_token(dict(request.headers))

    try:
        client = get_workspace_client(token=token)
        user = get_current_user(client)

        # Store in request state for later use
        request.state.user = user
        request.state.workspace_client = client

        return user

    except Exception as e:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

        # In development, allow anonymous access
        user = UserIdentity.anonymous()
        request.state.user = user
        return user


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
