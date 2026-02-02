"""Databricks authentication middleware."""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from deep_research.core.auth import (
    UserIdentity,
    get_current_user,
    get_workspace_client,
)
from deep_research.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


async def get_current_user_identity(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserIdentity:
    """FastAPI dependency to get current user identity.

    Priority order:
    1. OBO token from x-forwarded-access-token (actual user in Databricks Apps)
    2. Service principal auth (fallback for local development)
    3. Anonymous (development mode only)

    Args:
        request: FastAPI request object.
        settings: Application settings.

    Returns:
        UserIdentity of the authenticated user.

    Raises:
        HTTPException: If all authentication methods fail in production.
    """
    from deep_research.core.auth import extract_obo_token, get_user_workspace_client

    # Priority 1: OBO token (actual user in Databricks Apps)
    obo_token = extract_obo_token(dict(request.headers))
    if obo_token:
        try:
            user_client = get_user_workspace_client(obo_token)
            current_user = user_client.current_user.me()
            user = UserIdentity.from_workspace_user(current_user)

            # Keep service principal client for backend operations
            sp_client = get_workspace_client()
            request.state.user = user
            request.state.workspace_client = sp_client

            logger.info(f"OBO auth successful: user={user.email}, id={user.user_id}")
            return user

        except Exception as e:
            logger.warning(f"OBO auth failed, falling back to SP: {e}")

    # Priority 2: Service principal auth (existing logic)
    try:
        client = get_workspace_client()
        user = get_current_user(client)

        request.state.user = user
        request.state.workspace_client = client

        logger.debug(f"Service principal auth successful: user={user.email}")
        return user

    except Exception as e:
        logger.warning(f"Service principal auth failed: {e}")

    # Priority 3: Anonymous (development mode only)
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
