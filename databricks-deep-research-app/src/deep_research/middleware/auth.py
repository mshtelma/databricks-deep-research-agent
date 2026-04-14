"""Databricks authentication middleware."""

import asyncio
import logging
import time
from typing import Annotated

from databricks.sdk.errors import DatabricksError  # type: ignore[attr-defined]
from fastapi import Depends, HTTPException, Request, status

from deep_research.core.auth import (
    UserIdentity,
    get_current_user,
    get_workspace_client,
)
from deep_research.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Process-level cache: skip DB upsert if user was seen recently
_user_sync_cache: dict[str, float] = {}
_USER_SYNC_TTL = 300  # 5 minutes
_USER_SYNC_MAX_CACHE = 1024
_USER_SYNC_TIMEOUT = 3.0  # seconds


async def _sync_user_record(user: UserIdentity) -> None:
    """Upsert user identity to the users table (non-fatal).

    Uses its own session — the auth dependency runs before get_db,
    so the request session doesn't exist yet.
    """
    if user.user_id == "anonymous":
        return

    now = time.monotonic()
    if _user_sync_cache.get(user.user_id, 0) > now:
        return  # Recently synced

    try:
        from deep_research.db.session import get_session_maker
        from deep_research.services.user_service import UserService

        session_maker = get_session_maker()
        async with session_maker() as session:
            svc = UserService(session)
            await asyncio.wait_for(
                svc.upsert(
                    user_id=user.user_id,
                    email=user.email,
                    display_name=user.display_name,
                ),
                timeout=_USER_SYNC_TIMEOUT,
            )
            await asyncio.wait_for(session.commit(), timeout=_USER_SYNC_TIMEOUT)
        if len(_user_sync_cache) > _USER_SYNC_MAX_CACHE:
            _user_sync_cache.clear()
        _user_sync_cache[user.user_id] = now + _USER_SYNC_TTL
    except Exception:
        logger.warning("USER_SYNC_FAILED user_id=%s", user.user_id, exc_info=True)


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

            # T002: Preserve OBO token for enterprise data source access
            # Used by VectorSearchTool, GenieTool, KnowledgeAssistantTool
            request.state.obo_token = obo_token
            request.state.user_workspace_client = user_client

            logger.info(f"OBO auth successful: user={user.email}, id={user.user_id}")
            await _sync_user_record(user)
            return user

        except (ConnectionError, TimeoutError, ValueError, RuntimeError, DatabricksError) as e:
            logger.warning(f"OBO auth failed, falling back to SP: {e}")

    # Priority 2: Service principal auth (existing logic)
    try:
        client = get_workspace_client()
        user = get_current_user(client)

        request.state.user = user
        request.state.workspace_client = client

        logger.debug(f"Service principal auth successful: user={user.email}")
        await _sync_user_record(user)
        return user

    except (ConnectionError, TimeoutError, ValueError, RuntimeError, DatabricksError) as e:
        logger.warning(f"Service principal auth failed: {e}")

    # Priority 3: Anonymous (development mode only)
    if not settings.is_production:
        user = UserIdentity.anonymous()
        request.state.user = user
        logger.warning(
            "AUTH_ANONYMOUS_FALLBACK: Development mode anonymous user active. "
            "Ensure APP_ENV=production in deployment."
        )
        await _sync_user_record(user)
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
