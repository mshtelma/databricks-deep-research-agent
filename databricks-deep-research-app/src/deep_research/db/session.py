"""Database session management with Lakebase OAuth support."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from deep_research.core.config import Settings, get_settings
from deep_research.db.credential_provider import BaseLakebaseCredentialProvider

logger = logging.getLogger(__name__)

# Module-level state
_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None
_credential_provider: BaseLakebaseCredentialProvider | None = None


async def _dispose_engine_async(engine: AsyncEngine) -> None:
    """Safely dispose of an async engine.

    Args:
        engine: The async engine to dispose.
    """
    try:
        await engine.dispose()
    except Exception as e:
        logger.warning(f"Error disposing engine: {e}")


def get_credential_provider(settings: Settings) -> BaseLakebaseCredentialProvider | None:
    """Get or create Lakebase credential provider.

    Uses the credential factory for auto-detection of backend type
    (Provisioned vs Autoscaling).

    Args:
        settings: Application settings.

    Returns:
        Credential provider if Lakebase is configured, None otherwise.
    """
    global _credential_provider

    if not settings.use_lakebase:
        return None

    if _credential_provider is None:
        from deep_research.db.credential_factory import create_credential_provider

        _credential_provider = create_credential_provider(settings)

    return _credential_provider


def get_database_url(settings: Settings) -> str:
    """Get database URL, using Lakebase OAuth if configured.

    Args:
        settings: Application settings.

    Returns:
        Database connection URL.

    Raises:
        ValueError: If no database configuration is available.
    """
    if settings.use_lakebase:
        provider = get_credential_provider(settings)
        if provider:
            return provider.build_connection_url()

    # Fallback to static DATABASE_URL
    if settings.database_url:
        return str(settings.database_url)

    raise ValueError("No database configuration: set LAKEBASE_* or DATABASE_URL")


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Get or create async database engine with proactive credential refresh.

    Args:
        settings: Application settings (uses cached settings if None).

    Returns:
        SQLAlchemy async engine.

    Note:
        For Lakebase connections, this checks if the OAuth token is expired
        and refreshes it proactively before creating/reusing the engine.
    """
    global _engine, _async_session_maker, _credential_provider

    if settings is None:
        settings = get_settings()

    # Proactive token refresh check (Lakebase only)
    if settings.use_lakebase and _credential_provider is not None:
        cred = _credential_provider.current_credential
        if cred is not None:
            logger.info(
                "LAKEBASE_ENGINE_CHECK credential_exists=True expires_at=%s "
                "is_expired=%s engine_exists=%s",
                cred.expires_at.isoformat(),
                cred.is_expired,
                _engine is not None,
            )
        if cred is not None and cred.is_expired:
            logger.info("LAKEBASE_ENGINE_REFRESH_TRIGGERED reason=token_expired_or_expiring")
            # Force credential refresh FIRST
            _credential_provider.get_credential(force_refresh=True)
            # Clear engine and schedule disposal in the SAME event loop
            if _engine is not None:
                engine_to_dispose = _engine
                _engine = None
                _async_session_maker = None

                # Schedule disposal in the SAME event loop where connections live.
                # This avoids "Future attached to a different loop" errors that occur
                # when asyncio.run() creates a NEW event loop in a thread.
                #
                # Safety guarantees if disposal is delayed:
                # - pool_pre_ping=True validates connections before use
                # - pool_recycle=2700 refreshes connections every 45 min
                # - _dispose_engine_async() has try/except for safe disposal
                try:
                    loop = asyncio.get_running_loop()
                    # Fire-and-forget: schedule in same loop, don't block
                    loop.create_task(_dispose_engine_async(engine_to_dispose))
                    logger.info("LAKEBASE_ENGINE_DISPOSED scheduled=True")
                except RuntimeError:
                    # No running loop - extremely rare since get_engine() is always
                    # called from async contexts (get_db, job_manager, etc.)
                    # pool_pre_ping will validate connections on next use
                    logger.info("LAKEBASE_ENGINE_DISPOSED deferred=True no_running_loop=True")

    if _engine is None:
        database_url = get_database_url(settings)

        logger.info(f"Creating database engine (lakebase={settings.use_lakebase})")

        # For Lakebase: use SSL (asyncpg doesn't accept sslmode URL param)
        connect_args = {"ssl": True} if settings.use_lakebase else {}

        _engine = create_async_engine(
            database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            # For Lakebase: recycle connections at 45 min (2700s) to ensure they're
            # refreshed BEFORE the 5-minute token expiry buffer kicks in at 55 min.
            # This prevents pooled connections from holding stale tokens.
            pool_recycle=2700 if settings.use_lakebase else 3600,
            connect_args=connect_args,
        )

    return _engine


def get_session_maker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Get or create async session maker.

    Args:
        settings: Application settings (uses cached settings if None).

    Returns:
        Async session maker.
    """
    global _async_session_maker

    # CRITICAL: Always call get_engine() to trigger proactive token refresh.
    # If token is expired, get_engine() disposes the old engine and sets
    # _async_session_maker = None, forcing recreation below.
    engine = get_engine(settings)

    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_maker


async def refresh_engine_credentials() -> None:
    """Refresh database engine with new OAuth credentials.

    Call this if authentication fails due to expired token.
    """
    global _engine, _async_session_maker, _credential_provider

    settings = get_settings()

    if not settings.use_lakebase or _credential_provider is None:
        return

    logger.info("Refreshing Lakebase credentials and recreating engine")

    # Close existing engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None

    # Force credential refresh
    _credential_provider.get_credential(force_refresh=True)

    # Engine will be recreated on next request


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session with auto-refresh on auth failure.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...

    Yields:
        Database session.

    Note:
        If a database authentication error occurs (expired Lakebase token),
        this will trigger a credential refresh for the next request.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            # Check if this is an auth error that might be fixed by credential refresh
            error_str = str(e).lower()
            if "invalid" in error_str and (
                "password" in error_str or "authorization" in error_str
            ):
                logger.warning(f"Database auth failed: {e}")
                logger.info("Triggering credential refresh for next request...")
                await refresh_engine_credentials()
            raise
        finally:
            await session.close()


# Type alias for dependency injection
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def close_db() -> None:
    """Close database connections (call on app shutdown)."""
    global _engine, _async_session_maker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
