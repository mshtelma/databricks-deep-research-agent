"""Database session management with Lakebase OAuth support."""

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

from src.core.config import Settings, get_settings
from src.db.lakebase_auth import LakebaseCredentialProvider

logger = logging.getLogger(__name__)

# Module-level state
_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None
_credential_provider: LakebaseCredentialProvider | None = None


def get_credential_provider(settings: Settings) -> LakebaseCredentialProvider | None:
    """Get or create Lakebase credential provider.

    Args:
        settings: Application settings.

    Returns:
        Credential provider if Lakebase is configured, None otherwise.
    """
    global _credential_provider

    if not settings.use_lakebase:
        return None

    if _credential_provider is None:
        _credential_provider = LakebaseCredentialProvider(settings)

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
    """Get or create async database engine.

    Args:
        settings: Application settings (uses cached settings if None).

    Returns:
        SQLAlchemy async engine.
    """
    global _engine

    if _engine is None:
        if settings is None:
            settings = get_settings()

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
            # For Lakebase: shorter pool recycle to handle token expiry
            pool_recycle=3000 if settings.use_lakebase else 3600,
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

    if _async_session_maker is None:
        engine = get_engine(settings)
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
    """FastAPI dependency for database session.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...

    Yields:
        Database session.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
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
