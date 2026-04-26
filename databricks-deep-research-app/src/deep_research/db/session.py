"""Database session management with Lakebase OAuth support."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

import sqlalchemy.exc
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from deep_research.core.config import Settings, get_settings
from deep_research.db.asyncpg_config import (
    lakebase_asyncpg_connect_args,
    lakebase_engine_kwargs,
)
from deep_research.db.credential_provider import BaseLakebaseCredentialProvider

logger = logging.getLogger(__name__)


_STALE_CONNECTION_MARKERS = (
    "connection is closed",
    "connection was closed",
    "connection is not open",
    "connection is lost",
    "another operation is in progress",
)


def _is_stale_connection_error(exc: BaseException) -> bool:
    """Detect that the DB connection underlying ``exc`` is no longer usable.

    Long-running requests (SSE streams) hold the request-scoped session idle
    for minutes. PgBouncer/Lakebase will close the idle connection; when the
    request finally completes and FastAPI runs the ``get_db`` cleanup,
    ``session.commit()`` / ``rollback()`` / ``close()`` then raise
    ``InterfaceError: ... the underlying connection is closed``. At that
    point the writes the caller expected to persist are already lost —
    re-raising only turns a successful request into a 500 for the user and
    obscures real errors. We detect this narrow condition by exception
    type + message substring.
    """
    if not isinstance(exc, (sqlalchemy.exc.InterfaceError, sqlalchemy.exc.OperationalError)):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _STALE_CONNECTION_MARKERS)

# Module-level state
_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None
_credential_provider: BaseLakebaseCredentialProvider | None = None
# Tracks fire-and-forget engine-disposal tasks spawned during proactive
# token refresh so ``close_db()`` can await them before the event loop
# tears down. Without this set, a token-expiry storm could leak partially
# disposed engines and emit "Task was destroyed but it is pending!" on
# shutdown.
_pending_disposals: set[asyncio.Task[None]] = set()


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
                    # Fire-and-forget: schedule in same loop, don't block.
                    # Tracked in ``_pending_disposals`` so ``close_db()`` can
                    # drain pending disposals before the loop tears down.
                    task = loop.create_task(_dispose_engine_async(engine_to_dispose))
                    _pending_disposals.add(task)
                    task.add_done_callback(_pending_disposals.discard)
                    logger.info("LAKEBASE_ENGINE_DISPOSED scheduled=True")
                except RuntimeError:
                    # No running loop - extremely rare since get_engine() is always
                    # called from async contexts (get_db, job_manager, etc.)
                    # pool_pre_ping will validate connections on next use
                    logger.info("LAKEBASE_ENGINE_DISPOSED deferred=True no_running_loop=True")

    if _engine is None:
        database_url = get_database_url(settings)

        # PgBouncer-safe connection options (see db/asyncpg_config.py for why).
        connect_args = lakebase_asyncpg_connect_args(settings)
        engine_kwargs = lakebase_engine_kwargs(settings)

        logger.info(
            "DB_ENGINE_CREATED lakebase=%s statement_cache_size=%s "
            "prepared_statement_cache_size=%s command_timeout=%s",
            settings.use_lakebase,
            connect_args.get("statement_cache_size"),
            engine_kwargs.get("prepared_statement_cache_size"),
            connect_args.get("command_timeout"),
        )

        _engine = create_async_engine(
            database_url,
            echo=settings.debug and not settings.is_production,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            # For Lakebase: recycle connections at 45 min (2700s) to ensure they're
            # refreshed BEFORE the 5-minute token expiry buffer kicks in at 55 min.
            # This prevents pooled connections from holding stale tokens.
            pool_recycle=2700 if settings.use_lakebase else 3600,
            connect_args=connect_args,
            **engine_kwargs,
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

        If the underlying connection is closed mid-request (typically on
        long-running SSE streams whose request-scoped connection sits idle
        and is reaped by PgBouncer/Lakebase):

        * If the session had no pending writes (read-only or already
          committed via the service-level ``commit()``), the cleanup
          stale-connection error is swallowed — the request succeeded.
        * If the session had pending writes (the endpoint relied on the
          implicit cleanup commit and never called ``service.commit()``),
          a stale-connection error means data was lost. We raise 503
          rather than return a misleading success.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            # Snapshot pending writes BEFORE the commit attempt; a failed
            # commit puts the session into an aborted state where these
            # collections are unreliable.
            had_pending_writes = (
                bool(session.new) or bool(session.dirty) or bool(session.deleted)
            )
            try:
                await session.commit()
            except Exception as commit_err:
                if _is_stale_connection_error(commit_err):
                    if had_pending_writes:
                        logger.error(
                            "DB_COMMIT_STALE_CONNECTION_WITH_PENDING_WRITES: "
                            "request had uncommitted writes when the underlying "
                            "connection died — data was lost. error=%s",
                            str(commit_err)[:200],
                        )
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=(
                                "Database connection lost mid-request. "
                                "Pending writes were not persisted; please retry."
                            ),
                        ) from commit_err
                    logger.warning(
                        "DB_COMMIT_STALE_CONNECTION (no pending writes, benign): "
                        "session was already empty when cleanup commit fired "
                        "(likely idle during SSE stream). error=%s",
                        str(commit_err)[:200],
                    )
                else:
                    raise
        except Exception as e:
            try:
                await session.rollback()
            except Exception as rollback_err:
                if not _is_stale_connection_error(rollback_err):
                    logger.warning(
                        "DB_ROLLBACK_FAILED: %s", str(rollback_err)[:200]
                    )
            # Check if this is an auth error that might be fixed by credential refresh
            error_str = str(e).lower()
            if "invalid" in error_str and (
                "password" in error_str or "authorization" in error_str
            ):
                logger.warning(f"Database auth failed: {e}")
                logger.info("Triggering credential refresh for next request...")
                await refresh_engine_credentials()
            # Stale-connection on the yield itself (extremely rare) is still
            # not useful to propagate — the request already finished on the
            # caller side; turning it into a 500 helps no one.
            if _is_stale_connection_error(e):
                logger.warning(
                    "DB_YIELD_STALE_CONNECTION: %s", str(e)[:200]
                )
                return
            raise
        finally:
            try:
                await session.close()
            except Exception as close_err:
                if _is_stale_connection_error(close_err):
                    logger.debug(
                        "DB_CLOSE_STALE_CONNECTION (benign): %s",
                        str(close_err)[:200],
                    )
                else:
                    logger.warning(
                        "DB_CLOSE_FAILED: %s", str(close_err)[:200]
                    )


# Type alias for dependency injection
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def close_db() -> None:
    """Close database connections (call on app shutdown).

    Drains any pending engine-disposal tasks first so we don't leak them
    when the event loop tears down.
    """
    global _engine, _async_session_maker
    if _pending_disposals:
        # Snapshot so the discard callback can mutate the set safely.
        in_flight = list(_pending_disposals)
        await asyncio.gather(*in_flight, return_exceptions=True)
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
