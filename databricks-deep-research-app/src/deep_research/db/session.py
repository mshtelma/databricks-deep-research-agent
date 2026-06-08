"""Database session management with Lakebase OAuth support."""

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Annotated, Any

import asyncpg  # type: ignore[import-untyped]
import sqlalchemy.exc
from fastapi import Depends, HTTPException, status
from sqlalchemy import text
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
    lakebase_raw_asyncpg_kwargs,
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

_AUTH_FAILURE_MARKERS = (
    "password authentication failed",
    "invalid password",
    "invalid authorization",
    "invalid credentials",
    "authentication failed",
    "authorization failed",
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


def _is_db_auth_error(exc: BaseException) -> bool:
    """Detect DB authentication failures that require credential invalidation.

    Lakebase can reject an OAuth-derived password before our local
    ``expires_at`` threshold says it is stale. In that case asyncpg raises
    ``InvalidPasswordError: password authentication failed ...``; wrappers may
    then turn it into a storage-layer ``PermanentError``. Match both the
    exception class chain and the message so callers refresh the cached engine
    instead of repeatedly reusing a poisoned credential.
    """
    current: BaseException | None = exc
    while current is not None:
        if "invalidpassword" in type(current).__name__.lower():
            return True
        message = str(current).lower()
        if any(marker in message for marker in _AUTH_FAILURE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def log_lakebase_auth_failure_diagnostics(exc: BaseException) -> None:
    """Emit provider-specific, redacted context for Lakebase auth failures."""
    if _credential_provider is None:
        logger.warning(
            "LAKEBASE_DB_AUTH_FAILURE_DIAGNOSTIC provider_exists=False "
            "exc_class=%s error_fingerprint=%s",
            type(exc).__name__,
            _error_fingerprint(exc),
        )
        return

    diagnostics_logger = getattr(
        _credential_provider,
        "log_auth_failure_diagnostics",
        None,
    )
    if callable(diagnostics_logger):
        try:
            diagnostics_logger(exc)
            return
        except Exception:  # noqa: BLE001
            logger.exception("LAKEBASE_DB_AUTH_FAILURE_DIAGNOSTIC_FAILED")

    logger.warning(
        "LAKEBASE_DB_AUTH_FAILURE_DIAGNOSTIC backend=%s credential_exists=%s "
        "exc_class=%s error_fingerprint=%s",
        _credential_provider.get_backend_type(),
        _credential_provider.current_credential is not None,
        type(exc).__name__,
        _error_fingerprint(exc),
    )


def _error_fingerprint(exc: BaseException) -> str:
    message = str(exc)
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
    return f"sha256={digest}:len={len(message)}"


def _redact_db_identity(value: object | None) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    if "@" in rendered:
        kind = "email"
    elif len(rendered) == 36 and rendered.count("-") == 4:
        kind = "uuid"
    elif rendered.isdigit():
        kind = "numeric"
    else:
        kind = "other"
    return f"kind={kind}:sha256={digest}:len={len(rendered)}"


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


def _maybe_log_endpoint_state(provider: BaseLakebaseCredentialProvider) -> None:
    """Invoke the provider's non-fatal endpoint-state probe, if it has one."""
    probe = getattr(provider, "log_endpoint_state_diagnostics", None)
    if callable(probe):
        try:
            probe()
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            logger.debug("ENDPOINT_STATE_PROBE_FAILED", exc_info=True)


def _lakebase_connect_kwargs(
    settings: Settings,
    provider: BaseLakebaseCredentialProvider,
) -> dict[str, Any]:
    """Build raw ``asyncpg.connect`` kwargs from the current credential.

    Mirrors the connection the SQLAlchemy URL would have produced, but lets us
    own connection creation (and thus retry) via ``async_creator``. PgBouncer-
    safe options come from ``lakebase_raw_asyncpg_kwargs`` so we do not regress
    the statement-cache settings the URL path applied via ``connect_args``.
    """
    cred = provider.get_credential()
    kwargs: dict[str, Any] = {
        "host": provider.get_host(),
        "port": provider.get_port(),
        "user": cred.username,
        "password": cred.token,
        "database": provider.get_database(),
        "ssl": True,
        **lakebase_raw_asyncpg_kwargs(),
    }
    if settings.db_command_timeout is not None:
        kwargs["command_timeout"] = settings.db_command_timeout
    return kwargs


def _make_lakebase_async_creator(
    settings: Settings,
    provider: BaseLakebaseCredentialProvider,
) -> Callable[[], Awaitable[Any]]:
    """Return an async connection creator with bounded auth-failure retry.

    A freshly-minted Lakebase OAuth credential can be transiently rejected
    ("password authentication failed") by the PgBouncer/databricks_auth layer
    for a few seconds until it propagates. ``provider.get_credential()`` returns
    the cached (non-expired) credential, so retrying reuses the SAME token —
    which is exactly what a propagation race needs — rather than minting a new
    token that would hit the same race. Only when the credential is genuinely
    expired does ``get_credential()`` mint a new one. Bounded backoff caps the
    worst case so a truly invalid credential still fails fast.

    Retry/backoff settings are read inside the creator (per connection birth)
    rather than at factory time so engine construction stays side-effect free.
    """

    async def _creator() -> Any:
        attempts = max(1, settings.lakebase_auth_retry_attempts)
        base = max(0.0, settings.lakebase_auth_retry_base_delay_s)
        cap = max(base, settings.lakebase_auth_retry_max_delay_s)
        start = time.monotonic()
        first_token_fp: str | None = None
        reused_same_token = True
        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            kwargs = _lakebase_connect_kwargs(settings, provider)
            token_fp = hashlib.sha256(str(kwargs["password"]).encode()).hexdigest()[:12]
            if first_token_fp is None:
                first_token_fp = token_fp
            elif token_fp != first_token_fp:
                reused_same_token = False

            try:
                conn = await asyncpg.connect(**kwargs)
                if attempt > 1:
                    logger.warning(
                        "LAKEBASE_AUTH_RECOVERED attempts=%d elapsed_ms=%.0f same_token=%s",
                        attempt,
                        (time.monotonic() - start) * 1000.0,
                        reused_same_token,
                    )
                return conn
            except Exception as exc:  # noqa: BLE001 - re-raised below if not auth
                if not _is_db_auth_error(exc):
                    raise
                last_exc = exc
                cred = provider.current_credential
                expired = cred.is_expired if cred is not None else None
                if attempt == 1:
                    # Full redacted diagnostics + endpoint-state probe once.
                    log_lakebase_auth_failure_diagnostics(exc)
                    _maybe_log_endpoint_state(provider)
                if attempt >= attempts:
                    break
                delay = min(cap, base * (2 ** (attempt - 1)))
                logger.warning(
                    "LAKEBASE_AUTH_RETRY attempt=%d/%d delay_s=%.2f token_expired=%s",
                    attempt,
                    attempts,
                    delay,
                    expired,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        logger.error(
            "LAKEBASE_AUTH_RETRY_EXHAUSTED attempts=%d elapsed_ms=%.0f same_token=%s",
            attempts,
            (time.monotonic() - start) * 1000.0,
            reused_same_token,
        )
        assert last_exc is not None  # loop only exits via return or after setting last_exc
        raise last_exc

    return _creator


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
        # PgBouncer-safe connection options (see db/asyncpg_config.py for why).
        connect_args = lakebase_asyncpg_connect_args(settings)
        engine_kwargs = lakebase_engine_kwargs(settings)

        # For Lakebase we own connection creation via ``async_creator`` so we can
        # retry a transient "password authentication failed" on a freshly-minted
        # token (PgBouncer/databricks_auth propagation race) instead of 500-ing.
        # The creator supplies host/credentials + PgBouncer-safe options itself,
        # so we hand SQLAlchemy a credential-less dialect URL and omit connect_args.
        lakebase_provider = (
            get_credential_provider(settings) if settings.use_lakebase else None
        )

        logger.info(
            "DB_ENGINE_CREATED lakebase=%s auth_retry=%s statement_cache_size=%s "
            "command_timeout=%s",
            settings.use_lakebase,
            lakebase_provider is not None,
            connect_args.get("statement_cache_size"),
            connect_args.get("command_timeout"),
        )

        common_kwargs: dict[str, Any] = dict(
            echo=settings.debug and not settings.is_production,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            # For Lakebase: recycle connections at 45 min (2700s) to ensure they're
            # refreshed BEFORE the 5-minute token expiry buffer kicks in at 55 min.
            # This prevents pooled connections from holding stale tokens.
            pool_recycle=2700 if settings.use_lakebase else 3600,
            **engine_kwargs,
        )

        if lakebase_provider is not None:
            _engine = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=_make_lakebase_async_creator(settings, lakebase_provider),
                **common_kwargs,
            )
        else:
            _engine = create_async_engine(
                get_database_url(settings),
                connect_args=connect_args,
                **common_kwargs,
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


async def log_lakebase_connection_self_test(settings: Settings | None = None) -> None:
    """Run a non-fatal Lakebase connection probe and log redacted identity context."""
    if settings is None:
        settings = get_settings()
    if not settings.use_lakebase:
        return

    try:
        session_maker = get_session_maker(settings)
        async with session_maker() as session:
            result = await session.execute(
                text(
                    "SELECT current_user AS current_user, "
                    "current_database() AS database, current_schema() AS schema"
                )
            )
            row = result.mappings().one()
        logger.info(
            "LAKEBASE_SELF_TEST_OK current_user=%s database=%s schema=%s",
            _redact_db_identity(row["current_user"]),
            row["database"],
            row["schema"],
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic must not block startup
        auth_error = _is_db_auth_error(exc)
        if auth_error:
            log_lakebase_auth_failure_diagnostics(exc)
        logger.warning(
            "LAKEBASE_SELF_TEST_FAILED exc_class=%s auth_error=%s",
            type(exc).__name__,
            auth_error,
        )


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
            had_pending_writes = bool(session.new) or bool(session.dirty) or bool(session.deleted)
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
                    logger.warning("DB_ROLLBACK_FAILED: %s", str(rollback_err)[:200])
            # Auth failures that survive the connection-birth retry budget reach
            # here. Distinguish two cases by the cached credential's expiry:
            #  - expired/unknown → mint a fresh token so the NEXT request recovers.
            #  - still valid     → this was a transient PgBouncer/databricks_auth
            #                      propagation event, NOT a stale credential.
            #                      Force-refreshing would mint another age-0 token
            #                      that hits the same race, so we do NOT refresh;
            #                      we surface a 503 for the client to retry.
            if _is_db_auth_error(e):
                cred = (
                    _credential_provider.current_credential
                    if _credential_provider is not None
                    else None
                )
                expired = cred.is_expired if cred is not None else None
                if expired is None or expired:
                    logger.warning(
                        "Database auth failed on expired/unknown token; refreshing "
                        "Lakebase credentials. exc_class=%s",
                        type(e).__name__,
                    )
                    await refresh_engine_credentials()
                else:
                    cred_age = cred.age_s if cred is not None else None
                    logger.warning(
                        "LAKEBASE_AUTH_FAILURE_ON_FRESH_TOKEN exc_class=%s cred_age_s=%s "
                        "— not refreshing (transient propagation); surfacing 503.",
                        type(e).__name__,
                        f"{cred_age:.1f}" if cred_age is not None else None,
                    )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database temporarily unavailable (auth). Please retry.",
                    headers={"Retry-After": "2"},
                ) from e
            # Stale-connection on the yield itself (extremely rare) is still
            # not useful to propagate — the request already finished on the
            # caller side; turning it into a 500 helps no one.
            if _is_stale_connection_error(e):
                logger.warning("DB_YIELD_STALE_CONNECTION: %s", str(e)[:200])
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
                    logger.warning("DB_CLOSE_FAILED: %s", str(close_err)[:200])


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
