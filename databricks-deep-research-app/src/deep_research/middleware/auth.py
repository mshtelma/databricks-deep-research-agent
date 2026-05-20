"""Databricks authentication middleware.

Owns request-side user identity resolution (OBO → SP → anonymous) and
performs the user-identity upsert *synchronously on cache miss* so the
``users`` row is durable before any FK-bearing INSERT (chats,
research_sessions, etc.) fires later in the request.

The upsert goes through ``make_user_service``, which dispatches to the
``StorageBackend``-backed ``CachedUserService`` when running in cached
mode (any supported backend: Lakebase, SQL Warehouse, Fake) or to the
legacy SQLAlchemy ``UserService`` otherwise.

A process-level cache (``_user_sync_cache``) keyed by ``user_id`` short-
circuits the upsert for ``user_sync_success_ttl_sec`` after a successful
sync — the steady state for any active user is one cache hit per
request, no DB roundtrip.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Annotated, Literal

from databricks.sdk.errors import DatabricksError  # type: ignore[attr-defined]
from fastapi import Depends, HTTPException, Request, status

from deep_research.core.auth import (
    UserIdentity,
    get_current_user,
    get_workspace_client,
)
from deep_research.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# --- Sync bookkeeping --------------------------------------------------------


@dataclass(frozen=True)
class _SyncCacheEntry:
    """One cache entry per user_id.

    `kind` lets us distinguish "recently succeeded — skip for the full
    TTL" from "recently failed — skip for a shorter TTL so we don't retry
    on every request but still recover reasonably quickly". `valid_until`
    is a `time.monotonic()` timestamp.
    """

    kind: Literal["success", "failure"]
    valid_until: float


# Process-level caches. Bounded by `_enforce_cache_bound` on every write.
_user_sync_cache: dict[str, _SyncCacheEntry] = {}
# Per-user locks coalesce the thundering herd on first-request-per-user.
# The companion timestamp dict powers opportunistic eviction so we don't
# retain lock objects for users that have since gone quiet.
_user_sync_locks: dict[str, asyncio.Lock] = {}
_user_sync_lock_touched: dict[str, float] = {}


def _enforce_cache_bound(settings: Settings) -> None:
    """Evict everything if we exceed `user_sync_max_cache`.

    Called on every cache write (both success and failure paths) so
    unbounded growth under a sustained outage is impossible. A full
    clear is sufficient — the cache is regenerative and the sweep is
    cheap (dict clear is O(1) in CPython).
    """
    if len(_user_sync_cache) > settings.user_sync_max_cache:
        _user_sync_cache.clear()


def _record_sync_outcome(
    user_id: str,
    kind: Literal["success", "failure"],
    ttl_sec: float,
    settings: Settings,
) -> None:
    """Populate `_user_sync_cache` with a bounded entry."""
    _enforce_cache_bound(settings)
    _user_sync_cache[user_id] = _SyncCacheEntry(
        kind=kind,
        valid_until=time.monotonic() + ttl_sec,
    )


def _cache_hit(user_id: str) -> bool:
    """True iff we should skip sync for this user right now."""
    entry = _user_sync_cache.get(user_id)
    return entry is not None and entry.valid_until > time.monotonic()


def _lock_for(user_id: str, settings: Settings) -> asyncio.Lock:
    """Return the per-user lock, creating it if absent.

    Opportunistically evicts locks idle longer than
    `user_sync_lock_ttl_sec` so the lock dict doesn't grow forever
    under workloads with many unique users.
    """
    now = time.monotonic()
    lock = _user_sync_locks.get(user_id)
    if lock is None:
        lock = _user_sync_locks[user_id] = asyncio.Lock()
    _user_sync_lock_touched[user_id] = now

    # Cheap opportunistic eviction (constant amortized).
    if len(_user_sync_locks) > settings.user_sync_max_cache:
        stale_cutoff = now - settings.user_sync_lock_ttl_sec
        stale = [
            uid
            for uid, ts in _user_sync_lock_touched.items()
            if ts < stale_cutoff and not _user_sync_locks[uid].locked()
        ]
        for uid in stale:
            _user_sync_locks.pop(uid, None)
            _user_sync_lock_touched.pop(uid, None)

    return lock


# --- Public sync entry point ------------------------------------------------


async def _ensure_user_synced(
    user: UserIdentity,
    request: Request,
    settings: Settings,
) -> None:
    """Synchronously durably-persist the user identity on cache miss.

    Steady state (cache hit): returns immediately, no DB roundtrip.
    Cache miss: awaits ``_sync_user_record`` so the ``users`` row is
    durable BEFORE the dependency returns, satisfying FK constraints on
    ``chats``, ``research_sessions``, etc. introduced by migration 021.

    The ``user_sync_success_ttl_sec`` setting (default 300s) bounds how
    often a single user actually hits the DB.

    Anonymous user is skipped — migration 021 inserts an ``'anonymous'``
    sentinel row, so FKs resolve without per-request work.
    """
    if not settings.user_sync_enabled:
        return
    if user.user_id == "anonymous":
        return
    if _cache_hit(user.user_id):
        return

    await _sync_user_record(user, request, settings)


# --- Test-only legacy shim --------------------------------------------------
#
# ``_schedule_user_sync`` is preserved as a fire-and-forget wrapper around
# ``_ensure_user_synced`` solely for the existing unit-test suite at
# ``tests/unit/middleware/test_user_sync.py`` which asserts non-blocking
# scheduling, thundering-herd dedup, and bounded cache behaviour.
#
# Production paths must use ``await _ensure_user_synced(...)`` so the
# ``users`` row is durable before any FK-bearing INSERT runs later in the
# request. Calling the shim from request handlers re-introduces the race
# this refactor exists to eliminate.


def _schedule_user_sync(
    user: UserIdentity,
    request: Request,
    settings: Settings,
) -> None:
    """Test-only fire-and-forget wrapper. Production callers must
    ``await _ensure_user_synced(...)`` instead.
    """
    if not settings.user_sync_enabled:
        return
    if user.user_id == "anonymous":
        return
    if _cache_hit(user.user_id):
        return

    task = asyncio.create_task(
        _sync_user_record(user, request, settings),
        name=f"user_sync:{user.user_id}",
    )
    pending = getattr(request.app.state, "pending_user_syncs", None)
    if pending is not None:
        try:
            pending.add(task)
            task.add_done_callback(pending.discard)
        except AttributeError:
            pass
    task.add_done_callback(_log_task_exc)


def _log_task_exc(task: asyncio.Task[None]) -> None:
    """Done-callback for the legacy shim that surfaces task failures
    via WARNING log so asyncio's "Task exception was never retrieved"
    message doesn't fire on background-task errors.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "USER_SYNC_TASK_ERROR name=%s err=%r",
            task.get_name(),
            exc,
        )


# --- The sync body -----------------------------------------------------------


async def _sync_user_record(
    user: UserIdentity,
    request: Request,
    settings: Settings,
) -> None:
    """Upsert the current user identity via `IUserService`.

    Always safe to call concurrently; the per-user lock coalesces N
    simultaneous calls into one backend write. Idempotent (last-write
    wins on email/display_name/last_seen_at). Non-fatal: any exception
    is logged at WARNING and cached as a failure so the caller
    back-pressures retries instead of hammering the DB.
    """
    lock = _lock_for(user.user_id, settings)
    async with lock:
        # Re-check cache under the lock so concurrent winners short-circuit.
        if _cache_hit(user.user_id):
            return

        timeout = settings.effective_user_sync_timeout
        try:
            stack = getattr(request.app.state, "storage_stack", None)

            if settings.storage_service_impl == "cached":
                if stack is None:
                    # Happens during a narrow window at startup, or when
                    # cached mode fails to initialize the stack. Don't
                    # WARN; the root cause (if any) is already logged by
                    # the lifespan. Seed a short failure entry so we
                    # don't tight-loop.
                    logger.debug(
                        "USER_SYNC_SKIPPED reason=stack_unready user_id=%s",
                        user.user_id,
                    )
                    _record_sync_outcome(
                        user.user_id,
                        "failure",
                        settings.user_sync_failure_ttl_sec,
                        settings,
                    )
                    return

                from deep_research.services._impl_factory import (
                    make_user_service,
                )

                svc = make_user_service(settings, stack)
                await asyncio.wait_for(
                    svc.upsert(
                        user_id=user.user_id,
                        email=user.email,
                        display_name=user.display_name,
                    ),
                    timeout=timeout,
                )
            else:
                # Legacy SQLAlchemy path — still supported for deployments
                # that pin `storage_service_impl=sqlalchemy_legacy`. Here
                # we own the session (no request-scoped `get_db` because
                # this runs outside the request cycle as a background
                # task).
                from deep_research.db.session import get_session_maker
                from deep_research.services._impl_factory import (
                    make_user_service,
                )

                session_maker = get_session_maker()
                async with session_maker() as session:
                    svc = make_user_service(settings, session=session)
                    await asyncio.wait_for(
                        svc.upsert(
                            user_id=user.user_id,
                            email=user.email,
                            display_name=user.display_name,
                        ),
                        timeout=timeout,
                    )
                    await asyncio.wait_for(
                        session.commit(),
                        timeout=timeout,
                    )

            _record_sync_outcome(
                user.user_id,
                "success",
                settings.user_sync_success_ttl_sec,
                settings,
            )
            logger.info(
                "USER_SYNC_OK user_id=%s backend=%s",
                user.user_id,
                settings.storage_service_impl,
            )
        except Exception:
            # Cache failure briefly to suppress request-rate retries.
            _record_sync_outcome(
                user.user_id,
                "failure",
                settings.user_sync_failure_ttl_sec,
                settings,
            )
            logger.warning(
                "USER_SYNC_FAILED user_id=%s",
                user.user_id,
                exc_info=True,
            )


# --- FastAPI dependencies ----------------------------------------------------


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
            await _ensure_user_synced(user, request, settings)
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
        await _ensure_user_synced(user, request, settings)
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
        await _ensure_user_synced(user, request, settings)
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


# V1.5 OBO refresh hook: caller sets request.state.obo_token_rotated; middleware propagates
# as X-OBO-Token-Rotated header.  The header is injected by the response-finalize helper
# below.  Call site: wherever refresh_user_token() is invoked in a request handler, set
# ``request.state.obo_token_rotated = True`` on successful rotation.


def propagate_obo_rotation_header(request: Request, response: object) -> None:
    """Propagate X-OBO-Token-Rotated response header when a rotation occurred.

    Call this from any middleware or endpoint that has access to the Response
    object after the request handler completes.

    Args:
        request: The current FastAPI request (reads request.state.obo_token_rotated).
        response: The FastAPI Response object to mutate.
    """
    # After the request handler completes and before returning the response:
    if getattr(request.state, "obo_token_rotated", False):
        response.headers["X-OBO-Token-Rotated"] = "true"  # type: ignore[attr-defined]
