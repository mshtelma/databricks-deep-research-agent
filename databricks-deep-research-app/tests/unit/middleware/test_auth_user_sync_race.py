"""Race-prevention tests for the synchronous user-sync (PR-1.2 / migration 021).

Migration 021 adds FK constraints from ``chats.user_id`` (and friends) to
``users.user_id``. The original auth flow scheduled the user-row upsert
as a fire-and-forget background task, which races the very first
FK-bearing INSERT in the same request: the chat insert can fire before
the ``users`` row is durable, hitting an FK violation that the original
``hasattr(db, "commit"); except: pass`` cleanup silently swallowed.

The fix in ``middleware/auth.py`` is to await ``_ensure_user_synced``
synchronously on cache miss. These tests prove the synchronous semantics
hold under concurrent first-requests for the same user.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.core.auth import UserIdentity
from deep_research.middleware import auth as auth_module
from deep_research.middleware.auth import (
    _ensure_user_synced,
    _user_sync_cache,
    _user_sync_locks,
)


def _user(user_id: str = "u_race") -> UserIdentity:
    return UserIdentity(user_id=user_id, email=f"{user_id}@x.io", display_name=user_id)


def _settings() -> Any:
    s = MagicMock()
    s.storage_service_impl = "cached"
    s.storage_backend = "lakebase"
    s.user_sync_enabled = True
    s.user_sync_failure_ttl_sec = 30
    s.user_sync_success_ttl_sec = 300
    s.user_sync_max_cache = 1024
    s.user_sync_lock_ttl_sec = 60
    s.effective_user_sync_timeout = 15.0
    s.user_sync_timeout_sec = None
    return s


def _request(stack: Any | None = None) -> Any:
    request = MagicMock()
    request.app.state = SimpleNamespace(storage_stack=stack)
    return request


@pytest.fixture(autouse=True)
def _reset_module_state():
    _user_sync_cache.clear()
    _user_sync_locks.clear()
    auth_module._user_sync_lock_touched.clear()
    yield
    _user_sync_cache.clear()
    _user_sync_locks.clear()
    auth_module._user_sync_lock_touched.clear()


@pytest.mark.asyncio
async def test_ensure_user_synced_blocks_until_user_row_durable() -> None:
    """The synchronous semantic: when ``_ensure_user_synced`` returns,
    the user-service has durably persisted the row. A subsequent
    FK-bearing INSERT therefore cannot race the upsert."""
    backend = FakeBackend(latency_ms=20)
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()
    user = _user()

    # Track upsert-completion order vs. our caller-side observation.
    upsert_completed = asyncio.Event()
    original_write_user_doc = backend.write_user_doc

    async def _instrumented_write(*args, **kwargs):
        result = await original_write_user_doc(*args, **kwargs)
        upsert_completed.set()
        return result

    backend.write_user_doc = _instrumented_write  # type: ignore[assignment]

    await _ensure_user_synced(user, request, settings)

    # By the time _ensure_user_synced returns, the upsert MUST have
    # completed. If this fails the synchronous contract is broken and
    # the FK race is back.
    assert upsert_completed.is_set(), (
        "user upsert must complete BEFORE _ensure_user_synced returns; "
        "otherwise downstream FK-bearing INSERTs race the users row"
    )


@pytest.mark.asyncio
async def test_concurrent_first_requests_same_user_dedupe_to_one_write() -> None:
    """Ten simultaneous first-requests for the same brand-new user must
    coalesce into a single backend upsert. This is the thundering-herd
    invariant: an FK violation only happens if the upsert hasn't landed,
    and serialisation via the per-user lock + cache short-circuit is
    what makes the synchronous path cheap in steady state."""
    backend = FakeBackend(latency_ms=10)
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()
    user = _user(user_id="u_thunder")

    upsert_calls = 0
    original_write_user_doc = backend.write_user_doc

    async def _counted_write(*args, **kwargs):
        nonlocal upsert_calls
        upsert_calls += 1
        return await original_write_user_doc(*args, **kwargs)

    backend.write_user_doc = _counted_write  # type: ignore[assignment]

    await asyncio.gather(
        *(_ensure_user_synced(user, request, settings) for _ in range(10))
    )

    # All 10 callers see a durable users row; under the lock + cache,
    # the backend should be hit at most twice (one winner + at most one
    # late entrant before the cache populates).
    assert upsert_calls <= 2, (
        f"expected ≤2 upserts under thundering-herd dedup, got {upsert_calls}"
    )


@pytest.mark.asyncio
async def test_anonymous_does_not_hit_backend() -> None:
    """Anonymous fallback uses the migration-021 sentinel row and must
    never trigger an upsert during the request path — otherwise dev-mode
    would issue a write per request."""
    backend = FakeBackend(latency_ms=0)
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()

    upsert_calls = 0
    original_write_user_doc = backend.write_user_doc

    async def _counted_write(*args, **kwargs):
        nonlocal upsert_calls
        upsert_calls += 1
        return await original_write_user_doc(*args, **kwargs)

    backend.write_user_doc = _counted_write  # type: ignore[assignment]

    await _ensure_user_synced(UserIdentity.anonymous(), request, settings)
    assert upsert_calls == 0, "anonymous user must not hit the backend"
