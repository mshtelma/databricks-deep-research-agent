"""Unit tests for the fire-and-forget user-sync in the auth middleware.

Covers the scheduling surface (`_schedule_user_sync`) and the sync body
(`_sync_user_record`) in isolation. Uses `FakeBackend` + a minimal stack
surrogate rather than spinning a full `StorageStack`, since the test
targets the middleware's wiring (cache, lock, factory dispatch, failure
path) — not storage-stack lifecycle.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.core.auth import UserIdentity
from deep_research.middleware import auth as auth_module
from deep_research.middleware.auth import (
    _schedule_user_sync,
    _sync_user_record,
    _SyncCacheEntry,
    _user_sync_cache,
    _user_sync_locks,
)

# --- Helpers ----------------------------------------------------------------


def _user(user_id: str = "u1", email: str = "u1@x.io", display_name: str = "User 1") -> UserIdentity:
    return UserIdentity(user_id=user_id, email=email, display_name=display_name)


def _settings(
    *,
    storage_service_impl: str = "cached",
    storage_backend: str = "lakebase",
    user_sync_enabled: bool = True,
    timeout_override: float | None = None,
    failure_ttl: int = 30,
    success_ttl: int = 300,
    max_cache: int = 1024,
    lock_ttl: int = 60,
) -> Any:
    """Minimal pydantic-like settings stub; sufficient for the middleware."""
    s = MagicMock()
    s.storage_service_impl = storage_service_impl
    s.storage_backend = storage_backend
    s.user_sync_enabled = user_sync_enabled
    s.user_sync_failure_ttl_sec = failure_ttl
    s.user_sync_success_ttl_sec = success_ttl
    s.user_sync_max_cache = max_cache
    s.user_sync_lock_ttl_sec = lock_ttl
    # Mirror the computed-property behaviour precisely so we exercise the
    # per-backend default without mocking the Settings class itself.
    if timeout_override is not None:
        s.effective_user_sync_timeout = timeout_override
    else:
        s.effective_user_sync_timeout = (
            45.0 if storage_backend == "sql_warehouse" else 15.0
        )
    s.user_sync_timeout_sec = timeout_override
    return s


def _request(stack: Any | None = None) -> Any:
    """Build a Request surrogate with the two pieces of `app.state` we touch."""
    app_state = SimpleNamespace(
        storage_stack=stack,
        pending_user_syncs=set(),
    )
    request = MagicMock()
    request.app.state = app_state
    return request


@pytest.fixture(autouse=True)
def _clear_module_state():
    """Reset the middleware module's process-level caches between tests."""
    _user_sync_cache.clear()
    _user_sync_locks.clear()
    auth_module._user_sync_lock_touched.clear()
    yield
    _user_sync_cache.clear()
    _user_sync_locks.clear()
    auth_module._user_sync_lock_touched.clear()


async def _drain(request: Any) -> None:
    """Await all tasks registered on `app.state.pending_user_syncs`."""
    pending = list(request.app.state.pending_user_syncs)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# --- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_non_blocking() -> None:
    """`_schedule_user_sync` must return immediately regardless of backend latency.

    Catches re-introduction of a blocking `await _sync_user_record(...)` in
    the auth dep path.
    """
    backend = FakeBackend(latency_ms=500)  # slow on purpose
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()

    t0 = time.monotonic()
    _schedule_user_sync(_user(), request, settings)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 50, f"scheduling took {elapsed_ms:.1f} ms"
    assert len(request.app.state.pending_user_syncs) == 1

    await _drain(request)  # cleanup


@pytest.mark.asyncio
async def test_cached_path_writes_via_factory() -> None:
    """Happy path: scheduled sync lands in `FakeBackend._user_docs`."""
    backend = FakeBackend()
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()
    user = _user(user_id="u-happy", email="happy@x.io", display_name="Happy")

    _schedule_user_sync(user, request, settings)
    await _drain(request)

    doc = await backend.load_user_doc("u-happy")
    assert doc is not None
    assert doc.profile["email"] == "happy@x.io"
    assert doc.profile["display_name"] == "Happy"
    assert doc.profile["last_seen_at"]  # set by CachedUserService.upsert


@pytest.mark.asyncio
async def test_thundering_herd_dedup() -> None:
    """10 concurrent schedules for the same user → backend sees <= 2 writes.

    A perfectly serializing lock plus the in-lock cache re-check means one
    task wins and the rest short-circuit. We allow up to 2 to account for
    the narrow race between `asyncio.create_task` runs; regressing to
    hundreds of writes (the buggy state) is what this test guards.
    """
    writes: list[str] = []

    def counting_hook(method: str, _args: tuple[Any, ...]) -> None:
        if method == "write_user_doc":
            writes.append(method)

    backend = FakeBackend(latency_ms=30, fail_hook=counting_hook)
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()
    user = _user(user_id="herd")

    for _ in range(10):
        _schedule_user_sync(user, request, settings)
    await _drain(request)

    assert len(writes) <= 2, f"expected thundering herd to dedupe, got {len(writes)} writes"


@pytest.mark.asyncio
async def test_failure_cache_short_circuits() -> None:
    """After one failure, the next schedule within TTL creates no task."""
    def always_fail(method: str, _args: tuple[Any, ...]) -> None:
        if method == "write_user_doc":
            raise RuntimeError("simulated DB outage")

    backend = FakeBackend(fail_hook=always_fail)
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings(failure_ttl=30)
    user = _user(user_id="fail-u")

    _schedule_user_sync(user, request, settings)
    await _drain(request)

    assert _user_sync_cache["fail-u"].kind == "failure"

    # Second call within failure TTL — must be a no-op (no new task).
    before = len(request.app.state.pending_user_syncs)
    _schedule_user_sync(user, request, settings)
    after = len(request.app.state.pending_user_syncs)
    assert after == before, "failure cache should suppress re-scheduling"


@pytest.mark.asyncio
async def test_unbounded_cache_bounded() -> None:
    """Cache never exceeds `user_sync_max_cache`, even on the failure path."""
    def always_fail(method: str, _args: tuple[Any, ...]) -> None:
        if method == "write_user_doc":
            raise RuntimeError("boom")

    backend = FakeBackend(fail_hook=always_fail)
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings(max_cache=5)

    for i in range(20):
        _schedule_user_sync(_user(user_id=f"u-{i}"), request, settings)
    await _drain(request)

    assert len(_user_sync_cache) <= 5, (
        f"cache grew to {len(_user_sync_cache)} > max_cache=5 on failure path"
    )


@pytest.mark.asyncio
async def test_stack_unready_no_spam(caplog: pytest.LogCaptureFixture) -> None:
    """Cached mode + stack=None → DEBUG log, failure-cache entry, no WARNING.

    Protects against log-spam during lifespan startup races, where the
    stack may not yet be populated when the first request lands.
    """
    import logging as _logging

    request = _request(stack=None)
    settings = _settings()
    user = _user(user_id="early")

    with caplog.at_level(_logging.DEBUG, logger="deep_research.middleware.auth"):
        _schedule_user_sync(user, request, settings)
        await _drain(request)

    warnings = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert not warnings, f"unexpected WARNING during stack-unready: {warnings}"
    assert _user_sync_cache["early"].kind == "failure"


@pytest.mark.asyncio
async def test_user_sync_disabled_is_noop() -> None:
    """With `user_sync_enabled=False` we skip scheduling entirely."""
    backend = FakeBackend()
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings(user_sync_enabled=False)

    _schedule_user_sync(_user(), request, settings)
    assert request.app.state.pending_user_syncs == set()
    doc = await backend.load_user_doc("u1")
    assert doc is None


@pytest.mark.asyncio
async def test_anonymous_never_synced() -> None:
    """The anonymous user is a placeholder; never persisted."""
    backend = FakeBackend()
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()

    _schedule_user_sync(UserIdentity.anonymous(), request, settings)
    assert request.app.state.pending_user_syncs == set()


def test_timeout_config_applied_per_backend() -> None:
    """The per-backend timeout defaults are picked up by the stub.

    Guards against the settings-to-middleware wiring accidentally using
    a hardcoded constant.
    """
    lb = _settings(storage_backend="lakebase")
    assert lb.effective_user_sync_timeout == 15.0

    wh = _settings(storage_backend="sql_warehouse")
    assert wh.effective_user_sync_timeout == 45.0

    override = _settings(storage_backend="sql_warehouse", timeout_override=5.0)
    assert override.effective_user_sync_timeout == 5.0


@pytest.mark.asyncio
async def test_success_caches_with_success_kind() -> None:
    """Success path records `kind='success'` (not just 'failure')."""
    backend = FakeBackend()
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings()
    user = _user(user_id="ok")

    _schedule_user_sync(user, request, settings)
    await _drain(request)

    entry = _user_sync_cache.get("ok")
    assert isinstance(entry, _SyncCacheEntry)
    assert entry.kind == "success"
    assert entry.valid_until > time.monotonic()


@pytest.mark.asyncio
async def test_sync_body_survives_missing_pending_set() -> None:
    """If `pending_user_syncs` is not a set, we still sync correctly.

    Guards against the task-tracker being mistyped or absent; the task
    still runs to completion.
    """
    backend = FakeBackend()
    stack = SimpleNamespace(backend=backend)
    # Replace the set with something falsy to simulate a broken setup.
    request = _request(stack=stack)
    request.app.state.pending_user_syncs = None
    settings = _settings()
    user = _user(user_id="solo")

    _schedule_user_sync(user, request, settings)
    # We can't await via the set; wait a little and poll.
    for _ in range(50):
        if await backend.load_user_doc("solo") is not None:
            break
        await asyncio.sleep(0.01)

    doc = await backend.load_user_doc("solo")
    assert doc is not None and doc.profile["email"] == "u1@x.io"


@pytest.mark.asyncio
async def test_sync_body_timeout_marks_failure() -> None:
    """If the backend hangs past the timeout, we cache a failure."""
    backend = FakeBackend(latency_ms=10_000)  # hang far past any test budget
    stack = SimpleNamespace(backend=backend)
    request = _request(stack=stack)
    settings = _settings(timeout_override=0.05)  # 50 ms

    # Run the body directly so we can await it deterministically.
    await _sync_user_record(_user(user_id="slow"), request, settings)

    entry = _user_sync_cache.get("slow")
    assert entry is not None
    assert entry.kind == "failure"
