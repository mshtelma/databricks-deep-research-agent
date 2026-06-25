"""Fixtures for cached-service contract tests.

Parametrizes the `backend` fixture over the same three implementations as
`tests/contract/storage/conftest.py` (fake / lakebase / sql_warehouse).
The logic is duplicated (not `pytest_plugins`-imported) because pytest
forbids non-top-level `pytest_plugins`.

Wraps the backend in a minimal `StorageStack` — queue running so
append-only writes persist, cleanup + signal handlers skipped.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.storage.backend import StorageBackend
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.factory import StorageStack
from deep_research.storage.queue import WriteQueue


def _backend_params() -> list[pytest.param]:
    params: list[pytest.param] = [pytest.param("fake", id="fake")]
    if os.environ.get("STORAGE_TEST_LAKEBASE") == "1":
        params.append(pytest.param("lakebase", id="lakebase"))
    if os.environ.get("STORAGE_TEST_WAREHOUSE") == "1":
        params.append(pytest.param("sql_warehouse", id="sql_warehouse"))
    return params


@asynccontextmanager
async def _fake_backend_cm() -> AsyncIterator[StorageBackend]:
    b = FakeBackend()
    await b.migrate()
    try:
        yield b
    finally:
        await b.close()


@asynccontextmanager
async def _lakebase_backend_cm() -> AsyncIterator[StorageBackend]:
    from sqlalchemy import text

    from deep_research.core.config import get_settings
    from deep_research.db.session import get_session_maker
    from deep_research.storage.lakebase import LakebaseBackend, _split_sql

    settings = get_settings()
    schema_name = f"deep_research_test_{uuid.uuid4().hex[:12]}"
    sm = get_session_maker(settings)

    async def _run(sql: str) -> None:
        async with sm() as session, session.begin():
            await session.execute(text(sql))

    await _run(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')
    backend = LakebaseBackend(session_maker=sm)
    try:
        async with sm() as session, session.begin():
            await session.execute(text(f'SET search_path TO "{schema_name}"'))
            ddl_text = backend._ddl_path.read_text()
            for stmt in _split_sql(ddl_text):
                await session.execute(text(stmt))

        original = sm

        class _SchemaScopedMaker:
            def __call__(self, *args, **kwargs):  # type: ignore[override]
                session = original(*args, **kwargs)

                async def _enter():
                    await session.__aenter__()
                    await session.execute(
                        text(f'SET search_path TO "{schema_name}"')
                    )
                    return session

                class _CM:
                    async def __aenter__(self_inner):
                        return await _enter()

                    async def __aexit__(self_inner, *exc):
                        await session.__aexit__(*exc)

                return _CM()

        backend._sm = _SchemaScopedMaker()  # type: ignore[assignment]
        yield backend
    finally:
        try:
            await backend.close()
        except Exception:  # noqa: BLE001
            pass
        await _run(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')


@asynccontextmanager
async def _warehouse_backend_cm() -> AsyncIterator[StorageBackend]:
    from deep_research.storage.sql_warehouse import SQLWarehouseBackend

    warehouse_id = os.environ["STORAGE_WAREHOUSE_ID"]
    catalog = os.environ.get("STORAGE_CATALOG", "main")
    schema_prefix = os.environ.get("STORAGE_SCHEMA_PREFIX", "deep_research_test")
    schema_name = f"{schema_prefix}_{uuid.uuid4().hex[:12]}"

    backend = SQLWarehouseBackend(
        warehouse_id=warehouse_id, catalog=catalog, schema=schema_name,
    )
    try:
        await backend.migrate()
        yield backend
    finally:
        try:
            await backend._execute(
                f"DROP SCHEMA IF EXISTS {catalog}.{schema_name} CASCADE"
            )
        except Exception:  # noqa: BLE001
            pass
        await backend.close()


@pytest.fixture(params=_backend_params())
async def backend(request) -> AsyncIterator[StorageBackend]:  # type: ignore[no-untyped-def]
    name = request.param
    cm = {
        "fake": _fake_backend_cm,
        "lakebase": _lakebase_backend_cm,
        "sql_warehouse": _warehouse_backend_cm,
    }[name]
    async with cm() as b:
        yield b


@pytest.fixture
async def stack(backend: StorageBackend) -> AsyncIterator[StorageStack]:
    """Minimal `StorageStack` wrapping the parametrized backend.

    Starts the WriteQueue flush loop so append-only writes are persisted.
    """
    cold_cache = ColdReadCache(ttl_sec=5.0, max_entries=128)
    cache = ChatStateCache(backend, idle_ttl_min=5)
    queue = WriteQueue(
        backend, cache, flush_interval_sec=0.05, flush_size=50,
    )
    cache._on_dirty = queue.notify_dirty  # noqa: SLF001 — deliberate wire-up
    hydrator = Hydrator(cache, backend)

    stk = StorageStack(
        backend=backend,
        cache=cache,
        queue=queue,
        hydrator=hydrator,
        cold_cache=cold_cache,
    )
    stk.cache.start_reaper()
    stk.queue.start(event_tables=("research_events",))
    stk._started = True
    try:
        yield stk
    finally:
        try:
            await stk.queue.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            await stk.cache.stop_reaper()
        except Exception:  # noqa: BLE001
            pass
