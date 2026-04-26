"""Parametrized fixtures over every `StorageBackend` implementation.

The `backend` fixture is the axis every contract test depends on. It yields:

* `FakeBackend` — always.
* `LakebaseBackend` — gated on `STORAGE_TEST_LAKEBASE=1` (plus Lakebase creds
  already picked up by `Settings`). Creates a temp schema, runs `migrate()`,
  drops the schema on teardown regardless of outcome.
* `SQLWarehouseBackend` — gated on `STORAGE_TEST_WAREHOUSE=1`. Requires
  `STORAGE_WAREHOUSE_ID`, `STORAGE_CATALOG`, `STORAGE_SCHEMA_PREFIX` (default
  `deep_research_test`).

Every real-backend fixture isolates itself into its own schema so parallel
runs do not collide.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from deep_research.storage.backend import StorageBackend
from tests.fakes.fake_backend import FakeBackend


# --- Param generators -----------------------------------------------------


def _backend_params() -> list[pytest.param]:
    params: list[pytest.param] = [pytest.param("fake", id="fake")]
    if os.environ.get("STORAGE_TEST_LAKEBASE") == "1":
        params.append(pytest.param("lakebase", id="lakebase"))
    if os.environ.get("STORAGE_TEST_WAREHOUSE") == "1":
        params.append(pytest.param("sql_warehouse", id="sql_warehouse"))
    return params


# --- Fake ------------------------------------------------------------------


@asynccontextmanager
async def _fake_backend_cm() -> AsyncIterator[StorageBackend]:
    b = FakeBackend()
    await b.migrate()
    try:
        yield b
    finally:
        await b.close()


# --- Lakebase --------------------------------------------------------------


@asynccontextmanager
async def _lakebase_backend_cm() -> AsyncIterator[StorageBackend]:
    """Provisions a temp Postgres schema, runs DDL, tears down on exit."""
    from sqlalchemy import text

    from deep_research.core.config import get_settings
    from deep_research.db.session import get_session_maker
    from deep_research.storage.lakebase import LakebaseBackend

    settings = get_settings()
    schema_name = f"deep_research_test_{uuid.uuid4().hex[:12]}"
    sm = get_session_maker(settings)

    async def _run(sql: str, **params) -> None:
        async with sm() as session:
            async with session.begin():
                await session.execute(text(sql), params)

    # Create temp schema and apply DDL into it by replacing plain identifiers.
    # The Lakebase DDL uses unqualified table names — we rewrite it for the
    # temp schema via `SET search_path`.
    await _run(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

    backend = LakebaseBackend(session_maker=sm)

    try:
        async with sm() as session:
            async with session.begin():
                await session.execute(text(f'SET search_path TO "{schema_name}"'))
                # Re-run migrate under the new search_path.
                ddl_text = backend._ddl_path.read_text()
                from deep_research.storage.lakebase import _split_sql

                for stmt in _split_sql(ddl_text):
                    await session.execute(text(stmt))
        # Subsequent operations need a session whose search_path is already
        # set. Monkey-patch the session maker:
        original = sm
        from sqlalchemy.ext.asyncio import async_sessionmaker

        async def _set_search_path(session):  # type: ignore[no-untyped-def]
            await session.execute(text(f'SET search_path TO "{schema_name}"'))

        # Wrap session maker so every session sets search_path first.
        class _SchemaScopedMaker:
            def __call__(self, *args, **kwargs):  # type: ignore[override]
                session = original(*args, **kwargs)
                async def _enter():
                    await session.__aenter__()
                    await _set_search_path(session)
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


# --- SQL Warehouse ---------------------------------------------------------


@asynccontextmanager
async def _warehouse_backend_cm() -> AsyncIterator[StorageBackend]:
    """Provisions a temp UC schema, runs DDL, drops on exit."""
    from deep_research.storage.sql_warehouse import SQLWarehouseBackend

    warehouse_id = os.environ["STORAGE_WAREHOUSE_ID"]
    catalog = os.environ.get("STORAGE_CATALOG", "main")
    schema_prefix = os.environ.get("STORAGE_SCHEMA_PREFIX", "deep_research_test")
    schema_name = f"{schema_prefix}_{uuid.uuid4().hex[:12]}"

    backend = SQLWarehouseBackend(
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema_name,
    )
    try:
        await backend.migrate()
        yield backend
    finally:
        try:
            # Best-effort drop. CASCADE removes every table created by migrate.
            await backend._execute(f"DROP SCHEMA IF EXISTS {catalog}.{schema_name} CASCADE")
        except Exception:  # noqa: BLE001
            pass
        await backend.close()


# --- The pytest fixture --------------------------------------------------


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
