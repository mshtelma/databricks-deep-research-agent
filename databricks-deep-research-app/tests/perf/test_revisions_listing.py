"""Performance test: B-tree index validates sub-100ms p95 at 1k revisions.

Seeds 1000 AgentRevision rows for one agent, then times 50 sequential
GET /revisions?limit=20 calls.  Asserts that the p95 latency is < 100 ms,
which validates that the B-tree index on (agent_id, created_at DESC) is
doing its job.

Skipped unless ``RUN_INTEGRATION_TESTS=1`` is set (requires real DB).

Run:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/perf/test_revisions_listing.py -v -s
"""

from __future__ import annotations

import contextlib
import os
import statistics
import time
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.db.session import get_db  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402
from deep_research.models.agent_v2 import AgentRevision, AgentV2  # noqa: E402
from deep_research.models.visibility import AgentVisibility  # noqa: E402

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_RUN_DB_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

_DB_SKIP = pytest.mark.skipif(
    not _RUN_DB_TESTS,
    reason="Requires a real database connection; set RUN_INTEGRATION_TESTS=1 to enable",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_REVISIONS = 1000
_NUM_REQUESTS = 50
_P95_THRESHOLD_MS = 100.0

_PERF_USER = UserIdentity(
    user_id="perf-revisions-user",
    email="perf-rev@test.example",
    display_name="Perf Rev User",
)

_MINIMAL_DEFINITION: dict[str, Any] = {
    "id": "perf-wf",
    "name": "Perf Workflow",
    "version": 1,
    "root": {
        "id": "root-node",
        "type": "agent",
        "label": "researcher",
        "config": {"subtype": "researcher"},
        "children": [],
    },
    "tools": [],
    "pools": [],
    "sources": [],
    "models": {},
    "required_inputs": ["query"],
    "output_keys": ["output"],
    "token_budget": 0,
    "timeout_seconds": 1800,
}

# ---------------------------------------------------------------------------
# DB session fixture (same pattern as integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session() -> Any:
    import deep_research.db.session as _db_mod
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from deep_research.core.config import get_settings
    from deep_research.db.session import get_database_url

    _stale_engines: list[object] = []

    settings = get_settings()
    try:
        url = get_database_url(settings)
    except ValueError as exc:
        pytest.skip(f"No database configured — {exc}")

    if _db_mod._engine is not None:
        _stale_engines.append(_db_mod._engine)
        _db_mod._engine = None
        _db_mod._async_session_maker = None

    connect_args = {"ssl": True} if settings.use_lakebase else {}
    engine = create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with maker() as session:
        yield session
        await session.rollback()

    await engine.dispose()


# ---------------------------------------------------------------------------
# Helper: test client
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _db_client(
    user: UserIdentity,
    db_session: Any,
) -> Generator[TestClient, None, None]:
    async def _override_db() -> Any:
        yield db_session

    async def _override_user() -> UserIdentity:
        return user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        yield TestClient(app, raise_server_exceptions=True)
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


# ---------------------------------------------------------------------------
# Perf test
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_revisions_listing_p95_under_100ms(db_session: Any) -> None:
    """Seed 1000 revisions, time 50 GET /revisions calls, assert p95 < 100ms.

    This validates that the B-tree index on (agent_id, created_at DESC) keeps
    listing fast even at high revision counts.
    """
    import asyncio

    # --- seed data via raw async session ---
    async def _seed() -> str:
        agent_id = uuid.uuid4()
        now = datetime.now(UTC)
        agent = AgentV2(
            id=agent_id,
            owner_id=_PERF_USER.user_id,
            name="Perf Agent 1k Revisions",
            description=None,
            avatar_url=None,
            visibility=AgentVisibility.PRIVATE.value,
            definition=_MINIMAL_DEFINITION,
            schema_version=1,
            etag="seed-etag-initial",
            created_at=now,
            updated_at=now,
        )
        db_session.add(agent)
        await db_session.flush()

        revisions = [
            AgentRevision(
                rev_id=uuid.uuid4(),
                agent_id=agent_id,
                etag=f"etag-{i:04d}",
                definition=_MINIMAL_DEFINITION,
                created_at=now - timedelta(seconds=(_NUM_REVISIONS - i)),
                created_by=_PERF_USER.user_id,
            )
            for i in range(_NUM_REVISIONS)
        ]
        db_session.add_all(revisions)
        await db_session.commit()
        return str(agent_id)

    agent_id = asyncio.get_event_loop().run_until_complete(_seed())

    # --- time listing calls ---
    latencies: list[float] = []
    with _db_client(_PERF_USER, db_session) as client:
        for _ in range(_NUM_REQUESTS):
            t0 = time.monotonic()
            resp = client.get(f"/api/v1/agents-v2/{agent_id}/revisions?limit=20")
            elapsed_ms = (time.monotonic() - t0) * 1000
            assert resp.status_code == 200, resp.text
            latencies.append(elapsed_ms)

    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95_ms = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
    mean_ms = statistics.mean(latencies)

    print(
        f"\n[perf] {_NUM_REQUESTS} requests against {_NUM_REVISIONS} revisions: "
        f"mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms  threshold={_P95_THRESHOLD_MS}ms"
    )

    assert p95_ms < _P95_THRESHOLD_MS, (
        f"p95 latency {p95_ms:.1f}ms exceeds {_P95_THRESHOLD_MS}ms threshold — "
        f"B-tree index may be missing or not being used"
    )
