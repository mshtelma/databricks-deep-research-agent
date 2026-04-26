"""Integration smoke test for Lakebase PgBouncer safety.

Exercises the PgBouncer-safe asyncpg configuration by issuing 50 sequential
`UserService.upsert` calls against a real Lakebase instance. Before the
statement-cache fix, this pattern would hang in `asyncpg.protocol.prepare()`
or raise `DuplicatePreparedStatementError`.

Gated by `LAKEBASE_INTEGRATION=1` plus the usual Lakebase env vars
(`PGHOST`, `PGUSER`, `DATABRICKS_HOST`, service principal credentials, etc.)
so it does not run in normal CI.

Run with:
    LAKEBASE_INTEGRATION=1 uv run pytest \
        tests/integration/test_lakebase_user_upsert.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from statistics import median

import pytest

from deep_research.core.config import get_settings
from deep_research.db.session import get_session_maker
from deep_research.services.user_service import UserService

_INTEGRATION = os.environ.get("LAKEBASE_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not _INTEGRATION,
    reason="requires Lakebase; set LAKEBASE_INTEGRATION=1 to enable",
)

_ITERATIONS = 50
_PER_CALL_TIMEOUT = 5.0
_P95_BUDGET_SECONDS = 0.5  # 500 ms after cache fix


@pytest.mark.asyncio
async def test_upsert_survives_pgbouncer_churn() -> None:
    """50 sequential upserts must complete without hang or duplicate-stmt error."""
    settings = get_settings()
    assert settings.use_lakebase, (
        "Test requires use_lakebase=True; set PGHOST or ENDPOINT_NAME"
    )

    maker = get_session_maker(settings)
    latencies: list[float] = []
    run_id = uuid.uuid4().hex[:8]

    for i in range(_ITERATIONS):
        user_id = f"integ-test-{run_id}-{i}"
        started = time.monotonic()
        async with maker() as session:
            svc = UserService(session)
            await asyncio.wait_for(
                svc.upsert(
                    user_id=user_id,
                    email=f"{user_id}@example.invalid",
                    display_name=f"Integ Test {i}",
                ),
                timeout=_PER_CALL_TIMEOUT,
            )
            await asyncio.wait_for(session.commit(), timeout=_PER_CALL_TIMEOUT)
        latencies.append(time.monotonic() - started)

    latencies.sort()
    p50 = median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    p99 = latencies[int(len(latencies) * 0.99) - 1]

    print(f"\nLakebase upsert latency: p50={p50*1000:.1f}ms "
          f"p95={p95*1000:.1f}ms p99={p99*1000:.1f}ms")

    # p95 budget is intentionally loose — cold starts can breach briefly.
    # The test's primary assertion is "zero hangs, zero errors".
    assert p95 < _P95_BUDGET_SECONDS, (
        f"p95 latency {p95*1000:.1f}ms exceeded budget {_P95_BUDGET_SECONDS*1000}ms — "
        "possible regression of the asyncpg/PgBouncer prepared-statement issue"
    )
