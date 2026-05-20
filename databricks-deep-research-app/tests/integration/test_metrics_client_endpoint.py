"""Integration tests for POST /api/v1/metrics/client.

Tests are gated by RUN_INTEGRATION_TESTS=1 to match Phase 1 conventions.
The MetricsSink is swapped for a RecordingSink so no real external services
are touched.

Run:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_metrics_client_endpoint.py -v

Skip (clean module-level skip when env var absent):
    uv run pytest tests/integration/test_metrics_client_endpoint.py -q
"""
from __future__ import annotations

import json
import os

import pytest

# Must be set before importing `app` so that Settings() validation does not
# require LAKEBASE_*/DATABASE_URL.
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

# ---------------------------------------------------------------------------
# Module-level skip guard — matches Phase 1 pattern exactly
# ---------------------------------------------------------------------------

_RUN_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

if not _RUN_TESTS:
    pytest.skip("Requires RUN_INTEGRATION_TESTS=1", allow_module_level=True)

# ---------------------------------------------------------------------------
# Deferred imports (only reached when RUN_INTEGRATION_TESTS=1)
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402
from deep_research.storage.observability import RecordingSink, use_sink  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test user / dependency override
# ---------------------------------------------------------------------------

_TEST_USER = UserIdentity(
    user_id="metrics-test-user",
    email="metrics-test@test.example",
    display_name="Metrics Test User",
)


def _override_user() -> UserIdentity:
    return _TEST_USER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient with auth dependency overridden to bypass real auth."""
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_current_user_identity, None)


@pytest.fixture(autouse=True)
def _reset_rate_limit() -> None:  # type: ignore[return]
    """Clear per-user rate-limit state between tests to prevent cross-test bleed."""
    from deep_research.api.v1.metrics import _user_rate

    _user_rate.clear()
    yield
    _user_rate.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_emits_to_sink(client: TestClient) -> None:
    """POST with 3 valid events → 200, sink receives 3 emissions."""
    sink = RecordingSink()
    with use_sink(sink):
        resp = client.post(
            "/api/v1/metrics/client",
            json={
                "events": [
                    {"name": "block_render_count", "value": 5, "timestamp_ms": 1000},
                    {"name": "dnd_drop_failed", "timestamp_ms": 1001},
                    {"name": "revisions_tab_opened", "timestamp_ms": 1002},
                ]
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 3
    # block_render_count has a value → histogram; the other two → counter
    assert len(sink.emissions) == 3
    names = {e.name for e in sink.emissions}
    assert names == {
        "agent_designer.block_render_count",
        "agent_designer.dnd_drop_failed",
        "agent_designer.revisions_tab_opened",
    }


def test_oversize_returns_413(client: TestClient) -> None:
    """POST with body >1 KiB → 413."""
    # Build a payload that exceeds 1 KiB: pad a label value.
    big_label = "x" * 1100
    resp = client.post(
        "/api/v1/metrics/client",
        content=json.dumps(
            {
                "events": [
                    {
                        "name": "block_render_count",
                        "labels": {"big": big_label},
                        "timestamp_ms": 1000,
                    }
                ]
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_rate_limit_returns_429(client: TestClient) -> None:
    """POST 61 batches spread across a minute window → 429 on the 61st.

    We patch ``deep_research.api.v1.metrics.time`` so requests appear to arrive
    at 1-second intervals (well below the 10/sec per-second cap) but all within
    the 60-second sliding window so the per-minute cap is what triggers.
    """
    from deep_research.api.v1.metrics import RATE_LIMIT_PER_MIN

    # Base timestamp; each request is 1 second apart so per-sec limit (10)
    # is never hit, but per-min limit (60) is exhausted after 60 calls.
    base_ts = 1_000_000.0
    call_count = 0

    def _mock_time() -> float:
        nonlocal call_count
        # Advance by 1 second per call so per-sec window only ever has 1 event.
        t = base_ts + call_count
        call_count += 1
        return t

    sink = RecordingSink()
    with use_sink(sink), patch("deep_research.api.v1.metrics.time", _mock_time):
        for i in range(RATE_LIMIT_PER_MIN):
            r = client.post(
                "/api/v1/metrics/client",
                json={"events": [{"name": "dnd_drop_failed", "timestamp_ms": i}]},
            )
            assert r.status_code == 200, f"Expected 200 on request {i + 1}, got {r.status_code}"

        # The 61st request must be rate-limited.
        over_limit = client.post(
            "/api/v1/metrics/client",
            json={"events": [{"name": "dnd_drop_failed", "timestamp_ms": 9999}]},
        )
    assert over_limit.status_code == 429


def test_unknown_signal_dropped_silently(client: TestClient) -> None:
    """POST with an unknown signal name → 200 but sink receives 0 emissions."""
    sink = RecordingSink()
    with use_sink(sink):
        resp = client.post(
            "/api/v1/metrics/client",
            json={"events": [{"name": "evil_signal", "timestamp_ms": 1000}]},
        )
    assert resp.status_code == 200
    # "accepted" is the raw input count; sink should NOT have been called.
    assert len(sink.emissions) == 0


def test_malformed_json_returns_422(client: TestClient) -> None:
    """POST with invalid JSON → 422 (FastAPI validation error)."""
    resp = client.post(
        "/api/v1/metrics/client",
        content=b"not-valid-json{{{",
        headers={"Content-Type": "application/json"},
    )
    # FastAPI returns 422 for JSON parse failures (Pydantic validation).
    assert resp.status_code == 422
