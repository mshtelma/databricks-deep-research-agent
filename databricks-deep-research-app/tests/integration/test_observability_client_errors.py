"""Integration tests for POST /api/v1/observability/client-errors (Pillar 3).

Gated by RUN_INTEGRATION_TESTS=1 (matches the metrics-endpoint test convention). The
MetricsSink is swapped for a RecordingSink so no external services are touched.

Run:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_observability_client_errors.py -v
"""
from __future__ import annotations

import logging
import os

import pytest

# Must be set before importing `app` so Settings() validation does not require LAKEBASE_*.
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

_RUN_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"
if not _RUN_TESTS:
    pytest.skip("Requires RUN_INTEGRATION_TESTS=1", allow_module_level=True)

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402
from deep_research.storage.observability import RecordingSink, use_sink  # noqa: E402

_TEST_USER = UserIdentity(
    user_id="obs-test-user",
    email="obs-test@test.example",
    display_name="Obs Test User",
)


def _override_user() -> UserIdentity:
    return _TEST_USER


@pytest.fixture()
def client() -> TestClient:
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_current_user_identity, None)


@pytest.fixture(autouse=True)
def _reset_rate_limit() -> None:  # type: ignore[return]
    from deep_research.api.v1.observability import _user_rate

    _user_rate.clear()
    yield
    _user_rate.clear()


def test_happy_path_logs_warning_and_counts(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Valid report → 200, a WARNING log line, and a client_error counter emission."""
    sink = RecordingSink()
    with use_sink(sink), caplog.at_level(
        logging.WARNING, logger="deep_research.api.v1.observability"
    ):
        resp = client.post(
            "/api/v1/observability/client-errors",
            json={
                "kind": "render",
                "message": "Cannot read properties of undefined (reading 'length')",
                "boundary_name": "Surface",
                "route": "/chat",
                "bundle_id": "abc123",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1
    # Landed in the server logs (the whole point).
    assert any(
        "client_error" in r.message and "boundary=Surface" in r.getMessage()
        for r in caplog.records
    )
    # Emitted a counter for aggregation.
    assert any(e.name == "client_error" for e in sink.emissions)


def test_unknown_field_returns_422(client: TestClient) -> None:
    """extra='forbid' rejects unknown fields."""
    resp = client.post(
        "/api/v1/observability/client-errors",
        json={"kind": "render", "message": "boom", "evil": "x"},
    )
    assert resp.status_code == 422


def test_bad_kind_returns_422(client: TestClient) -> None:
    """kind outside the Literal is rejected."""
    resp = client.post(
        "/api/v1/observability/client-errors",
        json={"kind": "not-a-kind", "message": "boom"},
    )
    assert resp.status_code == 422


def test_oversize_field_returns_422(client: TestClient) -> None:
    """A stack past its 8 KiB field cap is the reachable oversize guard (Pydantic 422);
    the raw 24 KiB body check is defense-in-depth behind it."""
    resp = client.post(
        "/api/v1/observability/client-errors",
        json={"kind": "render", "message": "boom", "stack": "x" * 9000},
    )
    assert resp.status_code == 422


def test_rate_limit_returns_429(client: TestClient) -> None:
    """31st report within the minute window → 429 (per-minute cap)."""
    from deep_research.api.v1.observability import RATE_LIMIT_PER_MIN

    base_ts = 1_000_000.0
    call_count = 0

    def _mock_time() -> float:
        nonlocal call_count
        t = base_ts + call_count  # 1s apart → never hits per-sec cap
        call_count += 1
        return t

    sink = RecordingSink()
    with use_sink(sink), patch(
        "deep_research.api.v1.observability.time", _mock_time
    ):
        for i in range(RATE_LIMIT_PER_MIN):
            r = client.post(
                "/api/v1/observability/client-errors",
                json={"kind": "render", "message": f"e{i}"},
            )
            assert r.status_code == 200, f"req {i + 1} got {r.status_code}"
        over = client.post(
            "/api/v1/observability/client-errors",
            json={"kind": "render", "message": "over"},
        )
    assert over.status_code == 429
