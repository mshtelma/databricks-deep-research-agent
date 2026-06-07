"""Integration tests for GET /agents-v2/{id}/yaml (YAML export endpoint).

Tests covered:
  1.  test_export_returns_text_yaml_content_type
  2.  test_export_includes_registry_version
  3.  test_export_round_trips_via_load_workflow_from_dict
  4.  test_export_is_deterministic
  5.  test_export_owner_scoping
  6.  test_export_emits_yaml_export_ms

DB-backed tests (1-5) require a working Lakebase / PostgreSQL connection and are
skipped unless ``RUN_INTEGRATION_TESTS=1`` is set.

Metric test (6) is pure computation and always runs.

Run all:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_yaml_export.py -v

Run metric test only (no DB needed):
    uv run pytest tests/integration/test_yaml_export.py -v -k "emits"
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

# Must be set before importing `app` so that Settings() validation does not
# require LAKEBASE_*/DATABASE_URL (those are only checked when
# storage_service_impl == "cached"). The DB-backed tests inject a real session
# via dependency override; they do NOT go through the storage stack.
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from deep_research.agent_designer.registry import REGISTRY_VERSION  # noqa: E402
from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.db.session import get_db  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402

# ---------------------------------------------------------------------------
# Skip guard for DB-backed tests
# ---------------------------------------------------------------------------

_RUN_DB_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

_DB_SKIP = pytest.mark.skipif(
    not _RUN_DB_TESTS,
    reason=(
        "Requires a real database connection; "
        "set RUN_INTEGRATION_TESTS=1 to enable"
    ),
)


# ---------------------------------------------------------------------------
# Local DB session fixture (mirrors test_agent_v2_api.py pattern)
# ---------------------------------------------------------------------------

@pytest.fixture
async def db_session() -> Any:
    """Async DB session scoped to the test, rolls back after each test."""
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
        echo=settings.debug,
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
# Fixtures: valid WorkflowDefinition ASTs
# ---------------------------------------------------------------------------

VALID_DEFINITION: dict[str, Any] = {
    "id": "yaml-export-test-wf",
    "name": "YAML Export Test Workflow",
    "version": 1,
    "root": {
        "id": "root-seq",
        "type": "sequence",
        "label": "main",
        "config": {},
        "children": [
            {
                "id": "agent-node",
                "type": "agent",
                "label": "researcher",
                "config": {"subtype": "researcher"},
                "children": [],
            },
        ],
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
# User identities
# ---------------------------------------------------------------------------

USER_A = UserIdentity(
    user_id="yaml-export-test-user-a",
    email="yaml-export-user-a@test.example",
    display_name="YAML Export User A",
)

USER_B = UserIdentity(
    user_id="yaml-export-test-user-b",
    email="yaml-export-user-b@test.example",
    display_name="YAML Export User B",
)


# ---------------------------------------------------------------------------
# Helpers: test client builders
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _db_client(
    user: UserIdentity,
    db_session: Any,
) -> Generator[TestClient, None, None]:
    """TestClient with real DB session and given user identity injected."""

    async def _override_db() -> Any:
        yield db_session

    async def _override_user() -> UserIdentity:
        return user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        client = TestClient(app, raise_server_exceptions=True)
        yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


@contextlib.contextmanager
def _noauth_client() -> Generator[TestClient, None, None]:
    """TestClient with mock DB and anonymous user — for pure-computation tests."""

    async def _override_db() -> Any:
        yield AsyncMock()

    async def _override_user() -> UserIdentity:
        return UserIdentity.anonymous()

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        client = TestClient(app, raise_server_exceptions=True)
        yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def user_a_client(db_session: Any) -> Generator[TestClient, None, None]:
    with _db_client(USER_A, db_session) as client:
        yield client


@pytest.fixture
def user_b_client(db_session: Any) -> Generator[TestClient, None, None]:
    with _db_client(USER_B, db_session) as client:
        yield client


# ---------------------------------------------------------------------------
# Helper: create an agent and return its ID
# ---------------------------------------------------------------------------

def _create_agent(client: TestClient, definition: dict[str, Any] | None = None) -> str:
    resp = client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Export Test Agent",
            "definition": definition or VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# 1. Successful GET returns Content-Type: text/yaml
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_export_returns_text_yaml_content_type(user_a_client: TestClient) -> None:
    """GET /{id}/yaml returns HTTP 200 with Content-Type: text/yaml."""
    agent_id = _create_agent(user_a_client)

    resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    assert resp.status_code == 200, resp.text
    content_type = resp.headers.get("content-type", "")
    assert "text/yaml" in content_type, f"expected text/yaml, got: {content_type!r}"


# ---------------------------------------------------------------------------
# 2. Exported body has registry_version at top level
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_export_includes_registry_version(user_a_client: TestClient) -> None:
    """Parsed YAML from GET /{id}/yaml pins the CURRENT REGISTRY_VERSION.

    Regression guard: the export default MUST equal the constant the importer
    checks (``REGISTRY_VERSION``).  Previously export hardcoded ``"1.0"`` while
    import required ``"1.0.0"``, so every round-trip failed.
    """
    agent_id = _create_agent(user_a_client)

    resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    assert resp.status_code == 200, resp.text

    parsed = yaml.safe_load(resp.text)
    assert "registry_version" in parsed, "registry_version key must be present"
    assert parsed["registry_version"] == REGISTRY_VERSION, (
        f"expected {REGISTRY_VERSION!r}, got {parsed['registry_version']!r}"
    )


# ---------------------------------------------------------------------------
# 3. Round-trip: YAML → load_workflow_from_dict → model_dump() == definition
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_export_round_trips_via_load_workflow_from_dict(user_a_client: TestClient) -> None:
    """load_workflow_from_dict(yaml.safe_load(body)).model_dump() == stored definition.

    Validates that the exported YAML is a faithful, lossless representation
    of the AST stored in the database.
    """
    from databricks_deep_research import load_workflow_from_dict

    agent_id = _create_agent(user_a_client)

    # Fetch the agent definition via the JSON endpoint for reference
    get_resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}")
    assert get_resp.status_code == 200, get_resp.text
    stored_definition: dict[str, Any] = get_resp.json()["definition"]

    # Export as YAML and parse back
    yaml_resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    assert yaml_resp.status_code == 200, yaml_resp.text
    parsed = yaml.safe_load(yaml_resp.text)

    # Strip registry_version before feeding into the framework loader
    parsed.pop("registry_version", None)

    workflow = load_workflow_from_dict(parsed)
    round_tripped = workflow.model_dump()

    # The round-tripped model must reproduce the stored definition exactly
    assert round_tripped == stored_definition, (
        "round-trip mismatch: YAML export does not faithfully represent the stored AST"
    )


# ---------------------------------------------------------------------------
# 4. Determinism: two calls return byte-identical bodies
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_export_is_deterministic(user_a_client: TestClient) -> None:
    """Two successive GET /{id}/yaml calls return identical byte strings."""
    agent_id = _create_agent(user_a_client)

    resp1 = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    resp2 = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")

    assert resp1.status_code == 200, resp1.text
    assert resp2.status_code == 200, resp2.text
    assert resp1.text == resp2.text, "YAML export must be deterministic across calls"


# ---------------------------------------------------------------------------
# 5. Owner scoping: user_b GET returns 404 (NOT 403)
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_export_owner_scoping(
    user_a_client: TestClient,
    user_b_client: TestClient,
) -> None:
    """User A creates private agent; User B's GET /{id}/yaml returns 404."""
    agent_id = _create_agent(user_a_client)

    resp = user_b_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    assert resp.status_code == 404, (
        f"expected 404 for cross-user access, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# 5b. True round-trip: exported bytes are accepted by POST /import-yaml
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_exported_yaml_reimports_cleanly(user_a_client: TestClient) -> None:
    """GET /{id}/yaml output POSTed verbatim to /import-yaml returns 200.

    This is the regression the prior suite was missing: export and import each
    had their own test, but none exercised the actual export→import path that a
    user performs.  Before the registry_version fix this returned 400
    (registry_version_mismatch).
    """
    agent_id = _create_agent(user_a_client)

    yaml_resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    assert yaml_resp.status_code == 200, yaml_resp.text

    import_resp = user_a_client.post(
        "/api/v1/agent-designer/import-yaml",
        content=yaml_resp.text.encode("utf-8"),
        headers={"Content-Type": "text/yaml"},
    )
    assert import_resp.status_code == 200, import_resp.text
    body = import_resp.json()
    assert "definition" in body and "workflow_summary" in body


# ---------------------------------------------------------------------------
# 6. Metric emission: record_yaml_export_ms histogram called on success
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_export_emits_yaml_export_ms() -> None:
    """Mock the metric sink and assert histogram is called with correct name."""
    from deep_research.agent_designer.yaml_export import serialize_to_yaml
    from deep_research.storage.observability import MetricsSink

    mock_sink = MagicMock(spec=MetricsSink)

    # Build a minimal agent to return from the service mock
    from unittest.mock import AsyncMock as _AsyncMock
    from uuid import uuid4

    fake_agent = MagicMock()
    fake_agent.definition = VALID_DEFINITION.copy()

    async def _override_db() -> Any:
        yield _AsyncMock()

    async def _override_user() -> UserIdentity:
        return USER_A

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user_identity] = _override_user

    agent_id = uuid4()

    try:
        with (
            patch(
                "deep_research.api.v1.agents_v2.AgentV2Service"
            ) as MockService,
            patch(
                "deep_research.observability.agent_designer_metrics.get_sink",
                return_value=mock_sink,
            ),
        ):
            mock_svc_instance = MagicMock()
            mock_svc_instance.get_for_user = _AsyncMock(return_value=fake_agent)
            MockService.return_value = mock_svc_instance

            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/v1/agents-v2/{agent_id}/yaml")

        assert resp.status_code == 200, resp.text

        # Verify the histogram was called with the correct metric name
        mock_sink.histogram.assert_called_once()
        call_args = mock_sink.histogram.call_args
        metric_name = call_args[0][0] if call_args[0] else call_args[1].get("name")
        assert metric_name == "agent_designer.yaml_export_ms", (
            f"expected metric 'agent_designer.yaml_export_ms', got {metric_name!r}"
        )
        duration_value = call_args[0][1] if len(call_args[0]) > 1 else None
        assert duration_value is not None and duration_value >= 0.0, (
            "duration_ms must be a non-negative float"
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)
