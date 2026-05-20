"""Integration tests for AgentV2 CRUD and Agent Designer endpoints.

Tests covered:
  1.  test_create_agent_succeeds_and_returns_etag
  2.  test_create_agent_with_invalid_definition_returns_422
  3.  test_get_own_agent_succeeds
  4.  test_get_other_users_private_agent_returns_404
  5.  test_patch_without_if_match_returns_428
  6.  test_patch_with_stale_etag_returns_409
  7.  test_patch_with_correct_etag_succeeds
  8.  test_delete_own_agent_succeeds
  9.  test_delete_other_users_agent_returns_404
  10. test_list_returns_owners_and_workspace_agents
  11. test_validate_endpoint_with_valid_ast
  12. test_validate_endpoint_with_malformed_ast
  13. test_registry_endpoint_returns_8_node_types

DB-backed tests (1-10) require a working Lakebase / PostgreSQL connection
and are skipped unless ``RUN_INTEGRATION_TESTS=1`` is set.
validate/registry tests (11-13) are pure computation and always run.

Run all:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_agent_v2_api.py -v

Run validate/registry only:
    uv run pytest tests/integration/test_agent_v2_api.py -v -k "validate or registry"
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Must be set before importing `app` so that Settings() validation does not
# require LAKEBASE_*/DATABASE_URL (those are only checked when
# storage_service_impl == "cached"). The DB-backed tests inject a real session
# via dependency override; they do NOT go through the storage stack.
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from deep_research.api.v1 import agent_designer as agent_designer_api  # noqa: E402
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
# Local DB session fixture (overrides the integration conftest one)
# ---------------------------------------------------------------------------

@pytest.fixture
async def db_session() -> Any:
    """Async DB session scoped to the test.

    Skips gracefully when no database is configured (no LAKEBASE_* /
    DATABASE_URL in the environment) so that the test run doesn't error out
    — it simply reports a meaningful skip reason instead.

    When a real DB is available, it yields a real AsyncSession and rolls
    back after the test so each test starts clean.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    import deep_research.db.session as _db_mod
    from deep_research.core.config import get_settings
    from deep_research.db.session import get_database_url

    # Keep stale engines alive (same pattern as integration/conftest.py).
    _stale_engines: list[object] = []

    settings = get_settings()
    try:
        url = get_database_url(settings)
    except ValueError as exc:
        pytest.skip(f"No database configured — {exc}")

    # Detach any stale engine from module state without disposing it
    # (cross-loop disposal triggers asyncpg errors).
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
# Fixture: valid WorkflowDefinition AST (agent node only — minimal valid tree)
# ---------------------------------------------------------------------------

VALID_DEFINITION: dict[str, Any] = {
    "id": "test-wf",
    "name": "Test Workflow",
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

# An AST that is structurally broken (missing required "root" and "id" fields)
INVALID_DEFINITION: dict[str, Any] = {
    "not_a_valid_key": True,
    "broken": "garbage",
}

# ---------------------------------------------------------------------------
# User identities
# ---------------------------------------------------------------------------

USER_A = UserIdentity(
    user_id="integ-test-agent-v2-user-a",
    email="user-a@test.example",
    display_name="User A",
)

USER_B = UserIdentity(
    user_id="integ-test-agent-v2-user-b",
    email="user-b@test.example",
    display_name="User B",
)


# ---------------------------------------------------------------------------
# Helpers: test client builders
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _db_client(
    user: UserIdentity,
    db_session: Any,
) -> Generator[TestClient, None, None]:
    """TestClient with real DB session and given user identity injected.

    We do NOT use ``TestClient`` as a context manager (``with TestClient(app)``
    would trigger the FastAPI lifespan which tries to connect to Databricks /
    Lakebase).  Creating the client directly and using it without entering the
    ASGI lifespan is sufficient for endpoint-level testing — the same pattern
    used by all unit API tests in ``tests/unit/api/``.
    """

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
    """TestClient with mock DB and anonymous user — for validate/registry.

    validate/registry endpoints need no DB or Databricks auth.
    We bypass the lifespan by not entering TestClient as a context manager.
    """

    async def _override_db() -> Any:
        # Yield None for pure-computation endpoints (validate, registry); the registry
        # endpoint's tool_kinds_payload_with_custom handles session=None by falling
        # back to builtin tool kinds only (no DB query for custom_tool_defs).
        yield None

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
# 1. Create agent — happy path
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_create_agent_succeeds_and_returns_etag(user_a_client: TestClient) -> None:
    """POST /agents-v2 with valid AST returns 201, body with etag, ETag header."""
    resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "My Research Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "My Research Agent"
    assert body["visibility"] == "private"
    assert body["owner_id"] == USER_A.user_id
    assert body["etag"], "etag must be non-empty"
    assert "id" in body
    assert resp.headers.get("etag") == body["etag"]


# ---------------------------------------------------------------------------
# 2. Create agent — invalid AST returns 422
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_create_agent_with_invalid_definition_returns_422(user_a_client: TestClient) -> None:
    """POST with a broken AST must be rejected at schema-validation time (422)."""
    resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Bad Agent",
            "definition": INVALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    # FastAPI puts validation errors under "detail"
    assert "detail" in body


# ---------------------------------------------------------------------------
# 3. Get own agent
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_get_own_agent_succeeds(user_a_client: TestClient) -> None:
    """POST then GET returns 200 with matching ETag."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Gettable Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]
    created_etag = create_resp.json()["etag"]

    get_resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}")
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()["etag"] == created_etag
    assert get_resp.headers.get("etag") == created_etag


# ---------------------------------------------------------------------------
# 4. Get other user's private agent → 404
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_get_other_users_private_agent_returns_404(
    user_a_client: TestClient,
    user_b_client: TestClient,
) -> None:
    """User A creates private agent; User B's GET must return 404."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Private Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]

    get_resp = user_b_client.get(f"/api/v1/agents-v2/{agent_id}")
    assert get_resp.status_code == 404, get_resp.text


# ---------------------------------------------------------------------------
# 5. PATCH without If-Match → 428
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_patch_without_if_match_returns_428(user_a_client: TestClient) -> None:
    """PATCH without If-Match header must return 428 Precondition Required."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Agent for PATCH",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]

    patch_resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"name": "Updated Name"},
        # Intentionally no If-Match header
    )
    assert patch_resp.status_code == 428, patch_resp.text


# ---------------------------------------------------------------------------
# 6. PATCH with stale ETag → 409
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_patch_with_stale_etag_returns_409(user_a_client: TestClient) -> None:
    """POST → PATCH (succeeds, new etag) → PATCH again with old etag → 409."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "ETag Conflict Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]
    original_etag = create_resp.json()["etag"]

    # First PATCH — consume the etag
    patch1_resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"name": "First Update"},
        headers={"If-Match": original_etag},
    )
    assert patch1_resp.status_code == 200, patch1_resp.text

    # Second PATCH with the now-stale original etag
    patch2_resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"name": "Second Update"},
        headers={"If-Match": original_etag},  # stale
    )
    assert patch2_resp.status_code == 409, patch2_resp.text
    body = patch2_resp.json()
    # The router returns detail = {"message": ..., "current_etag": ...}
    detail = body.get("detail", {})
    assert "current_etag" in detail, f"expected current_etag in detail: {body}"


# ---------------------------------------------------------------------------
# 7. PATCH with correct ETag succeeds
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_patch_with_correct_etag_succeeds(user_a_client: TestClient) -> None:
    """POST then PATCH with matching If-Match → 200 + new etag different from original."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Patchable Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]
    etag = create_resp.json()["etag"]

    patch_resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"name": "Patched Name"},
        headers={"If-Match": etag},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    body = patch_resp.json()
    assert body["name"] == "Patched Name"
    new_etag = body["etag"]
    assert new_etag, "new etag must be non-empty"
    assert new_etag != etag, "etag must rotate after a successful PATCH"
    assert patch_resp.headers.get("etag") == new_etag


# ---------------------------------------------------------------------------
# 8. Delete own agent
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_delete_own_agent_succeeds(user_a_client: TestClient) -> None:
    """POST then DELETE → 204; subsequent GET → 404."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Deletable Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]

    delete_resp = user_a_client.delete(f"/api/v1/agents-v2/{agent_id}")
    assert delete_resp.status_code == 204, delete_resp.text

    get_resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}")
    assert get_resp.status_code == 404, get_resp.text


# ---------------------------------------------------------------------------
# 9. Delete other user's agent → 404
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_delete_other_users_agent_returns_404(
    user_a_client: TestClient,
    user_b_client: TestClient,
) -> None:
    """User A creates an agent; User B DELETE must return 404."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "A's Protected Agent",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]

    delete_resp = user_b_client.delete(f"/api/v1/agents-v2/{agent_id}")
    assert delete_resp.status_code == 404, delete_resp.text


# ---------------------------------------------------------------------------
# 10. List — visibility scoping
# ---------------------------------------------------------------------------

@_DB_SKIP
@pytest.mark.integration
def test_list_returns_owners_and_workspace_agents(
    user_a_client: TestClient,
    user_b_client: TestClient,
) -> None:
    """User A creates one private + one workspace agent.

    User B list must include the workspace one but NOT the private one (unless
    User B also owns other agents, those are also included).
    """
    # User A: private agent
    priv_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "A-Private",
            "definition": VALID_DEFINITION,
            "visibility": "private",
        },
    )
    assert priv_resp.status_code == 201, priv_resp.text
    private_id = priv_resp.json()["id"]

    # User A: workspace agent
    ws_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "A-Workspace",
            "definition": VALID_DEFINITION,
            "visibility": "workspace",
        },
    )
    assert ws_resp.status_code == 201, ws_resp.text
    workspace_id = ws_resp.json()["id"]

    # User B lists agents
    list_resp = user_b_client.get("/api/v1/agents-v2")
    assert list_resp.status_code == 200, list_resp.text
    body = list_resp.json()
    ids = {item["id"] for item in body["items"]}

    assert workspace_id in ids, "workspace agent must be visible to User B"
    assert private_id not in ids, "private agent of User A must NOT be visible to User B"


# ---------------------------------------------------------------------------
# 11. validate — valid AST
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_validate_endpoint_with_valid_ast() -> None:
    """POST /agent-designer/validate with valid AST → valid=true."""
    with _noauth_client() as client:
        resp = client.post(
            "/api/v1/agent-designer/validate",
            json={"definition": VALID_DEFINITION},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert body["workflow_summary"] is not None
    assert body["workflow_summary"]["node_count"] >= 1


# ---------------------------------------------------------------------------
# 12. validate — malformed AST
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_validate_endpoint_with_malformed_ast() -> None:
    """POST /agent-designer/validate with garbage → valid=false + structured errors."""
    with _noauth_client() as client:
        resp = client.post(
            "/api/v1/agent-designer/validate",
            json={"definition": INVALID_DEFINITION},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert len(body["errors"]) >= 1
    error = body["errors"][0]
    assert "message" in error
    assert error["kind"] in ("syntax", "schema", "validation")


@pytest.mark.integration
def test_validate_endpoint_rejects_missing_required_tool_config() -> None:
    definition = {
        **VALID_DEFINITION,
        "tools": [{"kind": "vector_search", "name": "vector_search", "config": {}}],
    }
    with _noauth_client() as client:
        resp = client.post(
            "/api/v1/agent-designer/validate",
            json={"definition": definition},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["errors"][0]["path"] == "tools[0].config.index_name"
    assert "requires config.index_name" in body["errors"][0]["message"]


@pytest.mark.integration
def test_resources_endpoint_returns_discovered_vector_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDiscoveryService:
        async def discover_all(self, **_: Any) -> Any:
            return SimpleNamespace(
                sources=[
                    SimpleNamespace(
                        source_type="vector_search",
                        source_id="idx-1",
                        name="Customer Index",
                        description="Customer embeddings",
                        status="ready",
                        capabilities=[],
                        metadata={"index_name": "main.sales.customer_index"},
                    )
                ]
            )

    monkeypatch.setattr(
        agent_designer_api,
        "DiscoveryService",
        lambda: FakeDiscoveryService(),
    )

    with _noauth_client() as client:
        resp = client.get("/api/v1/agent-designer/resources?kinds=vector_index")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    resource = body["resources"][0]
    assert resource["kind"] == "vector_index"
    assert resource["full_name"] == "main.sales.customer_index"


# ---------------------------------------------------------------------------
# 13. registry — 8 node types
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_registry_endpoint_returns_8_node_types() -> None:
    """GET /agent-designer/registry returns exactly 8 node_types with config_schema
    for loop, conditional, and plan_and_execute.
    """
    with _noauth_client() as client:
        resp = client.get("/api/v1/agent-designer/registry")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    node_types = body["node_types"]
    assert len(node_types) == 8, (
        f"expected 8 node_types, got {len(node_types)}: "
        f"{[nt['type'] for nt in node_types]}"
    )

    type_map = {nt["type"]: nt for nt in node_types}
    required_with_schema = ("loop", "conditional", "plan_and_execute", "tool")
    for nt_name in required_with_schema:
        assert nt_name in type_map, f"node type {nt_name!r} missing from registry"
        assert type_map[nt_name]["config_schema"] is not None, (
            f"node type {nt_name!r} must have a config_schema"
        )

    # Sanity-check the "parallel" type is present even though it has no config_schema
    assert "parallel" in type_map
    assert "sequence" in type_map
    assert "agent" in type_map
    assert "subworkflow" in type_map
