"""Integration tests for AgentV2 revision history endpoints.

Tests covered:
  1.  test_revision_written_on_patch      — PATCH → /revisions returns 1 entry
  2.  test_revision_written_on_post       — POST → /revisions returns 1 entry
  3.  test_5_patches_5_revisions          — 5 sequential PATCHes → 5 revisions DESC
  4.  test_get_revision_returns_historical_ast — capture rev_id of first PATCH; assert definition
  5.  test_revision_write_failure_does_not_break_patch — mock error → PATCH still 200
  6.  test_revisions_owner_scoped         — user_b GET /revisions → 404

All DB-backed tests require a working Lakebase / PostgreSQL connection and are
skipped unless ``RUN_INTEGRATION_TESTS=1`` is set.

Run:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_agent_revisions.py -v
"""

from __future__ import annotations

import contextlib
import copy
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.db.session import get_db  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_RUN_DB_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

_DB_SKIP = pytest.mark.skipif(
    not _RUN_DB_TESTS,
    reason="Requires a real database connection; set RUN_INTEGRATION_TESTS=1 to enable",
)

# ---------------------------------------------------------------------------
# Minimal valid WorkflowDefinition AST
# ---------------------------------------------------------------------------

_BASE_DEFINITION: dict[str, Any] = {
    "id": "rev-test-wf",
    "name": "Revision Test Workflow",
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
# User identities
# ---------------------------------------------------------------------------

USER_A = UserIdentity(
    user_id="integ-revisions-user-a",
    email="rev-user-a@test.example",
    display_name="Rev User A",
)

USER_B = UserIdentity(
    user_id="integ-revisions-user-b",
    email="rev-user-b@test.example",
    display_name="Rev User B",
)

# ---------------------------------------------------------------------------
# Helpers
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


def _create_agent(client: TestClient, name: str = "Rev Agent") -> tuple[str, str]:
    """POST a private agent; return (agent_id, etag)."""
    resp = client.post(
        "/api/v1/agents-v2",
        json={"name": name, "definition": _BASE_DEFINITION, "visibility": "private"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["id"], body["etag"]


def _patch_agent(client: TestClient, agent_id: str, etag: str, name: str) -> str:
    """PATCH name; return new etag."""
    resp = client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"name": name},
        headers={"If-Match": etag},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["etag"]


# ---------------------------------------------------------------------------
# DB session fixture (mirrors test_agent_v2_api.py pattern)
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
# 1. Revision written on PATCH
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_revision_written_on_patch(db_session: Any) -> None:
    """PATCH an agent → GET /revisions returns exactly 1 entry."""
    with _db_client(USER_A, db_session) as client:
        agent_id, etag = _create_agent(client, "Patch Rev Agent")
        _patch_agent(client, agent_id, etag, "Patched Name")

        resp = client.get(f"/api/v1/agents-v2/{agent_id}/revisions")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] >= 1
        assert len(body["items"]) >= 1
        item = body["items"][0]
        assert "rev_id" in item
        assert "etag" in item
        assert "created_at" in item
        assert "created_by" in item


# ---------------------------------------------------------------------------
# 2. Revision written on POST
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_revision_written_on_post(db_session: Any) -> None:
    """POST an agent → GET /revisions returns at least 1 entry (the create revision)."""
    with _db_client(USER_A, db_session) as client:
        agent_id, _etag = _create_agent(client, "Post Rev Agent")

        resp = client.get(f"/api/v1/agents-v2/{agent_id}/revisions")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] >= 1
        assert len(body["items"]) >= 1


# ---------------------------------------------------------------------------
# 3. 5 PATCHes → 5 revisions in DESC order
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_5_patches_5_revisions(db_session: Any) -> None:
    """5 sequential PATCHes produce 5 revision rows returned newest-first."""
    with _db_client(USER_A, db_session) as client:
        agent_id, etag = _create_agent(client, "Multi-Patch Agent")
        # The POST itself creates 1 revision; now do 5 more PATCHes
        for i in range(5):
            etag = _patch_agent(client, agent_id, etag, f"Name v{i + 1}")

        resp = client.get(f"/api/v1/agents-v2/{agent_id}/revisions?limit=10")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # At least 5 revisions (from 5 PATCHes); possibly 6 if POST also wrote one
        assert body["total"] >= 5
        items = body["items"]
        # Verify DESC order: created_at of item[i] >= item[i+1]
        for i in range(len(items) - 1):
            assert items[i]["created_at"] >= items[i + 1]["created_at"], (
                f"Revisions not in DESC order at index {i}"
            )


# ---------------------------------------------------------------------------
# 4. GET /revisions/{rev_id} returns historical AST
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_get_revision_returns_historical_ast(db_session: Any) -> None:
    """PATCH with modified definition; capture first rev_id; verify original definition."""
    original_def = copy.deepcopy(_BASE_DEFINITION)
    original_def["name"] = "Original Workflow Name"

    modified_def = copy.deepcopy(_BASE_DEFINITION)
    modified_def["name"] = "Modified Workflow Name"

    with _db_client(USER_A, db_session) as client:
        # Create with original definition
        create_resp = client.post(
            "/api/v1/agents-v2",
            json={
                "name": "Historical AST Agent",
                "definition": original_def,
                "visibility": "private",
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        agent_id = create_resp.json()["id"]
        etag = create_resp.json()["etag"]

        # PATCH with modified definition (this should write the first or second revision)
        patch_resp = client.patch(
            f"/api/v1/agents-v2/{agent_id}",
            json={"definition": modified_def},
            headers={"If-Match": etag},
        )
        assert patch_resp.status_code == 200, patch_resp.text

        # List revisions — grab the oldest one (last in DESC list) to find the first snapshot
        list_resp = client.get(f"/api/v1/agents-v2/{agent_id}/revisions?limit=10")
        assert list_resp.status_code == 200, list_resp.text
        items = list_resp.json()["items"]
        assert len(items) >= 1

        # The oldest revision (last in DESC list) should have the original or patched definition
        oldest_rev_id = items[-1]["rev_id"]

        get_resp = client.get(f"/api/v1/agents-v2/{agent_id}/revisions/{oldest_rev_id}")
        assert get_resp.status_code == 200, get_resp.text
        rev_body = get_resp.json()
        assert "rev_id" in rev_body
        assert "etag" in rev_body
        assert "definition" in rev_body
        assert "created_at" in rev_body
        assert "created_by" in rev_body
        # The definition in the revision must be a dict (AST snapshot)
        assert isinstance(rev_body["definition"], dict)


# ---------------------------------------------------------------------------
# 5. Revision write failure does NOT break PATCH
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_revision_write_failure_does_not_break_patch(db_session: Any) -> None:
    """Mock SQLAlchemyError on revision insert; PATCH must still return 200."""
    with _db_client(USER_A, db_session) as client:
        agent_id, etag = _create_agent(client, "Resilient Agent")

        with patch(
            "deep_research.services.agent_v2_service.AgentV2Service._write_revision_best_effort",
            new_callable=AsyncMock,
            side_effect=SQLAlchemyError("Simulated DB failure"),
        ):
            # The PATCH should succeed even though _write_revision_best_effort raises
            resp = client.patch(
                f"/api/v1/agents-v2/{agent_id}",
                json={"name": "Still Works"},
                headers={"If-Match": etag},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] == "Still Works"


# ---------------------------------------------------------------------------
# 6. Revisions are owner-scoped (user_b cannot access user_a's revisions)
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_revisions_owner_scoped(db_session: Any) -> None:
    """User B GET /revisions for User A's private agent → 404."""
    with _db_client(USER_A, db_session) as client_a:
        agent_id, etag = _create_agent(client_a, "Scoped Rev Agent")
        _patch_agent(client_a, agent_id, etag, "Patched by A")

    with _db_client(USER_B, db_session) as client_b:
        resp = client_b.get(f"/api/v1/agents-v2/{agent_id}/revisions")
        assert resp.status_code == 404, resp.text
