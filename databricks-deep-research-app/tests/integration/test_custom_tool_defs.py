"""Integration tests for custom_tool_defs CRUD API.

Tests covered:
  1.  test_create_appears_in_registry
  2.  test_create_duplicate_name_409
  3.  test_owner_scoping_returns_404_for_other_user
  4.  test_workspace_visibility_visible_to_others
  5.  test_delete_in_use_returns_409 (not applicable here; returns 204)
  6.  test_patch_with_stale_etag_409
  7.  test_list_paginates
  8.  test_factory_ref_in_allowlist_required
  9.  test_get_own_tool_succeeds
  10. test_delete_own_tool_succeeds
  11. test_patch_updates_fields_and_etag
  12. test_delete_other_users_private_tool_returns_404

DB-backed tests require a working Lakebase / PostgreSQL connection and are
skipped unless ``RUN_INTEGRATION_TESTS=1`` is set.

Run all:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_custom_tool_defs.py -v
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Must be set before importing `app`
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
    reason=(
        "Requires a real database connection; "
        "set RUN_INTEGRATION_TESTS=1 to enable"
    ),
)

# ---------------------------------------------------------------------------
# DB session fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session() -> Any:
    """Async DB session scoped to the test, rolled back afterwards."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    import deep_research.db.session as _db_mod
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
# User identities
# ---------------------------------------------------------------------------

USER_A = UserIdentity(
    user_id="integ-test-custom-tool-user-a",
    email="tool-user-a@test.example",
    display_name="Tool User A",
)

USER_B = UserIdentity(
    user_id="integ-test-custom-tool-user-b",
    email="tool-user-b@test.example",
    display_name="Tool User B",
)

# ---------------------------------------------------------------------------
# Valid payload helpers
# ---------------------------------------------------------------------------

_BASE_PAYLOAD: dict[str, Any] = {
    "name": "my_http_tool",
    "config_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
    "factory_ref": "web_search_v1",
    "visibility": "private",
}


def _payload(**overrides: Any) -> dict[str, Any]:
    return {**_BASE_PAYLOAD, **overrides}


def _error_payload(resp: Any) -> dict[str, Any]:
    """Extract error detail regardless of middleware response-wrapping.

    FastAPI emits ``{"detail": {...}}``; app middleware may transform to
    ``{"code": "HTTP_ERROR", "message": {...}}``.  Support both shapes.
    """
    body = resp.json()
    return body.get("detail") or body.get("message") or {}


# ---------------------------------------------------------------------------
# TestClient helpers
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
        client = TestClient(app, raise_server_exceptions=True)
        yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


@contextlib.contextmanager
def _mock_client(user: UserIdentity) -> Generator[TestClient, None, None]:
    """TestClient backed by a mock DB session — for allow-list / non-DB tests."""

    async def _override_db() -> Any:
        mock_session = AsyncMock()
        # Simulate execute returning no rows (empty scalars)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        yield mock_session

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


# ---------------------------------------------------------------------------
# 1. test_factory_ref_in_allowlist_required (no DB needed)
# ---------------------------------------------------------------------------


def test_factory_ref_in_allowlist_required() -> None:
    """factory_ref not in BUILTIN_FACTORIES must return 400 before any DB write."""
    with _mock_client(USER_A) as client:
        resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(factory_ref="os.system"),
        )
    assert resp.status_code == 400
    err = _error_payload(resp)
    assert err["error_kind"] == "factory_ref_not_in_allowlist"
    assert err["received"] == "os.system"


def test_factory_ref_dotted_path_rejected() -> None:
    """Dotted module paths as factory_ref must be rejected."""
    with _mock_client(USER_A) as client:
        resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(factory_ref="some.module.factory"),
        )
    assert resp.status_code == 400
    assert _error_payload(resp)["error_kind"] == "factory_ref_not_in_allowlist"


def test_factory_ref_empty_string_rejected() -> None:
    """Empty string factory_ref must be rejected with 400."""
    with _mock_client(USER_A) as client:
        resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(factory_ref=""),
        )
    assert resp.status_code == 400
    assert _error_payload(resp)["error_kind"] == "factory_ref_not_in_allowlist"


def test_patch_factory_ref_allowlist_enforced() -> None:
    """PATCH must also reject factory_ref not in allow-list."""
    import uuid

    tool_id = str(uuid.uuid4())
    with _mock_client(USER_A) as client:
        resp = client.patch(
            f"/api/v1/agent-designer/custom-tools/{tool_id}",
            json={"factory_ref": "arbitrary.__import__"},
            headers={"If-Match": "abc123"},
        )
    # 400 (allowlist) before any DB lookup
    assert resp.status_code == 400
    assert _error_payload(resp)["error_kind"] == "factory_ref_not_in_allowlist"


# ---------------------------------------------------------------------------
# DB-backed tests
# ---------------------------------------------------------------------------


@_DB_SKIP
def test_create_appears_in_registry(db_session: Any) -> None:
    """Creating a custom tool should surface it in the registry endpoint."""
    with _db_client(USER_A, db_session) as client:
        create_resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="registry_tool_test"),
        )
        assert create_resp.status_code == 201, create_resp.text
        created = create_resp.json()
        assert created["name"] == "registry_tool_test"
        assert "etag" in created

        registry_resp = client.get("/api/v1/agent-designer/registry")
        assert registry_resp.status_code == 200
        tool_kinds = registry_resp.json()["tool_kinds"]
        names = [k["kind"] for k in tool_kinds]
        assert "registry_tool_test" in names


@_DB_SKIP
def test_create_duplicate_name_409(db_session: Any) -> None:
    """Creating two tools with the same name for the same owner must return 409."""
    with _db_client(USER_A, db_session) as client:
        payload = _payload(name="dup_tool_409")
        r1 = client.post("/api/v1/agent-designer/custom-tools", json=payload)
        assert r1.status_code == 201, r1.text
        r2 = client.post("/api/v1/agent-designer/custom-tools", json=payload)
        assert r2.status_code == 409
        assert _error_payload(r2)["error_kind"] == "duplicate_name"


@_DB_SKIP
def test_owner_scoping_returns_404_for_other_user(db_session: Any) -> None:
    """User B cannot GET User A's private tool — must return 404."""
    with _db_client(USER_A, db_session) as client_a:
        create_resp = client_a.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="private_tool_scope", visibility="private"),
        )
        assert create_resp.status_code == 201
        tool_id = create_resp.json()["id"]

    with _db_client(USER_B, db_session) as client_b:
        get_resp = client_b.get(f"/api/v1/agent-designer/custom-tools/{tool_id}")
        assert get_resp.status_code == 404


@_DB_SKIP
def test_workspace_visibility_visible_to_others(db_session: Any) -> None:
    """Workspace-visible tools created by User A must be visible to User B."""
    with _db_client(USER_A, db_session) as client_a:
        create_resp = client_a.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="ws_visible_tool", visibility="workspace"),
        )
        assert create_resp.status_code == 201
        tool_id = create_resp.json()["id"]

    with _db_client(USER_B, db_session) as client_b:
        get_resp = client_b.get(f"/api/v1/agent-designer/custom-tools/{tool_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "ws_visible_tool"

        list_resp = client_b.get("/api/v1/agent-designer/custom-tools")
        assert list_resp.status_code == 200
        names = [t["name"] for t in list_resp.json()["items"]]
        assert "ws_visible_tool" in names


@_DB_SKIP
def test_patch_with_stale_etag_409(db_session: Any) -> None:
    """PATCH with a wrong If-Match etag must return 409."""
    with _db_client(USER_A, db_session) as client:
        create_resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="stale_etag_tool"),
        )
        assert create_resp.status_code == 201

        patch_resp = client.patch(
            f"/api/v1/agent-designer/custom-tools/{create_resp.json()['id']}",
            json={"name": "new_name"},
            headers={"If-Match": "stale_etag_value_that_is_wrong"},
        )
        assert patch_resp.status_code == 409
        assert _error_payload(patch_resp)["error_kind"] == "etag_conflict"


@_DB_SKIP
def test_list_paginates(db_session: Any) -> None:
    """List endpoint returns all tools owned by or visible to the user."""
    with _db_client(USER_A, db_session) as client:
        for i in range(3):
            r = client.post(
                "/api/v1/agent-designer/custom-tools",
                json=_payload(name=f"list_tool_{i}"),
            )
            assert r.status_code == 201, r.text

        list_resp = client.get("/api/v1/agent-designer/custom-tools")
        assert list_resp.status_code == 200
        body = list_resp.json()
        names = [t["name"] for t in body["items"]]
        assert "list_tool_0" in names
        assert "list_tool_1" in names
        assert "list_tool_2" in names
        assert body["total"] >= 3


@_DB_SKIP
def test_get_own_tool_succeeds(db_session: Any) -> None:
    """Owner can GET their own private tool by ID."""
    with _db_client(USER_A, db_session) as client:
        create_resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="get_own_tool"),
        )
        assert create_resp.status_code == 201
        tool_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/v1/agent-designer/custom-tools/{tool_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "get_own_tool"
        assert "ETag" in get_resp.headers


@_DB_SKIP
def test_delete_own_tool_succeeds(db_session: Any) -> None:
    """Owner can DELETE their own tool; subsequent GET returns 404."""
    with _db_client(USER_A, db_session) as client:
        create_resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="delete_own_tool"),
        )
        assert create_resp.status_code == 201
        tool_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/v1/agent-designer/custom-tools/{tool_id}")
        assert del_resp.status_code == 204

        get_resp = client.get(f"/api/v1/agent-designer/custom-tools/{tool_id}")
        assert get_resp.status_code == 404


@_DB_SKIP
def test_patch_updates_fields_and_etag(db_session: Any) -> None:
    """PATCH with correct etag updates fields and returns a fresh etag."""
    with _db_client(USER_A, db_session) as client:
        create_resp = client.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="patch_tool"),
        )
        assert create_resp.status_code == 201
        original = create_resp.json()
        tool_id = original["id"]
        original_etag = original["etag"]

        patch_resp = client.patch(
            f"/api/v1/agent-designer/custom-tools/{tool_id}",
            json={"name": "patch_tool_updated"},
            headers={"If-Match": original_etag},
        )
        assert patch_resp.status_code == 200
        updated = patch_resp.json()
        assert updated["name"] == "patch_tool_updated"
        assert updated["etag"] != original_etag


@_DB_SKIP
def test_delete_other_users_private_tool_returns_404(db_session: Any) -> None:
    """User B cannot DELETE User A's private tool."""
    with _db_client(USER_A, db_session) as client_a:
        create_resp = client_a.post(
            "/api/v1/agent-designer/custom-tools",
            json=_payload(name="cross_user_delete_tool"),
        )
        assert create_resp.status_code == 201
        tool_id = create_resp.json()["id"]

    with _db_client(USER_B, db_session) as client_b:
        del_resp = client_b.delete(f"/api/v1/agent-designer/custom-tools/{tool_id}")
        assert del_resp.status_code == 404
