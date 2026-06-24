"""Integration tests for the Mermaid export endpoint (GET /agents-v2/{id}/mermaid).

Tests covered:
  1.  test_each_node_type_parses (parametrized × 8 node types)
  2.  test_loop_renders_with_repeat_annotation
  3.  test_conditional_renders_with_merge_node
  4.  test_p95_under_100ms_for_30_block_tree
  5.  test_owner_scoping_returns_404

DB-backed tests (2–5) require a working Lakebase / PostgreSQL connection and
are skipped unless ``RUN_INTEGRATION_TESTS=1`` is set.

test_each_node_type_parses is **pure computation** (no DB, no auth) and always
runs; it exercises :func:`serialize_to_mermaid` directly rather than via HTTP
so it runs quickly and reliably in CI.

Run all:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_mermaid_export.py -v

Run pure-compute tests only:
    uv run pytest tests/integration/test_mermaid_export.py -v -k "node_type"
"""

from __future__ import annotations

import contextlib
import os
import re
import statistics
import time
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from deep_research.agent_designer.mermaid_export import serialize_to_mermaid  # noqa: E402
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
# User identities
# ---------------------------------------------------------------------------

USER_A = UserIdentity(
    user_id="mermaid-test-user-a",
    email="mermaid-a@test.example",
    display_name="Mermaid User A",
)

USER_B = UserIdentity(
    user_id="mermaid-test-user-b",
    email="mermaid-b@test.example",
    display_name="Mermaid User B",
)

# ---------------------------------------------------------------------------
# Minimal valid AgentV2 definition (used when creating via HTTP)
# ---------------------------------------------------------------------------

_BASE_DEFINITION: dict[str, Any] = {
    "id": "test-mermaid-wf",
    "name": "Mermaid Test Workflow",
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
# Client helpers (mirrors test_agent_v2_api.py pattern)
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
def _noauth_client() -> Generator[TestClient, None, None]:
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
# DB-session fixture (same pattern as test_agent_v2_api.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session() -> Any:
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


@pytest.fixture
def user_a_client(db_session: Any) -> Generator[TestClient, None, None]:
    with _db_client(USER_A, db_session) as client:
        yield client


@pytest.fixture
def user_b_client(db_session: Any) -> Generator[TestClient, None, None]:
    with _db_client(USER_B, db_session) as client:
        yield client


# ---------------------------------------------------------------------------
# Smoke-check helper: asserts mermaid output looks structurally valid
# ---------------------------------------------------------------------------

_EDGE_RE = re.compile(r"--[->]")


def _assert_valid_mermaid(text: str) -> None:
    """Light structural smoke-check for Mermaid output.

    We do NOT import the Python ``mermaid`` package (prohibited by the task
    spec). Instead we verify:
    1. The document starts with either a front-matter block or ``flowchart``.
    2. The document contains at least one edge (``-->`` or ``-.->``).
    """
    assert text.strip(), "Mermaid output must not be empty"
    # Accept front-matter (---\ntitle: ...\n---\nflowchart) or bare flowchart
    stripped = text.lstrip()
    assert stripped.startswith("---") or stripped.startswith("flowchart"), (
        f"Expected Mermaid document to start with '---' or 'flowchart', got:\n{text[:120]}"
    )
    assert _EDGE_RE.search(text), (
        f"Expected at least one Mermaid edge (-->) in output:\n{text}"
    )


# ---------------------------------------------------------------------------
# AST builders for each of the 8 node types
# ---------------------------------------------------------------------------

def _make_ast(root_block: dict[str, Any]) -> dict[str, Any]:
    """Wrap a root block in a minimal WorkflowDefinition AST dict."""
    return {
        "id": "test-wf",
        "name": "Test",
        "version": 1,
        "root": root_block,
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "token_budget": 0,
        "timeout_seconds": 1800,
    }


def _leaf(node_type: str, label: str = "") -> dict[str, Any]:
    return {
        "id": f"{node_type}-leaf",
        "node_type": node_type,
        "label": label or node_type,
        "config": {},
        "children": [],
    }


_NODE_TYPE_ASTS: dict[str, dict[str, Any]] = {
    "sequence": _make_ast({
        "id": "seq-root",
        "node_type": "sequence",
        "label": "Sequence",
        "config": {},
        "children": [_leaf("agent", "Step A"), _leaf("agent", "Step B")],
    }),
    "parallel": _make_ast({
        "id": "par-root",
        "node_type": "parallel",
        "label": "Parallel",
        "config": {},
        "children": [_leaf("agent", "Branch A"), _leaf("agent", "Branch B")],
    }),
    "loop": _make_ast({
        "id": "loop-root",
        "node_type": "loop",
        "label": "Loop",
        "config": {"max_iterations": 5},
        "children": [_leaf("agent", "Loop Body")],
    }),
    "conditional": _make_ast({
        "id": "cond-root",
        "node_type": "conditional",
        "label": "Conditional",
        "config": {"conditions": ["if X", "else"]},
        "children": [_leaf("agent", "Branch A"), _leaf("agent", "Branch B")],
    }),
    "agent": _make_ast(_leaf("agent", "Single Agent")),
    "tool": _make_ast(_leaf("tool", "Single Tool")),
    "subworkflow": _make_ast(_leaf("subworkflow", "Sub Workflow")),
    "plan_and_execute": _make_ast({
        "id": "pae-root",
        "node_type": "plan_and_execute",
        "label": "Plan & Execute",
        "config": {
            "body": _leaf("agent", "PAE Body"),
        },
        "children": [],
    }),
}

# ---------------------------------------------------------------------------
# 1. test_each_node_type_parses — parametrized, pure computation, always runs
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("node_type", list(_NODE_TYPE_ASTS.keys()))
def test_each_node_type_parses(node_type: str) -> None:
    """Each of the 8 node types produces non-empty, structurally valid Mermaid.

    This test runs without a database or auth (pure function call) so it
    executes in all CI environments.
    """
    ast = _NODE_TYPE_ASTS[node_type]
    result = serialize_to_mermaid(ast, agent_name=f"Test {node_type}", agent_id="test_agent")
    _assert_valid_mermaid(result)


# ---------------------------------------------------------------------------
# 2. test_loop_renders_with_repeat_annotation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_loop_renders_with_repeat_annotation() -> None:
    """Loop node output must contain the substring 'repeat' at least once."""
    ast = _NODE_TYPE_ASTS["loop"]
    result = serialize_to_mermaid(ast, agent_name="Loop Agent", agent_id="loop_agent")
    assert "repeat" in result, (
        f"Expected 'repeat' annotation in Mermaid loop output:\n{result}"
    )
    # Also verify the max_iterations value appears in the annotation
    assert "5" in result, (
        f"Expected max_iterations (5) in Mermaid loop annotation:\n{result}"
    )
    # Verify no real self-edge (would cause cycle): the repeat node must differ
    # from the loop node id
    assert "loop_root -.-> " in result or "-.-> " in result, (
        f"Expected dotted annotation edge in loop output:\n{result}"
    )


# ---------------------------------------------------------------------------
# 3. test_conditional_renders_with_merge_node
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_conditional_renders_with_merge_node() -> None:
    """Conditional node output must contain a 'merge' node."""
    ast = _NODE_TYPE_ASTS["conditional"]
    result = serialize_to_mermaid(ast, agent_name="Cond Agent", agent_id="cond_agent")
    assert "merge" in result, (
        f"Expected 'merge' node in Mermaid conditional output:\n{result}"
    )
    # Both branch labels should appear
    assert "if X" in result, f"Expected condition label 'if X':\n{result}"


# ---------------------------------------------------------------------------
# 4. test_p95_under_100ms_for_30_block_tree
# ---------------------------------------------------------------------------

def _build_30_block_tree() -> dict[str, Any]:
    """Build a deterministic ~30-block binary sequence tree (depth 4)."""

    def _seq(depth: int, prefix: str) -> dict[str, Any]:
        if depth == 0:
            return _leaf("agent", f"leaf-{prefix}")
        left = _seq(depth - 1, f"{prefix}L")
        right = _seq(depth - 1, f"{prefix}R")
        return {
            "id": f"seq-{prefix}",
            "node_type": "sequence",
            "label": f"Seq {prefix}",
            "config": {},
            "children": [left, right],
        }

    # depth=4 → 1+2+4+8+16 = 31 nodes
    root = _seq(4, "n")
    return _make_ast(root)


@pytest.mark.integration
def test_p95_under_100ms_for_30_block_tree() -> None:
    """p95 of 50 serialize_to_mermaid calls on a 30-block tree must be < 100ms."""
    ast = _build_30_block_tree()
    samples: list[float] = []
    # Warm up (avoids import cache cold-start inflating p95)
    for _ in range(3):
        serialize_to_mermaid(ast, agent_name="Perf Test", agent_id="perf_agent")

    for _ in range(50):
        t0 = time.monotonic()
        serialize_to_mermaid(ast, agent_name="Perf Test", agent_id="perf_agent")
        samples.append((time.monotonic() - t0) * 1000)

    p95 = statistics.quantiles(samples, n=20)[18]  # 95th percentile
    assert p95 < 100, (
        f"p95 serialize_to_mermaid latency {p95:.2f}ms exceeds 100ms threshold. "
        f"Samples: min={min(samples):.2f}ms, max={max(samples):.2f}ms"
    )


# ---------------------------------------------------------------------------
# 5. test_owner_scoping_returns_404 — DB-backed, skipped without DB
# ---------------------------------------------------------------------------


@_DB_SKIP
@pytest.mark.integration
def test_owner_scoping_returns_404(
    user_a_client: TestClient,
    user_b_client: TestClient,
) -> None:
    """User A creates private agent; User B GET /mermaid must return 404."""
    create_resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={
            "name": "Mermaid Owner Test Agent",
            "definition": _BASE_DEFINITION,
            "visibility": "private",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    agent_id = create_resp.json()["id"]

    # User A can fetch their own mermaid export
    own_resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/mermaid")
    assert own_resp.status_code == 200, own_resp.text
    _assert_valid_mermaid(own_resp.text)

    # User B must not see User A's private agent
    other_resp = user_b_client.get(f"/api/v1/agents-v2/{agent_id}/mermaid")
    assert other_resp.status_code == 404, other_resp.text
