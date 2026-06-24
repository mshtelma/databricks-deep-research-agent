"""V1.5 cross-cutting release-gate integration test.

Exercises the full V1.5 server-side chain end-to-end:
  import YAML → create → patch → 409 conflict → resolve via diff path →
  revisions recorded → YAML export → YAML round-trip → Mermaid export →
  assert all five V1.5 metric signals fired.

Skip-gate: requires RUN_INTEGRATION_TESTS=1 (Phase 1+2 + Phase 5 pattern).
Without the env var, the test skips cleanly.

Run (with DB configured):
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/agent_designer/test_v15_e2e.py -v

Skip check (no env var):
    uv run pytest tests/integration/agent_designer/test_v15_e2e.py -q
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import pytest

# Must be set before importing `app` so Settings() validation does not require
# LAKEBASE_*/DATABASE_URL (checked only when storage_service_impl == "cached").
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

# ---------------------------------------------------------------------------
# Module-level skip guard — matches Phase 1 convention used across this suite
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="V1.5 cross-cutting test requires DB + RUN_INTEGRATION_TESTS=1",
)

# ---------------------------------------------------------------------------
# Deferred imports (only reached when RUN_INTEGRATION_TESTS=1 at collection
# time; safe to import unconditionally because the module-level pytestmark
# causes pytest to skip before executing any test body, but the imports below
# are needed for fixture definitions which are always collected).
# ---------------------------------------------------------------------------

import yaml as yaml_module  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from deep_research.agent_designer.registry import REGISTRY_VERSION  # noqa: E402
from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.db.session import get_db  # noqa: E402
from deep_research.main import app  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402
from deep_research.storage.observability import RecordingSink, use_sink  # noqa: E402

# ---------------------------------------------------------------------------
# User identity
# ---------------------------------------------------------------------------

_USER_A = UserIdentity(
    user_id="v15-e2e-test-user-a",
    email="v15-e2e-user-a@test.example",
    display_name="V1.5 E2E User A",
)

# ---------------------------------------------------------------------------
# Minimal valid WorkflowDefinition AST (agent-only root — smallest valid tree)
# ---------------------------------------------------------------------------

_BASE_DEFINITION: dict[str, Any] = {
    "id": "v15-e2e-wf",
    "name": "V1.5 E2E Workflow",
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
# DB session fixture (mirrors test_agent_v2_api.py pattern exactly)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session() -> Any:
    """Async DB session scoped to the test, rolls back after each test.

    Skips gracefully when no database is configured so the test run does not
    error out — it simply reports a meaningful skip reason instead.
    """
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

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
# Test-client helper
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
        yield TestClient(app, raise_server_exceptions=True)
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


# ---------------------------------------------------------------------------
# mock_metrics_sink fixture — RecordingSink swapped in via use_sink()
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metrics_sink() -> Generator[RecordingSink, None, None]:
    """Swap the global MetricsSink for a RecordingSink for the duration of the test.

    RecordingSink.names() returns the set of all metric names emitted during
    the test, allowing the test to assert that each V1.5 signal fired.
    """
    sink = RecordingSink()
    with use_sink(sink):
        yield sink


# ---------------------------------------------------------------------------
# Fixture: user_a_client
# ---------------------------------------------------------------------------


@pytest.fixture
def user_a_client(db_session: Any) -> Generator[TestClient, None, None]:
    with _db_client(_USER_A, db_session) as client:
        yield client


# ---------------------------------------------------------------------------
# V1.5 YAML for import (registry_version + minimal AST)
# ---------------------------------------------------------------------------

_IMPORT_YAML_BYTES = yaml_module.safe_dump(
    {
        "registry_version": REGISTRY_VERSION,
        **_BASE_DEFINITION,
    },
    sort_keys=True,
    allow_unicode=True,
    indent=2,
).encode()


# ---------------------------------------------------------------------------
# The release-gate test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_v15_full_release_chain(
    user_a_client: TestClient,
    mock_metrics_sink: RecordingSink,
) -> None:
    """Cross-cutting V1.5 release-gate test.

    Scenario:
      1. Import YAML via POST /agent-designer/import-yaml  (US-611)
      2. Create agent from imported definition              (agents-v2 CRUD)
      3. PATCH the agent — records a revision              (US-620)
      4. Trigger 409 with stale etag                       (EtagConflictModal / US-613)
      5. Resolve 409 by re-PATCHing with current_etag      (diff-path resolution)
      6. Verify revision was recorded                       (US-620)
      7. Export YAML — GET /{id}/yaml                       (US-610)
      8. Round-trip: re-import exported YAML               (US-611)
      9. Mermaid sanity — GET /{id}/mermaid                 (US-612)
      10. Assert all five V1.5 server-side metric signals    (US-601/610/611/612/620)
      11. Assert total elapsed < 90 s                        (runtime budget)
    """
    start = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Step 1 — Import YAML via POST /agent-designer/import-yaml (US-611)
    # ------------------------------------------------------------------
    resp = user_a_client.post(
        "/api/v1/agent-designer/import-yaml",
        content=_IMPORT_YAML_BYTES,
        headers={"Content-Type": "text/yaml"},
    )
    assert resp.status_code == 200, f"import-yaml failed: {resp.text}"
    imported = resp.json()
    assert "definition" in imported, "import-yaml response must include 'definition'"
    definition: dict[str, Any] = imported["definition"]

    # ------------------------------------------------------------------
    # Step 2 — Create agent from imported definition
    # ------------------------------------------------------------------
    resp = user_a_client.post(
        "/api/v1/agents-v2",
        json={"name": "v15-e2e", "definition": definition, "visibility": "private"},
    )
    assert resp.status_code == 201, f"create agent failed: {resp.text}"
    body = resp.json()
    agent_id: str = body["id"]
    etag_initial: str = body["etag"]
    assert etag_initial, "etag must be non-empty after create"
    assert resp.headers.get("etag") == etag_initial

    # ------------------------------------------------------------------
    # Step 3 — PATCH (simulate user edit); revision written (US-620)
    # ------------------------------------------------------------------
    patched_def: dict[str, Any] = {**definition, "token_budget": 1}
    resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"definition": patched_def},
        headers={"If-Match": etag_initial},
    )
    assert resp.status_code == 200, f"first PATCH failed: {resp.text}"
    etag_after_patch: str = resp.json()["etag"]
    assert etag_after_patch != etag_initial, "etag must rotate after PATCH"

    # ------------------------------------------------------------------
    # Step 4 — Trigger 409 with the now-stale original etag
    # ------------------------------------------------------------------
    resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"definition": patched_def},
        headers={"If-Match": etag_initial},  # stale — triggers 409
    )
    assert resp.status_code == 409, f"expected 409, got {resp.status_code}: {resp.text}"
    conflict_body = resp.json()
    detail = conflict_body.get("detail", {})
    assert "current_etag" in detail, f"409 body must include current_etag: {conflict_body}"
    current_etag: str = detail["current_etag"]

    # ------------------------------------------------------------------
    # Step 5 — Resolve via diff path: re-PATCH with current_etag
    # (the EtagConflictModal would supply this etag on the frontend)
    # ------------------------------------------------------------------
    resp = user_a_client.patch(
        f"/api/v1/agents-v2/{agent_id}",
        json={"definition": patched_def},
        headers={"If-Match": current_etag},
    )
    assert resp.status_code == 200, f"conflict-resolve PATCH failed: {resp.text}"
    etag_after_resolve: str = resp.json()["etag"]
    assert etag_after_resolve, "etag must be non-empty after resolve PATCH"

    # ------------------------------------------------------------------
    # Step 6 — Verify revision was recorded (US-620)
    # ------------------------------------------------------------------
    resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/revisions")
    assert resp.status_code == 200, f"GET /revisions failed: {resp.text}"
    revisions_body = resp.json()
    revisions = revisions_body.get("items", [])
    assert len(revisions) >= 1, (
        f"expected at least 1 revision recorded, got 0; total={revisions_body.get('total')}"
    )

    # ------------------------------------------------------------------
    # Step 7 — Export YAML: GET /{id}/yaml  (US-610)
    # ------------------------------------------------------------------
    resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/yaml")
    assert resp.status_code == 200, f"YAML export failed: {resp.text}"
    content_type = resp.headers.get("content-type", "")
    assert "text/yaml" in content_type, f"expected text/yaml, got: {content_type!r}"
    exported_yaml_text: str = resp.text
    parsed = yaml_module.safe_load(exported_yaml_text)
    assert parsed.get("registry_version") == REGISTRY_VERSION, (
        f"registry_version must be {REGISTRY_VERSION!r}, got "
        f"{parsed.get('registry_version')!r}"
    )

    # ------------------------------------------------------------------
    # Step 8 — Round-trip: re-import exported YAML (US-611)
    # ------------------------------------------------------------------
    resp = user_a_client.post(
        "/api/v1/agent-designer/import-yaml",
        content=exported_yaml_text.encode(),
        headers={"Content-Type": "text/yaml"},
    )
    assert resp.status_code == 200, f"YAML round-trip import failed: {resp.text}"

    # ------------------------------------------------------------------
    # Step 9 — Mermaid sanity: GET /{id}/mermaid (US-612)
    # ------------------------------------------------------------------
    resp = user_a_client.get(f"/api/v1/agents-v2/{agent_id}/mermaid")
    assert resp.status_code == 200, f"Mermaid export failed: {resp.text}"
    mermaid_text: str = resp.text
    assert "flowchart" in mermaid_text or mermaid_text.lstrip().startswith("---"), (
        f"Mermaid output must contain 'flowchart' or start with front-matter:\n{mermaid_text[:200]}"
    )

    # ------------------------------------------------------------------
    # Step 10 — Assert V1.5 server-side metric signals fired (US-601/610/611/612/620)
    # ------------------------------------------------------------------
    emitted: set[str] = mock_metrics_sink.names()

    # US-611: yaml_import_outcome counter (fires on every successful import)
    assert "agent_designer.yaml_import_outcome" in emitted, (
        f"agent_designer.yaml_import_outcome not emitted; got: {sorted(emitted)}"
    )

    # US-610: yaml_export_ms histogram (fires on GET /{id}/yaml)
    assert "agent_designer.yaml_export_ms" in emitted, (
        f"agent_designer.yaml_export_ms not emitted; got: {sorted(emitted)}"
    )

    # US-613 / EtagConflictModal: save_etag_conflict counter (fires on 409)
    assert "agent_designer.save_etag_conflict" in emitted, (
        f"agent_designer.save_etag_conflict not emitted; got: {sorted(emitted)}"
    )

    # agents-v2 POST/PATCH save latency (fires on successful create/update)
    assert "agent_designer.designer_save_latency" in emitted, (
        f"agent_designer.designer_save_latency not emitted; got: {sorted(emitted)}"
    )

    # US-601: token_refresh_attempt only fires when OBO refresh is enabled
    if os.environ.get("AGENT_DESIGNER_OBO_REFRESH_ENABLED") == "1":
        assert "agent_designer.token_refresh_attempt" in emitted, (
            f"agent_designer.token_refresh_attempt not emitted (OBO enabled); "
            f"got: {sorted(emitted)}"
        )

    # ------------------------------------------------------------------
    # Step 11 — Runtime budget: total elapsed must be < 90 s
    # ------------------------------------------------------------------
    elapsed = (datetime.now(UTC) - start).total_seconds()
    assert elapsed < 90, (
        f"V1.5 e2e exceeded 90 s budget: {elapsed:.1f} s"
    )
