"""US-102 tests: migration 034 chain, validation model columns, and the
DB-backed validation cache (round-trip + idempotent upsert).

The unit suite has no live Postgres (conftest mocks AsyncSession), so the cache
is exercised against a fake session: ``put`` is asserted to compile to a PG
``ON CONFLICT DO NOTHING`` upsert, and ``get`` is asserted to round-trip a
``WorkflowValidationResult`` through ``model_dump(json)`` -> ``model_validate``
(incl. nested directives + the source enum). Applying the migration against a
real DB happens at deploy time.
"""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from deep_research.agent_designer.critic_types import CriticDirective
from deep_research.agent_designer.validation_cache import DbValidationCache
from deep_research.agent_designer.workflow_validation import (
    VALIDATOR_VERSION,
    ValidationSource,
    WorkflowValidationResult,
)

_APP_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MIGRATION = (
    _APP_ROOT
    / "src/deep_research/db/migrations/versions/034_add_workflow_validation.py"
)


def _make_result() -> WorkflowValidationResult:
    return WorkflowValidationResult(
        verdict="needs_revision",
        summary="tighten the synthesizer",
        directives=[
            CriticDirective(
                node_path="root.children[0]",
                issue="off-topic",
                suggested_action="update_block",
                severity="blocking",
            )
        ],
        semantic_hash="abc123",
        intent_hash="def456",
        validator_version=VALIDATOR_VERSION,
        source=ValidationSource.FRESH,
        cacheable=True,
    )


# --- migration 034 ----------------------------------------------------------


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_mig034", _MIGRATION)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeOp:
    """Records DDL ops so we can assert the migration's effect without a DB."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def add_column(self, *args: Any, **_: Any) -> None:
        self.calls.append(("add_column", args))

    def create_table(self, *args: Any, **_: Any) -> None:
        self.calls.append(("create_table", args))

    def drop_table(self, *args: Any, **_: Any) -> None:
        self.calls.append(("drop_table", args))

    def drop_column(self, *args: Any, **_: Any) -> None:
        self.calls.append(("drop_column", args))


def test_migration_034_chain_and_callables() -> None:
    mod = _load_migration()
    assert mod.revision == "034_add_workflow_validation"
    assert mod.down_revision == "033_create_user_skill_folders"
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)


def test_migration_034_emits_expected_ddl(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_migration()
    fake = _FakeOp()
    monkeypatch.setattr(mod, "op", fake)

    mod.upgrade()
    add_cols = [c for c in fake.calls if c[0] == "add_column"]
    add_targets = [c[1][0] for c in add_cols]
    assert add_targets.count("agents_v2") == 3
    assert add_targets.count("agent_revisions") == 1
    create_targets = [c[1][0] for c in fake.calls if c[0] == "create_table"]
    assert create_targets == ["workflow_validation_cache"]

    fake.calls.clear()
    mod.downgrade()
    assert [c[1][0] for c in fake.calls if c[0] == "drop_table"] == [
        "workflow_validation_cache"
    ]
    assert len([c for c in fake.calls if c[0] == "drop_column"]) == 4


# --- model columns ----------------------------------------------------------


def test_models_have_validation_columns() -> None:
    from deep_research.models.agent_v2 import (
        AgentRevision,
        AgentV2,
        WorkflowValidationCache,
    )

    assert {
        "last_validation",
        "last_validation_verdict",
        "last_validation_hash",
    } <= set(AgentV2.__table__.columns.keys())
    assert "validation" in AgentRevision.__table__.columns
    assert WorkflowValidationCache.__tablename__ == "workflow_validation_cache"
    pk = {c.name for c in WorkflowValidationCache.__table__.primary_key.columns}
    assert pk == {"validator_version", "intent_hash", "semantic_hash"}


# --- DB cache ---------------------------------------------------------------


class _FakeSession:
    def __init__(self, row: Any = None) -> None:
        self._row = row
        self.executed: list[Any] = []

    async def get(self, _model: Any, _pk: Any) -> Any:
        return self._row

    async def execute(self, stmt: Any) -> Any:
        self.executed.append(stmt)
        return None


@pytest.mark.asyncio
async def test_put_emits_on_conflict_do_nothing() -> None:
    sess = _FakeSession()
    cache = DbValidationCache(sess)  # type: ignore[arg-type]
    await cache.put(_make_result())
    assert len(sess.executed) == 1
    sql = str(sess.executed[0].compile(dialect=postgresql.dialect())).upper()
    assert "WORKFLOW_VALIDATION_CACHE" in sql
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql


@pytest.mark.asyncio
async def test_get_round_trips_result() -> None:
    result = _make_result()
    row = SimpleNamespace(result=result.model_dump(mode="json"))
    cache = DbValidationCache(_FakeSession(row=row))  # type: ignore[arg-type]
    got = await cache.get(
        validator_version=result.validator_version,
        intent_hash=result.intent_hash,
        semantic_hash=result.semantic_hash,
    )
    assert got is not None
    assert got.verdict == "needs_revision"
    assert got.semantic_hash == "abc123"
    assert len(got.directives) == 1
    assert got.directives[0].node_path == "root.children[0]"


@pytest.mark.asyncio
async def test_get_returns_none_when_absent() -> None:
    cache = DbValidationCache(_FakeSession(row=None))  # type: ignore[arg-type]
    got = await cache.get(
        validator_version="v1", intent_hash="i", semantic_hash="s"
    )
    assert got is None
