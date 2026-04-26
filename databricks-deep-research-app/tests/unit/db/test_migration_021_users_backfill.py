"""Static-correctness checks for migration 021 (create_users_table).

These tests guard against regressions of the P0 fixes landed in PR-1.1:
- The migration must backfill ``users`` from existing user_id/owner_id
  values BEFORE adding FK constraints.
- An ``'anonymous'`` sentinel row must be inserted up-front so dev-mode
  anonymous fallback in ``middleware/auth.py`` doesn't FK-violate.
- ``incognito_sessions`` must NOT be FK'd to ``users`` — incognito
  sessions are by-design ephemeral and not user-bound.

A real integration test that applies the migration to a live Postgres
instance lives separately under ``tests/integration/db/`` (gated by DB
credentials).
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

# Resolve the migration file from THIS worktree, not whatever
# ``deep_research.db.migrations.versions`` happens to import. The shared
# venv across worktrees can otherwise load a sibling copy. Loading the
# module by file path also avoids running through alembic's runtime.
_MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "deep_research"
    / "db"
    / "migrations"
    / "versions"
    / "021_create_users_table.py"
)


def _load_migration_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_migration_021_under_test", str(_MIGRATION_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def migration_source() -> str:
    """Read the migration file as text for static-pattern assertions."""
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def test_migration_module_imports() -> None:
    """The module must import cleanly even without a live DB."""
    mod = _load_migration_module()
    assert mod.revision == "021_create_users_table"
    assert mod.down_revision == "020_add_workflow_ref"


def test_user_id_tables_excludes_incognito_sessions() -> None:
    """``incognito_sessions`` must NOT have a FK to ``users``.

    Coupling incognito to ``users.user_id`` would make "delete user"
    cascades touch privacy-sensitive ephemeral state.
    """
    mod = _load_migration_module()
    assert "incognito_sessions" not in mod._USER_ID_TABLES, (
        "incognito_sessions must not be FK'd to users (privacy-sensitive)"
    )


def test_user_id_tables_includes_expected() -> None:
    """The FK list covers the expected user_id-bearing tables."""
    mod = _load_migration_module()
    assert set(mod._USER_ID_TABLES) == {
        "chats",
        "research_sessions",
        "audit_logs",
        "user_preferences",
        "message_feedback",
    }


def test_owner_id_tables_includes_expected() -> None:
    """The owner_id FK list covers all owner_id-bearing tables."""
    mod = _load_migration_module()
    assert set(mod._OWNER_ID_TABLES) == {
        "user_data_sources",
        "custom_agents",
        "prompt_templates",
        "uploaded_files",
    }


def test_anonymous_sentinel_inserted_before_backfill(migration_source: str) -> None:
    """``'anonymous'`` row must be inserted up-front so dev-mode FKs resolve."""
    assert "INSERT INTO users" in migration_source
    assert "'anonymous'" in migration_source
    assert "ON CONFLICT (user_id) DO NOTHING" in migration_source


def test_backfill_unions_all_referencing_tables(migration_source: str) -> None:
    """The backfill SQL must SELECT DISTINCT user_id/owner_id from every
    referencing table — without this, FK ADD on a non-empty DB fails."""
    expected_tables = [
        "chats",
        "research_sessions",
        "audit_logs",
        "user_preferences",
        "incognito_sessions",  # backfill includes incognito user_ids even
                                # though incognito has no FK; otherwise a
                                # legacy incognito-only user would be missing
                                # from ``users`` and break joins.
        "message_feedback",
        "user_data_sources",
        "custom_agents",
        "prompt_templates",
        "uploaded_files",
    ]
    backfill_match = re.search(
        r"INSERT INTO users.*?SELECT DISTINCT.*?FROM \(.*?\) u",
        migration_source,
        flags=re.DOTALL,
    )
    assert backfill_match is not None, "backfill UNION block missing"
    block = backfill_match.group(0)
    for table in expected_tables:
        assert table in block, f"backfill missing source table {table!r}"


def test_backfill_runs_before_fk_creation(migration_source: str) -> None:
    """The backfill INSERT must appear BEFORE any ``create_foreign_key`` call —
    otherwise FK ADD on a non-empty DB fails with constraint violation."""
    backfill_idx = migration_source.find("SELECT DISTINCT u.user_id")
    fk_idx = migration_source.find("op.create_foreign_key")
    assert backfill_idx > 0, "backfill SELECT DISTINCT not found"
    assert fk_idx > 0, "create_foreign_key call not found"
    assert backfill_idx < fk_idx, (
        "backfill must run BEFORE create_foreign_key (otherwise FK ADD fails)"
    )


def test_no_db_reset_required_in_docstring(migration_source: str) -> None:
    """The original docstring claimed `make db-reset` was required; the
    backfill makes that unnecessary, and the misleading claim must be gone."""
    assert "Requires DB recreation" not in migration_source
    assert "make db-reset" not in migration_source.lower()
