"""Static-correctness checks for migration 031 (designer_revisions).

Verifies the migration module loads cleanly, chains off the current head
(030_create_skills), and that upgrade/downgrade create/drop the append-only
audit table — without requiring a live database connection.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "deep_research"
    / "db"
    / "migrations"
    / "versions"
    / "031_create_designer_revisions.py"
)


def _load_migration_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_migration_031_under_test", str(_MIGRATION_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def migration_source() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def test_migration_module_imports_and_chains_off_head() -> None:
    """The module imports cleanly and chains off the prior head (030)."""
    mod = _load_migration_module()
    assert mod.revision == "031_create_designer_revisions"
    assert mod.down_revision == "030_create_skills"


def test_upgrade_creates_designer_revisions_table(migration_source: str) -> None:
    assert "create_table(" in migration_source
    assert '"designer_revisions"' in migration_source


def test_upgrade_has_audit_columns(migration_source: str) -> None:
    """The audit row must carry prev/new snapshots + verdict + actor + time."""
    for col in (
        '"subject_type"',
        '"subject_ref"',
        '"prev_snapshot"',
        '"new_snapshot"',
        '"security_verdict"',
        '"created_at"',
        '"created_by"',
    ):
        assert col in migration_source


def test_prev_snapshot_is_nullable_new_snapshot_is_not(migration_source: str) -> None:
    """prev is None on the first authored change; new is always present."""
    assert '"prev_snapshot"' in migration_source
    # security_verdict must be NOT NULL — no UNSAFE/ungated row is ever written.
    assert 'sa.Column("security_verdict", sa.Text(), nullable=False)' in migration_source


def test_upgrade_creates_subject_index(migration_source: str) -> None:
    assert "idx_designer_revisions_subject_created" in migration_source


def test_downgrade_drops_table_and_index(migration_source: str) -> None:
    assert 'drop_table("designer_revisions")' in migration_source
    assert 'drop_index(\n        "idx_designer_revisions_subject_created"' in (
        migration_source
    )
