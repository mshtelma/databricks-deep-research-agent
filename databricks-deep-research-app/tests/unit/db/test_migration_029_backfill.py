"""Static-correctness checks for migration 029 (backfill_in_app_deployments).

These tests verify the migration module loads cleanly and that the SQL
in upgrade/downgrade contains the expected clauses without requiring a
live database connection.
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
    / "029_backfill_in_app_deployments.py"
)


def _load_migration_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_migration_029_under_test", str(_MIGRATION_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def migration_source() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def test_migration_module_imports() -> None:
    """The module must import cleanly without a live DB."""
    mod = _load_migration_module()
    assert mod.revision == "029_backfill_in_app_deployments"
    assert mod.down_revision == "028_deployment_runtime_columns"


def test_upgrade_inserts_into_agent_deployments(migration_source: str) -> None:
    """upgrade() must INSERT into agent_deployments."""
    assert "INSERT INTO agent_deployments" in migration_source


def test_upgrade_targets_workspace_visibility(migration_source: str) -> None:
    """upgrade() must filter on visibility='workspace'."""
    assert "visibility = 'workspace'" in migration_source


def test_upgrade_uses_latest_revision(migration_source: str) -> None:
    """upgrade() must join against agent_revisions for the revision_id."""
    assert "agent_revisions" in migration_source
    assert "rev_id" in migration_source


def test_upgrade_skips_existing_active_in_app(migration_source: str) -> None:
    """upgrade() must skip agents that already have an active in_app deployment."""
    assert "NOT EXISTS" in migration_source
    assert "mode = 'in_app'" in migration_source
    assert "status = 'active'" in migration_source


def test_downgrade_deletes_synthetic_rows(migration_source: str) -> None:
    """downgrade() must DELETE only the synthetic rows by config + empty external_resource_ids."""
    assert "DELETE FROM agent_deployments" in migration_source
    assert '{"mode":"in_app"}' in migration_source
    assert "external_resource_ids = " in migration_source
