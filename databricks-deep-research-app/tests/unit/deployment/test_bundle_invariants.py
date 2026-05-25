"""Bundle invariant tests: every Lakebase-using target must declare
``lakebase-postgres`` in its app resources.

This is a regression guard. Without it, a future YAML edit could quietly
drop the DAB-managed binding and reintroduce the drift class that this
plan exists to fix.

We read ``databricks.yml`` directly rather than shelling out to
``databricks bundle validate``: the binding is statically defined in
the global ``apps.deep_research_agent.resources`` array and inherited
by every target, so YAML parsing is sufficient and avoids the network
call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DATABRICKS_YML = REPO_ROOT / "databricks.yml"

# Targets that bind the app to Lakebase. If a target legitimately does
# NOT use Lakebase in the future, exclude it here with a comment.
LAKEBASE_TARGETS = ("dev", "ais", "e2e", "local-dev")

REQUIRED_BUNDLE_VARS = ("postgres_branch", "postgres_database_resource")


@pytest.fixture(scope="module")
def databricks_yml() -> dict:
    assert DATABRICKS_YML.exists(), f"databricks.yml not found at {DATABRICKS_YML}"
    return yaml.safe_load(DATABRICKS_YML.read_text())


def test_postgres_binding_vars_declared(databricks_yml: dict) -> None:
    variables = databricks_yml.get("variables", {})
    for var in REQUIRED_BUNDLE_VARS:
        assert var in variables, f"databricks.yml is missing variable {var!r}"
        assert variables[var].get("default") == "pending", (
            f"{var} default must be 'pending' sentinel (was "
            f"{variables[var].get('default')!r}). Preflight relies on this "
            "to block unresolved deploys."
        )


def test_lakebase_postgres_resource_present_globally(databricks_yml: dict) -> None:
    app = databricks_yml["resources"]["apps"]["deep_research_agent"]
    by_name = {r["name"]: r for r in app["resources"]}

    assert "lakebase-postgres" in by_name, (
        "apps.deep_research_agent.resources must contain a "
        "'lakebase-postgres' entry — this is the single source of truth "
        "for the app↔Lakebase binding; removing it reintroduces the "
        "Apps-API drift class."
    )
    pg = by_name["lakebase-postgres"]["postgres"]
    assert pg["branch"] == "${var.postgres_branch}"
    assert pg["database"] == "${var.postgres_database_resource}"
    assert pg["permission"] == "CAN_CONNECT_AND_CREATE"


@pytest.mark.parametrize("target_name", LAKEBASE_TARGETS)
def test_target_inherits_lakebase_postgres_binding(
    databricks_yml: dict, target_name: str
) -> None:
    # Sanity check: target is defined and doesn't override .resources.apps
    # to drop lakebase-postgres. DAB merges target overrides over the
    # global config; if a target redeclares .apps.<name>.resources, it
    # REPLACES the global list. We refuse such an override unless this
    # test is updated.
    targets = databricks_yml.get("targets", {})
    assert target_name in targets, f"target {target_name!r} not declared"

    override = (
        targets[target_name]
        .get("resources", {})
        .get("apps", {})
        .get("deep_research_agent", {})
        .get("resources")
    )
    if override is None:
        return  # no override; inherits global config

    overridden_names = {r["name"] for r in override}
    assert "lakebase-postgres" in overridden_names, (
        f"target {target_name!r} overrides apps.deep_research_agent.resources "
        "but drops the 'lakebase-postgres' entry; the binding must be present "
        "for every Lakebase target."
    )
