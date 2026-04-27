"""Regression: every existing YAML workflow still loads after Phase 1 changes.

Iterates the framework's ``examples/`` directory plus the app's
``workflows/`` directory and asserts:

1. ``load_workflow(path)`` succeeds.
2. ``save_workflow(loaded)`` then ``load_workflow`` produces a structurally
   equal :class:`WorkflowDefinition`.

This catches any drift introduced by the new ``AgentNodeConfig.extras``
field (``extra="forbid"`` preservation) or the citation extraction
relocation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from databricks_deep_research.workflow.loader import load_workflow, save_workflow

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _FRAMEWORK_ROOT / "examples"
_APP_WORKFLOWS_DIR = (
    _FRAMEWORK_ROOT.parent / "databricks-deep-research-app"
    / "src" / "deep_research" / "agent" / "workflows"
)


def _gather_yaml_paths() -> list[Path]:
    paths: list[Path] = []
    if _EXAMPLES_DIR.is_dir():
        paths.extend(sorted(p for p in _EXAMPLES_DIR.glob("*.yaml") if p.is_file()))
    if _APP_WORKFLOWS_DIR.is_dir():
        paths.extend(sorted(p for p in _APP_WORKFLOWS_DIR.glob("*.yaml") if p.is_file()))
    return paths


_YAML_PATHS = _gather_yaml_paths()


@pytest.mark.parametrize("path", _YAML_PATHS, ids=lambda p: p.name)
def test_existing_yaml_loads_after_extras_added(path: Path) -> None:
    wf1 = load_workflow(path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        save_workflow(wf1, f.name)
        dump_path = f.name
    try:
        wf2 = load_workflow(dump_path)
        assert wf1.model_dump(mode="json") == wf2.model_dump(mode="json")
    finally:
        Path(dump_path).unlink(missing_ok=True)


def test_yaml_path_discovery_finds_at_least_one_fixture() -> None:
    """Sanity: if no fixtures are found, the parametrized test silently passes."""
    assert len(_YAML_PATHS) >= 1
