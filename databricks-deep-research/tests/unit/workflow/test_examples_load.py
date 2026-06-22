"""Every shipped example workflow must satisfy build-time validation.

``load_workflow`` runs ``validate_workflow`` (condition contracts + dataflow
contracts), so a stale/dangling condition or an unmodelled config field in any
shipped example surfaces here rather than at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from databricks_deep_research.workflow.loader import load_workflow

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"


def _example_files() -> list[Path]:
    return sorted(_EXAMPLES_DIR.glob("*.yaml"))


def test_examples_dir_is_nonempty() -> None:
    # Guards against the parametrized test silently covering nothing if the
    # examples directory moves.
    assert _example_files(), f"no example workflows found under {_EXAMPLES_DIR}"


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_workflow_loads(path: Path) -> None:
    load_workflow(path)
