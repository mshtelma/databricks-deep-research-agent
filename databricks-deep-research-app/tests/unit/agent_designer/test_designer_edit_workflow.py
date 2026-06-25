"""The edit-lane workflow loads and its tool list stays in sync with code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deep_research.agent_designer.framework_tools import _EDIT_AGENT_TOOL_NAMES

_WF = (
    Path(__file__).resolve().parents[3]
    / "src/deep_research/agent_designer/designer_edit_workflow.yaml"
)


def _load_raw() -> dict[str, Any]:
    return yaml.safe_load(_WF.read_text())


def _find(node: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    if node.get("id") == node_id:
        return node
    for child in node.get("children") or []:
        found = _find(child, node_id)
        if found is not None:
            return found
    return None


def test_edit_agent_tools_match_code_constant() -> None:
    raw = _load_raw()
    edit_agent = _find(raw["root"], "edit_agent")
    assert edit_agent is not None, "edit_agent node missing"
    tools = set(edit_agent["config"]["tools"])
    assert tools == set(_EDIT_AGENT_TOOL_NAMES), (
        f"YAML/code drift: only-in-yaml={tools - set(_EDIT_AGENT_TOOL_NAMES)}, "
        f"only-in-code={set(_EDIT_AGENT_TOOL_NAMES) - tools}"
    )


def test_edit_agent_cannot_rebuild() -> None:
    raw = _load_raw()
    tools = set(_find(raw["root"], "edit_agent")["config"]["tools"])
    assert "propose_workflow" not in tools
    assert "build_blueprint" not in tools


def test_edit_workflow_loads_via_framework_loader() -> None:
    from databricks_deep_research.workflow.loader import load_workflow

    wf = load_workflow(str(_WF))
    assert wf is not None
    assert getattr(wf, "root", None) is not None


def test_edit_workflow_has_grounding_then_edit() -> None:
    raw = _load_raw()
    child_ids = [c.get("id") for c in raw["root"]["children"]]
    assert child_ids == ["intent_grounding", "edit_agent"]
