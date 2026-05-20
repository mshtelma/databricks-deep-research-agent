"""Pure validation helpers extracted from orchestrator.py so they can survive the orchestrator's deletion.

These functions validate and analyse a workflow AST dictionary without any
I/O or session state. Keeping them in a dedicated module makes them easy to
import from tests and from other modules that must not take a dependency on
the full orchestrator.
"""

from __future__ import annotations

from typing import Any

from databricks_deep_research.workflow.loader import load_workflow_from_dict


def _node_count(node: dict[str, Any]) -> int:
    count = 1
    for child in node.get("children", []) or []:
        count += _node_count(child)
    config = node.get("config", {})
    if isinstance(config, dict):
        body = config.get("body")
        if isinstance(body, dict):
            count += _node_count(body)
        evaluator = config.get("evaluator")
        if isinstance(evaluator, dict):
            count += _node_count(evaluator)
    return count


def _validation_error(message: str, path: str | None = None) -> dict[str, Any]:
    return {"message": message, "path": path, "line": None, "kind": "validation"}


def _validate_ast(ast: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    """Returns (errors, summary). summary is None when invalid."""
    try:
        load_workflow_from_dict(ast)
    except Exception as exc:
        return [_validation_error(str(exc))], None
    summary = {
        "node_count": _node_count(ast.get("root", {})),
        "tool_count": len(ast.get("tools", []) or []),
        "source_count": len(ast.get("sources", []) or []),
    }
    return [], summary


def _quality_advice(ast: dict[str, Any]) -> list[dict[str, Any]]:
    """Surface per-agent specialization gaps AND topology mismatches as
    ADVICE for the LLM.

    These are not validation errors — they do not block save and they do not
    affect the ``valid`` flag returned to the chat consumer. They are
    structured suggestions so the LLM knows which follow-up tool call
    (``update_block`` / ``bind_tool_to_block`` / ``set_model_tier``) or which
    ``design_brief.topology`` change would fix the workflow.

    Aggregates two deterministic checks:
    * ``detect_unspecialized_agents`` — per-agent prompt / tools / tier
      defaults that survived from the scaffold.
    * ``detect_topology_mismatch`` — plan_and_execute workflows whose lane
      router is structurally redundant (would work better in parallel_lanes).
    """
    from deep_research.agent_designer.semantic_validation import (
        detect_generic_reflector_prompt,
        detect_generic_synthesizer_prompt,
        detect_topology_mismatch,
        detect_unspecialized_agents,
        detect_unspecialized_fallback_researcher,
    )

    defects = [
        *detect_unspecialized_agents(ast or {}),
        *detect_topology_mismatch(ast or {}),
        *detect_generic_synthesizer_prompt(ast or {}),
        *detect_generic_reflector_prompt(ast or {}),
        *detect_unspecialized_fallback_researcher(ast or {}),
    ]
    return [
        {
            "message": defect.message,
            "path": defect.path,
            "kind": defect.kind,
        }
        for defect in defects
    ]
