"""Workflow builder — programmatic WorkflowDefinition construction.

Level 1 abstraction: converts manual/preset steps into a WorkflowDefinition
tree without requiring YAML.  Used by config_translator for MANUAL and HYBRID
workflow modes.
"""

from __future__ import annotations

from typing import Any

from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)


def preset_steps_to_tree(
    *,
    steps: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    system_instructions: str | None = None,
) -> WorkflowNode:
    """Convert preset manual steps into a plan_and_execute node.

    In MANUAL mode the user provides pre-defined research steps instead of
    letting the planner generate them.  This builder creates a
    ``plan_and_execute`` node with a dummy planner that returns the preset
    steps verbatim and NO evaluator (since manual mode doesn't reflect).

    Args:
        steps: List of step dicts, each with at least ``title`` and ``search_queries``.
        tools: Tool references to attach to the researcher body node.
        system_instructions: Optional system prompt to inject.

    Returns:
        A ``WorkflowNode`` of type ``plan_and_execute`` ready to be inserted
        into a workflow tree.
    """
    body_config: dict[str, Any] = {
        "subtype": "researcher",
        "model_tier": "analytical",
        "output_key": "findings",
        "tools": tools or [],
        "pool_writes": [
            {"pool": "observations", "extract": "findings"},
            {"pool": "sources", "extract": "sources"},
        ],
        "max_tool_calls": 15,
    }

    if system_instructions:
        body_config["system_prompt"] = system_instructions

    body_node = {
        "id": "researcher",
        "type": "agent",
        "label": "Researcher",
        "config": body_config,
    }

    pe_config: dict[str, Any] = {
        "planner": {
            "subtype": "planner",
            "model_tier": "analytical",
            "output_key": "plan",
        },
        "items_path": "steps",
        "item_state_key": "current_step",
        "body": body_node,
        "evaluator": None,  # No evaluator for manual mode
        "preset_items": steps,  # Executor uses these instead of calling planner
        "max_iterations": len(steps),
        "min_iterations": len(steps),
        "max_replan_cycles": 0,
    }

    return WorkflowNode(
        id="research_cycle",
        type=NodeType.plan_and_execute,
        label="Research Cycle (Manual)",
        config=pe_config,
    )


def build_minimal_workflow(
    *,
    workflow_id: str = "custom",
    name: str = "Custom Workflow",
    children: list[WorkflowNode],
    pools: list[dict[str, Any]] | None = None,
) -> WorkflowDefinition:
    """Build a minimal WorkflowDefinition from a list of child nodes.

    Convenience wrapper for programmatic workflow construction.

    Args:
        workflow_id: Unique workflow identifier.
        name: Human-readable workflow name.
        children: Ordered list of child nodes for the root sequence.
        pools: Optional pool configurations.

    Returns:
        A complete ``WorkflowDefinition``.
    """
    root = WorkflowNode(
        id="main",
        type=NodeType.sequence,
        label=name,
        children=children,
    )

    return WorkflowDefinition(
        id=workflow_id,
        name=name,
        root=root,
        pools=pools or [],
        required_inputs=["query"],
        output_keys=["report"],
    )


__all__ = ["build_minimal_workflow", "preset_steps_to_tree"]
