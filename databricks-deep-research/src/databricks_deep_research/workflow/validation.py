"""Load-time structural validation of WorkflowDefinition trees.

Validates node-type constraints, duplicate IDs, required config fields,
and output-key conflicts before any runtime execution begins.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.agents.grounding import (
    uses_legacy_grounding_alias,
    validate_grounding_config,
)
from databricks_deep_research.errors import WorkflowValidationError
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-type classification
# ---------------------------------------------------------------------------

_LEAF_TYPES: frozenset[NodeType] = frozenset(
    {NodeType.agent, NodeType.tool, NodeType.subworkflow}
)

_COMPOSITE_TYPES: frozenset[NodeType] = frozenset(
    {NodeType.sequence, NodeType.parallel, NodeType.loop}
)

# Config fields that MUST be present for a given node type.
_REQUIRED_CONFIG: dict[NodeType, list[str]] = {
    NodeType.agent: ["subtype"],
    NodeType.tool: ["ref"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_errors(node: WorkflowNode, seen_ids: set[str], errors: list[str]) -> None:
    """Recursively walk *node* and append error strings to *errors*."""

    # -- Duplicate ID check --------------------------------------------------
    if node.id in seen_ids:
        errors.append(f"Duplicate node id: '{node.id}'")
    seen_ids.add(node.id)

    # -- Leaf nodes: no children allowed -------------------------------------
    if node.type in _LEAF_TYPES:
        if node.children:
            errors.append(
                f"Node '{node.id}' (type={node.type.value}) is a leaf and must have no children"
            )

    # -- Composite nodes: at least 1 child -----------------------------------
    elif node.type in _COMPOSITE_TYPES:
        if not node.children:
            errors.append(
                f"Node '{node.id}' (type={node.type.value}) must have at least 1 child"
            )

    # -- Conditional: at least 2 children (branches) -------------------------
    elif node.type is NodeType.conditional:
        if len(node.children) < 2:
            errors.append(
                f"Node '{node.id}' (type=conditional) must have at least 2 children (branches)"
            )
        conditions = node.config.get("conditions", [])
        default_branch = node.config.get("default_branch", len(node.children) - 1)
        if isinstance(conditions, list) and len(node.children) != len(conditions) + 1:
            errors.append(
                f"Node '{node.id}' (type=conditional) must have exactly one more "
                "child than config.conditions"
            )
        if isinstance(default_branch, int) and not (0 <= default_branch < len(node.children)):
            errors.append(
                f"Node '{node.id}' (type=conditional) has default_branch outside children range"
            )

    # -- plan_and_execute: exactly 0 children (body lives in config) ----------
    elif node.type is NodeType.plan_and_execute and node.children:
        errors.append(
            f"Node '{node.id}' (type=plan_and_execute) must have exactly 0 children"
        )

    # -- Parallel: non-overlapping output_keys in children --------------------
    if node.type is NodeType.parallel and node.children:
        seen_keys: set[str] = set()
        for child in node.children:
            key = child.config.get("output_key")
            if key is not None:
                if key in seen_keys:
                    errors.append(
                        f"Node '{node.id}' (type=parallel) has duplicate output_key "
                        f"'{key}' among its children"
                    )
                seen_keys.add(key)

    # -- Required config fields per type -------------------------------------
    required_fields = _REQUIRED_CONFIG.get(node.type, [])
    for field in required_fields:
        if field not in node.config:
            errors.append(
                f"Node '{node.id}' (type={node.type.value}) is missing required "
                f"config field '{field}'"
            )

    _validate_node_config(node, seen_ids, errors)

    # -- Pool write extract/output_key consistency ----------------------------
    if node.type is NodeType.agent:
        for warning in _validate_pool_write_extract(node.id, node.config):
            logger.warning(warning)

    # -- Recurse into children -----------------------------------------------
    for child in node.children:
        _collect_errors(child, seen_ids, errors)


def _validate_node_config(node: WorkflowNode, seen_ids: set[str], errors: list[str]) -> None:
    """Instantiate node-type config models so invalid configs fail at load time."""
    try:
        if node.type is NodeType.agent:
            config = AgentNodeConfig(**node.config)
            errors.extend(
                f"Node '{node.id}': {message}"
                for message in validate_grounding_config(config)
            )
            if uses_legacy_grounding_alias(config):
                logger.warning(
                    "WORKFLOW_LEGACY_GROUNDING_ALIAS node=%s subtype=%s",
                    node.id,
                    config.subtype,
                )
        elif node.type is NodeType.tool:
            ToolNodeConfig(**node.config)
        elif node.type is NodeType.loop:
            LoopNodeConfig(**node.config)
        elif node.type is NodeType.conditional:
            ConditionalNodeConfig(**node.config)
        elif node.type is NodeType.plan_and_execute:
            pae_config = PlanAndExecuteNodeConfig(**node.config)
            planner_config = AgentNodeConfig(**pae_config.planner)
            errors.extend(
                f"Node '{node.id}' planner: {message}"
                for message in validate_grounding_config(planner_config)
            )
            if uses_legacy_grounding_alias(planner_config):
                logger.warning(
                    "WORKFLOW_LEGACY_GROUNDING_ALIAS node=%s nested=planner",
                    node.id,
                )
            if pae_config.evaluator is not None:
                eval_config = AgentNodeConfig(**pae_config.evaluator)
                errors.extend(
                    f"Node '{node.id}' evaluator: {message}"
                    for message in validate_grounding_config(eval_config)
                )
                if uses_legacy_grounding_alias(eval_config):
                    logger.warning(
                        "WORKFLOW_LEGACY_GROUNDING_ALIAS node=%s nested=evaluator",
                        node.id,
                    )
            if pae_config.body:
                body_node = WorkflowNode(**pae_config.body)
                _collect_errors(body_node, seen_ids, errors)
    except Exception as exc:
        errors.append(
            f"Node '{node.id}' (type={node.type.value}) has invalid config: {exc}"
        )


# ---------------------------------------------------------------------------
# Pool write validation
# ---------------------------------------------------------------------------


def _validate_pool_write_extract(node_id: str, config: dict[str, Any]) -> list[str]:
    """Warn when pool_writes.extract won't match text/markdown output."""
    warnings: list[str] = []
    output_key = config.get("output_key", "output")
    output_format = config.get("output_format", "text")
    if output_format in ("text", "markdown"):
        for pw in config.get("pool_writes", []):
            extract = pw.get("extract", "") if isinstance(pw, dict) else getattr(pw, "extract", "")
            if extract != output_key and extract not in ("sources",):
                warnings.append(
                    f"Node '{node_id}': pool_writes.extract='{extract}' "
                    f"won't match output_key='{output_key}' for "
                    f"output_format='{output_format}'. "
                    f"Set extract='{output_key}' or use output_format='json'."
                )
    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_workflow(definition: WorkflowDefinition) -> list[str]:
    """Validate a :class:`WorkflowDefinition` and return a list of error messages.

    Returns an empty list when the workflow is structurally valid.

    Raises
    ------
    WorkflowValidationError
        If any validation errors are found.
    """
    errors: list[str] = []

    # -- Top-level required_inputs / output_keys must be non-empty -----------
    if not definition.required_inputs:
        errors.append("Workflow 'required_inputs' must be a non-empty list")
    if not definition.output_keys:
        errors.append("Workflow 'output_keys' must be a non-empty list")

    # -- Walk the node tree --------------------------------------------------
    seen_ids: set[str] = set()
    _collect_errors(definition.root, seen_ids, errors)

    if errors:
        raise WorkflowValidationError(errors=errors)

    return errors
