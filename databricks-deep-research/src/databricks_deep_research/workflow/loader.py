"""YAML loading and saving of :class:`WorkflowDefinition`.

This module is the single authority for YAML <-> ``WorkflowDefinition``
serialisation.  It also wires up the ``from_yaml`` / ``to_yaml`` convenience
methods on the definition model itself so callers can use either style::

    # Module-level functions
    defn = load_workflow("workflows/deep_research.yaml")
    save_workflow(defn, "out.yaml")

    # Class-method style (delegates here)
    defn = WorkflowDefinition.from_yaml("workflows/deep_research.yaml")
    defn.to_yaml("out.yaml")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from databricks_deep_research.errors import WorkflowValidationError
from databricks_deep_research.workflow.definition import (
    ErrorConfig,
    NodeType,
    SourceDefinition,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.validation import validate_workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_node(data: dict[str, Any]) -> WorkflowNode:
    """Recursively construct a :class:`WorkflowNode` from a raw dict.

    Handles ``children`` recursion and optional ``error_handling`` parsing.

    Raises
    ------
    WorkflowValidationError
        If a required field (``id``, ``type``, ``label``) is missing or
        ``type`` is not a recognised :class:`NodeType` value.
    """
    # -- Required fields -----------------------------------------------------
    missing = [f for f in ("id", "type", "label") if f not in data]
    if missing:
        raise WorkflowValidationError(
            errors=[f"Node is missing required field(s): {', '.join(missing)}"],
        )

    # -- Validate node type --------------------------------------------------
    raw_type = data["type"]
    try:
        node_type = NodeType(raw_type)
    except ValueError as exc:
        valid = ", ".join(t.value for t in NodeType)
        raise WorkflowValidationError(
            errors=[f"Unknown node type '{raw_type}'. Valid types: {valid}"],
        ) from exc

    # -- Recursively build children ------------------------------------------
    children = [_build_node(child) for child in data.get("children", [])]

    # -- Optional error handling ---------------------------------------------
    error_handling: ErrorConfig | None = None
    if data.get("error_handling") is not None:
        error_handling = ErrorConfig(**data["error_handling"])

    return WorkflowNode(
        id=data["id"],
        type=node_type,
        label=data["label"],
        config=data.get("config", {}),
        children=children,
        error_handling=error_handling,
        budget_seconds=data.get("budget_seconds"),
    )


_TOOL_KIND_ENDPOINT_KEY: dict[str, str] = {
    "vector_search": "index_name",
    "genie": "space_id",
    "knowledge_assistant": "endpoint_name",
}


def _sources_from_tools(tools: list[ToolDeclaration]) -> list[dict[str, Any]]:
    """Derive SourceDefinition dicts from tool declarations."""
    from databricks_deep_research.tools.protocol import tool_kind_to_source_kind

    sources: list[dict[str, Any]] = []
    for tool in tools:
        endpoint_key = _TOOL_KIND_ENDPOINT_KEY.get(tool.kind, "")
        sources.append({
            "name": tool.name,
            "kind": tool_kind_to_source_kind(tool.kind),
            "endpoint": tool.config.get(endpoint_key, "") if endpoint_key else "",
            "description": tool.description,
        })
    return sources


def _definition_from_raw(raw: dict[str, Any]) -> WorkflowDefinition:
    """Build a validated :class:`WorkflowDefinition` from a parsed YAML dict.

    Raises
    ------
    WorkflowValidationError
        If the raw dict is missing required top-level fields, or if structural
        validation (via :func:`validate_workflow`) fails.
    """
    missing = [f for f in ("id", "name", "root") if f not in raw]
    if missing:
        raise WorkflowValidationError(
            errors=[f"Workflow is missing required field(s): {', '.join(missing)}"],
        )

    root = _build_node(raw["root"])

    # Parse tool declarations
    tool_dicts = raw.get("tools", [])
    tool_declarations = [ToolDeclaration(**td) for td in tool_dicts]

    # Parse sources; auto-populate from tool declarations when empty
    sources_raw = raw.get("sources", [])
    if not sources_raw and tool_declarations:
        sources_raw = _sources_from_tools(tool_declarations)

    sources = [
        SourceDefinition(**s) if isinstance(s, dict) else s for s in sources_raw
    ]

    definition = WorkflowDefinition(
        id=raw["id"],
        name=raw["name"],
        description=raw.get("description", ""),
        version=raw.get("version", 1),
        root=root,
        tools=tool_declarations,
        pools=raw.get("pools", []),
        sources=sources,
        models=raw.get("models", {}),
        required_inputs=raw.get("required_inputs", ["query"]),
        output_keys=raw.get("output_keys", ["output"]),
        runtime_injected_keys=raw.get("runtime_injected_keys", []),
        token_budget=raw.get("token_budget", 0),
        timeout_seconds=raw.get("timeout_seconds", 1800),
    )

    validate_workflow(definition)
    logger.debug("WORKFLOW_LOADED id=%s name=%s", definition.id, definition.name)
    return definition


def _definition_to_dict(definition: WorkflowDefinition) -> dict[str, Any]:
    """Serialise a :class:`WorkflowDefinition` to a plain dict for YAML output."""
    return definition.model_dump(mode="json", exclude_defaults=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_workflow(path: str | Path) -> WorkflowDefinition:
    """Load a workflow definition from a YAML file.

    Parameters
    ----------
    path:
        Filesystem path to the YAML file.

    Returns
    -------
    WorkflowDefinition
        A fully parsed and validated workflow definition.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    WorkflowValidationError
        If the YAML content fails structural validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow file not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise WorkflowValidationError(
            errors=["YAML root must be a mapping, got " + type(raw).__name__],
        )

    return _definition_from_raw(raw)


def load_workflow_from_string(yaml_content: str) -> WorkflowDefinition:
    """Parse a workflow definition from a YAML string.

    Parameters
    ----------
    yaml_content:
        Raw YAML text.

    Returns
    -------
    WorkflowDefinition
        A fully parsed and validated workflow definition.

    Raises
    ------
    WorkflowValidationError
        If the YAML content fails structural validation.
    """
    raw = yaml.safe_load(yaml_content)
    if not isinstance(raw, dict):
        raise WorkflowValidationError(
            errors=["YAML root must be a mapping, got " + type(raw).__name__],
        )

    return _definition_from_raw(raw)


def load_workflow_from_dict(data: dict[str, Any]) -> WorkflowDefinition:
    """Build a workflow definition from a plain dictionary.

    The dictionary structure mirrors the YAML schema: top-level keys ``id``,
    ``name``, and ``root`` are required; all others have sensible defaults.
    Unknown top-level keys are silently ignored (forward-compatibility).

    Equivalent to ``yaml.safe_load()`` followed by ``load_workflow_from_string()``,
    but skips the YAML serialisation round-trip.  The input dict is read but not
    mutated.

    Parameters
    ----------
    data:
        Workflow specification as a plain dict (same schema as parsed YAML).

    Returns
    -------
    WorkflowDefinition
        A fully parsed and validated workflow definition.

    Raises
    ------
    WorkflowValidationError
        If *data* is not a dict or fails structural validation.
    """
    if not isinstance(data, dict):
        raise WorkflowValidationError(
            errors=[f"Expected a dict, got {type(data).__name__}"],
        )
    return _definition_from_raw(data)


def save_workflow(definition: WorkflowDefinition, path: str | Path) -> None:
    """Serialise a :class:`WorkflowDefinition` to a YAML file.

    Parameters
    ----------
    definition:
        The workflow definition to serialise.
    path:
        Destination file path.  Parent directories must already exist.
    """
    p = Path(path)
    data = _definition_to_dict(definition)
    p.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.debug("WORKFLOW_SAVED id=%s path=%s", definition.id, p)


# ---------------------------------------------------------------------------
# Wire up WorkflowDefinition convenience methods
# ---------------------------------------------------------------------------


def _from_yaml(_cls: type[WorkflowDefinition], path: str | Path) -> WorkflowDefinition:
    """Replacement for the stub ``WorkflowDefinition.from_yaml``."""
    return load_workflow(path)


def _to_yaml(self: WorkflowDefinition, path: str | Path) -> None:
    """Replacement for the stub ``WorkflowDefinition.to_yaml``."""
    save_workflow(self, path)


def _from_dict(_cls: type[WorkflowDefinition], data: dict[str, Any]) -> WorkflowDefinition:
    """Replacement for the stub ``WorkflowDefinition.from_dict``."""
    return load_workflow_from_dict(data)


# Monkey-patch the definition class so the convenience API works.
WorkflowDefinition.from_yaml = classmethod(_from_yaml)  # type: ignore[assignment]
WorkflowDefinition.from_dict = classmethod(_from_dict)  # type: ignore[assignment]
WorkflowDefinition.to_yaml = _to_yaml  # type: ignore[method-assign]
