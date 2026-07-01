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
from databricks_deep_research.tools.mcp import MCPServerConfig
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


# Provider-inheriting builtin web tools. A node may bind one of these by name
# while the workflow-level ``tools`` list omits the declaration — e.g. a
# designer scaffold whose architect returned an empty tool-plan, hand-written
# YAML, or an API import. Such kinds resolve via ``ctx.search_client`` /
# ``ctx.crawler`` like any default web tool, so a synthesized declaration is
# safe (no index name / endpoint to invent). Mirrored — NOT imported — by the
# two app-side copies (``framework_orchestrator._ensure_node_tools_declared``,
# ``ast_normalizer._reconcile_referenced_tools``) so the framework stays
# application-agnostic; ``framework_orchestrator`` now delegates here.
_AUTO_DECLARABLE_WEB_KINDS = frozenset({"web_search", "web_research", "web_crawl"})

# Sensible provider-inheriting defaults for a synthesized web-tool declaration.
# These mirror the app-side heals so a healed tool behaves like a designer
# default (web_research auto-fetches the top results in one call).
_HEAL_SYNTH_CONFIG: dict[str, dict[str, Any]] = {
    "web_research": {"total_results": 10, "auto_fetch_top_k": 5},
    "web_search": {"max_results": 10},
    "web_crawl": {},
}


def _collect_node_bound_tool_refs(root: WorkflowNode) -> list[str]:
    """Return ordered-unique STRING tool refs bound by any agent config.

    Traverses the whole node tree — parallel lanes, loop/conditional branches,
    and ``plan_and_execute`` planner/evaluator/body (which live in ``config``,
    not ``children``) — so callers are topology-agnostic. Inline tool dicts are
    skipped (they are self-contained and do not need a workflow-level
    declaration); only ``str`` refs are returned.
    """
    referenced: list[str] = []
    seen: set[str] = set()

    def _collect(agent_config: dict[str, Any]) -> None:
        for ref in agent_config.get("tools") or []:
            if isinstance(ref, str) and ref and ref not in seen:
                seen.add(ref)
                referenced.append(ref)

    def _visit_raw(raw_node: dict[str, Any]) -> None:
        config_dict = raw_node.get("config")
        if isinstance(config_dict, dict):
            if raw_node.get("type") == NodeType.agent.value:
                _collect(config_dict)
            elif raw_node.get("type") == NodeType.plan_and_execute.value:
                for nested_key in ("planner", "evaluator"):
                    nested = config_dict.get(nested_key)
                    if isinstance(nested, dict):
                        _collect(nested)
                body = config_dict.get("body")
                if isinstance(body, dict):
                    _visit_raw(body)
        for child in raw_node.get("children") or []:
            if isinstance(child, dict):
                _visit_raw(child)

    def _visit_node(node: WorkflowNode) -> None:
        if node.type == NodeType.agent:
            _collect(node.config)
        elif node.type == NodeType.plan_and_execute:
            for nested_key in ("planner", "evaluator"):
                nested = node.config.get(nested_key)
                if isinstance(nested, dict):
                    _collect(nested)
            body = node.config.get("body")
            if isinstance(body, dict):
                _visit_raw(body)
        for child in node.children:
            _visit_node(child)

    _visit_node(root)
    return referenced


def _synthesize_missing_web_declarations(
    refs: list[str], declared: set[str]
) -> list[ToolDeclaration]:
    """Build ``ToolDeclaration``s for builtin web refs missing from ``declared``.

    * a provider-inheriting builtin web kind gets a synthesized declaration
      (logged at WARNING) so it resolves via ``ctx.search_client`` /
      ``ctx.crawler`` like a default;
    * any other undeclared ref (a custom corpus / index tool that genuinely
      went missing) is logged at ERROR and skipped — we never invent config
      (index name, endpoint) we cannot know.
    """
    additions: list[ToolDeclaration] = []
    known = set(declared)
    for name in refs:
        if name in known:
            continue
        if name in _AUTO_DECLARABLE_WEB_KINDS:
            additions.append(
                ToolDeclaration(name=name, kind=name, config=dict(_HEAL_SYNTH_CONFIG[name]))
            )
            known.add(name)
            logger.warning(
                "WORKFLOW_HEAL_AUTO_DECLARED kind=%s reason=node_bound_undeclared "
                "(AST bypassed normalize_ast or builder binding/declaration mismatch)",
                name,
            )
        else:
            logger.error(
                "WORKFLOW_HEAL_UNDECLARED_NONWEB kind=%s — bound by an agent node but "
                "not declared at the workflow level and not an auto-declarable builtin "
                "web tool; execution will fail under strict tool resolution. Re-save the "
                "agent in the Designer or add the tool declaration.",
                name,
            )
    return additions


def heal_node_bound_web_tools(definition: WorkflowDefinition) -> None:
    """Auto-declare builtin web tools an agent node binds but the workflow omits.

    Runtime net for ASTs that bypass the designer normalizer — designer-chat
    scaffolds, pure UI saves, hand-written YAML, API imports, and verbatim
    shell-app exports. A researcher whose ``config.tools`` references e.g.
    ``web_research`` while the workflow-level ``tools`` declares only the
    architect's tool-plan tools otherwise fails at execution under strict tool
    resolution with ``WorkflowError "Node '<lane>-researcher' is missing
    declared tools: ['web_research']"``.

    In-place and idempotent (a workflow whose declarations already cover its
    bindings is unchanged). Mutates the *built* ``definition`` only — never a
    caller's input dict — so :func:`load_workflow_from_dict`'s no-mutation
    contract holds. ``load_workflow_from_dict`` already heals on every load (see
    :func:`_definition_from_raw`); this public entry point is for callers that
    hold a ``WorkflowDefinition`` built by other means.
    """
    additions = _synthesize_missing_web_declarations(
        _collect_node_bound_tool_refs(definition.root),
        {t.name for t in (definition.tools or [])},
    )
    definition.tools.extend(additions)


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

    # Heal node-bound-but-undeclared builtin web tools (designer scaffold /
    # shell-app export / API import) BEFORE deriving sources, so that a first
    # load equals a reload of the saved output (round-trip idempotent): the
    # synthesized web declarations participate in source auto-population exactly
    # like authored ones. Operates on the local ``tool_declarations`` list — the
    # input ``raw`` dict is untouched (no-mutation contract).
    tool_declarations.extend(
        _synthesize_missing_web_declarations(
            _collect_node_bound_tool_refs(root),
            {t.name for t in tool_declarations},
        )
    )

    # Parse sources; auto-populate from tool declarations when empty
    sources_raw = raw.get("sources", [])
    if not sources_raw and tool_declarations:
        sources_raw = _sources_from_tools(tool_declarations)

    sources = [
        SourceDefinition(**s) if isinstance(s, dict) else s for s in sources_raw
    ]

    # Parse declarative MCP servers. Previously dropped here: the constructor was
    # never given ``mcp_servers``, so persisted/loaded workflows silently lost
    # their MCP attachments before the orchestrator could inject them. Accept
    # dicts (YAML / ``model_dump``) or pre-built ``MCPServerConfig`` models.
    mcp_raw = raw.get("mcp_servers", []) or []
    mcp_servers = [
        MCPServerConfig(**m) if isinstance(m, dict) else m for m in mcp_raw
    ]

    definition = WorkflowDefinition(
        id=raw["id"],
        name=raw["name"],
        description=raw.get("description", ""),
        version=raw.get("version", 1),
        root=root,
        tools=tool_declarations,
        mcp_servers=mcp_servers,
        pools=raw.get("pools", []),
        sources=sources,
        models=raw.get("models", {}),
        required_inputs=raw.get("required_inputs", ["query"]),
        output_keys=raw.get("output_keys", ["output"]),
        runtime_injected_keys=raw.get("runtime_injected_keys", []),
        token_budget=raw.get("token_budget", 0),
        timeout_seconds=raw.get("timeout_seconds", 1800),
        research_effort=raw.get("research_effort"),
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
