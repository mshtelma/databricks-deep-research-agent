"""Python ``Agent`` → :class:`WorkflowDefinition` compiler.

Translates an :class:`Agent` (and any nested :class:`SubAgent` instances)
into the framework's existing :class:`WorkflowDefinition` IR. The output is
identical to what the YAML loader produces for an equivalent YAML file —
guaranteeing the YAML ↔ Python round-trip property documented in the plan.

Design points:

- The agent compiles to a single ``NodeType.agent`` node whose
  ``config.subtype`` selects the builtin subtype. Default is ``"custom"``.
- Sub-agents compile to ``NodeType.subworkflow`` children. The parent
  receives a synthesized ``task(agent_name, query)`` tool — generated at
  compile time, not at runtime — that delegates to the named subagent.
- ``@tool`` callables are inlined via the ``decorated`` factory: each
  becomes a :class:`ToolDeclaration` with ``kind="decorated"`` and the
  import path captured. The actual ``ResearchTool`` instance is also
  threaded into the runner via ``register_tools()`` so name resolution
  finds it without the YAML import lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from databricks_deep_research.api._model_resolver import resolve_tier_name
from databricks_deep_research.api.subagent import SubAgent
from databricks_deep_research.tools.api import _DecoratedTool
from databricks_deep_research.tools.api import tool as tool_decorator
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    ToolDefinition,
    tool_kind_to_source_kind,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    SourceDefinition,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)

if TYPE_CHECKING:
    from databricks_deep_research.api.agent import Agent

_TASK_TOOL_NAME = "task"


def _coerce_to_research_tool(spec: Any) -> ResearchTool:
    """Convert an arbitrary tool spec into a :class:`ResearchTool` instance.

    Supports:
    - Already-decorated ``_DecoratedTool`` (returned as-is).
    - Plain callables (auto-wrapped via :func:`tool`).
    - ``ResearchTool`` protocol instances (returned as-is).
    """
    if isinstance(spec, _DecoratedTool):
        return spec
    if isinstance(spec, ResearchTool):
        return spec
    if callable(spec):
        wrapped = tool_decorator(spec)
        return wrapped
    raise TypeError(
        f"Cannot coerce {type(spec).__name__!r} to ResearchTool; "
        "pass a callable, a @tool-decorated function, or a ResearchTool instance."
    )


def _detect_cycle(agent: Agent, seen: set[int] | None = None) -> None:
    """Raise ``ValueError`` if ``agent.subagents`` contains a cycle."""
    seen = seen or set()
    if id(agent) in seen:
        raise ValueError(
            f"Cycle detected in subagent graph at agent={agent.name!r}"
        )
    seen.add(id(agent))
    for sub in agent.subagents:
        for tool_obj in sub.tools:
            # SubAgent tools may include nested Agents; we don't traverse
            # those because subagents only delegate via task().
            if hasattr(tool_obj, "subagents"):
                _detect_cycle(tool_obj, set(seen))


def _build_task_tool(subagents: list[SubAgent]) -> _DecoratedTool:
    """Synthesize a ``task(agent_name, query)`` tool routing to subagents.

    Generated at compile time. The tool's body is unused at runtime — the
    subworkflow nodes carry the actual execution path. This tool exists so
    the LLM has a discoverable entry point with a typed schema.
    """

    @tool_decorator(
        name=_TASK_TOOL_NAME,
        description=(
            "Delegate work to a named sub-agent. Pass agent_name (one of: "
            f"{', '.join(repr(s.name) for s in subagents)}) and an optional "
            "query string."
        ),
    )
    def task(agent_name: str, query: str = "") -> str:
        """Delegate to a sub-agent by name (compile-time synthesized).

        Args:
            agent_name: Name of the sub-agent to delegate to.
            query: The query to forward.
        """
        return f"task delegated: agent={agent_name} query={query}"

    return task


def _agent_node_config(agent: Agent, *, tools: list[ResearchTool]) -> dict[str, Any]:
    """Build a config dict that satisfies :class:`AgentNodeConfig`.

    Note: ``output_model`` is intentionally omitted from the dict (Pydantic
    accepts ``None`` for ``Any``-typed fields, but Pydantic models are not
    JSON-serializable; we keep it on the runtime ``Agent`` object instead.)
    """
    config: dict[str, Any] = {
        "subtype": agent.subtype or "custom",
        "model_tier": resolve_tier_name(agent.model),
        "system_prompt": agent.instructions,
        "user_prompt_template": agent.user_prompt or "{query}",
        "tools": [t.definition.name for t in tools],
        "output_key": f"{agent.name}_output",
    }

    if agent.max_tool_calls is not None:
        config["max_tool_calls"] = agent.max_tool_calls
    elif tools:
        config["max_tool_calls"] = 8

    if agent.output_type is not None:
        config["output_format"] = "json"
        config["output_mode"] = "json"

    if agent.pool_writes:
        config["pool_writes"] = [w.to_dict() for w in agent.pool_writes]
    if agent.pool_inject:
        config["pool_inject"] = [i.to_dict() for i in agent.pool_inject]

    if agent.extras:
        config["extras"] = dict(agent.extras)

    return config


def _module_qualified_path(fn: Any) -> str | None:
    """Return ``module:attr`` for a function/callable, if recoverable."""
    module = getattr(fn, "__module__", None)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if module is None or name is None or "<lambda>" in name:
        return None
    return f"{module}:{name}"


def _build_tool_declaration(t: ResearchTool) -> ToolDeclaration:
    """Build a :class:`ToolDeclaration` for the YAML round-trip.

    For ``_DecoratedTool`` instances, capture the import path so YAML
    serialization can re-import the function via the ``decorated`` factory.
    For other ``ResearchTool`` instances, fall back to a ``custom`` kind
    with the tool name; YAML round-trip relies on ``register_tools()``
    plumbing the actual instance into the runner at execution time.
    """
    if isinstance(t, _DecoratedTool):
        path = _module_qualified_path(t.fn)
        config: dict[str, Any] = {}
        if path is not None:
            config["import"] = path
        return ToolDeclaration(
            name=t.definition.name,
            kind="decorated",
            config=config,
            description=t.definition.description,
        )
    definition: ToolDefinition = t.definition
    return ToolDeclaration(
        name=definition.name,
        kind="custom",
        config={},
        description=definition.description,
    )


def _compile_subagent_to_subworkflow(parent_name: str, sub: SubAgent) -> tuple[WorkflowNode, list[ToolDeclaration]]:
    """Compile a :class:`SubAgent` into a ``NodeType.subworkflow`` node.

    The inner workflow holds a single agent node configured per the subagent.
    """
    from databricks_deep_research.api.agent import Agent  # local import: avoid cycle

    inner_agent = Agent(**sub.to_inner_agent_kwargs())
    inner_wf = inner_agent.as_workflow()

    sub_node = WorkflowNode(
        id=sub.name,
        type=NodeType.subworkflow,
        label=sub.description or sub.name,
        config={
            "ref": f"{parent_name}.{sub.name}",
            "inline": inner_wf.model_dump(mode="json"),
            "pool_mode": sub.pool_mode,
        },
        children=[],
    )

    return sub_node, list(inner_wf.tools)


def compile(agent: Agent) -> WorkflowDefinition:
    """Build a :class:`WorkflowDefinition` from an :class:`Agent` declaration.

    Args:
        agent: The Python-level agent.

    Returns:
        A validated :class:`WorkflowDefinition` ready to execute via the
        existing :class:`WorkflowExecutor`.
    """
    _detect_cycle(agent)

    coerced_tools: list[ResearchTool] = [_coerce_to_research_tool(t) for t in agent.tools]

    subworkflow_nodes: list[WorkflowNode] = []
    tool_decls: list[ToolDeclaration] = []
    seen_tool_names: set[str] = set()

    if agent.subagents:
        for sub in agent.subagents:
            sub_node, sub_tool_decls = _compile_subagent_to_subworkflow(agent.name, sub)
            subworkflow_nodes.append(sub_node)
            for decl in sub_tool_decls:
                if decl.name in seen_tool_names:
                    continue
                seen_tool_names.add(decl.name)
                tool_decls.append(decl)

        task_tool = _build_task_tool(list(agent.subagents))
        coerced_tools.append(task_tool)

    for t in coerced_tools:
        if t.definition.name in seen_tool_names:
            continue
        seen_tool_names.add(t.definition.name)
        tool_decls.append(_build_tool_declaration(t))

    agent_node = WorkflowNode(
        id=agent.name,
        type=NodeType.agent,
        label=agent.name.replace("_", " ").title(),
        config=_agent_node_config(agent, tools=coerced_tools),
        children=[],
    )

    if subworkflow_nodes:
        # Wrap agent + subworkflow children under a sequence so the executor
        # exposes both branches in the same workflow tree.
        root = WorkflowNode(
            id=f"{agent.name}_with_subagents",
            type=NodeType.sequence,
            label=f"{agent.name} pipeline",
            config={},
            children=[agent_node, *subworkflow_nodes],
        )
    else:
        root = agent_node

    # Mirror ``loader._sources_from_tools``: auto-derive ``SourceDefinition``
    # entries from tool declarations so YAML round-trip is structurally equal.
    sources: list[SourceDefinition] = [
        SourceDefinition(
            name=decl.name,
            kind=tool_kind_to_source_kind(decl.kind),
            endpoint="",
            description=decl.description,
        )
        for decl in tool_decls
    ]

    return WorkflowDefinition(
        id=f"agent_{agent.name}",
        name=f"agent_{agent.name}",
        description=agent.instructions[:120] if agent.instructions else "",
        version=1,
        root=root,
        tools=tool_decls,
        pools=[],
        sources=sources,
        models={},
        required_inputs=["query"],
        output_keys=[f"{agent.name}_output"],
        token_budget=0,
    )


def coerce_tools(specs: list[Any]) -> list[ResearchTool]:
    """Public helper for runners: convert mixed Python tool specs to :class:`ResearchTool`."""
    return [_coerce_to_research_tool(t) for t in specs]


__all__ = ["compile", "coerce_tools"]
