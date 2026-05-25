"""Composition primitives: :class:`Sequence` and :class:`Parallel`.

Wrap one or more :class:`Agent` (and sub-pipelines) into a higher-level
workflow. Both compile to a single :class:`WorkflowDefinition` with a root
``NodeType.sequence`` / ``NodeType.parallel`` node.

Loop and Conditional composition primitives are deferred to a follow-up
plan — the IR already supports them via ``NodeType.loop`` and
``NodeType.conditional``, so the compile path is the same pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from databricks_deep_research.api.result import AgentResult


def _child_to_node(child: Any) -> tuple[WorkflowNode, list[ToolDeclaration]]:
    """Compile a child (Agent / Sequence / Parallel) to a (node, tools) tuple."""
    from databricks_deep_research.api.agent import Agent  # local import: avoid cycle

    if isinstance(child, Agent):
        wf = child.as_workflow()
        return wf.root, list(wf.tools)
    if isinstance(child, (Sequence, Parallel)):
        wf = child.as_workflow()
        return wf.root, list(wf.tools)
    raise TypeError(
        f"Unsupported composition child type: {type(child).__name__}; "
        "expected Agent, Sequence, or Parallel."
    )


def _composite_workflow(
    *,
    children: Iterable[Any],
    node_type: NodeType,
    composite_id: str,
) -> WorkflowDefinition:
    child_nodes: list[WorkflowNode] = []
    tool_decls: list[ToolDeclaration] = []
    seen_tool_names: set[str] = set()

    for child in children:
        node, tools = _child_to_node(child)
        child_nodes.append(node)
        for decl in tools:
            if decl.name in seen_tool_names:
                continue
            seen_tool_names.add(decl.name)
            tool_decls.append(decl)

    root = WorkflowNode(
        id=composite_id,
        type=node_type,
        label=composite_id.replace("_", " ").title(),
        config={},
        children=child_nodes,
    )

    return WorkflowDefinition(
        id=composite_id,
        name=composite_id,
        description="",
        version=1,
        root=root,
        tools=tool_decls,
        pools=[],
        sources=[],
        models={},
        required_inputs=["query"],
        output_keys=["output"],
        token_budget=0,
    )


@dataclass
class Sequence:
    """Run children in order. Each child sees the previous child's state.

    Compiles to ``NodeType.sequence``.
    """

    children: tuple[Any, ...] = field(default_factory=tuple)
    name: str = "sequence"

    def __init__(self, *children: Any, name: str = "sequence") -> None:
        self.children = tuple(children)
        self.name = name

    def as_workflow(self) -> WorkflowDefinition:
        return _composite_workflow(
            children=self.children,
            node_type=NodeType.sequence,
            composite_id=self.name,
        )

    async def arun(self, query: str, **kwargs: Any) -> AgentResult[BaseModel]:
        from databricks_deep_research.api.agent import Agent
        return await Agent._run_compiled_workflow(self.as_workflow(), query=query, **kwargs)

    async def astream(self, query: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        from databricks_deep_research.api.agent import Agent
        async for event in Agent._stream_compiled_workflow(
            self.as_workflow(), query=query, **kwargs,
        ):
            yield event


@dataclass
class Parallel:
    """Run children concurrently. Each sees the same starting state.

    Compiles to ``NodeType.parallel``.
    """

    children: tuple[Any, ...] = field(default_factory=tuple)
    name: str = "parallel"

    def __init__(self, *children: Any, name: str = "parallel") -> None:
        self.children = tuple(children)
        self.name = name

    def as_workflow(self) -> WorkflowDefinition:
        return _composite_workflow(
            children=self.children,
            node_type=NodeType.parallel,
            composite_id=self.name,
        )

    async def arun(self, query: str, **kwargs: Any) -> AgentResult[BaseModel]:
        from databricks_deep_research.api.agent import Agent
        return await Agent._run_compiled_workflow(self.as_workflow(), query=query, **kwargs)

    async def astream(self, query: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        from databricks_deep_research.api.agent import Agent
        async for event in Agent._stream_compiled_workflow(
            self.as_workflow(), query=query, **kwargs,
        ):
            yield event


__all__ = ["Sequence", "Parallel"]
