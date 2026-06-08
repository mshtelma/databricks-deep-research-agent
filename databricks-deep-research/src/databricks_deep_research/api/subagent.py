"""``SubAgent`` — a child agent invoked by a parent via task delegation.

A :class:`SubAgent` compiles to a ``NodeType.subworkflow`` node containing
an inner agent. The parent agent automatically receives a synthesized
``task(agent_name, query)`` tool that delegates to the named subagent.

The default ``pool_mode="inherit"`` shares the parent workflow's pools with
the subagent. Use ``"isolate"`` for a fresh pool scope, or ``"merge"`` for
write-through merging.

Phase 1 ships the subworkflow compile path only — the in-process opt-in
(``subagent_runtime="in_process"``) is deferred to a follow-up plan.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from databricks_deep_research.api._model_resolver import DEFAULT_TIER

PoolMode = Literal["inherit", "isolate", "merge"]


@dataclass
class SubAgent:
    """A child agent reachable via the parent's ``task()`` tool.

    Attributes:
        name: Unique identifier; the parent calls ``task(agent_name=name, ...)``.
        description: Human-readable description; surfaced to the LLM as
            documentation for ``task()``.
        model: Model spec; defaults to the parent's tier.
        instructions: System prompt for the subagent.
        tools: Tools available to the subagent (callables, ``@tool`` instances,
            or :class:`ResearchTool` objects).
        subtype: Builtin subtype (default ``"custom"``).
        output_type: Optional Pydantic model for structured output.
        max_tool_calls: Per-subagent tool budget.
        pool_mode: How to share pool state with the parent
            (``"inherit"`` | ``"isolate"`` | ``"merge"``).
    """

    name: str
    description: str = ""
    model: Any | None = None
    instructions: str = ""
    tools: Iterable[Any] = field(default_factory=list)
    subtype: str = "custom"
    output_type: type[BaseModel] | None = None
    max_tool_calls: int = 10
    pool_mode: PoolMode = "inherit"

    def to_inner_agent_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for constructing an inner :class:`Agent`."""
        return {
            "name": self.name,
            "instructions": self.instructions,
            "model": self.model,
            "tools": list(self.tools),
            "output_type": self.output_type,
            "max_tool_calls": self.max_tool_calls,
            "subtype": self.subtype or "custom",
        }

    @property
    def tier_name(self) -> str:
        from databricks_deep_research.api._model_resolver import resolve_tier_name

        return resolve_tier_name(self.model) if self.model is not None else DEFAULT_TIER


__all__ = ["SubAgent", "PoolMode"]
