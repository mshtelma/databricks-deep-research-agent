"""Multi-agent ``Team`` orchestration with two strategies.

- ``"delegate"``: A leader agent receives a synthesized ``task()`` tool that
  routes work to named members (each compiled as a :class:`SubAgent`).
- ``"round_robin"``: Members execute as a ``NodeType.sequence``; later
  members observe earlier members' state.

The deliberately rejected ``"vote"`` strategy raises ``ValueError`` —
LLM aggregation without citation verification is a hallucination risk;
users should compose ``Parallel(...)`` + a synthesizer agent explicitly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from databricks_deep_research.api.agent import Agent
from databricks_deep_research.api.compile import compile as compile_agent
from databricks_deep_research.api.composition import _composite_workflow
from databricks_deep_research.api.result import AgentResult
from databricks_deep_research.api.subagent import SubAgent
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
)

TeamStrategy = Literal["delegate", "round_robin", "vote"]


@dataclass
class Team:
    """Multi-agent team with a strategy for coordinating members."""

    members: list[Any] = field(default_factory=list)
    leader: Agent | None = None
    strategy: TeamStrategy = "delegate"
    share_memory: bool = False
    name: str = "team"

    def __post_init__(self) -> None:
        if self.strategy == "vote":
            raise ValueError(
                "Team(strategy='vote') is not supported; LLM aggregation "
                "without citation verification is a hallucination risk. "
                "Compose Parallel(...) + a synthesizer Agent manually."
            )
        if self.strategy == "delegate" and self.leader is None:
            raise ValueError(
                "Team(strategy='delegate') requires `leader` (an Agent)."
            )
        if not self.members:
            raise ValueError("Team requires at least one member.")

    def as_workflow(self) -> WorkflowDefinition:
        if self.strategy == "delegate":
            return self._compile_delegate()
        if self.strategy == "round_robin":
            return self._compile_round_robin()
        raise ValueError(f"Unsupported strategy: {self.strategy!r}")

    # -- delegate --------------------------------------------------------------

    def _compile_delegate(self) -> WorkflowDefinition:
        leader = self.leader
        assert leader is not None  # __post_init__ guarantees
        sub_members = [self._coerce_subagent(m) for m in self.members]
        composite = Agent(
            name=leader.name,
            instructions=leader.instructions,
            model=leader.model,
            tools=list(leader.tools),
            output_type=leader.output_type,
            max_tool_calls=leader.max_tool_calls,
            subtype=leader.subtype,
            user_prompt=leader.user_prompt,
            subagents=sub_members,
            extras=dict(leader.extras),
            pool_writes=list(leader.pool_writes),
            pool_inject=list(leader.pool_inject),
        )
        return compile_agent(composite)

    @staticmethod
    def _coerce_subagent(spec: Any) -> SubAgent:
        if isinstance(spec, SubAgent):
            return spec
        if isinstance(spec, Agent):
            return SubAgent(
                name=spec.name,
                description=spec.instructions[:200],
                model=spec.model,
                instructions=spec.instructions,
                tools=list(spec.tools),
                subtype=spec.subtype,
                output_type=spec.output_type,
                max_tool_calls=spec.max_tool_calls or 10,
                pool_mode="inherit",
            )
        raise TypeError(
            f"Team members must be Agent or SubAgent; got {type(spec).__name__}"
        )

    # -- round_robin -----------------------------------------------------------

    def _compile_round_robin(self) -> WorkflowDefinition:
        children: list[Agent] = []
        for member in self.members:
            if isinstance(member, Agent):
                children.append(member)
            elif isinstance(member, SubAgent):
                children.append(Agent(**member.to_inner_agent_kwargs()))
            else:
                raise TypeError(
                    f"Team members must be Agent or SubAgent; got {type(member).__name__}"
                )
        return _composite_workflow(
            children=children,
            node_type=NodeType.sequence,
            composite_id=self.name,
        )

    # -- runtime shims ---------------------------------------------------------

    async def arun(self, query: str, **kwargs: Any) -> AgentResult[BaseModel]:
        return await Agent._run_compiled_workflow(
            self.as_workflow(), query=query, **kwargs,
        )

    async def astream(self, query: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        async for event in Agent._stream_compiled_workflow(
            self.as_workflow(), query=query, **kwargs,
        ):
            yield event


__all__ = ["Team", "TeamStrategy"]
