"""``Agent`` — the Python authoring surface for a single LLM-driven node.

An :class:`Agent` compiles to one ``NodeType.agent`` :class:`WorkflowDefinition`
node and runs through the existing :class:`WorkflowExecutor` + harness +
ReactLoop. The default ``subtype="custom"`` registers as a no-hook builtin
(see :mod:`databricks_deep_research.agents.builtins.custom`) so it inherits
the bare execution path with zero hook invocation.

Use :meth:`Agent.arun` for collected results, :meth:`Agent.astream` for
event-by-event streaming, and :meth:`Agent.as_workflow` to obtain the IR
for serialization (``save_workflow(agent.as_workflow(), "x.yaml")``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from databricks_deep_research.api.pools import PoolInjectSpec, PoolWriteSpec
from databricks_deep_research.api.result import AgentResult
from databricks_deep_research.api.subagent import SubAgent
from databricks_deep_research.citation.extraction import (
    VerificationSummary,
    extract_verification,
    extract_verification_from_report,
)
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.state import WorkflowState

if TYPE_CHECKING:
    from databricks_deep_research.llm.client import FrameworkLLMClient

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """A single LLM-driven node, authored in Python.

    When ``approval_broker`` is set, ``user_id`` SHOULD also be set to enable
    per-request HITL authorization. Without ``user_id``, the HTTP layer falls
    back to any-authenticated-user (single-tenant assumption).

    Attributes:
        name: Identifier; used as the agent node's ``id``.
        instructions: System prompt.
        model: ``str`` (tier or endpoint), :class:`ModelTier`, or
            :class:`FrameworkLLMClient`.
        tools: Iterable of callables, ``@tool`` instances, or
            :class:`ResearchTool` objects.
        output_type: Optional Pydantic model class for structured output.
        max_tool_calls: Per-run tool budget. Defaults to ``8`` when tools are
            present, ``0`` otherwise.
        subtype: Builtin subtype (``"custom"`` by default).
        user_prompt: User-prompt template (Jinja-safe).
        subagents: List of :class:`SubAgent` instances. The compiler
            synthesizes a ``task()`` tool delegating to them.
        approval_broker: Phase-2 attachment point (HITL).
        files: Phase-2 attachment point (virtual filesystem).
        checkpointer: Phase-2 attachment point (DeltaCheckpointer).
        extras: Caller-supplied ``ctx.extras`` overrides.
        user_id: Authenticated user ID for HITL ownership. Set when
            ``approval_broker`` is set.
        pool_writes: Optional pool-write specs.
        pool_inject: Optional pool-inject specs.
    """

    name: str = "agent"
    instructions: str = ""
    model: Any | None = None
    tools: Iterable[Any] = field(default_factory=list)
    output_type: type[BaseModel] | None = None
    max_tool_calls: int | None = None
    subtype: str = "custom"
    user_prompt: str = "{query}"
    subagents: list[SubAgent] = field(default_factory=list)
    approval_broker: Any | None = None
    files: Any | None = None
    checkpointer: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    pool_writes: list[PoolWriteSpec] = field(default_factory=list)
    pool_inject: list[PoolInjectSpec] = field(default_factory=list)

    # ----------------------------------------------------------------- runtime

    def as_workflow(self) -> WorkflowDefinition:
        """Compile this agent into a :class:`WorkflowDefinition`."""
        from databricks_deep_research.api.compile import compile as compile_agent

        return compile_agent(self)

    async def arun(
        self,
        query: str,
        *,
        state: WorkflowState | None = None,
        thread_id: str | None = None,
        llm_client: FrameworkLLMClient | None = None,
    ) -> AgentResult:
        """Run the agent to completion and return an :class:`AgentResult`."""
        events: list[StreamEvent] = []
        async for event in self._stream(query, state=state, thread_id=thread_id, llm_client=llm_client):
            events.append(event)
        run_state = self._last_state or WorkflowState(query=query)
        return self._build_result(run_state, events)

    async def astream(
        self,
        query: str,
        *,
        state: WorkflowState | None = None,
        thread_id: str | None = None,
        llm_client: FrameworkLLMClient | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream :class:`StreamEvent` instances as the agent runs."""
        async for event in self._stream(query, state=state, thread_id=thread_id, llm_client=llm_client):
            yield event

    # ----------------------------------------------------------------- private

    _last_state: WorkflowState | None = field(default=None, init=False, repr=False)

    async def _stream(
        self,
        query: str,
        *,
        state: WorkflowState | None,
        thread_id: str | None,
        llm_client: FrameworkLLMClient | None,
    ) -> AsyncIterator[StreamEvent]:
        wf = self.as_workflow()
        run_state = state or WorkflowState(query=query)
        self._apply_extras_to_state(run_state, thread_id=thread_id)
        self._register_python_tools(run_state)
        client = llm_client or self._resolve_default_client()

        from databricks_deep_research.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(wf, client)
        try:
            async for event in executor.execute(run_state):
                yield event
        finally:
            self._last_state = run_state

    def _apply_extras_to_state(self, state: WorkflowState, *, thread_id: str | None) -> None:
        """Thread per-agent extras / capability handles into the workflow state.

        These end up on the :class:`ToolContext` consumed by ``@tool`` callables
        via the ``inject`` map. Phase 1 attaches user-supplied ``extras`` and
        ``thread_id``; Phase 2 layers ``approval_broker``, ``files``,
        ``checkpointer`` on top.
        """
        merged: dict[str, Any] = {}
        if thread_id is not None:
            merged["_framework_thread_id"] = thread_id
        if self.approval_broker is not None:
            merged["_framework_approval_broker"] = self.approval_broker
        if self.files is not None:
            merged["_framework_vfs"] = self.files
        if self.checkpointer is not None:
            merged["_framework_checkpointer"] = self.checkpointer
        merged.update(self.extras)
        # user_id is written AFTER merged.update(self.extras) so caller-
        # supplied extras cannot spoof the canonical _framework_user_id
        # (the HITL endpoint trusts this value for ownership authz).
        if self.user_id is not None:
            merged["_framework_user_id"] = self.user_id
        if merged:
            # WorkflowState carries arbitrary kwargs through to the executor's
            # ToolContext via the runtime ``extras`` plumbing in the harness.
            # Stash on the state object via the .append log so subsequent
            # nodes can read.
            state.append("_framework_api", "_framework_extras", dict(merged))

    def _register_python_tools(self, state: WorkflowState) -> None:
        """Plumb ``@tool``-decorated callables onto state.enterprise_tools.

        :class:`WorkflowExecutor` reads ``state.enterprise_tools`` when
        building its :class:`ToolResolver` overrides — placing tools here
        ensures the LLM resolves them by name without going through the
        decorated YAML factory.
        """
        from databricks_deep_research.api.compile import coerce_tools

        if not self.tools:
            return
        coerced = coerce_tools(list(self.tools))
        existing = list(state.enterprise_tools or [])
        existing_names = {t.definition.name for t in existing if hasattr(t, "definition")}
        for t in coerced:
            if t.definition.name in existing_names:
                continue
            existing.append(t)
            existing_names.add(t.definition.name)
        state.enterprise_tools = existing

    def _resolve_default_client(self) -> FrameworkLLMClient:
        """Use the explicit client if provided, else build a Databricks default."""
        from databricks_deep_research.llm.client import FrameworkLLMClient

        if isinstance(self.model, FrameworkLLMClient):
            return self.model
        return FrameworkLLMClient.from_databricks()

    def _build_result(
        self,
        state: WorkflowState,
        events: list[StreamEvent],
    ) -> AgentResult:
        """Assemble an :class:`AgentResult` from the run's final state."""
        output_key = f"{self.name}_output"
        content_obj = state.extract_output(output_key) if hasattr(state, "extract_output") else None
        content = str(content_obj) if content_obj is not None else ""

        parsed_output: Any
        ok = True
        if self.output_type is not None and content:
            try:
                parsed_output = self.output_type.model_validate_json(content)
            except (ValidationError, ValueError):
                try:
                    parsed_output = self.output_type.model_validate(json.loads(content))
                except Exception:
                    parsed_output = content
                    ok = False
        else:
            parsed_output = content

        verification: VerificationSummary | None = None
        if self.subtype == "synthesizer":
            verification = extract_verification(state, list(state.enterprise_tools or []))
            if not verification.claims and content:
                verification = extract_verification_from_report(content, list(state.enterprise_tools or []))

        usage: dict[str, int] = {}
        for ev in events:
            if hasattr(ev, "usage") and isinstance(ev.usage, dict):
                for k, v in ev.usage.items():
                    if isinstance(v, int):
                        usage[k] = usage.get(k, 0) + v

        return AgentResult(
            content=content,
            output=parsed_output,
            events=events,
            verification=verification,
            tool_calls=[],
            sources=[],
            usage=usage,
            ok=ok,
            run_id=f"agent_{self.name}",
        )

    # ------------------------------------------------------ composition shims

    @staticmethod
    async def _run_compiled_workflow(
        wf: WorkflowDefinition,
        *,
        query: str,
        state: WorkflowState | None = None,
        thread_id: str | None = None,  # noqa: ARG004 — Phase 2 wiring
        llm_client: FrameworkLLMClient | None = None,
    ) -> AgentResult:
        """Run a pre-compiled :class:`WorkflowDefinition` (used by Sequence/Parallel)."""
        events: list[StreamEvent] = []
        run_state = state or WorkflowState(query=query)

        from databricks_deep_research.llm.client import FrameworkLLMClient as _Client
        from databricks_deep_research.workflow.executor import WorkflowExecutor

        client = llm_client or _Client.from_databricks()
        executor = WorkflowExecutor(wf, client)
        async for event in executor.execute(run_state):
            events.append(event)

        # Best-effort output extraction: pick the first declared output_keys entry.
        content_obj = None
        for key in wf.output_keys:
            content_obj = run_state.extract_output(key) if hasattr(run_state, "extract_output") else None
            if content_obj:
                break
        content = str(content_obj) if content_obj is not None else ""
        return AgentResult(
            content=content,
            output=content,
            events=events,
            run_id=wf.id,
        )

    @staticmethod
    async def _stream_compiled_workflow(
        wf: WorkflowDefinition,
        *,
        query: str,
        state: WorkflowState | None = None,
        thread_id: str | None = None,  # noqa: ARG004 — Phase 2 wiring
        llm_client: FrameworkLLMClient | None = None,
    ) -> AsyncIterator[StreamEvent]:
        run_state = state or WorkflowState(query=query)

        from databricks_deep_research.llm.client import FrameworkLLMClient as _Client
        from databricks_deep_research.workflow.executor import WorkflowExecutor

        client = llm_client or _Client.from_databricks()
        executor = WorkflowExecutor(wf, client)
        async for event in executor.execute(run_state):
            yield event


def create_deep_agent(**kwargs: Any) -> Agent:
    """DeepAgents-compatible factory alias for :class:`Agent`.

    Provides DX parity for users coming from DeepAgents without a separate
    class. All keyword arguments forward to :class:`Agent`.
    """
    return Agent(**kwargs)


__all__ = ["Agent", "create_deep_agent"]
