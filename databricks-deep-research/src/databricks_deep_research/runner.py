"""WorkflowRunner — high-level convenience API for running framework workflows.

Reduces the typical "5 imports, 10 lines of setup" pattern to a single import
and a few lines::

    from databricks_deep_research import WorkflowRunner

    runner = WorkflowRunner.from_databricks()
    result = await runner.run("simple_research.yaml", query="What is AI?")
    print(result.output)

For streaming::

    async for event in runner.stream("simple_research.yaml", query="What is AI?"):
        print(event.event_type)
    print(runner.last_result.output)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.llm.client import (
    FrameworkLLMClient,
    ModelTierConfig,
    parse_model_config,
)
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.loader import load_workflow, load_workflow_from_dict
from databricks_deep_research.workflow.state import WorkflowState

if TYPE_CHECKING:
    from databricks_deep_research.tools.registry import ToolRegistry
    from databricks_deep_research.tools.resolver import ToolResolver
    from databricks_deep_research.workflow.context import ExecutionContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------


# Banner prepended when every claim failed verification but the synthesizer
# streamed substantive content. The framework hands the draft back to the
# caller with a clear warning rather than letting the post-Stage-8 empty
# string surface as a blank chat response. See plan v2.3 (citation pipeline
# UX backstop) for the failure-mode this guards against.
_UNVERIFIED_DRAFT_BANNER_TEMPLATE = (
    "> ⚠️ **Citations could not be verified.** "
    "The framework's entailment checker did not ground "
    "any of the {total_claims} claims this draft contains. "
    "Numbers below come directly from the retrieved corpus "
    "chunks; treat as a draft, not a final answer.\n\n"
)

# Minimum streamed-content length below which we don't bother surfacing
# the draft — short fragments are usually preamble headers, not answers.
_UNVERIFIED_DRAFT_MIN_CHARS = 200


def _verification_summary_from_events(
    events: list[StreamEvent],
) -> tuple[int, int] | None:
    """Return ``(total_claims, verified_claims)`` from the LAST
    ``verification_summary`` event in the stream, or ``None`` if no such
    event was emitted.

    A workflow can emit multiple verification_summary events (one per
    synthesizer pass under SYNTH_PIPELINE_V2); the last one reflects the
    final state Stage 8 acted on.
    """
    last: tuple[int, int] | None = None
    for event in events:
        if getattr(event, "event_type", None) == "verification_summary":
            total = int(getattr(event, "total_claims", 0) or 0)
            verified = int(getattr(event, "verified_claims", 0) or 0)
            last = (total, verified)
    return last


def _joined_stream_chunks(events: list[StreamEvent]) -> str:
    """Concatenate every ``agent_stream_chunk`` payload from *events*.

    This is the same fallback the main app's framework_orchestrator used
    before the backstop migrated into the framework — preserved here so
    every framework consumer (shell app, SDK, notebooks) gets the same
    safety net.
    """
    parts: list[str] = []
    for event in events:
        if getattr(event, "event_type", None) == "agent_stream_chunk":
            chunk = getattr(event, "chunk", None) or ""
            if chunk:
                parts.append(str(chunk))
    return "".join(parts)


@dataclass
class WorkflowResult:
    """Result of a completed workflow run."""

    state: WorkflowState
    events: list[StreamEvent] = field(default_factory=list)
    definition: WorkflowDefinition | None = None

    @property
    def runtime_state(self) -> Any | None:
        return self.state.runtime_state()

    @property
    def output(self) -> str:
        """Primary text output — prefer typed report artifact, then legacy output keys.

        Plan v2.3 backstop: when the citation pipeline marked zero claims
        as verified (Stage 4 returned ``unsupported`` / ``abstained`` for
        every claim, Stage 8 then surgically deleted all the claim text)
        AND the synthesizer streamed substantive content, surface the
        streamed draft with a clear banner rather than returning an empty
        string. The framework's ``verification_summary`` is still
        accurate; this only repairs the user-facing report.
        """
        runtime = self.runtime_state
        primary_text: str | None = None
        if runtime is not None and runtime.capabilities.synthesis is not None:
            artifact_id = runtime.capabilities.synthesis.report_artifact_id
            if artifact_id and artifact_id in runtime.artifacts:
                payload = runtime.artifacts[artifact_id].payload
                if payload is not None and str(payload):
                    primary_text = str(payload)
        if primary_text is None and self.definition is not None:
            for key in self.definition.output_keys:
                text = self.state.extract_output(key)
                if text is not None and text:
                    primary_text = str(text)
                    break
        if primary_text is None:
            primary_text = ""

        verif = _verification_summary_from_events(self.events)
        if verif is not None:
            total_claims, verified_claims = verif
            if total_claims > 0 and verified_claims == 0:
                joined = _joined_stream_chunks(self.events)
                if (
                    len(joined) >= _UNVERIFIED_DRAFT_MIN_CHARS
                    and len(primary_text) < len(joined) // 2
                ):
                    logger.warning(
                        "WORKFLOW_RESULT_UNVERIFIED_DRAFT_BACKSTOP "
                        "total_claims=%d verified_claims=%d "
                        "primary_text_len=%d streamed_chars=%d",
                        total_claims,
                        verified_claims,
                        len(primary_text),
                        len(joined),
                    )
                    banner = _UNVERIFIED_DRAFT_BANNER_TEMPLATE.format(
                        total_claims=total_claims
                    )
                    return banner + joined
        return primary_text

    @property
    def sources(self) -> list[Any]:
        """Sources from typed evidence state, falling back to legacy pool."""
        runtime = self.runtime_state
        if runtime is not None and runtime.capabilities.evidence is not None:
            typed_sources = [
                source.model_dump(mode="json")
                for source in runtime.capabilities.evidence.sources
            ]
            if typed_sources:
                return typed_sources
        pool = self.state.pools.get("sources")
        if pool is not None and hasattr(pool, "items"):
            return pool.items  # type: ignore[no-any-return]
        return []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """High-level API for running framework workflows.

    Wraps client creation, tool factory, workflow loading, and execution
    into a clean interface.  Designed for scripts, notebooks, and examples.

    Not thread-safe: ``last_result`` is instance-level state.  For concurrent
    use, create separate runner instances.
    """

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext | None = None,
    ) -> None:
        self._client = llm_client
        self._factory = factory_context or ToolFactoryContext.from_defaults()
        self._last_result: WorkflowResult | None = None
        self._registered_tools: list[Any] = []

    def register_tools(self, *tools: Any) -> None:
        """Register Python ``@tool``-decorated callables (or any
        :class:`ResearchTool`) so that workflows referencing them by name
        resolve correctly without needing an explicit ``decorated`` factory
        import.

        The registered tools are appended to ``state.enterprise_tools`` for
        every subsequent ``run()`` / ``stream()`` call on this runner. Pass
        already-resolved :class:`ResearchTool` instances or plain callables
        — callables are auto-wrapped via :func:`tool`.
        """
        from databricks_deep_research.api.compile import coerce_tools

        coerced = coerce_tools(list(tools))
        existing_names = {
            getattr(t, "definition", None) and t.definition.name
            for t in self._registered_tools
        }
        for t in coerced:
            if t.definition.name in existing_names:
                continue
            self._registered_tools.append(t)
            existing_names.add(t.definition.name)

    @classmethod
    def from_databricks(
        cls,
        *,
        model: str = "databricks-claude-haiku-4-5",
        model_mapping: dict[str, str | ModelTierConfig] | None = None,
        profile: str | None = None,
        factory_context: ToolFactoryContext | None = None,
        brave_api_key: str | None = None,
        user_token: str | None = None,
    ) -> WorkflowRunner:
        """Create a runner with Databricks auth.

        Same auth chain as ``FrameworkLLMClient.from_databricks()``:
        direct token (``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``), then
        SDK auto-detect (profiles, Azure MSI, etc.).

        Parameters
        ----------
        profile:
            Databricks CLI profile name from ``~/.databrickscfg``.
            Forwarded to ``FrameworkLLMClient.from_databricks()``.
        factory_context:
            Optional pre-built tool dependency context. Use this when the
            runtime must inject app-bound resources such as Databricks clients
            or search providers.
        brave_api_key:
            Optional Brave Search API key. Used to build a default
            ``ToolFactoryContext`` when *factory_context* is not provided.
        user_token:
            Optional OBO token forwarded into the default ``ToolFactoryContext``
            when *factory_context* is not provided.
        """
        client = FrameworkLLMClient.from_databricks(
            model=model,
            model_mapping=model_mapping,
            profile=profile,
        )
        resolved_factory_context = factory_context
        if resolved_factory_context is None and (brave_api_key or user_token):
            resolved_factory_context = ToolFactoryContext.from_defaults(
                brave_api_key=brave_api_key,
                user_token=user_token,
            )
        return cls(llm_client=client, factory_context=resolved_factory_context)

    async def run(
        self,
        workflow: str | Path | WorkflowDefinition | dict[str, Any],
        *,
        query: str = "",
        state: WorkflowState | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        tool_resolver: ToolResolver | None = None,
        tool_registry: ToolRegistry | None = None,
        context: ExecutionContext | None = None,
        strict_tool_resolution: bool = False,
    ) -> WorkflowResult:
        """Load and run a workflow to completion, returning a WorkflowResult.

        Parameters
        ----------
        workflow:
            Path to YAML, Path object, pre-loaded WorkflowDefinition, or
            a plain dict matching the YAML schema.
        query:
            Query string.  Ignored when *state* is provided.
        state:
            Pre-built state for advanced use (model_overrides, enterprise_tools,
            user_token, domain_filter).  When ``None``, a fresh state is created
            from *query*.
        conversation_history:
            Optional list of ``{role, content}`` dicts (OpenAI message format)
            representing prior conversation turns.  When provided, this value
            is seeded onto the run state and made available to every agent via
            ``AgentInput.conversation_history``.  If *state* was also provided
            and already contained history, this kwarg takes precedence (caller
            intent is the authority).
        tool_resolver:
            Custom :class:`ToolResolver` (with pre-built overrides or factories).
            When provided, takes precedence over the runner's default resolver.
        tool_registry:
            Custom :class:`ToolRegistry` pre-populated with tools (e.g., for
            ``type: tool`` workflow nodes that aren't routed via factories).
        context:
            Optional :class:`ExecutionContext` carrying per-run state such as
            HITL pause/resume tokens, chat_id, broker references.
        strict_tool_resolution:
            When True, raise :class:`WorkflowError` immediately if any agent
            node declares a tool that the resolver cannot construct. Default
            False matches notebook/playground use; production callers should
            opt in so deployment misconfigurations surface loudly.
        """
        definition = self._resolve(workflow)
        effective_client = self._resolve_client(definition)
        run_state = state if state is not None else WorkflowState(query=query)
        if conversation_history is not None:
            run_state.conversation_history = list(conversation_history)
        self._inject_registered_tools(run_state)
        executor = WorkflowExecutor(
            definition,
            effective_client,
            factory_context=self._factory,
            tool_resolver=tool_resolver,
            tool_registry=tool_registry,
            context=context,
            strict_tool_resolution=strict_tool_resolution,
        )
        events = [event async for event in executor.execute(run_state)]
        result = WorkflowResult(
            state=run_state,
            events=events,
            definition=definition,
        )
        self._last_result = result
        return result

    def _inject_registered_tools(self, state: WorkflowState) -> None:
        """Append registered Python tools to the state's enterprise_tools."""
        if not self._registered_tools:
            return
        existing = list(state.enterprise_tools or [])
        existing_names = {
            t.definition.name for t in existing if hasattr(t, "definition")
        }
        for t in self._registered_tools:
            if t.definition.name in existing_names:
                continue
            existing.append(t)
            existing_names.add(t.definition.name)
        state.enterprise_tools = existing

    async def stream(
        self,
        workflow: str | Path | WorkflowDefinition | dict[str, Any],
        *,
        query: str = "",
        state: WorkflowState | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        tool_resolver: ToolResolver | None = None,
        tool_registry: ToolRegistry | None = None,
        context: ExecutionContext | None = None,
        strict_tool_resolution: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Load and stream a workflow, yielding events.

        Access the final result via ``runner.last_result`` after iteration.

        Parameters
        ----------
        workflow:
            Path to YAML, Path object, pre-loaded WorkflowDefinition, or
            a plain dict matching the YAML schema.
        query:
            Query string.  Ignored when *state* is provided.
        state:
            Pre-built state for advanced use (model_overrides, enterprise_tools,
            user_token, domain_filter).  When ``None``, a fresh state is created
            from *query*.
        conversation_history:
            Optional list of ``{role, content}`` dicts (OpenAI message format)
            representing prior conversation turns.  When provided, this value
            is seeded onto the run state and made available to every agent via
            ``AgentInput.conversation_history``.  If *state* was also provided
            and already contained history, this kwarg takes precedence (caller
            intent is the authority).
        tool_resolver:
            Custom :class:`ToolResolver` (with pre-built overrides or factories).
            When provided, takes precedence over the runner's default resolver.
        tool_registry:
            Custom :class:`ToolRegistry` pre-populated with tools (e.g., for
            ``type: tool`` workflow nodes that aren't routed via factories).
        context:
            Optional :class:`ExecutionContext` carrying per-run state such as
            HITL pause/resume tokens, chat_id, broker references.
        strict_tool_resolution:
            When True, raise :class:`WorkflowError` immediately if any agent
            node declares a tool that the resolver cannot construct. Default
            False matches notebook/playground use; production callers should
            opt in so deployment misconfigurations surface loudly.
        """
        definition = self._resolve(workflow)
        effective_client = self._resolve_client(definition)
        run_state = state if state is not None else WorkflowState(query=query)
        if conversation_history is not None:
            run_state.conversation_history = list(conversation_history)
        self._inject_registered_tools(run_state)
        executor = WorkflowExecutor(
            definition,
            effective_client,
            factory_context=self._factory,
            tool_resolver=tool_resolver,
            tool_registry=tool_registry,
            context=context,
            strict_tool_resolution=strict_tool_resolution,
        )
        events: list[StreamEvent] = []
        try:
            async for event in executor.execute(run_state):
                events.append(event)
                yield event
        finally:
            self._last_result = WorkflowResult(
                state=run_state,
                events=events,
                definition=definition,
            )

    @property
    def last_result(self) -> WorkflowResult | None:
        """Result from the most recent ``run()`` or ``stream()`` call."""
        return self._last_result

    @property
    def factory_context(self) -> ToolFactoryContext:
        """Return the :class:`ToolFactoryContext` this runner uses.

        Exposed so callers (notably the app's ``framework_orchestrator``) can
        build their own :class:`ToolResolver` against the same context the
        runner will use internally — keeping factory wiring consistent between
        the app's resolver and the runner's executor.
        """
        return self._factory

    async def aclose(self) -> None:
        """Close the underlying LLM client."""
        await self._client.aclose()

    def _resolve_client(self, definition: WorkflowDefinition) -> FrameworkLLMClient:
        """Apply workflow-level model config from the YAML ``models:`` section.

        When the workflow defines a ``models:`` section, its entries are layered
        on top of the runner's base client via ``derive()``.  YAML models
        override Python-supplied mappings when present.
        """
        if not definition.models:
            return self._client
        yaml_mapping = parse_model_config(definition.models)
        logger.info(
            "RUNNER_APPLY_YAML_MODELS tiers=%s",
            list(yaml_mapping.keys()),
        )
        return self._client.derive(yaml_mapping)

    def _resolve(
        self,
        workflow: str | Path | WorkflowDefinition | dict[str, Any],
    ) -> WorkflowDefinition:
        """Accept str path, Path, pre-loaded definition, or raw dict."""
        if isinstance(workflow, WorkflowDefinition):
            return workflow
        if isinstance(workflow, dict):
            return load_workflow_from_dict(workflow)
        return load_workflow(str(workflow))
