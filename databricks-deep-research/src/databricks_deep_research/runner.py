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
from typing import Any

from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.llm.client import (
    FrameworkLLMClient,
    ModelTierConfig,
    parse_model_config,
)
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.loader import load_workflow
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------


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
        """Primary text output — prefer typed report artifact, then legacy output keys."""
        runtime = self.runtime_state
        if runtime is not None and runtime.capabilities.synthesis is not None:
            artifact_id = runtime.capabilities.synthesis.report_artifact_id
            if artifact_id and artifact_id in runtime.artifacts:
                payload = runtime.artifacts[artifact_id].payload
                if payload is not None and str(payload):
                    return str(payload)
        if self.definition is None:
            return ""
        for key in self.definition.output_keys:
            text = self.state.extract_output(key)
            if text is not None and text:
                return str(text)
        return ""

    @property
    def sources(self) -> list[Any]:
        """Sources from typed evidence state, falling back to legacy pool."""
        runtime = self.runtime_state
        if runtime is not None and runtime.capabilities.evidence is not None:
            return [source.model_dump(mode="json") for source in runtime.capabilities.evidence.sources]
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

    @classmethod
    def from_databricks(
        cls,
        *,
        model: str = "databricks-claude-haiku-4-5",
        model_mapping: dict[str, str | ModelTierConfig] | None = None,
    ) -> WorkflowRunner:
        """Create a runner with Databricks auth.

        Same auth chain as ``FrameworkLLMClient.from_databricks()``:
        direct token (``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``), then
        SDK auto-detect (profiles, Azure MSI, etc.).
        """
        client = FrameworkLLMClient.from_databricks(
            model=model,
            model_mapping=model_mapping,
        )
        return cls(llm_client=client)

    async def run(
        self,
        workflow: str | Path | WorkflowDefinition,
        *,
        query: str = "",
        state: WorkflowState | None = None,
    ) -> WorkflowResult:
        """Load and run a workflow to completion, returning a WorkflowResult.

        Parameters
        ----------
        workflow:
            Path to YAML, Path object, or pre-loaded WorkflowDefinition.
        query:
            Query string.  Ignored when *state* is provided.
        state:
            Pre-built state for advanced use (model_overrides, enterprise_tools,
            user_token, domain_filter).  When ``None``, a fresh state is created
            from *query*.
        """
        definition = self._resolve(workflow)
        effective_client = self._resolve_client(definition)
        run_state = state if state is not None else WorkflowState(query=query)
        executor = WorkflowExecutor(definition, effective_client, factory_context=self._factory)
        events = [event async for event in executor.execute(run_state)]
        result = WorkflowResult(
            state=run_state,
            events=events,
            definition=definition,
        )
        self._last_result = result
        return result

    async def stream(
        self,
        workflow: str | Path | WorkflowDefinition,
        *,
        query: str = "",
        state: WorkflowState | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Load and stream a workflow, yielding events.

        Access the final result via ``runner.last_result`` after iteration.
        """
        definition = self._resolve(workflow)
        effective_client = self._resolve_client(definition)
        run_state = state if state is not None else WorkflowState(query=query)
        executor = WorkflowExecutor(definition, effective_client, factory_context=self._factory)
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
        workflow: str | Path | WorkflowDefinition,
    ) -> WorkflowDefinition:
        """Accept str path, Path, or pre-loaded definition."""
        if isinstance(workflow, WorkflowDefinition):
            return workflow
        return load_workflow(str(workflow))
