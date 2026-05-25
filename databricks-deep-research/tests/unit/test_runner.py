"""Tests for WorkflowRunner and WorkflowResult."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.events.types import WorkflowStartedEvent
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.runner import WorkflowResult, WorkflowRunner
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
from databricks_deep_research.workflow.runtime_core.models import (
    EvidenceState,
    SourceRecord,
)
from databricks_deep_research.workflow.state import WorkflowState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_definition(
    output_keys: list[str] | None = None,
    models: dict | None = None,
) -> WorkflowDefinition:
    """Build a minimal WorkflowDefinition for testing."""
    return WorkflowDefinition(
        id="test-wf",
        name="Test Workflow",
        root=WorkflowNode(id="root", label="Root", type=NodeType.sequence, children=[]),
        output_keys=output_keys or ["output"],
        models=models or {},
    )


def _make_mock_client() -> MagicMock:
    """Return a MagicMock that passes isinstance checks for FrameworkLLMClient."""
    mock = MagicMock(spec=FrameworkLLMClient)
    mock.aclose = AsyncMock()
    return mock


async def _fake_execute(state: WorkflowState):  # type: ignore[no-untyped-def]
    """Async generator yielding a single WorkflowStartedEvent."""
    yield WorkflowStartedEvent(
        node_id="root",
        timestamp="2024-01-01T00:00:00Z",
        workflow_id="test-wf",
        workflow_name="Test Workflow",
    )


# ===================================================================
# WorkflowResult — pure logic, no mocks needed
# ===================================================================


class TestWorkflowResultOutput:
    def test_extracts_first_key(self) -> None:
        state = WorkflowState(query="q")
        state.append("node", "output", "hello")
        definition = _make_definition(output_keys=["output"])
        result = WorkflowResult(state=state, definition=definition)
        assert result.output == "hello"

    def test_skips_empty_keys(self) -> None:
        state = WorkflowState(query="q")
        state.append("node", "report", "hello")
        definition = _make_definition(output_keys=["output", "report"])
        result = WorkflowResult(state=state, definition=definition)
        assert result.output == "hello"

    def test_empty_state(self) -> None:
        state = WorkflowState(query="q")
        definition = _make_definition(output_keys=["output"])
        result = WorkflowResult(state=state, definition=definition)
        assert result.output == ""

    def test_no_definition(self) -> None:
        state = WorkflowState(query="q")
        result = WorkflowResult(state=state, definition=None)
        assert result.output == ""


class TestWorkflowResultSources:
    def test_sources_from_pool(self) -> None:
        state = WorkflowState(query="q")
        mock_pool = MagicMock()
        mock_pool.items = [{"url": "https://a.com", "title": "A"}]
        state.pools["sources"] = mock_pool
        result = WorkflowResult(state=state, definition=_make_definition())
        assert result.sources == [{"url": "https://a.com", "title": "A"}]

    def test_sources_from_typed_evidence_when_present(self) -> None:
        state = WorkflowState(query="q")
        store = TypedRuntimeStateStore(query="q")
        store.runtime().capabilities.evidence = EvidenceState(
            sources=[
                SourceRecord(
                    url="https://typed.example",
                    title="Typed",
                    snippet="Typed evidence.",
                    evidence_quality="full_text",
                )
            ]
        )
        state.runtime_store = store
        mock_pool = MagicMock()
        mock_pool.items = [{"url": "https://pool.example", "title": "Pool"}]
        state.pools["sources"] = mock_pool

        result = WorkflowResult(state=state, definition=_make_definition())

        assert result.sources[0]["url"] == "https://typed.example"

    def test_empty_typed_evidence_falls_back_to_pool(self) -> None:
        state = WorkflowState(query="q")
        store = TypedRuntimeStateStore(query="q")
        store.runtime().capabilities.evidence = EvidenceState()
        state.runtime_store = store
        mock_pool = MagicMock()
        mock_pool.items = [{"url": "https://pool.example", "title": "Pool"}]
        state.pools["sources"] = mock_pool

        result = WorkflowResult(state=state, definition=_make_definition())

        assert result.sources == [{"url": "https://pool.example", "title": "Pool"}]

    def test_sources_no_pool(self) -> None:
        state = WorkflowState(query="q")
        result = WorkflowResult(state=state, definition=_make_definition())
        assert result.sources == []


# ===================================================================
# WorkflowRunner._resolve
# ===================================================================


class TestResolve:
    def test_definition_passthrough(self) -> None:
        runner = WorkflowRunner(llm_client=_make_mock_client())
        defn = _make_definition()
        assert runner._resolve(defn) is defn

    @patch("databricks_deep_research.runner.load_workflow")
    def test_string_calls_load_workflow(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_definition()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        runner._resolve("path.yaml")
        mock_load.assert_called_once_with("path.yaml")

    @patch("databricks_deep_research.runner.load_workflow")
    def test_path_calls_load_workflow(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_definition()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        runner._resolve(Path("path.yaml"))
        mock_load.assert_called_once_with("path.yaml")

    @patch("databricks_deep_research.runner.load_workflow_from_dict")
    def test_dict_calls_load_workflow_from_dict(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_definition()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        runner._resolve({"id": "test", "name": "Test", "root": {}})
        mock_load.assert_called_once_with({"id": "test", "name": "Test", "root": {}})


# ===================================================================
# WorkflowRunner.run
# ===================================================================


class TestRun:
    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_returns_workflow_result(self, mock_executor_cls: MagicMock) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        runner = WorkflowRunner(llm_client=_make_mock_client())
        definition = _make_definition()
        result = await runner.run(definition, query="test")

        assert isinstance(result, WorkflowResult)
        assert result.state.query == "test"
        assert result.definition is definition
        assert len(result.events) == 1
        assert result.events[0].event_type == "workflow_started"

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_run_with_custom_state(self, mock_executor_cls: MagicMock) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        custom_state = WorkflowState(query="q", user_token="tok")
        runner = WorkflowRunner(llm_client=_make_mock_client())
        result = await runner.run(_make_definition(), state=custom_state)

        assert result.state is custom_state
        assert result.state.user_token == "tok"


# ===================================================================
# WorkflowRunner.stream
# ===================================================================


class TestStream:
    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_yields_events(self, mock_executor_cls: MagicMock) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        runner = WorkflowRunner(llm_client=_make_mock_client())
        events = [e async for e in runner.stream(_make_definition(), query="q")]

        assert len(events) == 1
        assert events[0].event_type == "workflow_started"

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_sets_last_result_on_completion(
        self, mock_executor_cls: MagicMock
    ) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        definition = _make_definition()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        async for _ in runner.stream(definition, query="q"):
            pass

        assert runner.last_result is not None
        assert runner.last_result.definition is definition

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_sets_last_result_on_early_break(
        self, mock_executor_cls: MagicMock
    ) -> None:
        """BUG 2 regression: last_result must be set even on early break."""

        async def _multi_events(state: WorkflowState):  # type: ignore[no-untyped-def]
            yield WorkflowStartedEvent(
                node_id="root", timestamp="2024-01-01T00:00:00Z",
                workflow_id="test-wf", workflow_name="Test",
            )
            yield WorkflowStartedEvent(
                node_id="root", timestamp="2024-01-01T00:00:00Z",
                workflow_id="test-wf", workflow_name="Test",
            )

        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_multi_events)
        mock_executor_cls.return_value = mock_executor

        definition = _make_definition()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        gen = runner.stream(definition, query="q")
        _ = await gen.__anext__()  # consume first event
        await gen.aclose()  # explicit close triggers finally block

        assert runner.last_result is not None
        assert runner.last_result.definition is definition


# ===================================================================
# WorkflowRunner.factory_context property + new kwargs (PR-1 unification)
# ===================================================================


class TestFactoryContextProperty:
    def test_property_returns_default_context(self) -> None:
        """Without an explicit context, the property returns the default
        ToolFactoryContext.from_defaults() instance the runner builds."""
        runner = WorkflowRunner(llm_client=_make_mock_client())
        assert runner.factory_context is not None
        assert runner.factory_context is runner._factory

    def test_property_returns_caller_provided_context(self) -> None:
        """Caller-supplied factory_context is exposed verbatim through the
        property — used by app code to wire the same context into a custom
        ToolResolver before delegating to runner.stream(...)."""
        ctx = ToolFactoryContext()
        runner = WorkflowRunner(llm_client=_make_mock_client(), factory_context=ctx)
        assert runner.factory_context is ctx


class TestExecutorKwargPassthrough:
    """PR-1 regression — new kwargs on run()/stream() must thread through to
    WorkflowExecutor.__init__ so app callers can collapse their direct
    executor construction into a single runner.stream(...) call."""

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_run_threads_tool_resolver(
        self, mock_executor_cls: MagicMock
    ) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        resolver_sentinel = MagicMock()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        await runner.run(
            _make_definition(), query="q", tool_resolver=resolver_sentinel
        )

        kwargs = mock_executor_cls.call_args.kwargs
        assert kwargs["tool_resolver"] is resolver_sentinel

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_run_threads_tool_registry(
        self, mock_executor_cls: MagicMock
    ) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        registry_sentinel = MagicMock()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        await runner.run(
            _make_definition(), query="q", tool_registry=registry_sentinel
        )

        kwargs = mock_executor_cls.call_args.kwargs
        assert kwargs["tool_registry"] is registry_sentinel

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_stream_threads_context_and_strict(
        self, mock_executor_cls: MagicMock
    ) -> None:
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        context_sentinel = MagicMock()
        runner = WorkflowRunner(llm_client=_make_mock_client())
        events = [
            e
            async for e in runner.stream(
                _make_definition(),
                query="q",
                context=context_sentinel,
                strict_tool_resolution=True,
            )
        ]

        assert len(events) == 1
        kwargs = mock_executor_cls.call_args.kwargs
        assert kwargs["context"] is context_sentinel
        assert kwargs["strict_tool_resolution"] is True

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_defaults_preserve_existing_behavior(
        self, mock_executor_cls: MagicMock
    ) -> None:
        """When the new kwargs aren't passed, they default to None/False —
        existing callers see no change."""
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        runner = WorkflowRunner(llm_client=_make_mock_client())
        await runner.run(_make_definition(), query="q")

        kwargs = mock_executor_cls.call_args.kwargs
        assert kwargs["tool_resolver"] is None
        assert kwargs["tool_registry"] is None
        assert kwargs["context"] is None
        assert kwargs["strict_tool_resolution"] is False


# ===================================================================
# WorkflowRunner.from_databricks
# ===================================================================


# ===================================================================
# WorkflowRunner._resolve_client
# ===================================================================


class TestResolveClient:
    def test_no_models_returns_original_client(self) -> None:
        """Empty models dict -> same client returned."""
        mock_client = _make_mock_client()
        runner = WorkflowRunner(llm_client=mock_client)
        definition = _make_definition(models={})
        assert runner._resolve_client(definition) is mock_client

    def test_yaml_models_returns_derived_client(self) -> None:
        """Definition with models -> derive() called, returns new client."""
        mock_client = _make_mock_client()
        derived_mock = _make_mock_client()
        mock_client.derive = MagicMock(return_value=derived_mock)
        runner = WorkflowRunner(llm_client=mock_client)
        definition = _make_definition(models={"simple": "new-model"})
        result = runner._resolve_client(definition)
        assert result is derived_mock
        mock_client.derive.assert_called_once()

    @pytest.mark.asyncio
    @patch("databricks_deep_research.runner.WorkflowExecutor")
    async def test_executor_receives_derived_client(
        self, mock_executor_cls: MagicMock
    ) -> None:
        """WorkflowExecutor constructor receives the resolved client."""
        mock_executor = MagicMock()
        mock_executor.execute = MagicMock(side_effect=_fake_execute)
        mock_executor_cls.return_value = mock_executor

        mock_client = _make_mock_client()
        derived_mock = _make_mock_client()
        mock_client.derive = MagicMock(return_value=derived_mock)

        runner = WorkflowRunner(llm_client=mock_client)
        definition = _make_definition(models={"simple": "yaml-model"})
        await runner.run(definition, query="test")

        # The executor should have been called with the derived client
        call_args = mock_executor_cls.call_args
        assert call_args[0][1] is derived_mock


# ===================================================================
# WorkflowRunner.from_databricks
# ===================================================================


class TestFromDatabricks:
    @patch(
        "databricks_deep_research.runner.FrameworkLLMClient.from_databricks"
    )
    def test_creates_runner(self, mock_factory: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_factory.return_value = mock_client

        runner = WorkflowRunner.from_databricks(model="test-model")

        mock_factory.assert_called_once_with(
            model="test-model", model_mapping=None, profile=None
        )
        assert runner._client is mock_client

    @patch(
        "databricks_deep_research.runner.FrameworkLLMClient.from_databricks"
    )
    def test_passes_profile_through(self, mock_factory: MagicMock) -> None:
        mock_factory.return_value = _make_mock_client()

        WorkflowRunner.from_databricks(model="m", profile="my-prof")

        mock_factory.assert_called_once_with(
            model="m", model_mapping=None, profile="my-prof"
        )

    @patch(
        "databricks_deep_research.runner.FrameworkLLMClient.from_databricks"
    )
    def test_accepts_factory_context(self, mock_factory: MagicMock) -> None:
        mock_factory.return_value = _make_mock_client()
        ctx = ToolFactoryContext(search_client=MagicMock())

        runner = WorkflowRunner.from_databricks(factory_context=ctx)

        assert runner._factory is ctx

    @patch(
        "databricks_deep_research.runner.FrameworkLLMClient.from_databricks"
    )
    @patch("databricks_deep_research.runner.ToolFactoryContext.from_defaults")
    def test_builds_context_from_brave_api_key(
        self,
        mock_context_factory: MagicMock,
        mock_llm_factory: MagicMock,
    ) -> None:
        mock_llm_factory.return_value = _make_mock_client()
        ctx = ToolFactoryContext(search_client=MagicMock(), user_token="tok")
        mock_context_factory.return_value = ctx

        runner = WorkflowRunner.from_databricks(
            brave_api_key="brave",
            user_token="tok",
        )

        mock_context_factory.assert_called_once_with(
            brave_api_key="brave",
            user_token="tok",
        )
        assert runner._factory is ctx
