"""Tests for WorkflowRunner and WorkflowResult."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.events.types import StreamEvent, WorkflowStartedEvent
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.runner import WorkflowResult, WorkflowRunner
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
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
            model="test-model", model_mapping=None
        )
        assert runner._client is mock_client
