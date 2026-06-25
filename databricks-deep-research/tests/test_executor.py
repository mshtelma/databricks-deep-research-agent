"""Tests for WorkflowExecutor — all 8 node types with mock agents/tools."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.agents.config import AgentNodeConfig, PlanAndExecuteNodeConfig
from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.errors import (
    PlanningContractError,
    WorkflowConditionEvaluationError,
    WorkflowExecutionError,
)
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    BranchSelectedEvent,
    LoopExitEvent,
    LoopIterationEvent,
    NodeErrorEvent,
    NodeSkippedEvent,
    NodeStartedEvent,
    PlanCreatedEvent,
    ReplanTriggeredEvent,
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
    WorkflowStartedEvent,
)
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ToolDefinition
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.conditions import ConditionBranch, StateCondition
from databricks_deep_research.workflow.definition import (
    ErrorConfig,
    NodeType,
    SourceDefinition,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import (
    NormalizedPlanContract,
    PlanCycleContext,
    WorkflowExecutor,
    _build_available_source_catalog,
    _build_evaluator_runtime_context,
    _build_planner_runtime_context,
    _extract_items,
    _extract_raw_plan_contract,
    _finalize_plan_contract,
    _format_all_observations,
    _format_source_quality,
    _normalize_executable_plan_contract,
    _populate_synthesis_state,
    run_workflow,
)
from databricks_deep_research.workflow.state import WorkflowState
from tests.conftest import (
    build_mock_llm_client as _mock_llm_client,
)
from tests.conftest import (
    collect_events as _collect_events,
)
from tests.conftest import (
    events_of_type as _events_of_type,
)


def _make_definition(root: WorkflowNode) -> WorkflowDefinition:
    """Wrap a root node into a minimal WorkflowDefinition."""
    return WorkflowDefinition(
        id="test-wf",
        name="Test Workflow",
        root=root,
    )


class TestStructuredOutputSerialization:
    """``WorkflowCompletedEvent.final_report`` must be valid JSON for
    structured-output deliverables, so the persisted ``message.content`` is
    parseable by the frontend's structured-output renderer.

    Regression: a dict report (what plugin assembler nodes write to
    ``state['report']``) was serialized via ``str()`` → a Python ``repr`` with
    single quotes → the frontend's ``JSON.parse(content)`` threw → the rich
    spec renderer never fired and the chat showed raw text.
    """

    async def _completed_for_report(self, report: Any) -> WorkflowCompletedEvent:
        async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            # The real harness writes the agent output into state; mirror that so
            # the completion step reads ``state["report"]``.
            st = kwargs.get("state")
            if st is not None:
                st.append(node_id, "report", report)
            return AgentOutput(content=report, output_key="report", events=[])

        root = WorkflowNode(
            id="assembler",
            type=NodeType.agent,
            label="assembler",
            config={"subtype": "researcher", "output_key": "report"},
        )
        executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())
        state = WorkflowState(query="test")
        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)
        completed = _events_of_type(events, WorkflowCompletedEvent)
        assert completed, "expected a WorkflowCompletedEvent"
        return completed[-1]  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_dict_report_with_output_type_serializes_as_json(self) -> None:
        report = {"output_type": "account_intel", "account_name": "Acme", "id": "abc"}
        evt = await self._completed_for_report(report)
        # Must round-trip as JSON (double-quoted), not a Python repr.
        parsed = json.loads(evt.final_report)  # type: ignore[attr-defined]
        assert parsed["output_type"] == "account_intel"
        assert parsed["account_name"] == "Acme"
        assert evt.structured_output == report  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_basemodel_report_serializes_as_json(self) -> None:
        from pydantic import BaseModel

        class _Out(BaseModel):
            output_type: str = "demo"
            title: str = "T"

        evt = await self._completed_for_report(_Out())
        parsed = json.loads(evt.final_report)  # type: ignore[attr-defined]
        assert parsed["output_type"] == "demo"
        assert evt.structured_output == {"output_type": "demo", "title": "T"}  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_plain_markdown_report_passthrough(self) -> None:
        evt = await self._completed_for_report("# Report\n\nbody")
        # Non-structured string reports stay as-is; no structured_output.
        assert evt.final_report == "# Report\n\nbody"  # type: ignore[attr-defined]
        assert evt.structured_output is None  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------


class TestSequenceNode:
    @pytest.mark.asyncio
    async def test_children_run_in_order(self) -> None:
        """Sequence node executes children one after another."""
        execution_order: list[str] = []

        AgentOutput(
            content="result",
            output_key="output",
            events=[],
        )

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            execution_order.append(node_id)
            return AgentOutput(
                content=f"result-{node_id}",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview=f"result-{node_id}",
                    )
                ],
            )

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="agent_a",
                    type=NodeType.agent,
                    label="Agent A",
                    config={
                        "subtype": "researcher",
                        "output_key": "findings_a",
                    },
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="Agent B",
                    config={
                        "subtype": "researcher",
                        "output_key": "findings_b",
                    },
                ),
            ],
        )
        defn = _make_definition(root)
        llm = _mock_llm_client()
        executor = WorkflowExecutor(defn, llm)
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        assert execution_order == ["agent_a", "agent_b"]
        assert isinstance(events[0], WorkflowStartedEvent)
        assert isinstance(events[-1], WorkflowCompletedEvent)


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


class TestParallelNode:
    @pytest.mark.asyncio
    async def test_children_run_concurrently(self) -> None:
        """Parallel node spawns children concurrently and collects events."""
        executed: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            executed.append(node_id)
            return AgentOutput(
                content=f"result-{node_id}",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="ok",
                    )
                ],
            )

        root = WorkflowNode(
            id="par",
            type=NodeType.parallel,
            label="parallel",
            children=[
                WorkflowNode(
                    id="child_1",
                    type=NodeType.agent,
                    label="C1",
                    config={"subtype": "researcher", "output_key": "r1"},
                ),
                WorkflowNode(
                    id="child_2",
                    type=NodeType.agent,
                    label="C2",
                    config={"subtype": "researcher", "output_key": "r2"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        # Both children ran (order may vary due to concurrency)
        assert set(executed) == {"child_1", "child_2"}
        started = _events_of_type(events, NodeStartedEvent)
        # root parallel + two children = at least 3 NodeStartedEvents
        assert len(started) >= 3


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class TestLoopNode:
    @pytest.mark.asyncio
    async def test_loop_exits_on_max_iterations(self) -> None:
        """Loop runs up to max_iterations when condition never fires."""
        call_count = 0

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            return AgentOutput(
                content="step",
                output_key="output",
                events=[],
            )

        until_cond = StateCondition(key="done", operator="exists")

        root = WorkflowNode(
            id="loop",
            type=NodeType.loop,
            label="loop",
            config={
                "until": until_cond.model_dump(),
                "min_iterations": 1,
                "max_iterations": 3,
            },
            children=[
                WorkflowNode(
                    id="body",
                    type=NodeType.agent,
                    label="body",
                    config={"subtype": "researcher", "output_key": "step_output"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        assert call_count == 3
        exits = _events_of_type(events, LoopExitEvent)
        assert len(exits) == 1
        assert exits[0].reason == "max_iterations"  # type: ignore[attr-defined]
        assert exits[0].total_iterations == 3  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_loop_exits_on_condition(self) -> None:
        """Loop exits early when the until-condition evaluates to True."""
        call_count = 0

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            # On second call, set state so condition fires
            st = kwargs.get("state")
            if st and call_count >= 2:
                st.append(node_id, "complete", True)
            return AgentOutput(content="step", output_key="output", events=[])

        until_cond = StateCondition(key="complete", operator="exists")

        root = WorkflowNode(
            id="loop",
            type=NodeType.loop,
            label="loop",
            config={
                "until": until_cond.model_dump(),
                "min_iterations": 1,
                "max_iterations": 10,
            },
            children=[
                WorkflowNode(
                    id="body",
                    type=NodeType.agent,
                    label="body",
                    config={"subtype": "researcher", "output_key": "step_output"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        exits = _events_of_type(events, LoopExitEvent)
        assert len(exits) == 1
        assert exits[0].reason == "condition_met"  # type: ignore[attr-defined]
        assert exits[0].total_iterations == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_loop_min_iterations_respected(self) -> None:
        """Loop does not exit before min_iterations even if condition is met."""
        call_count = 0

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            st = kwargs.get("state")
            # Set the condition true on the very first call
            if st and call_count == 1:
                st.append(node_id, "early_done", True)
            return AgentOutput(content="step", output_key="output", events=[])

        until_cond = StateCondition(key="early_done", operator="exists")

        root = WorkflowNode(
            id="loop",
            type=NodeType.loop,
            label="loop",
            config={
                "until": until_cond.model_dump(),
                "min_iterations": 3,
                "max_iterations": 10,
            },
            children=[
                WorkflowNode(
                    id="body",
                    type=NodeType.agent,
                    label="body",
                    config={"subtype": "researcher", "output_key": "step_output"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        # Should run at least min_iterations (3) before checking condition
        iterations = _events_of_type(events, LoopIterationEvent)
        assert len(iterations) >= 3


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------


class TestConditionalNode:
    @pytest.mark.asyncio
    async def test_selects_matching_branch(self) -> None:
        """Conditional selects the first matching condition's branch."""
        executed_children: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            executed_children.append(node_id)
            return AgentOutput(content="branch", output_key="output", events=[])

        # Condition on state: status == "ready"
        cond = StateCondition(key="status", operator="eq", value="ready")

        root = WorkflowNode(
            id="cond",
            type=NodeType.conditional,
            label="conditional",
            config={
                "conditions": [cond.model_dump()],
                "default_branch": 1,
            },
            children=[
                WorkflowNode(
                    id="branch_0",
                    type=NodeType.agent,
                    label="Branch 0",
                    config={"subtype": "researcher", "output_key": "b0"},
                ),
                WorkflowNode(
                    id="branch_1",
                    type=NodeType.agent,
                    label="Branch 1 (default)",
                    config={"subtype": "researcher", "output_key": "b1"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")
        state.append("init", "status", "ready")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        branch_events = _events_of_type(events, BranchSelectedEvent)
        assert len(branch_events) == 1
        assert branch_events[0].branch_index == 0  # type: ignore[attr-defined]
        assert "branch_0" in executed_children

    @pytest.mark.asyncio
    async def test_selects_condition_branch_child_index(self) -> None:
        """Conditional respects ConditionBranch.child_index, not condition list index."""
        executed_children: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            executed_children.append(node_id)
            return AgentOutput(content="branch", output_key="output", events=[])

        branch = ConditionBranch(
            condition=StateCondition(key="status", operator="eq", value="ready"),
            child_index=1,
        )

        root = WorkflowNode(
            id="cond",
            type=NodeType.conditional,
            label="conditional",
            config={
                "conditions": [branch.model_dump()],
                "default_branch": 0,
            },
            children=[
                WorkflowNode(
                    id="branch_0",
                    type=NodeType.agent,
                    label="Branch 0 (default)",
                    config={"subtype": "researcher", "output_key": "b0"},
                ),
                WorkflowNode(
                    id="branch_1",
                    type=NodeType.agent,
                    label="Branch 1",
                    config={"subtype": "researcher", "output_key": "b1"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")
        state.append("init", "status", "ready")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        branch_events = _events_of_type(events, BranchSelectedEvent)
        assert len(branch_events) == 1
        assert branch_events[0].branch_index == 1  # type: ignore[attr-defined]
        assert "branch_1" in executed_children
        assert "branch_0" not in executed_children

    @pytest.mark.asyncio
    async def test_falls_through_to_default(self) -> None:
        """Conditional falls through to default_branch when no condition matches."""
        executed_children: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            executed_children.append(node_id)
            return AgentOutput(content="default", output_key="output", events=[])

        # Condition that won't match
        cond = StateCondition(key="status", operator="eq", value="impossible")

        root = WorkflowNode(
            id="cond",
            type=NodeType.conditional,
            label="conditional",
            config={
                "conditions": [cond.model_dump()],
                "default_branch": 1,
            },
            children=[
                WorkflowNode(
                    id="branch_0",
                    type=NodeType.agent,
                    label="Branch 0",
                    config={"subtype": "researcher", "output_key": "b0"},
                ),
                WorkflowNode(
                    id="branch_1",
                    type=NodeType.agent,
                    label="Branch 1 (default)",
                    config={"subtype": "researcher", "output_key": "b1"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")
        state.append("init", "status", "pending")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        branch_events = _events_of_type(events, BranchSelectedEvent)
        assert branch_events[0].branch_index == 1  # type: ignore[attr-defined]
        assert "no conditions matched" in branch_events[0].condition_summary
        assert "branch_1" in executed_children

    @pytest.mark.asyncio
    async def test_missing_condition_operand_raises_instead_of_defaulting(self) -> None:
        """Missing condition operands fail closed instead of selecting default_branch."""
        executed_children: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            executed_children.append(node_id)
            return AgentOutput(content="branch", output_key="output", events=[])

        cond = StateCondition(key="missing.status", operator="eq", value="ready")

        root = WorkflowNode(
            id="cond",
            type=NodeType.conditional,
            label="conditional",
            config={
                "conditions": [cond.model_dump()],
                "default_branch": 1,
            },
            children=[
                WorkflowNode(
                    id="branch_0",
                    type=NodeType.agent,
                    label="Branch 0",
                    config={"subtype": "researcher", "output_key": "b0"},
                ),
                WorkflowNode(
                    id="branch_1",
                    type=NodeType.agent,
                    label="Branch 1 (default)",
                    config={"subtype": "researcher", "output_key": "b1"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with (
            patch(
                "databricks_deep_research.workflow.executor.execute_agent",
                side_effect=fake_execute_agent,
            ),
            pytest.raises(WorkflowConditionEvaluationError, match="condition\\[0\\]"),
        ):
            await _collect_events(executor, state)

        assert executed_children == []


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TestAgentNode:
    @pytest.mark.asyncio
    async def test_agent_writes_output_to_state(self) -> None:
        """Agent node calls execute_agent and output is written to state."""
        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            # The harness writes to state internally; simulate that
            state = kwargs.get("state")
            if state:
                state.append(node_id, "analysis", "deep analysis result")
            return AgentOutput(
                content="deep analysis result",
                output_key="analysis",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="analysis",
                        output_preview="deep analysis result",
                    )
                ],
            )

        root = WorkflowNode(
            id="agent_node",
            type=NodeType.agent,
            label="Analyst",
            config={
                "subtype": "researcher",
                "output_key": "analysis",
                "model_tier": "analytical",
            },
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="analyze this")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        assert state.get("analysis") == "deep analysis result"
        output_events = _events_of_type(events, AgentOutputEvent)
        assert len(output_events) >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_skip_on_error(self) -> None:
        """Node with on_error=skip emits NodeSkippedEvent and continues."""
        async def failing_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            raise RuntimeError("boom")

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="failing",
                    type=NodeType.agent,
                    label="Failing Agent",
                    config={"subtype": "researcher", "output_key": "out"},
                    error_handling=ErrorConfig(on_error="skip"),
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=failing_agent,
        ):
            events = await _collect_events(executor, state)

        skipped = _events_of_type(events, NodeSkippedEvent)
        assert len(skipped) == 1
        assert "boom" in skipped[0].reason  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_retry_on_error(self) -> None:
        """Node with on_error=retry retries the specified number of times."""
        call_count = 0

        async def flaky_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return AgentOutput(
                content="success",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="success",
                    )
                ],
            )

        root = WorkflowNode(
            id="retryable",
            type=NodeType.agent,
            label="Retryable Agent",
            config={"subtype": "researcher", "output_key": "out"},
            error_handling=ErrorConfig(
                on_error="retry", max_retries=3, retry_delay_seconds=0.01
            ),
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=flaky_agent,
        ):
            events = await _collect_events(executor, state)

        # First call fails, then retry 1 fails, retry 2 succeeds
        retry_errors = _events_of_type(events, NodeErrorEvent)
        assert len(retry_errors) >= 1
        assert any(e.will_retry for e in retry_errors)  # type: ignore[attr-defined]
        # The workflow should complete (not raise)
        assert isinstance(events[-1], WorkflowCompletedEvent)

    @pytest.mark.asyncio
    async def test_error_propagates_with_fail_policy(self) -> None:
        """Node with default on_error=fail raises the exception."""
        async def failing_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            raise RuntimeError("fatal error")

        root = WorkflowNode(
            id="fatal",
            type=NodeType.agent,
            label="Fatal Agent",
            config={"subtype": "researcher", "output_key": "out"},
            # No error_handling => defaults to fail
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=failing_agent,
        ), pytest.raises(RuntimeError, match="fatal error"):
            await _collect_events(executor, state)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_state_stops_execution(self) -> None:
        """Setting is_cancelled on state stops further node execution."""
        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="child",
                    type=NodeType.agent,
                    label="Agent",
                    config={"subtype": "researcher", "output_key": "out"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")
        state.is_cancelled = True

        # Should not raise — cancellation is caught
        events = await _collect_events(executor, state)

        # Should get WorkflowStarted and WorkflowCompleted, but no child
        started = _events_of_type(events, NodeStartedEvent)
        # The seq node itself should not start because the cancellation
        # check happens at the beginning of _exec_node
        assert len(started) == 0


# ---------------------------------------------------------------------------
# run_workflow convenience function
# ---------------------------------------------------------------------------


class TestRunWorkflow:
    @pytest.mark.asyncio
    async def test_run_workflow_returns_state_and_events(self) -> None:
        """The run_workflow convenience function collects events."""
        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            state = kwargs.get("state")
            if state:
                state.append(node_id, "output", "final result")
            return AgentOutput(
                content="final result",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="final result",
                    )
                ],
            )

        defn = WorkflowDefinition(
            id="wf-1",
            name="Simple",
            root=WorkflowNode(
                id="agent",
                type=NodeType.agent,
                label="Agent",
                config={"subtype": "synthesizer", "output_key": "output"},
            ),
        )
        llm = _mock_llm_client()

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            state, events = await run_workflow(
                defn, llm, initial_state={"query": "hello"}
            )

        assert state.get("output") == "final result"
        assert any(isinstance(e, WorkflowStartedEvent) for e in events)
        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_run_workflow_wraps_failure_with_partial_events(self) -> None:
        async def failing_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            raise RuntimeError("fatal agent crash")

        defn = WorkflowDefinition(
            id="wf-fail",
            name="Failure",
            root=WorkflowNode(
                id="agent",
                type=NodeType.agent,
                label="Agent",
                config={"subtype": "researcher", "output_key": "output"},
            ),
        )
        llm = _mock_llm_client()

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=failing_execute_agent,
        ), pytest.raises(WorkflowExecutionError, match="fatal agent crash") as exc_info:
            await run_workflow(defn, llm, initial_state={"query": "hello"})

        assert isinstance(exc_info.value.cause, RuntimeError)
        assert any(isinstance(e, WorkflowStartedEvent) for e in exc_info.value.events)
        assert any(isinstance(e, WorkflowFailedEvent) for e in exc_info.value.events)


class TestExtractItems:
    def test_from_dict(self) -> None:
        """Standard dict path navigation."""
        items = _extract_items({"steps": [{"id": "s1"}]}, "steps")
        assert len(items) == 1
        assert items[0]["id"] == "s1"

    def test_from_string_json(self) -> None:
        """_extract_items recovers items from raw JSON string."""
        raw = '{"steps": [{"id": "step-1", "title": "Research"}]}'
        items = _extract_items(raw, "steps")
        assert len(items) == 1
        assert items[0]["id"] == "step-1"

    def test_from_codeblock(self) -> None:
        """_extract_items recovers items from markdown code block."""
        raw = 'Here is the plan:\n```json\n{"steps": [{"id": "s1"}]}\n```'
        items = _extract_items(raw, "steps")
        assert len(items) == 1

    def test_from_plain_text(self) -> None:
        """_extract_items returns [] for non-JSON string."""
        items = _extract_items("This is a research plan about Tokyo", "steps")
        assert items == []

    def test_empty_steps_list(self) -> None:
        """_extract_items returns [] for empty steps list."""
        items = _extract_items({"steps": []}, "steps")
        assert items == []

    def test_from_object_with_attr(self) -> None:
        """_extract_items navigates object attributes."""
        class Plan:
            steps = [{"id": "s1"}, {"id": "s2"}]
        items = _extract_items(Plan(), "steps")
        assert len(items) == 2


# ---------------------------------------------------------------------------
# Fix 2: Parallel node uses merged queue (no busy-wait)
# ---------------------------------------------------------------------------


class TestParallelNodeMergedQueue:
    @pytest.mark.asyncio
    async def test_no_busy_wait(self) -> None:
        """Parallel node collects events without asyncio.sleep polling."""
        import asyncio

        executed: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            await asyncio.sleep(0.01)  # simulate work
            executed.append(node_id)
            return AgentOutput(
                content=f"result-{node_id}",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="ok",
                    )
                ],
            )

        root = WorkflowNode(
            id="par",
            type=NodeType.parallel,
            label="parallel",
            children=[
                WorkflowNode(
                    id=f"child_{i}",
                    type=NodeType.agent,
                    label=f"C{i}",
                    config={"subtype": "researcher", "output_key": f"r{i}"},
                )
                for i in range(5)
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            await _collect_events(executor, state)

        assert set(executed) == {f"child_{i}" for i in range(5)}


# ---------------------------------------------------------------------------
# Fix 4: Reflector observation cap
# ---------------------------------------------------------------------------


class TestReflectorObservationCap:
    def test_cap_at_max_observations(self) -> None:
        """_format_all_observations caps at _REFLECTOR_MAX_OBSERVATIONS."""
        from databricks_deep_research.pools.pool_state import PoolConfig, PoolState

        pool = PoolState(PoolConfig(name="observations", dedup_content_hash=False))
        for i in range(50):
            pool.add(f"observation about topic {i}")

        result = _format_all_observations({"observations": pool})

        # Should mention older observations were omitted
        assert "omitted" in result
        # Should be under the char cap
        assert len(result) <= 2100  # allow small overshoot from final line

    def test_empty_pool(self) -> None:
        """Empty pool returns placeholder text."""
        result = _format_all_observations({})
        assert result == "(no observations yet)"

    def test_source_quality_format(self) -> None:
        """_format_source_quality returns readable metrics."""
        from databricks_deep_research.pools.pool_state import PoolConfig, PoolState

        pool = PoolState(PoolConfig(name="sources", dedup_content_hash=False))
        pool.add({"url": "https://a.com", "title": "A", "snippet": "some text", "admission_status": "accepted", "evidence_quality": "full_text"})
        pool.add({"url": "https://b.com", "title": "B", "admission_status": "accepted_low_value", "evidence_quality": "metadata_only"})
        pool.add({"url": "https://a.com/other", "title": "C", "snippet": "more", "admission_status": "accepted", "evidence_quality": "full_text"})

        result = _format_source_quality({"sources": pool})

        assert "3 sources" in result
        assert "2 domains" in result
        assert "2/3 with evidence" in result
        assert "substantive=2" in result
        assert "low_value=1" in result


# ---------------------------------------------------------------------------
# Fix 6: Always-evaluate
# ---------------------------------------------------------------------------


class TestAlwaysEvaluate:
    @pytest.mark.asyncio
    async def test_evaluator_runs_before_min_iterations(self) -> None:
        """Evaluator runs on step 0 and under-min exhaustion triggers a degraded exit."""
        evaluation_calls: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            config = kwargs.get("config")
            # Detect evaluator vs researcher by subtype
            if config and config.subtype == "reflector":
                evaluation_calls.append(node_id)
                return AgentOutput(
                    content={"decision": "complete", "reasoning": "enough data"},
                    output_key="evaluation",
                    events=[],
                )
            elif config and config.subtype == "planner":
                return AgentOutput(
                    content={"steps": [{"id": "step-1", "title": "Research A"}]},
                    output_key="plan",
                    events=[],
                )
            else:
                return AgentOutput(
                    content="findings here",
                    output_key="findings",
                    events=[],
                )

        from databricks_deep_research.events.types import (
            EvaluationDecisionEvent,
            PlanAndExecuteExitEvent,
        )

        root = WorkflowNode(
            id="pe",
            type=NodeType.plan_and_execute,
            label="Research",
            config={
                "planner": {"subtype": "planner", "output_key": "plan"},
                "items_path": "steps",
                "item_state_key": "current_step",
                "body": {
                    "id": "researcher",
                    "type": "agent",
                    "label": "Researcher",
                    "config": {"subtype": "researcher", "output_key": "findings"},
                },
                "evaluator": {"subtype": "reflector", "output_key": "evaluation"},
                "max_iterations": 5,
                "min_iterations": 3,  # min is 3 but only 1 step
                "max_replan_cycles": 0,
            },
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        # Evaluator SHOULD have been called (always-evaluate)
        assert len(evaluation_calls) >= 1

        # Check that evaluation decision events exist
        eval_decisions = _events_of_type(events, EvaluationDecisionEvent)
        assert len(eval_decisions) >= 1

        # The "complete" decision should have been gated to "continue"
        # since we're before min_iterations
        exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(exit_events) == 1
        assert exit_events[0].reason == "min_iterations_unmet"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fix 11: Synthesis state population
# ---------------------------------------------------------------------------


class TestSynthesisState:
    def test_populate_synthesis_state(self) -> None:
        """_populate_synthesis_state writes expected keys."""
        from databricks_deep_research.pools.pool_state import PoolConfig, PoolState

        state = WorkflowState(query="test")
        sources = PoolState(PoolConfig(name="sources", dedup_content_hash=False))
        sources.add({"url": "http://a.com", "title": "A"})
        observations = PoolState(PoolConfig(name="observations", dedup_content_hash=False))
        observations.add("observation 1")

        _populate_synthesis_state(
            "pe_node", state,
            {"sources": sources, "observations": observations},
            total_items_processed=3,
            replan_cycles=1,
        )

        assert state.get("steps_executed") == "3"
        assert state.get("plan_iterations") == "2"
        assert state.get("sources_count") == "1"
        assert "observation 1" in state.get("all_observations")
        assert "[A](http://a.com)" in state.get("sources_list")


@pytest.mark.asyncio
async def test_plan_and_execute_treats_adjust_as_replan() -> None:
    evaluation_calls = 0

    async def fake_execute_agent(
        node_id: str, **kwargs: Any
    ) -> AgentOutput:
        nonlocal evaluation_calls
        config = kwargs.get("config")
        if config and config.subtype == "planner":
            return AgentOutput(
                content={"steps": [{"id": "step-1", "title": "Research A"}]},
                output_key="plan",
                events=[],
            )
        if config and config.subtype == "reflector":
            evaluation_calls += 1
            return AgentOutput(
                content={"decision": "adjust", "reasoning": "Need a different plan"},
                output_key="evaluation",
                events=[],
            )
        return AgentOutput(
            content="finding",
            output_key="findings",
            events=[],
        )

    root = WorkflowNode(
        id="pe",
        type=NodeType.plan_and_execute,
        label="Research",
        config={
            "planner": {"subtype": "planner", "output_key": "plan"},
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": {
                "id": "researcher",
                "type": "agent",
                "label": "Researcher",
                "config": {"subtype": "researcher", "output_key": "findings"},
            },
            "evaluator": {"subtype": "reflector", "output_key": "evaluation"},
            "max_iterations": 2,
            "min_iterations": 1,
            "max_replan_cycles": 1,
        },
    )
    executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())
    state = WorkflowState(query="test")

    with patch(
        "databricks_deep_research.workflow.executor.execute_agent",
        side_effect=fake_execute_agent,
    ):
        events = await _collect_events(executor, state)

    decisions = [
        event for event in events
        if event.__class__.__name__ == "EvaluationDecisionEvent"
    ]
    replans = _events_of_type(events, ReplanTriggeredEvent)

    assert evaluation_calls >= 1
    assert any(getattr(event, "decision", None) == "replan" for event in decisions)
    assert replans


@pytest.mark.asyncio
async def test_plan_and_execute_empty_plan_replans_and_fails_when_no_progress() -> None:
    planner_calls = 0

    async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        nonlocal planner_calls
        config = kwargs.get("config")
        if config and config.subtype == "planner":
            planner_calls += 1
            return AgentOutput(
                content={"title": "Empty", "steps": [], "has_enough_context": False},
                output_key="plan",
                events=[],
            )
        return AgentOutput(content="finding", output_key="findings", events=[])

    root = WorkflowNode(
        id="pe",
        type=NodeType.plan_and_execute,
        label="Research",
        config={
            "planner": {"subtype": "planner", "output_key": "plan"},
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": {
                "id": "researcher",
                "type": "agent",
                "label": "Researcher",
                "config": {"subtype": "researcher", "output_key": "findings"},
            },
            "max_iterations": 2,
            "min_iterations": 1,
            "max_replan_cycles": 1,
        },
    )
    executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())
    state = WorkflowState(query="test")

    with patch(
        "databricks_deep_research.workflow.executor.execute_agent",
        side_effect=fake_execute_agent,
    ), pytest.raises(PlanningContractError, match="zero executable steps"):
        await _collect_events(executor, state)

    assert planner_calls == 2


@pytest.mark.asyncio
async def test_plan_and_execute_under_min_iterations_triggers_replan_then_exit() -> None:
    planner_calls = 0

    async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        nonlocal planner_calls
        config = kwargs.get("config")
        if config and config.subtype == "planner":
            planner_calls += 1
            if planner_calls == 1:
                return AgentOutput(
                    content={"title": "Plan A", "steps": [{"id": "step-1", "title": "Research A", "needs_search": False}]},
                    output_key="plan",
                    events=[],
                )
            return AgentOutput(
                content={"title": "Plan B", "steps": [{"id": "step-2", "title": "Research B", "needs_search": False}]},
                output_key="plan",
                events=[],
            )
        return AgentOutput(content="finding", output_key="findings", events=[])

    from databricks_deep_research.events.types import PlanAndExecuteExitEvent

    root = WorkflowNode(
        id="pe",
        type=NodeType.plan_and_execute,
        label="Research",
        config={
            "planner": {"subtype": "planner", "output_key": "plan"},
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": {
                "id": "researcher",
                "type": "agent",
                "label": "Researcher",
                "config": {"subtype": "researcher", "output_key": "findings"},
            },
            "max_iterations": 4,
            "min_iterations": 3,
            "max_replan_cycles": 1,
        },
    )
    executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())
    state = WorkflowState(query="test")

    with patch(
        "databricks_deep_research.workflow.executor.execute_agent",
        side_effect=fake_execute_agent,
    ):
        events = await _collect_events(executor, state)

    replans = _events_of_type(events, ReplanTriggeredEvent)
    exit_events = _events_of_type(events, PlanAndExecuteExitEvent)

    assert planner_calls == 2
    assert any(getattr(event, "reason", None) == "min_iterations_unmet" for event in replans)
    assert exit_events[-1].reason == "min_iterations_unmet"  # type: ignore[attr-defined]


def test_planner_runtime_context_covers_default_prompt_variables() -> None:
    from databricks_deep_research.agents.prompts.planner import PLANNER_USER_PROMPT

    renderer = SafeTemplateRenderer()
    variables = renderer.extract_variables(PLANNER_USER_PROMPT)
    config = PlanAndExecuteNodeConfig(planner={"subtype": "planner"})
    state = WorkflowState(query="test")
    state.append("background", "background", {"summary": "context"})
    state.append("upload", "file_context", "file contents")
    cycle_ctx = PlanCycleContext()
    context = _build_planner_runtime_context(
        config=config,
        state=state,
        pools={},
        cycle_ctx=cycle_ctx,
    )

    assert variables <= {"query"} | set(context.keys()) | set(state._latest_index.keys())


def test_evaluator_runtime_context_covers_default_prompt_variables() -> None:
    from databricks_deep_research.agents.prompts.reflector import REFLECTOR_USER_PROMPT

    renderer = SafeTemplateRenderer()
    variables = renderer.extract_variables(REFLECTOR_USER_PROMPT)
    config = PlanAndExecuteNodeConfig(planner={"subtype": "planner"})
    state = WorkflowState(query="test")
    state.append("body", "plan", {"steps": [{"id": "step-1", "title": "A"}]})
    state.append("body", "findings", "obs")
    context = _build_evaluator_runtime_context(
        config=config,
        state=state,
        pools={},
        items=[{"id": "step-1", "title": "A"}],
        current_idx=0,
        current_item={"id": "step-1", "title": "A"},
        cycle=0,
        total_items_processed=1,
    )

    assert variables <= {"query"} | set(context.keys())


@pytest.mark.asyncio
async def test_available_source_catalog_uses_body_tools_and_excludes_helpers() -> None:
    definition = WorkflowDefinition(
        id="wf",
        name="wf",
        tools=[
            ToolDeclaration(name="web_search", kind="web_search", description="Public web search"),
            ToolDeclaration(name="web_crawl", kind="web_crawl", description="Crawler helper"),
            ToolDeclaration(name="vector_search", kind="vector_search", description="Internal docs"),
        ],
        sources=[
            SourceDefinition(name="web_search", kind="web", description="Public web search"),
            SourceDefinition(name="vector_search", kind="vector_index", description="Internal docs"),
        ],
        root=WorkflowNode(
            id="root",
            type=NodeType.plan_and_execute,
            label="Plan",
            config={
                "planner": {"subtype": "planner"},
                "body": {
                    "id": "researcher",
                    "type": "agent",
                    "label": "Researcher",
                    "config": {
                        "subtype": "researcher",
                        "tools": ["web_search", "web_crawl", "vector_search"],
                    },
                },
            },
        ),
    )

    catalog = await _build_available_source_catalog(
        definition,
        ToolResolver(declarations=list(definition.tools)),
        definition.root.config["body"],
    )

    assert [item.source_name for item in catalog] == ["web_search", "vector_search"]


@pytest.mark.asyncio
async def test_available_source_catalog_includes_attached_mcp_tools() -> None:
    class _FakeMCPTool:
        definition = ToolDefinition(
            name="tavily_search",
            description="Search the live web through Tavily MCP.",
            parameters={"type": "object", "properties": {}},
            source_type="mcp",
            source_kind="qa_assistant",
            metadata={
                "source_name": "tavily_mcp",
                "source_url": "mcp://tavily_mcp/tavily_search",
            },
        )

    definition = WorkflowDefinition(
        id="wf",
        name="wf",
        root=WorkflowNode(
            id="root",
            type=NodeType.plan_and_execute,
            label="Plan",
            config={
                "planner": {"subtype": "planner"},
                "body": {
                    "id": "researcher",
                    "type": "agent",
                    "label": "Researcher",
                    "config": {
                        "subtype": "researcher",
                        "tools": [],
                        "mcp_servers": ["tavily_mcp"],
                    },
                },
            },
        ),
    )
    resolver = ToolResolver(
        factory_context=ToolFactoryContext(
            extras={"_mcp_tools_by_server": {"tavily_mcp": [_FakeMCPTool()]}}
        )
    )

    catalog = await _build_available_source_catalog(
        definition,
        resolver,
        definition.root.config["body"],
    )

    assert [(item.source_name, item.tool_kind, item.source_kind) for item in catalog] == [
        ("tavily_search", "mcp", "qa_assistant")
    ]


@pytest.mark.asyncio
async def test_agent_compute_namespace_gets_table_and_vector_providers() -> None:
    from types import SimpleNamespace

    from databricks_deep_research.tools.builtins.compute import PythonComputeTool
    from databricks_deep_research.tools.builtins.text_table import (
        BindingInfo,
        BindingSource,
        RoleMap,
        Schema,
        SchemaColumn,
        TableBindingRegistry,
    )
    from databricks_deep_research.tools.factory import ToolFactoryContext
    from databricks_deep_research.tools.protocol import ToolContext

    class _SchemaCache:
        def get(self, fqn: str, user_token: str) -> Schema:
            return Schema(
                fqn=fqn,
                columns=(
                    SchemaColumn(name="chunk_id", data_type="string"),
                    SchemaColumn(name="content", data_type="string"),
                ),
            )

    class _VectorIndexes:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def query_index(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(
                manifest=SimpleNamespace(
                    columns=[
                        SimpleNamespace(name="id"),
                        SimpleNamespace(name="content"),
                        SimpleNamespace(name="score"),
                    ],
                ),
                result=SimpleNamespace(
                    data_array=[["v1", "vector hit", 0.98]],
                ),
            )

    registry = TableBindingRegistry()
    registry.register_bound(
        BindingInfo(
            name="docs",
            fqn="cat.sch.docs",
            source=BindingSource.BOUND,
            roles=RoleMap(id_column="chunk_id", content_column="content"),
        )
    )

    def sql_executor(
        sql: str, params: list[dict[str, Any]], user_token: str
    ) -> list[dict[str, Any]]:
        assert "LIKE" in sql
        assert params
        assert user_token == ""
        return [{"chunk_id": "r1", "content": "hello table hit"}]

    vector_indexes = _VectorIndexes()
    workspace_client = SimpleNamespace(vector_search_indexes=vector_indexes)
    definition = WorkflowDefinition(
        id="wf",
        name="wf",
        tools=[
            ToolDeclaration(name="compute", kind="compute", config={}),
            ToolDeclaration(name="table_search", kind="table_search", config={}),
            ToolDeclaration(
                name="vs",
                kind="vector_search",
                config={
                    "index_name": "cat.sch.docs_vs",
                    "columns": ["id", "content", "score"],
                    "num_results": 1,
                },
            ),
        ],
        root=WorkflowNode(
            id="agent",
            type=NodeType.agent,
            label="Agent",
            config={
                "subtype": "researcher",
                "output_key": "answer",
                "tools": ["compute", "table_search", "vs"],
            },
        ),
    )

    async def fake_execute_agent(
        node_id: str, tools: list[Any], **kwargs: Any
    ) -> AgentOutput:
        config = kwargs["config"]
        assert "## Available text tables" in config.system_prompt
        assert "binding: docs" in config.system_prompt
        compute = next(tool for tool in tools if isinstance(tool, PythonComputeTool))
        first = await compute.execute(
            {
                "code": "\n".join(
                    [
                        "print('docs' in bindings)",
                        "print(vector_indexes['vs']['index_name'])",
                        "print(table_search(binding='docs', query='hello')[0]['id'])",
                        "print(vector_search('hello', num_results=1)[0]['content'])",
                    ]
                )
            },
            ToolContext(),
        )
        assert first.success is True
        assert "True" in first.content
        assert "cat.sch.docs_vs" in first.content
        assert "r1" in first.content
        assert "vector hit" in first.content

        registry.register_discovered(
            BindingInfo(
                name="late",
                fqn="cat.sch.late",
                source=BindingSource.DISCOVERED,
            )
        )
        second = await compute.execute(
            {"code": "print('late' in bindings)"},
            ToolContext(),
        )
        assert second.success is True
        assert "True" in second.content

        return AgentOutput(
            content="ok",
            output_key="answer",
            events=[
                AgentOutputEvent(
                    node_id=node_id,
                    timestamp="T",
                    output_key="answer",
                    output_preview="ok",
                )
            ],
        )

    executor = WorkflowExecutor(
        definition,
        _mock_llm_client(),
        factory_context=ToolFactoryContext(
            workspace_client=workspace_client,
            table_registry=registry,
            schema_cache=_SchemaCache(),
            sql_executor=sql_executor,
        ),
    )

    with patch(
        "databricks_deep_research.workflow.executor.execute_agent",
        side_effect=fake_execute_agent,
    ):
        events = await _collect_events(executor, WorkflowState(query="test"))

    assert isinstance(events[-1], WorkflowCompletedEvent)
    assert vector_indexes.calls[0]["index_name"] == "cat.sch.docs_vs"


def test_planner_event_uses_normalized_executable_steps() -> None:
    from databricks_deep_research.agents.builtins.planner import _post_process
    from databricks_deep_research.agents.output_models import PlanOutput

    output = PlanOutput(
        title="Environmental Impacts of Lithium Mining for EV Batteries",
        thought="The query has strong discovered evidence and should be investigated in a focused step.",
        steps=[],
        has_enough_context=False,
        iteration=1,
    )
    events = _post_process(
        "planner",
        output,
        AgentNodeConfig(subtype="planner", output_key="plan"),
        WorkflowState(query="test"),
    )

    assert len(events) == 1
    assert isinstance(events[0], PlanCreatedEvent)
    assert len(events[0].steps) == 1
    assert events[0].steps[0]["title"] == "Environmental Impacts of Lithium Mining for EV Batteries"



class TestExecutablePlanNormalization:
    def test_normalize_executable_plan_contract_synthesizes_step(self) -> None:
        contract = _normalize_executable_plan_contract({"title": "Research Plan", "thought": "Investigate the topic", "steps": []}, "steps")
        assert len(contract["items"]) == 1
        step = contract["items"][0]
        assert step["title"] == "Research Plan"
        assert contract["repair_mode"] == "synthesized_from_empty_steps"


class TestTypedPlanContract:
    def test_extract_raw_plan_contract(self) -> None:
        contract = _extract_raw_plan_contract({"title": "Plan", "thought": "Think", "steps": [{}, {"title": "Step"}]}, "steps")
        assert isinstance(contract, NormalizedPlanContract)
        assert contract.title == "Plan"
        assert len(contract.raw_items) == 2

    def test_finalize_plan_contract_filters_empty_items(self) -> None:
        raw = _extract_raw_plan_contract({"title": "Plan", "thought": "Think", "steps": [{}, {"title": "Step"}]}, "steps")
        finalized = _finalize_plan_contract(raw, {"title": "Plan", "thought": "Think", "steps": [{}, {"title": "Step"}]})
        assert len(finalized.items) == 1
