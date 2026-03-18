"""Tests for cancellation behavior and token tracking across workflow execution."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.errors import NodeBudgetExceededError
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    NodeBudgetExceededEvent,
    NodeSkippedEvent,
    NodeStartedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
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


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_executor.py)
# ---------------------------------------------------------------------------
def _make_definition(root: WorkflowNode) -> WorkflowDefinition:
    """Wrap a root node into a minimal WorkflowDefinition."""
    return WorkflowDefinition(
        id="test-wf",
        name="Test Workflow",
        root=root,
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCancellationAndBudget:
    """Tests for cancellation semantics and token usage tracking."""

    # -- 1. Cancel before execution ----------------------------------------

    @pytest.mark.asyncio
    async def test_cancel_before_execution(self) -> None:
        """Setting is_cancelled before execute emits start/complete but no agents."""
        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="agent_a",
                    type=NodeType.agent,
                    label="Agent A",
                    config={"subtype": "researcher", "output_key": "out_a"},
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="Agent B",
                    config={"subtype": "researcher", "output_key": "out_b"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")
        state.is_cancelled = True

        events = await _collect_events(executor, state)

        # No agent node should have started
        started = _events_of_type(events, NodeStartedEvent)
        assert len(started) == 0

        # Workflow bookend events must still be emitted
        assert any(isinstance(e, WorkflowStartedEvent) for e in events)
        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)

    # -- 2. Cancel mid-sequence --------------------------------------------

    @pytest.mark.asyncio
    async def test_cancel_mid_sequence(self) -> None:
        """Cancellation after the 2nd agent in a 3-agent sequence stops execution."""
        execution_order: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            execution_order.append(node_id)
            st = kwargs.get("state")
            if st and len(execution_order) == 2:
                st.is_cancelled = True
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
                    label="A",
                    config={"subtype": "researcher", "output_key": "out_a"},
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="B",
                    config={"subtype": "researcher", "output_key": "out_b"},
                ),
                WorkflowNode(
                    id="agent_c",
                    type=NodeType.agent,
                    label="C",
                    config={"subtype": "researcher", "output_key": "out_c"},
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

        # Only the first 2 agents should have run
        assert execution_order == ["agent_a", "agent_b"]
        # Workflow should complete (cancellation is caught gracefully)
        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)

    # -- 3. Cancel mid plan-and-execute ------------------------------------

    @pytest.mark.asyncio
    async def test_cancel_mid_plan_and_execute(self) -> None:
        """Cancellation during plan-and-execute stops after the cancelled step."""
        execution_order: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            execution_order.append(node_id)
            st = kwargs.get("state")

            # Planner: return a plan with 4 steps
            if "planner" in node_id:
                return AgentOutput(
                    content={
                        "title": "Test plan",
                        "thought": "planning",
                        "steps": [
                            {"title": "Step 1", "query": "s1"},
                            {"title": "Step 2", "query": "s2"},
                            {"title": "Step 3", "query": "s3"},
                            {"title": "Step 4", "query": "s4"},
                        ],
                    },
                    output_key="plan",
                    events=[],
                    token_usage={"prompt_tokens": 50, "completion_tokens": 50},
                )

            # Evaluator: always continue
            if "eval" in node_id:
                return AgentOutput(
                    content={"decision": "continue", "reasoning": "keep going"},
                    output_key="evaluation",
                    events=[],
                    token_usage={"prompt_tokens": 20, "completion_tokens": 10},
                )

            # Body researcher: cancel on 2nd step
            body_count = sum(1 for nid in execution_order if "body" in nid)
            if st and body_count >= 2:
                st.is_cancelled = True
            return AgentOutput(
                content=f"findings-{node_id}",
                output_key="findings",
                events=[],
                token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            )

        root = WorkflowNode(
            id="pae",
            type=NodeType.plan_and_execute,
            label="plan-and-execute",
            config={
                "planner": {
                    "subtype": "planner",
                    "output_key": "plan",
                    "output_mode": "json",
                },
                "items_path": "steps",
                "item_state_key": "current_step",
                "body": {
                    "id": "body",
                    "type": "agent",
                    "label": "Researcher",
                    "config": {
                        "subtype": "researcher",
                        "output_key": "findings",
                    },
                },
                "evaluator": {
                    "subtype": "evaluator",
                    "output_key": "evaluation",
                    "output_mode": "json",
                },
                "max_iterations": 10,
                "min_iterations": 1,
                "max_replan_cycles": 1,
                "complete_on_exhaustion": True,
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

        # Steps 3 and 4 should NOT have run
        body_executions = [nid for nid in execution_order if "body" in nid]
        assert len(body_executions) <= 2

        # Workflow completes gracefully
        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)

    # -- 4. Token tracking across agents -----------------------------------


    @pytest.mark.asyncio
    async def test_workflow_completed_reports_total_steps_executed(self) -> None:
        async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            if "planner" in node_id:
                return AgentOutput(
                    content={
                        "title": "Plan",
                        "thought": "Do one step",
                        "steps": [{"title": "Step 1", "needs_search": False}],
                    },
                    output_key="plan",
                    events=[],
                )
            if "eval" in node_id:
                return AgentOutput(
                    content={"decision": "complete", "reasoning": "done"},
                    output_key="evaluation",
                    events=[],
                )
            return AgentOutput(content="finding", output_key="findings", events=[])

        root = WorkflowNode(
            id="pae",
            type=NodeType.plan_and_execute,
            label="plan-and-execute",
            config={
                "planner": {"subtype": "planner", "output_key": "plan", "output_mode": "json"},
                "items_path": "steps",
                "item_state_key": "current_step",
                "body": {
                    "id": "body",
                    "type": "agent",
                    "label": "Researcher",
                    "config": {"subtype": "researcher", "output_key": "findings"},
                },
                "evaluator": {"subtype": "evaluator", "output_key": "evaluation", "output_mode": "json"},
                "max_iterations": 5,
                "min_iterations": 1,
                "max_replan_cycles": 0,
                "complete_on_exhaustion": True,
            },
        )
        executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())
        with patch("databricks_deep_research.workflow.executor.execute_agent", side_effect=fake_execute_agent):
            events = await _collect_events(executor, WorkflowState(query="test"))

        completed = _events_of_type(events, WorkflowCompletedEvent)
        assert len(completed) == 1
        assert completed[0].total_steps_executed == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_token_tracking_across_agents(self) -> None:
        """Token usage from 3 sequential agents sums correctly."""
        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
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
                token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            )

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="agent_a",
                    type=NodeType.agent,
                    label="A",
                    config={"subtype": "researcher", "output_key": "out_a"},
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="B",
                    config={"subtype": "researcher", "output_key": "out_b"},
                ),
                WorkflowNode(
                    id="agent_c",
                    type=NodeType.agent,
                    label="C",
                    config={"subtype": "researcher", "output_key": "out_c"},
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

        completed = _events_of_type(events, WorkflowCompletedEvent)
        assert len(completed) == 1
        assert completed[0].total_tokens == 450  # 3 * (100 + 50)  # type: ignore[attr-defined]

    # -- 5. Token tracking in plan-and-execute -----------------------------

    @pytest.mark.asyncio
    async def test_token_tracking_in_plan_and_execute(self) -> None:
        """Body steps in plan_and_execute accumulate tokens via _exec_node.

        Note: the planner and evaluator call execute_agent directly (not
        through _exec_node → _exec_agent), so only body-node tokens are
        tracked by the executor's _total_tokens accumulator.  With 2 body
        steps each contributing 300 tokens, the total is 600.
        """
        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            # Planner
            if "planner" in node_id:
                return AgentOutput(
                    content={
                        "title": "Plan",
                        "thought": "thinking",
                        "steps": [
                            {"title": "Step 1", "needs_search": False},
                            {"title": "Step 2", "needs_search": False},
                        ],
                    },
                    output_key="plan",
                    events=[],
                    token_usage={"prompt_tokens": 200, "completion_tokens": 100},
                )

            # Evaluator
            if "eval" in node_id:
                return AgentOutput(
                    content={"decision": "continue", "reasoning": "go on"},
                    output_key="evaluation",
                    events=[],
                    token_usage={"prompt_tokens": 200, "completion_tokens": 100},
                )

            # Body researcher
            return AgentOutput(
                content="findings",
                output_key="findings",
                events=[],
                token_usage={"prompt_tokens": 200, "completion_tokens": 100},
            )

        root = WorkflowNode(
            id="pae",
            type=NodeType.plan_and_execute,
            label="plan-and-execute",
            config={
                "planner": {
                    "subtype": "planner",
                    "output_key": "plan",
                    "output_mode": "json",
                },
                "items_path": "steps",
                "item_state_key": "current_step",
                "body": {
                    "id": "body",
                    "type": "agent",
                    "label": "Researcher",
                    "config": {
                        "subtype": "researcher",
                        "output_key": "findings",
                    },
                },
                "evaluator": {
                    "subtype": "evaluator",
                    "output_key": "evaluation",
                    "output_mode": "json",
                },
                "max_iterations": 10,
                "min_iterations": 1,
                "max_replan_cycles": 0,
                "complete_on_exhaustion": True,
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

        completed = _events_of_type(events, WorkflowCompletedEvent)
        assert len(completed) == 1
        # Only body nodes go through _exec_agent which tracks tokens.
        # 2 body steps * (200 + 100) = 600.
        total = completed[0].total_tokens  # type: ignore[attr-defined]
        assert total == 600

    # -- 6. Cancel flag checked per node -----------------------------------

    @pytest.mark.asyncio
    async def test_cancel_flag_checked_per_node(self) -> None:
        """Cancel after 3rd agent in a 5-agent sequence: exactly 3 executed."""
        execution_order: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            execution_order.append(node_id)
            st = kwargs.get("state")
            if st and len(execution_order) == 3:
                st.is_cancelled = True
            return AgentOutput(
                content="ok",
                output_key="output",
                events=[],
            )

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id=f"agent_{i}",
                    type=NodeType.agent,
                    label=f"Agent {i}",
                    config={"subtype": "researcher", "output_key": f"out_{i}"},
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
            events = await _collect_events(executor, state)

        assert len(execution_order) == 3
        assert execution_order == ["agent_0", "agent_1", "agent_2"]
        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)

    # -- 7. Cancel during parallel -----------------------------------------

    @pytest.mark.asyncio
    async def test_cancel_during_parallel(self) -> None:
        """Cancellation during parallel execution does not crash."""
        first_done = asyncio.Event()

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            st = kwargs.get("state")
            if node_id == "child_0":
                # First child completes quickly and sets cancel
                if st:
                    st.is_cancelled = True
                first_done.set()
                return AgentOutput(
                    content="fast", output_key="output", events=[],
                )
            # Other children yield to let cancellation propagate
            await asyncio.sleep(0.05)
            return AgentOutput(
                content="slow", output_key="output", events=[],
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
                for i in range(3)
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            # Must not raise
            events = await _collect_events(executor, state)

        # Workflow should still emit completion bookend
        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)

    # -- 8. WorkflowCompletedEvent carries total_tokens --------------------

    @pytest.mark.asyncio
    async def test_workflow_completed_event_has_total_tokens(self) -> None:
        """A single agent's token usage appears in the final completed event."""
        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            return AgentOutput(
                content="analysis",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="analysis",
                    )
                ],
                token_usage={"prompt_tokens": 500, "completion_tokens": 250},
            )

        root = WorkflowNode(
            id="single",
            type=NodeType.agent,
            label="Single Agent",
            config={"subtype": "researcher", "output_key": "output"},
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        completed = _events_of_type(events, WorkflowCompletedEvent)
        assert len(completed) == 1
        assert completed[0].total_tokens == 750  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_token_tracking_prefers_total_tokens_field(self) -> None:
        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            return AgentOutput(
                content="analysis",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="analysis",
                    )
                ],
                token_usage={
                    "prompt_tokens": 500,
                    "completion_tokens": 250,
                    "total_tokens": 750,
                },
            )

        root = WorkflowNode(
            id="single",
            type=NodeType.agent,
            label="Single Agent",
            config={"subtype": "researcher", "output_key": "output"},
        )
        executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, WorkflowState(query="test"))

        completed = _events_of_type(events, WorkflowCompletedEvent)
        assert completed[0].total_tokens == 750  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_node_budget_exceeded_fails_node(self) -> None:
        async def slow_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            await asyncio.sleep(0.05)
            return AgentOutput(content=node_id, output_key="output", events=[])

        root = WorkflowNode(
            id="slow",
            type=NodeType.agent,
            label="Slow Agent",
            budget_seconds=0.01,
            config={"subtype": "researcher", "output_key": "output"},
        )
        executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=slow_execute_agent,
        ), pytest.raises(NodeBudgetExceededError):
            await _collect_events(executor, WorkflowState(query="test"))

    @pytest.mark.asyncio
    async def test_node_budget_exceeded_can_skip(self) -> None:
        async def slow_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            await asyncio.sleep(0.05)
            return AgentOutput(content=node_id, output_key="output", events=[])

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="Sequence",
            children=[
                WorkflowNode(
                    id="slow",
                    type=NodeType.agent,
                    label="Slow Agent",
                    budget_seconds=0.01,
                    config={"subtype": "researcher", "output_key": "slow_out"},
                    error_handling={"on_error": "skip"},
                ),
                WorkflowNode(
                    id="fast",
                    type=NodeType.agent,
                    label="Fast Agent",
                    config={"subtype": "researcher", "output_key": "fast_out"},
                ),
            ],
        )
        executor = WorkflowExecutor(_make_definition(root), _mock_llm_client())

        async def side_effect(node_id: str, **kwargs: Any) -> AgentOutput:
            if node_id == "slow":
                await asyncio.sleep(0.05)
            return AgentOutput(content=node_id, output_key="output", events=[])

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=side_effect,
        ):
            events = await _collect_events(executor, WorkflowState(query="test"))

        assert _events_of_type(events, NodeBudgetExceededEvent)
        assert _events_of_type(events, NodeSkippedEvent)
        assert any(isinstance(event, WorkflowCompletedEvent) for event in events)
