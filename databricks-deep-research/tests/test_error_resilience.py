"""Tests for error resilience across multi-node workflows with mocked LLM clients.

Tier 2: fast, fully mocked, no credentials needed, <30s total.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    LoopExitEvent,
    LoopIterationEvent,
    NodeErrorEvent,
    NodeSkippedEvent,
    NodeStartedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from databricks_deep_research.workflow.conditions import StateCondition
from databricks_deep_research.workflow.definition import (
    ErrorConfig,
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState
from tests.conftest import (
    build_mock_llm_client as _mock_llm_client,
    collect_events as _collect_events,
    events_of_type as _events_of_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_TARGET = "databricks_deep_research.workflow.executor.execute_agent"
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


class TestErrorResilience:
    """Error handling tests across multi-node workflows."""

    # 1. Skip failing researcher in a sequence
    @pytest.mark.asyncio
    async def test_skip_failing_researcher_in_sequence(self) -> None:
        """Sequence of 3 agents: middle one fails with on_error=skip.

        First and third should complete; middle should be skipped.
        """
        execution_order: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            if node_id == "agent_b":
                raise RuntimeError("agent_b exploded")
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
                    config={"subtype": "researcher", "output_key": "out_a"},
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="Agent B",
                    config={"subtype": "researcher", "output_key": "out_b"},
                    error_handling=ErrorConfig(on_error="skip"),
                ),
                WorkflowNode(
                    id="agent_c",
                    type=NodeType.agent,
                    label="Agent C",
                    config={"subtype": "researcher", "output_key": "out_c"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            events = await _collect_events(executor, state)

        # First and third ran; middle was skipped
        assert execution_order == ["agent_a", "agent_c"]

        skipped = _events_of_type(events, NodeSkippedEvent)
        assert len(skipped) == 1
        assert "agent_b exploded" in skipped[0].reason  # type: ignore[attr-defined]

        assert isinstance(events[-1], WorkflowCompletedEvent)

    # 2. Retry succeeds on second attempt
    @pytest.mark.asyncio
    async def test_retry_succeeds_second_attempt(self) -> None:
        """Agent fails first call, succeeds on retry (max_retries=2)."""
        call_count = 0

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            return AgentOutput(
                content="recovered",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="recovered",
                    )
                ],
            )

        root = WorkflowNode(
            id="retryable",
            type=NodeType.agent,
            label="Retryable Agent",
            config={"subtype": "researcher", "output_key": "out"},
            error_handling=ErrorConfig(
                on_error="retry", max_retries=2, retry_delay_seconds=0.01
            ),
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            events = await _collect_events(executor, state)

        # Should see a NodeErrorEvent with will_retry=True
        error_events = _events_of_type(events, NodeErrorEvent)
        assert len(error_events) >= 1
        assert any(
            e.will_retry for e in error_events  # type: ignore[attr-defined]
        )

        # Workflow should complete successfully
        assert isinstance(events[-1], WorkflowCompletedEvent)

    # 3. Retry exhaustion raises
    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises(self) -> None:
        """Agent always fails; on_error=retry, max_retries=2. Should raise."""

        async def always_fail(node_id: str, **kwargs: Any) -> AgentOutput:
            raise RuntimeError("permanent failure")

        root = WorkflowNode(
            id="doomed",
            type=NodeType.agent,
            label="Doomed Agent",
            config={"subtype": "researcher", "output_key": "out"},
            error_handling=ErrorConfig(
                on_error="retry", max_retries=2, retry_delay_seconds=0.01
            ),
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=always_fail):
            with pytest.raises(RuntimeError, match="permanent failure"):
                await _collect_events(executor, state)

    # 4. Skip in parallel node
    @pytest.mark.asyncio
    async def test_skip_in_parallel_node(self) -> None:
        """3 parallel children, 1 fails with on_error=skip. 2 complete, 1 skipped."""
        executed: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            if node_id == "par_b":
                raise RuntimeError("par_b failed")
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
                    id="par_a",
                    type=NodeType.agent,
                    label="A",
                    config={"subtype": "researcher", "output_key": "r_a"},
                ),
                WorkflowNode(
                    id="par_b",
                    type=NodeType.agent,
                    label="B",
                    config={"subtype": "researcher", "output_key": "r_b"},
                    error_handling=ErrorConfig(on_error="skip"),
                ),
                WorkflowNode(
                    id="par_c",
                    type=NodeType.agent,
                    label="C",
                    config={"subtype": "researcher", "output_key": "r_c"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            events = await _collect_events(executor, state)

        # Two children completed normally
        assert set(executed) == {"par_a", "par_c"}

        # One was skipped
        skipped = _events_of_type(events, NodeSkippedEvent)
        assert len(skipped) == 1
        assert "par_b failed" in skipped[0].reason  # type: ignore[attr-defined]

        assert isinstance(events[-1], WorkflowCompletedEvent)

    # 5. Error in plan_and_execute body skips a step
    @pytest.mark.asyncio
    async def test_error_in_plan_and_execute_body_skips_step(self) -> None:
        """plan_and_execute with body that fails on step 2 (on_error=skip).

        Planner returns 3 steps; body should skip step 2, run steps 1 and 3.
        """
        body_calls: list[int] = []
        step_counter = 0

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            nonlocal step_counter
            # Planner node produces the plan
            if "_planner_" in node_id:
                plan = {
                    "steps": [
                        {"description": "step 1", "needs_search": False},
                        {"description": "step 2", "needs_search": False},
                        {"description": "step 3", "needs_search": False},
                    ]
                }
                return AgentOutput(
                    content=plan,
                    output_key="plan",
                    events=[],
                    token_usage={"total": 10},
                )

            # Body node: fail on step 2
            step_counter += 1
            current_step = step_counter
            if current_step == 2:
                raise RuntimeError("step 2 body failed")
            body_calls.append(current_step)
            return AgentOutput(
                content=f"body result step {current_step}",
                output_key="findings",
                events=[],
                token_usage={"total": 5},
            )

        root = WorkflowNode(
            id="pae",
            type=NodeType.plan_and_execute,
            label="Plan and Execute",
            config={
                "planner": {
                    "subtype": "planner",
                    "output_key": "plan",
                },
                "body": {
                    "id": "body_agent",
                    "type": "agent",
                    "label": "Body Agent",
                    "config": {
                        "subtype": "researcher",
                        "output_key": "findings",
                    },
                    "error_handling": {
                        "on_error": "skip",
                    },
                },
                "items_path": "steps",
                "item_state_key": "current_step",
                "max_iterations": 10,
                "min_iterations": 1,
                "max_replan_cycles": 0,
                "complete_on_exhaustion": True,
            },
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            events = await _collect_events(executor, state)

        # Steps 1 and 3 ran; step 2 was skipped
        assert body_calls == [1, 3]

        skipped = _events_of_type(events, NodeSkippedEvent)
        assert len(skipped) == 1
        assert "step 2 body failed" in skipped[0].reason  # type: ignore[attr-defined]

    # 6. Cascade failure in sequence
    @pytest.mark.asyncio
    async def test_cascade_failure_in_sequence(self) -> None:
        """Sequence of 2: first fails (default on_error=fail). Second never runs."""
        executed: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            if node_id == "first":
                raise RuntimeError("first agent crashed")
            executed.append(node_id)
            return AgentOutput(
                content="result",
                output_key="output",
                events=[],
            )

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="first",
                    type=NodeType.agent,
                    label="First",
                    config={"subtype": "researcher", "output_key": "out1"},
                    # No error_handling => defaults to fail
                ),
                WorkflowNode(
                    id="second",
                    type=NodeType.agent,
                    label="Second",
                    config={"subtype": "researcher", "output_key": "out2"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            with pytest.raises(RuntimeError, match="first agent crashed"):
                await _collect_events(executor, state)

        # Second agent never ran
        assert executed == []

    # 7. Retry with exponential backoff
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self) -> None:
        """Agent fails 2 times then succeeds. Verify exponential sleep calls."""
        call_count = 0

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient")
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
            id="backoff_agent",
            type=NodeType.agent,
            label="Backoff Agent",
            config={"subtype": "researcher", "output_key": "out"},
            error_handling=ErrorConfig(
                on_error="retry", max_retries=3, retry_delay_seconds=0.1
            ),
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with (
            patch(PATCH_TARGET, side_effect=fake_execute_agent),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            events = await _collect_events(executor, state)

        # Retry logic: retry_delay_seconds * (2 ** attempt)
        # First call fails -> retry attempt=0: sleep(0.1 * 2^0 = 0.1)
        # Second call fails -> retry attempt=1: sleep(0.1 * 2^1 = 0.2)
        # Third call succeeds
        assert mock_sleep.call_count == 2
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_args[0] == pytest.approx(0.1)
        assert sleep_args[1] == pytest.approx(0.2)

        assert isinstance(events[-1], WorkflowCompletedEvent)

    # 8. Error preserves partial state
    @pytest.mark.asyncio
    async def test_error_preserves_partial_state(self) -> None:
        """Sequence of 2: first writes state, second fails with on_error=skip.

        First agent's state should be preserved.
        """

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            st = kwargs.get("state")
            if node_id == "writer":
                if st:
                    st.append(node_id, "important_data", "preserved value")
                return AgentOutput(
                    content="wrote state",
                    output_key="important_data",
                    events=[],
                )
            if node_id == "failer":
                raise RuntimeError("failer blew up")
            return AgentOutput(content="", output_key="output", events=[])

        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="sequence",
            children=[
                WorkflowNode(
                    id="writer",
                    type=NodeType.agent,
                    label="Writer",
                    config={"subtype": "researcher", "output_key": "important_data"},
                ),
                WorkflowNode(
                    id="failer",
                    type=NodeType.agent,
                    label="Failer",
                    config={"subtype": "researcher", "output_key": "out2"},
                    error_handling=ErrorConfig(on_error="skip"),
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            events = await _collect_events(executor, state)

        # First agent's state is preserved
        assert state.get("important_data") == "preserved value"

        # Second was skipped
        skipped = _events_of_type(events, NodeSkippedEvent)
        assert len(skipped) == 1

        assert isinstance(events[-1], WorkflowCompletedEvent)

    # 9. All parallel children fail
    @pytest.mark.asyncio
    async def test_all_parallel_children_fail(self) -> None:
        """3 parallel children, all fail (default on_error=fail). First error propagates."""

        async def always_fail(node_id: str, **kwargs: Any) -> AgentOutput:
            raise RuntimeError(f"{node_id} failed")

        root = WorkflowNode(
            id="par",
            type=NodeType.parallel,
            label="parallel",
            children=[
                WorkflowNode(
                    id="c1",
                    type=NodeType.agent,
                    label="C1",
                    config={"subtype": "researcher", "output_key": "r1"},
                ),
                WorkflowNode(
                    id="c2",
                    type=NodeType.agent,
                    label="C2",
                    config={"subtype": "researcher", "output_key": "r2"},
                ),
                WorkflowNode(
                    id="c3",
                    type=NodeType.agent,
                    label="C3",
                    config={"subtype": "researcher", "output_key": "r3"},
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=always_fail):
            with pytest.raises(RuntimeError, match="failed"):
                await _collect_events(executor, state)

    # 10. Nested error handling: sequence in loop with skippable failing agent
    @pytest.mark.asyncio
    async def test_nested_error_handling_sequence_in_loop(self) -> None:
        """Loop with body=sequence containing agent that fails (on_error=skip).

        Loop: min_iterations=1, max_iterations=3. The failing agent is skipped
        each iteration but the loop still runs all 3 iterations.
        """
        ok_calls: list[str] = []

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            if node_id == "failing_step":
                raise RuntimeError("always fails in loop")
            ok_calls.append(node_id)
            return AgentOutput(
                content=f"result-{node_id}",
                output_key="output",
                events=[],
            )

        until_cond = StateCondition(key="never_set", operator="exists")

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
                    id="loop_body_seq",
                    type=NodeType.sequence,
                    label="loop body sequence",
                    children=[
                        WorkflowNode(
                            id="ok_step",
                            type=NodeType.agent,
                            label="OK Step",
                            config={
                                "subtype": "researcher",
                                "output_key": "step_out",
                            },
                        ),
                        WorkflowNode(
                            id="failing_step",
                            type=NodeType.agent,
                            label="Failing Step",
                            config={
                                "subtype": "researcher",
                                "output_key": "fail_out",
                            },
                            error_handling=ErrorConfig(on_error="skip"),
                        ),
                    ],
                ),
            ],
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(PATCH_TARGET, side_effect=fake_execute_agent):
            events = await _collect_events(executor, state)

        # ok_step ran 3 times (once per iteration)
        assert ok_calls == ["ok_step", "ok_step", "ok_step"]

        # failing_step was skipped 3 times
        skipped = _events_of_type(events, NodeSkippedEvent)
        assert len(skipped) == 3
        for s in skipped:
            assert "always fails in loop" in s.reason  # type: ignore[attr-defined]

        # Loop ran all 3 iterations
        iterations = _events_of_type(events, LoopIterationEvent)
        assert len(iterations) == 3

        # Loop exited due to max_iterations
        exits = _events_of_type(events, LoopExitEvent)
        assert len(exits) == 1
        assert exits[0].reason == "max_iterations"  # type: ignore[attr-defined]
        assert exits[0].total_iterations == 3  # type: ignore[attr-defined]
