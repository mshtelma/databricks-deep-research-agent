"""Tests for event sequence contracts across different workflow shapes.

Tier 2: fast mocked tests (no credentials needed, <30s total).
Verifies that events are emitted in the correct order for sequences,
parallel nodes, loops, plan-and-execute, and workflow boundaries.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    BranchSelectedEvent,
    CoordinatorClassifiedEvent,
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemsExtractedEvent,
    LoopExitEvent,
    LoopIterationEvent,
    NodeCompletedEvent,
    NodeStartedEvent,
    PlanAndExecuteExitEvent,
    PlanCreatedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from databricks_deep_research.workflow.conditions import StateCondition
from databricks_deep_research.workflow.definition import (
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
def _make_definition(root: WorkflowNode) -> WorkflowDefinition:
    """Wrap a root node into a minimal WorkflowDefinition."""
    return WorkflowDefinition(
        id="test-wf",
        name="Test Workflow",
        root=root,
    )

# ---------------------------------------------------------------------------
# Event ordering tests
# ---------------------------------------------------------------------------


class TestEventOrdering:
    """Verify event sequence contracts across different workflow shapes."""

    @pytest.mark.asyncio
    async def test_workflow_started_always_first(self) -> None:
        """WorkflowStartedEvent is always the first event emitted."""

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            return AgentOutput(
                content="result",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="result",
                    )
                ],
            )

        root = WorkflowNode(
            id="single_agent",
            type=NodeType.agent,
            label="Single Agent",
            config={"subtype": "researcher", "output_key": "findings"},
        )
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="test")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=fake_execute_agent,
        ):
            events = await _collect_events(executor, state)

        assert len(events) > 0
        assert isinstance(events[0], WorkflowStartedEvent)

    @pytest.mark.asyncio
    async def test_workflow_completed_always_last(self) -> None:
        """WorkflowCompletedEvent is always the last event emitted."""

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
                    config={"subtype": "researcher", "output_key": "findings_a"},
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="Agent B",
                    config={"subtype": "researcher", "output_key": "findings_b"},
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

        assert len(events) > 0
        assert isinstance(events[-1], WorkflowCompletedEvent)

    @pytest.mark.asyncio
    async def test_node_started_before_completed(self) -> None:
        """For each node, NodeStartedEvent comes before NodeCompletedEvent."""

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
                    config={"subtype": "researcher", "output_key": "findings_a"},
                ),
                WorkflowNode(
                    id="agent_b",
                    type=NodeType.agent,
                    label="Agent B",
                    config={"subtype": "researcher", "output_key": "findings_b"},
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

        # Collect all node_ids that have both started and completed events
        started_indices: dict[str, int] = {}
        completed_indices: dict[str, int] = {}
        for idx, event in enumerate(events):
            if isinstance(event, NodeStartedEvent):
                # Record first occurrence
                if event.node_id not in started_indices:
                    started_indices[event.node_id] = idx
            if isinstance(event, NodeCompletedEvent):
                # Record last occurrence
                completed_indices[event.node_id] = idx

        # Every node that has both events must have started < completed
        common_ids = set(started_indices.keys()) & set(completed_indices.keys())
        assert len(common_ids) > 0, "Should have at least one node with start+complete"
        for node_id in common_ids:
            assert started_indices[node_id] < completed_indices[node_id], (
                f"Node {node_id}: started at index {started_indices[node_id]} "
                f"but completed at index {completed_indices[node_id]}"
            )

    @pytest.mark.asyncio
    async def test_plan_and_execute_event_sequence(self) -> None:
        """Plan-and-execute emits events in the correct order."""

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            # Planner: returns a dict with steps list
            if "_planner_" in node_id:
                plan = {"steps": [
                    {"title": "Step 1", "query": "research topic A", "needs_search": False},
                    {"title": "Step 2", "query": "research topic B", "needs_search": False},
                ]}
                return AgentOutput(
                    content=plan,
                    output_key="plan",
                    events=[
                        AgentOutputEvent(
                            node_id=node_id,
                            timestamp="T",
                            output_key="plan",
                            output_preview="plan with 2 steps",
                        )
                    ],
                )
            # Evaluator: return continue decision
            if "_eval_" in node_id:
                return AgentOutput(
                    content={"decision": "continue", "reasoning": "more work needed"},
                    output_key="evaluation",
                    events=[
                        AgentOutputEvent(
                            node_id=node_id,
                            timestamp="T",
                            output_key="evaluation",
                            output_preview="continue",
                        )
                    ],
                )
            # Body: return findings
            return AgentOutput(
                content=f"findings-{node_id}",
                output_key="findings",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="findings",
                        output_preview=f"findings-{node_id}",
                    )
                ],
            )

        root = WorkflowNode(
            id="pae",
            type=NodeType.plan_and_execute,
            label="Plan and Execute",
            config={
                "planner": {
                    "subtype": "planner",
                    "output_key": "plan",
                    "output_format": "json",
                },
                "items_path": "steps",
                "item_state_key": "current_step",
                "body": {
                    "id": "body_researcher",
                    "type": "agent",
                    "label": "Body Researcher",
                    "config": {
                        "subtype": "researcher",
                        "output_key": "findings",
                    },
                },
                "evaluator": {
                    "subtype": "reflector",
                    "output_key": "evaluation",
                    "output_format": "json",
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

        # Planner AgentOutputEvent comes before ItemsExtractedEvent
        planner_output_events = [
            e for e in events
            if isinstance(e, AgentOutputEvent) and e.output_key == "plan"
        ]
        items_extracted_events = _events_of_type(events, ItemsExtractedEvent)
        assert len(planner_output_events) >= 1
        assert len(items_extracted_events) >= 1

        planner_idx = events.index(planner_output_events[0])
        extracted_idx = events.index(items_extracted_events[0])
        assert planner_idx < extracted_idx, (
            "Planner AgentOutputEvent must come before ItemsExtractedEvent"
        )

        # ItemStartedEvent[0] before ItemCompletedEvent[0]
        item_started = _events_of_type(events, ItemStartedEvent)
        item_completed = _events_of_type(events, ItemCompletedEvent)
        assert len(item_started) >= 2
        assert len(item_completed) >= 2

        item_started_0_idx = events.index(item_started[0])
        item_completed_0_idx = events.index(item_completed[0])
        assert item_started_0_idx < item_completed_0_idx, (
            "ItemStartedEvent[0] must come before ItemCompletedEvent[0]"
        )

        # ItemStartedEvent[1] before ItemCompletedEvent[1]
        item_started_1_idx = events.index(item_started[1])
        item_completed_1_idx = events.index(item_completed[1])
        assert item_started_1_idx < item_completed_1_idx, (
            "ItemStartedEvent[1] must come before ItemCompletedEvent[1]"
        )

        # PlanAndExecuteExitEvent is last before node completion
        pae_exits = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(pae_exits) == 1
        pae_exit_idx = events.index(pae_exits[0])
        # The PlanAndExecuteExitEvent should come before the NodeCompletedEvent
        # for the pae node (which is immediately followed by WorkflowCompletedEvent)
        node_completed_for_pae = [
            e for e in events
            if isinstance(e, NodeCompletedEvent) and e.node_id == "pae"
        ]
        assert len(node_completed_for_pae) == 1
        pae_node_completed_idx = events.index(node_completed_for_pae[0])
        assert pae_exit_idx < pae_node_completed_idx, (
            "PlanAndExecuteExitEvent must come before the pae NodeCompletedEvent"
        )

    @pytest.mark.asyncio
    async def test_loop_iteration_events_ordered(self) -> None:
        """Loop emits sequential LoopIterationEvents followed by LoopExitEvent."""

        async def fake_execute_agent(
            node_id: str, **kwargs: Any
        ) -> AgentOutput:
            return AgentOutput(
                content="step",
                output_key="output",
                events=[
                    AgentOutputEvent(
                        node_id=node_id,
                        timestamp="T",
                        output_key="output",
                        output_preview="step",
                    )
                ],
            )

        # Condition that never fires: key "never_exists" must exist
        until_cond = StateCondition(key="never_exists", operator="exists")

        root = WorkflowNode(
            id="loop",
            type=NodeType.loop,
            label="loop",
            config={
                "until": until_cond.model_dump(),
                "min_iterations": 3,
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

        # LoopIterationEvent.iteration values are sequential [1, 2, 3]
        loop_iterations = _events_of_type(events, LoopIterationEvent)
        assert len(loop_iterations) == 3
        iteration_values = [e.iteration for e in loop_iterations]  # type: ignore[attr-defined]
        assert iteration_values == [1, 2, 3]

        # LoopExitEvent comes after all iterations
        loop_exits = _events_of_type(events, LoopExitEvent)
        assert len(loop_exits) == 1
        last_iteration_idx = events.index(loop_iterations[-1])
        exit_idx = events.index(loop_exits[0])
        assert exit_idx > last_iteration_idx, (
            "LoopExitEvent must come after all LoopIterationEvents"
        )

    @pytest.mark.asyncio
    async def test_agent_output_count_matches_agents(self) -> None:
        """Exactly one AgentOutputEvent per agent node in a sequence of 3."""

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
                    id="agent_1",
                    type=NodeType.agent,
                    label="Agent 1",
                    config={"subtype": "researcher", "output_key": "r1"},
                ),
                WorkflowNode(
                    id="agent_2",
                    type=NodeType.agent,
                    label="Agent 2",
                    config={"subtype": "researcher", "output_key": "r2"},
                ),
                WorkflowNode(
                    id="agent_3",
                    type=NodeType.agent,
                    label="Agent 3",
                    config={"subtype": "researcher", "output_key": "r3"},
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

        agent_output_events = _events_of_type(events, AgentOutputEvent)
        assert len(agent_output_events) == 3

    @pytest.mark.asyncio
    async def test_parallel_events_from_all_children(self) -> None:
        """Parallel node emits NodeStarted/NodeCompleted for all 3 children."""

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
                        output_preview=f"result-{node_id}",
                    )
                ],
            )

        child_ids = ["child_a", "child_b", "child_c"]
        root = WorkflowNode(
            id="par",
            type=NodeType.parallel,
            label="parallel",
            children=[
                WorkflowNode(
                    id=cid,
                    type=NodeType.agent,
                    label=f"Agent {cid}",
                    config={"subtype": "researcher", "output_key": f"r_{cid}"},
                )
                for cid in child_ids
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

        # 3 agent children + 1 parallel parent = at least 4 NodeStartedEvents
        started_events = _events_of_type(events, NodeStartedEvent)
        assert len(started_events) >= 4

        # Check all 3 child node_ids appear in NodeStartedEvent events
        started_node_ids = {e.node_id for e in started_events}  # type: ignore[attr-defined]
        for cid in child_ids:
            assert cid in started_node_ids, (
                f"NodeStartedEvent missing for child {cid}"
            )

        # Check all 3 child node_ids appear in NodeCompletedEvent events
        completed_events = _events_of_type(events, NodeCompletedEvent)
        completed_node_ids = {e.node_id for e in completed_events}  # type: ignore[attr-defined]
        for cid in child_ids:
            assert cid in completed_node_ids, (
                f"NodeCompletedEvent missing for child {cid}"
            )

    @pytest.mark.asyncio
    async def test_workflow_events_bracket_all_nodes(self) -> None:
        """All node events fall between WorkflowStarted and WorkflowCompleted."""

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
                    id="agent_x",
                    type=NodeType.agent,
                    label="Agent X",
                    config={"subtype": "researcher", "output_key": "rx"},
                ),
                WorkflowNode(
                    id="agent_y",
                    type=NodeType.agent,
                    label="Agent Y",
                    config={"subtype": "researcher", "output_key": "ry"},
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

        # Find workflow boundary indices
        wf_started_idx = next(
            i for i, e in enumerate(events) if isinstance(e, WorkflowStartedEvent)
        )
        wf_completed_idx = next(
            i for i, e in enumerate(events) if isinstance(e, WorkflowCompletedEvent)
        )

        # Collect all NodeStartedEvent and NodeCompletedEvent indices
        node_event_indices = [
            i for i, e in enumerate(events)
            if isinstance(e, (NodeStartedEvent, NodeCompletedEvent))
        ]

        assert len(node_event_indices) > 0, "Should have node events"
        for idx in node_event_indices:
            assert wf_started_idx < idx < wf_completed_idx, (
                f"Node event at index {idx} is not between "
                f"WorkflowStarted ({wf_started_idx}) and "
                f"WorkflowCompleted ({wf_completed_idx})"
            )
