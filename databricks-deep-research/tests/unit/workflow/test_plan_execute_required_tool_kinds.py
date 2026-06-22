"""Plan-and-execute required tool-kind completion gates."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.events.types import (
    EvaluationDecisionEvent,
    PlanAndExecuteExitEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState
from tests.conftest import build_mock_llm_client as _mock_llm_client
from tests.conftest import collect_events as _collect_events
from tests.conftest import events_of_type as _events_of_type


def _definition(root: WorkflowNode) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="test-required-tool-gates",
        name="Required Tool Gates",
        tools=[
            ToolDeclaration(
                name="vector_search",
                kind="vector_search",
                config={"index_name": "main.officeqa.idx"},
            ),
            ToolDeclaration(
                name="table_read_2",
                kind="table_read",
                config={"table_name": "main.officeqa.treasury_tables"},
            ),
        ],
        root=root,
    )


def _node() -> WorkflowNode:
    return WorkflowNode(
        id="pe",
        type=NodeType.plan_and_execute,
        label="Plan and execute",
        config={
            "planner": {"subtype": "planner", "output_key": "plan"},
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": {
                "id": "researcher",
                "type": "agent",
                "label": "Researcher",
                "config": {
                    "subtype": "researcher",
                    "output_key": "findings",
                    "tools": ["vector_search", "table_read_2"],
                },
            },
            "evaluator": {"subtype": "reflector", "output_key": "evaluation"},
            "max_iterations": 3,
            "min_iterations": 1,
            "max_replan_cycles": 0,
            "required_tool_kind_groups": [
                ["vector_search"],
                ["table_search", "table_read", "table_load"],
            ],
        },
    )


@pytest.mark.asyncio
async def test_evaluator_complete_cannot_skip_required_tool_kind_groups() -> None:
    researcher_calls = 0

    async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        nonlocal researcher_calls
        config = kwargs.get("config")

        if config and config.subtype == "planner":
            return AgentOutput(
                content={
                    "steps": [
                        {
                            "id": "semantic",
                            "title": "Find vector candidates",
                            "needs_search": True,
                        },
                        {
                            "id": "exact",
                            "title": "Read exact table evidence",
                            "needs_search": True,
                        },
                    ]
                },
                output_key="plan",
                events=[],
            )

        if config and config.subtype == "reflector":
            return AgentOutput(
                content={"decision": "complete", "reasoning": "Enough evidence"},
                output_key="evaluation",
                events=[],
            )

        researcher_calls += 1
        tool_name = "vector_search" if researcher_calls == 1 else "table_read_2"
        return AgentOutput(
            content={"findings": f"used {tool_name}"},
            output_key="findings",
            events=[
                ToolCallEvent(
                    node_id=node_id,
                    timestamp="2026-05-27T00:00:00Z",
                    tool_name=tool_name,
                    arguments={},
                ),
                ToolResultEvent(
                    node_id=node_id,
                    timestamp="2026-05-27T00:00:00Z",
                    tool_name=tool_name,
                    result_summary="ok",
                    source_count=1,
                    accepted_source_count=1,
                    tool_success=True,
                ),
            ],
        )

    executor = WorkflowExecutor(_definition(_node()), _mock_llm_client())
    state = WorkflowState(query="OfficeQA table question")

    with patch(
        "databricks_deep_research.workflow.executor.execute_agent",
        side_effect=fake_execute_agent,
    ):
        events = await _collect_events(executor, state)

    eval_events = _events_of_type(events, EvaluationDecisionEvent)
    assert [event.decision for event in eval_events] == ["continue", "complete"]
    assert "Required tool kind groups are still missing" in eval_events[0].reasoning

    exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
    assert len(exit_events) == 1
    assert exit_events[0].reason == "evaluator_complete"
    assert exit_events[0].total_items_processed == 2
    assert state.get("observed_tool_kinds") == ["table_read", "vector_search"]
    assert state.get("missing_required_tool_kind_groups") == []
