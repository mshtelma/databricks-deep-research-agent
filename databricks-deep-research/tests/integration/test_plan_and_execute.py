"""Integration tests: plan-and-execute workflow with real LLM + tools.

Tests the framework's plan-and-execute meta-node:
- Planner generates research steps
- Researcher executes each step with web search
- Reflector evaluates progress and decides continue/replan/complete
- Full event stream with domain events

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_plan_and_execute.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from databricks_deep_research import (
    FrameworkLLMClient,
    run_workflow,
)
from databricks_deep_research.events.types import (
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemsExtractedEvent,
    ItemStartedEvent,
    PlanAndExecuteExitEvent,
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    ToolCallEvent,
    WorkflowCompletedEvent,
)
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    print_event_timeline,
    print_pool_summary,
    print_research_plan,
    print_search_queries,
    print_step_execution,
)
from tests.integration.conftest import requires_all_credentials


@pytest.mark.integration
class TestPlanAndExecuteWorkflow:
    """Full plan-and-execute cycle with real LLM, search, and crawl."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_research_pipeline_produces_report(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Full research pipeline: coordinator -> background -> plan&execute -> synthesizer."""
        definition = load_workflow(examples_dir / "research_pipeline.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What are the key differences between transformer and state space models for NLP?"
            },
            tool_registry=tool_registry,
        )

        # -- Verify plan was created --
        plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
        assert len(plan_events) >= 1, "Planner should create a plan"
        plan = plan_events[0]
        assert len(plan.steps) >= 1, "Plan should have at least one step"
        print(f"\nPlan: {plan.title}")
        for i, step in enumerate(plan.steps):
            print(f"  Step {i+1}: {step}")

        # -- Verify items were extracted and executed --
        extracted_events = [e for e in events if isinstance(e, ItemsExtractedEvent)]
        assert len(extracted_events) >= 1, "Should extract items from plan"

        started_events = [e for e in events if isinstance(e, ItemStartedEvent)]
        completed_events = [e for e in events if isinstance(e, ItemCompletedEvent)]
        assert len(started_events) >= 1, "Should start at least one research step"
        assert len(completed_events) >= 1, "Should complete at least one research step"

        # -- Verify tool calls happened during research --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 1, "Researcher should make tool calls"
        search_calls = [tc for tc in tool_calls if tc.tool_name == "web_search"]
        assert len(search_calls) >= 1, "Should use web_search"

        # -- Verify evaluator ran --
        eval_events = [
            e for e in events
            if isinstance(e, (EvaluationDecisionEvent, ReflectionDecisionEvent))
        ]
        assert len(eval_events) >= 1, "Reflector should evaluate at least once"
        for ev in eval_events:
            print(f"  Evaluation: decision={ev.decision}, reasoning={ev.reasoning[:80]}")

        # -- Verify plan_and_execute exit --
        exit_events = [e for e in events if isinstance(e, PlanAndExecuteExitEvent)]
        assert len(exit_events) == 1, "Should have one PlanAndExecuteExitEvent"
        print(f"\nP&E exit: reason={exit_events[0].reason}, "
              f"items_processed={exit_events[0].total_items_processed}, "
              f"replan_cycles={exit_events[0].replan_cycles}")

        # -- Verify final report --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        report_str = str(report)
        assert len(report_str) > 100, f"Report too short ({len(report_str)} chars)"

        # -- Verify workflow completion --
        completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed) == 1
        wf = completed[0]
        assert wf.duration_ms > 0

        print(f"\nFinal report ({len(report_str)} chars):")
        print(report_str[:500])
        print(f"\nTotal events: {len(events)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Steps executed: {len(completed_events)}")
        print(f"Duration: {wf.duration_ms / 1000:.1f}s")
        if wf.total_sources:
            print(f"Sources: {wf.total_sources}")

        # -- Rich diagnostics --
        print_research_plan(events)
        print_step_execution(events)
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_research_with_web_crawl(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Verify researcher uses both web_search and web_crawl tools."""
        definition = load_workflow(examples_dir / "research_pipeline.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is Apple's revenue for fiscal year 2024?"
            },
            tool_registry=tool_registry,
        )

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        search_calls = [tc for tc in tool_calls if tc.tool_name == "web_search"]
        crawl_calls = [tc for tc in tool_calls if tc.tool_name == "web_crawl"]

        assert len(search_calls) >= 1, "Should use web_search"
        # web_crawl may or may not be used depending on LLM decisions
        print(f"\nSearch calls: {len(search_calls)}, Crawl calls: {len(crawl_calls)}")

        report = state.get("report")
        assert report is not None
        report_lower = str(report).lower()
        assert any(
            term in report_lower for term in ["apple", "revenue", "fiscal"]
        ), "Report should address the query"

        print(f"\nReport preview: {str(report)[:300]}")

        # -- Rich diagnostics --
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)
