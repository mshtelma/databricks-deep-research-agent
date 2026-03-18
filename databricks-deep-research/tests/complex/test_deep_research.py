"""Complex, long-running framework research tests.

These tests verify the standalone framework can execute full research
pipelines end-to-end — the same capability as the app's orchestrator
but driven entirely by YAML workflow definitions.

Tests verify:
- Multi-step research with plan-and-execute
- Pool accumulation (observations, sources)
- Reflector evaluation decisions
- Synthesized report quality
- Event stream completeness

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- BRAVE_API_KEY
- 3-10 minutes per test

Run with:
    cd databricks-deep-research
    uv run pytest tests/complex/test_deep_research.py -v -s --timeout=600
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from databricks_deep_research import (
    FrameworkLLMClient,
    run_workflow,
)
from databricks_deep_research.events.types import (
    BackgroundCompletedEvent,
    CoordinatorClassifiedEvent,
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    ToolCallEvent,
)
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.loader import load_workflow
from tests.complex.conftest import requires_all_credentials
from tests.helpers import (
    assert_report_has_substance,
    event_summary,
    print_evaluator_decisions,
    print_event_timeline,
    print_pool_summary,
    print_research_plan,
    print_search_queries,
    print_step_execution,
)


@pytest.mark.complex
class TestDeepResearch:
    """Full research pipeline tests with the standalone framework."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_multi_step_research(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Full pipeline: coordinator -> background -> plan&execute -> synthesizer.

        Verifies the framework can plan, execute multiple research steps,
        reflect on progress, and synthesize a coherent report.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the key differences between transformer "
                    "architectures and state space models like Mamba "
                    "for natural language processing?"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Coordinator classified the query --
        coord_events = [e for e in events if isinstance(e, CoordinatorClassifiedEvent)]
        assert len(coord_events) == 1, "Should classify query once"
        assert coord_events[0].is_simple is False, "Should be a complex query"

        # -- Background gathered initial context --
        bg_events = [e for e in events if isinstance(e, BackgroundCompletedEvent)]
        assert len(bg_events) >= 1, "Should run background search"

        # -- Plan was created --
        plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
        assert len(plan_events) >= 1, "Should create research plan"
        plan = plan_events[0]
        assert len(plan.steps) >= 1, "Plan should have steps"

        # -- Research steps were executed --
        items_started = [e for e in events if isinstance(e, ItemStartedEvent)]
        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        assert len(items_started) >= 1, "Should start research steps"
        assert len(items_completed) >= 1, "Should complete research steps"

        # -- Tool calls happened --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 2, "Should make multiple tool calls"

        # -- Reflector evaluated --
        eval_events = [
            e for e in events
            if isinstance(e, (EvaluationDecisionEvent, ReflectionDecisionEvent))
        ]
        assert len(eval_events) >= 1, "Should evaluate research progress"

        # -- Pools accumulated data --
        obs_pool = state.pools.get("observations")
        src_pool = state.pools.get("sources")
        obs_count = obs_pool.count() if obs_pool else 0
        src_count = src_pool.count() if src_pool else 0
        assert obs_count > 0, (
            f"Observations pool is empty after {len(tool_calls)} tool calls. "
            "Pool writes from researcher are not working."
        )
        assert src_count > 0, (
            f"Sources pool is empty after {len(tool_calls)} tool calls. "
            "Source extraction from ReAct loop is not working."
        )

        # -- Report was synthesized with substance --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(
            report_str,
            min_length=500,
            required_term_groups=(
                ("transformer", "attention", "architecture"),
                ("mamba", "state space", "ssm", "s4"),
            ),
        )

        # -- Summary --
        print(f"\n{'='*60}")
        print(f"Multi-step research completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Plan steps: {len(plan.steps)}")
        print(f"Research steps executed: {len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Evaluations: {len(eval_events)}")
        print(f"Observations pool: {obs_count} items")
        print(f"Sources pool: {src_count} items")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")

        # -- Rich diagnostics --
        print_research_plan(events)
        print_step_execution(events)
        print_search_queries(events)
        print_pool_summary(state)
        print_evaluator_decisions(events)
        print_event_timeline(events)

        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_comparative_research(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Compare multiple entities requiring multi-step research.

        Similar to the app's test_multi_entity_comparison but driven
        entirely by the standalone framework.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Compare Apple and Microsoft's annual revenue, "
                    "market capitalization, and AI strategy as of 2024"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]

        # -- Pool diagnostics --
        obs_pool = state.pools.get("observations")
        src_pool = state.pools.get("sources")
        obs_count = obs_pool.count() if obs_pool else 0
        src_count = src_pool.count() if src_pool else 0
        assert obs_count > 0, "Observations pool should not be empty"
        assert src_count > 0, "Sources pool should not be empty"

        report = state.get("report")
        assert report is not None, "Should produce report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=500)

        report_lower = report_str.lower()
        assert "apple" in report_lower, "Should mention Apple"
        assert "microsoft" in report_lower, "Should mention Microsoft"
        assert any(
            term in report_lower for term in ["revenue", "market cap", "capitalization"]
        ), "Should discuss financials"
        assert any(
            term in report_lower for term in ["ai", "artificial intelligence", "copilot", "intelligence"]
        ), "Should discuss AI strategy"

        print(f"\n{'='*60}")
        print(f"Comparative research completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Steps executed: {len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Observations: {obs_count}, Sources: {src_count}")
        print(f"Report length: {len(report_str)} chars")

        # -- Rich diagnostics --
        print_research_plan(events)
        print_step_execution(events)
        print_search_queries(events)
        print_pool_summary(state)
        print_evaluator_decisions(events)
        print_event_timeline(events)

        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_analytical_query(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Complex analytical query requiring synthesis of diverse sources.

        Verifies the framework can handle queries that need:
        - Multiple research angles
        - Balanced pros/cons analysis
        - Source diversity
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the economic and environmental trade-offs "
                    "of electric vehicles versus hydrogen fuel cell vehicles "
                    "for long-haul trucking?"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]

        # -- Pool diagnostics --
        obs_pool = state.pools.get("observations")
        src_pool = state.pools.get("sources")
        obs_count = obs_pool.count() if obs_pool else 0
        src_count = src_pool.count() if src_pool else 0
        assert obs_count > 0, "Observations pool should not be empty"
        assert src_count > 0, "Sources pool should not be empty"

        report = state.get("report")
        assert report is not None, "Should produce report"
        report_str = str(report)
        assert_report_has_substance(
            report_str,
            min_length=500,
            required_term_groups=(
                ("electric", "battery", "ev"),
                ("hydrogen", "fuel cell"),
                ("cost", "economic", "price", "investment", "infrastructure"),
                ("emission", "environment", "carbon", "clean", "climate"),
            ),
        )

        print(f"\n{'='*60}")
        print(f"Analytical research completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Steps executed: {len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Observations: {obs_count}, Sources: {src_count}")
        print(f"Report length: {len(report_str)} chars")

        # -- Rich diagnostics --
        print_research_plan(events)
        print_step_execution(events)
        print_search_queries(events)
        print_pool_summary(state)
        print_evaluator_decisions(events)
        print_event_timeline(events)

        print(f"\nReport preview:\n{report_str[:500]}")


@pytest.mark.complex
class TestEventStreamCompleteness:
    """Verify the event stream captures the full execution lifecycle."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_event_stream_has_all_lifecycle_events(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Every expected event type should appear in a full pipeline run."""
        definition = load_workflow(examples_dir / "research_pipeline.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What are the latest advances in quantum computing in 2025?"
            },
            tool_registry=tool_registry,
        )

        event_types = {type(e).__name__ for e in events}

        # These events MUST appear in any full research pipeline run
        required_events = {
            "WorkflowStartedEvent",
            "WorkflowCompletedEvent",
            "NodeStartedEvent",
            "NodeCompletedEvent",
            "AgentOutputEvent",
            "CoordinatorClassifiedEvent",
            "PlanCreatedEvent",
            "ItemsExtractedEvent",
            "ItemStartedEvent",
            "ItemCompletedEvent",
            "PlanAndExecuteExitEvent",
            "ToolCallEvent",
            "ToolResultEvent",
        }

        missing = required_events - event_types
        assert not missing, (
            f"Missing required events: {missing}\n"
            f"Got: {sorted(event_types)}"
        )

        # Verify event ordering: workflow_started must come first
        assert isinstance(events[0], type(events[0]))  # sanity
        first_event_type = type(events[0]).__name__
        assert first_event_type == "WorkflowStartedEvent", (
            f"First event should be WorkflowStartedEvent, got {first_event_type}"
        )

        # Last event should be WorkflowCompletedEvent
        last_event_type = type(events[-1]).__name__
        assert last_event_type == "WorkflowCompletedEvent", (
            f"Last event should be WorkflowCompletedEvent, got {last_event_type}"
        )

        # -- Report should have substance --
        report = state.get("report")
        assert report is not None, "Should produce report"
        assert_report_has_substance(str(report), min_length=200)

        print("\nAll required events present.")
        print(f"Event summary: {event_summary(events)}")
        print(f"Total events: {len(events)}")
        print_event_timeline(events)
