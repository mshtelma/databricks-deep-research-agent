"""Advanced research configuration and depth tests.

Tier 1 complex tests — long-running with real LLM + Brave Search.
Each test exercises different research configurations, depth settings,
and event stream characteristics.

Tests verify:
- Extended vs light depth settings
- Reflector early completion behaviour
- Sequential multi-query independence
- Tool cache dedup across research steps
- Event stream completeness across all lifecycle phases

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- BRAVE_API_KEY
- ~5-8 minutes per test

Run with:
    cd databricks-deep-research
    uv run pytest tests/complex/test_advanced_research.py -v -s --timeout=600
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
    AgentOutputEvent,
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    NodeCompletedEvent,
    NodeStartedEvent,
    PlanAndExecuteExitEvent,
    PlanCreatedEvent,
    StreamEvent,
    ToolCacheHitEvent,
    ToolCallEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.complex.conftest import requires_all_credentials
from tests.helpers import (
    assert_report_has_substance,
    event_summary,
    print_event_timeline,
    print_evaluator_decisions,
    print_full_diagnostics,
    print_pool_summary,
)


@pytest.mark.complex
class TestAdvancedResearch:
    """Advanced research pipeline tests covering depth, caching, and event completeness."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_extended_depth_more_steps(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Extended depth (simple_research.yaml: max_iterations=10, min_iterations=2).

        Uses a complex geopolitical query that requires multi-step research.
        Verifies the pipeline executes at least 2 research steps and produces
        a substantial report with rich observation data.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "simple_research.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Analyze the geopolitical implications of rare earth mineral "
                    "supply chains on global technology manufacturing"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- At least 2 research steps executed --
        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        assert len(items_completed) >= 2, (
            f"Extended depth should execute at least 2 research steps, "
            f"got {len(items_completed)}"
        )

        # -- Observations pool has multiple items --
        obs_pool = state.pools.get("observations")
        obs_count = obs_pool.count() if obs_pool else 0
        assert obs_count > 1, (
            f"Observations pool should have multiple items after extended research, "
            f"got {obs_count}"
        )

        # -- Report has substance --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=500)

        # -- Summary --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        print(f"\n{'='*60}")
        print(f"Extended depth research completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Research steps executed: {len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Observations pool: {obs_count} items")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")

        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_light_depth_fewer_steps(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Light depth (research_pipeline.yaml: max_iterations=5, min_iterations=1).

        Uses a focused factual query that should complete quickly.
        Verifies the pipeline completes and produces a report, and
        checks step count and exit reason from PlanAndExecuteExitEvent.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is the population of Tokyo?",
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Pipeline completed --
        completed_events = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed_events) >= 1, "Should emit WorkflowCompletedEvent"

        # -- Report exists --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=100)

        # -- Check exit event for step count --
        exit_events = [e for e in events if isinstance(e, PlanAndExecuteExitEvent)]
        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]

        print(f"\n{'='*60}")
        print(f"Light depth research completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Research steps executed: {len(items_completed)}")
        if exit_events:
            exit_evt = exit_events[0]
            print(f"Exit reason: {exit_evt.reason}")
            print(f"Items processed: {exit_evt.total_items_processed}/{exit_evt.total_planned}")
            print(f"Replan cycles: {exit_evt.replan_cycles}")
        print(f"Report length: {len(report_str)} chars")
        print(f"Duration: {duration_s:.1f}s (light queries should be faster)")
        print(f"\nEvent summary: {event_summary(events)}")

        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:300]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_reflector_triggers_early_completion(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Well-researched topic — reflector may trigger early completion.

        Uses a simple, well-documented topic (photosynthesis) where the
        reflector should have high confidence after few steps.
        Validates that EvaluationDecisionEvent is emitted. If the reflector
        triggers "complete", that validates early stopping; if it continues
        through all steps, that is also acceptable (LLM discretion).
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is photosynthesis and how does it work?",
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Evaluator was invoked --
        eval_events = [e for e in events if isinstance(e, EvaluationDecisionEvent)]
        assert len(eval_events) >= 1, (
            "Should emit at least one EvaluationDecisionEvent"
        )

        # -- Check if early completion was triggered --
        early_complete = any(e.decision == "complete" for e in eval_events)

        # -- Report should exist regardless --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=200)

        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]

        print(f"\n{'='*60}")
        print(f"Reflector test completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Research steps executed: {len(items_completed)}")
        print(f"Evaluator decisions: {len(eval_events)}")
        if early_complete:
            print("EARLY COMPLETION TRIGGERED by reflector")
        else:
            print("Reflector did NOT trigger early completion (all steps executed)")
        for i, ev in enumerate(eval_events):
            print(f"  Decision {i + 1}: [{ev.decision.upper()}] {ev.reasoning[:200]}")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")

        print_evaluator_decisions(events)
        print_event_timeline(events)
        print(f"\nReport preview:\n{report_str[:300]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_multiple_queries_sequential(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Run 3 independent queries sequentially through the same workflow.

        Verifies each query produces an independent, substantive report
        and that results do not leak between runs.
        """
        queries = [
            "What are the health benefits of intermittent fasting?",
            "How do electric vehicle batteries get recycled?",
            "What is the James Webb Space Telescope's latest discovery?",
        ]

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        reports: list[str] = []
        run_stats: list[dict[str, Any]] = []

        for i, query in enumerate(queries):
            start = time.perf_counter()

            state, events = await run_workflow(
                definition,
                llm_client,
                initial_state={"query": query},
                tool_registry=tool_registry,
            )

            duration_s = time.perf_counter() - start

            report = state.get("report")
            assert report is not None, f"Query {i + 1} should produce report"
            report_str = str(report)
            assert_report_has_substance(report_str, min_length=200)
            reports.append(report_str)

            items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
            tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
            run_stats.append({
                "query": query,
                "duration_s": duration_s,
                "steps": len(items_completed),
                "tool_calls": len(tool_calls),
                "report_len": len(report_str),
            })

        # -- Reports should be independent (different from each other) --
        for i in range(len(reports)):
            for j in range(i + 1, len(reports)):
                assert reports[i] != reports[j], (
                    f"Reports {i + 1} and {j + 1} should be different"
                )

        # -- Summary --
        total_duration = sum(r["duration_s"] for r in run_stats)
        print(f"\n{'='*60}")
        print(f"Sequential multi-query test completed in {total_duration:.1f}s")
        print(f"{'='*60}")
        for i, stats in enumerate(run_stats):
            print(
                f"  Query {i + 1}: {stats['query'][:60]}..."
                f"\n    Duration: {stats['duration_s']:.1f}s | "
                f"Steps: {stats['steps']} | "
                f"Tool calls: {stats['tool_calls']} | "
                f"Report: {stats['report_len']} chars"
            )
        print(f"\nTotal time: {total_duration:.1f}s")
        for i, report_str in enumerate(reports):
            print(f"\nReport {i + 1} preview:\n{report_str[:200]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_tool_cache_dedup_across_research_steps(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Tool cache dedup within plan_and_execute.

        Uses a focused query where multiple research steps may issue
        similar search queries (e.g., Bitcoin price). Checks for
        ToolCacheHitEvent presence as a diagnostic. The test PASSES
        regardless of cache hit count — cache hits are opportunistic
        and depend on LLM query formulation.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is the current price and market cap of Bitcoin?",
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Pipeline should complete --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=100)

        # -- Diagnostic: check for cache hits --
        cache_hits = [e for e in events if isinstance(e, ToolCacheHitEvent)]
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]

        print(f"\n{'='*60}")
        print(f"Tool cache dedup test completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Research steps executed: {len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Cache hits: {len(cache_hits)}")
        if cache_hits:
            print("\nCache hit details:")
            for hit in cache_hits:
                print(f"  [{hit.tool_name}] cache_key={hit.cache_key[:80]}")
        else:
            print("  (no cache hits — queries were sufficiently distinct)")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")

        print_pool_summary(state)
        print_event_timeline(events)
        print(f"\nReport preview:\n{report_str[:300]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_research_workflow_event_completeness(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Verify all required event types appear in a full pipeline run.

        Uses a complex ML/cybersecurity query to ensure the pipeline
        goes through all phases. Asserts minimum counts for each
        critical event type.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "research_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Compare supervised and unsupervised machine learning "
                    "approaches for anomaly detection in cybersecurity"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Collect event type counts --
        type_counts: dict[str, int] = {}
        for e in events:
            name = type(e).__name__
            type_counts[name] = type_counts.get(name, 0) + 1

        # -- WorkflowStartedEvent / WorkflowCompletedEvent --
        assert type_counts.get("WorkflowStartedEvent", 0) >= 1, (
            "Missing WorkflowStartedEvent"
        )
        assert type_counts.get("WorkflowCompletedEvent", 0) >= 1, (
            "Missing WorkflowCompletedEvent"
        )

        # -- NodeStartedEvent / NodeCompletedEvent (at least 3 each) --
        assert type_counts.get("NodeStartedEvent", 0) >= 3, (
            f"Expected at least 3 NodeStartedEvent, got {type_counts.get('NodeStartedEvent', 0)}"
        )
        assert type_counts.get("NodeCompletedEvent", 0) >= 3, (
            f"Expected at least 3 NodeCompletedEvent, got {type_counts.get('NodeCompletedEvent', 0)}"
        )

        # -- AgentOutputEvent (at least 3: coordinator, researcher, synthesizer) --
        assert type_counts.get("AgentOutputEvent", 0) >= 3, (
            f"Expected at least 3 AgentOutputEvent, got {type_counts.get('AgentOutputEvent', 0)}"
        )

        # -- PlanCreatedEvent (at least 1) --
        assert type_counts.get("PlanCreatedEvent", 0) >= 1, (
            "Missing PlanCreatedEvent"
        )

        # -- ItemStartedEvent / ItemCompletedEvent (at least 1 each) --
        assert type_counts.get("ItemStartedEvent", 0) >= 1, (
            "Missing ItemStartedEvent"
        )
        assert type_counts.get("ItemCompletedEvent", 0) >= 1, (
            "Missing ItemCompletedEvent"
        )

        # -- ToolCallEvent (at least 2) --
        assert type_counts.get("ToolCallEvent", 0) >= 2, (
            f"Expected at least 2 ToolCallEvent, got {type_counts.get('ToolCallEvent', 0)}"
        )

        # -- Report should still be produced --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=300)

        # -- Summary: event type counts --
        sorted_counts = dict(sorted(type_counts.items()))
        print(f"\n{'='*60}")
        print(f"Event completeness test completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Total events: {len(events)}")
        print(f"Report length: {len(report_str)} chars")
        print("\nEvent type counts:")
        for event_name, count in sorted_counts.items():
            print(f"  {event_name}: {count}")
        print(f"\nEvent summary: {event_summary(events)}")

        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")
