"""Integration tests: lightweight/quick workflow modes with REAL LLM.

Tier 1 integration tests verifying the framework operates correctly in
lightweight and quick workflow configurations against real Databricks
LLM endpoints and Brave Search API.

Covers:
- search_and_summarize.yaml  (background search + synthesizer)
- single_agent.yaml          (coordinator classification only)
- simple_research.yaml       (full pipeline with plan_and_execute)

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_lightweight_workflow.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
    AgentOutputEvent,
    BackgroundCompletedEvent,
    CoordinatorClassifiedEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    PlanCreatedEvent,
    ToolCallEvent,
    WorkflowCompletedEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    assert_report_has_substance,
    event_summary,
    print_event_timeline,
    print_full_diagnostics,
    print_pool_summary,
)
from tests.integration.conftest import requires_all_credentials, requires_databricks


@pytest.mark.integration
class TestLightweightWorkflow:
    """Lightweight/quick workflow modes with real LLM and search."""

    # ------------------------------------------------------------------
    # 1. search_and_summarize: background search + synthesizer
    # ------------------------------------------------------------------

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_search_and_summarize_fast(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Background agent searches, synthesizer produces report — fast path.

        Validates:
        - web_search tool is called at least once
        - A report is generated with meaningful content (len > 50)
        - BackgroundCompletedEvent or AgentOutputEvent is emitted
        - Total wall-clock time stays under 120 seconds
        """
        definition = load_workflow(str(examples_dir / "search_and_summarize.yaml"))

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is the current status of the Mars Perseverance rover mission?"
            },
            tool_registry=tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- Assert web_search was invoked --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        search_calls = [tc for tc in tool_calls if tc.tool_name == "web_search"]
        assert len(search_calls) >= 1, (
            f"Expected at least one web_search ToolCallEvent, got {len(search_calls)}. "
            f"All tool calls: {[tc.tool_name for tc in tool_calls]}"
        )

        # -- Assert report was produced --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report in state['report']"
        report_str = str(report)
        assert len(report_str) > 50, (
            f"Report too short ({len(report_str)} chars). Preview: {report_str[:200]}"
        )

        # -- Assert domain events emitted --
        bg_events = [e for e in events if isinstance(e, BackgroundCompletedEvent)]
        output_events = [e for e in events if isinstance(e, AgentOutputEvent)]
        assert len(bg_events) >= 1 or len(output_events) >= 1, (
            "Expected BackgroundCompletedEvent or AgentOutputEvent to be emitted. "
            f"Event types seen: {list(event_summary(events).keys())}"
        )

        # -- Assert timing --
        assert elapsed < 120, (
            f"Search-and-summarize took {elapsed:.1f}s, expected < 120s"
        )

        # -- Print diagnostics --
        counts = event_summary(events)
        print("\n=== search_and_summarize_fast ===")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Report length: {len(report_str)} chars")
        print(f"Tool calls: {len(tool_calls)} (web_search: {len(search_calls)})")
        print(f"Event counts: {counts}")
        print(f"Report preview: {report_str[:300]}")
        print_pool_summary(state)
        print_event_timeline(events)

    # ------------------------------------------------------------------
    # 2. single_agent: coordinator classifies queries of varying complexity
    # ------------------------------------------------------------------

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_single_agent_coordinator_classifies(
        self,
        llm_client: FrameworkLLMClient,
        examples_dir: Path,
    ) -> None:
        """Coordinator classifies three queries of varying complexity.

        Runs sequentially (each ~5-10s) and verifies that
        CoordinatorClassifiedEvent is emitted for every query.
        """
        queries = [
            ("What is 5+5?", "simple"),
            ("Explain quantum computing", "moderate"),
            (
                "Compare transformer architectures with state-space models "
                "for long-context tasks",
                "complex",
            ),
        ]

        definition = load_workflow(str(examples_dir / "single_agent.yaml"))

        results: list[dict[str, Any]] = []

        for query, expected_band in queries:
            t0 = time.monotonic()
            state, events = await run_workflow(
                definition,
                llm_client,
                initial_state={"query": query},
            )
            elapsed = time.monotonic() - t0

            # -- Assert CoordinatorClassifiedEvent emitted --
            classified = [
                e for e in events if isinstance(e, CoordinatorClassifiedEvent)
            ]
            assert len(classified) >= 1, (
                f"CoordinatorClassifiedEvent not emitted for query: {query!r}. "
                f"Event types: {list(event_summary(events).keys())}"
            )

            ce = classified[0]
            coordination = state.get("coordination")

            results.append({
                "query": query,
                "expected_band": expected_band,
                "complexity": ce.complexity,
                "is_simple": ce.is_simple,
                "elapsed": elapsed,
                "coordination": coordination,
            })

        # -- Print classification results --
        print("\n=== single_agent_coordinator_classifies ===")
        for r in results:
            print(
                f"  [{r['expected_band']:>8s}] "
                f"complexity={r['complexity']:<10s} "
                f"is_simple={r['is_simple']!s:<6s} "
                f"({r['elapsed']:.1f}s) "
                f"query={r['query'][:60]}"
            )

    # ------------------------------------------------------------------
    # 3. simple_research: full lifecycle with plan_and_execute
    # ------------------------------------------------------------------

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_simple_research_pipeline_full_lifecycle(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Full simple_research pipeline: coordinator -> background -> P&E -> synthesizer.

        Validates the complete lifecycle:
        - CoordinatorClassifiedEvent is emitted
        - PlanCreatedEvent is emitted with steps
        - At least 1 ItemStartedEvent and ItemCompletedEvent
        - At least 2 ToolCallEvents (searching)
        - Report has substance (>= 300 chars, no failure phrases)
        - WorkflowCompletedEvent with duration_ms > 0
        """
        definition = load_workflow(str(examples_dir / "simple_research.yaml"))

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the environmental impacts of lithium mining "
                    "for EV batteries?"
                )
            },
            tool_registry=tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- CoordinatorClassifiedEvent --
        classified = [e for e in events if isinstance(e, CoordinatorClassifiedEvent)]
        assert len(classified) >= 1, (
            "CoordinatorClassifiedEvent not emitted. "
            f"Event types: {list(event_summary(events).keys())}"
        )

        # -- PlanCreatedEvent with steps --
        plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
        assert len(plan_events) >= 1, (
            "PlanCreatedEvent not emitted. "
            f"Event types: {list(event_summary(events).keys())}"
        )
        plan = plan_events[0]
        assert len(plan.steps) >= 1, (
            f"Plan should have at least 1 step, got {len(plan.steps)}"
        )

        # -- Item lifecycle events --
        items_started = [e for e in events if isinstance(e, ItemStartedEvent)]
        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        assert len(items_started) >= 1, (
            f"Expected >= 1 ItemStartedEvent, got {len(items_started)}"
        )
        assert len(items_completed) >= 1, (
            f"Expected >= 1 ItemCompletedEvent, got {len(items_completed)}"
        )

        # -- Tool calls (searching) --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 2, (
            f"Expected >= 2 ToolCallEvents, got {len(tool_calls)}. "
            f"Tools called: {[tc.tool_name for tc in tool_calls]}"
        )

        # -- Report substance --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=300)

        # -- WorkflowCompletedEvent --
        completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed) == 1, (
            f"Expected exactly 1 WorkflowCompletedEvent, got {len(completed)}"
        )
        assert completed[0].duration_ms > 0, (
            f"WorkflowCompletedEvent.duration_ms should be > 0, "
            f"got {completed[0].duration_ms}"
        )

        # -- Print full diagnostics --
        counts = event_summary(events)
        print("\n=== simple_research_pipeline_full_lifecycle ===")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Report length: {len(report_str)} chars")
        print(f"Plan steps: {len(plan.steps)}")
        print(f"Items started/completed: {len(items_started)}/{len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Workflow duration: {completed[0].duration_ms / 1000:.1f}s")
        print(f"Event counts: {counts}")
        print(f"\nReport preview:\n{report_str[:500]}")
        print_full_diagnostics(events, state)
