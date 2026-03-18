"""Integration tests: conditional branching workflow with real LLM.

Tests the framework's conditional node:
- Coordinator classifies query complexity
- Simple queries take a direct synthesis branch (no research)
- Complex queries trigger full background + plan_and_execute + synthesis

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_conditional_workflow.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
    BranchSelectedEvent,
    ItemStartedEvent,
    NodeStartedEvent,
    StreamEvent,
    ToolCallEvent,
    WorkflowCompletedEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import event_summary, print_event_timeline
from tests.integration.conftest import requires_all_credentials


@pytest.mark.integration
class TestConditionalWorkflow:
    """Conditional branching: simple queries skip research, complex queries get full pipeline."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_simple_query_takes_simple_branch(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Simple query should be classified as simple; either branch is acceptable."""
        definition = load_workflow(str(examples_dir / "conditional_research.yaml"))

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": "What is 2+2?"},
            tool_registry=tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- BranchSelectedEvent must exist --
        branch_events = [e for e in events if isinstance(e, BranchSelectedEvent)]
        assert len(branch_events) >= 1, (
            "Conditional node must emit a BranchSelectedEvent"
        )

        branch_index = branch_events[0].branch_index
        print(f"\nBranch selected: {branch_index} ({branch_events[0].condition_summary})")

        if branch_index == 0:
            # Simple path: no research items should have started
            item_started = [e for e in events if isinstance(e, ItemStartedEvent)]
            assert len(item_started) == 0, (
                f"Simple branch should skip research, but {len(item_started)} items started"
            )
            print("  -> Simple path taken: no research items executed (as expected)")
        else:
            # LLM classified "What is 2+2?" as complex -- acceptable but unusual
            print("  -> Complex path taken (LLM classified simple query as complex)")

        # -- Report must exist either way --
        report = state.get("report")
        assert report is not None, "Both branches should produce a report"
        print(f"\nReport ({len(str(report))} chars): {str(report)[:300]}")

        # -- Diagnostics --
        summary = event_summary(events)
        print(f"\nEvent summary: {summary}")
        print(f"Duration: {elapsed:.1f}s")
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_complex_query_triggers_research(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Complex query must trigger full research pipeline with tools."""
        definition = load_workflow(str(examples_dir / "conditional_research.yaml"))

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Compare the economic impact of renewable energy adoption "
                    "across developing nations in Southeast Asia, including "
                    "policy frameworks, investment patterns, and technological readiness"
                ),
            },
            tool_registry=tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- Must branch to complex path --
        branch_events = [e for e in events if isinstance(e, BranchSelectedEvent)]
        assert len(branch_events) >= 1, "Must emit BranchSelectedEvent"
        assert branch_events[0].branch_index == 1, (
            f"Complex query should take branch 1 (complex path), "
            f"got branch {branch_events[0].branch_index}"
        )
        print(f"\nBranch selected: {branch_events[0].branch_index} "
              f"({branch_events[0].condition_summary})")

        # -- Research items must have been executed --
        item_started = [e for e in events if isinstance(e, ItemStartedEvent)]
        assert len(item_started) > 0, (
            "Complex path must execute research items (ItemStartedEvent)"
        )
        print(f"Research items started: {len(item_started)}")

        # -- Tools must have been called --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) > 0, (
            "Complex path must invoke tools (web_search/web_crawl)"
        )
        search_calls = [tc for tc in tool_calls if tc.tool_name == "web_search"]
        crawl_calls = [tc for tc in tool_calls if tc.tool_name == "web_crawl"]
        print(f"Tool calls: {len(tool_calls)} total "
              f"({len(search_calls)} search, {len(crawl_calls)} crawl)")

        # -- Report must have substance --
        report = state.get("report")
        assert report is not None, "Synthesizer must produce a report"
        report_str = str(report)
        assert len(report_str) > 200, (
            f"Report too short for complex query ({len(report_str)} chars). "
            f"Preview: {report_str[:300]}"
        )
        print(f"\nReport ({len(report_str)} chars): {report_str[:500]}")

        # -- Workflow completion --
        completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed) == 1
        print(f"\nDuration: {elapsed:.1f}s (event: {completed[0].duration_ms:.0f}ms)")

        # -- Diagnostics --
        summary = event_summary(events)
        print(f"\nEvent summary: {summary}")
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_conditional_produces_report_both_paths(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Both simple and complex paths must produce a report key in state."""
        definition = load_workflow(str(examples_dir / "conditional_research.yaml"))

        # -- Run simple query --
        t0 = time.monotonic()
        simple_state, simple_events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": "What is the capital of France?"},
            tool_registry=tool_registry,
        )
        simple_elapsed = time.monotonic() - t0

        simple_branch_events = [
            e for e in simple_events if isinstance(e, BranchSelectedEvent)
        ]
        simple_branch = (
            simple_branch_events[0].branch_index if simple_branch_events else None
        )
        print(f"\nSimple query branch: {simple_branch}")
        if simple_branch_events:
            print(f"  condition: {simple_branch_events[0].condition_summary}")
        print(f"  duration: {simple_elapsed:.1f}s")

        # -- Run complex query --
        t1 = time.monotonic()
        complex_state, complex_events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "Analyze the competitive landscape of cloud AI platforms in 2025",
            },
            tool_registry=tool_registry,
        )
        complex_elapsed = time.monotonic() - t1

        complex_branch_events = [
            e for e in complex_events if isinstance(e, BranchSelectedEvent)
        ]
        complex_branch = (
            complex_branch_events[0].branch_index if complex_branch_events else None
        )
        print(f"\nComplex query branch: {complex_branch}")
        if complex_branch_events:
            print(f"  condition: {complex_branch_events[0].condition_summary}")
        print(f"  duration: {complex_elapsed:.1f}s")

        # -- Both must produce a report --
        simple_report = simple_state.get("report")
        complex_report = complex_state.get("report")

        assert simple_report is not None, (
            "Simple path must produce a report key in state"
        )
        assert complex_report is not None, (
            "Complex path must produce a report key in state"
        )

        print(f"\nSimple report ({len(str(simple_report))} chars): "
              f"{str(simple_report)[:200]}")
        print(f"\nComplex report ({len(str(complex_report))} chars): "
              f"{str(complex_report)[:200]}")

        # -- Diagnostics for both --
        print("\n=== Simple query events ===")
        print(f"  Event summary: {event_summary(simple_events)}")
        print_event_timeline(simple_events)

        print("\n=== Complex query events ===")
        print(f"  Event summary: {event_summary(complex_events)}")
        print_event_timeline(complex_events)
