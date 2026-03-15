"""Integration tests: search workflow with real tools.

Tests the framework's ability to:
- Execute a multi-step YAML workflow with real LLM + Brave Search
- Use web_search tool via the ReAct loop
- Write results to pools and inject them into downstream agents
- Generate a synthesized report from search findings

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_search_workflow.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from databricks_deep_research import (
    FrameworkLLMClient,
    run_workflow,
)
from databricks_deep_research.events.types import (
    BackgroundCompletedEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkflowCompletedEvent,
)
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    print_event_timeline,
    print_pool_summary,
    print_search_queries,
)
from tests.integration.conftest import requires_all_credentials


@pytest.mark.integration
class TestSearchAndSummarizeWorkflow:
    """Search + synthesize pipeline with real Brave Search + LLM."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_search_and_summarize(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Background agent searches, synthesizer produces report."""
        definition = load_workflow(examples_dir / "search_and_summarize.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": "What are the latest advances in AI agents in 2025?"},
            tool_registry=tool_registry,
        )

        # Verify background completed
        bg_events = [e for e in events if isinstance(e, BackgroundCompletedEvent)]
        assert len(bg_events) >= 1, "Background agent should complete"
        assert bg_events[0].sources_discovered >= 0

        # Verify tool calls happened
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 1, "Should have at least one tool call (web_search)"
        search_calls = [tc for tc in tool_calls if tc.tool_name == "web_search"]
        assert len(search_calls) >= 1, "Should have web_search calls"

        # Verify tool results
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) >= 1, "Should have tool results"

        # Verify report was generated
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        assert len(str(report)) > 50, f"Report too short: {str(report)[:100]}"

        # Verify workflow completed
        completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].duration_ms > 0

        print(f"\nReport ({len(str(report))} chars):")
        print(str(report)[:500])
        print(f"\nTool calls: {len(tool_calls)}")
        print(f"Sources discovered: {bg_events[0].sources_discovered}")
        print(f"Duration: {completed[0].duration_ms / 1000:.1f}s")

        # -- Rich diagnostics --
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_search_produces_sources(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Verify web search finds real sources and they're surfaced in events."""
        definition = load_workflow(examples_dir / "search_and_summarize.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": "NVIDIA revenue Q4 2024 earnings"},
            tool_registry=tool_registry,
        )

        # Check that search returned actual results
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        source_counts = [e.source_count for e in result_events if e.source_count > 0]
        assert len(source_counts) > 0, "Search should discover sources"

        report = state.get("report")
        assert report is not None
        report_str = str(report).lower()
        assert any(
            term in report_str for term in ["nvidia", "revenue", "earning", "quarter"]
        ), f"Report should mention NVIDIA earnings: {report_str[:200]}"

        print(f"\nSources found via tool calls: {sum(source_counts)}")
        print(f"Report preview: {str(report)[:300]}")

        # -- Rich diagnostics --
        print_search_queries(events)
        print_pool_summary(state)
