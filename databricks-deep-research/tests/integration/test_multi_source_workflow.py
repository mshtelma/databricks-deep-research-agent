"""Integration test: multi-source research with per-lane pools.

Three parallel research lanes (earnings VS, transcript VS, web search),
each writing to its own observation pool with per-lane section synthesis,
followed by a final report synthesizer that merges all sections.

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- BRAVE_API_KEY for web search lane
- Access to VS indexes configured via FRAMEWORK_TEST_VS_INDEX and
  FRAMEWORK_TEST_TRANSCRIPT_VS_INDEX

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_multi_source_workflow.py -v -s --timeout=600
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
    NodeStartedEvent,
    ToolCallEvent,
    WorkflowCompletedEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    assert_report_has_substance,
    event_summary,
    print_citation_details,
    print_event_timeline,
    print_pool_summary,
    print_search_queries,
    print_verification_summary,
)
from tests.integration.conftest import (
    BraveSearchAdapter,
    requires_all_credentials,
)

_QUERY = "What are the recent Kroger earnings results and financial outlook?"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factory_context(
    workspace_client: Any,
    brave_adapter: BraveSearchAdapter,
) -> ToolFactoryContext:
    """ToolFactoryContext with real workspace client and Brave search adapter."""
    return ToolFactoryContext(
        workspace_client=workspace_client,
        search_client=brave_adapter,
    )


# ---------------------------------------------------------------------------
# Helper: run the workflow once (shared logic across tests)
# ---------------------------------------------------------------------------


async def _run_multi_source(
    llm_client: FrameworkLLMClient,
    factory_context: ToolFactoryContext,
    examples_dir: Path,
) -> tuple[Any, list[Any], float]:
    """Execute the multi_source_research workflow and return (state, events, elapsed)."""
    definition = load_workflow(examples_dir / "multi_source_research.yaml")

    t0 = time.monotonic()
    state, events = await run_workflow(
        definition,
        llm_client,
        initial_state={"query": _QUERY},
        factory_context=factory_context,
        strict_tool_resolution=True,
    )
    elapsed = time.monotonic() - t0
    return state, events, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiSourceWorkflow:
    """Multi-source parallel research with per-lane pools and section synthesis."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_all_three_lanes_execute(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext,
        examples_dir: Path,
    ) -> None:
        """All 6 agent nodes start and the workflow completes."""
        state, events, elapsed = await _run_multi_source(
            llm_client, factory_context, examples_dir
        )

        node_started = [e for e in events if isinstance(e, NodeStartedEvent)]
        started_ids = {e.node_id for e in node_started}

        expected_nodes = {
            "researcher_earnings",
            "earnings_synthesizer",
            "researcher_transcripts",
            "transcript_synthesizer",
            "researcher_web",
            "web_synthesizer",
            "final_synthesizer",
        }
        missing = expected_nodes - started_ids
        assert not missing, (
            f"Expected all agent nodes to start. Missing: {missing}. "
            f"Started: {started_ids}"
        )

        completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed) == 1, (
            f"Expected exactly 1 WorkflowCompletedEvent, got {len(completed)}"
        )

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Started node_ids: {sorted(started_ids)}")
        print(f"Event summary: {event_summary(events)}")
        print_event_timeline(events)
        print_search_queries(events)
        print_verification_summary(state, events)
        print_citation_details(state, events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_per_lane_pools_populated(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext,
        examples_dir: Path,
    ) -> None:
        """Each per-lane observation pool and the shared sources pool have items."""
        state, events, elapsed = await _run_multi_source(
            llm_client, factory_context, examples_dir
        )

        for pool_name in ("earnings_observations", "transcript_observations", "web_observations"):
            pool = state.pools.get(pool_name)
            assert pool is not None, f"Pool '{pool_name}' should exist"
            assert pool.count() > 0, (
                f"Pool '{pool_name}' should have items, got {pool.count()}"
            )

        sources_pool = state.pools.get("sources")
        assert sources_pool is not None, "Shared sources pool should exist"
        assert sources_pool.count() > 0, (
            f"Sources pool should have items, got {sources_pool.count()}"
        )

        print(f"\nElapsed: {elapsed:.1f}s")
        print_pool_summary(state)
        print_search_queries(events)
        print_verification_summary(state, events)
        print_citation_details(state, events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_section_summaries_produced(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext,
        examples_dir: Path,
    ) -> None:
        """Each lane produces a non-trivial section summary in state."""
        state, events, elapsed = await _run_multi_source(
            llm_client, factory_context, examples_dir
        )

        for key in ("earnings_section", "transcript_section", "web_section"):
            section = state.get(key)
            assert section is not None, f"State key '{key}' should exist"
            section_str = str(section)
            assert len(section_str) >= 50, (
                f"'{key}' is too short ({len(section_str)} chars). "
                f"Preview: {section_str[:200]}"
            )

        print(f"\nElapsed: {elapsed:.1f}s")
        for key in ("earnings_section", "transcript_section", "web_section"):
            section_str = str(state.get(key))
            print(f"\n--- {key} ({len(section_str)} chars) ---")
            print(section_str)
        print_search_queries(events)
        print_verification_summary(state, events)
        print_citation_details(state, events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_tool_diversity(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext,
        examples_dir: Path,
    ) -> None:
        """All three tool types (earnings_vs, transcript_vs, web_search) appear in events."""
        state, events, elapsed = await _run_multi_source(
            llm_client, factory_context, examples_dir
        )

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_names_used = {tc.tool_name for tc in tool_calls}

        expected_tools = {"earnings_vs", "transcript_vs", "web_search"}
        missing = expected_tools - tool_names_used
        assert not missing, (
            f"Expected all tool types to be called. Missing: {missing}. "
            f"Tools used: {tool_names_used}"
        )

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Tool names used: {sorted(tool_names_used)}")
        tool_dist: dict[str, int] = {}
        for tc in tool_calls:
            tool_dist[tc.tool_name] = tool_dist.get(tc.tool_name, 0) + 1
        print(f"Tool distribution: {tool_dist}")
        print_search_queries(events)
        print_verification_summary(state, events)
        print_citation_details(state, events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_final_report_has_substance(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext,
        examples_dir: Path,
    ) -> None:
        """Final report is >= 300 chars and contains financial + transcript keywords."""
        state, events, elapsed = await _run_multi_source(
            llm_client, factory_context, examples_dir
        )

        report = state.get("report")
        assert report is not None, "Final synthesizer should produce a report"
        report_str = str(report)

        assert len(report_str) >= 300, (
            f"Report too short ({len(report_str)} chars). Preview: {report_str[:200]}"
        )
        assert_report_has_substance(report_str, min_length=300)

        lowered = report_str.lower()
        # Financial keywords from earnings lane
        financial_keywords = ("earnings", "revenue", "profit", "financial", "quarter", "fiscal")
        has_financial = any(kw in lowered for kw in financial_keywords)
        assert has_financial, (
            f"Report should contain financial keywords. "
            f"Checked: {financial_keywords}. Preview: {report_str[:400]}"
        )

        # Transcript keywords from transcript lane
        transcript_keywords = (
            "transcript", "call", "presentation", "executive", "ceo", "cfo",
            "guidance", "outlook", "commentary", "management",
        )
        has_transcript = any(kw in lowered for kw in transcript_keywords)
        assert has_transcript, (
            f"Report should contain transcript-related keywords. "
            f"Checked: {transcript_keywords}. Preview: {report_str[:400]}"
        )

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Report length: {len(report_str)} chars")
        print("\n--- FULL REPORT ---")
        print(report_str)
        print_pool_summary(state)
        print_search_queries(events)
        print_verification_summary(state, events)
        print_citation_details(state, events)
