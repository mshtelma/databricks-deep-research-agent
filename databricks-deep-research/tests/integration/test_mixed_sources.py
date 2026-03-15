"""Integration tests: mixed-source research with real LLM + web + mock enterprise tools.

Tests the framework's ability to:
- Execute a workflow that combines web search (Brave) with enterprise tools (Genie, Vector Search)
- Route tool calls appropriately based on query intent
- Deduplicate items across source types via pool dedup tracking
- Synthesize a report that incorporates findings from both web and enterprise sources

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_mixed_sources.py -v -s
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    assert_plan_executed,
    assert_report_has_substance,
    event_summary,
    print_event_timeline,
    print_pool_summary,
    print_search_queries,
    tool_calls_for_node,
)
from tests.integration.conftest import requires_all_credentials


@pytest.mark.integration
class TestMixedSources:
    """Mixed web + enterprise source research with real Brave Search + LLM."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_mixed_web_and_enterprise_research(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Workflow uses both web_search and enterprise tools (genie/vector_search)."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "mixed_sources.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What does our internal revenue data show about "
                    "cloud computing growth, and how does it compare "
                    "to public industry trends?"
                ),
            },
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        # -- Verify research-cycle tool calls include both web and enterprise sources --
        tool_calls = tool_calls_for_node(events, "researcher")
        tool_names = {tc.tool_name for tc in tool_calls}

        web_calls = [tc for tc in tool_calls if tc.tool_name == "web_search"]
        enterprise_calls = [
            tc for tc in tool_calls
            if tc.tool_name in ("genie", "vector_search", "knowledge_assistant")
        ]

        assert len(web_calls) >= 1, (
            f"Expected at least one web_search call; got tool names: {tool_names}"
        )
        assert len(enterprise_calls) >= 1, (
            f"Expected at least one enterprise tool call (genie/vector_search); "
            f"got tool names: {tool_names}"
        )

        assert_plan_executed(events)

        # -- Verify sources pool has items from both public and enterprise families --
        sources_pool = state.pools.get("sources")
        assert sources_pool is not None, "Workflow should create a 'sources' pool"
        assert sources_pool.count() > 0, "Sources pool should contain items"
        pooled_sources = sources_pool.snapshot()
        web_sources = [
            item for item in pooled_sources
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://"))
        ]
        enterprise_sources = [
            item for item in pooled_sources
            if isinstance(item, dict) and "enterprise://" in str(item.get("url", ""))
        ]
        assert web_sources, "Sources pool should contain public web evidence"
        assert enterprise_sources, "Sources pool should contain enterprise evidence"

        # -- Verify report has substance --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        assert_report_has_substance(str(report))

        # -- Diagnostics --
        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Total events: {len(events)}")
        print(f"Tool calls: {len(tool_calls)} (web: {len(web_calls)}, "
              f"enterprise: {len(enterprise_calls)})")
        print(f"Sources pool: {sources_pool.count()} items")
        print(f"Report length: {len(str(report))} chars")
        print(f"\nEvent summary: {event_summary(events)}")
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_source_diversity_in_report(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Report incorporates content from both enterprise mock data and web search."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "mixed_sources.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Using our internal data warehouse metrics and deployment "
                    "pipeline documentation, analyze our cloud architecture "
                    "and compare it against public industry best practices "
                    "for microservices"
                ),
            },
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        assert_plan_executed(events)

        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=300)

        researcher_tool_calls = tool_calls_for_node(events, "researcher")
        assert any(tc.tool_name == "web_search" for tc in researcher_tool_calls)
        assert any(
            tc.tool_name in {"vector_search", "genie", "knowledge_assistant"}
            for tc in researcher_tool_calls
        )

        sources_pool = state.pools.get("sources")
        assert sources_pool is not None and sources_pool.count() > 0
        pooled_sources = sources_pool.snapshot()
        assert any(
            isinstance(item, dict) and "enterprise://" in str(item.get("url", ""))
            for item in pooled_sources
        ), "Expected enterprise evidence in sources pool"
        assert any(
            isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://"))
            for item in pooled_sources
        ), "Expected public web evidence in sources pool"

        # -- Diagnostics --
        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_dedup_across_source_types(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Pool dedup tracking works across mixed web and enterprise sources."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "mixed_sources.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What is the current state of our system architecture "
                    "and how does it compare to modern cloud-native patterns?"
                ),
            },
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        assert_plan_executed(events)

        # -- Verify pool dedup tracking has entries --
        sources_pool = state.pools.get("sources")
        observations_pool = state.pools.get("observations")

        has_dedup_tracking = False

        if sources_pool is not None:
            sources_seen_keys = len(sources_pool.seen_keys)
            print(f"\nSources pool: {sources_pool.count()} items, "
                  f"{sources_seen_keys} seen_keys")
            if sources_seen_keys > 0:
                has_dedup_tracking = True

        if observations_pool is not None:
            observations_seen_hashes = len(observations_pool.seen_hashes)
            print(f"Observations pool: {observations_pool.count()} items, "
                  f"{observations_seen_hashes} seen_hashes")
            if observations_seen_hashes > 0:
                has_dedup_tracking = True

        assert has_dedup_tracking, (
            "At least one pool should have dedup tracking entries "
            "(sources.seen_keys or observations.seen_hashes)"
        )
        assert sources_pool is not None and sources_pool.count() > 0
        pooled_sources = sources_pool.snapshot()
        assert any(
            isinstance(item, dict) and "enterprise://" in str(item.get("url", ""))
            for item in pooled_sources
        ), "Expected enterprise evidence in deduplicated sources pool"
        assert any(
            isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://"))
            for item in pooled_sources
        ), "Expected public web evidence in deduplicated sources pool"

        # -- Diagnostics --
        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"\nEvent summary: {event_summary(events)}")
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_researcher_selects_appropriate_tools(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Researcher makes tool calls when 4 tools are available (web + enterprise)."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "mixed_sources.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What internal documentation exists about our "
                    "deployment pipeline?"
                ),
            },
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        assert_plan_executed(events)

        # -- Verify researcher tool calls were made --
        tool_calls = tool_calls_for_node(events, "researcher")
        assert len(tool_calls) >= 1, (
            f"Expected at least 1 tool call; got {len(tool_calls)}"
        )
        assert any(
            tc.tool_name in {"vector_search", "genie", "knowledge_assistant"}
            for tc in tool_calls
        ), "Internal documentation query should use an enterprise retrieval tool"

        # -- Print tool call distribution --
        tool_counts: Counter[str] = Counter()
        for tc in tool_calls:
            tool_counts[tc.tool_name] += 1

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Total tool calls: {len(tool_calls)}")
        print("\nTool call distribution:")
        for tool_name, count in tool_counts.most_common():
            print(f"  {tool_name}: {count}")

        print(f"\nEvent summary: {event_summary(events)}")
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)
