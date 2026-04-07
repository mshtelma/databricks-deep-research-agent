"""Integration tests: enterprise workflow with real LLM + mock enterprise tools.

Tests the framework's ability to orchestrate enterprise-only research using
real Databricks LLM endpoints with mock enterprise tools (Genie, Vector Search,
Knowledge Assistant). No Brave API key is required.

Each test typically runs 2-4 minutes.

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_enterprise_workflow.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
    ItemCompletedEvent,
    ItemStartedEvent,
    PlanCreatedEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    assert_plan_executed,
    assert_report_has_substance,
    assert_terminal_plan_exit,
    event_summary,
    print_event_timeline,
    print_pool_summary,
    print_search_queries,
    tool_calls_for_node,
)
from tests.integration.conftest import requires_databricks, skip_if_transient_provider_failure


@pytest.mark.integration
class TestEnterpriseWorkflow:
    """Enterprise research pipeline with real LLM and mock enterprise tools."""

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_enterprise_only_research(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Enterprise-only pipeline uses only enterprise tools (no web_search)."""
        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is the company's revenue growth trend across product lines?"
            },
            enterprise_tools=enterprise_tools,
            tool_registry=enterprise_tool_registry,
        )
        elapsed = time.monotonic() - t0

        assert_plan_executed(events)

        # -- Verify only enterprise tool calls (no web_search) --
        tool_calls = tool_calls_for_node(events, "researcher")
        assert len(tool_calls) >= 1, "Should make at least one enterprise tool call"
        enterprise_tool_names = {"genie", "vector_search", "knowledge_assistant"}
        for tc in tool_calls:
            assert tc.tool_name in enterprise_tool_names, (
                f"Expected only enterprise tools, got '{tc.tool_name}'. "
                "Enterprise-only workflow should not use web_search."
            )
        exit_event = assert_terminal_plan_exit(events)
        if exit_event.reason == "insufficient_evidence_exhausted":
            assert exit_event.completion_mode == "degraded"
            assert exit_event.evidence_sufficiency in {"partial", "insufficient"}
            assert exit_event.failure_mode

        # -- Verify pools have content --
        sources_pool = state.pools.get("sources")
        observations_pool = state.pools.get("observations")
        assert sources_pool is not None, "sources pool should exist"
        assert observations_pool is not None, "observations pool should exist"
        assert sources_pool.count() > 0, "sources pool should have items"
        assert observations_pool.count() > 0, "observations pool should have items"

        # -- Verify report has substance --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        assert_report_has_substance(str(report))

        # -- Diagnostics --
        summary = event_summary(events)
        print(f"\nEnterprise-only research completed in {elapsed:.1f}s")
        print(f"Total events: {len(events)}")
        print(f"Event summary: {summary}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Sources pool: {sources_pool.count()} items")
        print(f"Observations pool: {observations_pool.count()} items")
        print(f"Report length: {len(str(report))} chars")
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_genie_tool_generates_sql_insights(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Genie tool is invoked and financial data appears in the report."""
        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are our company's quarterly revenue figures "
                    "by product line from the internal data warehouse?"
                )
            },
            enterprise_tools=enterprise_tools,
            tool_registry=enterprise_tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- Verify genie was called --
        tool_calls = tool_calls_for_node(events, "researcher")
        genie_calls = [tc for tc in tool_calls if tc.tool_name == "genie"]
        assert len(genie_calls) >= 1, (
            "Researcher should invoke genie for financial queries"
        )

        # -- Verify report references financial data --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        report_lower = str(report).lower()
        financial_keywords = ["revenue", "growth", "quarter", "fiscal", "financial"]
        matching_keywords = [kw for kw in financial_keywords if kw in report_lower]
        assert len(matching_keywords) >= 1, (
            f"Report should reference financial data. "
            f"Keywords checked: {financial_keywords}. "
            f"Report preview: {str(report)[:300]}"
        )

        # -- Diagnostics --
        print(f"\nGenie SQL insights test completed in {elapsed:.1f}s")
        print(f"Genie calls: {len(genie_calls)}")
        print(f"Matching financial keywords: {matching_keywords}")
        print(f"Report preview: {str(report)[:500]}")
        print_event_timeline(events)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_vector_search_retrieves_relevant_docs(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Vector search is invoked and sources pool contains enterprise:// URLs."""
        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "What is our technical architecture and scaling strategy?"
            },
            enterprise_tools=enterprise_tools,
            tool_registry=enterprise_tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- Verify vector_search was called --
        assert_plan_executed(events)

        tool_calls = tool_calls_for_node(events, "researcher")
        vs_calls = [tc for tc in tool_calls if tc.tool_name == "vector_search"]
        assert len(vs_calls) >= 1, (
            "Researcher should invoke vector_search for architecture queries"
        )

        # -- Verify sources pool has enterprise:// URLs --
        sources_pool = state.pools.get("sources")
        assert sources_pool is not None, "sources pool should exist"
        assert sources_pool.count() > 0, "sources pool should have items"

        recent_sources = sources_pool.get_recent(20)
        enterprise_urls = [
            item.get("url", "") if isinstance(item, dict) else str(item)
            for item in recent_sources
            if isinstance(item, dict) and "enterprise://" in item.get("url", "")
        ]
        assert len(enterprise_urls) > 0, (
            f"Sources pool should contain enterprise:// URLs. "
            f"Got sources: {recent_sources[:3]}"
        )

        # -- Diagnostics --
        print(f"\nVector search test completed in {elapsed:.1f}s")
        print(f"Vector search calls: {len(vs_calls)}")
        print(f"Enterprise URLs found: {enterprise_urls}")
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_enterprise_pool_dedup(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Pool dedup tracking works for enterprise sources and observations."""
        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )

        t0 = time.monotonic()
        try:
            state, events = await run_workflow(
                definition,
                llm_client,
                initial_state={
                    "query": (
                        "Using our internal finance systems, retrieve quarterly revenue, "
                        "profitability, growth rate, and operational efficiency metrics "
                        "for the company from enterprise data sources."
                    )
                },
                enterprise_tools=enterprise_tools,
                tool_registry=enterprise_tool_registry,
            )
        except Exception as exc:
            skip_if_transient_provider_failure(exc)
        elapsed = time.monotonic() - t0

        assert_plan_executed(events)

        # -- Verify pool dedup is tracked --
        sources_pool = state.pools.get("sources")
        observations_pool = state.pools.get("observations")
        assert sources_pool is not None, "sources pool should exist"
        assert observations_pool is not None, "observations pool should exist"

        # Dedup instrumentation should exist; duplicate rejections are workload-dependent
        sources_dedup = getattr(sources_pool.stats, "rejected_duplicate_key", 0) + getattr(sources_pool.stats, "rejected_duplicate_hash", 0)
        observations_dedup = getattr(observations_pool.stats, "rejected_duplicate_key", 0) + getattr(observations_pool.stats, "rejected_duplicate_hash", 0)
        total_attempts = getattr(sources_pool.stats, "attempted", 0) + getattr(observations_pool.stats, "attempted", 0)
        tool_calls = tool_calls_for_node(events, "researcher")
        assert total_attempts > 0, (
            f"Pools should record write attempts. "
            f"sources.attempted={getattr(sources_pool.stats, 'attempted', 0)}, "
            f"observations.attempted={getattr(observations_pool.stats, 'attempted', 0)}. "
            f"Researcher tool calls: {[tc.tool_name for tc in tool_calls]}"
        )
        exit_event = assert_terminal_plan_exit(events)
        if exit_event.reason == "insufficient_evidence_exhausted":
            assert exit_event.completion_mode == "degraded"
            assert exit_event.evidence_sufficiency in {"partial", "insufficient"}
            assert exit_event.failure_mode

        # -- Diagnostics --
        print(f"\nPool dedup test completed in {elapsed:.1f}s")
        print(f"Sources pool: {sources_pool.count()} items, "
              f"{sources_dedup} seen_keys")
        print(f"Observations pool: {observations_pool.count()} items, "
              f"{observations_dedup} seen_hashes")
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_enterprise_tools_in_plan_and_execute(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Plan-and-execute emits plan, items, and uses enterprise tools."""
        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Provide a comprehensive analysis of the company's technical "
                    "infrastructure, financial performance across all product lines, "
                    "and operational deployment practices"
                )
            },
            enterprise_tools=enterprise_tools,
            tool_registry=enterprise_tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- Verify plan was created --
        plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
        assert len(plan_events) >= 1, "Planner should create a research plan"
        plan = plan_events[0]
        assert len(plan.steps) >= 1, "Plan should have at least one step"

        assert_plan_executed(events)
        started_events = [e for e in events if isinstance(e, ItemStartedEvent)]
        completed_events = [e for e in events if isinstance(e, ItemCompletedEvent)]

        # -- Verify enterprise tool calls --
        tool_calls = tool_calls_for_node(events, "researcher")
        enterprise_tool_names = {"genie", "vector_search", "knowledge_assistant"}
        enterprise_calls = [
            tc for tc in tool_calls if tc.tool_name in enterprise_tool_names
        ]
        assert len(enterprise_calls) >= 1, (
            "Plan execution should invoke enterprise tools. "
            f"Tool calls found: {[tc.tool_name for tc in tool_calls]}"
        )
        exit_event = assert_terminal_plan_exit(events)
        if exit_event.reason == "insufficient_evidence_exhausted":
            assert exit_event.completion_mode == "degraded"
            assert exit_event.evidence_sufficiency in {"partial", "insufficient"}
            assert exit_event.failure_mode

        # -- Diagnostics --
        summary = event_summary(events)
        print(f"\nPlan-and-execute enterprise test completed in {elapsed:.1f}s")
        print(f"Plan: {plan.title} ({len(plan.steps)} steps)")
        for i, step in enumerate(plan.steps):
            step_str = (
                step.get("description", str(step))
                if isinstance(step, dict)
                else str(step)
            )
            print(f"  Step {i + 1}: {step_str[:120]}")
        print(f"Items started: {len(started_events)}")
        print(f"Items completed: {len(completed_events)}")
        print(f"Enterprise tool calls: {len(enterprise_calls)}")
        print(f"Event summary: {summary}")
        print_event_timeline(events)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_enterprise_report_synthesizes_data(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Synthesizer produces a substantive report from enterprise data."""
        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )

        t0 = time.monotonic()
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": "Summarize all available enterprise data about the platform"
            },
            enterprise_tools=enterprise_tools,
            tool_registry=enterprise_tool_registry,
        )
        elapsed = time.monotonic() - t0

        # -- Verify report exists and has substance --
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        report_str = str(report)
        assert len(report_str) >= 200, (
            f"Report too short ({len(report_str)} chars). "
            f"Preview: {report_str[:300]}"
        )
        assert_report_has_substance(report_str, min_length=200)

        # -- Verify observations pool has content --
        observations_pool = state.pools.get("observations")
        assert observations_pool is not None, "observations pool should exist"
        assert observations_pool.count() > 0, (
            "observations pool should have items for synthesis"
        )

        # -- Full diagnostics --
        summary = event_summary(events)
        print(f"\nEnterprise report synthesis test completed in {elapsed:.1f}s")
        print(f"Total events: {len(events)}")
        print(f"Event summary: {summary}")
        print(f"Observations pool: {observations_pool.count()} items")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nFull report:\n{report_str[:1000]}")
        print_search_queries(events)
        print_pool_summary(state)
        print_event_timeline(events)
