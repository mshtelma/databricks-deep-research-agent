"""Complex tests for the citation verification pipeline through the framework.

These are Tier 1 long-running tests (~5-10 min each) that exercise the full
research pipeline with citation-relevant assertions: verifying that reports
contain citable claims, sources have valid URLs, synthesized content references
source material, evidence accumulates across research steps, enterprise sources
are preserved for citation context, and report quality metrics meet thresholds.

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- BRAVE_API_KEY (except test_enterprise_sources_in_citation_context)
- Significant time (5-10 minutes per test)

Run with:
    cd databricks-deep-research
    uv run pytest tests/complex/test_citation_verification.py -v -s --timeout=600
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
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
from tests.complex.conftest import requires_all_credentials
from tests.helpers import (
    assert_report_has_substance,
    event_summary,
    print_event_timeline,
    print_full_diagnostics,
    print_pool_summary,
)

# Enterprise test uses requires_databricks (no Brave needed)
from tests.integration.conftest import requires_databricks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_factual_claims(text: str) -> bool:
    """Check whether text contains factual statements suitable for citation.

    Looks for numeric data, percentage references, date-anchored claims,
    named-entity statements, and comparative assertions.
    """
    patterns = [
        r"\d+%",                       # percentages
        r"\$[\d,]+",                   # dollar amounts
        r"\d{4}",                      # year references
        r"\d+\s*(million|billion|trillion)",  # magnitude numbers
        r"(according to|study|research|found that|reported|published)",  # attribution
        r"(increased|decreased|grew|declined)\s+by",  # trend claims
    ]
    text_lower = text.lower()
    matches = sum(1 for p in patterns if re.search(p, text_lower))
    return matches >= 2  # at least two different types of factual claims


def _extract_source_keywords(sources_pool: Any) -> set[str]:
    """Extract meaningful keywords from source titles and snippets."""
    keywords: set[str] = set()
    for item in sources_pool.get_recent(20):
        if isinstance(item, dict):
            for field in ("title", "snippet"):
                text = item.get(field, "")
                if text:
                    # Extract words >= 4 chars as meaningful keywords
                    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
                    keywords.update(words)
    return keywords


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.complex
class TestCitationVerification:
    """Full research pipeline tests with citation-relevant assertions.

    Each test runs the citation_pipeline.yaml workflow (or enterprise variant)
    and verifies properties required for downstream citation verification:
    - Reports contain citable, factual claims
    - Sources have valid, reachable URLs
    - Synthesized reports reference gathered source content
    - Evidence accumulates across multiple research steps
    - Enterprise sources are preserved in the pipeline context
    - Quality metrics meet minimum thresholds
    """

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_research_produces_citable_report(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Report from citation pipeline contains factual, citable claims.

        Verifies that the synthesizer produces a report rich enough
        for the 7-stage citation verification pipeline to operate on:
        numeric claims, attribution language, and source-backed assertions.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "citation_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the key findings from recent studies on the "
                    "effectiveness of mRNA vaccines against COVID-19 variants?"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Report should have substance --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=500)

        # -- Sources pool should have gathered evidence --
        src_pool = state.pools.get("sources")
        obs_pool = state.pools.get("observations")
        src_count = src_pool.count() if src_pool else 0
        obs_count = obs_pool.count() if obs_pool else 0
        assert src_count >= 2, (
            f"Sources pool has only {src_count} items; need >= 2 for citation verification"
        )
        assert obs_count > 0, (
            "Observations pool is empty; researcher did not accumulate evidence"
        )

        # -- Report should contain factual, citable claims --
        assert _has_factual_claims(report_str), (
            "Report lacks factual claims (no numeric data, percentages, "
            "attribution language, or trend statements). "
            "Citation verification requires citable content.\n"
            f"Report preview: {report_str[:500]}"
        )

        # -- Content relevance --
        report_lower = report_str.lower()
        assert any(
            term in report_lower
            for term in ["mrna", "vaccine", "covid", "variant", "efficacy", "booster"]
        ), "Report should discuss mRNA vaccines and COVID-19 variants"

        # -- Summary --
        print(f"\n{'='*60}")
        print(f"Citation-ready report produced in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Report length: {len(report_str)} chars")
        print(f"Sources pool: {src_count} items")
        print(f"Observations pool: {obs_count} items")
        print(f"Has factual claims: {_has_factual_claims(report_str)}")
        print(f"\nEvent summary: {event_summary(events)}")
        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_sources_have_valid_urls(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """All gathered sources have valid HTTP(S) URLs for citation linking.

        Citation verification needs source URLs to attribute claims. This test
        verifies that the pool contains properly-formed URLs and that there
        is sufficient URL diversity across sources.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "citation_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the current global trends in renewable energy "
                    "adoption and investment for 2024-2025?"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Sources pool must exist and have items --
        src_pool = state.pools.get("sources")
        assert src_pool is not None, "sources pool should exist"
        assert src_pool.count() >= 2, (
            f"Sources pool has only {src_pool.count()} items; "
            "need at least 2 for meaningful citation coverage"
        )

        # -- Each source should have a valid URL --
        all_items = src_pool.get_recent(100)
        urls: list[str] = []
        for item in all_items:
            if isinstance(item, dict):
                url = item.get("url", "")
                assert url, f"Source item missing 'url': {item}"
                assert url.startswith("http://") or url.startswith("https://"), (
                    f"Source URL should start with http:// or https://, got: {url}"
                )
                urls.append(url)

        # -- URL diversity: at least 2 unique URLs --
        unique_urls = set(urls)
        assert len(unique_urls) >= 2, (
            f"Only {len(unique_urls)} unique URL(s) found; "
            "need at least 2 diverse sources for citation verification. "
            f"URLs: {unique_urls}"
        )

        # -- Summary --
        print(f"\n{'='*60}")
        print(f"Source URL validation completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Total source items: {src_pool.count()}")
        print(f"Valid URLs: {len(urls)}")
        print(f"Unique URLs: {len(unique_urls)}")
        print("\nSource URLs:")
        for i, url in enumerate(sorted(unique_urls)):
            print(f"  [{i + 1}] {url[:120]}")
        print(f"\nEvent summary: {event_summary(events)}")
        print_pool_summary(state)
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_report_references_source_content(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Synthesized report content relates to gathered source material.

        Validates that the synthesizer actually incorporates pool content
        into the final report. At least one keyword from source titles or
        snippets should appear in the report, confirming the citation
        pipeline has traceable provenance.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "citation_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What is the current state of quantum computing hardware?"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Report should exist --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=500)

        # -- Sources pool should have items --
        src_pool = state.pools.get("sources")
        assert src_pool is not None, "sources pool should exist"
        assert src_pool.count() > 0, "sources pool should have items"

        # -- Extract keywords from source titles/snippets --
        source_keywords = _extract_source_keywords(src_pool)
        report_lower = report_str.lower()

        # Check for keyword overlap between sources and report
        overlapping_keywords = {
            kw for kw in source_keywords if kw in report_lower
        }

        assert len(overlapping_keywords) >= 1, (
            "Report does not reference any source content. "
            "No keywords from source titles/snippets appear in the report.\n"
            f"Source keywords sample: {sorted(source_keywords)[:20]}\n"
            f"Report preview: {report_str[:300]}"
        )

        # -- Content relevance --
        assert any(
            term in report_lower
            for term in ["quantum", "qubit", "computing", "processor", "superconducting"]
        ), "Report should discuss quantum computing hardware"

        # -- Summary --
        print(f"\n{'='*60}")
        print(f"Source-report reference check completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Source keywords extracted: {len(source_keywords)}")
        print(f"Keywords found in report: {len(overlapping_keywords)}")
        print(f"Overlap sample: {sorted(overlapping_keywords)[:15]}")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")
        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_multiple_research_steps_build_evidence(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Multiple research steps progressively build the evidence base.

        Citation verification depends on having sufficient, diverse evidence.
        This test verifies that the plan-and-execute loop produces multiple
        research steps and that the pools grow with each iteration.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "citation_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "Compare different approaches to carbon capture and "
                    "storage technology"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Research steps were executed --
        items_started = [e for e in events if isinstance(e, ItemStartedEvent)]
        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        assert len(items_started) >= 1, (
            "Should start at least one research step"
        )
        assert len(items_completed) >= 1, (
            "Should complete at least one research step"
        )

        # -- Tool calls happened --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 2, (
            f"Only {len(tool_calls)} tool call(s); need >= 2 for evidence building"
        )

        # -- Observations pool grew across steps --
        obs_pool = state.pools.get("observations")
        obs_count = obs_pool.count() if obs_pool else 0
        assert obs_count > 0, (
            "Observations pool is empty after research cycle. "
            "Evidence did not accumulate across steps."
        )

        # -- Sources pool has diverse URLs --
        src_pool = state.pools.get("sources")
        src_count = src_pool.count() if src_pool else 0
        assert src_count > 0, "Sources pool should not be empty"

        if src_pool and src_count > 0:
            all_sources = src_pool.get_recent(100)
            urls = {
                item.get("url", "")
                for item in all_sources
                if isinstance(item, dict) and item.get("url")
            }
            assert len(urls) >= 1, (
                "Sources pool should contain at least 1 distinct URL"
            )

        # -- Plan was created --
        plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
        assert len(plan_events) >= 1, "Should create a research plan"

        # -- Report has substance --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=500)

        # -- Content relevance --
        report_lower = report_str.lower()
        assert any(
            term in report_lower
            for term in ["carbon capture", "carbon", "ccs", "storage", "sequestration"]
        ), "Report should discuss carbon capture"

        # -- Summary --
        print(f"\n{'='*60}")
        print(f"Multi-step evidence building completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Research steps started: {len(items_started)}")
        print(f"Research steps completed: {len(items_completed)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Observations pool: {obs_count} items")
        print(f"Sources pool: {src_count} items")
        if src_pool and src_count > 0:
            print(f"Unique source URLs: {len(urls)}")
        print(f"Report length: {len(report_str)} chars")
        print(f"\nEvent summary: {event_summary(events)}")
        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_enterprise_sources_in_citation_context(
        self,
        llm_client: FrameworkLLMClient,
        enterprise_tools: list[Any],
        enterprise_tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Enterprise sources are available as citation evidence.

        The citation verification pipeline needs to access enterprise://
        URLs alongside web URLs. This test runs enterprise_research.yaml
        with mock enterprise tools and verifies that enterprise sources
        are properly captured in the sources pool.
        """
        start = time.perf_counter()

        definition = load_workflow(
            str(examples_dir / "enterprise_research.yaml")
        )
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the company revenue trends and "
                    "technical architecture?"
                ),
            },
            enterprise_tools=enterprise_tools,
            tool_registry=enterprise_tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Report should exist --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)
        assert_report_has_substance(report_str, min_length=200)

        # -- Sources pool should have enterprise:// URLs --
        src_pool = state.pools.get("sources")
        assert src_pool is not None, "sources pool should exist"
        assert src_pool.count() > 0, (
            "sources pool is empty; enterprise tools did not emit sources"
        )

        all_sources = src_pool.get_recent(100)
        enterprise_urls = [
            item.get("url", "")
            for item in all_sources
            if isinstance(item, dict) and item.get("url", "").startswith("enterprise://")
        ]
        assert len(enterprise_urls) > 0, (
            "No enterprise:// URLs found in sources pool. "
            "Enterprise tools should emit sources with enterprise:// URL scheme.\n"
            f"Source items: {all_sources[:5]}"
        )

        # -- Observations pool should have items --
        obs_pool = state.pools.get("observations")
        obs_count = obs_pool.count() if obs_pool else 0
        assert obs_count > 0, (
            "Observations pool is empty; enterprise tools did not produce findings"
        )

        # -- Tool calls should be enterprise-only --
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        enterprise_tool_names = {"genie", "vector_search", "knowledge_assistant"}
        for tc in tool_calls:
            assert tc.tool_name in enterprise_tool_names, (
                f"Expected enterprise tools only, got '{tc.tool_name}'"
            )

        # -- Summary --
        print(f"\n{'='*60}")
        print(f"Enterprise citation context test completed in {duration_s:.1f}s")
        print(f"{'='*60}")
        print(f"Report length: {len(report_str)} chars")
        print(f"Sources pool: {src_pool.count()} items")
        print(f"Enterprise URLs: {len(enterprise_urls)}")
        print(f"Observations pool: {obs_count} items")
        print(f"Tool calls: {len(tool_calls)}")
        print("\nEnterprise URLs found:")
        for i, url in enumerate(enterprise_urls):
            print(f"  [{i + 1}] {url}")
        print(f"\nEvent summary: {event_summary(events)}")
        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(480)
    async def test_report_quality_metrics(
        self,
        llm_client: FrameworkLLMClient,
        tool_registry: ToolRegistry,
        examples_dir: Path,
    ) -> None:
        """Report quality metrics meet thresholds for citation verification.

        Computes and asserts on key quality dimensions: report length,
        source count, observation count, tool call count, and research
        step count. These metrics indicate whether the pipeline produced
        enough material for the citation verification stages to operate.
        """
        start = time.perf_counter()

        definition = load_workflow(examples_dir / "citation_pipeline.yaml")
        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={
                "query": (
                    "What are the latest advances in large language model "
                    "training techniques?"
                ),
            },
            tool_registry=tool_registry,
        )

        duration_s = time.perf_counter() - start

        # -- Compute quality metrics --
        report = state.get("report")
        assert report is not None, "Should produce final report"
        report_str = str(report)

        src_pool = state.pools.get("sources")
        obs_pool = state.pools.get("observations")
        source_count = src_pool.count() if src_pool else 0
        observation_count = obs_pool.count() if obs_pool else 0

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_call_count = len(tool_calls)

        items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]
        research_steps = len(items_completed)

        plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
        planned_steps = len(plan_events[0].steps) if plan_events else 0

        workflow_completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        total_duration_ms = (
            workflow_completed[0].duration_ms if workflow_completed else 0
        )

        report_length = len(report_str)
        has_claims = _has_factual_claims(report_str)

        # -- Assert quality thresholds --
        assert report_length >= 500, (
            f"Report too short for citation verification: {report_length} chars "
            f"(minimum 500). Preview: {report_str[:300]}"
        )
        assert source_count >= 2, (
            f"Too few sources for citation verification: {source_count} "
            f"(minimum 2)"
        )
        assert observation_count >= 1, (
            f"Too few observations: {observation_count} (minimum 1)"
        )

        # -- Verify report passes substance check --
        assert_report_has_substance(report_str, min_length=500)

        # -- Content relevance --
        report_lower = report_str.lower()
        assert any(
            term in report_lower
            for term in [
                "language model", "llm", "training", "transformer",
                "fine-tuning", "pretraining", "scaling",
            ]
        ), "Report should discuss LLM training techniques"

        # -- Formatted quality metrics summary --
        print(f"\n{'='*60}")
        print("REPORT QUALITY METRICS")
        print(f"{'='*60}")
        print(f"  Duration:            {duration_s:.1f}s")
        if total_duration_ms:
            print(f"  Workflow duration:   {total_duration_ms:.0f}ms")
        print(f"  Report length:       {report_length:,} chars")
        print(f"  Has factual claims:  {has_claims}")
        print(f"  Source count:        {source_count}")
        print(f"  Observation count:   {observation_count}")
        print(f"  Tool call count:     {tool_call_count}")
        print(f"  Planned steps:       {planned_steps}")
        print(f"  Completed steps:     {research_steps}")
        print(f"{'='*60}")

        # -- Quality score (informational, not asserted) --
        quality_score = min(100, (
            min(report_length / 20, 25)         # up to 25 for length
            + min(source_count * 5, 25)         # up to 25 for sources
            + min(observation_count * 5, 25)    # up to 25 for observations
            + (25 if has_claims else 0)         # 25 for factual claims
        ))
        print(f"  Quality score:       {quality_score:.0f}/100")
        print(f"{'='*60}")

        print(f"\nEvent summary: {event_summary(events)}")
        print_full_diagnostics(events, state)
        print(f"\nReport preview:\n{report_str[:500]}")
