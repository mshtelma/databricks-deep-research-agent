"""Shared helpers for real enterprise integration tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from databricks_deep_research.events.types import (
    ClaimGeneratedEvent,
    ClaimVerifiedEvent,
    SynthesisStartedEvent,
    ToolCallEvent,
    VerificationSummaryEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.executor import run_workflow
from databricks_deep_research.workflow.loader import load_workflow
from tests.helpers import (
    assert_report_has_substance,
    event_summary,
    print_event_timeline,
    print_pool_summary,
    print_search_queries,
    print_verification_summary,
)


@dataclass(slots=True)
class EnterpriseDatasetProfile:
    """Intent/configuration for an enterprise integration run."""

    label: str
    mode: str
    query: str
    expected_tool_name: str
    required_keywords: tuple[str, ...]
    required_term_groups: tuple[tuple[str, ...], ...] = ()
    min_report_length: int = 200


@dataclass(slots=True)
class EnterpriseRunArtifacts:
    """Pre-computed integration test facts from one enterprise workflow run."""

    profile: EnterpriseDatasetProfile
    state: Any
    events: list[Any]
    elapsed: float
    report_str: str
    tool_calls: list[Any]
    matching_tool_calls: list[Any]
    sources_pool: Any
    observations_pool: Any | None
    recent_sources: list[Any]
    enterprise_urls: list[str]


def build_enterprise_registry(*tools: ResearchTool) -> ToolRegistry:
    """Build an external-only enterprise tool registry."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register_external(tool.definition.name, tool)
    return registry


async def run_enterprise_case(
    *,
    workflow_path: Path,
    llm_client: FrameworkLLMClient,
    profile: EnterpriseDatasetProfile,
    enterprise_tools: list[ResearchTool],
    tool_registry: ToolRegistry,
) -> EnterpriseRunArtifacts:
    """Execute one enterprise workflow and gather common artifacts."""
    definition = load_workflow(str(workflow_path))

    t0 = time.monotonic()
    state, events = await run_workflow(
        definition,
        llm_client,
        initial_state={"query": profile.query},
        enterprise_tools=list(enterprise_tools),
        tool_registry=tool_registry,
        strict_tool_resolution=True,
    )
    elapsed = time.monotonic() - t0

    tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]
    matching_tool_calls = [
        event for event in tool_calls if event.tool_name == profile.expected_tool_name
    ]
    sources_pool = state.pools.get("sources")
    observations_pool = state.pools.get("observations")
    recent_sources = sources_pool.get_recent(50) if sources_pool is not None else []
    enterprise_urls = [
        item.get("url", "") if isinstance(item, dict) else str(item)
        for item in recent_sources
        if isinstance(item, dict) and "enterprise://" in str(item.get("url", ""))
    ]
    report = state.get("report")
    report_str = str(report) if report is not None else ""

    return EnterpriseRunArtifacts(
        profile=profile,
        state=state,
        events=events,
        elapsed=elapsed,
        report_str=report_str,
        tool_calls=tool_calls,
        matching_tool_calls=matching_tool_calls,
        sources_pool=sources_pool,
        observations_pool=observations_pool,
        recent_sources=recent_sources,
        enterprise_urls=enterprise_urls,
    )


def assert_enterprise_baseline(result: EnterpriseRunArtifacts) -> None:
    """Shared assertions for real enterprise runs across reclaim/classical lanes."""
    assert result.matching_tool_calls, (
        f"{result.profile.label}: expected tool '{result.profile.expected_tool_name}' "
        f"to be called. Tool calls found: {[tc.tool_name for tc in result.tool_calls]}"
    )

    assert result.sources_pool is not None, (
        f"{result.profile.label}: sources pool should exist"
    )
    assert result.sources_pool.count() > 0, (
        f"{result.profile.label}: sources pool should have items"
    )

    assert result.enterprise_urls, (
        f"{result.profile.label}: sources pool should contain enterprise:// URLs. "
        f"Got sources: {result.recent_sources[:3]}"
    )
    assert any(
        isinstance(item, dict) and item.get("content")
        for item in result.recent_sources
    ), (
        f"{result.profile.label}: enterprise sources should preserve full content, "
        "not only snippets"
    )
    assert any(
        isinstance(item, dict)
        and (
            item.get("source_type") == result.profile.expected_tool_name
            or f"enterprise://{result.profile.expected_tool_name}/" in str(item.get("url", ""))
        )
        for item in result.recent_sources
    ), (
        f"{result.profile.label}: sources pool should preserve the expected enterprise "
        f"tool family '{result.profile.expected_tool_name}'. "
        f"Got: {result.recent_sources[:3]}"
    )

    assert result.report_str, (
        f"{result.profile.label}: synthesizer should produce a report"
    )
    assert_report_has_substance(
        result.report_str,
        min_length=result.profile.min_report_length,
        required_term_groups=result.profile.required_term_groups,
    )
    assert "<analysis>" not in result.report_str and "<free>" not in result.report_str, (
        f"{result.profile.label}: final report should not expose internal reclaim tags. "
        f"Preview: {result.report_str[:400]}"
    )

    assert any(
        keyword in result.report_str.lower() for keyword in result.profile.required_keywords
    ), (
        f"{result.profile.label}: report should be relevant to the dataset/query. "
        f"Keywords checked: {result.profile.required_keywords}. "
        f"Preview: {result.report_str[:400]}"
    )


def assert_reclaim_enterprise_output(result: EnterpriseRunArtifacts) -> None:
    """Assertions specific to reclaim-mode enterprise synthesis."""
    events = result.events
    synthesis_events = [event for event in events if isinstance(event, SynthesisStartedEvent)]
    claim_generated = [event for event in events if isinstance(event, ClaimGeneratedEvent)]
    claim_verified = [event for event in events if isinstance(event, ClaimVerifiedEvent)]
    verification_summaries = [
        event for event in events if isinstance(event, VerificationSummaryEvent)
    ]

    assert synthesis_events, (
        f"{result.profile.label}: reclaim mode should emit SynthesisStartedEvent"
    )
    assert claim_generated, (
        f"{result.profile.label}: reclaim mode should emit ClaimGeneratedEvent"
    )
    assert claim_verified, (
        f"{result.profile.label}: reclaim mode should emit ClaimVerifiedEvent"
    )
    assert verification_summaries, (
        f"{result.profile.label}: reclaim mode should emit VerificationSummaryEvent"
    )

    verification_summary = result.state.get("verification_summary")
    verification_details = result.state.get("verification_details")
    analysis_summary = result.state.get("analysis_summary")
    assert isinstance(verification_summary, dict) and verification_summary, (
        f"{result.profile.label}: reclaim mode should persist verification_summary"
    )
    assert isinstance(verification_details, dict) and verification_details, (
        f"{result.profile.label}: reclaim mode should persist verification_details"
    )
    assert isinstance(analysis_summary, dict), (
        f"{result.profile.label}: reclaim mode should persist analysis_summary"
    )


def assert_classical_enterprise_output(result: EnterpriseRunArtifacts) -> None:
    """Assertions specific to classical_lite grounded enterprise synthesis."""
    events = result.events
    claim_generated = [event for event in events if isinstance(event, ClaimGeneratedEvent)]
    claim_verified = [event for event in events if isinstance(event, ClaimVerifiedEvent)]
    verification_summaries = [
        event for event in events if isinstance(event, VerificationSummaryEvent)
    ]

    assert claim_generated, (
        f"{result.profile.label}: classical_lite runs should emit ClaimGeneratedEvent"
    )
    assert claim_verified, (
        f"{result.profile.label}: classical_lite runs should emit ClaimVerifiedEvent"
    )
    assert verification_summaries, (
        f"{result.profile.label}: classical_lite runs should emit VerificationSummaryEvent"
    )

    verification_summary = result.state.get("verification_summary")
    verification_details = result.state.get("verification_details")
    analysis_summary = result.state.get("analysis_summary")
    claims = result.state.get("claims")

    assert isinstance(verification_summary, dict) and verification_summary, (
        f"{result.profile.label}: classical_lite runs should persist verification_summary"
    )
    assert isinstance(verification_details, dict) and verification_details, (
        f"{result.profile.label}: classical_lite runs should persist verification_details"
    )
    assert isinstance(analysis_summary, dict), (
        f"{result.profile.label}: classical_lite runs should persist analysis_summary"
    )
    assert isinstance(claims, list) and claims, (
        f"{result.profile.label}: classical_lite runs should persist claim details"
    )


def print_enterprise_case_diagnostics(result: EnterpriseRunArtifacts) -> None:
    """Emit rich diagnostics with dataset/mode labels for trace correlation."""
    summary = event_summary(result.events)
    obs_count = result.observations_pool.count() if result.observations_pool else 0

    print(f"\n{'=' * 72}")
    print(
        f"{result.profile.label} "
        f"[mode={result.profile.mode}, tool={result.profile.expected_tool_name}] "
        f"completed in {result.elapsed:.1f}s"
    )
    print(f"{'=' * 72}")
    print(f"Total events: {len(result.events)}")
    print(f"Event summary: {summary}")
    print(f"Tool calls: {len(result.tool_calls)}")
    print(f"Matching calls ({result.profile.expected_tool_name}): {len(result.matching_tool_calls)}")
    print(f"Sources pool: {result.sources_pool.count() if result.sources_pool else 0} items")
    print(f"Observations pool: {obs_count} items")
    print(f"Enterprise URLs: {len(result.enterprise_urls)}")
    print(f"Report length: {len(result.report_str)} chars")
    print(f"\n--- REPORT PREVIEW ({len(result.report_str)} chars) ---")
    print(result.report_str[:1000])

    if result.state.get("verification_summary"):
        print_verification_summary(result.state, result.events)
    print_search_queries(result.events)
    print_pool_summary(result.state)
    print_event_timeline(result.events)
