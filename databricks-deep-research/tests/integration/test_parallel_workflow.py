"""Integration tests: parallel workflow with real LLM + mixed tools.

Tests the framework's parallel execution:
- Two researchers (web + enterprise) running concurrently
- Shared pool writes with no data loss
- Events emitted from both parallel children
- Synthesizer consuming findings from both tracks

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_parallel_workflow.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.events.types import (
    NodeStartedEvent,
    StreamEvent,
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
    print_pool_summary,
)
from tests.integration.conftest import requires_all_credentials

_QUERY = "What are the best practices for building scalable AI systems?"


@pytest.mark.integration
class TestParallelWorkflow:
    """Parallel research pipeline with web + enterprise researchers and shared pools."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_parallel_researchers_both_produce_findings(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Both parallel researchers complete and contribute to the observations pool."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "parallel_research.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": _QUERY},
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        # Both researchers should have started
        node_started = [e for e in events if isinstance(e, NodeStartedEvent)]
        started_ids = {e.node_id for e in node_started}
        assert "researcher_web" in started_ids, (
            f"researcher_web not started. Started node_ids: {started_ids}"
        )
        assert "researcher_enterprise" in started_ids, (
            f"researcher_enterprise not started. Started node_ids: {started_ids}"
        )

        # Observations pool should have content from at least one researcher
        observations_pool = state.pools.get("observations")
        assert observations_pool is not None, "observations pool should exist"
        assert observations_pool.count() > 0, (
            f"observations pool is empty — researchers did not write findings"
        )

        # Final report should have substance
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        assert_report_has_substance(str(report), min_length=200)

        # Workflow should complete
        completed = [e for e in events if isinstance(e, WorkflowCompletedEvent)]
        assert len(completed) == 1

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Observations pool count: {observations_pool.count()}")
        print(f"Report length: {len(str(report))} chars")
        print(f"Total events: {len(events)}")
        print(f"Event summary: {event_summary(events)}")
        print_pool_summary(state)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_parallel_pool_writes_no_data_loss(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Parallel pool writes do not lose data — both researchers contribute."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "parallel_research.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": _QUERY},
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        # Observations pool should have items (at least something from researchers)
        observations_pool = state.pools.get("observations")
        assert observations_pool is not None, "observations pool should exist"
        obs_count = observations_pool.count()
        assert obs_count >= 1, (
            f"observations pool should have >= 1 items, got {obs_count}"
        )

        # Sources pool should also have items
        sources_pool = state.pools.get("sources")
        assert sources_pool is not None, "sources pool should exist"
        src_count = sources_pool.count()
        assert src_count > 0, (
            f"sources pool should have items, got {src_count}"
        )

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Observations pool: {obs_count} items")
        print(f"Sources pool: {src_count} items")
        print_pool_summary(state)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_parallel_events_from_both_children(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Events are emitted from both parallel children (web + enterprise)."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "parallel_research.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": _QUERY},
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        # NodeStartedEvent for both researchers
        node_started = [e for e in events if isinstance(e, NodeStartedEvent)]
        started_ids = {e.node_id for e in node_started}
        assert "researcher_web" in started_ids, (
            f"researcher_web missing from NodeStartedEvents. Got: {started_ids}"
        )
        assert "researcher_enterprise" in started_ids, (
            f"researcher_enterprise missing from NodeStartedEvents. Got: {started_ids}"
        )

        # At least one ToolCallEvent should exist
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 1, (
            f"Expected at least 1 ToolCallEvent, got {len(tool_calls)}"
        )

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"NodeStartedEvent node_ids: {started_ids}")
        print(f"ToolCallEvent count: {len(tool_calls)}")
        tool_names = [tc.tool_name for tc in tool_calls]
        print(f"Tool names called: {tool_names}")
        print(f"Event summary: {event_summary(events)}")
        print_event_timeline(events)

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_synthesizer_uses_parallel_findings(
        self,
        llm_client: FrameworkLLMClient,
        mixed_tool_registry: ToolRegistry,
        enterprise_tools: list[Any],
        examples_dir: Path,
    ) -> None:
        """Synthesizer produces a substantive report from parallel research findings."""
        t0 = time.monotonic()

        definition = load_workflow(examples_dir / "parallel_research.yaml")

        state, events = await run_workflow(
            definition,
            llm_client,
            initial_state={"query": _QUERY},
            enterprise_tools=enterprise_tools,
            tool_registry=mixed_tool_registry,
        )

        elapsed = time.monotonic() - t0

        # Report must exist and have substance
        report = state.get("report")
        assert report is not None, "Synthesizer should produce a report"
        report_str = str(report)
        assert len(report_str) > 200, (
            f"Report too short ({len(report_str)} chars). Preview: {report_str[:200]}"
        )
        assert_report_has_substance(report_str, min_length=200)

        # Pool counts for diagnostics
        observations_pool = state.pools.get("observations")
        sources_pool = state.pools.get("sources")
        obs_count = observations_pool.count() if observations_pool else 0
        src_count = sources_pool.count() if sources_pool else 0

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Report length: {len(report_str)} chars")
        print(f"Observations pool: {obs_count} items")
        print(f"Sources pool: {src_count} items")
        print(f"\nReport preview:\n{report_str[:500]}")
        print_pool_summary(state)
