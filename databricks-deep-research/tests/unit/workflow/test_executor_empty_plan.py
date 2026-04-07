"""Tests for empty-plan recovery paths in the plan_and_execute executor.

Covers the following scenarios:
1. Empty plan + discovered_sources in state -> hydrate pools, exit as planner_sufficient_context
2. has_enough_context=True + pools have background items (total_items_processed=0) -> planner_sufficient_context exit
3. Empty plan + no discovered_sources + empty pools -> PlanningContractError raised
4. Duplicate discovered_sources items are not added twice to pools (dedup)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.errors import PlanningContractError
from databricks_deep_research.events.types import (
    PlanAndExecuteExitEvent,
    ReplanTriggeredEvent,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState
from tests.conftest import (
    build_mock_llm_client as _mock_llm_client,
)
from tests.conftest import (
    collect_events as _collect_events,
)
from tests.conftest import (
    events_of_type as _events_of_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_definition(
    root: WorkflowNode,
    pools: list[dict[str, Any]] | None = None,
) -> WorkflowDefinition:
    """Wrap a root node into a minimal WorkflowDefinition with optional pool configs."""
    return WorkflowDefinition(
        id="test-wf",
        name="Test Workflow",
        root=root,
        pools=pools or [
            {"name": "sources", "dedup_key": "url"},
            {"name": "observations"},
        ],
    )


def _pe_root(
    *,
    max_iterations: int = 2,
    min_iterations: int = 1,
    max_replan_cycles: int = 1,
) -> WorkflowNode:
    """Build a standard plan_and_execute root node for tests."""
    return WorkflowNode(
        id="pe",
        type=NodeType.plan_and_execute,
        label="Research",
        config={
            "planner": {"subtype": "planner", "output_key": "plan"},
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": {
                "id": "researcher",
                "type": "agent",
                "label": "Researcher",
                "config": {"subtype": "researcher", "output_key": "findings"},
            },
            "max_iterations": max_iterations,
            "min_iterations": min_iterations,
            "max_replan_cycles": max_replan_cycles,
        },
    )


def _empty_plan_agent(has_enough_context: bool = False) -> Any:
    """Return a fake_execute_agent that always returns an empty plan."""

    async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        config = kwargs.get("config")
        if config and config.subtype == "planner":
            return AgentOutput(
                content={
                    "title": "Empty Plan",
                    "steps": [],
                    "has_enough_context": has_enough_context,
                },
                output_key="plan",
                events=[],
            )
        return AgentOutput(content="finding", output_key="findings", events=[])

    return fake_execute_agent


# ---------------------------------------------------------------------------
# Test 1: discovered_sources recovery
# ---------------------------------------------------------------------------


class TestDiscoveredSourcesRecovery:
    """When planner returns empty plan but state has discovered_sources,
    hydrate pools and exit as planner_sufficient_context."""

    @pytest.mark.asyncio
    async def test_empty_plan_with_discovered_sources_exits_as_sufficient_context(
        self,
    ) -> None:
        """Empty plan + discovered_sources -> hydrate pools, exit as
        planner_sufficient_context (not PlanningContractError)."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")
        state.append(
            "background",
            "discovered_sources",
            [
                {
                    "url": "https://example.com/a",
                    "title": "Source A",
                    "summary": "Summary of source A",
                },
                {
                    "url": "https://example.com/b",
                    "title": "Source B",
                    "summary": "Summary of source B",
                },
            ],
        )

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            events = await _collect_events(executor, state)

        exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(exit_events) == 1
        assert exit_events[0].reason == "planner_sufficient_context"
        assert exit_events[0].total_items_processed == 0

    @pytest.mark.asyncio
    async def test_discovered_sources_hydrate_sources_pool(self) -> None:
        """discovered_sources items are added to the 'sources' pool."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        discovered = [
            {
                "url": "https://example.com/a",
                "title": "Source A",
                "summary": "Summary A",
            },
        ]
        state = WorkflowState(query="test query")
        state.append("background", "discovered_sources", discovered)

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            await _collect_events(executor, state)

        sources_pool = executor._pools.get("sources")
        assert sources_pool is not None
        assert sources_pool.count() >= 1
        added_item = sources_pool.items[0]
        assert added_item["url"] == "https://example.com/a"

    @pytest.mark.asyncio
    async def test_discovered_sources_hydrate_observations_pool(self) -> None:
        """discovered_sources summaries are added to the 'observations' pool."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        discovered = [
            {
                "url": "https://example.com/c",
                "title": "Source C",
                "summary": "Key findings about topic C",
            },
        ]
        state = WorkflowState(query="test query")
        state.append("background", "discovered_sources", discovered)

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            await _collect_events(executor, state)

        observations_pool = executor._pools.get("observations")
        assert observations_pool is not None
        assert observations_pool.count() >= 1
        obs = observations_pool.items[0]
        assert obs["text"] == "Key findings about topic C"
        assert obs["source"] == "discovered"

    @pytest.mark.asyncio
    async def test_recovery_populates_synthesis_state(self) -> None:
        """After discovered_sources recovery, synthesis state keys are written."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")
        state.append(
            "background",
            "discovered_sources",
            [
                {
                    "url": "https://example.com/d",
                    "title": "Source D",
                    "summary": "Summary D",
                },
            ],
        )

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            await _collect_events(executor, state)

        # _populate_synthesis_state writes these keys
        assert state.get("steps_executed") is not None
        assert state.get("plan_iterations") is not None

    @pytest.mark.asyncio
    async def test_recovery_after_replan_exhaustion(self) -> None:
        """Recovery from discovered_sources happens after all replan cycles
        are exhausted, not before."""
        planner_calls = 0

        async def counting_planner(node_id: str, **kwargs: Any) -> AgentOutput:
            nonlocal planner_calls
            config = kwargs.get("config")
            if config and config.subtype == "planner":
                planner_calls += 1
                return AgentOutput(
                    content={
                        "title": "Empty",
                        "steps": [],
                        "has_enough_context": False,
                    },
                    output_key="plan",
                    events=[],
                )
            return AgentOutput(content="finding", output_key="findings", events=[])

        root = _pe_root(max_replan_cycles=2)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")
        state.append(
            "background",
            "discovered_sources",
            [{"url": "https://example.com/x", "title": "X", "summary": "X info"}],
        )

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=counting_planner,
        ):
            events = await _collect_events(executor, state)

        # Should have replanned max_replan_cycles times before falling through
        replan_events = _events_of_type(events, ReplanTriggeredEvent)
        assert len(replan_events) == 2

        # planner called: 1 initial + 2 replans = 3
        assert planner_calls == 3

        exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(exit_events) == 1
        assert exit_events[0].reason == "planner_sufficient_context"


# ---------------------------------------------------------------------------
# Test 2: has_enough_context with pool content from background
# ---------------------------------------------------------------------------


class TestHasEnoughContextWithPoolContent:
    """When planner says has_enough_context=True and pools have items
    (even if total_items_processed == 0), allow planner_sufficient_context exit."""

    @pytest.mark.asyncio
    async def test_has_enough_context_with_pool_items_exits_immediately(self) -> None:
        """has_enough_context=True + pools have background items -> immediate exit
        as planner_sufficient_context, no replan needed."""
        root = _pe_root(min_iterations=0, max_replan_cycles=1)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        # Pre-populate pools with background data (simulating background node)
        sources_pool = executor._pools.get("sources")
        if sources_pool is not None:
            sources_pool.add({
                "url": "https://bg.example.com",
                "title": "Background Source",
            })
        observations_pool = executor._pools.get("observations")
        if observations_pool is not None:
            observations_pool.add({
                "text": "Background observation data",
                "source": "background",
            })

        state = WorkflowState(query="test query")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=True),
        ):
            events = await _collect_events(executor, state)

        exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(exit_events) == 1
        assert exit_events[0].reason == "planner_sufficient_context"
        assert exit_events[0].total_items_processed == 0

        # No replan should have occurred
        replan_events = _events_of_type(events, ReplanTriggeredEvent)
        assert len(replan_events) == 0

    @pytest.mark.asyncio
    async def test_has_enough_context_with_min_iterations_met(self) -> None:
        """has_enough_context=True + min_iterations already met (0 == 0)
        -> exits as planner_sufficient_context."""
        root = _pe_root(min_iterations=0, max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=True),
        ):
            events = await _collect_events(executor, state)

        exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(exit_events) == 1
        assert exit_events[0].reason == "planner_sufficient_context"


# ---------------------------------------------------------------------------
# Test 3: PlanningContractError when no recovery is possible
# ---------------------------------------------------------------------------


class TestPlanningContractErrorOnTrulyEmptyPlan:
    """When both discovered_sources and pools are empty and planner returns
    no items, PlanningContractError must still be raised."""

    @pytest.mark.asyncio
    async def test_empty_plan_no_discovered_sources_no_pools_raises(self) -> None:
        """Empty plan + no discovered_sources + empty pools -> PlanningContractError."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")
        # No discovered_sources set in state

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ), pytest.raises(PlanningContractError, match="zero executable steps"):
            await _collect_events(executor, state)

    @pytest.mark.asyncio
    async def test_empty_plan_empty_discovered_sources_raises(self) -> None:
        """Empty plan + discovered_sources is empty list -> PlanningContractError."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")
        state.append("background", "discovered_sources", [])

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ), pytest.raises(PlanningContractError, match="zero executable steps"):
            await _collect_events(executor, state)

    @pytest.mark.asyncio
    async def test_error_includes_node_id(self) -> None:
        """PlanningContractError message references the failing node id."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        state = WorkflowState(query="test query")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ), pytest.raises(PlanningContractError) as exc_info:
            await _collect_events(executor, state)

        assert "pe" in str(exc_info.value)
        assert exc_info.value.reason == "empty_plan"


# ---------------------------------------------------------------------------
# Test 4: Dedup on discovered_sources
# ---------------------------------------------------------------------------


class TestDiscoveredSourcesDedup:
    """Duplicate discovered_sources items should not be added twice to pools."""

    @pytest.mark.asyncio
    async def test_duplicate_urls_deduplicated_in_sources_pool(self) -> None:
        """Two discovered_sources with the same URL result in only one pool entry."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        duplicate_sources = [
            {
                "url": "https://example.com/same",
                "title": "Source Same",
                "summary": "First copy",
            },
            {
                "url": "https://example.com/same",
                "title": "Source Same",
                "summary": "First copy",
            },
        ]
        state = WorkflowState(query="test query")
        state.append("background", "discovered_sources", duplicate_sources)

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            events = await _collect_events(executor, state)

        sources_pool = executor._pools.get("sources")
        assert sources_pool is not None
        # The sources pool has dedup_key="url", so only one should be added
        assert sources_pool.count() == 1

        exit_events = _events_of_type(events, PlanAndExecuteExitEvent)
        assert len(exit_events) == 1
        assert exit_events[0].reason == "planner_sufficient_context"

    @pytest.mark.asyncio
    async def test_distinct_urls_both_added(self) -> None:
        """Two discovered_sources with different URLs are both added."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        distinct_sources = [
            {
                "url": "https://example.com/one",
                "title": "Source One",
                "summary": "Summary one",
            },
            {
                "url": "https://example.com/two",
                "title": "Source Two",
                "summary": "Summary two",
            },
        ]
        state = WorkflowState(query="test query")
        state.append("background", "discovered_sources", distinct_sources)

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            await _collect_events(executor, state)

        sources_pool = executor._pools.get("sources")
        assert sources_pool is not None
        assert sources_pool.count() == 2

    @pytest.mark.asyncio
    async def test_observations_from_duplicate_sources_also_deduplicated(self) -> None:
        """Observations derived from duplicate sources are deduplicated
        via content-hash dedup on the observations pool."""
        root = _pe_root(max_replan_cycles=0)
        defn = _make_definition(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())

        duplicate_sources = [
            {
                "url": "https://example.com/dup",
                "title": "Dup Source",
                "summary": "Same summary text",
            },
            {
                "url": "https://example.com/dup",
                "title": "Dup Source",
                "summary": "Same summary text",
            },
        ]
        state = WorkflowState(query="test query")
        state.append("background", "discovered_sources", duplicate_sources)

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_empty_plan_agent(has_enough_context=False),
        ):
            await _collect_events(executor, state)

        observations_pool = executor._pools.get("observations")
        assert observations_pool is not None
        # Content-hash dedup should prevent the identical observation dict
        # from being added twice
        assert observations_pool.count() == 1
