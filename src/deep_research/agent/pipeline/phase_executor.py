"""
Phase Executor - Executes custom research phases in dependency order.

This module provides the PhaseExecutor class which:
1. Builds execution order from PhaseInsertion configurations
2. Groups phases that can run in parallel (same insert_after)
3. Executes phases respecting dependencies
4. Streams phase events for UI feedback
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from deep_research.agent.pipeline.protocols import (
    CustomPhase,
    PipelineCustomization,
)

if TYPE_CHECKING:
    from deep_research.agent.state import ResearchState
    from deep_research.agent.tools.base import ResearchContext

logger = logging.getLogger(__name__)


@dataclass
class PhaseEvent:
    """Event emitted during phase execution."""

    event_type: str  # "started", "completed", "error", "skipped"
    phase_name: str
    duration_ms: float = 0
    error: str | None = None
    sources_count: int = 0


@dataclass
class PhaseExecutionGroup:
    """Group of phases that can execute in parallel."""

    phases: list[str] = field(default_factory=list)
    insert_after: str = ""


@dataclass
class PhaseExecutor:
    """Executes custom research phases in dependency order.

    Phases are grouped by their insert_after value:
    - Phases in the same group can run in parallel
    - Groups execute in topological order based on dependencies

    Example execution order for SAPreSalesBot:
    1. [crm_context] - after coordinator
    2. [company_intel, industry_trends, attendee_research] - after crm_context (parallel)
    3. [customer_competitive] - after company_intel
    4. [vendor_competitive] - after customer_competitive
    """

    phases: dict[str, CustomPhase]
    customization: PipelineCustomization
    _execution_groups: list[PhaseExecutionGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Build execution order from phase insertions."""
        self._execution_groups = self._build_execution_groups()
        logger.info(
            "PhaseExecutor initialized with %d phases in %d groups",
            len(self.phases),
            len(self._execution_groups),
        )
        for i, group in enumerate(self._execution_groups):
            logger.debug(
                "Execution group %d (after '%s'): %s",
                i + 1,
                group.insert_after,
                group.phases,
            )

    def _build_execution_groups(self) -> list[PhaseExecutionGroup]:
        """Build execution groups from phase insertions.

        Groups phases by their insert_after value, then orders groups
        by dependency chain (topological sort).
        """
        # Group phases by their insert_after target
        groups_by_after: dict[str, list[str]] = {}
        for insertion in self.customization.phase_insertions:
            if not insertion.enabled:
                continue
            after = insertion.insert_after or "coordinator"
            if after not in groups_by_after:
                groups_by_after[after] = []
            groups_by_after[after].append(insertion.phase_name)

        # Build ordered execution groups via topological traversal
        execution_groups: list[PhaseExecutionGroup] = []
        visited: set[str] = set()

        def process_after(after: str) -> None:
            """Process phases that come after a given point."""
            if after in visited:
                return
            visited.add(after)

            if after in groups_by_after:
                group = PhaseExecutionGroup(
                    phases=groups_by_after[after],
                    insert_after=after,
                )
                execution_groups.append(group)

                # Process phases that depend on these phases
                for phase_name in groups_by_after[after]:
                    process_after(phase_name)

        # Start from coordinator (standard entry point)
        process_after("coordinator")

        return execution_groups

    async def execute_all(
        self,
        context: "ResearchContext",
        state: "ResearchState",
        config: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[PhaseEvent, "ResearchState"]]:
        """Execute all phases in dependency order.

        Yields:
            Tuple of (PhaseEvent, updated ResearchState) for each phase event
        """
        config = config or {}

        for group in self._execution_groups:
            # Determine which phases in this group should run
            phases_to_run: list[tuple[str, CustomPhase]] = []

            for phase_name in group.phases:
                phase = self.phases.get(phase_name)
                if not phase:
                    logger.warning("Phase '%s' not found in registry", phase_name)
                    continue

                # Check if phase should run
                should_run = True
                if hasattr(phase, "should_run"):
                    try:
                        should_run = phase.should_run(context)
                    except Exception as e:
                        logger.warning(
                            "Phase '%s' should_run() failed: %s",
                            phase_name,
                            e,
                        )
                        should_run = False

                if not should_run:
                    logger.info("Skipping phase '%s' (should_run=False)", phase_name)
                    yield (
                        PhaseEvent(event_type="skipped", phase_name=phase_name),
                        state,
                    )
                    continue

                phases_to_run.append((phase_name, phase))

            if not phases_to_run:
                continue

            # Emit start events for all phases in group
            for phase_name, _ in phases_to_run:
                yield (
                    PhaseEvent(event_type="started", phase_name=phase_name),
                    state,
                )

            # Execute phases in parallel
            tasks = [
                self._execute_phase(phase, context, state, config)
                for _, phase in phases_to_run
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for (phase_name, _), result in zip(phases_to_run, results):
                if isinstance(result, Exception):
                    logger.error("Phase '%s' failed: %s", phase_name, result)
                    yield (
                        PhaseEvent(
                            event_type="error",
                            phase_name=phase_name,
                            error=str(result),
                        ),
                        state,
                    )
                else:
                    # Merge phase result into state
                    phase_state, duration_ms, sources_count = result
                    state = self._merge_state(state, phase_state, phase_name)
                    yield (
                        PhaseEvent(
                            event_type="completed",
                            phase_name=phase_name,
                            duration_ms=duration_ms,
                            sources_count=sources_count,
                        ),
                        state,
                    )

    async def _execute_phase(
        self,
        phase: CustomPhase,
        context: "ResearchContext",
        state: "ResearchState",
        config: dict[str, Any],
    ) -> tuple["ResearchState", float, int]:
        """Execute a single phase.

        Returns:
            Tuple of (result_state, duration_ms, sources_count)
        """
        start_time = time.perf_counter()
        logger.info("Executing phase: %s", phase.name)

        try:
            result_state = await phase.execute(context, state, config)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Count sources added by this phase
            sources_count = 0
            if hasattr(result_state, "sources"):
                sources_count = len(result_state.sources) - len(state.sources)

            logger.info(
                "Phase '%s' completed in %.1fms with %d new sources",
                phase.name,
                duration_ms,
                sources_count,
            )
            return result_state, duration_ms, sources_count

        except Exception as e:
            logger.error("Phase '%s' execution failed: %s", phase.name, e)
            raise

    def _merge_state(
        self,
        current: "ResearchState",
        phase_result: "ResearchState",
        phase_name: str,
    ) -> "ResearchState":
        """Merge phase result into current state.

        Merges:
        - sources: Append new sources
        - all_observations: Append new observations
        - phase_results: Store phase output
        """
        # Merge sources (with deduplication)
        if hasattr(phase_result, "sources"):
            existing_urls = {s.url for s in current.sources if hasattr(s, "url")}
            for source in phase_result.sources:
                if hasattr(source, "url") and source.url not in existing_urls:
                    current.sources.append(source)
                    existing_urls.add(source.url)

        # Merge observations
        if hasattr(phase_result, "all_observations"):
            current.all_observations.extend(phase_result.all_observations)

        # Store phase result if method exists
        if hasattr(current, "add_phase_result") and hasattr(
            phase_result, "last_observation"
        ):
            from deep_research.agent.nodes.custom_phase_executor import PhaseResult

            current.add_phase_result(
                phase_name,
                PhaseResult(
                    output=phase_result.last_observation or "",
                    sources=[
                        s.__dict__ if hasattr(s, "__dict__") else s
                        for s in phase_result.sources
                    ],
                    success=True,
                ),
            )

        return current


__all__ = ["PhaseExecutor", "PhaseEvent", "PhaseExecutionGroup"]
