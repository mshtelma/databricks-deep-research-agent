"""Integration tests: single agent execution with real LLM.

Tests the framework's ability to:
- Load a YAML workflow and execute it
- Run a single coordinator agent against a real Databricks endpoint
- Parse structured output (CoordinatorOutput)
- Emit proper framework events

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_single_agent.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from databricks_deep_research import (
    FrameworkLLMClient,
    run_workflow_from_yaml,
)
from databricks_deep_research.agents.output_models import CoordinatorOutput
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    CoordinatorClassifiedEvent,
    NodeCompletedEvent,
    NodeStartedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from tests.integration.conftest import requires_databricks


@pytest.mark.integration
class TestSingleAgentWorkflow:
    """Run a single coordinator agent from YAML against a real LLM."""

    @requires_databricks
    @pytest.mark.asyncio
    async def test_coordinator_classifies_query(
        self,
        llm_client: FrameworkLLMClient,
        examples_dir: Path,
    ) -> None:
        """Coordinator agent classifies a query and returns structured output."""
        state, events = await run_workflow_from_yaml(
            str(examples_dir / "single_agent.yaml"),
            llm_client,
            initial_state={"query": "What is quantum computing?"},
        )

        # Verify state has coordination output
        coordination = state.get("coordination")
        assert coordination is not None, "Coordinator should produce output"

        # Output may be CoordinatorOutput (Pydantic model) or dict
        if isinstance(coordination, CoordinatorOutput):
            assert coordination.complexity in ("simple", "moderate", "complex")
        elif isinstance(coordination, dict):
            assert "complexity" in coordination or "is_simple" in coordination
        else:
            pytest.fail(f"Unexpected output type: {type(coordination)}")

        # Verify events were emitted
        event_types = {type(e) for e in events}
        assert WorkflowStartedEvent in event_types, "Should emit WorkflowStartedEvent"
        assert WorkflowCompletedEvent in event_types, "Should emit WorkflowCompletedEvent"
        assert NodeStartedEvent in event_types, "Should emit NodeStartedEvent"
        assert NodeCompletedEvent in event_types, "Should emit NodeCompletedEvent"
        assert AgentOutputEvent in event_types, "Should emit AgentOutputEvent"
        assert CoordinatorClassifiedEvent in event_types, "Should emit CoordinatorClassifiedEvent"

        # Print results
        print(f"\nCoordination output: {coordination}")
        print(f"Events emitted: {len(events)}")
        for e in events:
            if isinstance(e, CoordinatorClassifiedEvent):
                print(f"  Complexity: {e.complexity}")
                print(f"  Recommended depth: {e.recommended_depth}")
                print(f"  Is simple: {e.is_simple}")

    @requires_databricks
    @pytest.mark.asyncio
    async def test_coordinator_handles_simple_query(
        self,
        llm_client: FrameworkLLMClient,
        examples_dir: Path,
    ) -> None:
        """Coordinator should recognize a simple factual query."""
        state, events = await run_workflow_from_yaml(
            str(examples_dir / "single_agent.yaml"),
            llm_client,
            initial_state={"query": "What is 2 + 2?"},
        )

        coordination = state.get("coordination")
        assert coordination is not None

        # Simple arithmetic should be classified as simple
        classified_events = [e for e in events if isinstance(e, CoordinatorClassifiedEvent)]
        assert len(classified_events) == 1
        assert classified_events[0].is_simple is True, (
            "Simple arithmetic should be classified as simple"
        )

        print(f"\nSimple query classification: {coordination}")

    @requires_databricks
    @pytest.mark.asyncio
    async def test_coordinator_handles_complex_query(
        self,
        llm_client: FrameworkLLMClient,
        examples_dir: Path,
    ) -> None:
        """Coordinator should recognize a complex research query as non-simple."""
        state, events = await run_workflow_from_yaml(
            str(examples_dir / "single_agent.yaml"),
            llm_client,
            initial_state={
                "query": (
                    "Compare the economic and environmental trade-offs of "
                    "electric vehicles versus hydrogen fuel cells for "
                    "long-haul commercial trucking in the European Union, "
                    "including infrastructure costs, total cost of ownership, "
                    "well-to-wheel emissions, and regulatory frameworks"
                )
            },
        )

        coordination = state.get("coordination")
        assert coordination is not None

        classified_events = [e for e in events if isinstance(e, CoordinatorClassifiedEvent)]
        assert len(classified_events) == 1

        # A detailed multi-faceted research query should not be simple
        ce = classified_events[0]
        assert ce.complexity in ("moderate", "complex"), (
            f"Multi-faceted query should be moderate/complex, got '{ce.complexity}'"
        )

        print(f"\nComplex query classification: {coordination}")
        print(f"  Complexity: {ce.complexity}")
        print(f"  Recommended depth: {ce.recommended_depth}")
        print(f"  Is simple: {ce.is_simple}")
