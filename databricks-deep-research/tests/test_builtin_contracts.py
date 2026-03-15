"""Parametrized builtin contract tests."""
from __future__ import annotations

import pytest

# Force registration of all builtins
import databricks_deep_research.agents.builtins  # noqa: F401
from databricks_deep_research.agents.builtins.registry import get_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.output_models import PlanOutput, ReflectionOutput
from databricks_deep_research.events.types import (
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    SynthesisStartedEvent,
)
from databricks_deep_research.pools.pool_state import PoolConfig, PoolState
from databricks_deep_research.workflow.state import WorkflowState


@pytest.mark.parametrize(
    ("subtype", "expected_output_model"),
    [
        ("researcher", None),
        ("planner", PlanOutput),
        ("reflector", ReflectionOutput),
        ("synthesizer", None),
    ],
)
def test_builtin_registration_contract(
    subtype: str,
    expected_output_model: type[object] | None,
) -> None:
    builtin = get_builtin(subtype)

    assert builtin is not None
    assert builtin.subtype == subtype
    if expected_output_model is not None:
        assert builtin.output_model is expected_output_model


@pytest.mark.parametrize(
    ("subtype", "config", "assertion_name", "expected"),
    [
        ("researcher", AgentNodeConfig(subtype="researcher", max_tool_calls=None), "max_tool_calls", 15),
        ("researcher", AgentNodeConfig(subtype="researcher", system_prompt=""), "system_prompt_nonempty", True),
        ("planner", AgentNodeConfig(subtype="planner", output_model=None), "output_model", PlanOutput),
        ("reflector", AgentNodeConfig(subtype="reflector", output_model=None), "output_model", ReflectionOutput),
        ("synthesizer", AgentNodeConfig(subtype="synthesizer", max_tool_calls=None), "max_tool_calls", 10),
    ],
)
def test_enrich_applies_builtin_defaults(
    subtype: str,
    config: AgentNodeConfig,
    assertion_name: str,
    expected: object,
) -> None:
    builtin = get_builtin(subtype)
    state = WorkflowState(query="test")

    assert builtin is not None and builtin.enrich_config is not None
    enriched = builtin.enrich_config(config, state)

    if assertion_name == "max_tool_calls":
        assert enriched.max_tool_calls == expected
    elif assertion_name == "output_model":
        assert enriched.output_model is expected
    else:
        assert enriched.system_prompt != ""


@pytest.mark.parametrize(
    ("subtype", "config", "expected"),
    [
        ("researcher", AgentNodeConfig(subtype="researcher", max_tool_calls=0), 0),
        ("researcher", AgentNodeConfig(subtype="researcher", max_tool_calls=5), 5),
        ("synthesizer", AgentNodeConfig(subtype="synthesizer", max_tool_calls=0), 0),
        ("synthesizer", AgentNodeConfig(subtype="synthesizer", max_tool_calls=20), 20),
    ],
)
def test_enrich_preserves_explicit_max_tool_calls(
    subtype: str,
    config: AgentNodeConfig,
    expected: int,
) -> None:
    builtin = get_builtin(subtype)

    assert builtin is not None and builtin.enrich_config is not None
    enriched = builtin.enrich_config(config, WorkflowState(query="test"))
    assert enriched.max_tool_calls == expected


def test_researcher_post_process_returns_no_events() -> None:
    builtin = get_builtin("researcher")

    assert builtin is not None and builtin.post_process is not None
    assert builtin.post_process(
        "researcher_1",
        {"findings": "data"},
        AgentNodeConfig(subtype="researcher"),
        WorkflowState(query="test"),
    ) == []


@pytest.mark.parametrize(
    "output",
    [
        PlanOutput(
            title="Test Plan",
            thought="Investigating",
            steps=[{"id": "s1", "title": "step 1"}],
            iteration=1,
            has_enough_context=False,
        ),
        {
            "title": "Dict Plan",
            "thought": "Thinking",
            "steps": [{"query": "s1"}, {"query": "s2"}],
            "iteration": 2,
            "has_enough_context": True,
        },
    ],
)
def test_planner_post_process_emits_plan_created_event(
    output: PlanOutput | dict[str, object],
) -> None:
    builtin = get_builtin("planner")

    assert builtin is not None and builtin.post_process is not None
    events = builtin.post_process(
        "planner_1",
        output,
        AgentNodeConfig(subtype="planner"),
        WorkflowState(query="test"),
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PlanCreatedEvent)
    assert event.title in {"Test Plan", "Dict Plan"}


@pytest.mark.parametrize(
    "output",
    [
        ReflectionOutput(
            decision="continue",
            reasoning="Need more data on quantum effects",
            suggested_changes=[],
            evidence_sufficiency="insufficient",
            failure_mode="metadata_only",
        ),
        {
            "decision": "complete",
            "reasoning": "All aspects covered",
        },
    ],
)
def test_reflector_post_process_emits_reflection_event(
    output: ReflectionOutput | dict[str, str],
) -> None:
    builtin = get_builtin("reflector")

    assert builtin is not None and builtin.post_process is not None
    events = builtin.post_process(
        "reflector_1",
        output,
        AgentNodeConfig(subtype="reflector"),
        WorkflowState(query="test"),
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ReflectionDecisionEvent)
    assert event.decision in {"continue", "complete"}


def test_unexpected_planner_and_reflector_output_returns_no_events() -> None:
    planner = get_builtin("planner")
    reflector = get_builtin("reflector")
    state = WorkflowState(query="test")

    assert planner is not None and planner.post_process is not None
    assert reflector is not None and reflector.post_process is not None
    assert planner.post_process("planner_1", 42, AgentNodeConfig(subtype="planner"), state) == []
    assert reflector.post_process(
        "reflector_1",
        "unexpected string",
        AgentNodeConfig(subtype="reflector"),
        state,
    ) == []


def test_synthesizer_post_process_uses_state_counts() -> None:
    state = WorkflowState(query="test")
    state.append("r1", "findings", "Finding about AI")
    state.append("r2", "findings", "Finding about ML")
    state.append("r1", "sources", [{"url": "https://a.com"}, {"url": "https://b.com"}])
    state.append("r2", "sources", [{"url": "https://c.com"}])
    builtin = get_builtin("synthesizer")

    assert builtin is not None and builtin.post_process is not None
    events = builtin.post_process(
        "synth_1",
        {"report": "Done"},
        AgentNodeConfig(subtype="synthesizer"),
        state,
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SynthesisStartedEvent)
    assert event.total_observations == 2
    assert event.total_sources == 3


def test_synthesizer_post_process_falls_back_to_pool_counts() -> None:
    state = WorkflowState(query="test")
    obs_pool = PoolState(PoolConfig(name="observations"))
    obs_pool.add("Finding about Kroger")
    obs_pool.add("Finding about impairment")
    src_pool = PoolState(PoolConfig(name="sources"))
    src_pool.add({"url": "https://a.com"})
    src_pool.add({"url": "https://b.com"})
    src_pool.add({"url": "https://c.com"})
    state.pools = {"observations": obs_pool, "sources": src_pool}
    builtin = get_builtin("synthesizer")

    assert builtin is not None and builtin.post_process is not None
    events = builtin.post_process(
        "synth_1",
        {"report": "Done"},
        AgentNodeConfig(subtype="synthesizer"),
        state,
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SynthesisStartedEvent)
    assert event.total_observations == 2
    assert event.total_sources == 3


def test_planner_post_process_filters_empty_steps() -> None:
    builtin = get_builtin("planner")
    assert builtin is not None and builtin.post_process is not None
    events = builtin.post_process(
        "planner_1",
        {"title": "Plan", "thought": "Think", "steps": [{}, {"title": "Do it"}]},
        AgentNodeConfig(subtype="planner"),
        WorkflowState(query="test"),
    )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PlanCreatedEvent)
    assert len(event.steps) == 1
