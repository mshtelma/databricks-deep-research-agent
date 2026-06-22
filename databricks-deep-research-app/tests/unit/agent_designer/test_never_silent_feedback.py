"""Phase 1 — never-silent feedback.

The designer's only user-facing surface is the pending-mutation card, and all
designer agents suppress their own prose. So a turn that produces no AST change
used to render NOTHING (the reported "Best of N edit does nothing" bug). The
orchestrator now guarantees a visible message/error on every turn; the
user-facing explanation is built by the pure helper exercised here.
"""

from __future__ import annotations

import json

from databricks_deep_research.workflow.state import WorkflowState

from deep_research.agent_designer.orchestrator import (
    _GENERIC_NO_CHANGE_MESSAGE,
    _terminal_feedback_message,
)


def _state(**keys: object) -> WorkflowState:
    state = WorkflowState()
    for key, value in keys.items():
        state.append("test", key, value)
    return state


def test_empty_state_returns_generic_capability_message() -> None:
    msg = _terminal_feedback_message(_state())
    assert msg == _GENERIC_NO_CHANGE_MESSAGE
    # Generic across topologies — Best-of-N is only an example, never hardcoded.
    assert "parallel lanes" in msg and "plan-and-execute" in msg


def test_revision_request_dict_takes_priority() -> None:
    state = _state(
        revision_request={
            "reason": "the requested shape is not one of the deterministic topologies",
            "fields_to_reconsider": ["coordination_pattern"],
        },
        critic_verdict={"directives": [{"issue": "should not surface — lower priority"}]},
    )
    msg = _terminal_feedback_message(state)
    assert "reconsider the workflow structure" in msg
    assert "not one of the deterministic topologies" in msg
    assert "coordination_pattern" in msg
    assert "lower priority" not in msg


def test_revision_request_accepts_json_string() -> None:
    state = _state(revision_request=json.dumps({"reason": "structure mismatch"}))
    assert "structure mismatch" in _terminal_feedback_message(state)


def test_critic_verdict_directives_when_no_revision() -> None:
    state = _state(
        critic_verdict={
            "approve": False,
            "directives": [
                {"issue": "synthesizer missing required section"},
                {"issue": "lane prompt references sibling lane"},
            ],
        }
    )
    msg = _terminal_feedback_message(state)
    assert "unresolved issues" in msg
    assert "synthesizer missing required section" in msg
    assert "lane prompt references sibling lane" in msg


def test_gate_failures_when_no_revision_or_verdict() -> None:
    state = _state(
        gate_result=json.dumps(
            {"status": "fail", "failures": [{"message": "placeholder_pending not cleared"}]}
        )
    )
    msg = _terminal_feedback_message(state)
    assert "structural checks" in msg
    assert "placeholder_pending not cleared" in msg


def test_passing_gate_does_not_produce_gate_message() -> None:
    # A gate that passed must not be reported as a failure; fall through to generic.
    state = _state(gate_result={"status": "pass", "failures": []})
    assert _terminal_feedback_message(state) == _GENERIC_NO_CHANGE_MESSAGE


def test_malformed_values_fall_through_to_generic() -> None:
    state = _state(
        revision_request="not json at all",
        critic_verdict=12345,
        gate_result=["unexpected", "shape"],
    )
    assert _terminal_feedback_message(state) == _GENERIC_NO_CHANGE_MESSAGE
