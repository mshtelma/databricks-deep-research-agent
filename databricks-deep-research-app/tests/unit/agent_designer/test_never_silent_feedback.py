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

from deep_research.agent_designer.critic_types import CriticVerdict
from deep_research.agent_designer.orchestrator import (
    _GENERIC_NO_CHANGE_MESSAGE,
    _build_critic_review_event,
    _critic_review_event,
    _signature_review_failed,
    _terminal_error_message,
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


def test_terminal_error_when_signature_loop_exhausts_unapproved() -> None:
    state = _state(
        signature_loop_done={
            "signature_loop_done": True,
            "critic_approved": False,
            "has_revision_request": True,
            "revision_count": 2,
            "exhausted": True,
        },
        revision_request={
            "reason": "the classifier kept selecting a topology without coverage review",
            "fields_to_reconsider": ["independent_workstreams_count"],
        },
    )
    msg = _terminal_error_message(state)
    assert msg is not None
    assert "couldn't create the agent" in msg
    assert "did not pass designer review" in msg
    assert "coverage review" in msg


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


def test_terminal_feedback_reads_critic_verdict_model() -> None:
    # REGRESSION (root cause B): critic_verdict is a CriticVerdict MODEL object in
    # production state; the old JSON-only coercer dropped it → generic message.
    state = _state(
        critic_verdict=CriticVerdict(
            approve=False,
            directives=[
                {
                    "node_path": "agents.final-report-synthesizer",
                    "issue": "synthesizer must enumerate all required outputs",
                    "suggested_action": "list fundamentals, risk, earnings in the prompt",
                    "severity": "blocking",
                }
            ],
        ),
    )
    msg = _terminal_feedback_message(state)
    assert msg != _GENERIC_NO_CHANGE_MESSAGE
    assert "unresolved issues" in msg
    assert "synthesizer must enumerate all required outputs" in msg
    assert "final-report-synthesizer" in msg  # node label surfaced


def test_gate_failure_takes_priority_over_stale_critic_verdict() -> None:
    # If the LAST iteration failed the structural gate the critic didn't run, so
    # critic_verdict is stale — the fresh gate failure must win (reorder).
    state = _state(
        gate_result={"status": "fail", "failures": [{"message": "placeholder not cleared"}]},
        critic_verdict=CriticVerdict(
            approve=False, directives=[{"issue": "stale critic note"}]
        ),
    )
    msg = _terminal_feedback_message(state)
    assert "structural checks" in msg
    assert "placeholder not cleared" in msg
    assert "stale critic note" not in msg


def test_signature_review_failed() -> None:
    assert _signature_review_failed(_state(signature_loop_done={"critic_approved": False})) is True
    assert _signature_review_failed(_state(signature_loop_done={"critic_approved": True})) is False
    assert _signature_review_failed(_state()) is False


def test_build_critic_review_event_from_directives() -> None:
    state = _state(
        critic_verdict=CriticVerdict(
            approve=False,
            directives=[
                {
                    "node_path": "agents.synth",
                    "issue": "missing earnings section",
                    "suggested_action": "add it",
                    "severity": "blocking",
                }
            ],
        ),
        signature_loop_done={"critic_approved": False, "revision_count": 2},
    )
    event = _build_critic_review_event(state)
    assert event is not None
    assert event.verdict == "needs_revision"
    assert event.agent_findings[0]["finding"] == "missing earnings section"
    assert event.agent_findings[0]["label"] == "synth"
    assert event.agent_findings[0]["node_path"] == "agents.synth"


def test_build_critic_review_event_none_when_no_directives() -> None:
    assert _build_critic_review_event(_state(critic_verdict=CriticVerdict(approve=True))) is None
    assert _build_critic_review_event(_state()) is None


def test_critic_review_event_pass_when_approved() -> None:
    # Always-on critic: an approved verdict now yields a "pass" card so the
    # reviewer visibly reports something on a successful build (it used to be
    # silent on success).
    event = _critic_review_event(_state(critic_verdict=CriticVerdict(approve=True)))
    assert event is not None
    assert event.verdict == "pass"
    assert event.agent_findings == []
    assert "approved" in event.summary.lower()


def test_critic_review_event_needs_revision_from_directives() -> None:
    state = _state(
        critic_verdict=CriticVerdict(
            approve=False,
            directives=[
                {
                    "node_path": "agents.synth",
                    "issue": "missing earnings section",
                    "suggested_action": "add it",
                    "severity": "blocking",
                }
            ],
        ),
    )
    event = _critic_review_event(state)
    assert event is not None
    assert event.verdict == "needs_revision"
    assert event.agent_findings[0]["finding"] == "missing earnings section"


def test_critic_review_event_none_when_critic_never_ran() -> None:
    # No verdict at all, AND the real per-turn init value (""), both mean the
    # critic never produced output (e.g. the surgical-edit / topology lanes,
    # which skip the architect-critic loop) → no empty card.
    assert _critic_review_event(_state()) is None
    assert _critic_review_event(_state(critic_verdict="")) is None
    assert _critic_review_event(_state(critic_verdict="   ")) is None


def test_critic_review_event_none_when_unapproved_without_directives() -> None:
    # e.g. a structural-gate failure: not approved but no actionable directives.
    # The terminal-feedback path explains that case instead of an empty card.
    assert _critic_review_event(_state(critic_verdict=CriticVerdict(approve=False))) is None
