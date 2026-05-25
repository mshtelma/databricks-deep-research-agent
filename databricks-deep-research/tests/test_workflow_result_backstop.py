"""Plan v2.3 — ``WorkflowResult.output`` UX backstop.

When the citation pipeline returns ``verified_claims=0`` AND Stage 8
surgically removed every claim text, the framework's primary-text path
returns an empty string. The shell app (and every other consumer of
``WorkflowRunner``) reads ``WorkflowResult.output`` directly, so an
empty string surfaces as "No response returned." in the UI.

The backstop surfaces the streamed synthesizer content prepended with a
clear warning banner — better than silent emptiness. The framework's
``verification_summary`` event is still emitted with the real numbers,
so the verification badge in any UI remains accurate.
"""
from __future__ import annotations

from databricks_deep_research.events.types import (
    AgentStreamChunkEvent,
    VerificationSummaryEvent,
)
from databricks_deep_research.runner import WorkflowResult
from databricks_deep_research.workflow.state import WorkflowState

_TS = "2026-05-25T12:00:00Z"


def _chunks(*parts: str) -> list[AgentStreamChunkEvent]:
    return [
        AgentStreamChunkEvent(node_id="synthesizer", timestamp=_TS, chunk=p)
        for p in parts
    ]


def _verif(total: int, verified: int) -> VerificationSummaryEvent:
    return VerificationSummaryEvent(
        node_id="synthesizer",
        timestamp=_TS,
        total_claims=total,
        verified_claims=verified,
        corrected_citations=0,
        removed_claims=0,
        softened_claims=0,
        overall_confidence=0.0,
    )


def test_backstop_fires_when_no_claims_verify_and_chunks_substantive() -> None:
    """Real failure-mode reproduction: Stage 8 stripped every claim text,
    so the primary-text path returns empty. The synthesizer streamed real
    content. Backstop surfaces the draft with the banner."""
    state = WorkflowState(query="treasury 1945")
    body = (
        "## Total War Expenditures\n\n"
        "The United States government spent $90.5 billion on war activities "
        "in fiscal year 1945, representing the peak of mobilization across "
        "the Army, Navy, and Maritime Commission programs.\n\n"
        "Munitions accounted for $58.5 billion, pay and subsistence "
        "for $21.6 billion, and miscellaneous war expenditures for "
        "$10.4 billion." * 2
    )
    result = WorkflowResult(
        state=state,
        events=[*_chunks(body), _verif(total=55, verified=0)],
        definition=None,
    )
    out = result.output
    assert "Citations could not be verified" in out
    assert "$90.5 billion" in out
    # Banner names the claim count so the user can judge severity.
    assert "55" in out


def test_backstop_skipped_when_some_claims_verify() -> None:
    """Mixed verification result — at least one claim grounded — must
    return the primary text unchanged. No banner."""
    state = WorkflowState(query="x")
    # Seed a state entry that the WorkflowResult primary-text fallback can
    # read via extract_output. We use a no-definition path that simply
    # returns empty primary text; the assertion is that the banner is NOT
    # emitted because verified_claims > 0.
    result = WorkflowResult(
        state=state,
        events=[
            *_chunks("substantive synthesizer content " * 20),
            _verif(total=10, verified=4),
        ],
        definition=None,
    )
    out = result.output
    assert "Citations could not be verified" not in out


def test_backstop_skipped_when_chunks_below_min_length() -> None:
    """A few-character streamed fragment is usually preamble/headers, not
    a real draft. Don't surface it under the banner."""
    state = WorkflowState(query="x")
    result = WorkflowResult(
        state=state,
        events=[*_chunks("short"), _verif(total=5, verified=0)],
        definition=None,
    )
    assert "Citations could not be verified" not in result.output


def test_backstop_skipped_when_no_verification_summary() -> None:
    """Workflows that don't run the citation pipeline (e.g. tool-only
    runs, simple_response mode) emit no verification_summary event and
    must keep returning whatever the primary-text path produces."""
    state = WorkflowState(query="x")
    result = WorkflowResult(
        state=state,
        events=[*_chunks("a real draft from a non-citation pipeline " * 20)],
        definition=None,
    )
    assert "Citations could not be verified" not in result.output


def test_backstop_does_not_fire_when_primary_text_substantial() -> None:
    """Even with verified_claims=0, when the primary text is still
    substantial (e.g. only a few claims were stripped), don't override
    with the banner. The user gets the partially-cleaned report."""
    state = WorkflowState(query="x")
    state.append("synth", "report", "A" * 5000)  # substantial primary text
    # Mock a definition that exposes "report" as the output_key.
    from databricks_deep_research.workflow.definition import (
        WorkflowDefinition,
        WorkflowNode,
    )

    definition = WorkflowDefinition(
        id="t",
        name="t",
        version=1,
        required_inputs=[],
        output_keys=["report"],
        root=WorkflowNode(id="r", type="sequence", label="r", config={}, children=[]),
    )
    chunks = "B" * 1000
    result = WorkflowResult(
        state=state,
        events=[*_chunks(chunks), _verif(total=10, verified=0)],
        definition=definition,
    )
    # primary_text length 5000, chunks length 1000 → primary not "less
    # than half of chunks" → backstop does not fire.
    assert "Citations could not be verified" not in result.output
    assert "AAAA" in result.output


def test_backstop_uses_latest_verification_summary() -> None:
    """When the synthesizer ran two passes (SYNTH_PIPELINE_V2), two
    verification_summary events are emitted. The backstop must use the
    LAST one — that's what Stage 8 acted on."""
    state = WorkflowState(query="x")
    result = WorkflowResult(
        state=state,
        events=[
            *_chunks("a real long draft " * 30),
            _verif(total=20, verified=20),  # first pass: all good
            _verif(total=5, verified=0),  # second pass: all stripped
        ],
        definition=None,
    )
    out = result.output
    assert "Citations could not be verified" in out
    assert "5" in out  # banner reflects the final pass
