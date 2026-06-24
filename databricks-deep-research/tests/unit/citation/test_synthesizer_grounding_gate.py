from __future__ import annotations

from databricks_deep_research.agents.builtins.synthesizer import (
    _build_reclaim_generation_instructions,
    _classify_grounding,
    _GroundingOutcome,
)
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.citation.types import ClaimInfo, ClaimRole


def _claim(
    verdict: str | None,
    confidence: float | None = None,
    *,
    abstained: bool = False,
    role: str = ClaimRole.FACT.value,
) -> ClaimInfo:
    c = ClaimInfo(
        claim_text="test",
        claim_type="factual",
        position_start=0,
        position_end=4,
        claim_role=role,
    )
    c.verification_verdict = verdict
    c.verification_confidence = confidence
    c.abstained = abstained
    return c


def test_grounding_gate_fails_when_verifier_finds_zero_grounded_claims() -> None:
    """Real grounding failure: verifier ran on all claims with non-zero confidence
    and rejected them. Expect HARD_FAIL — the report should be overwritten with
    the canned "Insufficient Evidence" template."""
    claims = [_claim("unsupported", 0.6) for _ in range(32)]
    verdict = _classify_grounding(claims, report_content="Some report")
    assert verdict.outcome == _GroundingOutcome.HARD_FAIL
    assert "32" in verdict.reason
    assert "unsupported or contradicted" in verdict.reason


def test_grounding_gate_soft_warns_when_verifier_did_not_really_run() -> None:
    """The actual 1945-query case: 19 abstained + 24 unsupported with
    confidence=0 (verifier-crash). Expect SOFT_WARN — report flows through
    with a banner, not the canned fallback."""
    claims = (
        [_claim("abstained", 0.0, abstained=True) for _ in range(19)]
        + [_claim("unsupported", 0.0) for _ in range(24)]
    )
    verdict = _classify_grounding(claims, report_content="Some report")
    assert verdict.outcome == _GroundingOutcome.SOFT_WARN
    assert "could not judge" in verdict.reason


def test_grounding_gate_allows_supported_claims() -> None:
    """At least one positive claim → OK, no gate trip."""
    claims = [_claim("supported", 0.9)] + [_claim("unsupported", 0.6) for _ in range(3)]
    verdict = _classify_grounding(claims, report_content="Some report")
    assert verdict.outcome == _GroundingOutcome.OK
    assert verdict.reason == ""


def test_grounding_gate_allows_no_claims_and_no_content() -> None:
    """Nothing happened → OK (defer to other gates)."""
    verdict = _classify_grounding([], report_content="")
    assert verdict.outcome == _GroundingOutcome.OK


def test_grounding_gate_flags_no_claims_extracted_when_content_present() -> None:
    """Synthesizer produced report content but the citation pipeline extracted
    no fact claims → NO_CLAIMS_EXTRACTED, banner-prefixed but report flows."""
    verdict = _classify_grounding([], report_content="Some content")
    assert verdict.outcome == _GroundingOutcome.NO_CLAIMS_EXTRACTED


def test_grounding_gate_defers_when_no_pipeline_run() -> None:
    """claims=None → OK (no pipeline ran; caller's other gates handle it)."""
    verdict = _classify_grounding(None, report_content="Some content")
    assert verdict.outcome == _GroundingOutcome.OK


def test_grounding_gate_ignores_free_role_claims() -> None:
    """FREE-role claims don't count toward fact-claim bucketing."""
    claims = [_claim("unsupported", 0.0, role=ClaimRole.FREE.value) for _ in range(5)]
    verdict = _classify_grounding(claims, report_content="Some content")
    assert verdict.outcome == _GroundingOutcome.NO_CLAIMS_EXTRACTED


def test_grounding_gate_unknown_verdict_counts_as_no_judgment() -> None:
    """Defensive: unknown verdict strings get counted into no_judgment (conservative)."""
    claims = [_claim("mystery_verdict", 0.0) for _ in range(3)]
    verdict = _classify_grounding(claims, report_content="Some report")
    assert verdict.outcome == _GroundingOutcome.SOFT_WARN


def test_reclaim_generation_instructions_include_designer_contract() -> None:
    config = AgentNodeConfig(
        subtype="synthesizer",
        output_schema={
            "report_contract": {
                "required_outputs": ["Executive Summary", "Risk Review"],
                "quality_gates": ["No unsupported recommendations"],
            }
        },
        system_prompt=(
            "Base prompt\n\n"
            "## Workflow-Specific Report Format\n"
            "Use the exact headings requested by the workflow."
        ),
        user_prompt_template=(
            "Base user prompt\n\n"
            "## Workflow-Specific Instructions\n"
            "## Investment Thesis & Recommendation\n"
            "Include a recommendation only when supported."
        ),
    )

    instructions = _build_reclaim_generation_instructions(config)

    assert "Executive Summary" in instructions
    assert "Risk Review" in instructions
    assert "No unsupported recommendations" in instructions
    assert "Use the exact headings requested by the workflow" in instructions
    assert "Investment Thesis & Recommendation" in instructions
