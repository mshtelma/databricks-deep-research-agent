from __future__ import annotations

from databricks_deep_research.agents.builtins.synthesizer import (
    _build_reclaim_generation_instructions,
    _grounding_failure_reason,
)
from databricks_deep_research.agents.config import AgentNodeConfig


def test_grounding_gate_fails_when_verifier_finds_zero_grounded_claims() -> None:
    reason = _grounding_failure_reason(
        {
            "total_claims": 32,
            "verified_claims": 0,
            "supported_count": 0,
            "partial_count": 0,
            "claims_fully_verified": 0,
            "claims_partially_softened": 0,
            "atomic_facts_verified": 0,
        }
    )

    assert "zero supported" in reason
    assert "32" in reason


def test_grounding_gate_allows_supported_claims() -> None:
    reason = _grounding_failure_reason(
        {
            "total_claims": 4,
            "verified_claims": 1,
            "supported_count": 1,
            "partial_count": 0,
        }
    )

    assert reason == ""


def test_grounding_gate_allows_empty_claim_summary() -> None:
    assert _grounding_failure_reason({"total_claims": 0}) == ""


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
