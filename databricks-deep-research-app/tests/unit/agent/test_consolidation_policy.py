"""Unit tests for the consolidation trust policy.

Only verified claims may become durable chat knowledge: a refuted, contradicted,
unsupported, or abstained claim must NEVER be persisted as a finding (it would
resurface as "knowledge" in later turns). This policy is the trust gate between
the citation pipeline's output and the durable store.
"""

from __future__ import annotations

import pytest

from deep_research.agent.consolidation_policy import extract_consolidatable_claims

pytestmark = pytest.mark.unit


def test_keeps_only_supported_and_partial_with_confidence_mapping() -> None:
    vdata = {
        "claims": [
            {"claim_text": "A.", "verification_verdict": "supported", "abstained": False},
            {"claim_text": "B.", "verification_verdict": "partial", "abstained": False},
            {"claim_text": "C.", "verification_verdict": "unsupported", "abstained": False},
            {"claim_text": "D.", "verification_verdict": "contradicted", "abstained": False},
            {"claim_text": "E.", "verification_verdict": "supported", "abstained": True},
            {"claim_text": "", "verification_verdict": "supported", "abstained": False},
        ]
    }
    out = extract_consolidatable_claims(vdata)
    mapped = {c["claim_text"]: c["confidence"] for c in out}
    assert mapped == {"A.": "high", "B.": "medium"}


def test_empty_or_missing_is_empty() -> None:
    assert extract_consolidatable_claims(None) == []
    assert extract_consolidatable_claims({}) == []
    assert extract_consolidatable_claims({"summary": {}}) == []
    assert extract_consolidatable_claims({"claims": []}) == []


def test_abstained_excluded_even_if_supported() -> None:
    vdata = {"claims": [{"claim_text": "X.", "verification_verdict": "supported", "abstained": True}]}
    assert extract_consolidatable_claims(vdata) == []
