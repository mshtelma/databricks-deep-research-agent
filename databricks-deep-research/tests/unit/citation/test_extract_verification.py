"""Framework-side verification extraction tests.

Three golden fixtures derived from the app's previous behavior. Each
asserts that the framework's :func:`extract_verification` produces the
expected ``Claim`` / ``SummaryInfo`` structure.
"""

from __future__ import annotations

from databricks_deep_research.citation.extraction import (
    Claim,
    SummaryInfo,
    VerificationSummary,
    extract_verification,
    extract_verification_from_report,
)


class _FakeState:
    """Minimal stand-in for WorkflowState exposing ``.get(key)``."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key: str):  # type: ignore[no-untyped-def]
        return self._data.get(key)


def test_empty_state_returns_empty_summary() -> None:
    v = extract_verification(None, [])
    assert isinstance(v, VerificationSummary)
    assert v.claims == []
    assert v.summary is None


def test_state_with_native_claims_extracts_typed() -> None:
    state = _FakeState({
        "claims": [
            {
                "claim_text": "AI is powerful",
                "claim_type": "general",
                "position_start": 0,
                "position_end": 14,
                "evidence": {
                    "source_url": "https://example.com",
                    "quote_text": "AI is powerful",
                    "has_numeric_content": False,
                },
                "confidence_level": "high",
                "verification_verdict": "supported",
            },
        ],
        "verification_summary": {
            "total_claims": 1,
            "verified_claims": 1,
            "softened_claims": 0,
            "removed_claims": 0,
            "corrected_citations": 0,
        },
    })

    v = extract_verification(state, [])
    assert len(v.claims) == 1
    c = v.claims[0]
    assert isinstance(c, Claim)
    assert c.claim_text == "AI is powerful"
    assert c.evidence is not None
    assert c.evidence.source_url == "https://example.com"

    assert isinstance(v.summary, SummaryInfo)
    assert v.summary.total_claims == 1
    assert v.summary.supported_count == 1


def test_claim_without_evidence_resolves_via_citation_key() -> None:
    sources = [
        {"url": "http://s0", "snippet": "snippet 0"},
        {"url": "http://s1", "snippet": "snippet 1"},
    ]
    state = _FakeState({
        "claims": [
            {
                "claim_text": "Indexed claim",
                "claim_type": "general",
                "position_start": 0,
                "position_end": 13,
                "citation_key": "1",
            },
        ],
    })
    v = extract_verification(state, sources)
    assert len(v.claims) == 1
    e = v.claims[0].evidence
    assert e is not None
    assert e.source_url == "http://s1"


def test_extract_from_report_parses_markers() -> None:
    report = "AI advances rapidly [1]. Models are larger [2]."
    sources = [
        {"url": "http://a", "snippet": "AI snippet"},
        {"url": "http://b", "snippet": "Model snippet"},
    ]
    v = extract_verification_from_report(report, sources)
    assert len(v.claims) == 2
    assert v.claims[0].evidence is not None
    assert v.claims[0].evidence.source_url == "http://a"
    assert v.claims[1].evidence.source_url == "http://b"
    assert v.summary is not None
    assert v.summary.total_claims == 2
    assert v.summary.supported_count == 2


def test_extract_from_report_detects_numeric_claim_type() -> None:
    report = "Revenue grew $35.1B in Q3 [1]."
    sources = [{"url": "http://a", "snippet": "rev"}]
    v = extract_verification_from_report(report, sources)
    assert v.claims[0].claim_type == "numeric"


def test_extract_from_report_index_offset_detection() -> None:
    # 0-indexed report (rare but supported). Sentences must be at least 10
    # characters AFTER stripping ``[N]`` markers — that's the framework's
    # filter for trivially short claims.
    report = "First detailed claim here [0]. Second detailed claim there [1]."
    sources = [{"url": "http://a"}, {"url": "http://b"}]
    v = extract_verification_from_report(report, sources)
    assert v.claims[0].evidence.source_url == "http://a"
    assert v.claims[1].evidence.source_url == "http://b"


def test_summary_accessors_match_event_payload_shape() -> None:
    state = _FakeState({
        "claims": [],
        "verification_summary": {
            "total_claims": 5,
            "verified_claims": 4,
            "softened_claims": 1,
            "removed_claims": 0,
            "corrected_citations": 2,
        },
    })
    v = extract_verification(state, [])
    assert v.total_claims == 5
    assert v.verified_claims == 4
    assert v.softened_claims == 1
    assert v.removed_claims == 0
    assert v.corrected_citations == 2
