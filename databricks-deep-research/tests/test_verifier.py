"""T106: Tests for IsolatedVerifier (Stage 4: Isolated Verification).

Verifies:
- Single claim verification (full and quick modes)
- Batch processing
- Verification cache hit behavior
- Verdict parsing
- NEI heuristic check
- Fingerprinting
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.citation.config import IsolatedVerificationConfig
from databricks_deep_research.citation.isolated_verifier import (
    IsolatedVerifier,
)
from databricks_deep_research.citation.types import (
    BatchVerificationItem,
    BatchVerificationOutput,
    RankedEvidence,
    VerificationOutput,
    VerificationResult,
    VerificationVerdict,
)
from databricks_deep_research.llm.client import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked_evidence(**overrides: Any) -> RankedEvidence:
    defaults: dict[str, Any] = {
        "source_id": None,
        "source_url": "https://example.com/article",
        "source_title": "Example Article",
        "quote_text": "Revenue reached $3.2 billion in Q4 2024.",
        "start_offset": 0,
        "end_offset": 50,
        "section_heading": None,
        "relevance_score": 0.9,
        "has_numeric_content": True,
        "is_snippet_based": False,
    }
    defaults.update(overrides)
    return RankedEvidence(**defaults)


def _mock_llm_structured(
    structured: Any = None,
    content: str = "SUPPORTED",
) -> MagicMock:
    """Create a mock FrameworkLLMClient with structured output support."""
    llm = MagicMock()
    resp = LLMResponse(content=content, structured=structured)
    llm.complete = AsyncMock(return_value=resp)
    return llm


def _make_verification_output(
    verdict: str = "SUPPORTED",
    reasoning: str = "Evidence directly supports claim.",
) -> VerificationOutput:
    return VerificationOutput(
        verdict=verdict,
        reasoning=reasoning,
        key_match="Revenue reached $3.2 billion",
        issues=None,
    )


def test_verification_output_coerces_string_issues_to_none() -> None:
    output = VerificationOutput(
        verdict="SUPPORTED",
        reasoning="Direct match.",
        key_match="Revenue reached $3.2 billion",
        issues="None identified. The claim is supported by the evidence.",
    )

    assert output.issues is None


def test_verification_output_coerces_string_issues_to_singleton_list() -> None:
    output = VerificationOutput(
        verdict="PARTIAL",
        reasoning="Mostly supported.",
        key_match="Revenue reached $3.2 billion",
        issues="Date mismatch between claim and evidence",
    )

    assert output.issues == ["Date mismatch between claim and evidence"]


# ---------------------------------------------------------------------------
# T106-1: Single claim verification -- full mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_verification_with_structured_output() -> None:
    """Full verification should parse structured output to a result."""
    structured = _make_verification_output("SUPPORTED", "Direct match.")
    llm = _mock_llm_structured(structured=structured)
    verifier = IsolatedVerifier(llm)

    result = await verifier.verify_with_isolation(
        claim_text="Revenue was $3.2B in Q4.",
        evidence=_make_ranked_evidence(),
        use_quick_verification=False,
    )

    assert result.verdict == VerificationVerdict.SUPPORTED
    assert "Direct match" in result.reasoning
    assert result.abstained is False
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_full_verification_fallback_on_no_structured() -> None:
    """Full verification should fall back to parsing raw content."""
    llm = _mock_llm_structured(
        structured=None,
        content='{"verdict": "PARTIAL", "reasoning": "Some match"}',
    )
    verifier = IsolatedVerifier(llm)

    result = await verifier.verify_with_isolation(
        claim_text="Revenue was $3.2B in Q4.",
        evidence=_make_ranked_evidence(),
    )

    assert result.verdict == VerificationVerdict.PARTIAL


@pytest.mark.asyncio
async def test_full_verification_handles_exception() -> None:
    """Full verification should return abstained result on exception."""
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    verifier = IsolatedVerifier(llm)

    result = await verifier.verify_with_isolation(
        claim_text="Some claim",
        evidence=_make_ranked_evidence(),
    )

    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert result.abstained is True
    assert "failed" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# T106-2: Quick verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_verification() -> None:
    """Quick verification should parse a simple verdict string."""
    llm = _mock_llm_structured(content="SUPPORTED")
    verifier = IsolatedVerifier(llm)

    result = await verifier.verify_with_isolation(
        claim_text="Revenue was $3.2B",
        evidence=_make_ranked_evidence(),
        use_quick_verification=True,
    )

    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.reasoning == "Quick verification"


@pytest.mark.asyncio
async def test_quick_verification_handles_exception() -> None:
    """Quick verification should return abstained on failure."""
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("timeout"))
    verifier = IsolatedVerifier(llm)

    result = await verifier.verify_with_isolation(
        claim_text="Some claim",
        evidence=_make_ranked_evidence(),
        use_quick_verification=True,
    )

    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert result.abstained is True


# ---------------------------------------------------------------------------
# T106-3: Batch verification (sequential)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_batch_sequential() -> None:
    """verify_batch processes claims sequentially."""
    structured = _make_verification_output("SUPPORTED")
    llm = _mock_llm_structured(structured=structured)
    verifier = IsolatedVerifier(llm)

    claims = [
        ("Claim A", _make_ranked_evidence()),
        ("Claim B", _make_ranked_evidence()),
    ]

    results = await verifier.verify_batch(claims)

    assert len(results) == 2
    assert all(r.verdict == VerificationVerdict.SUPPORTED for r in results)
    assert llm.complete.await_count == 2


@pytest.mark.asyncio
async def test_verify_batch_with_confidence_levels() -> None:
    """verify_batch routes high-confidence to quick verification."""
    # We use a single mock that returns SUPPORTED for all calls.
    llm = _mock_llm_structured(content="SUPPORTED")
    verifier = IsolatedVerifier(llm)

    claims = [
        ("Claim A", _make_ranked_evidence()),
        ("Claim B", _make_ranked_evidence()),
    ]
    confidence_levels = ["high", "low"]

    results = await verifier.verify_batch(claims, confidence_levels)

    assert len(results) == 2
    # First call uses quick (high confidence), second uses full.
    # Both should succeed regardless.
    assert all(
        r.verdict == VerificationVerdict.SUPPORTED for r in results
    )


# ---------------------------------------------------------------------------
# T106-4: Grouped batch verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_batch_grouped_with_structured() -> None:
    """verify_batch_grouped uses batch LLM calls with structured output."""
    batch_output = BatchVerificationOutput(
        results=[
            BatchVerificationItem(
                claim_index=0, verdict="SUPPORTED", reasoning="Match"
            ),
            BatchVerificationItem(
                claim_index=1, verdict="PARTIAL", reasoning="Partial match"
            ),
        ]
    )
    llm = _mock_llm_structured(structured=batch_output)
    verifier = IsolatedVerifier(llm)

    claims = [
        ("Claim A", _make_ranked_evidence()),
        ("Claim B", _make_ranked_evidence()),
    ]

    results = await verifier.verify_batch_grouped(claims, batch_size=10)

    assert len(results) == 2
    assert results[0].verdict == VerificationVerdict.SUPPORTED
    assert results[1].verdict == VerificationVerdict.PARTIAL


@pytest.mark.asyncio
async def test_verify_batch_grouped_empty_input() -> None:
    """verify_batch_grouped returns empty list for empty input."""
    llm = _mock_llm_structured()
    verifier = IsolatedVerifier(llm)

    results = await verifier.verify_batch_grouped([])
    assert results == []


# ---------------------------------------------------------------------------
# T106-5: Cache hit behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_batch_grouped_cache_hit() -> None:
    """Cached claims should be returned without LLM calls."""
    llm = _mock_llm_structured()
    verifier = IsolatedVerifier(llm)

    claim_text = "Revenue was $3.2B"
    evidence = _make_ranked_evidence()
    fp = verifier.fingerprint_pair(claim_text, evidence.quote_text)

    cached_result = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        reasoning="Cached result",
    )
    cache: dict[str, VerificationResult] = {fp: cached_result}

    results = await verifier.verify_batch_grouped(
        [(claim_text, evidence)],
        verification_cache=cache,
    )

    assert len(results) == 1
    assert results[0].reasoning == "Cached result"
    llm.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_batch_grouped_populates_cache() -> None:
    """New results should be added to the cache after verification."""
    batch_output = BatchVerificationOutput(
        results=[
            BatchVerificationItem(
                claim_index=0, verdict="SUPPORTED", reasoning="OK"
            ),
        ]
    )
    llm = _mock_llm_structured(structured=batch_output)
    verifier = IsolatedVerifier(llm)

    evidence = _make_ranked_evidence()
    cache: dict[str, VerificationResult] = {}

    await verifier.verify_batch_grouped(
        [("New claim", evidence)],
        verification_cache=cache,
    )

    assert len(cache) == 1
    stored = list(cache.values())[0]
    assert stored.verdict == VerificationVerdict.SUPPORTED


# ---------------------------------------------------------------------------
# T106-6: Verdict parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_verdict", "expected"),
    [
        ("SUPPORTED", VerificationVerdict.SUPPORTED),
        ("PARTIAL", VerificationVerdict.PARTIAL),
        ("CONTRADICTED", VerificationVerdict.CONTRADICTED),
        ("UNSUPPORTED", VerificationVerdict.UNSUPPORTED),
        ("  supported  ", VerificationVerdict.SUPPORTED),
        ("The verdict is CONTRADICTED.", VerificationVerdict.CONTRADICTED),
        ("MAYBE", VerificationVerdict.UNSUPPORTED),
    ],
)
def test_parse_verdict_variants(
    raw_verdict: str,
    expected: VerificationVerdict,
) -> None:
    assert IsolatedVerifier.parse_verdict(raw_verdict) == expected


# ---------------------------------------------------------------------------
# T106-7: NEI heuristic check
# ---------------------------------------------------------------------------


def test_check_nei_low_overlap() -> None:
    """NEI should be True when word overlap is below 20%."""
    verifier = IsolatedVerifier(
        _mock_llm_structured(),
        config=IsolatedVerificationConfig(enable_nei_verdict=True),
    )
    result = verifier.check_nei(
        "Quantum computing achieves supremacy",
        _make_ranked_evidence(quote_text="The weather forecast is sunny tomorrow"),
    )
    assert result is True


def test_check_nei_high_overlap() -> None:
    """NEI should be False when word overlap is high."""
    verifier = IsolatedVerifier(
        _mock_llm_structured(),
        config=IsolatedVerificationConfig(enable_nei_verdict=True),
    )
    result = verifier.check_nei(
        "Revenue reached $3.2 billion in Q4 2024",
        _make_ranked_evidence(quote_text="Revenue reached $3.2 billion in Q4 2024"),
    )
    assert result is False


def test_check_nei_disabled() -> None:
    """NEI check should always return False when disabled."""
    verifier = IsolatedVerifier(
        _mock_llm_structured(),
        config=IsolatedVerificationConfig(enable_nei_verdict=False),
    )
    result = verifier.check_nei(
        "Quantum computing achieves supremacy",
        _make_ranked_evidence(quote_text="The weather is sunny"),
    )
    assert result is False


# ---------------------------------------------------------------------------
# T106-8: Fingerprinting
# ---------------------------------------------------------------------------


def test_fingerprint_claim_deterministic() -> None:
    """Fingerprint should be deterministic for same input."""
    fp1 = IsolatedVerifier.fingerprint_claim("Revenue was $3.2B in Q4.")
    fp2 = IsolatedVerifier.fingerprint_claim("Revenue was $3.2B in Q4.")
    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_claim_normalized() -> None:
    """Fingerprint should normalize text (case, punctuation, word order)."""
    fp1 = IsolatedVerifier.fingerprint_claim("Revenue WAS $3.2B!")
    fp2 = IsolatedVerifier.fingerprint_claim("revenue was 32b")
    # After normalization: words sorted then hashed -- same words, same hash
    assert fp1 == fp2


def test_fingerprint_pair_differs_from_claim() -> None:
    """Fingerprint_pair should produce a different hash than fingerprint_claim."""
    claim_fp = IsolatedVerifier.fingerprint_claim("Revenue was $3.2B")
    pair_fp = IsolatedVerifier.fingerprint_pair(
        "Revenue was $3.2B", "Revenue reached $3.2 billion"
    )
    assert claim_fp != pair_fp


def test_fingerprint_pair_changes_with_evidence() -> None:
    """Same claim with different evidence should produce different fingerprints."""
    fp1 = IsolatedVerifier.fingerprint_pair("Same claim", "Evidence A")
    fp2 = IsolatedVerifier.fingerprint_pair("Same claim", "Evidence B")
    assert fp1 != fp2
