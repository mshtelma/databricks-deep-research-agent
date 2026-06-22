"""Pool-wide numeric evidence recovery (Stage 4/6).

A numeric FACT claim that the synthesizer mis-cited to an evidence span lacking
its figure must be re-verified against the figure-bearing source elsewhere in the
evidence pool — flipping a *false* unsupported/contradicted verdict to supported
and re-pointing the citation to the source that actually contains the number.

This guards the OfficeQA failure mode: monthly table values (e.g. ``January:
4,294``) were cited to vector "Entities:" metadata chunks that contain no
figures, while the cell lived in a structured ``table_load`` row. The numeric
verifier could not find ``4,294`` in the cited metadata, so the claim was
hedged ``[unverified]`` or removed even though the value is in-corpus.

Generic across source kinds: recovery keys only on numeric content +
``source_kind`` (corpus/structured preferred) — no domain, table, or corpus
identifiers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.citation.config import CitationConfig
from databricks_deep_research.citation.numeric_verifier import is_exact_numeric_match
from databricks_deep_research.citation.pipeline import CitationVerificationPipeline
from databricks_deep_research.citation.types import (
    ClaimInfo,
    EvidenceInfo,
    NumericValue,
    NumericVerificationResult,
    RankedEvidence,
    VerificationResult,
    VerificationVerdict,
)

pytestmark = pytest.mark.asyncio

_FIGURE = "4,294"
_METADATA_URL = "https://corpus/vector-metadata"
_STRUCTURED_URL = "https://corpus/table-load-row"


def _ranked(**overrides: Any) -> RankedEvidence:
    defaults: dict[str, Any] = {
        "source_url": _METADATA_URL,
        "quote_text": "Entities: 1938, 1939, 1940; Columns: Fiscal year | Total",
        "relevance_score": 0.9,
        "source_kind": "vector_index",
        "has_numeric_content": False,
        "source_pool_index": 0,
        "evidence_pool_index": 0,
    }
    defaults.update(overrides)
    return RankedEvidence(**defaults)


def _structured(**overrides: Any) -> RankedEvidence:
    """A structured table-read span that actually contains the figure."""
    defaults: dict[str, Any] = {
        "source_url": _STRUCTURED_URL,
        "quote_text": 'row label 1945: {"col_2": "4,294", "col_3": "3,866"}',
        "relevance_score": 0.55,
        "source_kind": "sql_analytics",
        "has_numeric_content": True,
        "source_pool_index": 1,
        "evidence_pool_index": 1,
    }
    defaults.update(overrides)
    return RankedEvidence(**defaults)


def _evidence_info(ranked: RankedEvidence) -> EvidenceInfo:
    return EvidenceInfo(
        source_url=ranked.source_url,
        quote_text=ranked.quote_text,
        relevance_score=ranked.relevance_score,
        has_numeric_content=ranked.has_numeric_content,
        source_pool_index=ranked.source_pool_index,
        evidence_pool_index=ranked.evidence_pool_index,
        source_kind=ranked.source_kind,
    )


def _numeric_claim(cited: EvidenceInfo, **overrides: Any) -> ClaimInfo:
    defaults: dict[str, Any] = {
        "claim_text": "January: 4,294",
        "claim_type": "numeric",
        "position_start": 0,
        "position_end": 15,
        "evidence": cited,
        "evidences": [cited],
        "confidence_level": "medium",
        "citation_key": "0",
        "citation_keys": ["0"],
    }
    defaults.update(overrides)
    return ClaimInfo(**defaults)


def _numeric_verifier_real_match() -> MagicMock:
    """Numeric verifier mock mirroring the real exact-match heuristic: a strong
    match (0.95) when the figure is present in the evidence, weak (0.2) otherwise."""
    verifier = MagicMock()

    async def _verify(
        *, claim_text: str, evidence: RankedEvidence
    ) -> NumericVerificationResult:
        match = is_exact_numeric_match(claim_text, evidence.quote_text)
        return NumericVerificationResult(
            claim_text=claim_text,
            parsed_value=NumericValue(
                raw_text=claim_text, normalized_value=None, unit=None, entity=None
            ),
            qa_results=[],
            overall_match=match,
            derivation_type="direct",
            confidence=0.95 if match else 0.2,
        )

    verifier.verify_numeric_claim = AsyncMock(side_effect=_verify)
    return verifier


def _build_pipeline(
    miss_verdict: VerificationVerdict, **overrides: Any
) -> CitationVerificationPipeline:
    """Pipeline whose NLI verifier returns SUPPORTED iff the evidence contains the
    figure, else *miss_verdict* — so the verdict tracks which evidence it sees."""
    verifier = MagicMock()

    async def _verify_iso(
        *,
        claim_text: str,
        evidence: RankedEvidence,
        use_quick_verification: bool = False,
    ) -> VerificationResult:
        if _FIGURE in (evidence.quote_text or ""):
            return VerificationResult(
                verdict=VerificationVerdict.SUPPORTED,
                reasoning="figure present in evidence",
                confidence=0.9,
            )
        return VerificationResult(
            verdict=miss_verdict, reasoning="figure absent from evidence", confidence=0.75
        )

    verifier.verify_with_isolation = AsyncMock(side_effect=_verify_iso)

    kwargs: dict[str, Any] = {
        "llm": MagicMock(),
        "evidence_selector": MagicMock(),
        "claim_generator": MagicMock(),
        "confidence_classifier": MagicMock(),
        "isolated_verifier": verifier,
        "citation_corrector": MagicMock(),
        "numeric_verifier": _numeric_verifier_real_match(),
        "config": CitationConfig(),
    }
    kwargs.update(overrides)
    return CitationVerificationPipeline(**kwargs)


@pytest.mark.parametrize(
    "miss_verdict",
    [VerificationVerdict.UNSUPPORTED, VerificationVerdict.CONTRADICTED],
)
async def test_recovers_figure_from_structured_pool_source(
    miss_verdict: VerificationVerdict,
) -> None:
    """Claim mis-cited to a figure-less metadata chunk is rescued from the
    structured pool source that holds the cell — for both a false ``unsupported``
    and a false ``contradicted`` initial verdict."""
    metadata = _ranked()
    structured = _structured()
    pipeline = _build_pipeline(miss_verdict)
    pipeline.last_evidence_pool = [metadata, structured]
    claim = _numeric_claim(_evidence_info(metadata))

    await pipeline._verify_claim_once(0, claim)

    assert claim.verification_verdict == VerificationVerdict.SUPPORTED.value
    # Citation re-pointed to the figure-bearing structured source.
    assert claim.evidence is not None
    assert claim.evidence.source_url == _STRUCTURED_URL
    assert claim.evidence.source_kind == "sql_analytics"


async def test_no_recovery_when_pool_lacks_figure() -> None:
    """No figure-bearing source in the pool → verdict and citation unchanged
    (recovery never fabricates support)."""
    metadata = _ranked()
    unrelated = _ranked(
        source_url="https://corpus/unrelated",
        quote_text="some unrelated 1,234 and 5,678 values",
        source_pool_index=1,
        evidence_pool_index=1,
    )
    pipeline = _build_pipeline(VerificationVerdict.UNSUPPORTED)
    pipeline.last_evidence_pool = [metadata, unrelated]
    claim = _numeric_claim(_evidence_info(metadata))

    await pipeline._verify_claim_once(0, claim)

    assert claim.verification_verdict == VerificationVerdict.UNSUPPORTED.value
    assert claim.evidence is not None
    assert claim.evidence.source_url == _METADATA_URL


async def test_no_recovery_when_cited_evidence_already_has_figure() -> None:
    """When the cited evidence already contains the figure the normal path
    verifies it; the pool is not searched and the citation is untouched."""
    cited = _ranked(quote_text="cell for 1945 col_2 = 4,294", has_numeric_content=True)
    decoy = _structured(source_url="https://corpus/decoy")
    pipeline = _build_pipeline(VerificationVerdict.UNSUPPORTED)
    pipeline.last_evidence_pool = [cited, decoy]
    claim = _numeric_claim(_evidence_info(cited))

    await pipeline._verify_claim_once(0, claim)

    assert claim.verification_verdict == VerificationVerdict.SUPPORTED.value
    assert claim.evidence is not None
    assert claim.evidence.source_url == _METADATA_URL  # original cited source retained


async def test_non_numeric_claim_is_not_recovered() -> None:
    """Recovery is scoped to numeric claims; a general claim with an unsupported
    verdict is left as-is even if a figure-bearing source exists in the pool."""
    metadata = _ranked()
    structured = _structured()
    pipeline = _build_pipeline(VerificationVerdict.UNSUPPORTED)
    pipeline.last_evidence_pool = [metadata, structured]
    claim = _numeric_claim(
        _evidence_info(metadata),
        claim_text="Spending rose sharply that year.",
        claim_type="general",
    )

    await pipeline._verify_claim_once(0, claim)

    assert claim.verification_verdict == VerificationVerdict.UNSUPPORTED.value
    assert claim.evidence is not None
    assert claim.evidence.source_url == _METADATA_URL
