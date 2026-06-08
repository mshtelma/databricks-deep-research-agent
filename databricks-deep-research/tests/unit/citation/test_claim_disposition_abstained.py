"""Unit tests for the confidence-aware abstained-unsupported branch (#4 fix).

When NLI returns ``verdict in (unsupported, contradicted) AND abstained=True``
with low confidence, the new code path promotes the disposition from KEEP to
REMOVE so the claim doesn't leak through the report uncited.

The behavior is gated by
``GroundingValidationConfig.abstained_unsupported_remove_threshold`` (default
0.5). Below the threshold → REMOVE; at-or-above → KEEP (legacy behavior).
"""

import asyncio
from types import SimpleNamespace

from databricks_deep_research.citation.pipeline import (
    CitationVerificationPipeline,
)
from databricks_deep_research.citation.types import ClaimInfo, ClaimRole


def _make_claim(
    *,
    abstained: bool,
    verdict: str | None,
    confidence: float | None,
    text: str = "Some assertion text.",
    position_start: int = 0,
) -> ClaimInfo:
    return ClaimInfo(
        claim_text=text,
        claim_type="general",
        position_start=position_start,
        position_end=position_start + len(text),
        abstained=abstained,
        verification_verdict=verdict,
        verification_confidence=confidence,
        claim_role=ClaimRole.FACT.value,
    )


def _build_minimal_self(*, threshold: float = 0.5) -> SimpleNamespace:
    """Build a SimpleNamespace satisfying ``process_unverified_claims``.

    We replicate just enough of CitationVerificationPipeline's attribute
    surface so the disposition lookup path runs.
    """
    from databricks_deep_research.citation.config import ClaimDisposition

    return SimpleNamespace(
        config=SimpleNamespace(
            claim_disposition=SimpleNamespace(
                abstained=ClaimDisposition.KEEP,
                unsupported=ClaimDisposition.REMOVE,
                contradicted=ClaimDisposition.REMOVE,
                partial=ClaimDisposition.KEEP,
                supported=ClaimDisposition.KEEP,
                analysis_partial=ClaimDisposition.KEEP,
                analysis_unsupported=ClaimDisposition.REMOVE,
            ),
            grounding_validation=SimpleNamespace(
                abstained_unsupported_remove_threshold=threshold,
                hedging_prefix="Based on the cited facts, ",
            ),
        ),
        _merge_overlapping_modifications=(
            CitationVerificationPipeline._merge_overlapping_modifications
        ),
    )


def _run(fake_self: SimpleNamespace, content: str, claims: list[ClaimInfo]) -> tuple[str, int, int, int]:
    return asyncio.run(
        CitationVerificationPipeline.process_unverified_claims(
            fake_self, content, claims
        )
    )


def test_abstained_unsupported_low_confidence_is_removed() -> None:
    """confidence < threshold + abstained + unsupported → REMOVE."""
    text = "Some assertion text."
    claim = _make_claim(abstained=True, verdict="unsupported", confidence=0.3, text=text)
    fake_self = _build_minimal_self(threshold=0.5)
    content = f"{text} More body after."

    _, removed, _softened, _rewritten = _run(fake_self, content, [claim])

    assert removed == 1, "Low-confidence abstained-unsupported claim should be removed"


def test_abstained_unsupported_high_confidence_is_kept() -> None:
    """confidence >= threshold + abstained + unsupported → KEEP (legacy behavior)."""
    text = "Some assertion text."
    claim = _make_claim(abstained=True, verdict="unsupported", confidence=0.8, text=text)
    fake_self = _build_minimal_self(threshold=0.5)
    content = f"{text} More body after."

    result_content, removed, _softened, _rewritten = _run(fake_self, content, [claim])

    assert removed == 0, "High-confidence abstained claims should remain in the text"
    assert text in result_content


def test_abstained_supported_is_not_promoted() -> None:
    """abstained=True but verdict=supported → KEEP, regardless of confidence."""
    text = "Some assertion text."
    claim = _make_claim(abstained=True, verdict="supported", confidence=0.1, text=text)
    fake_self = _build_minimal_self(threshold=0.5)
    content = f"{text} Body."

    _, removed, _softened, _rewritten = _run(fake_self, content, [claim])

    assert removed == 0, "Promotion must require verdict in (unsupported, contradicted)"


def test_threshold_zero_disables_promotion() -> None:
    """threshold=0.0 → never promote (claims with confidence>=0 are kept)."""
    text = "Some assertion text."
    claim = _make_claim(abstained=True, verdict="unsupported", confidence=0.0, text=text)
    fake_self = _build_minimal_self(threshold=0.0)
    content = f"{text} Body."

    _, removed, _softened, _rewritten = _run(fake_self, content, [claim])

    # confidence=0.0 is NOT < threshold=0.0, so KEEP path runs.
    assert removed == 0


def test_contradicted_low_confidence_is_also_promoted() -> None:
    """contradicted + abstained + low confidence → REMOVE."""
    text = "Some assertion text."
    claim = _make_claim(abstained=True, verdict="contradicted", confidence=0.2, text=text)
    fake_self = _build_minimal_self(threshold=0.5)
    content = f"{text} Body."

    _, removed, _softened, _rewritten = _run(fake_self, content, [claim])

    assert removed == 1
