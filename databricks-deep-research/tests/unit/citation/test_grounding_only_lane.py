"""Grounding-only (cheap) lane: ``enable_isolated_verification=False``.

The cheap citation tier generates + links + renders ``[N]`` citations (Stages
1-3) but skips the expensive per-claim NLI overlay (Stage 4a/4b) AND the
verdict-based Stage 8 disposition, so claims persist as resolvable-but-
unverified. These tests pin the two framework gates plus the config plumbing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from databricks_deep_research.citation.config import ClaimDisposition
from databricks_deep_research.citation.pipeline import CitationVerificationPipeline
from databricks_deep_research.citation.types import ClaimInfo, ClaimRole


def _make_claim(
    *,
    abstained: bool = True,
    verdict: str | None = "unsupported",
    confidence: float | None = 0.3,
    text: str = "Some assertion text.",
) -> ClaimInfo:
    return ClaimInfo(
        claim_text=text,
        claim_type="general",
        position_start=0,
        position_end=len(text),
        abstained=abstained,
        verification_verdict=verdict,
        verification_confidence=confidence,
        claim_role=ClaimRole.FACT.value,
    )


# ---------------------------------------------------------------------------
# Stage 4 (verify_claims) gate — covers both the FACT (4a) and ANALYSIS (4b)
# passes, which both route through verify_claims.
# ---------------------------------------------------------------------------


def test_verify_claims_skipped_when_isolated_verification_disabled() -> None:
    """With the flag off, Stage 4 yields no events and never touches a claim."""
    fake_self = SimpleNamespace(
        config=SimpleNamespace(enabled=True, enable_isolated_verification=False)
    )

    async def _collect() -> list[object]:
        events: list[object] = []
        async for ev in CitationVerificationPipeline.verify_claims(
            fake_self, [_make_claim()], target_roles={ClaimRole.FACT.value}
        ):
            events.append(ev)
        return events

    assert asyncio.run(_collect()) == []


def test_verify_claims_still_gated_by_master_enabled() -> None:
    """The master ``enabled=False`` short-circuit is preserved (regression)."""
    fake_self = SimpleNamespace(
        config=SimpleNamespace(enabled=False, enable_isolated_verification=True)
    )

    async def _collect() -> list[object]:
        return [
            ev
            async for ev in CitationVerificationPipeline.verify_claims(
                fake_self, [_make_claim()], target_roles={ClaimRole.FACT.value}
            )
        ]

    assert asyncio.run(_collect()) == []


# ---------------------------------------------------------------------------
# Stage 8 (process_unverified_claims) gate
# ---------------------------------------------------------------------------


def _disposition_self(*, enable_isolated_verification: bool) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            enable_isolated_verification=enable_isolated_verification,
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
                abstained_unsupported_remove_threshold=0.5,
                hedging_prefix="Based on the cited facts, ",
            ),
        ),
        _merge_overlapping_modifications=(
            CitationVerificationPipeline._merge_overlapping_modifications
        ),
    )


def _run_stage8(
    fake_self: SimpleNamespace, content: str, claims: list[ClaimInfo]
) -> tuple[str, int, int, int]:
    return asyncio.run(
        CitationVerificationPipeline.process_unverified_claims(
            fake_self, content, claims
        )
    )


def test_stage8_noop_when_isolated_verification_disabled() -> None:
    """Grounding-only: a claim that WOULD be removed is kept, content unchanged."""
    text = "Some assertion text."
    content = f"{text} More body after."
    fake_self = _disposition_self(enable_isolated_verification=False)

    result_content, removed, softened, rewritten = _run_stage8(
        fake_self, content, [_make_claim(text=text, confidence=0.3)]
    )

    assert (removed, softened, rewritten) == (0, 0, 0)
    assert result_content == content  # byte-identical: no disposition applied


def test_stage8_still_disposes_when_isolated_verification_enabled() -> None:
    """Contrast: the SAME claim IS removed when verification is enabled — proves
    the flag (not some other condition) is what suppresses disposition."""
    text = "Some assertion text."
    content = f"{text} More body after."
    fake_self = _disposition_self(enable_isolated_verification=True)

    _, removed, _softened, _rewritten = _run_stage8(
        fake_self, content, [_make_claim(text=text, confidence=0.3)]
    )

    assert removed == 1


# ---------------------------------------------------------------------------
# Config plumbing — synthesizer output_schema → CitationConfig
# ---------------------------------------------------------------------------


def test_build_citation_config_reads_isolated_verification_flag() -> None:
    from databricks_deep_research.agents.builtins.synthesizer import (
        _build_citation_config,
    )
    from databricks_deep_research.agents.config import AgentNodeConfig

    cheap = AgentNodeConfig(
        subtype="synthesizer",
        output_schema={
            "enable_isolated_verification": False,
            "enable_citation_correction": False,
            "enable_numeric_qa_verification": False,
        },
    )
    cfg = _build_citation_config(cheap)
    assert cfg.enable_isolated_verification is False
    assert cfg.enable_citation_correction is False
    assert cfg.enable_numeric_qa_verification is False

    # Default (no override) preserves full verification.
    full = AgentNodeConfig(subtype="synthesizer", output_schema={})
    assert _build_citation_config(full).enable_isolated_verification is True
