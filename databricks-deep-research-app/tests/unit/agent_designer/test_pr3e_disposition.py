"""PR3-E — claim_disposition_applier + is_negative_existence classifier
+ force-REMOVE logic gated by SYNTH_PIPELINE_V2."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.citation.claim_classifier import (
    classify_negative_existence,
)
from databricks_deep_research.citation.types import ClaimInfo

from deep_research.agent_designer.disposition_applier import (
    claim_disposition_applier,
)


# ---------------------------------------------------------------------------
# is_negative_existence classifier — happy paths and verdict scope
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal duck-typed LLM client whose ``complete`` returns a fixed dict."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        prompt: str,
        model_tier: str,
        response_format: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "model_tier": model_tier,
                "response_format": response_format,
            }
        )
        return self.payload


def _claim(text: str, *, verdict: str = "abstained") -> ClaimInfo:
    return ClaimInfo(
        claim_text=text,
        claim_type="general",
        position_start=0,
        position_end=len(text),
        verification_verdict=verdict,
    )


async def test_classifier_returns_true_on_positive_fixture() -> None:
    stub = _StubLLM(
        {"is_negative_existence": True, "reasoning": "absence assertion"}
    )
    claim = _claim("Calendar-year 1945 Army data is not available in this corpus")
    flag, reason = await classify_negative_existence(claim, stub)
    assert flag is True
    assert reason == "absence assertion"
    assert stub.calls and stub.calls[0]["model_tier"] == "fast"


async def test_classifier_returns_false_on_positive_factual_claim() -> None:
    stub = _StubLLM(
        {"is_negative_existence": False, "reasoning": "asserts a positive fact"}
    )
    claim = _claim("Army FY1945 expenditures totaled $50,337M", verdict="unsupported")
    flag, _ = await classify_negative_existence(claim, stub)
    assert flag is False


async def test_classifier_skipped_when_verdict_supported() -> None:
    stub = _StubLLM({"is_negative_existence": True, "reasoning": "x"})
    claim = _claim("anything", verdict="supported")
    flag, reason = await classify_negative_existence(claim, stub)
    assert flag is False
    assert reason is None
    # Classifier short-circuited — no LLM call.
    assert stub.calls == []


async def test_classifier_runs_on_partial_verdict() -> None:
    """ADR finding (hypothesis a): partial verdicts must be in scope so
    normalized-from-contradicted claims still get classified."""
    stub = _StubLLM({"is_negative_existence": True, "reasoning": "negation"})
    claim = _claim("no data exists", verdict="partial")
    flag, _ = await classify_negative_existence(claim, stub)
    assert flag is True


async def test_classifier_runs_when_abstained_flag_true() -> None:
    stub = _StubLLM({"is_negative_existence": True, "reasoning": "x"})
    claim = ClaimInfo(
        claim_text="X is not present",
        claim_type="general",
        position_start=0,
        position_end=15,
        verification_verdict=None,
        abstained=True,
    )
    flag, _ = await classify_negative_existence(claim, stub)
    assert flag is True


async def test_classifier_returns_false_on_malformed_response() -> None:
    stub = _StubLLM({"unexpected_key": "garbage"})
    claim = _claim("anything")
    flag, reason = await classify_negative_existence(claim, stub)
    assert flag is False
    assert reason is None


async def test_classifier_handles_string_json_response() -> None:
    stub = _StubLLM(
        '{"is_negative_existence": true, "reasoning": "ok"}'  # type: ignore[arg-type]
    )
    claim = _claim("X is unavailable")
    flag, _ = await classify_negative_existence(claim, stub)
    # The string-JSON path is handled inside the classifier's coercion.
    assert flag is True


# ---------------------------------------------------------------------------
# Force-REMOVE logic in pipeline.process_unverified_claims (PR3-E R2.2)
# ---------------------------------------------------------------------------


async def _run_disposition(
    claim: ClaimInfo,
    *,
    flag_enabled: bool,
) -> tuple[str, int, int, int]:
    from databricks_deep_research.citation.config import (
        CitationConfig,
        ClaimDisposition,
        ClaimDispositionConfig,
    )
    from databricks_deep_research.citation.pipeline import (
        CitationVerificationPipeline,
    )

    env = {"SYNTH_PIPELINE_V2": "true" if flag_enabled else "false"}
    with patch.dict(os.environ, env, clear=False):
        # Defaults: abstained=SOFTEN (the legacy hole the plan flags).
        config = CitationConfig(
            claim_disposition=ClaimDispositionConfig(
                supported=ClaimDisposition.KEEP,
                abstained=ClaimDisposition.SOFTEN,
                unsupported=ClaimDisposition.SOFTEN,
                contradicted=ClaimDisposition.REMOVE,
            ),
        )
        # Stage 8 only reads self.config; bypass __init__ which requires
        # all 7-stage protocol dependencies.
        pipeline = object.__new__(CitationVerificationPipeline)
        pipeline.config = config  # type: ignore[attr-defined]
        # Use claim text as content so position-based modifications work.
        content = claim.claim_text + " (additional sentence.)"
        claim.position_start = 0
        claim.position_end = len(claim.claim_text)
        return await pipeline.process_unverified_claims(content, [claim])


async def test_force_remove_fires_when_flag_on_and_negative_existence_true() -> None:
    claim = _claim("Calendar-year 1945 data is not available", verdict="abstained")
    claim.is_negative_existence = True
    _, removed, softened, _ = await _run_disposition(claim, flag_enabled=True)
    assert removed == 1
    assert softened == 0


async def test_force_remove_skipped_when_flag_off() -> None:
    claim = _claim("Calendar-year 1945 data is not available", verdict="abstained")
    claim.is_negative_existence = True
    _, removed, softened, _ = await _run_disposition(claim, flag_enabled=False)
    # Default policy: abstained → SOFTEN → no removal under flag-off.
    assert removed == 0
    assert softened == 1


async def test_force_remove_skipped_when_negative_existence_false() -> None:
    claim = _claim("Army FY1945 expenditures totaled $50,337M", verdict="abstained")
    claim.is_negative_existence = False
    _, removed, softened, _ = await _run_disposition(claim, flag_enabled=True)
    # Default policy still applies — flag=true alone doesn't force REMOVE.
    assert removed == 0
    assert softened == 1


async def test_force_remove_skipped_when_verdict_supported() -> None:
    """Force-REMOVE must NOT fire for fully-supported claims even if the
    flag is on and is_negative_existence is somehow set."""
    claim = _claim("Some fact", verdict="supported")
    claim.is_negative_existence = True
    _, removed, softened, _ = await _run_disposition(claim, flag_enabled=True)
    assert removed == 0


async def test_force_remove_fires_for_partial_verdict_via_adr_broadening() -> None:
    """ADR hypothesis (a) findings broadened the scope to ``partial``."""
    claim = _claim("X does not exist", verdict="partial")
    claim.is_negative_existence = True
    _, removed, _, _ = await _run_disposition(claim, flag_enabled=True)
    assert removed == 1


# ---------------------------------------------------------------------------
# claim_disposition_applier — delegation to process_unverified_claims
# ---------------------------------------------------------------------------


async def test_pipeline_classifier_batch_sets_flags() -> None:
    """The pipeline's _classify_negative_existence_batch helper iterates
    eligible claims and sets ``is_negative_existence`` on each.

    Confirms the runtime wiring path (pipeline.py) — distinct from
    classifier function unit tests above.
    """
    from databricks_deep_research.citation.pipeline import (
        _classify_negative_existence_batch,
    )

    stub = _StubLLM({"is_negative_existence": True, "reasoning": "neg"})
    claims = [
        _claim("X is not available", verdict="abstained"),
        _claim("Y has value 100", verdict="supported"),  # ineligible
        _claim("Z cannot be computed", verdict="partial"),
    ]
    await _classify_negative_existence_batch(claims, stub)
    # Eligible claims got the flag set.
    assert claims[0].is_negative_existence is True
    assert claims[2].is_negative_existence is True
    # Supported claim is skipped (no classifier call).
    assert claims[1].is_negative_existence is False


async def test_claim_disposition_applier_returns_summary() -> None:
    from databricks_deep_research.citation.config import (
        ClaimDisposition,
        ClaimDispositionConfig,
    )

    policy = ClaimDispositionConfig(
        supported=ClaimDisposition.KEEP,
        abstained=ClaimDisposition.REMOVE,
        unsupported=ClaimDisposition.SOFTEN,
        contradicted=ClaimDisposition.REMOVE,
    )
    claim = _claim("Some sentence", verdict="abstained")
    claim.position_start = 0
    claim.position_end = len("Some sentence")
    final_md, summary = await claim_disposition_applier(
        "Some sentence in the draft.", [claim], policy
    )
    assert isinstance(final_md, str)
    assert "removed_claims" in summary
    assert "softened_claims" in summary
    assert "rewritten_claims" in summary
    assert summary["total_claims"] == 1
