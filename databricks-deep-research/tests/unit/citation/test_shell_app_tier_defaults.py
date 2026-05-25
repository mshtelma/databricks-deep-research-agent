"""Regression tests for the 2026-05-25 shell-app citation-pipeline fix.

A user-deployed shell-app surfaced a soft-warn banner ("verifier could not
judge 45 of 45 claims") on every report. Two distinct defects collided:

1. **Stage-2 framing.** ``claim_generator.py`` appended observations as
   ``"Previous content: ...\\n\\nContinue from here:"`` which primed the LLM
   to autoregress prose without ``[N]`` citation markers. Stage 4 then had
   nothing to entail against and abstained on every claim.
2. **App-only LLM tiers in framework defaults.** Pydantic defaults on
   ``IsolatedVerificationConfig`` and ``VerificationRetrievalConfig`` named
   tiers (``bulk_analysis``, ``fast``) that the bare framework
   ``FrameworkLLMClient.from_databricks(...)`` used by shell-app deployments
   does not register. Plus the synthesizer hardcoded those same literals
   regardless of agent overrides.

These tests pin the post-fix invariants so neither regression can return.
"""

from __future__ import annotations

from databricks_deep_research.citation.claim_generator import (
    _STRICT_GENERATION_PROMPT,
)
from databricks_deep_research.citation.config import (
    IsolatedVerificationConfig,
    VerificationRetrievalConfig,
)
from databricks_deep_research.llm.client import ModelTier


def _framework_canonical_tiers() -> set[str]:
    return {tier.value for tier in ModelTier}


def test_isolated_verification_defaults_are_framework_canonical() -> None:
    """Stage 4 NLI tiers must resolve under the bare framework LLM client.

    Shell-app deployments use ``FrameworkLLMClient.from_databricks(...)``
    which only knows ``simple|analytical|complex``. Defaulting to
    ``complex``/``analytical`` (or worse, app-only tiers) breaks shell-app
    citation verification and produces the 45-of-45 soft-warn banner.
    """
    cfg = IsolatedVerificationConfig()
    canonical = _framework_canonical_tiers()
    assert cfg.verification_model_tier in canonical, (
        f"verification_model_tier={cfg.verification_model_tier!r} not in framework "
        f"canonical tiers {canonical}"
    )
    assert cfg.quick_verification_tier in canonical, (
        f"quick_verification_tier={cfg.quick_verification_tier!r} not in framework "
        f"canonical tiers {canonical}"
    )


def test_isolated_verification_defaults_are_cost_aware() -> None:
    """Defaults must be cost-conscious for shell-app deployments.

    ``complex`` is reserved for explicit overrides — it would be too
    expensive as the silent default for all shell-app reports.
    """
    cfg = IsolatedVerificationConfig()
    assert cfg.verification_model_tier == "analytical"
    assert cfg.quick_verification_tier == "simple"


def test_verification_retrieval_defaults_are_framework_canonical() -> None:
    """Stage 7 ARE retrieval tiers must resolve under the bare framework
    LLM client. Pre-fix defaults included ``bulk_analysis`` and ``fast``,
    both unknown to ``FrameworkLLMClient.from_databricks(...)``.
    """
    cfg = VerificationRetrievalConfig()
    canonical = _framework_canonical_tiers()
    for field, value in (
        ("decomposition_tier", cfg.decomposition_tier),
        ("entailment_tier", cfg.entailment_tier),
        ("reconstruction_tier", cfg.reconstruction_tier),
        ("softening_tier", cfg.softening_tier),
    ):
        assert value in canonical, (
            f"{field}={value!r} not in framework canonical tiers {canonical}"
        )


def test_strict_prompt_still_requires_numeric_markers() -> None:
    """Lock the contract that the strict claim-generation prompt teaches
    ``[N]`` notation. If a future edit drops the example markers, the
    Stage-2 LLM has no spec to follow and citations vanish — exactly the
    failure mode that produced the 45-of-45 banner.
    """
    assert "[0]" in _STRICT_GENERATION_PROMPT
    assert "[1]" in _STRICT_GENERATION_PROMPT


async def test_previous_content_framing_does_not_invite_continuation() -> None:
    """Lock the prompt-fix contract.

    The pre-fix branch appended ``"Previous content:\\n...\\n\\nContinue
    from here:"`` to the user prompt. That phrasing primed the LLM to
    autoregress prose without citations, derailing the strict prompt's
    [N] requirement. The fix reframes ``previous_content`` as background
    notes and re-asserts the citation requirement.

    This test exercises ``synthesize_with_streaming`` with a sentinel
    ``previous_content``, captures the rendered user prompt sent to the
    LLM, and asserts (a) the buggy framing strings do not appear in the
    rendered prompt and (b) the corrective re-instruction does.
    """
    from unittest.mock import AsyncMock, MagicMock

    from databricks_deep_research.citation.claim_generator import (
        GenerationMode,
        InterleavedGenerationConfig,
        InterleavedGenerator,
    )
    from databricks_deep_research.citation.types import RankedEvidence
    from databricks_deep_research.llm.client import LLMResponse

    captured: dict[str, str] = {}

    async def _fake_complete(*, messages, tier, max_tokens):  # type: ignore[no-untyped-def]
        captured["prompt"] = messages[0]["content"]
        return LLMResponse(content="Anything [0].", structured=None)

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=_fake_complete)

    evidence = [
        RankedEvidence(
            source_url="https://example.com/doc",
            quote_text="A short citeable quote.",
            relevance_score=0.9,
            source_title="Doc",
            has_numeric_content=False,
        )
    ]
    gen = InterleavedGenerator(
        llm,
        config=InterleavedGenerationConfig(generation_mode=GenerationMode.STRICT),
    )

    sentinel_previous = "PRIOR_OBSERVATION_TEXT_SENTINEL_xyz123"
    async for _content, _claim in gen.synthesize_with_streaming(
        query="q",
        evidence_pool=evidence,
        target_word_count=100,
        max_tokens=500,
        previous_content=sentinel_previous,
    ):
        pass

    rendered = captured.get("prompt", "")
    assert sentinel_previous in rendered, (
        "previous_content was not threaded into the prompt at all"
    )
    # Pre-fix framing must not survive in the rendered prompt.
    assert "Previous content:" not in rendered, (
        "Pre-fix 'Previous content:' framing leaked into the rendered prompt — "
        "this primes the LLM to autoregress prose without [N] markers "
        "(45-of-45 grounding warning regression)."
    )
    assert "Continue from here:" not in rendered, (
        "Pre-fix 'Continue from here:' framing leaked into the rendered prompt — "
        "this primes the LLM to autoregress prose without [N] markers "
        "(45-of-45 grounding warning regression)."
    )
    # The replacement framing must explicitly disclaim continuation and
    # re-assert the [N] citation requirement.
    assert "background context" in rendered
    assert "[N]" in rendered
