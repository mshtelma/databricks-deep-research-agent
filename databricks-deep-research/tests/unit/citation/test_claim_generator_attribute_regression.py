"""Regression test for the 2026-05-25 hotfix.

`claim_generator.py:511` referenced ``self._cfg.max_evidence_chars`` when the
actual attribute on ``InterleavedGenerator`` is ``self._config`` (set at
line 336). The bug crashed every synthesizer run with
``AttributeError: 'InterleavedGenerator' object has no attribute '_cfg'``.

The existing ``tests/test_interleaved.py::test_synthesize_with_streaming_natural_mode``
test exercises this code path with a non-empty evidence pool and SHOULD have
caught the typo, but it lives outside ``tests/unit/citation/`` and was not
picked up by the citation-only filter the user ran before deploy. This file
co-locates a focused regression test with the rest of the citation suite so
the next attribute-typo regression cannot escape that filter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.citation.claim_generator import (
    GenerationMode,
    InterleavedGenerationConfig,
    InterleavedGenerator,
)
from databricks_deep_research.citation.types import (
    InterleavedClaim,
    RankedEvidence,
)
from databricks_deep_research.llm.client import LLMResponse


def _evidence(quote: str = "A short citeable quote.") -> RankedEvidence:
    return RankedEvidence(
        source_url="https://example.com/doc",
        quote_text=quote,
        relevance_score=0.9,
        source_title="Doc",
        has_numeric_content=True,
    )


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content=content, structured=None))
    return llm


@pytest.mark.asyncio
async def test_synthesize_with_streaming_reads_max_evidence_chars_via_config() -> None:
    """The prompt-formatting branch at claim_generator.py:511 must read
    self._config.max_evidence_chars without raising AttributeError.

    A long quote forces the truncation branch (``len(quote_text) > evidence_cap``)
    so the f-string at line 514 is exercised end-to-end, not just the
    attribute load.
    """
    cap = 200
    long_quote = "x" * 5000  # well above the cap, forces the truncation arm
    evidence = [_evidence(long_quote)]
    gen = InterleavedGenerator(
        _mock_llm("Generated text [0]."),
        config=InterleavedGenerationConfig(
            generation_mode=GenerationMode.NATURAL,
            max_evidence_chars=cap,
        ),
    )

    results: list[tuple[str, InterleavedClaim | None]] = []
    async for content, claim in gen.synthesize_with_streaming(
        query="q",
        evidence_pool=evidence,
        target_word_count=100,
        max_tokens=500,
    ):
        results.append((content, claim))

    # The bug raised AttributeError before the first yield. If we reach here
    # at all, line 511 successfully read self._config.max_evidence_chars.
    assert results, "expected at least one yield from synthesize_with_streaming"


@pytest.mark.asyncio
async def test_match_claim_to_evidence_reads_max_evidence_chars_via_config() -> None:
    """The sibling truncation site at claim_generator.py:417 also reads
    self._config.max_evidence_chars. Cover both sites in one place.
    """
    cap = 150
    long_quote = "y" * 4000
    evidence = [_evidence(long_quote)]
    gen = InterleavedGenerator(
        _mock_llm("0"),
        config=InterleavedGenerationConfig(max_evidence_chars=cap),
    )

    # Must not raise AttributeError.
    idx, entailment, reasoning = await gen.match_claim_to_evidence(
        "Some claim text.",
        evidence,
    )
    # We don't assert on the values (the mock LLM controls those); we only
    # assert the attribute read at line 417 succeeded.
    assert reasoning is not None


def test_interleaved_generator_stores_config_under_known_attribute() -> None:
    """Lock the attribute name. If a future refactor renames ``_config``
    to ``_cfg``, every call site in claim_generator.py needs to move with
    it — this test fails fast on attribute drift.
    """
    cfg = InterleavedGenerationConfig(max_evidence_chars=1234)
    gen = InterleavedGenerator(_mock_llm("ok"), config=cfg)
    assert hasattr(gen, "_config"), "InterleavedGenerator must expose ._config"
    assert gen._config is cfg
    assert gen._config.max_evidence_chars == 1234


def test_interleaved_generation_config_has_max_evidence_chars() -> None:
    """Lock the config field name and default. If renamed or removed, the
    pipeline factory at synthesizer.py:591 silently drops the cap and the
    1945 regression returns.
    """
    cfg = InterleavedGenerationConfig()
    assert hasattr(cfg, "max_evidence_chars")
    assert cfg.max_evidence_chars == 3000

    cfg2 = InterleavedGenerationConfig(max_evidence_chars=4500)
    assert cfg2.max_evidence_chars == 4500
