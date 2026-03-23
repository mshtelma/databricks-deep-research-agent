"""T107: Tests for InterleavedGenerator (Stage 2: Interleaved Generation).

Verifies:
- ReClaim interleaved generation
- Citation marker placement
- Parsing of generated content
- Evidence selection heuristic
- Constrained claim generation
- Claim-evidence matching
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.citation.claim_generator import (
    GenerationMode,
    InterleavedGenerationConfig,
    InterleavedGenerator,
    _parse_interleaved_content,
)
from databricks_deep_research.citation.utils import has_numeric_content
from databricks_deep_research.citation.types import (
    InterleavedClaim,
    RankedEvidence,
)
from databricks_deep_research.llm.client import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked_evidence(**overrides: Any) -> RankedEvidence:
    defaults: dict[str, Any] = {
        "source_id": None,
        "source_url": "https://arxiv.org/abs/123",
        "source_title": "Research Paper",
        "quote_text": "The model achieved 95% accuracy on the benchmark.",
        "start_offset": 0,
        "end_offset": 50,
        "section_heading": None,
        "relevance_score": 0.9,
        "has_numeric_content": True,
        "is_snippet_based": False,
    }
    defaults.update(overrides)
    return RankedEvidence(**defaults)


def _mock_llm(content: str = "Generated text [0].") -> MagicMock:
    """Mock FrameworkLLMClient returning plain text."""
    llm = MagicMock()
    resp = LLMResponse(content=content, structured=None)
    llm.complete = AsyncMock(return_value=resp)
    return llm


def _mock_llm_structured(structured: Any = None, content: str = "") -> MagicMock:
    llm = MagicMock()
    resp = LLMResponse(content=content, structured=structured)
    llm.complete = AsyncMock(return_value=resp)
    return llm


# ---------------------------------------------------------------------------
# T107-1: _parse_interleaved_content with numeric markers
# ---------------------------------------------------------------------------


def test_parse_content_numeric_markers() -> None:
    """Parse content with [0], [1] numeric markers."""
    evidence = [
        _make_ranked_evidence(source_url="https://arxiv.org/abs/1"),
        _make_ranked_evidence(source_url="https://github.com/repo"),
    ]

    content = "The model achieved 95% accuracy [0]. It is open source [1]."
    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 2

    # First claim: evidence linked via [0]
    assert claims[0].evidence_index == 0
    assert claims[0].evidence is not None
    assert claims[0].evidence.source_url == "https://arxiv.org/abs/1"
    assert "95% accuracy" in claims[0].claim_text
    assert claims[0].claim_type == "numeric"

    # Second claim: evidence linked via [1]
    assert claims[1].evidence_index == 1
    assert claims[1].evidence is not None


def test_parse_content_with_key_based_markers() -> None:
    """Parse content with human-readable [Arxiv], [Github] markers."""
    evidence = [
        _make_ranked_evidence(source_url="https://arxiv.org/abs/1"),
        _make_ranked_evidence(source_url="https://github.com/repo"),
    ]
    reverse_key_map = {"Arxiv": 0, "Github": 1}

    content = "The model achieved 95% accuracy [Arxiv]. It is open source [Github]."
    claims = _parse_interleaved_content(content, evidence, reverse_key_map)

    assert len(claims) == 2
    assert claims[0].citation_key == "Arxiv"
    assert claims[0].evidence_index == 0
    assert claims[1].citation_key == "Github"
    assert claims[1].evidence_index == 1


def test_parse_content_no_citations() -> None:
    """Sentences without citations should still be parsed as claims."""
    evidence = [_make_ranked_evidence()]

    content = "This sentence has no citation. Another sentence without one."
    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 2
    assert claims[0].evidence is None
    assert claims[0].evidence_index is None


def test_parse_content_multi_key_sentence() -> None:
    """A sentence with multiple citation keys should capture all keys."""
    evidence = [
        _make_ranked_evidence(source_url="https://arxiv.org/abs/1"),
        _make_ranked_evidence(source_url="https://github.com/repo"),
    ]
    reverse_key_map = {"Arxiv": 0, "Github": 1}

    content = "Growth was strong [Arxiv] [Github]."
    claims = _parse_interleaved_content(content, evidence, reverse_key_map)

    assert len(claims) == 1
    assert claims[0].citation_keys == ["Arxiv", "Github"]
    # Primary key is the first one
    assert claims[0].citation_key == "Arxiv"
    assert claims[0].evidence_indices == [0, 1]
    assert len(claims[0].evidences) == 2


def test_parse_content_empty_string() -> None:
    """Parsing empty content should return no claims."""
    claims = _parse_interleaved_content("", [], None)
    assert claims == []


def test_parse_content_position_tracking() -> None:
    """Claims should have correct position_start and position_end."""
    evidence = [_make_ranked_evidence()]
    content = "First sentence [0]. Second sentence [0]."
    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 2
    assert claims[0].position_start == 0
    assert claims[0].position_end > 0
    assert claims[1].position_start > claims[0].position_start


def test_parse_content_separates_table_from_commentary() -> None:
    """Markdown tables should not collapse into the following commentary claim."""
    evidence = [
        _make_ranked_evidence(source_url="https://example.com/q4"),
        _make_ranked_evidence(source_url="https://example.com/q1"),
    ]
    reverse_key_map = {"Q4": 0, "Q1": 1}
    content = (
        "| Quarter | Growth |\n"
        "|---------|--------|\n"
        "| Q4 | 2.4% [Q4] |\n"
        "| Q1 | 3.2% [Q1] |\n\n"
        "The comparable sales growth pattern remained within a narrow range [Q4] [Q1]."
    )

    claims = _parse_interleaved_content(content, evidence, reverse_key_map)

    assert len(claims) == 3
    assert claims[0].claim_text.startswith("| Q4 | 2.4%")
    assert claims[1].claim_text.startswith("| Q1 | 3.2%")
    assert claims[2].claim_text.startswith("The comparable sales growth pattern")


def test_parse_analysis_block_sets_claim_role() -> None:
    """Tagged analysis blocks should be parsed with an analysis role."""
    evidence = [_make_ranked_evidence()]
    content = (
        "Revenue increased 12% [0].\n\n"
        "<analysis>This may indicate stronger enterprise demand.</analysis>"
    )

    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 2
    assert claims[0].claim_role == "fact"
    assert claims[1].claim_role == "analysis"


def test_parse_free_block_sets_claim_role() -> None:
    """Tagged free blocks should be marked as structural content."""
    evidence = [_make_ranked_evidence()]
    content = "<free>Overview section.</free>\n\nRevenue increased 12% [0]."

    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 2
    assert claims[0].claim_role == "free"
    assert claims[0].from_free_block is True
    assert claims[1].claim_role == "fact"


def test_parse_content_preserves_common_abbreviations() -> None:
    """Abbreviations like U.S. should not be split into standalone fake claims."""
    evidence = [_make_ranked_evidence()]
    content = "Some sources indicate that the U.S. market expanded in 2025 [0]."

    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 1
    assert claims[0].claim_text == "Some sources indicate that the U.S. market expanded in 2025."


def test_parse_content_skips_numbered_list_fragments() -> None:
    """List markers and markdown labels should not become standalone claims."""
    evidence = [_make_ranked_evidence()]
    content = (
        "1. **Company name** for targeted research\n"
        "2. Revenue increased 12% in 2025 [0].\n"
    )

    claims = _parse_interleaved_content(content, evidence)

    assert len(claims) == 1
    assert claims[0].claim_text == "Revenue increased 12% in 2025."


# ---------------------------------------------------------------------------
# T107-2: _has_numeric_content helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Growth was 15.3%", True),
        ("Revenue was $3.2B", True),
        ("Sold 1,500,000 units", True),
        ("Worth 5 billion dollars", True),
        ("The cat sat on the mat", False),
    ],
)
def test_has_numeric_content_variants(text: str, expected: bool) -> None:
    assert has_numeric_content(text) is expected


# ---------------------------------------------------------------------------
# T107-3: select_best_evidence heuristic
# ---------------------------------------------------------------------------


def test_select_best_evidence_keyword_match() -> None:
    """select_best_evidence should pick the most relevant evidence by keyword overlap."""
    generator = InterleavedGenerator(
        _mock_llm(),
        config=InterleavedGenerationConfig(min_evidence_similarity=0.3),
    )
    evidence_pool = [
        _make_ranked_evidence(
            quote_text="The weather is sunny today",
            relevance_score=0.5,
        ),
        _make_ranked_evidence(
            quote_text="Revenue growth reached 95% accuracy on benchmark tests",
            relevance_score=0.9,
        ),
    ]

    best, index = generator.select_best_evidence(
        "benchmark accuracy", "accuracy on benchmark", evidence_pool
    )

    assert index == 1
    assert best is not None
    assert "benchmark" in best.quote_text.lower()


def test_select_best_evidence_no_match() -> None:
    """select_best_evidence returns None when no evidence meets threshold."""
    generator = InterleavedGenerator(
        _mock_llm(),
        config=InterleavedGenerationConfig(min_evidence_similarity=0.9),
    )
    evidence_pool = [
        _make_ranked_evidence(
            quote_text="Unrelated topic about weather",
            relevance_score=0.1,
        ),
    ]

    best, index = generator.select_best_evidence(
        "quantum computing", "quantum supremacy", evidence_pool
    )

    assert best is None
    assert index is None


def test_select_best_evidence_empty_pool() -> None:
    """select_best_evidence returns None for empty evidence pool."""
    generator = InterleavedGenerator(_mock_llm())

    best, index = generator.select_best_evidence("query", "context", [])
    assert best is None
    assert index is None


# ---------------------------------------------------------------------------
# T107-4: generate_constrained_claim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_constrained_claim() -> None:
    """generate_constrained_claim should produce a claim from evidence."""
    llm = _mock_llm("The model achieved 95% accuracy.")
    generator = InterleavedGenerator(llm)

    result = await generator.generate_constrained_claim(
        query="What is the model accuracy?",
        evidence=_make_ranked_evidence(),
        context="Testing the model",
    )

    assert "95%" in result
    llm.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# T107-5: match_claim_to_evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_claim_to_evidence_structured() -> None:
    """match_claim_to_evidence should return index and entailment from structured output."""
    from databricks_deep_research.citation.claim_generator import (
        ClaimEvidenceMatchOutput,
    )

    structured = ClaimEvidenceMatchOutput(
        evidence_index=0, entailment="full", reasoning="Direct match."
    )
    llm = _mock_llm_structured(structured=structured)
    generator = InterleavedGenerator(llm)

    idx, entailment, reasoning = await generator.match_claim_to_evidence(
        "Accuracy was 95%",
        [_make_ranked_evidence()],
    )

    assert idx == 0
    assert entailment == "full"
    assert "Direct match" in reasoning


@pytest.mark.asyncio
async def test_match_claim_to_evidence_empty_pool() -> None:
    """match_claim_to_evidence returns none when pool is empty."""
    generator = InterleavedGenerator(_mock_llm())

    idx, entailment, reasoning = await generator.match_claim_to_evidence(
        "Some claim", []
    )

    assert idx is None
    assert entailment == "none"


@pytest.mark.asyncio
async def test_match_claim_to_evidence_fallback_on_error() -> None:
    """match_claim_to_evidence should return 'none' on LLM failure."""
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("LLM failure"))
    generator = InterleavedGenerator(llm)

    idx, entailment, reasoning = await generator.match_claim_to_evidence(
        "Some claim", [_make_ranked_evidence()]
    )

    assert idx is None
    assert entailment == "none"


# ---------------------------------------------------------------------------
# T107-6: synthesize_with_streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_with_streaming_natural_mode() -> None:
    """synthesize_with_streaming should yield content then claims in natural mode."""
    evidence = [
        _make_ranked_evidence(source_url="https://arxiv.org/abs/1"),
    ]
    llm = _mock_llm("The model achieved 95% accuracy on the benchmark [0].")
    generator = InterleavedGenerator(
        llm,
        config=InterleavedGenerationConfig(generation_mode=GenerationMode.NATURAL),
    )

    results: list[tuple[str, InterleavedClaim | None]] = []
    async for content, claim in generator.synthesize_with_streaming(
        query="test query",
        evidence_pool=evidence,
        target_word_count=100,
        max_tokens=500,
    ):
        results.append((content, claim))

    # First yield: content with human-readable keys
    content_items = [(c, cl) for c, cl in results if c]
    claim_items = [(c, cl) for c, cl in results if cl is not None]

    assert len(content_items) >= 1
    # Content should have human-readable key, not [0]
    first_content = content_items[0][0]
    assert "[Arxiv]" in first_content or "[0]" not in first_content

    # Claims should be parsed from the content
    assert len(claim_items) >= 1


@pytest.mark.asyncio
async def test_synthesize_with_streaming_strict_mode() -> None:
    """synthesize_with_streaming should work in strict mode too."""
    evidence = [_make_ranked_evidence()]
    llm = _mock_llm("Claim text [0].")
    generator = InterleavedGenerator(
        llm,
        config=InterleavedGenerationConfig(generation_mode=GenerationMode.STRICT),
    )

    results: list[tuple[str, InterleavedClaim | None]] = []
    async for content, claim in generator.synthesize_with_streaming(
        query="query",
        evidence_pool=evidence,
    ):
        results.append((content, claim))

    assert len(results) >= 1


@pytest.mark.asyncio
async def test_synthesize_with_streaming_empty_pool() -> None:
    """synthesize_with_streaming yields nothing with empty evidence pool."""
    generator = InterleavedGenerator(_mock_llm())

    results: list[tuple[str, InterleavedClaim | None]] = []
    async for content, claim in generator.synthesize_with_streaming(
        query="query",
        evidence_pool=[],
    ):
        results.append((content, claim))

    assert len(results) == 0


# ---------------------------------------------------------------------------
# T107-7: synthesize_with_interleaving (claim-only wrapper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_with_interleaving_yields_only_claims() -> None:
    """synthesize_with_interleaving should yield only InterleavedClaim objects."""
    evidence = [_make_ranked_evidence(source_url="https://arxiv.org/abs/1")]
    llm = _mock_llm("Accuracy was 95% [0]. Another fact [0].")
    generator = InterleavedGenerator(llm)

    claims: list[InterleavedClaim] = []
    async for claim in generator.synthesize_with_interleaving(
        query="test",
        evidence_pool=evidence,
    ):
        claims.append(claim)

    assert len(claims) >= 1
    assert all(isinstance(c, InterleavedClaim) for c in claims)


# ---------------------------------------------------------------------------
# T107-8: Generation mode validation
# ---------------------------------------------------------------------------


def test_generation_mode_enum() -> None:
    """GenerationMode enum should have STRICT and NATURAL values."""
    assert GenerationMode.STRICT.value == "strict"
    assert GenerationMode.NATURAL.value == "natural"


def test_config_defaults() -> None:
    """InterleavedGenerationConfig should have sensible defaults."""
    config = InterleavedGenerationConfig()
    assert config.min_evidence_similarity == 0.5
    assert config.generation_mode == GenerationMode.NATURAL
