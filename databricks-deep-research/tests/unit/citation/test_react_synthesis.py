"""Unit tests for REACT synthesis mode (citation/react_generator + synthesis_tools)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from databricks_deep_research.citation.react_generator import (
    ReactGenerator,
    _parse_react_content,
    _post_process_react_content,
)
from databricks_deep_research.citation.utils import has_numeric_content
from databricks_deep_research.citation.synthesis_tools import (
    SYNTHESIS_TOOLS,
    EvidenceSearchIndex,
    SynthesisToolExecutor,
    build_assistant_message,
    build_evidence_source_index,
)
from databricks_deep_research.citation.types import ClaimRole, InterleavedClaim, RankedEvidence
from databricks_deep_research.llm.client import LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_evidence(
    title: str = "TestSource",
    url: str = "https://example.com",
    text: str = "Sample evidence text.",
    section: str | None = None,
    relevance: float = 0.8,
    numeric: bool = False,
) -> RankedEvidence:
    return RankedEvidence(
        source_id=uuid4(),
        source_url=url,
        quote_text=text,
        start_offset=0,
        end_offset=len(text),
        section_heading=section,
        relevance_score=relevance,
        has_numeric_content=numeric,
        source_title=title,
    )


@pytest.fixture()
def evidence_pool() -> list[RankedEvidence]:
    return [
        _make_evidence(
            title="Arxiv Paper",
            url="https://arxiv.org/abs/123",
            text="GPT-4 achieves 86.4% on MMLU benchmark.",
            section="Results",
            numeric=True,
        ),
        _make_evidence(
            title="Github Repo",
            url="https://github.com/example",
            text="The library provides async/await support for Python 3.10+.",
            section="Features",
        ),
        _make_evidence(
            title="Wikipedia",
            url="https://en.wikipedia.org/wiki/AI",
            text="Artificial intelligence was founded as a discipline in 1956.",
            section="History",
        ),
    ]


@pytest.fixture()
def key_map() -> dict[int, str]:
    return {0: "Arxiv", 1: "Github", 2: "Wikipedia"}


# ---------------------------------------------------------------------------
# EvidenceSearchIndex
# ---------------------------------------------------------------------------

class TestEvidenceSearchIndex:
    def test_create_and_search(self, evidence_pool: list[RankedEvidence]) -> None:
        idx = EvidenceSearchIndex.create(evidence_pool)
        # Search is async -- run it
        import asyncio
        results = asyncio.get_event_loop().run_until_complete(
            idx.search("GPT-4 accuracy", limit=3)
        )
        assert isinstance(results, list)
        # Should return indices in valid range
        for r in results:
            assert 0 <= r < len(evidence_pool)

    def test_empty_pool(self) -> None:
        idx = EvidenceSearchIndex.create([])
        import asyncio
        results = asyncio.get_event_loop().run_until_complete(
            idx.search("anything", limit=5)
        )
        assert results == []


# ---------------------------------------------------------------------------
# SynthesisToolExecutor
# ---------------------------------------------------------------------------

class TestSynthesisToolExecutor:
    @pytest.fixture()
    def executor(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> SynthesisToolExecutor:
        search_index = EvidenceSearchIndex.create(evidence_pool)
        return SynthesisToolExecutor(evidence_pool, key_map, search_index)

    @pytest.mark.asyncio()
    async def test_search_evidence(self, executor: SynthesisToolExecutor) -> None:
        result = await executor.execute(
            "search_evidence", json.dumps({"query": "GPT accuracy", "limit": 3})
        )
        assert "Found relevant evidence" in result or "No matching" in result

    @pytest.mark.asyncio()
    async def test_read_snippet(self, executor: SynthesisToolExecutor) -> None:
        result = await executor.execute(
            "read_snippet", json.dumps({"index": 0})
        )
        assert "GPT-4 achieves 86.4%" in result
        assert "Arxiv" in result
        assert 0 in executor.read_indices

    @pytest.mark.asyncio()
    async def test_read_snippet_invalid_index(
        self, executor: SynthesisToolExecutor
    ) -> None:
        result = await executor.execute(
            "read_snippet", json.dumps({"index": 99})
        )
        assert "Invalid index" in result

    @pytest.mark.asyncio()
    async def test_invalid_json(self, executor: SynthesisToolExecutor) -> None:
        result = await executor.execute("search_evidence", "not json{")
        assert "Invalid JSON" in result

    @pytest.mark.asyncio()
    async def test_unknown_tool(self, executor: SynthesisToolExecutor) -> None:
        result = await executor.execute("unknown_tool", "{}")
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# build_assistant_message
# ---------------------------------------------------------------------------

class TestBuildAssistantMessage:
    def test_content_only(self) -> None:
        resp = LLMResponse(content="Hello world")
        msg = build_assistant_message(resp)
        assert msg == {"role": "assistant", "content": "Hello world"}

    def test_tool_calls_only(self) -> None:
        resp = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    function_name="search_evidence",
                    arguments='{"query": "test"}',
                )
            ],
        )
        msg = build_assistant_message(resp)
        assert msg["role"] == "assistant"
        assert "content" not in msg  # empty content not included
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search_evidence"

    def test_content_and_tool_calls(self) -> None:
        resp = LLMResponse(
            content="Let me search.",
            tool_calls=[
                ToolCall(id="tc_1", function_name="read_snippet", arguments='{"index": 0}')
            ],
        )
        msg = build_assistant_message(resp)
        assert msg["content"] == "Let me search."
        assert len(msg["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# build_evidence_source_index
# ---------------------------------------------------------------------------

class TestBuildEvidenceSourceIndex:
    def test_basic(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        result = build_evidence_source_index(evidence_pool, key_map)
        assert "Evidence map" in result
        assert "3 spans" in result
        assert "3 sources" in result
        assert "[Arxiv]" in result
        assert "[Github]" in result
        assert "[Wikipedia]" in result
        assert "1 numeric" in result  # Arxiv has numeric
        assert "Results" in result  # Section heading

    def test_empty_pool(self) -> None:
        result = build_evidence_source_index([], {})
        assert "0 spans" in result
        assert "0 sources" in result


# ---------------------------------------------------------------------------
# _parse_react_content
# ---------------------------------------------------------------------------

class TestParseReactContent:
    def test_cite_tags(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = '<cite key="Arxiv">GPT-4 scores 86.4% on MMLU.</cite>'
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert "GPT-4 scores 86.4% on MMLU." in assembled
        assert "[Arxiv]" in assembled
        assert len(claims) == 1
        assert claims[0].claim_role == ClaimRole.FACT.value
        assert claims[0].evidence is not None
        assert claims[0].evidence_index == 0
        assert claims[0].citation_key == "Arxiv"
        assert claims[0].claim_type == "numeric"

    def test_analysis_tags(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = "<analysis>This suggests a significant improvement.</analysis>"
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert "This suggests a significant improvement." in assembled
        assert len(claims) == 1
        assert claims[0].claim_role == ClaimRole.ANALYSIS.value

    def test_free_tags(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = "<free>## Key Findings</free>"
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert "## Key Findings" in assembled
        assert len(claims) == 1
        assert claims[0].claim_role == ClaimRole.FREE.value
        assert claims[0].from_free_block is True

    def test_unverified_tags(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = "<unverified>Training cost over $100 million.</unverified>"
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert "Training cost over $100 million." in assembled
        assert len(claims) == 1
        assert claims[0].claim_role == ClaimRole.FACT.value
        assert claims[0].confidence_score == 0.0
        assert claims[0].claim_type == "numeric"

    def test_mixed_tags(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = (
            '<free>## Introduction</free>\n'
            '<cite key="Arxiv">GPT-4 scores 86.4%.</cite>\n'
            '<analysis>This is impressive.</analysis>\n'
            '<free>## Conclusion</free>'
        )
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert len(claims) == 4
        assert claims[0].claim_role == ClaimRole.FREE.value
        assert claims[1].claim_role == ClaimRole.FACT.value
        assert claims[2].claim_role == ClaimRole.ANALYSIS.value
        assert claims[3].claim_role == ClaimRole.FREE.value
        # Positions should be ordered
        for i in range(1, len(claims)):
            assert claims[i].position_start >= claims[i - 1].position_start

    def test_unknown_key(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = '<cite key="UnknownSource">Some claim.</cite>'
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert len(claims) == 1
        assert claims[0].evidence is None
        assert claims[0].evidence_index is None

    def test_empty_tags_skipped(
        self, evidence_pool: list[RankedEvidence], key_map: dict[int, str]
    ) -> None:
        raw = '<cite key="Arxiv"></cite><analysis></analysis>'
        assembled, claims = _parse_react_content(raw, evidence_pool, key_map)
        assert len(claims) == 0


# ---------------------------------------------------------------------------
# _post_process_react_content
# ---------------------------------------------------------------------------

class TestPostProcessReactContent:
    def test_dedup_reports(self) -> None:
        raw = (
            "<free>## Introduction</free>First attempt.\n"
            "<free>## Introduction</free>Second attempt.\n"
            "<free>## Conclusion</free>Done."
        )
        result = _post_process_react_content(raw)
        assert result.count("<free>## Introduction</free>") == 1
        assert "<free>## Conclusion</free>" in result

    def test_no_dedup_single_intro(self) -> None:
        raw = "<free>## Introduction</free>Content.\n<free>## Conclusion</free>Done."
        result = _post_process_react_content(raw)
        assert result == raw

    def test_empty_content(self) -> None:
        assert _post_process_react_content("") == ""
        assert _post_process_react_content("   ") == "   "


# ---------------------------------------------------------------------------
# _has_numeric_content
# ---------------------------------------------------------------------------

class TestHasNumericContent:
    def test_percentage(self) -> None:
        assert has_numeric_content("Accuracy is 86.4%") is True

    def test_currency(self) -> None:
        assert has_numeric_content("Revenue was $5.2B") is True

    def test_large_number(self) -> None:
        assert has_numeric_content("Over 1,000,000 users") is True

    def test_written_number(self) -> None:
        assert has_numeric_content("about 5 billion parameters") is True

    def test_no_numbers(self) -> None:
        assert has_numeric_content("This is plain text.") is False


# ---------------------------------------------------------------------------
# SYNTHESIS_TOOLS structure
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_two_tools_defined(self) -> None:
        assert len(SYNTHESIS_TOOLS) == 2
        names = {t["function"]["name"] for t in SYNTHESIS_TOOLS}
        assert names == {"search_evidence", "read_snippet"}

    def test_search_evidence_schema(self) -> None:
        tool = SYNTHESIS_TOOLS[0]
        params = tool["function"]["parameters"]
        assert "query" in params["properties"]
        assert "query" in params["required"]

    def test_read_snippet_schema(self) -> None:
        tool = SYNTHESIS_TOOLS[1]
        params = tool["function"]["parameters"]
        assert "index" in params["properties"]
        assert "index" in params["required"]


# ---------------------------------------------------------------------------
# ReactGenerator end-to-end (mocked LLM)
# ---------------------------------------------------------------------------

class TestReactGeneratorE2E:
    @pytest.mark.asyncio()
    async def test_synthesize_with_mocked_llm(
        self, evidence_pool: list[RankedEvidence]
    ) -> None:
        """Mock LLM returns tool calls then content. Verify full loop produces claims."""
        # Turn 1: LLM requests search_evidence
        turn1 = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    function_name="search_evidence",
                    arguments='{"query": "GPT-4 accuracy"}',
                )
            ],
        )
        # Turn 2: LLM requests read_snippet
        turn2 = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc_2",
                    function_name="read_snippet",
                    arguments='{"index": 0}',
                )
            ],
        )
        # Turn 3: LLM writes the report
        turn3 = LLMResponse(
            content=(
                '<free>## Results</free>\n'
                '<cite key="Arxiv">GPT-4 achieves 86.4% accuracy on MMLU.</cite>\n'
                '<analysis>This represents a significant advancement.</analysis>\n'
                '<free>## Conclusion</free>\n'
                '<analysis>In summary, large models continue to improve.</analysis>'
            ),
            tool_calls=[],
        )

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[turn1, turn2, turn3])

        gen = ReactGenerator(mock_llm)
        results: list[tuple[str, InterleavedClaim | None]] = []
        async for content, claim in gen.synthesize(
            query="What is GPT-4's accuracy?",
            evidence_pool=evidence_pool,
            max_tool_calls=10,
        ):
            results.append((content, claim))

        # First yield is assembled content
        assert results[0][0]  # non-empty content
        assert results[0][1] is None

        # Remaining yields are claims (5 tags: 2 free + 1 cite + 2 analysis)
        claims = [r[1] for r in results[1:] if r[1] is not None]
        assert len(claims) == 5

        roles = [c.claim_role for c in claims]
        assert roles.count(ClaimRole.FREE.value) == 2
        assert roles.count(ClaimRole.FACT.value) == 1
        assert roles.count(ClaimRole.ANALYSIS.value) == 2

        # Verify LLM was called 3 times
        assert mock_llm.complete.call_count == 3

    @pytest.mark.asyncio()
    async def test_no_tool_calls_fallback(
        self, evidence_pool: list[RankedEvidence]
    ) -> None:
        """LLM returns content without tool calls — should still produce output."""
        response = LLMResponse(
            content="Just a plain text response without any XML tags.",
            tool_calls=[],
        )
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=response)

        gen = ReactGenerator(mock_llm)
        results: list[tuple[str, Any]] = []
        async for content, claim in gen.synthesize(
            query="test",
            evidence_pool=evidence_pool,
            max_tool_calls=10,
        ):
            results.append((content, claim))

        # Should still yield content (raw fallback)
        assert len(results) >= 1
        assert results[0][0]  # non-empty content
