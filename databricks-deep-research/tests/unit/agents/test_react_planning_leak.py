"""Tests for the Phase 2 planning-text leak guard.

When a ReAct researcher exhausts its tool-call budget mid-thought, the LLM
sometimes emits a planning sentence ("Let me crawl...") as its final
``response.content`` instead of a structured observation JSON. These tests
verify (a) the heuristic that detects such leakage and (b) the output
normalizer's defense-in-depth that strips planning text rather than embedding
it as ``findings``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.execution.output_normalizer import (
    normalize_research_output,
)
from databricks_deep_research.agents.react_loop import ReactLoop, _looks_like_planning
from databricks_deep_research.llm.client import LLMResponse


class TestLooksLikePlanning:
    """Heuristic that flags planning-text leakage at ReAct exit."""

    @pytest.mark.parametrize(
        "text",
        [
            "Let me crawl some sources to get more detail.",
            "I'll search the SEC filing directly.",
            "Now I'll examine the data center segment.",
            "First, I need to check the AMD comparables.",
            "To answer this, I'll start by gathering...",
            "I need to find the latest quarterly report.",
            "I now need to add the final coverage reflector node.",
            "I've used my tool budget and still need to finish the AST.",
            "Let's start by understanding the macro context.",
        ],
    )
    def test_planning_prefixes_detected(self, text: str) -> None:
        assert _looks_like_planning(text)

    @pytest.mark.parametrize(
        "text",
        [
            '{"observation": "NVDA reported $26B revenue", "findings": "..."}',
            "[{\"key\": \"value\"}]",
            "NVDA reported $26B revenue with 76% gross margin.",
            "The data center segment grew 73% YoY in Q1.",
            "",
            "   ",
        ],
    )
    def test_real_content_not_flagged(self, text: str) -> None:
        assert not _looks_like_planning(text)

    def test_long_planning_text_not_flagged(self) -> None:
        """Length cap (>=400 chars) avoids false positives on long observations."""
        text = "Let me " + "describe this in detail. " * 30  # well over 400 chars
        assert not _looks_like_planning(text)


class TestOptInPlanningFinalOutputSuppression:
    """Non-researcher tool agents can opt into suppressing leaked planning text."""

    @pytest.mark.asyncio
    async def test_opt_in_suppresses_bare_planning_final_output(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content=(
                    "I've used my tool budget and still need to finish the AST. "
                    + "More planning details. " * 50
                ),
                tool_calls=[],
                model="test-model",
            )
        )

        loop = ReactLoop(
            llm,
            tools=[],
            node_id="architect",
            subtype="coordinator",
            suppress_planning_final_output=True,
        )
        result = await loop.execute([{"role": "user", "content": "design"}])

        assert result.content == ""
        assert result.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_opt_in_preserves_prefaced_json_patch_output(self) -> None:
        llm = AsyncMock()
        content = (
            "I'll proceed with patches keyed by lane_key.\n\n"
            "```json\n"
            '{"node_patches": {"lane_a": {"system_prompt": "Use the corpus."}}}\n'
            "```"
        )
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content=content,
                tool_calls=[],
                model="test-model",
            )
        )

        loop = ReactLoop(
            llm,
            tools=[],
            node_id="architect",
            subtype="custom",
            suppress_planning_final_output=True,
        )
        result = await loop.execute([{"role": "user", "content": "design"}])

        assert result.content == content
        assert result.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_suppression_is_off_by_default(self) -> None:
        llm = AsyncMock()
        content = "I now need to add the final coverage reflector node."
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content=content,
                tool_calls=[],
                model="test-model",
            )
        )

        loop = ReactLoop(llm, tools=[], node_id="ordinary", subtype="coordinator")
        result = await loop.execute([{"role": "user", "content": "design"}])

        assert result.content == content


class TestNormalizerStripsPlanningText:
    """Defense-in-depth: output_normalizer drops planning text rather than
    propagating it as observation/findings."""

    def _researcher_config(self) -> AgentNodeConfig:
        return AgentNodeConfig(
            subtype="researcher",
            input_keys=["query"],
            output_key="findings",
        )

    def test_planning_text_with_no_sources_yields_incomplete(self) -> None:
        config = self._researcher_config()
        result = normalize_research_output(
            "Let me crawl some sources to get more detail.",
            config,
            tool_sources=[],
        )
        assert result is not None
        assert result.findings_text == ""
        assert result.observation_text == ""
        assert result.research_status == "incomplete"
        assert result.blocking_reason == "tool_budget_exhausted"
        assert result.repair_mode == "planning_leak_dropped"
        assert result.skip_observation_writes is True

    def test_planning_text_with_sources_repairs_from_sources(self) -> None:
        """When tool sources are present, repair the observation from them
        instead of dropping outright (better than empty)."""
        config = self._researcher_config()
        sources = [
            {"url": "https://example.com/a", "title": "Source A", "snippet": "Snippet A"},
        ]
        result = normalize_research_output(
            "Let me crawl some sources to get more detail.",
            config,
            tool_sources=sources,
        )
        assert result is not None
        assert result.research_status == "ok"
        assert result.repair_mode == "source_backed_observation"
        assert "Source A" in result.findings_text or result.findings_text  # non-empty
        assert len(result.sources) == 1

    def test_real_observation_passes_through(self) -> None:
        config = self._researcher_config()
        result = normalize_research_output(
            "NVDA Q1 FY2025 revenue was $26.04B with 76% gross margin.",
            config,
            tool_sources=[],
        )
        assert result is not None
        assert result.findings_text.startswith("NVDA Q1 FY2025")
        assert result.research_status == "ok"
        assert result.repair_mode is None

    def test_dict_output_strips_planning_control_fields(self) -> None:
        config = self._researcher_config()
        result = normalize_research_output(
            {
                "planned_queries": [
                    {
                        "sub_question": "Find revenue",
                        "query": "NVDA revenue FY2025",
                        "expected_source": "10-K",
                    }
                ],
                "search_plan": "Run more searches if needed.",
                "findings": "NVDA reported concrete revenue evidence.",
                "search_queries": ["NVDA revenue FY2025"],
                "sources": [
                    {
                        "url": "https://example.com/a",
                        "title": "Source A",
                        "snippet": "Revenue evidence",
                    }
                ],
            },
            config,
            tool_sources=[],
        )
        assert result is not None
        assert "planned_queries" not in result.state_text
        assert "search_plan" not in result.state_text
        assert "search_queries" not in result.state_text
        assert "NVDA reported concrete revenue evidence" in result.state_text
        assert result.search_queries == ["NVDA revenue FY2025"]
        assert len(result.sources) == 1

    def test_empty_string_unchanged(self) -> None:
        config = self._researcher_config()
        result = normalize_research_output("", config, tool_sources=[])
        assert result is not None
        assert result.findings_text == ""
        assert result.research_status == "insufficient_data"
