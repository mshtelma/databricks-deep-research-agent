"""Phase 2c-1: SearchChatMemoryTool — on-demand recall of prior verified
findings, with an explicit citation contract (do NOT cite findings directly;
re-ground before citing) per Codex §7.
"""

from __future__ import annotations

import pytest

from deep_research.agent.tools.search_chat_memory import SearchChatMemoryTool

pytestmark = pytest.mark.unit


class _Finding:
    def __init__(self, content: str, confidence: str) -> None:
        self.content = content
        self.confidence = confidence


class _FakeMemory:
    def __init__(self, results: list[_Finding]) -> None:
        self._r = results

    async def search_findings(self, query: str, k: int = 5) -> list[_Finding]:
        return self._r[:k]

    def snapshot(self) -> object:
        results = self._r

        class _Snap:
            empty = not results

        return _Snap()


async def test_returns_prior_findings_with_citation_contract() -> None:
    tool = SearchChatMemoryTool(_FakeMemory([_Finding("Acme FY24 revenue grew 12%.", "high")]))
    res = await tool.execute({"query": "Acme revenue"}, None)
    assert res.success
    assert "Acme FY24 revenue grew 12%." in res.content
    # Citation contract: findings are orientation, not citable.
    assert "do not cite" in res.content.lower()
    assert res.sources == []


async def test_empty_memory_is_graceful() -> None:
    tool = SearchChatMemoryTool(_FakeMemory([]))
    res = await tool.execute({"query": "anything"}, None)
    assert res.success
    assert "no prior" in res.content.lower()


async def test_blank_query_is_handled() -> None:
    tool = SearchChatMemoryTool(_FakeMemory([_Finding("x", "high")]))
    res = await tool.execute({"query": "   "}, None)
    assert res.success


def test_definition_shape() -> None:
    tool = SearchChatMemoryTool(_FakeMemory([]))
    d = tool.definition
    assert d.name == "search_chat_memory"
    assert d.source_type == "file_search"
    assert "query" in d.parameters["properties"]
