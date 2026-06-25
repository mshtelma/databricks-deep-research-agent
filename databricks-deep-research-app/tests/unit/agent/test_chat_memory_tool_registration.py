"""Phase 2c-2: the orchestrator registers SearchChatMemoryTool alongside the
file tools, gated by CHAT_MEMORY_UNIFIED. Tests the extracted, pure
``_build_chat_memory_tools`` helper (the orchestrator calls it).
"""

from __future__ import annotations

import pytest

from deep_research.agent.framework_orchestrator import _build_chat_memory_tools

pytestmark = pytest.mark.unit


class _FakeMemory:
    """Minimal stand-in — the tools only store the reference at construction."""


def _names(tools: list) -> set[str]:
    return {t.definition.name for t in tools}


def test_includes_search_when_flag_on() -> None:
    names = _names(
        _build_chat_memory_tools(_FakeMemory(), file_service=None, include_search=True)
    )
    assert "search_chat_memory" in names
    assert "list_attached_files" in names
    assert "get_file_entities" in names
    assert "read_attached_file" not in names  # no file_service provided


def test_excludes_search_when_flag_off() -> None:
    names = _names(
        _build_chat_memory_tools(_FakeMemory(), file_service=None, include_search=False)
    )
    assert "search_chat_memory" not in names
    assert "list_attached_files" in names  # baseline tools unaffected


def test_read_tool_present_when_file_service_given() -> None:
    names = _names(
        _build_chat_memory_tools(_FakeMemory(), file_service=object(), include_search=False)
    )
    assert "read_attached_file" in names
