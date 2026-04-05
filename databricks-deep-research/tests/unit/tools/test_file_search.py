"""Tests for file_search table detection and registration."""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.tools.builtins.file_search import FileSearchTool
from databricks_deep_research.tools.protocol import TableRegistry, ToolContext, UrlRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE_CHUNK = (
    "Revenue breakdown by region:\n\n"
    "| Region | Revenue |\n"
    "|---|---|\n"
    "| North America | 12B |\n"
    "| Europe | 8B |\n"
    "| Asia | 5B |\n\n"
    "Total revenue increased 15% year-over-year."
)

_PLAIN_CHUNK = "This is a plain text chunk with no tables. " * 5


class _FakeFileIndex:
    """Minimal FileIndex protocol implementation for testing."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def get_chunks(self) -> list[dict[str, Any]]:
        return self._chunks


def _make_context(*, table_registry: TableRegistry | None = None) -> ToolContext:
    return ToolContext(url_registry=UrlRegistry(), table_registry=table_registry)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileSearchTableDetection:
    @pytest.fixture()
    def index_with_table(self) -> _FakeFileIndex:
        return _FakeFileIndex([
            {"content": _TABLE_CHUNK, "source": "report.pdf", "file_id": "f1"},
            {"content": _PLAIN_CHUNK, "source": "notes.txt", "file_id": "f2"},
        ])

    @pytest.fixture()
    def index_plain(self) -> _FakeFileIndex:
        return _FakeFileIndex([
            {"content": _PLAIN_CHUNK, "source": "notes.txt", "file_id": "f1"},
        ])

    @pytest.mark.asyncio()
    async def test_tables_detected_and_registered(
        self, index_with_table: _FakeFileIndex,
    ) -> None:
        """File chunks with markdown tables → detected and registered."""
        reg = TableRegistry()
        tool = FileSearchTool(index_with_table, top_k=5)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "revenue region"}), ctx,
        )

        assert result.success
        assert result.data["num_results"] >= 1
        assert result.data.get("table_count", 0) >= 1
        assert len(reg) >= 1
        assert "table_idx=" in result.content

        entry = reg.resolve(0)
        assert entry is not None
        assert entry.source_kind == "file"
        assert entry.source_label == "report.pdf"

    @pytest.mark.asyncio()
    async def test_no_tables_in_plain_text(
        self, index_plain: _FakeFileIndex,
    ) -> None:
        """Plain text chunks → no tables detected."""
        reg = TableRegistry()
        tool = FileSearchTool(index_plain, top_k=5)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "plain text chunk"}), ctx,
        )

        assert result.success
        assert len(reg) == 0
        assert "table_count" not in result.data

    @pytest.mark.asyncio()
    async def test_no_crash_without_registry(
        self, index_with_table: _FakeFileIndex,
    ) -> None:
        """Table detection is skipped when table_registry is None."""
        tool = FileSearchTool(index_with_table, top_k=5)
        ctx = _make_context(table_registry=None)

        result = await tool.execute(
            tool.validate_arguments({"query": "revenue"}), ctx,
        )

        assert result.success
        # No crash, but no table_count since registry is None
        assert "table_count" not in result.data

    @pytest.mark.asyncio()
    async def test_capacity_overflow_handled(
        self, index_with_table: _FakeFileIndex,
    ) -> None:
        """Registry at capacity → tables skipped gracefully."""
        reg = TableRegistry(max_tables=0)
        tool = FileSearchTool(index_with_table, top_k=5)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "revenue"}), ctx,
        )

        assert result.success
        assert len(reg) == 0
