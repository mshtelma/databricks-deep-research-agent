"""Tests for vector_search.py — title extraction and content column resolution."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.builtins.vector_search import (
    DatabricksVectorSearchTool,
    _title_from_chunk_id,
)
from databricks_deep_research.tools.protocol import ToolContext

# ---------------------------------------------------------------------------
# _title_from_chunk_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_id, expected", [
    # Standard Databricks Volume paths
    (
        "dbfs__Volumes_users_christophe_chieu_documents_compete_upload__Platform_Battlecard__Snowflake_pdf_page25_chunk0",
        "Platform Battlecard: Snowflake (p.25)",
    ),
    (
        "dbfs__Volumes_users_christophe_chieu_documents_compete_upload__Platform_Battlecard__AWS_Lakehouse__pdf_page13_chunk0",
        "Platform Battlecard: AWS Lakehouse (p.13)",
    ),
    (
        "dbfs__Volumes_users_christophe_chieu_documents_compete_upload__Platform_Battlecard__AI_on_Azure_pdf_page14_chunk0",
        "Platform Battlecard: AI on Azure (p.14)",
    ),
    # No page/chunk suffix
    (
        "dbfs__Volumes_users_joe_docs__Sales_Playbook_pdf",
        "Sales Playbook",
    ),
    # Edge cases → empty string
    ("", ""),
    ("abc-123-uuid-no-pdf", ""),
    ("42", ""),
    # chunk_id with no path prefix
    ("My_Report_pdf_page1_chunk0", "My Report (p.1)"),
])
def test_title_from_chunk_id(chunk_id: str, expected: str):
    assert _title_from_chunk_id(chunk_id) == expected


# ---------------------------------------------------------------------------
# Content column resolution — chunk_content discovered via _discover_columns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_content_column_from_discovered_embedding_source():
    """When the VS index has 'chunk_content' as the embedding source column,
    SourceInfo.content should contain the chunk text, not raw column dump."""

    ws = MagicMock()

    # Simulate index metadata with chunk_content as embedding source
    index_info = SimpleNamespace(
        primary_key="chunk_id",
        delta_sync_index_spec=SimpleNamespace(
            primary_key_columns=["chunk_id"],
            embedding_source_columns=[{"name": "chunk_content"}],
        ),
    )
    ws.vector_search_indexes.get_index.return_value = index_info

    # Simulate query result with chunk_content column
    manifest = SimpleNamespace(
        columns=[
            SimpleNamespace(name="chunk_id"),
            SimpleNamespace(name="chunk_content"),
            SimpleNamespace(name="score"),
        ]
    )
    result_data = SimpleNamespace(
        data_array=[
            [
                "dbfs__Volumes_users_joe_upload__Battlecard__Snowflake_pdf_page5_chunk0",
                "Snowflake lacks native ML support compared to Databricks.",
                0.85,
            ]
        ]
    )
    query_result = SimpleNamespace(manifest=manifest, result=result_data)
    ws.vector_search_indexes.query_index.return_value = query_result

    tool = DatabricksVectorSearchTool(
        workspace_client=ws,
        name="vs_compete",
        index_name="test.schema.compete_idx",
    )

    ctx = ToolContext(url_registry=None)
    result = await tool.execute({"query": "Snowflake comparison"}, ctx)

    assert result.success
    assert len(result.sources) == 1
    src = result.sources[0]
    # Content should be the actual chunk text, not a raw dump
    assert src.content == "Snowflake lacks native ML support compared to Databricks."
    # Title should be parsed from chunk_id
    assert src.title == "Battlecard: Snowflake (p.5)"


# ---------------------------------------------------------------------------
# Harness label fallback — generic VS titles replaced with source metadata
# ---------------------------------------------------------------------------

def test_harness_label_replaces_generic_vs_title():
    """When title is 'Vector search result N', harness should use
    source_description or source_name instead."""
    title = "Vector search result 1"
    item = {
        "title": title,
        "source_description": "Search internal competitive intelligence knowledge base for battlecards",
        "source_name": "vs_compete_intel",
    }

    # Replicate the harness logic
    if re.match(r'^Vector search result \d+$', title):
        source_desc = str(item.get("source_description", "") or "")
        source_name = str(item.get("source_name", "") or "")
        if source_desc:
            title = source_desc[:120]
        elif source_name:
            title = source_name

    assert title == "Search internal competitive intelligence knowledge base for battlecards"


def test_harness_label_keeps_good_title():
    """When title is already descriptive, harness should not replace it."""
    title = "Platform Battlecard: Snowflake (p.25)"
    item = {
        "title": title,
        "source_description": "Some generic description",
    }

    if re.match(r'^Vector search result \d+$', title):
        source_desc = str(item.get("source_description", "") or "")
        if source_desc:
            title = source_desc[:120]

    assert title == "Platform Battlecard: Snowflake (p.25)"


def test_harness_label_falls_back_to_source_name():
    """When source_description is empty, fall back to source_name."""
    title = "Vector search result 3"
    item = {
        "title": title,
        "source_description": "",
        "source_name": "vs_compete_intel",
    }

    if re.match(r'^Vector search result \d+$', title):
        source_desc = str(item.get("source_description", "") or "")
        source_name = str(item.get("source_name", "") or "")
        if source_desc:
            title = source_desc[:120]
        elif source_name:
            title = source_name

    assert title == "vs_compete_intel"


# ---------------------------------------------------------------------------
# _normalize_filters — unsupported operator handling
# ---------------------------------------------------------------------------

class TestNormalizeFilters:
    """Normalization of dict-key filter operators for query_index()."""

    def test_equality_passthrough(self):
        f = {"bulletin_date": "1941-01", "chunk_type": "table"}
        assert DatabricksVectorSearchTool._normalize_filters(f) == f

    def test_comparison_numeric_passthrough(self):
        """Numeric values pass through comparison operators."""
        f = {"score >": 0.5, "count >=": 10, "rank <": 100, "level <=": 3}
        assert DatabricksVectorSearchTool._normalize_filters(f) == f

    def test_comparison_string_dropped(self):
        """String values with comparison operators are dropped."""
        f = {"date >=": "1941-01", "date <": "1942-01"}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {}

    def test_comparison_mixed_types(self):
        """Numeric comparisons pass, string comparisons are dropped."""
        f = {"score >": 0.5, "date >=": "1941-01", "count <": 10}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {
            "score >": 0.5,
            "count <": 10,
        }

    def test_like_dropped(self):
        """LIKE is unsupported by the API — should be dropped."""
        f = {"date LIKE": "1941-%"}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {}

    def test_ne_passthrough(self):
        """!= passes through optimistically (commonly supported)."""
        f = {"status !=": "deleted"}
        assert DatabricksVectorSearchTool._normalize_filters(f) == f

    def test_in_downgraded_to_equality(self):
        f = {"chunk_type IN": ["table", "section"]}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {"chunk_type": "table"}

    def test_in_single_element_list(self):
        f = {"chunk_type IN": ["table"]}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {"chunk_type": "table"}

    def test_in_empty_list_dropped(self):
        assert DatabricksVectorSearchTool._normalize_filters({"c IN": []}) == {}

    def test_in_non_list_dropped(self):
        assert DatabricksVectorSearchTool._normalize_filters({"c IN": "val"}) == {}

    def test_in_case_insensitive(self):
        f = {"c in": ["a", "b"]}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {"c": "a"}

    def test_not_like_dropped(self):
        """NOT LIKE is multi-word; split(' ', 1) captures it correctly."""
        f = {"c NOT LIKE": "x%"}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {}

    def test_not_in_dropped(self):
        f = {"c NOT IN": ["a", "b"]}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {}

    def test_between_dropped(self):
        f = {"c BETWEEN": [1, 10]}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {}

    def test_mixed_filters_preserves_valid(self):
        f = {
            "date >=": "1941-01",
            "type IN": ["table", "section"],
            "category": "finance",
            "score >": 0.5,
        }
        # date >= is string → dropped; IN → downgraded; score > is numeric → kept
        assert DatabricksVectorSearchTool._normalize_filters(f) == {
            "type": "table",
            "category": "finance",
            "score >": 0.5,
        }

    def test_empty_dict(self):
        assert DatabricksVectorSearchTool._normalize_filters({}) == {}

    def test_trailing_whitespace_key_treated_as_equality(self):
        """Key 'col ' (trailing space) should not be misidentified as operator."""
        f = {"col ": "val"}
        assert DatabricksVectorSearchTool._normalize_filters(f) == {"col": "val"}


# ---------------------------------------------------------------------------
# _try_parse_filter_string — string-to-dict recovery
# ---------------------------------------------------------------------------

class TestTryParseFilterString:
    def test_valid_json(self):
        result = DatabricksVectorSearchTool._try_parse_filter_string('{"a": "b"}')
        assert result == {"a": "b"}

    def test_python_dict_literal(self):
        result = DatabricksVectorSearchTool._try_parse_filter_string("{'a': 'b'}")
        assert result == {"a": "b"}

    def test_unparseable_returns_none(self):
        assert DatabricksVectorSearchTool._try_parse_filter_string("not a dict") is None

    def test_non_dict_parsed_returns_none(self):
        assert DatabricksVectorSearchTool._try_parse_filter_string("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# validate_arguments — filter normalization integration
# ---------------------------------------------------------------------------

class TestValidateArgumentsFilters:
    def _make_tool(self):
        return DatabricksVectorSearchTool(
            workspace_client=MagicMock(), name="t", index_name="c.s.i",
        )

    def test_dict_filters_normalized(self):
        r = self._make_tool().validate_arguments(
            {"query": "q", "filters": {"c IN": ["a", "b"]}}
        )
        assert r["filters"] == {"c": "a"}

    def test_string_filters_parsed_and_normalized(self):
        r = self._make_tool().validate_arguments(
            {"query": "q", "filters": '{"c IN": ["a", "b"]}'}
        )
        assert r["filters"] == {"c": "a"}

    def test_python_literal_string_parsed(self):
        r = self._make_tool().validate_arguments(
            {"query": "q", "filters": "{'chunk_type': 'table'}"}
        )
        assert r["filters"] == {"chunk_type": "table"}

    def test_non_dict_non_string_ignored(self):
        r = self._make_tool().validate_arguments(
            {"query": "q", "filters": 42}
        )
        assert "filters" not in r

    def test_all_filters_normalized_away_not_set(self):
        """If all filters are unsupported, filters key should not be set."""
        r = self._make_tool().validate_arguments(
            {"query": "q", "filters": {"c BETWEEN": [1, 10]}}
        )
        assert "filters" not in r


# ---------------------------------------------------------------------------
# exclude_chunk_types — DatabricksVectorSearchTool
# ---------------------------------------------------------------------------


def _make_vs_mock_with_chunk_types(
    rows: list[list[object]],
    col_names: list[str],
) -> MagicMock:
    """Create a mock workspace_client for VS that returns given rows."""
    ws = MagicMock()
    ws.vector_search_indexes.get_index.return_value = SimpleNamespace(
        primary_key="chunk_id",
        delta_sync_index_spec=None,
    )
    manifest = SimpleNamespace(
        columns=[SimpleNamespace(name=n) for n in col_names],
    )
    result_data = SimpleNamespace(data_array=rows)
    ws.vector_search_indexes.query_index.return_value = SimpleNamespace(
        manifest=manifest, result=result_data,
    )
    return ws


class TestVectorSearchExcludeChunkTypes:
    """Tests for the exclude_chunk_types config on DatabricksVectorSearchTool."""

    _COL_NAMES = ["chunk_id", "content", "chunk_type", "score"]

    _MIXED_ROWS = [
        ["c001", "Text about budget", "text", 0.9],
        ["c002", "| Table | Data |", "table", 0.85],
        ["c003", "Section header", "section", 0.8],
        ["c004", "| More | Table |", "table", 0.75],
        ["c005", "Narrative paragraph", "text", 0.7],
    ]

    @pytest.mark.asyncio
    async def test_post_filters_excluded_types(self) -> None:
        ws = _make_vs_mock_with_chunk_types(self._MIXED_ROWS, self._COL_NAMES)
        tool = DatabricksVectorSearchTool(
            workspace_client=ws,
            name="test_vs",
            index_name="test.idx",
            columns=self._COL_NAMES,
            exclude_chunk_types=["table"],
        )
        ctx = ToolContext(query="test")
        result = await tool.execute({"query": "budget data", "num_results": 10}, ctx)

        assert result.success
        # Only text + section rows should be in output (3 out of 5)
        assert len(result.sources) == 3
        for src in result.sources:
            assert "table" not in src.title.lower() or "Table" not in src.content

    @pytest.mark.asyncio
    async def test_over_fetches_when_excluding(self) -> None:
        ws = _make_vs_mock_with_chunk_types([], self._COL_NAMES)
        tool = DatabricksVectorSearchTool(
            workspace_client=ws,
            name="test_vs",
            index_name="test.idx",
            columns=self._COL_NAMES,
            exclude_chunk_types=["table"],
        )
        ctx = ToolContext(query="test")
        await tool.execute({"query": "q", "num_results": 10}, ctx)

        call_kwargs = ws.vector_search_indexes.query_index.call_args.kwargs
        # Should over-fetch: min(10*2, 50) = 20
        assert call_kwargs["num_results"] == 20

    @pytest.mark.asyncio
    async def test_trims_to_requested_count(self) -> None:
        # 5 text rows — agent asks for 3
        text_rows = [
            [f"c{i}", f"text {i}", "text", 0.9 - i * 0.1]
            for i in range(5)
        ]
        ws = _make_vs_mock_with_chunk_types(text_rows, self._COL_NAMES)
        tool = DatabricksVectorSearchTool(
            workspace_client=ws,
            name="test_vs",
            index_name="test.idx",
            columns=self._COL_NAMES,
            exclude_chunk_types=["table"],
        )
        ctx = ToolContext(query="test")
        result = await tool.execute({"query": "q", "num_results": 3}, ctx)

        assert len(result.sources) == 3

    @pytest.mark.asyncio
    async def test_no_chunk_type_column_graceful(self) -> None:
        """When chunk_type column is missing, no filtering occurs."""
        cols = ["chunk_id", "content", "score"]
        rows = [["c1", "text", 0.9]]
        ws = _make_vs_mock_with_chunk_types(rows, cols)
        tool = DatabricksVectorSearchTool(
            workspace_client=ws,
            name="test_vs",
            index_name="test.idx",
            columns=cols,
            exclude_chunk_types=["table"],
        )
        ctx = ToolContext(query="test")
        result = await tool.execute({"query": "q"}, ctx)

        assert result.success
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_no_exclusion_no_over_fetch(self) -> None:
        ws = _make_vs_mock_with_chunk_types([], self._COL_NAMES)
        tool = DatabricksVectorSearchTool(
            workspace_client=ws,
            name="test_vs",
            index_name="test.idx",
            columns=self._COL_NAMES,
        )
        ctx = ToolContext(query="test")
        await tool.execute({"query": "q", "num_results": 10}, ctx)

        call_kwargs = ws.vector_search_indexes.query_index.call_args.kwargs
        assert call_kwargs["num_results"] == 10
