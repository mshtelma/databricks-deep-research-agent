"""Tests for delta_read.py — DeltaReadTool and DeltaGrepTool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.builtins.delta_read import (
    DeltaContextTool,
    DeltaReadTool,
    DeltaGrepTool,
)
from databricks_deep_research.tools.protocol import ToolContext, UrlRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ws_mock(rows: list[list[object]], col_names: list[str]) -> MagicMock:
    """Create a mock workspace_client that returns the given rows."""
    ws = MagicMock()
    columns = [SimpleNamespace(name=n) for n in col_names]
    ws.statement_execution.execute_statement.return_value = SimpleNamespace(
        result=SimpleNamespace(data_array=rows),
        manifest=SimpleNamespace(
            schema=SimpleNamespace(columns=columns),
        ),
    )
    return ws


def _make_ws_empty() -> MagicMock:
    """Mock that returns no rows."""
    return _make_ws_mock([], [])


def _make_ws_error(exc: Exception) -> MagicMock:
    """Mock that raises on execute_statement."""
    ws = MagicMock()
    ws.statement_execution.execute_statement.side_effect = exc
    return ws


_COL_NAMES = ["chunk_id", "file_name", "bulletin_date", "page_info", "content", "chunk_type"]
_COLUMNS = _COL_NAMES

_SAMPLE_ROWS = [
    ["file:chunk_000", "treasury_bulletin_1941_01.txt", "1941-01", "Table 1", "National defense | 2,602", "table"],
    ["file:chunk_001", "treasury_bulletin_1941_01.txt", "1941-01", "Table 1", "International | 103", "table"],
]


def _tool_kwargs() -> dict:
    return dict(
        name="test_read",
        description="Test tool",
        table_name="main.test.chunks",
        columns=_COLUMNS,
        warehouse_id="wh-123",
        content_column="content",
        order_by="chunk_id",
    )


# ---------------------------------------------------------------------------
# DeltaReadTool
# ---------------------------------------------------------------------------


class TestDeltaReadTool:
    def test_definition_has_required_fields(self) -> None:
        ws = _make_ws_empty()
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        defn = tool.definition
        assert defn.name == "test_read"
        assert "file_name" in defn.parameters["properties"]
        assert "file_name" in defn.parameters["required"]

    def test_validate_arguments_file_name_required(self) -> None:
        ws = _make_ws_empty()
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        with pytest.raises(ValueError, match="file_name"):
            tool.validate_arguments({})

    def test_validate_arguments_limit_capped(self) -> None:
        ws = _make_ws_empty()
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        args = tool.validate_arguments({"file_name": "test.txt", "limit": 999})
        assert args["limit"] == 100  # capped at _MAX_LIMIT

    @pytest.mark.asyncio
    async def test_execute_returns_formatted_content(self) -> None:
        ws = _make_ws_mock(_SAMPLE_ROWS, _COL_NAMES)
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        result = await tool.execute({"file_name": "treasury_bulletin_1941_01.txt"}, ctx)

        assert result.success
        assert "[0]" in result.content
        assert "[1]" in result.content
        assert "2,602" in result.content
        assert len(result.sources) == 2
        assert result.data["row_count"] == 2

    @pytest.mark.asyncio
    async def test_execute_empty_results(self) -> None:
        ws = _make_ws_empty()
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        ctx = ToolContext(query="test")

        result = await tool.execute({"file_name": "nonexistent.txt"}, ctx)

        assert result.success
        assert "No chunks found" in result.content

    @pytest.mark.asyncio
    async def test_execute_with_chunk_type_filter(self) -> None:
        ws = _make_ws_mock(_SAMPLE_ROWS, _COL_NAMES)
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        ctx = ToolContext(query="test")

        await tool.execute({"file_name": "test.txt", "chunk_type": "table"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        params = call_args.kwargs.get("parameters", call_args[1].get("parameters", []))
        assert "chunk_type" in sql
        assert any(p.name == "chunk_type" for p in params)

    @pytest.mark.asyncio
    async def test_execute_handles_api_error(self) -> None:
        ws = _make_ws_error(RuntimeError("warehouse unavailable"))
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        ctx = ToolContext(query="test")

        result = await tool.execute({"file_name": "test.txt"}, ctx)

        assert not result.success
        assert "warehouse unavailable" in result.content

    @pytest.mark.asyncio
    async def test_execute_registers_urls(self) -> None:
        ws = _make_ws_mock(_SAMPLE_ROWS, _COL_NAMES)
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        registry = UrlRegistry()
        ctx = ToolContext(query="test", url_registry=registry)

        await tool.execute({"file_name": "test.txt"}, ctx)

        assert len(registry) == 2

    @pytest.mark.asyncio
    async def test_execute_parameterized_sql(self) -> None:
        """Verify SQL uses parameterized queries, never string interpolation."""
        from databricks.sdk.service.sql import StatementParameterListItem

        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaReadTool(**_tool_kwargs(), workspace_client=ws)
        ctx = ToolContext(query="test")

        # file_name with SQL injection attempt
        await tool.execute({"file_name": "'; DROP TABLE--"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        params = call_args.kwargs.get("parameters", call_args[1].get("parameters", []))
        # SQL should use :file_name placeholder, not the raw value
        assert ":file_name" in sql
        assert "DROP" not in sql
        # Params must be StatementParameterListItem, not dicts
        assert all(isinstance(p, StatementParameterListItem) for p in params)
        # Raw value should be in params only
        assert any(p.value == "'; DROP TABLE--" for p in params)


# ---------------------------------------------------------------------------
# DeltaGrepTool
# ---------------------------------------------------------------------------


class TestDeltaGrepTool:
    def _make_tool(self, ws: MagicMock | None = None) -> DeltaGrepTool:
        return DeltaGrepTool(
            **{**_tool_kwargs(), "name": "test_grep"},
            workspace_client=ws or _make_ws_empty(),
        )

    def test_definition_requires_pattern(self) -> None:
        tool = self._make_tool()
        assert "pattern" in tool.definition.parameters["required"]

    def test_validate_arguments_pattern_required(self) -> None:
        tool = self._make_tool()
        with pytest.raises(ValueError, match="pattern"):
            tool.validate_arguments({"file_name": "test.txt"})

    def test_validate_arguments_invalid_regex(self) -> None:
        tool = self._make_tool()
        with pytest.raises(ValueError, match="Invalid regex"):
            tool.validate_arguments({
                "file_name": "test.txt",
                "pattern": "[invalid",
                "mode": "regex",
            })

    def test_validate_arguments_valid_regex(self) -> None:
        tool = self._make_tool()
        args = tool.validate_arguments({
            "file_name": "test.txt",
            "pattern": r"Defense.*Military",
            "mode": "regex",
        })
        assert args["mode"] == "regex"
        assert args["pattern"] == r"Defense.*Military"

    @pytest.mark.asyncio
    async def test_execute_substring_mode_uses_ilike(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaGrepTool(**{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws)
        ctx = ToolContext(query="test")

        await tool.execute({"file_name": "test.txt", "pattern": "Defense"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        assert "ILIKE" in sql

    @pytest.mark.asyncio
    async def test_execute_regex_mode_uses_rlike(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaGrepTool(**{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws)
        ctx = ToolContext(query="test")

        await tool.execute({
            "file_name": "test.txt",
            "pattern": r"Defense.*\d+",
            "mode": "regex",
        }, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        assert "RLIKE" in sql

    @pytest.mark.asyncio
    async def test_execute_escapes_ilike_wildcards(self) -> None:
        """Verify % and _ in pattern are escaped for ILIKE mode."""
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaGrepTool(**{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws)
        ctx = ToolContext(query="test")

        await tool.execute({
            "file_name": "test.txt",
            "pattern": "100% of_total",
            "mode": "substring",
        }, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        params = call_args.kwargs.get("parameters", call_args[1].get("parameters", []))
        pattern_param = next(p for p in params if p.name == "pattern")
        # Should escape % and _ and wrap in %...%
        assert r"\%" in pattern_param.value
        assert r"\_" in pattern_param.value

    @pytest.mark.asyncio
    async def test_execute_returns_results(self) -> None:
        ws = _make_ws_mock(_SAMPLE_ROWS, _COL_NAMES)
        tool = DeltaGrepTool(**{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws)
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        result = await tool.execute({
            "file_name": "treasury_bulletin_1941_01.txt",
            "pattern": "defense",
        }, ctx)

        assert result.success
        assert len(result.sources) == 2
        assert result.data["pattern"] == "defense"

    @pytest.mark.asyncio
    async def test_execute_no_matches(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaGrepTool(**{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws)
        ctx = ToolContext(query="test")

        result = await tool.execute({
            "file_name": "test.txt",
            "pattern": "nonexistent",
        }, ctx)

        assert result.success
        assert "No matches" in result.content


# ---------------------------------------------------------------------------
# DeltaContextTool
# ---------------------------------------------------------------------------


class TestDeltaContextTool:
    def _make_tool(self, ws: MagicMock | None = None) -> DeltaContextTool:
        return DeltaContextTool(
            **{**_tool_kwargs(), "name": "test_context"},
            workspace_client=ws or _make_ws_empty(),
        )

    def test_definition_requires_file_name_and_chunk_id(self) -> None:
        tool = self._make_tool()
        defn = tool.definition
        assert "file_name" in defn.parameters["required"]
        assert "chunk_id" in defn.parameters["required"]
        assert "window" in defn.parameters["properties"]

    def test_validate_arguments_chunk_id_required(self) -> None:
        tool = self._make_tool()
        with pytest.raises(ValueError, match="chunk_id"):
            tool.validate_arguments({"file_name": "test.txt"})

    def test_validate_arguments_chunk_id_kept_as_string(self) -> None:
        tool = self._make_tool()
        args = tool.validate_arguments({"file_name": "test.txt", "chunk_id": "42"})
        assert args["chunk_id"] == "42"
        assert isinstance(args["chunk_id"], str)

    def test_validate_arguments_chunk_id_int_converted_to_string(self) -> None:
        tool = self._make_tool()
        args = tool.validate_arguments({"file_name": "test.txt", "chunk_id": 42})
        assert args["chunk_id"] == "42"
        assert isinstance(args["chunk_id"], str)

    def test_validate_arguments_chunk_id_compound(self) -> None:
        tool = self._make_tool()
        args = tool.validate_arguments({"file_name": "test.txt", "chunk_id": "test_c0042"})
        assert args["chunk_id"] == "test_c0042"

    def test_validate_arguments_window_capped(self) -> None:
        tool = self._make_tool()
        args = tool.validate_arguments({"file_name": "test.txt", "chunk_id": 10, "window": 99})
        assert args["window"] == 5  # _MAX_WINDOW

    def test_validate_arguments_default_window(self) -> None:
        tool = self._make_tool()
        args = tool.validate_arguments({"file_name": "test.txt", "chunk_id": 10})
        assert args["window"] == 2

    @pytest.mark.asyncio
    async def test_execute_returns_surrounding_chunks(self) -> None:
        ws = _make_ws_mock(_SAMPLE_ROWS, _COL_NAMES)
        tool = self._make_tool(ws)
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        result = await tool.execute(
            {"file_name": "test.txt", "chunk_id": 10, "window": 2}, ctx
        )

        assert result.success
        assert len(result.sources) == 2
        assert result.data["chunk_id"] == "10"
        assert result.data["window"] == 2

        # Verify SQL uses parameterized range query
        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        assert ":start_id" in sql
        assert ":end_id" in sql
        assert ":file_name" in sql

    @pytest.mark.asyncio
    async def test_execute_empty_results(self) -> None:
        ws = _make_ws_empty()
        tool = self._make_tool(ws)
        ctx = ToolContext(query="test")

        result = await tool.execute(
            {"file_name": "test.txt", "chunk_id": 999, "window": 2}, ctx
        )

        assert result.success
        assert "No chunks found" in result.content

    @pytest.mark.asyncio
    async def test_execute_handles_api_error(self) -> None:
        ws = _make_ws_error(RuntimeError("connection lost"))
        tool = DeltaContextTool(
            **{**_tool_kwargs(), "name": "test_context"},
            workspace_client=ws,
        )
        ctx = ToolContext(query="test")

        result = await tool.execute(
            {"file_name": "test.txt", "chunk_id": 10, "window": 2}, ctx
        )

        assert not result.success
        assert "connection lost" in result.content

    def test_definition_source_kind_is_delta_table(self) -> None:
        tool = self._make_tool()
        assert str(tool.definition.source_kind) == "delta_table"

    def test_compute_range_compound_id(self) -> None:
        start, end = DeltaContextTool._compute_range("treasury_bulletin_1941_01_c0027", 3)
        assert start == "treasury_bulletin_1941_01_c0024"
        assert end == "treasury_bulletin_1941_01_c0030"

    def test_compute_range_bare_number(self) -> None:
        start, end = DeltaContextTool._compute_range("67", 2)
        assert start == "65"
        assert end == "69"

    def test_compute_range_clamp_to_zero(self) -> None:
        start, end = DeltaContextTool._compute_range("prefix_c0002", 5)
        assert start == "prefix_c0000"
        assert end == "prefix_c0007"

    def test_compute_range_unknown_format_exact_match(self) -> None:
        start, end = DeltaContextTool._compute_range("unknown_format", 3)
        assert start == "unknown_format"
        assert end == "unknown_format"

    def test_compute_range_preserves_zero_padding_width(self) -> None:
        start, end = DeltaContextTool._compute_range("prefix_c00005", 2)
        assert start == "prefix_c00003"
        assert end == "prefix_c00007"


# ---------------------------------------------------------------------------
# exclude_chunk_types — DeltaReadTool
# ---------------------------------------------------------------------------


class TestDeltaReadToolExcludeChunkTypes:
    """Tests for the exclude_chunk_types config on DeltaReadTool."""

    @pytest.mark.asyncio
    async def test_exclude_adds_not_in_to_sql(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaReadTool(
            **_tool_kwargs(), workspace_client=ws,
            exclude_chunk_types=["table"],
        )
        ctx = ToolContext(query="test")
        await tool.execute({"file_name": "test.txt"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        params = call_args.kwargs.get("parameters", call_args[1].get("parameters", []))
        assert "NOT IN" in sql
        assert any(p.name == "excl_ct_0" and p.value == "table" for p in params)

    def test_schema_removes_excluded_from_enum(self) -> None:
        tool = DeltaReadTool(
            **_tool_kwargs(), workspace_client=_make_ws_empty(),
            exclude_chunk_types=["table"],
        )
        defn = tool.definition
        ct_prop = defn.parameters["properties"].get("chunk_type")
        assert ct_prop is not None
        assert "table" not in ct_prop["enum"]
        assert "section" in ct_prop["enum"]
        assert "text" in ct_prop["enum"]

    def test_schema_removes_param_when_all_excluded(self) -> None:
        tool = DeltaReadTool(
            **_tool_kwargs(), workspace_client=_make_ws_empty(),
            exclude_chunk_types=["table", "section", "text"],
        )
        defn = tool.definition
        assert "chunk_type" not in defn.parameters["properties"]

    @pytest.mark.asyncio
    async def test_empty_exclusion_no_not_in(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaReadTool(
            **_tool_kwargs(), workspace_client=ws,
            exclude_chunk_types=[],
        )
        ctx = ToolContext(query="test")
        await tool.execute({"file_name": "test.txt"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        assert "NOT IN" not in sql


# ---------------------------------------------------------------------------
# exclude_chunk_types — DeltaGrepTool
# ---------------------------------------------------------------------------


class TestDeltaGrepToolExcludeChunkTypes:
    """Tests for the exclude_chunk_types config on DeltaGrepTool."""

    @pytest.mark.asyncio
    async def test_exclude_adds_not_in_to_sql(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaGrepTool(
            **{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws,
            exclude_chunk_types=["table"],
        )
        ctx = ToolContext(query="test")
        await tool.execute({"file_name": "test.txt", "pattern": "something"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        params = call_args.kwargs.get("parameters", call_args[1].get("parameters", []))
        assert "NOT IN" in sql
        assert any(p.name == "excl_ct_0" and p.value == "table" for p in params)

    def test_schema_removes_excluded_from_enum(self) -> None:
        tool = DeltaGrepTool(
            **{**_tool_kwargs(), "name": "test_grep"}, workspace_client=_make_ws_empty(),
            exclude_chunk_types=["table"],
        )
        defn = tool.definition
        ct_prop = defn.parameters["properties"].get("chunk_type")
        assert ct_prop is not None
        assert "table" not in ct_prop["enum"]
        assert "section" in ct_prop["enum"]

    @pytest.mark.asyncio
    async def test_multiple_exclusions(self) -> None:
        ws = _make_ws_mock([], _COL_NAMES)
        tool = DeltaGrepTool(
            **{**_tool_kwargs(), "name": "test_grep"}, workspace_client=ws,
            exclude_chunk_types=["table", "section"],
        )
        ctx = ToolContext(query="test")
        await tool.execute({"file_name": "test.txt", "pattern": "x"}, ctx)

        call_args = ws.statement_execution.execute_statement.call_args
        sql = call_args.kwargs.get("statement", call_args[1].get("statement", ""))
        params = call_args.kwargs.get("parameters", call_args[1].get("parameters", []))
        assert "NOT IN" in sql
        assert any(p.name == "excl_ct_0" and p.value == "table" for p in params)
        assert any(p.name == "excl_ct_1" and p.value == "section" for p in params)
