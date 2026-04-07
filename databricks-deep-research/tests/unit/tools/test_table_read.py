"""Tests for table_read.py — TableReadTool and TableRegistry integration."""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.protocol import (
    TableRegistry,
    ToolContext,
    UrlRegistry,
)

# ---------------------------------------------------------------------------
# TableRegistry
# ---------------------------------------------------------------------------


class TestTableRegistry:
    def test_register_and_resolve(self) -> None:
        reg = TableRegistry()
        tj = {"headers": [{"name": "A", "index": 0}], "rows": [{"label": "r1", "cells": {"A": "1"}}]}
        idx = reg.register(tj, source_kind="web", source_label="http://example.com")
        assert idx == 0
        entry = reg.resolve(idx)
        assert entry is not None
        assert entry.table_json == tj
        assert entry.table_json is not tj  # deep-copied on registration
        assert entry.source_kind == "web"
        assert entry.source_label == "http://example.com"

    def test_register_returns_sequential_indices(self) -> None:
        reg = TableRegistry()
        tj = {"headers": [], "rows": []}
        assert reg.register(tj) == 0
        assert reg.register(tj) == 1
        assert reg.register(tj) == 2
        assert len(reg) == 3

    def test_resolve_out_of_range(self) -> None:
        reg = TableRegistry()
        assert reg.resolve(0) is None
        assert reg.resolve(-1) is None
        assert reg.resolve(999) is None

    def test_register_validates_dict_type(self) -> None:
        reg = TableRegistry()
        with pytest.raises(TypeError, match="must be a dict"):
            reg.register("not a dict")  # type: ignore[arg-type]

    def test_register_validates_required_keys(self) -> None:
        reg = TableRegistry()
        with pytest.raises(ValueError, match="headers.*rows"):
            reg.register({"headers": []})
        with pytest.raises(ValueError, match="headers.*rows"):
            reg.register({"rows": []})

    def test_capacity_limit(self) -> None:
        reg = TableRegistry(max_tables=3)
        tj = {"headers": [], "rows": []}
        reg.register(tj)
        reg.register(tj)
        reg.register(tj)
        with pytest.raises(ValueError, match="capacity limit"):
            reg.register(tj)

    def test_repr(self) -> None:
        reg = TableRegistry()
        assert "count=0" in repr(reg)
        reg.register({"headers": [], "rows": []})
        assert "count=1" in repr(reg)

    def test_registered_table_markdown(self) -> None:
        reg = TableRegistry()
        idx = reg.register(
            {"headers": [], "rows": []},
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        )
        entry = reg.resolve(idx)
        assert entry is not None
        assert "| A | B |" in entry.markdown


# ---------------------------------------------------------------------------
# ToolContext with table_registry
# ---------------------------------------------------------------------------


class TestToolContextTableRegistry:
    def test_default_none(self) -> None:
        ctx = ToolContext()
        assert ctx.table_registry is None

    def test_with_registry(self) -> None:
        reg = TableRegistry()
        ctx = ToolContext(table_registry=reg)
        assert ctx.table_registry is reg

    def test_backward_compatible(self) -> None:
        """Existing code that doesn't pass table_registry should still work."""
        ctx = ToolContext(
            query="test",
            url_registry=UrlRegistry(),
        )
        assert ctx.table_registry is None
        assert ctx.query == "test"


# ---------------------------------------------------------------------------
# TableReadTool
# ---------------------------------------------------------------------------


class TestTableReadTool:
    @pytest.fixture()
    def sample_table_json(self) -> dict:
        return {
            "headers": [
                {"name": "Category", "parent": None, "index": 0},
                {"name": "Value", "parent": None, "index": 1},
            ],
            "rows": [
                {"label": "GDP", "cells": {"Value": "1.2T"}, "is_group_header": False, "is_total": False},
                {"label": "Population", "cells": {"Value": "330M"}, "is_group_header": False, "is_total": False},
            ],
            "row_count": 2,
            "data_row_count": 2,
        }

    @pytest.fixture()
    def registry_with_table(self, sample_table_json: dict) -> TableRegistry:
        reg = TableRegistry()
        reg.register(
            sample_table_json,
            source_kind="web",
            source_label="http://example.com/data",
            markdown="| Category | Value |\n|---|---|\n| GDP | 1.2T |",
        )
        return reg

    @pytest.mark.asyncio()
    async def test_execute_success(self, registry_with_table: TableRegistry) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        tool = TableReadTool()
        ctx = ToolContext(table_registry=registry_with_table)
        result = await tool.execute({"table_index": 0}, ctx)

        assert result.success is True
        assert "STRUCTURAL ANALYSIS" in result.content
        assert "Category" in result.content
        assert "Value" in result.content
        assert "GDP" in result.content
        assert result.data["table_index"] == 0
        assert result.data["source_kind"] == "web"

    @pytest.mark.asyncio()
    async def test_execute_not_found(self) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        tool = TableReadTool()
        reg = TableRegistry()
        ctx = ToolContext(table_registry=reg)
        result = await tool.execute({"table_index": 5}, ctx)

        assert result.success is False
        assert "not found" in result.content
        assert "empty" in result.content

    @pytest.mark.asyncio()
    async def test_execute_no_registry(self) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        tool = TableReadTool()
        ctx = ToolContext()
        result = await tool.execute({"table_index": 0}, ctx)

        assert result.success is False
        assert "No table registry" in result.content

    def test_validate_arguments(self) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        tool = TableReadTool()
        assert tool.validate_arguments({"table_index": 3}) == {"table_index": 3}
        assert tool.validate_arguments({"table_index": "5"}) == {"table_index": 5}

    def test_validate_arguments_errors(self) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        tool = TableReadTool()
        with pytest.raises(ValueError, match="required"):
            tool.validate_arguments({})
        with pytest.raises(ValueError, match="non-negative"):
            tool.validate_arguments({"table_index": -1})
        with pytest.raises(ValueError, match="integer"):
            tool.validate_arguments({"table_index": "abc"})

    def test_definition(self) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        tool = TableReadTool(name="load_table")
        defn = tool.definition
        assert defn.name == "load_table"
        assert "table_index" in defn.parameters["properties"]
        assert defn.source_kind == "builtin"

    @pytest.mark.asyncio()
    async def test_compute_injection(
        self, registry_with_table: TableRegistry
    ) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        injected: dict[str, object] = {}

        class FakeCompute:
            def inject_variable(self, name: str, value: object) -> None:
                injected[name] = value

        tool = TableReadTool(
            store_in_compute="my_table",
            compute_resolver=lambda: FakeCompute(),
        )
        ctx = ToolContext(table_registry=registry_with_table)
        result = await tool.execute({"table_index": 0}, ctx)

        assert result.success is True
        assert "my_table" in injected
        # Should be a Table instance (not raw dict)
        assert hasattr(injected["my_table"], "cell")
        assert "my_table" in result.content  # analysis mentions variable name

    @pytest.mark.asyncio()
    async def test_analysis_shows_total_rows(self) -> None:
        from databricks_deep_research.tools.builtins.table_read import TableReadTool

        reg = TableRegistry()
        reg.register(
            {
                "headers": [
                    {"name": "Category", "parent": None, "index": 0},
                    {"name": "Amount", "parent": None, "index": 1},
                ],
                "rows": [
                    {"label": "Item A", "cells": {"Amount": "100"}, "is_group_header": False, "is_total": False},
                    {"label": "Total", "cells": {"Amount": "100"}, "is_group_header": False, "is_total": True},
                ],
                "row_count": 2,
                "data_row_count": 1,
            },
            source_kind="file",
            source_label="report.csv",
        )

        tool = TableReadTool()
        ctx = ToolContext(table_registry=reg)
        result = await tool.execute({"table_index": 0}, ctx)

        assert result.success is True
        assert "Total" in result.content
        assert "file" in result.content
