from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from databricks_deep_research.tools.builtins.text_table import (
    StatementExecutionTableSQL,
    TableBindingRegistry,
    schema_from_describe_rows,
    wire_statement_execution_text_table_context,
)
from databricks_deep_research.tools.factory import ToolFactoryContext


def _statement_response(
    *,
    columns: list[str],
    rows: list[list[object]],
    state: str = "SUCCEEDED",
) -> SimpleNamespace:
    return SimpleNamespace(
        statement_id="stmt-1",
        status=SimpleNamespace(state=state),
        manifest=SimpleNamespace(
            schema=SimpleNamespace(
                columns=[SimpleNamespace(name=name) for name in columns]
            )
        ),
        result=SimpleNamespace(data_array=rows),
    )


def test_schema_from_describe_rows_parses_databricks_describe_output() -> None:
    schema = schema_from_describe_rows(
        "main.default.docs",
        [
            {"col_name": "id", "data_type": "BIGINT"},
            {"col_name": "body", "data_type": "STRING"},
            {"col_name": "# Partition Information", "data_type": ""},
        ],
    )

    assert schema.fqn == "main.default.docs"
    assert [col.name for col in schema.columns] == ["id", "body"]
    assert [col.data_type for col in schema.columns] == ["BIGINT", "STRING"]


def test_wire_statement_execution_text_table_context_installs_shared_dependencies() -> None:
    statement_execution = MagicMock()
    statement_execution.execute_statement.return_value = _statement_response(
        columns=["col_name", "data_type"],
        rows=[["id", "BIGINT"], ["body", "STRING"]],
    )
    workspace_client = SimpleNamespace(statement_execution=statement_execution)
    discovery_provider = object()
    ctx = ToolFactoryContext(workspace_client=workspace_client)

    result = wire_statement_execution_text_table_context(
        ctx,
        warehouse_id=" wh-123 ",
        table_discovery_provider=discovery_provider,
    )

    assert result is ctx
    assert isinstance(ctx.table_registry, TableBindingRegistry)
    assert ctx.table_discovery_provider is discovery_provider
    assert isinstance(ctx.sql_executor, StatementExecutionTableSQL)
    assert ctx.schema_cache is not None
    schema = ctx.schema_cache.get("main.default.docs", "obo-token")
    assert [col.name for col in schema.columns] == ["id", "body"]

    statement_execution.execute_statement.assert_called_once()
    call = statement_execution.execute_statement.call_args.kwargs
    assert call["statement"] == "DESCRIBE TABLE `main`.`default`.`docs`"
    assert call["warehouse_id"] == "wh-123"
    assert call["parameters"] is None


def test_wire_statement_execution_text_table_context_without_warehouse_keeps_registry(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
    ctx = ToolFactoryContext(workspace_client=object())

    wire_statement_execution_text_table_context(ctx, warehouse_id=None)

    assert isinstance(ctx.table_registry, TableBindingRegistry)
    assert ctx.table_discovery_provider is None
    assert ctx.sql_executor is None
    assert ctx.schema_cache is None
