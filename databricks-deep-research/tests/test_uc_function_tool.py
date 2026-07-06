"""uc_function tool: SQL build, param binding, result parsing, error contract, factory."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest

from databricks_deep_research.tools.builtins.uc_function import UCFunctionTool
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.validation import validate_workflow

SqlExec = Callable[[str, list[Any], str], list[dict[str, Any]]]


def _executor(
    rows: list[dict[str, Any]], capture: dict[str, Any] | None = None
) -> SqlExec:
    def _exec(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        if capture is not None:
            capture["sql"] = sql
            capture["params"] = params
            capture["token"] = token
            capture["thread"] = threading.current_thread().name
        return rows

    return _exec


class TestDefinitionAndSchema:
    def test_definition_source_kind_and_schema(self) -> None:
        tool = UCFunctionTool(
            name="pct",
            function_name="msh.dre_e2e.pct_change",
            sql_executor=_executor([]),
            params=[
                {"name": "old_value", "type": "number", "required": True},
                {"name": "new_value", "type": "number", "required": True},
            ],
        )
        d = tool.definition
        assert d.source_type == "uc_function"
        assert d.source_kind == "sql_analytics"
        assert d.parameters["required"] == ["old_value", "new_value"]
        assert d.parameters["properties"]["old_value"]["type"] == "number"

    def test_citeable_false_flips_source_kind_to_builtin(self) -> None:
        tool = UCFunctionTool(
            name="pct",
            function_name="c.s.f",
            sql_executor=_executor([]),
            citeable=False,
        )
        assert tool.definition.source_kind == "builtin"

    def test_bad_fqn_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="three-part"):
            UCFunctionTool(
                name="x", function_name="only.two", sql_executor=_executor([])
            )
        # hyphenated catalog names are unsupported (charset excludes '-')
        with pytest.raises(ValueError, match="three-part"):
            UCFunctionTool(
                name="x", function_name="a.b.c-d", sql_executor=_executor([])
            )


class TestValidateArguments:
    def test_declared_fills_defaults_drops_undeclared(self) -> None:
        tool = UCFunctionTool(
            name="f",
            function_name="c.s.f",
            sql_executor=_executor([]),
            params=[
                {"name": "a", "type": "integer", "required": True},
                {"name": "b", "type": "integer", "default": 5},
            ],
        )
        assert tool.validate_arguments({"a": 1, "zz": 9}) == {"a": 1, "b": 5}

    def test_missing_required_raises(self) -> None:
        tool = UCFunctionTool(
            name="f",
            function_name="c.s.f",
            sql_executor=_executor([]),
            params=[{"name": "a", "required": True}],
        )
        with pytest.raises(ValueError, match="missing required"):
            tool.validate_arguments({})

    def test_empty_params_passes_all_args_through(self) -> None:
        # No declared params (introspection not run): all provided args survive
        # (a declared-only filter would drop them -> SELECT fn() arity error).
        tool = UCFunctionTool(
            name="f", function_name="c.s.f", sql_executor=_executor([])
        )
        assert tool.validate_arguments({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}


class TestExecute:
    async def test_builds_named_arg_sql_typed_params_and_parses_result(self) -> None:
        capture: dict[str, Any] = {}
        tool = UCFunctionTool(
            name="pct",
            function_name="msh.dre_e2e.pct_change",
            sql_executor=_executor([{"result": "40.0"}], capture),
            params=[
                {"name": "old_value", "type": "number", "required": True},
                {"name": "new_value", "type": "number", "required": True},
            ],
            description="pct change",
        )
        result = await tool.execute(
            {"old_value": 100, "new_value": 140}, ToolContext()
        )
        assert result.success
        assert result.data["result"] == "40.0"
        assert capture["sql"] == (
            "SELECT `msh`.`dre_e2e`.`pct_change`"
            "(old_value => :old_value, new_value => :new_value) AS result"
        )
        by_name = {p.name: p for p in capture["params"]}
        assert by_name["old_value"].value == "100"
        assert by_name["old_value"].type == "DOUBLE"
        # Runs off the event loop (asyncio.to_thread) — a blocking 30s SQL call
        # must not stall the loop / starve SSE heartbeats.
        assert capture["thread"] != threading.main_thread().name
        assert len(result.sources) == 1
        assert result.sources[0].url.startswith(
            "uc-function://msh.dre_e2e.pct_change/"
        )
        assert result.sources[0].source_kind == "sql_analytics"

    async def test_integer_binds_bigint(self) -> None:
        capture: dict[str, Any] = {}
        tool = UCFunctionTool(
            name="f",
            function_name="c.s.f",
            sql_executor=_executor([{"result": "1"}], capture),
            params=[{"name": "n", "type": "integer", "required": True}],
        )
        await tool.execute({"n": 3_000_000_000}, ToolContext())
        assert {p.name: p.type for p in capture["params"]}["n"] == "BIGINT"

    async def test_empty_rows_yields_none_result(self) -> None:
        tool = UCFunctionTool(
            name="f", function_name="c.s.f", sql_executor=_executor([])
        )
        result = await tool.execute({}, ToolContext())
        assert result.success and result.data["result"] is None

    async def test_timeout_error_wrapped_not_raised(self) -> None:
        # A leaked TimeoutError would be caught upstream and mislabeled
        # RESEARCH_TIMEOUT — the tool must convert it to success=False.
        def _boom(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
            raise TimeoutError("statement timed out after 30s")

        tool = UCFunctionTool(name="f", function_name="c.s.f", sql_executor=_boom)
        result = await tool.execute({"a": 1}, ToolContext())
        assert result.success is False
        assert "timed out" in (result.error or "")

    async def test_runtime_error_wrapped(self) -> None:
        def _boom(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
            raise RuntimeError("FUNCTION_NOT_FOUND")

        tool = UCFunctionTool(name="f", function_name="c.s.f", sql_executor=_boom)
        result = await tool.execute({"a": 1}, ToolContext())
        assert result.success is False
        assert "FUNCTION_NOT_FOUND" in (result.error or "")

    async def test_invalid_arg_name_fails_soft(self) -> None:
        tool = UCFunctionTool(
            name="f", function_name="c.s.f", sql_executor=_executor([{"result": "1"}])
        )
        result = await tool.execute({"bad name": 1}, ToolContext())
        assert result.success is False
        assert "invalid argument name" in (result.error or "")


def _decl(config: dict[str, Any], name: str = "ucf") -> ToolDeclaration:
    return ToolDeclaration(name=name, kind="uc_function", config=config)


class TestFactory:
    async def test_requires_sql_executor(self) -> None:
        factory = BuiltinToolFactory()
        with pytest.raises(ValueError, match="sql_executor required"):
            await factory.create(_decl({"function": "c.s.f"}), ToolFactoryContext())

    async def test_constructs_with_sql_executor(self) -> None:
        factory = BuiltinToolFactory()
        ctx = ToolFactoryContext(sql_executor=_executor([{"result": "1"}]))
        tool = await factory.create(
            _decl(
                {
                    "function": "msh.dre_e2e.pct_change",
                    "params": [{"name": "a", "type": "number"}],
                }
            ),
            ctx,
        )
        assert tool.definition.source_type == "uc_function"

    async def test_missing_function_config_raises(self) -> None:
        factory = BuiltinToolFactory()
        ctx = ToolFactoryContext(sql_executor=_executor([]))
        with pytest.raises(ValueError, match="config.function"):
            await factory.create(_decl({}), ctx)


def _definition(tools: list[ToolDeclaration]) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="wf",
        name="wf",
        required_inputs=["query"],
        output_keys=["out"],
        tools=tools,
        root=WorkflowNode(
            id="a1",
            type=NodeType.agent,
            label="a1",
            config={"subtype": "researcher", "output_key": "out"},
        ),
    )


def test_uc_function_declaration_passes_validation() -> None:
    """The Phase-2 guard is gone: uc_function is a first-class executable kind."""
    definition = _definition(
        [ToolDeclaration(name="pct", kind="uc_function", config={"function": "c.s.f"})]
    )
    assert validate_workflow(definition) == []
