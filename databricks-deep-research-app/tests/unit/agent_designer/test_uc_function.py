"""uc_function (first-class, SQL/OBO): normalization survival, semantic
validation, tool-node refs, and save-time parameter introspection."""

from __future__ import annotations

from typing import Any

from deep_research.agent_designer.ast_normalizer import normalize_ast
from deep_research.agent_designer.semantic_validation import (
    semantic_validation_errors,
)
from deep_research.agent_designer.uc_function_introspect import (
    introspect_and_fill_uc_params,
)


def _ast(
    tools: list[dict[str, Any]],
    children: list[dict[str, Any]] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ast: dict[str, Any] = {
        "tools": tools,
        "root": {
            "id": "seq",
            "type": "sequence",
            "label": "seq",
            "children": children or [],
        },
    }
    if mcp_servers is not None:
        ast["mcp_servers"] = mcp_servers
    return ast


def _uc_tool(name: str, function: str, **config: Any) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "uc_function",
        "config": {"function": function, **config},
    }


class TestUcFunctionNormalizationSurvival:
    """uc_function is first-class now — normalization must NOT lift it to MCP."""

    def test_decl_survives_as_first_class_tool(self) -> None:
        ast = _ast([_uc_tool("sma", "finance.indicators.sma")])
        new_ast, fixes = normalize_ast(ast)

        kinds = [t["kind"] for t in new_ast["tools"]]
        assert "uc_function" in kinds
        # No managed-MCP lift happens anymore.
        assert not new_ast.get("mcp_servers")
        assert not any(str(f.kind).startswith("uc_function_lift") for f in fixes)

    def test_tool_node_ref_to_uc_function_left_intact(self) -> None:
        node = {
            "id": "t1",
            "type": "tool",
            "label": "t1",
            "config": {"ref": {"name": "sma"}, "output_key": "out"},
        }
        ast = _ast([_uc_tool("sma", "finance.indicators.sma")], children=[node])
        new_ast, _fixes = normalize_ast(ast)

        ref = new_ast["root"]["children"][0]["config"]["ref"]
        # Not rewritten to {type: mcp, ...} — resolver binds by the decl name.
        assert ref.get("name") == "sma"
        assert ref.get("type") != "mcp"


class TestUcFunctionSemanticValidation:
    def test_invalid_fqn_rejected(self) -> None:
        definition = _ast([_uc_tool("bad", "not-a-function")])
        errors = semantic_validation_errors(definition)
        assert any("catalog.schema.function" in e.message for e in errors)

    def test_valid_fqn_passes(self) -> None:
        definition = _ast([_uc_tool("sma", "finance.indicators.sma")])
        errors = semantic_validation_errors(definition)
        assert not [e for e in errors if "config.function" in (e.path or "")]

    def test_missing_function_reports_required(self) -> None:
        definition = _ast([{"name": "sma", "kind": "uc_function", "config": {}}])
        errors = semantic_validation_errors(definition)
        assert any("requires config.function" in e.message for e in errors)


class TestToolNodeRefValidation:
    def _node(self, ref: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "t1",
            "type": "tool",
            "label": "t1",
            "config": {"ref": ref, "output_key": "out"},
        }

    def test_unknown_ref_flagged(self) -> None:
        definition = _ast([], children=[self._node({"name": "ghost"})])
        errors = semantic_validation_errors(definition)
        assert any("unknown tool 'ghost'" in e.message for e in errors)

    def test_uc_function_decl_ref_passes(self) -> None:
        definition = _ast(
            [_uc_tool("sma", "finance.indicators.sma")],
            children=[self._node({"name": "sma"})],
        )
        errors = semantic_validation_errors(definition)
        assert not [e for e in errors if "unknown tool" in e.message]


# --------------------------------------------------------------------------
# Save-time signature introspection (DESCRIBE FUNCTION — BROWSE-sufficient)
# --------------------------------------------------------------------------

SqlRow = dict[str, Any]


def _executor(
    rows: list[SqlRow], capture: dict[str, Any] | None = None
) -> Any:
    def _exec(sql: str, params: list[Any], token: str) -> list[SqlRow]:
        if capture is not None:
            capture["sql"] = sql
            capture["params"] = params
        return rows

    return _exec


def _describe(lines: list[str]) -> list[SqlRow]:
    """DESCRIBE FUNCTION EXTENDED output: one text column ``function_desc``."""
    return [{"function_desc": line} for line in lines]


class TestParamIntrospection:
    async def test_fills_params_and_maps_types(self) -> None:
        lines = [
            "Function: msh.dre_e2e.pct_change",
            "Type: SCALAR",
            "Input: old_value DOUBLE 'previous'",
            "       new_value DOUBLE 'current'",
            "Returns: DOUBLE",
        ]
        defn = _ast([_uc_tool("pct", "msh.dre_e2e.pct_change")])
        capture: dict[str, Any] = {}
        warnings = await introspect_and_fill_uc_params(
            defn, _executor(_describe(lines), capture)
        )
        cfg = defn["tools"][0]["config"]
        params = cfg["params"]
        assert [p["name"] for p in params] == ["old_value", "new_value"]
        assert params[0]["type"] == "number"
        assert params[0]["required"] is True
        assert cfg["returns_table"] is False
        # DESCRIBE the exact function — never a batched/LIKE query.
        assert "DESCRIBE FUNCTION EXTENDED" in capture["sql"]
        assert warnings == []

    async def test_optional_param_marked_not_required(self) -> None:
        lines = [
            "Function: c.s.f",
            "Type: SCALAR",
            "Input: n INT DEFAULT 10 'count'",
            "Returns: INT",
        ]
        defn = _ast([_uc_tool("f", "c.s.f")])
        await introspect_and_fill_uc_params(defn, _executor(_describe(lines)))
        params = defn["tools"][0]["config"]["params"]
        assert params[0]["type"] == "integer"
        assert params[0]["required"] is False

    async def test_uppercase_fqn_lowercased(self) -> None:
        defn = _ast([_uc_tool("pct", "MSH.DRE_E2E.PCT_CHANGE")])
        capture: dict[str, Any] = {}
        await introspect_and_fill_uc_params(defn, _executor([], capture))
        assert "`msh`.`dre_e2e`.`pct_change`" in capture["sql"]

    async def test_table_function_sets_returns_table(self) -> None:
        lines = [
            "Function: mcp.default.get_orders",
            "Type: TABLE",
            "Input: cust_id STRING 'customer id'",
            "Returns: sale_id STRING",
        ]
        defn = _ast([_uc_tool("orders", "mcp.default.get_orders")])
        await introspect_and_fill_uc_params(defn, _executor(_describe(lines)))
        cfg = defn["tools"][0]["config"]
        assert cfg["returns_table"] is True
        assert cfg["params"] == [
            {"name": "cust_id", "type": "string", "required": True}
        ]

    async def test_explicit_params_preserved_but_returns_table_filled(self) -> None:
        # Author params win, but returns_table is still corrected from the signature.
        lines = [
            "Function: msh.dre_e2e.pct_change",
            "Type: TABLE",
            "Input: z STRING 'z'",
            "Returns: r STRING",
        ]
        defn = _ast(
            [_uc_tool("pct", "msh.dre_e2e.pct_change", params=[{"name": "z"}])]
        )
        await introspect_and_fill_uc_params(defn, _executor(_describe(lines)))
        cfg = defn["tools"][0]["config"]
        assert cfg["params"] == [{"name": "z"}]
        assert cfg["returns_table"] is True

    async def test_array_param_rejected_with_warning(self) -> None:
        lines = [
            "Function: c.s.reshaper",
            "Type: SCALAR",
            "Input: items ARRAY<STRING> 'labels'",
            "Returns: STRING",
        ]
        defn = _ast([_uc_tool("r", "c.s.reshaper")])
        warnings = await introspect_and_fill_uc_params(defn, _executor(_describe(lines)))
        # non-scalar => params left unset (untyped pass-through at runtime)
        assert "params" not in defn["tools"][0]["config"]
        assert any("non-scalar" in w for w in warnings)

    async def test_query_error_is_fail_soft(self) -> None:
        def _boom(sql: str, params: list[Any], token: str) -> list[SqlRow]:
            raise RuntimeError("permission denied")

        defn = _ast([_uc_tool("pct", "msh.dre_e2e.pct_change")])
        warnings = await introspect_and_fill_uc_params(defn, _boom)
        assert "params" not in defn["tools"][0]["config"]
        assert warnings  # non-empty, author-facing

    async def test_no_uc_functions_is_noop(self) -> None:
        defn = _ast([{"name": "web", "kind": "web_search", "config": {}}])
        called = {"n": 0}

        def _exec(sql: str, params: list[Any], token: str) -> list[SqlRow]:
            called["n"] += 1
            return []

        warnings = await introspect_and_fill_uc_params(defn, _exec)
        assert warnings == []
        assert called["n"] == 0
