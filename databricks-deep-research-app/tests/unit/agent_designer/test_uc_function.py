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
# Save-time parameter introspection
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


def _param_row(
    fn: str,
    name: str,
    data_type: str,
    ordinal: int,
    *,
    default: Any = None,
    full: str | None = None,
) -> SqlRow:
    return {
        "specific_name": fn,
        "parameter_name": name,
        "data_type": data_type,
        "full_data_type": full or data_type,
        "ordinal_position": ordinal,
        "parameter_default": default,
    }


class TestParamIntrospection:
    async def test_fills_params_and_maps_types(self) -> None:
        rows = [
            _param_row("pct_change", "old_value", "DOUBLE", 1),
            _param_row("pct_change", "new_value", "DOUBLE", 2),
        ]
        defn = _ast([_uc_tool("pct", "msh.dre_e2e.pct_change")])
        capture: dict[str, Any] = {}
        warnings = await introspect_and_fill_uc_params(
            defn, _executor(rows, capture)
        )
        params = defn["tools"][0]["config"]["params"]
        assert [p["name"] for p in params] == ["old_value", "new_value"]
        assert params[0]["type"] == "number"
        assert params[0]["required"] is True
        # Exact-match query, never a LIKE-prefix (which would ingest siblings).
        assert "specific_name IN (" in capture["sql"]
        assert "LIKE" not in capture["sql"].upper()
        assert warnings == []

    async def test_optional_param_marked_not_required(self) -> None:
        rows = [_param_row("f", "n", "INT", 1, default="10")]
        defn = _ast([_uc_tool("f", "c.s.f")])
        await introspect_and_fill_uc_params(defn, _executor(rows))
        params = defn["tools"][0]["config"]["params"]
        assert params[0]["type"] == "integer"
        assert params[0]["required"] is False

    async def test_uppercase_fqn_lowercased(self) -> None:
        defn = _ast([_uc_tool("pct", "MSH.DRE_E2E.PCT_CHANGE")])
        capture: dict[str, Any] = {}
        await introspect_and_fill_uc_params(defn, _executor([], capture))
        by_name = {p.name: p.value for p in capture["params"]}
        assert by_name["schema"] == "dre_e2e"
        assert by_name["fn0"] == "pct_change"
        assert "`msh`.information_schema.parameters" in capture["sql"]

    async def test_sibling_specific_name_not_confused(self) -> None:
        # A sibling function's rows must never leak into a different decl.
        rows = [_param_row("pct_change_2", "x", "INT", 1)]
        defn = _ast([_uc_tool("pct", "msh.dre_e2e.pct_change")])
        await introspect_and_fill_uc_params(defn, _executor(rows))
        assert defn["tools"][0]["config"]["params"] == []

    async def test_explicit_params_skip_introspection(self) -> None:
        defn = _ast(
            [_uc_tool("pct", "msh.dre_e2e.pct_change", params=[{"name": "z"}])]
        )
        called = {"n": 0}

        def _exec(sql: str, params: list[Any], token: str) -> list[SqlRow]:
            called["n"] += 1
            return []

        await introspect_and_fill_uc_params(defn, _exec)
        assert defn["tools"][0]["config"]["params"] == [{"name": "z"}]
        assert called["n"] == 0

    async def test_array_param_rejected_with_warning(self) -> None:
        rows = [_param_row("reshaper", "items", "ARRAY", 1, full="ARRAY<STRING>")]
        defn = _ast([_uc_tool("r", "c.s.reshaper")])
        warnings = await introspect_and_fill_uc_params(defn, _executor(rows))
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
