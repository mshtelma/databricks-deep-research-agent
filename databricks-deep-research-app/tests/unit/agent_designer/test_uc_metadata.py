"""Unit tests for uc_metadata — OBO-SQL Unity Catalog browse + signature helpers.

Browse/signature use SHOW/DESCRIBE (BROWSE-sufficient) rather than
information_schema (which needs USE CATALOG). A fake SqlExecutor dispatches canned
rows by matching a substring of the issued SQL, so the pure query-building +
row-parsing logic is exercised without a warehouse.
"""

from __future__ import annotations

from typing import Any

import pytest

from deep_research.agent_designer import uc_metadata


def _executor(responses: list[tuple[str, list[dict[str, Any]]]]):
    """Return a fake SqlExecutor returning the first canned rows whose pattern is
    a substring of the SQL. Records calls on ``.calls``."""
    calls: list[tuple[str, list[Any], str]] = []

    def _exec(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        calls.append((sql, params, token))
        for pattern, rows in responses:
            if pattern in sql:
                return rows
        return []

    _exec.calls = calls  # type: ignore[attr-defined]
    return _exec


def _describe(lines: list[str]) -> list[dict[str, Any]]:
    """DESCRIBE FUNCTION EXTENDED returns one text column ``function_desc``."""
    return [{"function_desc": line} for line in lines]


# --- catalogs (SHOW CATALOGS, unchanged) -----------------------------------


def test_list_catalogs_filters_hyphenated_lowercases_and_sorts() -> None:
    ex = _executor(
        [("SHOW CATALOGS", [{"catalog": "Sales"}, {"catalog": "my-cat"}, {"catalog": "MAIN"}])]
    )
    result = uc_metadata.list_catalogs(ex)
    assert [c["name"] for c in result] == ["main", "sales"]
    assert result[0]["full_name"] == "main"


def test_list_catalogs_name_prefix_client_side() -> None:
    ex = _executor(
        [("SHOW CATALOGS", [{"catalog": "sales"}, {"catalog": "staging"}, {"catalog": "main"}])]
    )
    result = uc_metadata.list_catalogs(ex, name_prefix="sa")
    assert [c["name"] for c in result] == ["sales"]


# --- schemas (SHOW SCHEMAS IN, BROWSE-sufficient) --------------------------


def test_list_schemas_uses_show_schemas_and_filters_prefix() -> None:
    ex = _executor(
        [
            (
                "SHOW SCHEMAS",
                [{"databaseName": "finance"}, {"databaseName": "fin_raw"}, {"databaseName": "sales"}],
            )
        ]
    )
    result = uc_metadata.list_schemas(ex, "MAIN", name_prefix="fin")
    # prefix-filtered client-side, lowercased, sorted ("_" sorts before "a")
    assert result == [
        {"name": "fin_raw", "full_name": "main.fin_raw"},
        {"name": "finance", "full_name": "main.finance"},
    ]
    sql, _params, _ = ex.calls[0]  # type: ignore[attr-defined]
    assert sql == "SHOW SCHEMAS IN `main`"


def test_list_schemas_rejects_bad_catalog_identifier() -> None:
    ex = _executor([])
    with pytest.raises(ValueError):
        uc_metadata.list_schemas(ex, "bad-catalog")


# --- functions (SHOW USER FUNCTIONS + session context) ---------------------


def test_list_functions_uses_show_user_functions_and_parses_fqn() -> None:
    ex = _executor(
        [
            (
                "SHOW USER FUNCTIONS",
                [
                    {"function": "main.finance.get_price"},
                    {"function": "main.finance.calc_tax"},
                    {"function": "other.sch.ignore_me"},
                ],
            )
        ]
    )
    result = uc_metadata.list_functions(ex, "MAIN", "Finance")
    # only this schema's functions, bare name kept, sorted
    assert result == [
        {"name": "calc_tax", "full_name": "main.finance.calc_tax"},
        {"name": "get_price", "full_name": "main.finance.get_price"},
    ]
    sql, _params, _ = ex.calls[0]  # type: ignore[attr-defined]
    assert sql == "SHOW USER FUNCTIONS"


def test_list_functions_name_prefix_client_side() -> None:
    ex = _executor(
        [
            (
                "SHOW USER FUNCTIONS",
                [{"function": "main.fin.get_price"}, {"function": "main.fin.calc_tax"}],
            )
        ]
    )
    result = uc_metadata.list_functions(ex, "main", "fin", name_prefix="get")
    assert [r["name"] for r in result] == ["get_price"]


def test_list_functions_rejects_bad_identifier() -> None:
    ex = _executor([])
    with pytest.raises(ValueError):
        uc_metadata.list_functions(ex, "bad-catalog", "sch")


# --- signature (DESCRIBE FUNCTION EXTENDED) --------------------------------


def test_get_signature_scalar_function() -> None:
    ex = _executor(
        [
            (
                "DESCRIBE FUNCTION",
                _describe(
                    [
                        "Function:      main.finance.get_price",
                        "Type:          SCALAR",
                        "Input:         ticker STRING 'the ticker symbol'",
                        "               as_of DATE DEFAULT current_date() 'as-of date'",
                        "Returns:       DOUBLE",
                        "Deterministic: true",
                    ]
                ),
            )
        ]
    )
    sig = uc_metadata.get_signature(ex, "main.finance.get_price")
    assert sig["scalar"] is True
    assert sig["returns_table"] is False
    assert sig["function"] == "main.finance.get_price"
    assert sig["params"] == [
        {"name": "ticker", "type": "string", "required": True},
        {"name": "as_of", "type": "string", "required": False},
    ]
    sql, _params, _ = ex.calls[0]  # type: ignore[attr-defined]
    assert sql == "DESCRIBE FUNCTION EXTENDED `main`.`finance`.`get_price`"


def test_get_signature_table_function_sets_returns_table() -> None:
    ex = _executor(
        [
            (
                "DESCRIBE FUNCTION",
                _describe(
                    [
                        "Function:      mcp.default.get_orders",
                        "Type:          TABLE",
                        "Input:         input_customer_id STRING COLLATE UTF8_BINARY 'The customer ID (format: C0001)'",
                        "Returns:       sale_id STRING",
                        "               revenue DOUBLE",
                    ]
                ),
            )
        ]
    )
    sig = uc_metadata.get_signature(ex, "mcp.default.get_orders")
    assert sig["returns_table"] is True
    assert sig["scalar"] is True
    # only the Input param (Returns continuation lines are not params)
    assert sig["params"] == [
        {"name": "input_customer_id", "type": "string", "required": True}
    ]


def test_get_signature_non_scalar_param_marks_not_scalar() -> None:
    ex = _executor(
        [
            (
                "DESCRIBE FUNCTION",
                _describe(
                    [
                        "Function:      main.ml.classify",
                        "Type:          SCALAR",
                        "Input:         labels ARRAY<STRING> 'candidate labels'",
                        "Returns:       STRING",
                    ]
                ),
            )
        ]
    )
    sig = uc_metadata.get_signature(ex, "main.ml.classify")
    assert sig["scalar"] is False
    assert sig["params"] == []


def test_get_signature_no_arg_function_is_scalar_empty() -> None:
    ex = _executor(
        [
            (
                "DESCRIBE FUNCTION",
                _describe(
                    ["Function: main.util.now", "Type: SCALAR", "Input: ()", "Returns: TIMESTAMP"]
                ),
            )
        ]
    )
    sig = uc_metadata.get_signature(ex, "main.util.now")
    assert sig["scalar"] is True
    assert sig["params"] == []
    assert sig["returns_table"] is False


def test_get_signature_rejects_malformed_fqn() -> None:
    ex = _executor([])
    with pytest.raises(ValueError):
        uc_metadata.get_signature(ex, "not_a_fqn")
    with pytest.raises(ValueError):
        uc_metadata.get_signature(ex, "two.parts")


# --- run-readiness (USE CATALOG litmus) + error classification -------------


def test_check_use_catalog_true_on_success() -> None:
    ex = _executor([("information_schema.schemata", [{"1": 1}])])
    assert uc_metadata.check_use_catalog(ex, "main") is True
    sql, _params, _ = ex.calls[0]  # type: ignore[attr-defined]
    assert "`main`.information_schema.schemata" in sql


def test_check_use_catalog_false_on_permission_error() -> None:
    def _exec(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        raise RuntimeError(
            "ServiceErrorCode.BAD_REQUEST: [INSUFFICIENT_PERMISSIONS] Insufficient "
            "privileges: User does not have USE CATALOG on Catalog 'x'. SQLSTATE: 42501"
        )

    assert uc_metadata.check_use_catalog(_exec, "x") is False


def test_check_use_catalog_reraises_non_permission_error() -> None:
    def _exec(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        raise RuntimeError("transient network blip")

    with pytest.raises(RuntimeError):
        uc_metadata.check_use_catalog(_exec, "main")


def test_classify_sql_error() -> None:
    assert (
        uc_metadata.classify_sql_error("[INSUFFICIENT_PERMISSIONS] USE CATALOG ... 42501")
        == "permission"
    )
    assert (
        uc_metadata.classify_sql_error("[TABLE_OR_VIEW_NOT_FOUND] cannot be found 42P01")
        == "not_found"
    )
    assert uc_metadata.classify_sql_error("something unexpected") == "other"
