"""Unit tests for uc_metadata — OBO-SQL Unity Catalog browse + signature helpers.

Uses a fake SqlExecutor that dispatches canned rows by matching a substring of the
issued SQL, so the pure query-building + row-parsing logic is exercised without a
warehouse.
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


def test_list_catalogs_filters_hyphenated_lowercases_and_sorts() -> None:
    ex = _executor(
        [("SHOW CATALOGS", [{"catalog": "Sales"}, {"catalog": "my-cat"}, {"catalog": "MAIN"}])]
    )
    result = uc_metadata.list_catalogs(ex)
    # hyphenated dropped (v1 unsupported), lowercased, sorted
    assert [c["name"] for c in result] == ["main", "sales"]
    assert result[0]["full_name"] == "main"


def test_list_catalogs_name_prefix_client_side() -> None:
    ex = _executor([("SHOW CATALOGS", [{"catalog": "sales"}, {"catalog": "staging"}, {"catalog": "main"}])])
    result = uc_metadata.list_catalogs(ex, name_prefix="sa")
    assert [c["name"] for c in result] == ["sales"]


def test_list_schemas_scopes_to_catalog_and_binds_prefix() -> None:
    ex = _executor([("information_schema.schemata", [{"schema_name": "finance"}])])
    result = uc_metadata.list_schemas(ex, "main", name_prefix="fin")
    assert result == [{"name": "finance", "full_name": "main.finance"}]
    sql, params, _ = ex.calls[0]  # type: ignore[attr-defined]
    assert "`main`.information_schema.schemata" in sql
    assert "schema_name LIKE :q" in sql
    assert any(getattr(p, "value", None) == "fin%" for p in params)


def test_list_schemas_rejects_bad_catalog_identifier() -> None:
    ex = _executor([])
    with pytest.raises(ValueError):
        uc_metadata.list_schemas(ex, "bad-catalog")


def test_list_functions_returns_fqn_and_return_type_hint() -> None:
    ex = _executor(
        [("information_schema.routines", [{"routine_name": "Get_Price", "data_type": "DOUBLE"}])]
    )
    result = uc_metadata.list_functions(ex, "main", "finance")
    assert result == [
        {
            "name": "get_price",
            "full_name": "main.finance.get_price",
            "description": "returns DOUBLE",
        }
    ]
    sql, params, _ = ex.calls[0]  # type: ignore[attr-defined]
    assert "routine_type = 'FUNCTION'" in sql
    assert any(getattr(p, "value", None) == "finance" for p in params)


def test_get_signature_scalar_function() -> None:
    rows = [
        {
            "specific_name": "get_price",
            "parameter_name": "ticker",
            "data_type": "STRING",
            "full_data_type": "STRING",
            "ordinal_position": 1,
            "parameter_default": None,
        },
        {
            "specific_name": "get_price",
            "parameter_name": "as_of",
            "data_type": "DATE",
            "full_data_type": "DATE",
            "ordinal_position": 2,
            "parameter_default": "current_date()",
        },
    ]
    ex = _executor([("information_schema.parameters", rows)])
    sig = uc_metadata.get_signature(ex, "main.finance.get_price")
    assert sig["scalar"] is True
    assert sig["function"] == "main.finance.get_price"
    assert sig["params"] == [
        {"name": "ticker", "type": "string", "required": True},
        {"name": "as_of", "type": "string", "required": False},
    ]


def test_get_signature_non_scalar_param_marks_not_scalar() -> None:
    rows = [
        {
            "specific_name": "classify",
            "parameter_name": "labels",
            "data_type": "ARRAY",
            "full_data_type": "ARRAY<STRING>",
            "ordinal_position": 1,
            "parameter_default": None,
        }
    ]
    ex = _executor([("information_schema.parameters", rows)])
    sig = uc_metadata.get_signature(ex, "main.ml.classify")
    assert sig["scalar"] is False
    assert sig["params"] == []


def test_get_signature_no_arg_function_is_scalar_empty() -> None:
    ex = _executor([("information_schema.parameters", [])])
    sig = uc_metadata.get_signature(ex, "main.util.now")
    assert sig["scalar"] is True
    assert sig["params"] == []


def test_get_signature_rejects_malformed_fqn() -> None:
    ex = _executor([])
    with pytest.raises(ValueError):
        uc_metadata.get_signature(ex, "not_a_fqn")
    with pytest.raises(ValueError):
        uc_metadata.get_signature(ex, "two.parts")


def test_run_parameters_query_rejects_empty_fn_names() -> None:
    ex = _executor([])
    with pytest.raises(ValueError):
        uc_metadata.run_parameters_query(ex, "main", "finance", [])
