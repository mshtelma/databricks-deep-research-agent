"""Tests for the parameterized SQL compiler.

Covers:
- Basic SELECT * compiles with backtick-quoted FQN
- Column projection validates against schema
- Unknown column raises INVALID_COLUMN
- WHERE compiles via TableFilter and produces parameter list
- ORDER BY with ASC/DESC
- LIMIT clamp at PER_STMT_LIMIT_ROWS
- OFFSET pagination
- Invalid FQN raises INVALID_BINDING
- Filter values flow through parameters (no string concat)
"""

from __future__ import annotations

import re

import pytest
from databricks.sdk.service.sql import StatementParameterListItem

from databricks_deep_research.tools.builtins.text_table.budgets import (
    PER_STMT_LIMIT_ROWS,
)
from databricks_deep_research.tools.builtins.text_table.error_codes import (
    ErrorCode,
    ToolErrorException,
)
from databricks_deep_research.tools.builtins.text_table.filter_dsl import (
    AndFilter,
    FlatTableFilter,
    NotFilter,
    OrFilter,
)
from databricks_deep_research.tools.builtins.text_table.schema_cache import (
    Schema,
    SchemaColumn,
)
from databricks_deep_research.tools.builtins.text_table.sql_compiler import (
    compile_select,
)

SCHEMA = Schema(
    fqn="cat.s.tbl",
    columns=(
        SchemaColumn("id", "STRING"),
        SchemaColumn("content", "STRING"),
        SchemaColumn("file_name", "STRING"),
        SchemaColumn("chunk_index", "BIGINT"),
    ),
)


def test_select_star_when_columns_none() -> None:
    sql, params = compile_select("cat.s.tbl", SCHEMA)
    assert "SELECT *" in sql
    assert "FROM `cat`.`s`.`tbl`" in sql
    assert params == []


def test_select_with_projected_columns() -> None:
    sql, params = compile_select(
        "cat.s.tbl",
        SCHEMA,
        columns=["id", "content"],
        limit=10,
    )
    assert "SELECT `id`, `content`" in sql
    assert "FROM `cat`.`s`.`tbl`" in sql
    assert "LIMIT 10" in sql
    assert params == []


def test_unknown_column_raises_invalid_column() -> None:
    with pytest.raises(ToolErrorException) as exc:
        compile_select(
            "cat.s.tbl",
            SCHEMA,
            columns=["nope"],
        )
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_unknown_order_by_column_raises_invalid_column() -> None:
    with pytest.raises(ToolErrorException) as exc:
        compile_select(
            "cat.s.tbl",
            SCHEMA,
            order_by=["unknown"],
        )
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_where_compiles_via_table_filter_with_parameters() -> None:
    flt = FlatTableFilter(eq={"file_name": "doc.pdf"})
    sql, params = compile_select(
        "cat.s.tbl",
        SCHEMA,
        where=flt,
        limit=5,
    )
    assert "WHERE" in sql
    assert isinstance(params, list)
    assert len(params) == 1
    assert isinstance(params[0], StatementParameterListItem)
    assert params[0].value == "doc.pdf"


def test_where_filter_value_not_in_sql_text() -> None:
    """SQL injection guard — user-supplied filter value must NOT appear literally."""
    secret_value = "'; DROP TABLE users; --"
    flt = FlatTableFilter(eq={"file_name": secret_value})
    sql, params = compile_select(
        "cat.s.tbl",
        SCHEMA,
        where=flt,
    )
    assert secret_value not in sql
    assert not re.search(re.escape("DROP TABLE"), sql)
    assert any(p.value == secret_value for p in params)


def test_text_search_compiles_parameterized_like() -> None:
    query = "apple%_\\needle"
    sql, params = compile_select(
        "cat.s.tbl",
        SCHEMA,
        text_search=("content", query),
        limit=5,
    )
    assert "LOWER(`content`) LIKE LOWER(:p_text_search_1)" in sql
    assert "ESCAPE '\\\\'" in sql
    assert query not in sql
    assert len(params) == 1
    assert params[0].name == "p_text_search_1"
    assert params[0].value == "%apple\\%\\_\\\\needle%"


def test_text_search_and_where_are_and_combined() -> None:
    flt = FlatTableFilter(eq={"file_name": "doc.pdf"})
    sql, params = compile_select(
        "cat.s.tbl",
        SCHEMA,
        where=flt,
        text_search=("content", "revenue"),
        limit=5,
    )
    assert "WHERE" in sql
    assert " AND " in sql
    assert "LOWER(`content`) LIKE LOWER(:p_text_search_1)" in sql
    assert len(params) == 2


def test_order_by_asc_and_desc() -> None:
    sql, _ = compile_select(
        "cat.s.tbl",
        SCHEMA,
        order_by=["file_name", "-chunk_index"],
    )
    assert "ORDER BY `file_name` ASC, `chunk_index` DESC" in sql


def test_limit_clamped_at_per_stmt_limit_rows() -> None:
    sql, _ = compile_select(
        "cat.s.tbl",
        SCHEMA,
        limit=999_999,
    )
    assert f"LIMIT {PER_STMT_LIMIT_ROWS}" in sql


def test_no_limit_when_none_passed() -> None:
    sql, _ = compile_select("cat.s.tbl", SCHEMA)
    # When limit is None we still cap at PER_STMT_LIMIT_ROWS to be safe.
    assert f"LIMIT {PER_STMT_LIMIT_ROWS}" in sql


def test_offset_pagination() -> None:
    sql, _ = compile_select(
        "cat.s.tbl",
        SCHEMA,
        limit=10,
        offset=20,
    )
    assert "LIMIT 10" in sql
    assert "OFFSET 20" in sql


def test_zero_offset_omitted() -> None:
    sql, _ = compile_select(
        "cat.s.tbl",
        SCHEMA,
        limit=10,
        offset=0,
    )
    assert "OFFSET" not in sql


def test_invalid_fqn_two_parts_raises_invalid_binding() -> None:
    with pytest.raises(ToolErrorException) as exc:
        compile_select("schema.tbl", SCHEMA)
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING


def test_invalid_fqn_four_parts_raises_invalid_binding() -> None:
    with pytest.raises(ToolErrorException) as exc:
        compile_select("a.b.c.d", SCHEMA)
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING


def test_invalid_fqn_empty_part_raises_invalid_binding() -> None:
    with pytest.raises(ToolErrorException) as exc:
        compile_select("cat..tbl", SCHEMA)
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING


def test_backtick_in_column_rejected_as_invalid_column() -> None:
    sql, _ = compile_select("cat.s.tbl", SCHEMA, columns=["id"])
    assert "`id`" in sql
    # A column name containing a backtick is not in the schema, so INVALID_COLUMN.
    with pytest.raises(ToolErrorException) as exc:
        compile_select("cat.s.tbl", SCHEMA, columns=["id`"])
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_complete_select_sql_shape() -> None:
    """Smoke test: full statement assembles in canonical order."""
    flt = FlatTableFilter(eq={"file_name": "doc.pdf"})
    sql, params = compile_select(
        "cat.s.tbl",
        SCHEMA,
        columns=["id", "content"],
        where=flt,
        order_by=["chunk_index"],
        limit=20,
        offset=40,
    )
    # Expected ordering: SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT ... OFFSET ...
    select_idx = sql.index("SELECT")
    from_idx = sql.index("FROM")
    where_idx = sql.index("WHERE")
    order_idx = sql.index("ORDER BY")
    limit_idx = sql.index("LIMIT")
    offset_idx = sql.index("OFFSET")
    assert select_idx < from_idx < where_idx < order_idx < limit_idx < offset_idx
    assert len(params) == 1


# --- WHERE column-key validation (C1 SQL-injection guard) -------------------


def test_where_eq_column_is_backtick_quoted() -> None:
    """A valid WHERE column key is backtick-quoted in the SQL (not raw)."""
    flt = FlatTableFilter(eq={"file_name": "doc.pdf"})
    sql, _ = compile_select("cat.s.tbl", SCHEMA, where=flt)
    assert "`file_name` = :" in sql


def test_where_malicious_column_key_raises_invalid_column() -> None:
    """An injection attempt in a WHERE column *key* is rejected, not compiled.

    Filter values were already parameterized; the column identifier was the open
    vector — it must be validated against the schema before reaching the SQL text.
    """
    flt = FlatTableFilter(eq={"id = 1 OR 1=1 --": "x"})
    with pytest.raises(ToolErrorException) as exc:
        compile_select("cat.s.tbl", SCHEMA, where=flt)
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_where_unknown_gt_column_raises_invalid_column() -> None:
    flt = FlatTableFilter(gt={"not_a_column": 5})
    with pytest.raises(ToolErrorException) as exc:
        compile_select("cat.s.tbl", SCHEMA, where=flt)
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_where_is_null_unknown_column_raises_invalid_column() -> None:
    flt = FlatTableFilter(is_null=["nope"])
    with pytest.raises(ToolErrorException) as exc:
        compile_select("cat.s.tbl", SCHEMA, where=flt)
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_where_composite_nested_column_validated() -> None:
    """Column validation reaches nested and/or/not sub-filters, not just the top."""
    bad = AndFilter(
        and_=[
            FlatTableFilter(eq={"file_name": "doc.pdf"}),
            OrFilter(
                or_=[
                    FlatTableFilter(eq={"id": "1"}),
                    NotFilter(not_=FlatTableFilter(eq={"evil`col": "x"})),
                ]
            ),
        ]
    )
    with pytest.raises(ToolErrorException) as exc:
        compile_select("cat.s.tbl", SCHEMA, where=bad)
    assert exc.value.error.error_code is ErrorCode.INVALID_COLUMN


def test_where_injection_key_never_reaches_sql_text() -> None:
    """Defense in depth: a raw injection key must never appear in compiled SQL."""
    injection = "x`); DROP TABLE secrets; --"
    flt = FlatTableFilter(eq={injection: "v"})
    with pytest.raises(ToolErrorException):
        compile_select("cat.s.tbl", SCHEMA, where=flt)
