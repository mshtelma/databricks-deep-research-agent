"""Parameterized SQL compiler for text-table tools.

Produces a ``SELECT`` statement plus a list of
``StatementParameterListItem`` for the Databricks SQL execution API.

Safety contract:
- The FQN must be a 3-part identifier ``catalog.schema.table``.
- All column references and the FQN are backtick-quoted.
- Every column referenced in ``columns`` and ``order_by`` is validated
  against ``schema.column_map`` before going into the SQL text.
- Filter values are bound as parameters; user-supplied data is never
  concatenated into SQL text.
- ``LIMIT`` is hard-clamped to ``PER_STMT_LIMIT_ROWS`` from
  ``text_table.budgets``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .budgets import PER_STMT_LIMIT_ROWS
from .error_codes import ErrorCode, ToolError, ToolErrorException
from .filter_dsl import (
    AndFilter,
    FlatTableFilter,
    NotFilter,
    OrFilter,
    compile_filter,
)
from .schema_cache import Schema

if TYPE_CHECKING:
    from databricks.sdk.service.sql import StatementParameterListItem

TableFilterLike = FlatTableFilter | AndFilter | OrFilter | NotFilter


def _raise_invalid_column(column: str, fqn: str) -> None:
    raise ToolErrorException(
        ToolError(
            error_code=ErrorCode.INVALID_COLUMN,
            message=f"column {column!r} not in schema for {fqn}",
            details={"column": column, "fqn": fqn},
        )
    )


def _raise_invalid_binding(fqn: str, reason: str) -> None:
    raise ToolErrorException(
        ToolError(
            error_code=ErrorCode.INVALID_BINDING,
            message=f"invalid FQN {fqn!r}: {reason}",
            details={"fqn": fqn, "reason": reason},
        )
    )


def _quote_ident(name: str) -> str:
    return f"`{name}`"


def _quote_fqn(fqn: str) -> str:
    parts = fqn.split(".")
    if len(parts) != 3:
        _raise_invalid_binding(
            fqn, "must be three-part 'catalog.schema.table'"
        )
    if any(not p for p in parts):
        _raise_invalid_binding(fqn, "FQN parts must be non-empty")
    if any("`" in p for p in parts):
        _raise_invalid_binding(fqn, "FQN parts must not contain backticks")
    return ".".join(_quote_ident(p) for p in parts)


def _validate_columns(
    columns: Sequence[str] | None, schema: Schema, fqn: str
) -> Sequence[str] | None:
    if columns is None:
        return None
    for col in columns:
        if col not in schema.column_map:
            _raise_invalid_column(col, fqn)
    return columns


def _to_sdk_params(
    raw_params: list[dict[str, object]],
) -> list[StatementParameterListItem]:
    """Convert compile_filter's dict params to the SDK shape."""
    from databricks.sdk.service.sql import StatementParameterListItem

    out: list[StatementParameterListItem] = []
    for p in raw_params:
        name = p["name"]
        value = p["value"]
        type_ = p.get("type")
        out.append(
            StatementParameterListItem(
                name=str(name),
                value=str(value) if value is not None else None,
                type=str(type_) if type_ is not None else None,
            )
        )
    return out


def _escape_like(value: str) -> str:
    """Escape user text for a SQL LIKE pattern using backslash escaping."""
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def compile_select(
    fqn: str,
    schema: Schema,
    *,
    columns: Sequence[str] | None = None,
    where: TableFilterLike | None = None,
    text_search: tuple[str, str] | None = None,
    order_by: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> tuple[str, list[StatementParameterListItem]]:
    """Compile a SELECT statement against the given schema.

    Parameters
    ----------
    fqn:
        Three-part identifier ``catalog.schema.table``. Reject anything else
        with ``INVALID_BINDING``.
    schema:
        Cached table schema. Used to validate column projections and ORDER BY
        keys.
    columns:
        Optional column projection. ``None`` produces ``SELECT *``. Each name
        must be in ``schema.column_map``; otherwise ``INVALID_COLUMN``.
    where:
        Optional ``TableFilter`` (recursive DSL) compiled via
        ``filter_dsl.compile_filter``. Filter values become bound parameters.
    text_search:
        Optional ``(column, query)`` pair compiled into a parameterized,
        case-insensitive ``LIKE`` predicate against a string content column.
    order_by:
        Optional list of column names. Prefix a name with ``-`` to sort
        descending (``"-chunk_index"``). Each name must be in the schema.
    limit:
        Hard-capped to ``PER_STMT_LIMIT_ROWS``. ``None`` defaults to the cap.
        Negative values are clamped to ``0``.
    offset:
        Pagination offset. Falsy / zero / negative values omit the OFFSET
        clause.

    Returns
    -------
    sql:
        The fully-assembled SQL statement.
    params:
        A list of ``StatementParameterListItem`` ready to pass to the
        Databricks SQL execution API.
    """
    quoted_fqn = _quote_fqn(fqn)
    _validate_columns(columns, schema, fqn)

    if columns is None:
        select_clause = "SELECT *"
    else:
        select_clause = "SELECT " + ", ".join(_quote_ident(c) for c in columns)

    parts: list[str] = [select_clause, f"FROM {quoted_fqn}"]
    sdk_params: list[StatementParameterListItem] = []
    where_fragments: list[str] = []
    raw_search_params: list[dict[str, object]] = []

    if where is not None:

        def _filter_col(col: str) -> str:
            # Validate every WHERE column key against the schema and backtick-quote
            # it. ``compile_filter`` parameterizes filter *values* but not column
            # identifiers; without this they would be interpolated raw into the SQL
            # (injection, plus breakage on reserved/spaced names). Mirrors the
            # projection / ORDER BY / text_search validation done elsewhere here.
            if col not in schema.column_map:
                _raise_invalid_column(col, fqn)
            return _quote_ident(col)

        sql_fragment, raw_params = compile_filter(where, column_quoter=_filter_col)
        if sql_fragment:
            where_fragments.append(sql_fragment)
        sdk_params.extend(_to_sdk_params(raw_params))

    if text_search is not None:
        search_col, query = text_search
        if search_col not in schema.column_map:
            _raise_invalid_column(search_col, fqn)
        param_name = "p_text_search_1"
        where_fragments.append(
            f"LOWER({_quote_ident(search_col)}) LIKE LOWER(:{param_name}) ESCAPE '\\\\'"
        )
        raw_search_params.append(
            {
                "name": param_name,
                "value": f"%{_escape_like(query)}%",
                "type": "STRING",
            }
        )

    if where_fragments:
        parts.append("WHERE " + " AND ".join(f"({frag})" for frag in where_fragments))
    if raw_search_params:
        sdk_params.extend(_to_sdk_params(raw_search_params))

    if order_by:
        order_clauses: list[str] = []
        for ob in order_by:
            descending = ob.startswith("-")
            col_name = ob[1:] if descending else ob
            if col_name not in schema.column_map:
                _raise_invalid_column(col_name, fqn)
            direction = "DESC" if descending else "ASC"
            order_clauses.append(f"{_quote_ident(col_name)} {direction}")
        parts.append("ORDER BY " + ", ".join(order_clauses))

    effective_limit = PER_STMT_LIMIT_ROWS if limit is None else limit
    if effective_limit < 0:
        effective_limit = 0
    capped_limit = min(effective_limit, PER_STMT_LIMIT_ROWS)
    parts.append(f"LIMIT {capped_limit}")

    if offset is not None and offset > 0:
        parts.append(f"OFFSET {offset}")

    return " ".join(parts), sdk_params
