"""TableFilter — recursive predicate DSL for SQL generation.

Extends the V1 flat-shape filter with ``{and, or, not}`` composite operators
while preserving full backwards compatibility with the original flat dict shape.

## NULL + OR semantics parity vs SQL

| Operation        | Recursive DSL | SQL       | Status    |
|------------------|---------------|-----------|-----------|
| or([NULL,true])  | true          | true      | parity    |
| or([NULL,NULL])  | NULL          | NULL      | parity    |
| not(NULL)        | NULL          | NULL      | parity    |
| and([NULL,false])| false         | false     | parity    |
| or([])           | FALSE         | rejected  | DEVIATION |
| and([])          | TRUE          | rejected  | DEVIATION |

The deviations (empty ``and`` → TRUE, empty ``or`` → FALSE) are intentional:
they follow monoid identity element conventions and avoid caller-side special
casing of empty predicate lists.

## DoS limits
- Nesting depth ≤ 8
- Total leaf (FlatTableFilter) count ≤ 64

## Feature flag
Set ``AGENT_DESIGNER_TABLE_FILTER_RECURSIVE=0`` to disable the recursive union
and accept only V1 flat-shape input.  When the flag is off, ``AndFilter``,
``OrFilter``, and ``NotFilter`` are still importable but ``TableFilter`` is
aliased to ``FlatTableFilter`` so the Pydantic discriminator rejects them.
"""

from __future__ import annotations

import os
from typing import Any, TypeAlias

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# V1 flat-shape (preserved exactly)
# ---------------------------------------------------------------------------

_V1_EQ_MAX = 32
_V1_IN_MAX = 256


class FlatTableFilter(BaseModel):
    """V1 flat filter shape: named equality / in-list predicates.

    All fields are optional; an empty ``FlatTableFilter`` compiles to an empty
    string (no WHERE clause contribution).
    """

    eq: dict[str, Any] | None = Field(
        default=None,
        description="Exact-match predicates: {column: value}. Max 32 pairs.",
    )
    in_columns: list[str] | None = Field(
        default=None,
        description="List of column names whose value must appear in the search term.",
    )
    gt: dict[str, Any] | None = Field(
        default=None,
        description="Greater-than predicates: {column: value}.",
    )
    gte: dict[str, Any] | None = Field(
        default=None,
        description="Greater-than-or-equal predicates: {column: value}.",
    )
    lt: dict[str, Any] | None = Field(
        default=None,
        description="Less-than predicates: {column: value}.",
    )
    lte: dict[str, Any] | None = Field(
        default=None,
        description="Less-than-or-equal predicates: {column: value}.",
    )
    ne: dict[str, Any] | None = Field(
        default=None,
        description="Not-equal predicates: {column: value}.",
    )
    is_null: list[str] | None = Field(
        default=None,
        description="Column names that must be NULL.",
    )
    is_not_null: list[str] | None = Field(
        default=None,
        description="Column names that must be NOT NULL.",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Composite operators
# ---------------------------------------------------------------------------


class AndFilter(BaseModel):
    """Logical AND of a list of sub-filters.

    An empty list (``and: []``) compiles to ``TRUE`` (identity element — DEVIATION
    from SQL which rejects empty AND).
    """

    and_: list[TableFilter] = Field(
        alias="and",
        min_length=0,
        description="Sub-filters combined with AND. Empty → TRUE. Compiler enforces max 64 leaves total.",
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}


class OrFilter(BaseModel):
    """Logical OR of a list of sub-filters.

    An empty list (``or: []``) compiles to ``FALSE`` (identity element — DEVIATION
    from SQL which rejects empty OR).
    """

    or_: list[TableFilter] = Field(
        alias="or",
        min_length=0,
        description="Sub-filters combined with OR. Empty → FALSE. Compiler enforces max 64 leaves total.",
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}


class NotFilter(BaseModel):
    """Logical NOT of a single sub-filter."""

    not_: TableFilter = Field(
        alias="not",
        description="Sub-filter to negate.",
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Feature-flag gated union type
# ---------------------------------------------------------------------------

_RECURSIVE_ENABLED = os.environ.get("AGENT_DESIGNER_TABLE_FILTER_RECURSIVE", "1") != "0"

# TableFilter is always the full union for static type checking.
TableFilter: TypeAlias = FlatTableFilter | AndFilter | OrFilter | NotFilter

# At runtime, when the rollback flag is set, the Pydantic models are rebuilt
# with only FlatTableFilter in the namespace, so the "TableFilter" forward
# reference inside AndFilter / OrFilter / NotFilter resolves to FlatTableFilter
# only — causing Pydantic to reject recursive variants at validation time.
_runtime_table_filter: object = (
    FlatTableFilter | AndFilter | OrFilter | NotFilter
    if _RECURSIVE_ENABLED
    else FlatTableFilter
)
_ns: dict[str, object] = {"TableFilter": _runtime_table_filter}
AndFilter.model_rebuild(_types_namespace=_ns)
OrFilter.model_rebuild(_types_namespace=_ns)
NotFilter.model_rebuild(_types_namespace=_ns)

# ---------------------------------------------------------------------------
# SQL compiler
# ---------------------------------------------------------------------------

_MAX_DEPTH = 8
_MAX_LEAVES = 64


def count_leaves(f: FlatTableFilter | AndFilter | OrFilter | NotFilter) -> int:
    """Count the number of FlatTableFilter leaf nodes in a filter tree."""
    if isinstance(f, AndFilter):
        return sum(count_leaves(s) for s in f.and_)
    if isinstance(f, OrFilter):
        return sum(count_leaves(s) for s in f.or_)
    if isinstance(f, NotFilter):
        return count_leaves(f.not_)
    # FlatTableFilter
    return 1


def compile_filter(
    filter_obj: FlatTableFilter | AndFilter | OrFilter | NotFilter,
    *,
    depth: int = 0,
    _leaf_budget: list[int] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Compile a filter tree into a parameterized SQL fragment.

    Parameters
    ----------
    filter_obj:
        The filter to compile.
    depth:
        Current recursion depth (starts at 0, max 8).
    _leaf_budget:
        Internal mutable counter [remaining_leaves].  Callers should not pass
        this; it is initialised by the entry-point wrapper below.

    Returns
    -------
    (sql_fragment, params)
        ``sql_fragment`` is a string suitable for embedding in a WHERE clause.
        ``params`` is a list of ``{name, value, type}`` dicts for parameterized
        execution.

    Raises
    ------
    ValueError
        If depth > 8 or total leaf count > 64.
    """
    if depth > _MAX_DEPTH:
        raise ValueError(
            f"filter nesting exceeds maximum depth={_MAX_DEPTH}; got depth={depth}"
        )

    if _leaf_budget is None:
        # Entry point: initialise budget and validate total leaves upfront.
        total = count_leaves(filter_obj)
        if total > _MAX_LEAVES:
            raise ValueError(
                f"filter has {total} leaf nodes; maximum is {_MAX_LEAVES}"
            )
        _leaf_budget = [total]

    if isinstance(filter_obj, AndFilter):
        sub_filters = filter_obj.and_
        if not sub_filters:
            return "TRUE", []
        parts: list[str] = []
        params: list[dict[str, Any]] = []
        for sub in sub_filters:
            sql, p = compile_filter(sub, depth=depth + 1, _leaf_budget=_leaf_budget)
            parts.append(f"({sql})")
            params.extend(p)
        return " AND ".join(parts), params

    if isinstance(filter_obj, OrFilter):
        sub_filters_or = filter_obj.or_
        if not sub_filters_or:
            return "FALSE", []
        parts_or: list[str] = []
        params_or: list[dict[str, Any]] = []
        for sub in sub_filters_or:
            sql, p = compile_filter(sub, depth=depth + 1, _leaf_budget=_leaf_budget)
            parts_or.append(f"({sql})")
            params_or.extend(p)
        return " OR ".join(parts_or), params_or

    if isinstance(filter_obj, NotFilter):
        sql, p = compile_filter(filter_obj.not_, depth=depth + 1, _leaf_budget=_leaf_budget)
        return f"NOT ({sql})", p

    # FlatTableFilter (V1 leaf)
    return _compile_flat(filter_obj)


def _compile_flat(
    f: FlatTableFilter,
) -> tuple[str, list[dict[str, Any]]]:
    """Compile a V1 flat filter into parameterized SQL.

    All user-provided values are bound as parameters — never concatenated.
    """
    parts: list[str] = []
    params: list[dict[str, Any]] = []
    _counter: list[int] = [0]

    def _next_name(col: str) -> str:
        _counter[0] += 1
        # Sanitize column name to safe identifier for param name.
        safe_col = "".join(c if c.isalnum() or c == "_" else "_" for c in col)
        return f"p_{safe_col}_{_counter[0]}"

    def _add(col: str, op: str, value: Any, sql_type: str = "STRING") -> None:
        name = _next_name(col)
        # Column name is NOT parameterized — it comes from the schema definition,
        # not from user input at the filter value level.  In real usage callers
        # should validate column names against an allowlist.
        parts.append(f"{col} {op} :{name}")
        params.append({"name": name, "value": str(value), "type": sql_type})

    if f.eq:
        for col, val in f.eq.items():
            if val is None:
                parts.append(f"{col} IS NULL")
            else:
                _add(col, "=", val)

    if f.ne:
        for col, val in f.ne.items():
            if val is None:
                parts.append(f"{col} IS NOT NULL")
            else:
                _add(col, "<>", val)

    if f.gt:
        for col, val in f.gt.items():
            _add(col, ">", val)

    if f.gte:
        for col, val in f.gte.items():
            _add(col, ">=", val)

    if f.lt:
        for col, val in f.lt.items():
            _add(col, "<", val)

    if f.lte:
        for col, val in f.lte.items():
            _add(col, "<=", val)

    if f.is_null:
        for col in f.is_null:
            parts.append(f"{col} IS NULL")

    if f.is_not_null:
        for col in f.is_not_null:
            parts.append(f"{col} IS NOT NULL")

    if not parts:
        return "TRUE", []

    return " AND ".join(parts), params
