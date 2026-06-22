"""TableAggregateTool — count / sum / avg / min / max with optional GROUP BY.

The SQL compiler does not natively support GROUP BY, so this tool builds
its statement directly. All identifiers are validated against the schema
``column_map`` BEFORE assembly; user values flow exclusively through the
``compile_filter`` parameter binder. No string concatenation of user input
into SQL.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)

from ..budgets import PER_STMT_LIMIT_GROUPS, PER_STMT_LIMIT_ROWS, Budget3D
from ..error_codes import ErrorCode, ToolError, ToolErrorException
from ..filter_dsl import (
    AndFilter,
    FlatTableFilter,
    NotFilter,
    OrFilter,
    compile_filter,
)
from ..registry import TableBindingRegistry
from ._common import (
    SqlExecutor,
    TableFilterLike,
    _SchemaCacheLike,
    get_user_token,
    quote_fqn,
    quote_ident,
    require_column,
    resolve_binding,
)

if TYPE_CHECKING:
    from databricks.sdk.service.sql import StatementParameterListItem

__all__ = ["TableAggregateTool"]

_VALID_OPS = ("count", "sum", "avg", "min", "max")
_NUMERIC_OPS = ("sum", "avg", "min", "max")
_DEFAULT_LIMIT = 100


def _error_result(error: ToolError) -> ToolResult:
    return ToolResult(
        content=json.dumps({"error": error.to_dict()}),
        success=False,
        error=str(error.error_code),
        data={"error": error.to_dict()},
    )


def _parse_filter(raw: dict[str, Any] | None) -> TableFilterLike | None:
    if raw is None:
        return None
    if "and" in raw:
        return AndFilter.model_validate(raw)
    if "or" in raw:
        return OrFilter.model_validate(raw)
    if "not" in raw:
        return NotFilter.model_validate(raw)
    return FlatTableFilter.model_validate(raw)


def _to_sdk_params(
    raw_params: list[dict[str, Any]],
) -> list[StatementParameterListItem]:
    from databricks.sdk.service.sql import StatementParameterListItem

    out: list[StatementParameterListItem] = []
    for p in raw_params:
        out.append(
            StatementParameterListItem(
                name=str(p["name"]),
                value=str(p["value"]) if p.get("value") is not None else None,
                type=str(p.get("type")) if p.get("type") is not None else None,
            )
        )
    return out


class TableAggregateTool:
    """Aggregate rows from a registered binding."""

    def __init__(
        self,
        *,
        registry: TableBindingRegistry,
        schema_cache: _SchemaCacheLike,
        sql_executor: SqlExecutor,
        budget: Budget3D | None = None,
        name: str = "table_aggregate",
        description: str | None = None,
        default_binding: str | None = None,
    ) -> None:
        self._registry = registry
        self._schema_cache = schema_cache
        self._sql_executor = sql_executor
        self._budget = budget
        self._name = name
        self._description = description
        self._default_binding = default_binding

    @property
    def definition(self) -> ToolDefinition:
        description = self._description or (
            "Compute count / sum / avg / min / max over a registered "
            "table, with optional WHERE filter and GROUP BY."
        )
        if self._default_binding:
            description = (
                f"{description} Default binding: {self._default_binding!r}; "
                "omit 'binding' to use it."
            )
        return ToolDefinition(
            name=self._name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "binding": {"type": "string"},
                    "op": {
                        "type": "string",
                        "enum": list(_VALID_OPS),
                    },
                    "column": {
                        "type": "string",
                        "description": "Required for op != 'count'.",
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "where": {"type": "object"},
                    "having": {"type": "object"},
                    "limit": {
                        "type": "integer",
                        "default": _DEFAULT_LIMIT,
                    },
                },
                "required": ["op"] if self._default_binding else ["binding", "op"],
            },
            source_type="builtin",
            source_kind="text_table",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        binding = arguments.get("binding", self._default_binding)
        if not isinstance(binding, str) or not binding:
            raise ValueError("'binding' is required and must be a non-empty string")

        op = arguments.get("op")
        if not isinstance(op, str) or op not in _VALID_OPS:
            raise ValueError(
                f"'op' must be one of {_VALID_OPS}; got {op!r}"
            )

        column = arguments.get("column")
        if op != "count":
            if not isinstance(column, str) or not column:
                raise ValueError(
                    f"'column' is required for op={op!r}"
                )
        elif column is not None and not isinstance(column, str):
            raise ValueError("'column' must be a string when provided")

        group_by = arguments.get("group_by")
        if group_by is not None and (
            not isinstance(group_by, list)
            or not all(isinstance(g, str) for g in group_by)
        ):
            raise ValueError("'group_by' must be a list of strings")

        where = arguments.get("where")
        if where is not None and not isinstance(where, dict):
            raise ValueError("'where' must be a dict when provided")

        having = arguments.get("having")
        if having is not None and not isinstance(having, dict):
            raise ValueError("'having' must be a dict when provided")

        limit_raw = arguments.get("limit", _DEFAULT_LIMIT)
        if not isinstance(limit_raw, int) or limit_raw < 1:
            raise ValueError("'limit' must be a positive integer")
        limit = min(limit_raw, PER_STMT_LIMIT_ROWS)

        return {
            "binding": binding,
            "op": op,
            "column": column,
            "group_by": list(group_by) if group_by else None,
            "where": where,
            "having": having,
            "limit": limit,
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        try:
            return await self._run(arguments, context)
        except ToolErrorException as exc:
            return _error_result(exc.error)

    async def _run(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        binding_name: str = arguments["binding"]
        op: str = arguments["op"]
        column: str | None = arguments.get("column")
        group_by: list[str] | None = arguments.get("group_by")
        having_raw: dict[str, Any] | None = arguments.get("having")
        limit: int = arguments["limit"]

        if having_raw is not None:
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_FILTER,
                    message="HAVING not yet supported by table_aggregate",
                    binding=binding_name,
                    hint=(
                        "Filter post-aggregation in the caller, or wait "
                        "for HAVING support in a future release."
                    ),
                )
            )

        info = resolve_binding(self._registry, binding_name)
        user_token = get_user_token(context.extras)
        schema = self._schema_cache.get(info.fqn, user_token)

        # Validate column membership.
        if op != "count" and column is not None:
            require_column(schema, column, fqn=info.fqn)
        if (
            op in _NUMERIC_OPS
            and column is not None
            and column not in info.numeric_columns
        ):
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_COLUMN,
                    message=(
                        f"column {column!r} is not in "
                        f"binding.numeric_columns for {binding_name!r}"
                    ),
                    binding=binding_name,
                    hint=(
                        "column must appear in binding.numeric_columns "
                        f"for aggregate ops {_NUMERIC_OPS}"
                    ),
                    details={
                        "column": column,
                        "numeric_columns": list(info.numeric_columns),
                    },
                )
            )

        if group_by:
            for g in group_by:
                require_column(schema, g, fqn=info.fqn)

        # Build the aggregate SQL.
        quoted_fqn = quote_fqn(info.fqn)

        if op == "count":
            agg_expr = (
                f"COUNT({quote_ident(column)})" if column else "COUNT(*)"
            )
        elif op == "sum":
            agg_expr = f"SUM({quote_ident(column)})" if column else "SUM(*)"
        elif op == "avg":
            agg_expr = f"AVG({quote_ident(column)})" if column else "AVG(*)"
        elif op == "min":
            agg_expr = f"MIN({quote_ident(column)})" if column else "MIN(*)"
        else:  # max
            agg_expr = f"MAX({quote_ident(column)})" if column else "MAX(*)"

        select_parts: list[str] = []
        if group_by:
            select_parts.extend(quote_ident(g) for g in group_by)
        select_parts.append(f"{agg_expr} AS `__agg`")

        sql_parts: list[str] = [
            "SELECT " + ", ".join(select_parts),
            f"FROM {quoted_fqn}",
        ]

        sdk_params: list[StatementParameterListItem] = []
        where_filter = _parse_filter(arguments.get("where"))
        if where_filter is not None:
            fragment, raw_params = compile_filter(where_filter)
            if fragment:
                sql_parts.append(f"WHERE {fragment}")
            sdk_params.extend(_to_sdk_params(raw_params))

        if group_by:
            sql_parts.append(
                "GROUP BY " + ", ".join(quote_ident(g) for g in group_by)
            )
            # Detect group cardinality overflow by fetching one extra row.
            cap_groups = min(PER_STMT_LIMIT_GROUPS + 1, PER_STMT_LIMIT_ROWS)
            effective_limit = min(limit, cap_groups)
        else:
            effective_limit = min(limit, PER_STMT_LIMIT_ROWS)

        sql_parts.append(f"LIMIT {effective_limit}")
        sql = " ".join(sql_parts)

        rows = self._sql_executor(sql, sdk_params, user_token)
        if self._budget is not None:
            self._budget.tick(rows=len(rows))

        if group_by and len(rows) > PER_STMT_LIMIT_GROUPS:
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.GROUP_CARDINALITY_EXCEEDED,
                    message=(
                        f"GROUP BY produced more than {PER_STMT_LIMIT_GROUPS} "
                        "groups; tighten the WHERE filter or aggregate at a "
                        "coarser granularity"
                    ),
                    binding=binding_name,
                    details={"limit": PER_STMT_LIMIT_GROUPS},
                )
            )

        payload = {
            "rows": rows,
            "op": op,
            "column": column,
            "group_by": group_by,
            "binding": binding_name,
        }
        return ToolResult(
            content=json.dumps(payload),
            data=payload,
        )

    # -- ComputeCallableProvider --------------------------------------------

    @property
    def compute_name(self) -> str:
        return "table_aggregate"

    def to_compute_callable(
        self, *, compute: Any
    ) -> Callable[..., list[dict[str, Any]]]:
        """Return a synchronous callable usable inside the compute sandbox.

        Returns the list of aggregate row dicts. Raises ``ToolErrorException``
        on any error.
        """
        del compute
        registry = self._registry
        schema_cache = self._schema_cache
        sql_executor = self._sql_executor
        budget = self._budget

        def _call(
            *,
            binding: str,
            op: str,
            user_token: str = "",
            column: str | None = None,
            group_by: list[str] | None = None,
            where: dict[str, Any] | None = None,
            having: dict[str, Any] | None = None,
            limit: int = _DEFAULT_LIMIT,
        ) -> list[dict[str, Any]]:
            if not isinstance(binding, str) or not binding:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_BINDING,
                        message="'binding' is required and must be a non-empty string",
                    )
                )
            if not isinstance(op, str) or op not in _VALID_OPS:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message=f"'op' must be one of {_VALID_OPS}; got {op!r}",
                    )
                )
            if op != "count" and (not isinstance(column, str) or not column):
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_COLUMN,
                        message=f"'column' is required for op={op!r}",
                    )
                )
            if having is not None:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message="HAVING not yet supported by table_aggregate",
                        binding=binding,
                    )
                )

            limit_eff = min(max(int(limit), 1), PER_STMT_LIMIT_ROWS)

            info = resolve_binding(registry, binding)
            schema = schema_cache.get(info.fqn, user_token)

            if op != "count" and column is not None:
                require_column(schema, column, fqn=info.fqn)
            if (
                op in _NUMERIC_OPS
                and column is not None
                and column not in info.numeric_columns
            ):
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_COLUMN,
                        message=(
                            f"column {column!r} is not in "
                            f"binding.numeric_columns for {binding!r}"
                        ),
                        binding=binding,
                        details={
                            "column": column,
                            "numeric_columns": list(info.numeric_columns),
                        },
                    )
                )

            gb = list(group_by) if group_by else None
            if gb:
                for g in gb:
                    require_column(schema, g, fqn=info.fqn)

            quoted_fqn = quote_fqn(info.fqn)
            if op == "count":
                agg_expr = (
                    f"COUNT({quote_ident(column)})" if column else "COUNT(*)"
                )
            elif op == "sum":
                agg_expr = (
                    f"SUM({quote_ident(column)})" if column else "SUM(*)"
                )
            elif op == "avg":
                agg_expr = (
                    f"AVG({quote_ident(column)})" if column else "AVG(*)"
                )
            elif op == "min":
                agg_expr = (
                    f"MIN({quote_ident(column)})" if column else "MIN(*)"
                )
            else:  # max
                agg_expr = (
                    f"MAX({quote_ident(column)})" if column else "MAX(*)"
                )

            select_parts: list[str] = []
            if gb:
                select_parts.extend(quote_ident(g) for g in gb)
            select_parts.append(f"{agg_expr} AS `__agg`")

            sql_parts: list[str] = [
                "SELECT " + ", ".join(select_parts),
                f"FROM {quoted_fqn}",
            ]

            sdk_params: list[StatementParameterListItem] = []
            where_filter = _parse_filter(where)
            if where_filter is not None:
                fragment, raw_params = compile_filter(where_filter)
                if fragment:
                    sql_parts.append(f"WHERE {fragment}")
                sdk_params.extend(_to_sdk_params(raw_params))

            if gb:
                sql_parts.append(
                    "GROUP BY " + ", ".join(quote_ident(g) for g in gb)
                )
                cap_groups = min(PER_STMT_LIMIT_GROUPS + 1, PER_STMT_LIMIT_ROWS)
                effective_limit = min(limit_eff, cap_groups)
            else:
                effective_limit = min(limit_eff, PER_STMT_LIMIT_ROWS)

            sql_parts.append(f"LIMIT {effective_limit}")
            sql = " ".join(sql_parts)

            rows = sql_executor(sql, sdk_params, user_token)
            if budget is not None:
                budget.tick(rows=len(rows))

            if gb and len(rows) > PER_STMT_LIMIT_GROUPS:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.GROUP_CARDINALITY_EXCEEDED,
                        message=(
                            f"GROUP BY produced more than {PER_STMT_LIMIT_GROUPS} "
                            "groups"
                        ),
                        binding=binding,
                        details={"limit": PER_STMT_LIMIT_GROUPS},
                    )
                )
            return rows

        return _call
