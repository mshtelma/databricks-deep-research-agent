"""TableReadTool — paginated row read with where/columns/order_by."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)

from ..budgets import PER_STMT_LIMIT_ROWS, Budget3D
from ..error_codes import ErrorCode, ToolError, ToolErrorException
from ..filter_dsl import (
    WHERE_PARAM_SCHEMA,
    AndFilter,
    FlatTableFilter,
    NotFilter,
    OrFilter,
    coerce_flat_filter_shape,
)
from ..registry import TableBindingRegistry
from ..sql_compiler import compile_select
from ._common import (
    SqlExecutor,
    TableFilterLike,
    _SchemaCacheLike,
    get_user_token,
    require_column,
    resolve_binding,
)

__all__ = ["TableReadTool"]

_DEFAULT_LIMIT = 50


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
    raw = coerce_flat_filter_shape(raw)
    if "and" in raw:
        return AndFilter.model_validate(raw)
    if "or" in raw:
        return OrFilter.model_validate(raw)
    if "not" in raw:
        return NotFilter.model_validate(raw)
    return FlatTableFilter.model_validate(raw)


class TableReadTool:
    """Read rows from a registered binding with where/columns/order_by/pagination."""

    def __init__(
        self,
        *,
        registry: TableBindingRegistry,
        schema_cache: _SchemaCacheLike,
        sql_executor: SqlExecutor,
        budget: Budget3D | None = None,
        name: str = "table_read",
        description: str | None = None,
        default_binding: str | None = None,
        default_columns: list[str] | None = None,
        default_order_by: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._schema_cache = schema_cache
        self._sql_executor = sql_executor
        self._budget = budget
        self._name = name
        self._description = description
        self._default_binding = default_binding
        self._default_columns = list(default_columns) if default_columns else None
        self._default_order_by = list(default_order_by) if default_order_by else None

    @property
    def definition(self) -> ToolDefinition:
        description = self._description or (
            "Read rows from a registered table with optional filter, "
            "column projection, ordering, and pagination."
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
                    "where": WHERE_PARAM_SCHEMA,
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "order_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Prefix column with '-' for DESC.",
                    },
                    "limit": {"type": "integer", "default": _DEFAULT_LIMIT},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": [] if self._default_binding else ["binding"],
            },
            source_type="builtin",
            source_kind="text_table",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        binding = arguments.get("binding", self._default_binding)
        if not isinstance(binding, str) or not binding:
            raise ValueError("'binding' is required and must be a non-empty string")

        where = arguments.get("where")
        if where is not None and not isinstance(where, dict):
            raise ValueError("'where' must be a dict when provided")

        columns = arguments.get("columns", self._default_columns)
        if columns is not None and (
            not isinstance(columns, list)
            or not all(isinstance(c, str) for c in columns)
        ):
            raise ValueError("'columns' must be a list of strings")

        order_by = arguments.get("order_by", self._default_order_by)
        if order_by is not None and (
            not isinstance(order_by, list)
            or not all(isinstance(o, str) for o in order_by)
        ):
            raise ValueError("'order_by' must be a list of strings")

        limit_raw = arguments.get("limit", _DEFAULT_LIMIT)
        if not isinstance(limit_raw, int) or limit_raw < 1:
            raise ValueError("'limit' must be a positive integer")
        limit = min(limit_raw, PER_STMT_LIMIT_ROWS)

        offset_raw = arguments.get("offset", 0)
        if not isinstance(offset_raw, int) or offset_raw < 0:
            raise ValueError("'offset' must be a non-negative integer")

        return {
            "binding": binding,
            "where": where,
            "columns": list(columns) if columns is not None else None,
            "order_by": list(order_by) if order_by is not None else None,
            "limit": limit,
            "offset": offset_raw,
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
        info = resolve_binding(self._registry, binding_name)
        user_token = get_user_token(context.extras)
        schema = self._schema_cache.get(info.fqn, user_token)

        cols: list[str] | None = arguments.get("columns")
        if cols:
            for c in cols:
                require_column(schema, c, fqn=info.fqn)

        where = _parse_filter(arguments.get("where"))

        sql, params = compile_select(
            info.fqn,
            schema,
            columns=cols,
            where=where,
            order_by=arguments.get("order_by"),
            limit=arguments["limit"],
            offset=arguments["offset"],
        )
        rows = self._sql_executor(sql, params, user_token)
        if self._budget is not None:
            self._budget.tick(rows=len(rows))

        return ToolResult(
            content=json.dumps({"rows": rows, "count": len(rows)}),
            data={
                "rows": rows,
                "count": len(rows),
                "binding": binding_name,
            },
        )

    # -- ComputeCallableProvider --------------------------------------------

    @property
    def compute_name(self) -> str:
        return "table_read"

    def to_compute_callable(
        self, *, compute: Any
    ) -> Callable[..., list[dict[str, Any]]]:
        """Return a synchronous callable usable inside the compute sandbox.

        Returns the raw list of row dicts. Raises ``ToolErrorException`` on
        any error.
        """
        del compute
        registry = self._registry
        schema_cache = self._schema_cache
        sql_executor = self._sql_executor
        budget = self._budget

        def _call(
            *,
            binding: str,
            user_token: str = "",
            where: dict[str, Any] | None = None,
            columns: list[str] | None = None,
            order_by: list[str] | None = None,
            limit: int = _DEFAULT_LIMIT,
            offset: int = 0,
        ) -> list[dict[str, Any]]:
            if not isinstance(binding, str) or not binding:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_BINDING,
                        message="'binding' is required and must be a non-empty string",
                    )
                )
            limit_eff = min(max(int(limit), 1), PER_STMT_LIMIT_ROWS)
            offset_eff = max(int(offset), 0)

            info = resolve_binding(registry, binding)
            schema = schema_cache.get(info.fqn, user_token)

            cols = list(columns) if columns is not None else None
            if cols:
                for c in cols:
                    require_column(schema, c, fqn=info.fqn)

            where_filter = _parse_filter(where)
            sql, params = compile_select(
                info.fqn,
                schema,
                columns=cols,
                where=where_filter,
                order_by=list(order_by) if order_by is not None else None,
                limit=limit_eff,
                offset=offset_eff,
            )
            rows = sql_executor(sql, params, user_token)
            if budget is not None:
                budget.tick(rows=len(rows))
            return rows

        return _call
