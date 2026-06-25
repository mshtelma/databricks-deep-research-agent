"""TableLoadTool — fetch row(s) by id, optionally inject into a compute namespace.

The compute namespace mutation is decoupled from the framework via a
``compute_namespace_setter`` callback. When the callback is ``None`` the
tool simply returns the rows as JSON — callers running on the external
ReAct surface get the same payload as the in-compute path.
"""

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
from ..filter_dsl import FlatTableFilter, OrFilter
from ..registry import TableBindingRegistry
from ..sql_compiler import compile_select
from ..table_api import Table
from ._common import (
    SqlExecutor,
    _SchemaCacheLike,
    ensure_roles,
    get_user_token,
    require_column,
    resolve_binding,
)

__all__ = ["TableLoadTool"]


def _error_result(error: ToolError) -> ToolResult:
    return ToolResult(
        content=json.dumps({"error": error.to_dict()}),
        success=False,
        error=str(error.error_code),
        data={"error": error.to_dict()},
    )


def _row_to_table(row: dict[str, Any]) -> Table:
    """Build a Table from a single SQL row.

    Each column becomes a header; the row becomes a single data row keyed
    by column name in the ``cells`` map. We use the row's own column-name
    set as the label fallback when no role-mapped label_column is present.
    """
    headers = [{"name": k, "parent": ""} for k in row]
    cells = {k: ("" if v is None else str(v)) for k, v in row.items()}
    label = next(iter(cells.values()), "")
    table_json: dict[str, Any] = {
        "headers": headers,
        "rows": [{"label": label, "cells": cells}],
        "row_count": 1,
        "data_row_count": 1,
    }
    return Table(table_json)


def _append_namespace_tables(compute: Any, tables: list[Table]) -> list[Table]:
    if compute is None or not hasattr(compute, "get_variable"):
        return list(tables)
    existing = compute.get_variable("tables", [])
    if not isinstance(existing, list):
        return list(tables)
    return [*existing, *tables]


class TableLoadTool:
    """Materialise specific row(s) into a compute namespace and/or return as JSON."""

    def __init__(
        self,
        *,
        registry: TableBindingRegistry,
        schema_cache: _SchemaCacheLike,
        sql_executor: SqlExecutor,
        compute_namespace_setter: Callable[[str, Any], None] | None = None,
        budget: Budget3D | None = None,
        name: str = "table_load",
        description: str | None = None,
        default_binding: str | None = None,
        default_columns: list[str] | None = None,
        default_as_var: str | None = None,
    ) -> None:
        self._registry = registry
        self._schema_cache = schema_cache
        self._sql_executor = sql_executor
        self._namespace_setter = compute_namespace_setter
        self._budget = budget
        self._name = name
        self._description = description
        self._default_binding = default_binding
        self._default_columns = list(default_columns) if default_columns else None
        self._default_as_var = default_as_var

    @property
    def definition(self) -> ToolDefinition:
        description = self._description or (
            "Load row(s) by id into the compute namespace as Table object(s). "
            "Returns the rows as JSON regardless. Use 'as_var' to bind a "
            "specific name; otherwise the row is available under "
            "'last_table' and appended to 'tables'."
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
                    "id": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                    },
                    "as_var": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "roles": {
                        "type": "object",
                        "description": (
                            "Optional role override for DISCOVERED bindings, "
                            "for example {'id': 'id', 'content': 'body'}."
                        ),
                    },
                },
                "required": ["id"] if self._default_binding else ["binding", "id"],
            },
            source_type="builtin",
            source_kind="text_table",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        binding = arguments.get("binding", self._default_binding)
        if not isinstance(binding, str) or not binding:
            raise ValueError("'binding' is required and must be a non-empty string")

        raw_id = arguments.get("id")
        if raw_id is None:
            raise ValueError("'id' is required")
        if isinstance(raw_id, (str, int, float)):
            ids = [str(raw_id)]
        elif isinstance(raw_id, list):
            if not raw_id:
                raise ValueError("'id' list must be non-empty")
            if not all(isinstance(x, (str, int, float)) for x in raw_id):
                raise ValueError("'id' list elements must be scalars")
            ids = [str(x) for x in raw_id]
        else:
            raise ValueError("'id' must be a string or array of strings")

        as_var = arguments.get("as_var", self._default_as_var)
        if as_var is not None and (
            not isinstance(as_var, str) or not as_var.isidentifier()
        ):
            raise ValueError(
                "'as_var' must be a valid Python identifier when provided"
            )

        columns = arguments.get("columns", self._default_columns)
        if columns is not None and (
            not isinstance(columns, list)
            or not all(isinstance(c, str) for c in columns)
        ):
            raise ValueError("'columns' must be a list of strings")

        roles = arguments.get("roles")
        if roles is not None and not isinstance(roles, dict):
            raise ValueError("'roles' must be a dict when provided")

        return {
            "binding": binding,
            "ids": ids,
            "as_var": as_var,
            "columns": list(columns) if columns is not None else None,
            "roles": dict(roles) if roles is not None else None,
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
        ids: list[str] = arguments["ids"]
        as_var: str | None = arguments.get("as_var")
        cols: list[str] | None = arguments.get("columns")
        explicit_roles: dict[str, Any] | None = arguments.get("roles")

        info = resolve_binding(self._registry, binding_name)
        user_token = get_user_token(context.extras)
        info, schema = ensure_roles(
            registry=self._registry,
            binding_name=binding_name,
            info=info,
            schema_cache=self._schema_cache,
            sql_executor=self._sql_executor,
            user_token=user_token,
            explicit_roles=explicit_roles,
        )
        assert info.roles is not None
        id_column = info.roles.id_column
        require_column(schema, id_column, fqn=info.fqn)

        projection: list[str] | None = None
        if cols:
            for c in cols:
                require_column(schema, c, fqn=info.fqn)
            projection = list(cols)
            if id_column not in projection:
                projection.append(id_column)

        # Build IN-list filter via OR over equalities — filter_dsl has no
        # native ``in`` operator, but a flat eq is one leaf and we cap at 64
        # leaves anyway.
        where: FlatTableFilter | OrFilter
        if len(ids) == 1:
            where = FlatTableFilter.model_validate({"eq": {id_column: ids[0]}})
        else:
            sub = [
                FlatTableFilter.model_validate({"eq": {id_column: i}})
                for i in ids
            ]
            where = OrFilter.model_validate({"or": sub})

        sql, params = compile_select(
            info.fqn,
            schema,
            columns=projection,
            where=where,
            limit=min(len(ids), PER_STMT_LIMIT_ROWS),
        )
        rows = self._sql_executor(sql, params, user_token)
        if self._budget is not None:
            self._budget.tick(rows=len(rows))

        tables = [_row_to_table(row) for row in rows]

        # Slot precedence: <as_var> > last_table > tables[-1]
        if self._namespace_setter is not None and tables:
            if as_var is not None:
                # Single anchor variable. For a multi-row load, we bind the
                # first table under as_var; the rest still flow into the
                # `tables` list. This matches spec §6.2 conservatively.
                self._namespace_setter(as_var, tables[0])
            self._namespace_setter("last_table", tables[-1])
            self._namespace_setter("tables", tables)

        # Serialise rows for the JSON content; Table objects are not JSON-
        # serialisable directly so we expose the raw rows.
        payload = {
            "rows": rows,
            "ids": ids,
            "binding": binding_name,
            "loaded": len(rows),
        }
        return ToolResult(
            content=json.dumps(payload),
            data={
                **payload,
                "tables": tables,
            },
        )

    # -- ComputeCallableProvider --------------------------------------------

    @property
    def compute_name(self) -> str:
        return "table_load"

    def to_compute_callable(
        self, *, compute: Any
    ) -> Callable[..., Table | list[Table]]:
        """Return a synchronous callable usable inside the compute sandbox.

        Mutates the hosting compute tool's namespace via the public
        ``inject_variable`` method (when present), and returns the loaded
        ``Table`` object(s). Raises ``ToolErrorException`` on any error.

        ``compute`` should be the hosting :class:`PythonComputeTool`. When
        ``None`` is passed, the callable still returns the rows but does
        not mutate any namespace.
        """
        registry = self._registry
        schema_cache = self._schema_cache
        sql_executor = self._sql_executor
        budget = self._budget

        # Prefer the compute tool's public injection method when available;
        # fall back to the constructor-injected setter so external callers
        # (e.g. the ReAct surface) keep working.
        injector: Callable[[str, Any], None] | None
        if compute is not None and hasattr(compute, "inject_variable"):
            injector = compute.inject_variable
        else:
            injector = self._namespace_setter

        def _call(
            *,
            binding: str,
            id: str | int | float | list[Any],
            user_token: str = "",
            as_var: str | None = None,
            columns: list[str] | None = None,
            roles: dict[str, Any] | None = None,
        ) -> Table | list[Table]:
            if not isinstance(binding, str) or not binding:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_BINDING,
                        message="'binding' is required and must be a non-empty string",
                    )
                )
            if id is None:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message="'id' is required",
                    )
                )
            single_id = isinstance(id, (str, int, float))
            if single_id:
                ids: list[str] = [str(id)]
            elif isinstance(id, list):
                if not id:
                    raise ToolErrorException(
                        ToolError(
                            error_code=ErrorCode.INVALID_FILTER,
                            message="'id' list must be non-empty",
                        )
                    )
                if not all(isinstance(x, (str, int, float)) for x in id):
                    raise ToolErrorException(
                        ToolError(
                            error_code=ErrorCode.INVALID_FILTER,
                            message="'id' list elements must be scalars",
                        )
                    )
                ids = [str(x) for x in id]
            else:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message="'id' must be a scalar or list of scalars",
                    )
                )
            if as_var is not None and (
                not isinstance(as_var, str) or not as_var.isidentifier()
            ):
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message=(
                            "'as_var' must be a valid Python identifier"
                        ),
                    )
                )

            info = resolve_binding(registry, binding)
            info, schema = ensure_roles(
                registry=registry,
                binding_name=binding,
                info=info,
                schema_cache=schema_cache,
                sql_executor=sql_executor,
                user_token=user_token,
                explicit_roles=roles,
            )
            assert info.roles is not None
            id_column = info.roles.id_column
            require_column(schema, id_column, fqn=info.fqn)

            projection: list[str] | None = None
            if columns:
                for c in columns:
                    require_column(schema, c, fqn=info.fqn)
                projection = list(columns)
                if id_column not in projection:
                    projection.append(id_column)

            where: FlatTableFilter | OrFilter
            if len(ids) == 1:
                where = FlatTableFilter.model_validate(
                    {"eq": {id_column: ids[0]}}
                )
            else:
                sub = [
                    FlatTableFilter.model_validate({"eq": {id_column: i}})
                    for i in ids
                ]
                where = OrFilter.model_validate({"or": sub})

            sql, params = compile_select(
                info.fqn,
                schema,
                columns=projection,
                where=where,
                limit=min(len(ids), PER_STMT_LIMIT_ROWS),
            )
            rows = sql_executor(sql, params, user_token)
            if budget is not None:
                budget.tick(rows=len(rows))

            tables = [_row_to_table(row) for row in rows]
            if injector is not None and tables:
                if as_var is not None:
                    injector(as_var, tables[0])
                injector("last_table", tables[-1])
                injector("tables", _append_namespace_tables(compute, tables))
            if single_id and len(tables) == 1:
                return tables[0]
            return tables

        return _call
