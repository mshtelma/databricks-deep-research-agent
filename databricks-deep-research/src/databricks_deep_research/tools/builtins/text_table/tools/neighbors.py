"""TableNeighborsTool — fetch sibling rows by partition_column + order_column."""

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
from ..filter_dsl import AndFilter, FlatTableFilter
from ..registry import TableBindingRegistry
from ..sql_compiler import compile_select
from ._common import (
    SqlExecutor,
    _SchemaCacheLike,
    ensure_roles,
    get_user_token,
    require_column,
    resolve_binding,
)

__all__ = ["TableNeighborsTool"]


def _error_result(error: ToolError) -> ToolResult:
    return ToolResult(
        content=json.dumps({"error": error.to_dict()}),
        success=False,
        error=str(error.error_code),
        data={"error": error.to_dict()},
    )


class TableNeighborsTool:
    """Return rows neighbouring a given anchor by (partition, order)."""

    def __init__(
        self,
        *,
        registry: TableBindingRegistry,
        schema_cache: _SchemaCacheLike,
        sql_executor: SqlExecutor,
        budget: Budget3D | None = None,
        name: str = "table_neighbors",
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
            "For an anchor row identified by id, return sibling rows in "
            "the same partition (by partition_column) within a window of "
            "[order - before, order + after] on order_column."
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
                        "type": "string",
                        "description": "ID value of the anchor row.",
                    },
                    "before": {"type": "integer", "default": 1},
                    "after": {"type": "integer", "default": 1},
                    "roles": {
                        "type": "object",
                        "description": (
                            "Optional role override for DISCOVERED bindings, "
                            "for example {'id': 'id', 'content': 'body', "
                            "'partition': 'doc_id', 'order': 'chunk_index'}."
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
        id_value = arguments.get("id")
        if id_value is None or not isinstance(id_value, (str, int, float)):
            raise ValueError("'id' is required and must be a scalar")
        before = arguments.get("before", 1)
        if not isinstance(before, int) or before < 0:
            raise ValueError("'before' must be a non-negative integer")
        after = arguments.get("after", 1)
        if not isinstance(after, int) or after < 0:
            raise ValueError("'after' must be a non-negative integer")
        roles = arguments.get("roles")
        if roles is not None and not isinstance(roles, dict):
            raise ValueError("'roles' must be a dict when provided")
        return {
            "binding": binding,
            "id": str(id_value),
            "before": before,
            "after": after,
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
        id_value: str = arguments["id"]
        before: int = arguments["before"]
        after: int = arguments["after"]
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

        roles = info.roles
        if (
            roles.id_column is None
            or roles.partition_column is None
            or roles.order_column is None
        ):
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.NEIGHBOR_CONFIG_MISSING,
                    message=(
                        f"binding {binding_name!r} is missing one of "
                        "id_column / partition_column / order_column required "
                        "for neighbours"
                    ),
                    binding=binding_name,
                    details={
                        "id_column": roles.id_column,
                        "partition_column": roles.partition_column,
                        "order_column": roles.order_column,
                    },
                )
            )

        for col in (roles.id_column, roles.partition_column, roles.order_column):
            require_column(schema, col, fqn=info.fqn)

        # Step 1 — fetch anchor partition + order
        anchor_filter = FlatTableFilter.model_validate(
            {"eq": {roles.id_column: id_value}}
        )
        sql_a, params_a = compile_select(
            info.fqn,
            schema,
            columns=[
                roles.id_column,
                roles.partition_column,
                roles.order_column,
            ],
            where=anchor_filter,
            limit=2,
        )
        anchor_rows = self._sql_executor(sql_a, params_a, user_token)
        if self._budget is not None:
            self._budget.tick(rows=len(anchor_rows))

        if not anchor_rows:
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_BINDING,
                    message=(
                        f"anchor id {id_value!r} not found in {binding_name!r}"
                    ),
                    binding=binding_name,
                    details={"id": id_value, "fqn": info.fqn},
                )
            )

        anchor = anchor_rows[0]
        partition_value = anchor.get(roles.partition_column)
        order_value = anchor.get(roles.order_column)

        if order_value is None:
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.NEIGHBOR_CONFIG_MISSING,
                    message=(
                        f"anchor row order_column={roles.order_column!r} "
                        "is NULL; cannot compute window"
                    ),
                    binding=binding_name,
                )
            )

        # Step 2 — fetch siblings within [order - before, order + after].
        # We rely on numeric order_column for the arithmetic; if the column
        # is non-numeric in the DB the filter compiler will pass strings
        # through and the DB will surface the error.
        try:
            order_int = int(order_value)
        except (TypeError, ValueError):
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.NEIGHBOR_CONFIG_MISSING,
                    message=(
                        f"order_column={roles.order_column!r} value "
                        f"{order_value!r} is not coercible to int; window "
                        "arithmetic requires an integer-typed column"
                    ),
                    binding=binding_name,
                    details={"value": str(order_value)},
                )
            ) from None

        lower = order_int - before
        upper = order_int + after

        # Combine: same partition AND order in window range.
        partition_pred = FlatTableFilter.model_validate(
            {"eq": {roles.partition_column: partition_value}}
        )
        range_pred = FlatTableFilter.model_validate(
            {
                "gte": {roles.order_column: lower},
                "lte": {roles.order_column: upper},
            }
        )
        combined = AndFilter.model_validate(
            {"and": [partition_pred, range_pred]}
        )

        window_size = before + after + 1
        capped = min(window_size, PER_STMT_LIMIT_ROWS)

        sql_n, params_n = compile_select(
            info.fqn,
            schema,
            where=combined,
            order_by=[roles.order_column],
            limit=capped,
        )
        rows = self._sql_executor(sql_n, params_n, user_token)
        if self._budget is not None:
            self._budget.tick(rows=len(rows))

        payload = {
            "rows": rows,
            "anchor_id": id_value,
            "binding": binding_name,
            "window": {"lower": lower, "upper": upper},
        }
        return ToolResult(
            content=json.dumps(payload),
            data=payload,
        )

    # -- ComputeCallableProvider --------------------------------------------

    @property
    def compute_name(self) -> str:
        return "table_neighbors"

    def to_compute_callable(
        self, *, compute: Any
    ) -> Callable[..., list[dict[str, Any]]]:
        """Return a synchronous callable usable inside the compute sandbox.

        Returns the list of neighbour row dicts for the given anchor id.
        Raises ``ToolErrorException`` on any error.
        """
        del compute
        registry = self._registry
        schema_cache = self._schema_cache
        sql_executor = self._sql_executor
        budget = self._budget

        def _call(
            *,
            binding: str,
            id: str | int | float,
            user_token: str = "",
            before: int = 1,
            after: int = 1,
            roles: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            if not isinstance(binding, str) or not binding:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_BINDING,
                        message="'binding' is required and must be a non-empty string",
                    )
                )
            if id is None or not isinstance(id, (str, int, float)):
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message="'id' is required and must be a scalar",
                    )
                )
            id_str = str(id)
            before_eff = max(int(before), 0)
            after_eff = max(int(after), 0)

            explicit_roles = roles
            info = resolve_binding(registry, binding)
            info, schema = ensure_roles(
                registry=registry,
                binding_name=binding,
                info=info,
                schema_cache=schema_cache,
                sql_executor=sql_executor,
                user_token=user_token,
                explicit_roles=explicit_roles,
            )
            assert info.roles is not None
            role_map = info.roles
            if (
                role_map.id_column is None
                or role_map.partition_column is None
                or role_map.order_column is None
            ):
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.NEIGHBOR_CONFIG_MISSING,
                        message=(
                            f"binding {binding!r} is missing one of "
                            "id_column / partition_column / order_column"
                        ),
                        binding=binding,
                    )
                )

            for col in (
                role_map.id_column,
                role_map.partition_column,
                role_map.order_column,
            ):
                require_column(schema, col, fqn=info.fqn)

            anchor_filter = FlatTableFilter.model_validate(
                {"eq": {role_map.id_column: id_str}}
            )
            sql_a, params_a = compile_select(
                info.fqn,
                schema,
                columns=[
                    role_map.id_column,
                    role_map.partition_column,
                    role_map.order_column,
                ],
                where=anchor_filter,
                limit=2,
            )
            anchor_rows = sql_executor(sql_a, params_a, user_token)
            if budget is not None:
                budget.tick(rows=len(anchor_rows))
            if not anchor_rows:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_BINDING,
                        message=(
                            f"anchor id {id_str!r} not found in {binding!r}"
                        ),
                        binding=binding,
                        details={"id": id_str, "fqn": info.fqn},
                    )
                )
            anchor = anchor_rows[0]
            partition_value = anchor.get(role_map.partition_column)
            order_value = anchor.get(role_map.order_column)
            if order_value is None:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.NEIGHBOR_CONFIG_MISSING,
                        message=(
                            f"anchor row order_column={role_map.order_column!r} "
                            "is NULL; cannot compute window"
                        ),
                        binding=binding,
                    )
                )
            try:
                order_int = int(order_value)
            except (TypeError, ValueError):
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.NEIGHBOR_CONFIG_MISSING,
                        message=(
                            f"order_column={role_map.order_column!r} value "
                            f"{order_value!r} is not coercible to int"
                        ),
                        binding=binding,
                        details={"value": str(order_value)},
                    )
                ) from None

            lower = order_int - before_eff
            upper = order_int + after_eff
            partition_pred = FlatTableFilter.model_validate(
                {"eq": {role_map.partition_column: partition_value}}
            )
            range_pred = FlatTableFilter.model_validate(
                {
                    "gte": {role_map.order_column: lower},
                    "lte": {role_map.order_column: upper},
                }
            )
            combined = AndFilter.model_validate(
                {"and": [partition_pred, range_pred]}
            )
            window_size = before_eff + after_eff + 1
            capped = min(window_size, PER_STMT_LIMIT_ROWS)

            sql_n, params_n = compile_select(
                info.fqn,
                schema,
                where=combined,
                order_by=[role_map.order_column],
                limit=capped,
            )
            rows = sql_executor(sql_n, params_n, user_token)
            if budget is not None:
                budget.tick(rows=len(rows))
            return rows

        return _call
