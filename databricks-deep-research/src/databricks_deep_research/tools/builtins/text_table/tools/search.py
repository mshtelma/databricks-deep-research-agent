"""TableSearchTool — SQL-side substring search over a binding's content column.

The relevance score is a placeholder (1.0 for every match). Real BM25 /
vector ranking is a future enhancement; for now we expose a deterministic
substring filter so callers can verify wiring end-to-end.

Filter strategy: ``filter_dsl`` handles structured predicates while the
query text compiles to a parameterized, case-insensitive ``LIKE`` predicate.
User text is never interpolated into the SQL string.
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

from ..binding import BindingInfo
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
    ensure_roles,
    get_user_token,
    require_column,
    resolve_binding,
)

__all__ = ["TableSearchTool"]

_SNIPPET_LEN = 512
_DEFAULT_LIMIT = 10
_MAX_QUERY_LEN = 500


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
    # Accept any of the four shapes; let pydantic raise if invalid.
    if "and" in raw:
        return AndFilter.model_validate(raw)
    if "or" in raw:
        return OrFilter.model_validate(raw)
    if "not" in raw:
        return NotFilter.model_validate(raw)
    return FlatTableFilter.model_validate(raw)


def _make_snippet(content: str, query: str) -> str:
    """Return a deterministic snippet, centered near the match when possible."""
    if len(content) <= _SNIPPET_LEN:
        return content
    idx = content.lower().find(query.lower())
    if idx < 0:
        return content[:_SNIPPET_LEN]
    start = max(0, idx - (_SNIPPET_LEN // 3))
    end = start + _SNIPPET_LEN
    if end > len(content):
        end = len(content)
        start = max(0, end - _SNIPPET_LEN)
    return content[start:end]


def _matches_query(content: Any, query: str) -> bool:
    return isinstance(content, str) and query.lower() in content.lower()


class TableSearchTool:
    """Substring search over the binding's content column."""

    def __init__(
        self,
        *,
        registry: TableBindingRegistry,
        schema_cache: _SchemaCacheLike,
        sql_executor: SqlExecutor,
        budget: Budget3D | None = None,
        name: str = "table_search",
        description: str | None = None,
        default_binding: str | None = None,
        default_columns: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._schema_cache = schema_cache
        self._sql_executor = sql_executor
        self._budget = budget
        self._name = name
        self._description = description
        self._default_binding = default_binding
        self._default_columns = list(default_columns) if default_columns else None

    @property
    def definition(self) -> ToolDefinition:
        description = self._description or (
            "Search a registered table for rows whose content column "
            "contains the query substring. Optional 'where' filter "
            "narrows the candidate rows before substring matching."
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
                    "binding": {
                        "type": "string",
                        "description": "Name of a registered binding.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Substring to match (case-insensitive).",
                    },
                    "where": WHERE_PARAM_SCHEMA,
                    "limit": {
                        "type": "integer",
                        "default": _DEFAULT_LIMIT,
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                    },
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
                "required": (
                    ["query"] if self._default_binding else ["binding", "query"]
                ),
            },
            source_type="builtin",
            source_kind="text_table",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        binding = arguments.get("binding", self._default_binding)
        if not isinstance(binding, str) or not binding:
            raise ValueError("'binding' is required and must be a non-empty string")

        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("'query' is required and must be a non-empty string")
        if len(query) > _MAX_QUERY_LEN:
            raise ValueError(f"'query' must be {_MAX_QUERY_LEN} characters or less")

        where = arguments.get("where")
        if where is not None and not isinstance(where, dict):
            raise ValueError("'where' must be a dict when provided")

        limit_raw = arguments.get("limit", _DEFAULT_LIMIT)
        if not isinstance(limit_raw, int) or limit_raw < 1:
            raise ValueError("'limit' must be a positive integer")
        limit = min(limit_raw, PER_STMT_LIMIT_ROWS)

        offset_raw = arguments.get("offset", 0)
        if not isinstance(offset_raw, int) or offset_raw < 0:
            raise ValueError("'offset' must be a non-negative integer")

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
            "query": query,
            "where": where,
            "limit": limit,
            "offset": offset_raw,
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
        query: str = arguments["query"]
        where_dict: dict[str, Any] | None = arguments.get("where")
        limit: int = arguments["limit"]
        offset: int = arguments["offset"]
        extra_cols: list[str] | None = arguments.get("columns")
        explicit_roles: dict[str, Any] | None = arguments.get("roles")

        info: BindingInfo = resolve_binding(self._registry, binding_name)
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
        content_column = info.roles.content_column
        require_column(schema, id_column, fqn=info.fqn)
        require_column(schema, content_column, fqn=info.fqn)

        projection: list[str] = [id_column, content_column]
        if extra_cols:
            for c in extra_cols:
                require_column(schema, c, fqn=info.fqn)
                if c not in projection:
                    projection.append(c)

        where_filter = _parse_filter(where_dict)

        sql, params = compile_select(
            info.fqn,
            schema,
            columns=projection,
            where=where_filter,
            text_search=(content_column, query),
            limit=limit,
            offset=offset,
        )

        rows = self._sql_executor(sql, params, user_token)
        if self._budget is not None:
            self._budget.tick(rows=len(rows))

        results: list[dict[str, Any]] = []
        for row in rows:
            content = row.get(content_column, "") or ""
            if not _matches_query(content, query):
                continue
            snippet = _make_snippet(content, query) if isinstance(content, str) else ""
            results.append(
                {
                    "id": row.get(id_column),
                    "snippet": snippet,
                    "score": 1.0,
                    "row": row,
                }
            )

        payload = {
            "binding": binding_name,
            "query": query,
            "results": results,
            "total_matched": len(results),
        }
        return ToolResult(
            content=json.dumps(payload),
            data={
                "rows": results,
                "binding": binding_name,
                "query": query,
                "total_matched": len(results),
            },
        )

    # -- ComputeCallableProvider --------------------------------------------

    @property
    def compute_name(self) -> str:
        return "table_search"

    def to_compute_callable(
        self, *, compute: Any
    ) -> Callable[..., list[dict[str, Any]]]:
        """Return a synchronous callable usable inside the compute sandbox.

        Validates and runs the substring search; raises ``ToolErrorException``
        on any error rather than returning a JSON envelope. Returns the list
        of result dicts ``{"id", "snippet", "score", "row"}``.
        """
        del compute
        registry = self._registry
        schema_cache = self._schema_cache
        sql_executor = self._sql_executor
        budget = self._budget

        def _call(
            *,
            binding: str,
            query: str,
            user_token: str = "",
            where: dict[str, Any] | None = None,
            limit: int = _DEFAULT_LIMIT,
            offset: int = 0,
            columns: list[str] | None = None,
            roles: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            if not isinstance(binding, str) or not binding:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_BINDING,
                        message="'binding' is required and must be a non-empty string",
                    )
                )
            if not isinstance(query, str) or not query:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message="'query' is required and must be a non-empty string",
                    )
                )
            if len(query) > _MAX_QUERY_LEN:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message=f"'query' must be {_MAX_QUERY_LEN} chars or fewer",
                    )
                )
            limit_eff = min(max(int(limit), 1), PER_STMT_LIMIT_ROWS)
            offset_eff = max(int(offset), 0)

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
            content_column = info.roles.content_column
            require_column(schema, id_column, fqn=info.fqn)
            require_column(schema, content_column, fqn=info.fqn)

            projection: list[str] = [id_column, content_column]
            if columns:
                for c in columns:
                    require_column(schema, c, fqn=info.fqn)
                    if c not in projection:
                        projection.append(c)

            where_filter = _parse_filter(where)

            sql, params = compile_select(
                info.fqn,
                schema,
                columns=projection,
                where=where_filter,
                text_search=(content_column, query),
                limit=limit_eff,
                offset=offset_eff,
            )
            rows = sql_executor(sql, params, user_token)
            if budget is not None:
                budget.tick(rows=len(rows))

            results: list[dict[str, Any]] = []
            for row in rows:
                content = row.get(content_column, "") or ""
                if not _matches_query(content, query):
                    continue
                snippet = (
                    _make_snippet(content, query) if isinstance(content, str) else ""
                )
                results.append(
                    {
                        "id": row.get(id_column),
                        "snippet": snippet,
                        "score": 1.0,
                        "row": row,
                    }
                )
            return results

        return _call
