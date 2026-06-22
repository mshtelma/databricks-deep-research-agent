"""Shared helpers for table_* tools.

Avoid leaking these into the public surface — they are tool-implementation
internals.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..binding import BindingInfo, BindingSource, RoleMap
from ..error_codes import ErrorCode, ToolError, ToolErrorException
from ..filter_dsl import (
    AndFilter,
    FlatTableFilter,
    NotFilter,
    OrFilter,
)
from ..registry import TableBindingRegistry
from ..role_inference import infer_roles
from ..schema_cache import Schema

if TYPE_CHECKING:
    from databricks.sdk.service.sql import StatementParameterListItem

# Public type aliases used across all table_* tools.
TableFilterLike = FlatTableFilter | AndFilter | OrFilter | NotFilter
SqlExecutor = Callable[
    [str, list["StatementParameterListItem"], str], list[dict[str, Any]]
]


@runtime_checkable
class _SchemaCacheLike(Protocol):
    """Anything with a ``.get(fqn, user_token) -> Schema`` method."""

    def get(self, fqn: str, user_token: str) -> Schema: ...


_ROLE_ALIASES: dict[str, str] = {
    "id": "id_column",
    "id_column": "id_column",
    "content": "content_column",
    "content_column": "content_column",
    "order": "order_column",
    "order_column": "order_column",
    "partition": "partition_column",
    "partition_column": "partition_column",
    "label": "label_column",
    "label_column": "label_column",
    "type": "type_column",
    "type_column": "type_column",
    "date": "date_column",
    "date_column": "date_column",
}


def get_user_token(extras: dict[str, Any]) -> str:
    """Extract the OBO user token from ``ToolContext.extras``.

    Falls back to an empty string when missing — downstream callers will
    surface auth errors from the SQL executor if a token is actually
    required. This keeps the framework decoupled from the auth layer.
    """
    token = extras.get("user_token", "")
    if not isinstance(token, str):
        return ""
    return token


def resolve_binding(
    registry: TableBindingRegistry, name: str
) -> BindingInfo:
    """Look up a binding by name, raising ToolErrorException on miss."""
    return registry.get(name)


def require_roles_or_raise(info: BindingInfo) -> None:
    """Raise INVALID_BINDING if a non-inferable binding has no RoleMap."""
    if info.roles is None:
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_BINDING,
                message=(
                    f"binding {info.name!r} has no role mapping; "
                    "DISCOVERED bindings require explicit roles=… or a "
                    "first-call inference pass to populate roles"
                ),
                binding=info.name,
                details={"name": info.name, "fqn": info.fqn},
            )
        )


def parse_roles(raw: dict[str, Any] | None) -> RoleMap | None:
    """Parse user-supplied role aliases into a ``RoleMap``.

    Tool callers may use compact role names from the spec (``id``,
    ``content``, ``partition``) or the dataclass field names
    (``id_column``, ``content_column``, ...). Unknown keys are rejected so a
    typo cannot silently produce a bad binding.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_BINDING,
                message="'roles' must be an object when provided",
            )
        )
    parsed: dict[str, str | None] = {
        "order_column": None,
        "partition_column": None,
        "label_column": None,
        "type_column": None,
        "date_column": None,
    }
    for key, value in raw.items():
        canonical = _ROLE_ALIASES.get(key)
        if canonical is None:
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_BINDING,
                    message=f"unknown role key {key!r}",
                    details={"allowed": sorted(_ROLE_ALIASES)},
                )
            )
        if value is not None and not isinstance(value, str):
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_BINDING,
                    message=f"role {key!r} must be a string or null",
                )
            )
        parsed[canonical] = value

    id_column = parsed.get("id_column")
    content_column = parsed.get("content_column")
    if not isinstance(id_column, str) or not id_column:
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_BINDING,
                message="'roles' must include id/id_column",
            )
        )
    if not isinstance(content_column, str) or not content_column:
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_BINDING,
                message="'roles' must include content/content_column",
            )
        )
    return RoleMap(
        id_column=id_column,
        content_column=content_column,
        order_column=parsed["order_column"],
        partition_column=parsed["partition_column"],
        label_column=parsed["label_column"],
        type_column=parsed["type_column"],
        date_column=parsed["date_column"],
    )


def _validate_role_columns(info: BindingInfo, schema: Schema, roles: RoleMap) -> None:
    required = {
        "id_column": roles.id_column,
        "content_column": roles.content_column,
    }
    optional = {
        "order_column": roles.order_column,
        "partition_column": roles.partition_column,
        "label_column": roles.label_column,
        "type_column": roles.type_column,
        "date_column": roles.date_column,
    }
    for column in {**required, **optional}.values():
        if column is not None:
            require_column(schema, column, fqn=info.fqn)


def _sample_rows_for_inference(
    *,
    info: BindingInfo,
    schema: Schema,
    sql_executor: SqlExecutor,
    user_token: str,
    sample_limit: int = 1000,
) -> list[dict[str, Any]]:
    from ..sql_compiler import compile_select

    sql, params = compile_select(info.fqn, schema, limit=sample_limit)
    return sql_executor(sql, params, user_token)


def ensure_roles(
    *,
    registry: TableBindingRegistry,
    binding_name: str,
    info: BindingInfo,
    schema_cache: _SchemaCacheLike,
    sql_executor: SqlExecutor,
    user_token: str,
    explicit_roles: dict[str, Any] | None = None,
) -> tuple[BindingInfo, Schema]:
    """Return a binding with roles, inferring DISCOVERED roles when needed."""
    schema = schema_cache.get(info.fqn, user_token)
    roles = parse_roles(explicit_roles)
    if roles is not None:
        _validate_role_columns(info, schema, roles)
        updated = registry.update_roles(binding_name, roles)
        return updated, schema

    if info.roles is not None:
        _validate_role_columns(info, schema, info.roles)
        return info, schema

    if info.source is not BindingSource.DISCOVERED:
        require_roles_or_raise(info)

    try:
        sample_rows = _sample_rows_for_inference(
            info=info,
            schema=schema,
            sql_executor=sql_executor,
            user_token=user_token,
        )
        inferred = infer_roles(schema, sample_rows=sample_rows)
        _validate_role_columns(info, schema, inferred)
    except ToolErrorException as exc:
        details = {
            **dict(exc.error.details),
            "binding": binding_name,
            "fqn": info.fqn,
        }
        raise ToolErrorException(
            ToolError(
                error_code=exc.error.error_code,
                message=exc.error.message,
                binding=binding_name,
                hint=(
                    exc.error.hint
                    or "Pass roles={'id': '...', 'content': '...'} explicitly."
                ),
                details=details,
            )
        ) from exc

    updated = registry.update_roles(binding_name, inferred)
    return updated, schema


def require_column(
    schema: Schema, column: str, *, fqn: str
) -> None:
    """Raise INVALID_COLUMN if column is missing from schema."""
    if column not in schema.column_map:
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_COLUMN,
                message=f"column {column!r} not in schema for {fqn}",
                details={"column": column, "fqn": fqn},
            )
        )


def quote_ident(name: str) -> str:
    """Backtick-quote a single identifier."""
    if "`" in name:
        # The SQL compiler also rejects backticks in idents; mirror here.
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_COLUMN,
                message=f"identifier {name!r} contains a backtick",
                details={"identifier": name},
            )
        )
    return f"`{name}`"


def quote_fqn(fqn: str) -> str:
    """Backtick-quote a 3-part identifier ``catalog.schema.table``."""
    parts = fqn.split(".")
    if len(parts) != 3 or any(not p for p in parts) or any("`" in p for p in parts):
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_BINDING,
                message=(
                    f"invalid FQN {fqn!r}: must be three-part "
                    "'catalog.schema.table' with non-empty parts and no backticks"
                ),
                details={"fqn": fqn},
            )
        )
    return ".".join(quote_ident(p) for p in parts)
