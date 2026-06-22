"""Boot-time validation for BOUND text-table bindings.

Run once at app startup against the BOUND entries in
:class:`TableBindingRegistry`. Confirms each binding's ``fqn`` is reachable
through the schema cache, that every column referenced in the ``RoleMap`` and
``numeric_columns`` exists in the live schema, and that role/numeric column
types are compatible with the text-table contract.

Two entry points:

- :func:`validate_bound_binding` — single binding, raises on first failure.
- :func:`validate_all_bound_bindings` — registry-wide. Defaults to raise-
  first; pass ``raise_on_error=False`` to collect and return a
  :class:`BindingValidationReport` listing every failure.

DISCOVERED bindings are intentionally skipped — their schema is queried
lazily at first use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .binding import BindingInfo, BindingSource
from .error_codes import ErrorCode, ToolError, ToolErrorException
from .parsers import get_parser
from .registry import TableBindingRegistry
from .schema_cache import Schema, SchemaColumn

__all__ = [
    "BindingValidationReport",
    "validate_bound_binding",
    "validate_all_bound_bindings",
]


class _SchemaCacheLike(Protocol):
    """Minimal protocol — only ``.get(fqn, user_token)`` is required."""

    def get(self, fqn: str, user_token: str) -> Schema: ...


@dataclass(frozen=True)
class BindingValidationReport:
    """Outcome of registry-wide boot validation.

    Attributes
    ----------
    validated:
        Tuple of binding names that passed validation, in registration
        order.
    errors:
        Tuple of :class:`ToolError` accumulated across all failures
        (only populated when ``raise_on_error=False``).
    """

    validated: tuple[str, ...]
    errors: tuple[ToolError, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_column_error(
    binding_name: str, column: str, fqn: str, role: str
) -> ToolError:
    return ToolError(
        error_code=ErrorCode.INVALID_COLUMN,
        message=(
            f"binding {binding_name!r} declares {role}={column!r} but the "
            f"column is not present in {fqn!r}"
        ),
        binding=binding_name,
        hint=(
            "Edit the binding YAML so the role column matches a real "
            "column in the table schema, or update the table to add the "
            "missing column."
        ),
        details={"fqn": fqn, "column": column, "role": role},
    )


def _invalid_binding_error(
    binding_name: str, fqn: str, message: str, details: dict[str, object]
) -> ToolError:
    return ToolError(
        error_code=ErrorCode.INVALID_BINDING,
        message=message,
        binding=binding_name,
        hint=(
            "Edit the BOUND table binding so it declares a complete role map "
            "and valid structured passage parser names."
        ),
        details={"fqn": fqn, **details},
    )


def _invalid_type_error(
    binding_name: str,
    column: SchemaColumn,
    fqn: str,
    role: str,
    expected: str,
) -> ToolError:
    return ToolError(
        error_code=ErrorCode.INVALID_COLUMN,
        message=(
            f"binding {binding_name!r} declares {role}={column.name!r} "
            f"with incompatible type {column.data_type!r}; expected {expected}"
        ),
        binding=binding_name,
        hint=(
            "Update the role map or table schema so role columns match the "
            "expected text-table type contract."
        ),
        details={
            "fqn": fqn,
            "column": column.name,
            "role": role,
            "actual_type": column.data_type,
            "expected": expected,
        },
    )


def _normalized_type(data_type: str) -> str:
    return data_type.strip().lower().split("(", 1)[0]


def _is_string_type(column: SchemaColumn) -> bool:
    typ = _normalized_type(column.data_type)
    return typ in {"char", "string", "text", "varchar"}


def _is_integer_type(column: SchemaColumn) -> bool:
    typ = _normalized_type(column.data_type)
    return typ in {"bigint", "int", "integer", "long", "smallint", "tinyint"}


def _is_numeric_type(column: SchemaColumn) -> bool:
    typ = _normalized_type(column.data_type)
    return _is_integer_type(column) or typ in {
        "decimal",
        "double",
        "float",
        "number",
        "numeric",
        "real",
    }


def _is_date_type(column: SchemaColumn) -> bool:
    typ = _normalized_type(column.data_type)
    return typ in {"date", "datetime", "string", "timestamp", "timestamp_ltz", "timestamp_ntz"}


def _validate_role_type(
    *,
    info: BindingInfo,
    schema: Schema,
    column_name: str,
    role: str,
    predicate: Callable[[SchemaColumn], bool],
    expected: str,
) -> ToolError | None:
    column = schema.get_column(column_name)
    if column is None:
        return _missing_column_error(info.name, column_name, info.fqn, role)
    if not predicate(column):
        return _invalid_type_error(info.name, column, info.fqn, role, expected)
    return None


def _validate_structured_passages(info: BindingInfo, schema: Schema) -> list[ToolError]:
    if not info.structured_passages:
        return []

    errors: list[ToolError] = []
    if info.roles is None or not info.roles.type_column:
        errors.append(
            _invalid_binding_error(
                info.name,
                info.fqn,
                (
                    f"binding {info.name!r} declares structured_passages but "
                    "does not declare roles.type_column"
                ),
                {"structured_passages": dict(info.structured_passages)},
            )
        )
        return errors

    type_column = schema.get_column(info.roles.type_column)
    if type_column is not None and not _is_string_type(type_column):
        errors.append(
            _invalid_type_error(
                info.name,
                type_column,
                info.fqn,
                "type",
                "string/varchar/text",
            )
        )

    sample_by_parser = {
        "html": "<table><tr><th>name</th></tr><tr><td>value</td></tr></table>",
        "markdown": "# Heading\nBody",
        "json": '{"ok": true}',
    }
    for type_value, parser_name in info.structured_passages.items():
        try:
            parser = get_parser(parser_name)  # type: ignore[arg-type]
            parser(sample_by_parser.get(parser_name, "sample"))
        except Exception as exc:  # noqa: BLE001 — parser registry is the boundary
            errors.append(
                _invalid_binding_error(
                    info.name,
                    info.fqn,
                    (
                        f"binding {info.name!r} declares parser "
                        f"{parser_name!r} for type {type_value!r}, but the "
                        f"parser is not valid: {exc}"
                    ),
                    {
                        "type_value": type_value,
                        "parser": parser_name,
                        "cause": str(exc),
                    },
                )
            )
    return errors


def _validate_columns(
    info: BindingInfo, schema: Schema
) -> list[ToolError]:
    """Return the list of role/numeric column errors (empty if all OK)."""
    errors: list[ToolError] = []

    if info.roles is None:
        errors.append(
            _invalid_binding_error(
                info.name,
                info.fqn,
                f"BOUND binding {info.name!r} must declare roles",
                {},
            )
        )
        return errors

    roles = info.roles
    role_specs = (
        ("id", roles.id_column, _is_string_type, "string/varchar/text"),
        ("content", roles.content_column, _is_string_type, "string/varchar/text"),
        ("partition", roles.partition_column, _is_string_type, "string/varchar/text"),
        ("label", roles.label_column, _is_string_type, "string/varchar/text"),
        ("type", roles.type_column, _is_string_type, "string/varchar/text"),
        ("date", roles.date_column, _is_date_type, "date/timestamp/string"),
        ("order", roles.order_column, _is_integer_type, "integer/bigint"),
    )
    for role_name, value, predicate, expected in role_specs:
        if value:
            error = _validate_role_type(
                info=info,
                schema=schema,
                column_name=value,
                role=role_name,
                predicate=predicate,
                expected=expected,
            )
            if error is not None:
                errors.append(error)

    for col in info.numeric_columns:
        schema_col = schema.get_column(col)
        if schema_col is None:
            errors.append(
                _missing_column_error(info.name, col, info.fqn, "numeric")
            )
        elif not _is_numeric_type(schema_col):
            errors.append(
                _invalid_type_error(
                    info.name, schema_col, info.fqn, "numeric", "numeric"
                )
            )

    errors.extend(_validate_structured_passages(info, schema))
    return errors


def validate_bound_binding(
    info: BindingInfo,
    *,
    schema_cache: _SchemaCacheLike,
    user_token: str,
) -> None:
    """Validate one BOUND binding against the live schema.

    Raises :class:`ToolErrorException` on the first failure.
    """
    if info.source is not BindingSource.BOUND:
        raise ValueError(
            f"validate_bound_binding requires source=BOUND, got {info.source!r}"
        )

    try:
        schema = schema_cache.get(info.fqn, user_token)
    except Exception as exc:  # noqa: BLE001 — fetch surface intentionally broad
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.SCHEMA_FETCH_FAILED,
                message=(
                    f"failed to fetch schema for binding {info.name!r} "
                    f"(fqn={info.fqn!r}): {exc}"
                ),
                binding=info.name,
                hint=(
                    "Confirm the binding fqn is correct and the app SP has "
                    "USE / SELECT on the table."
                ),
                details={"fqn": info.fqn, "cause": str(exc)},
            )
        ) from exc

    errors = _validate_columns(info, schema)
    if errors:
        # Single-binding API — raise the first error so the caller sees a
        # crisp boot failure.
        raise ToolErrorException(errors[0])


def validate_all_bound_bindings(
    registry: TableBindingRegistry,
    *,
    schema_cache: _SchemaCacheLike,
    user_token: str,
    raise_on_error: bool = True,
) -> BindingValidationReport:
    """Validate every BOUND binding in the registry.

    Parameters
    ----------
    registry:
        The :class:`TableBindingRegistry` to validate.
    schema_cache:
        Live schema cache (must implement ``.get(fqn, user_token) -> Schema``).
    user_token:
        Caller's user token for schema fetch authorization.
    raise_on_error:
        When True (default), raise :class:`ToolErrorException` on the first
        failure — appropriate for app boot. When False, collect every
        failure into the returned :class:`BindingValidationReport`.

    Returns
    -------
    BindingValidationReport
        Names that passed and errors collected.
    """
    snapshot = registry.metadata_snapshot()
    validated: list[str] = []
    errors: list[ToolError] = []

    for name, info in snapshot.items():
        if info.source is not BindingSource.BOUND:
            continue
        try:
            validate_bound_binding(
                info, schema_cache=schema_cache, user_token=user_token
            )
        except ToolErrorException as exc:
            if raise_on_error:
                raise
            errors.append(exc.error)
            continue
        validated.append(name)

    return BindingValidationReport(
        validated=tuple(validated), errors=tuple(errors)
    )
