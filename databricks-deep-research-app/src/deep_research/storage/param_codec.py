"""Python ↔ Databricks SDK parameter conversion.

Every SQL statement executed via the `SQLWarehouseBackend` passes its literals
through this module; there is zero string interpolation in the SQL path. This
both prevents injection bugs and gives Delta correct type information for
partition pruning.

Conversion rules (Python → SDK type):

* `None`                   → value=None   (type omitted; SDK handles NULL)
* `bool`                   → BOOLEAN      ("true"|"false")
* `int` (bool-excluded)    → BIGINT       (str)
* `float`                  → DOUBLE       (str)
* `Decimal`                → DECIMAL      (str)
* `datetime`               → TIMESTAMP    (isoformat, UTC if naive)
* `date`                   → DATE         (isoformat)
* `UUID`                   → STRING       (str)
* `Enum`                   → STRING       (value)
* `bytes`                  → BINARY       (hex)  — rare, supported for completeness
* `dict` / `list`          → STRING       (json.dumps)  — caller is expected
  to parse via `from_json(..., 'SCHEMA')` on the Delta side or `:jsonb` on
  Postgres; this module only encodes.
* everything else          → STRING       (str())

Two entry points:

- `to_param(name, value)` — single `StatementParameterListItem`
- `params(mapping)`       — convenience that flattens a dict
- `row_params(prefix, row)` — helper to bind a dict-row using `:prefix_<col>` names

The SDK import is lazy so test environments without `databricks-sdk` can still
import this module for the Lakebase path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from databricks.sdk.service.sql import StatementParameterListItem


def _encode_value(value: Any) -> tuple[str | None, str | None]:
    """Return (stringified_value, type_name). Type is `None` for SQL NULL."""
    if value is None:
        return None, None
    # bool before int because bool is a subclass of int.
    if isinstance(value, bool):
        return ("true" if value else "false"), "BOOLEAN"
    if isinstance(value, int):
        return str(value), "BIGINT"
    if isinstance(value, float):
        return str(value), "DOUBLE"
    if isinstance(value, Decimal):
        return str(value), "DECIMAL"
    if isinstance(value, datetime):
        # Normalize naive timestamps to UTC so the warehouse parses consistently.
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat(), "TIMESTAMP"
    if isinstance(value, date):
        return value.isoformat(), "DATE"
    if isinstance(value, UUID):
        return str(value), "STRING"
    if isinstance(value, Enum):
        return str(value.value), "STRING"
    if isinstance(value, bytes):
        return value.hex(), "BINARY"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default, sort_keys=True), "STRING"
    return str(value), "STRING"


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder for nested UUIDs, datetimes, Decimals, Enums."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_param(name: str, value: Any) -> StatementParameterListItem:
    """Build a single `StatementParameterListItem` from a Python value."""
    try:
        from databricks.sdk.service.sql import StatementParameterListItem
    except ImportError as exc:  # pragma: no cover — only reached in stripped envs
        raise RuntimeError(
            "databricks-sdk not installed; "
            "install it to use the SQL Warehouse backend"
        ) from exc

    str_value, type_name = _encode_value(value)
    return StatementParameterListItem(name=name, value=str_value, type=type_name)


def params(mapping: Mapping[str, Any]) -> list[StatementParameterListItem]:
    """Convert a mapping of name → value to the SDK's list form."""
    return [to_param(name, value) for name, value in mapping.items()]


def row_params(prefix: str, row: Mapping[str, Any]) -> list[StatementParameterListItem]:
    """Bind a dict-row under prefixed parameter names.

    Used when inlining a row into a MERGE's USING clause: each column becomes
    `:{prefix}_{col}` and the caller constructs the SQL referencing the same
    prefixed names.
    """
    return params({f"{prefix}_{key}": value for key, value in row.items()})


# -- Lightweight stub for importability without databricks-sdk -------------

def encoded(value: Any) -> tuple[str | None, str | None]:
    """Public wrapper around `_encode_value` for tests and non-SDK callers."""
    return _encode_value(value)
