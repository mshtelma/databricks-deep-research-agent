"""Shared OBO-SQL Unity Catalog metadata reads for Designer tool authoring.

Every read goes through the ``text_table`` :data:`SqlExecutor` (OBO, ``sql``
scope) — the path proven to work under Databricks Apps OBO. The Unity Catalog
REST list APIs and the managed-functions MCP server 403 under Apps OBO, so a
warehouse-backed ``information_schema`` query is the reliable transport here (the
same one the save-time ``uc_function`` param introspection already uses).

Injection posture: values are bound via parameter markers (``:name``); only
identifiers validated against :data:`IDENT_RE` are interpolated into SQL text
(backtick-quoted). Identifiers are lowercased — ``information_schema`` stores
them lowercase and Unity Catalog resolves catalog/schema/function names
case-insensitively, so the lowercased FQN the picker emits is what
``UCFunctionTool`` invokes at runtime.

This module is the single home for these reads: ``uc_function_introspect``
(batch, save-time) and the ``/resources`` browse + signature routes (live,
authoring-time) both build on the helpers here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databricks.sdk.service.sql import StatementParameterListItem
    from databricks_deep_research.tools.builtins.text_table.tools._common import (
        SqlExecutor,
    )

# A single Databricks-legal identifier segment (letters / digits / underscore).
# Matches ``UCFunctionTool``'s per-part rule: hyphenated catalogs are a v1 limit,
# so they are filtered from listings to keep "what you can pick" == "what you can
# run" (no drill-in dead ends). Also the interpolation guard for FROM clauses.
IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Strict 3-part FQN of IDENT_RE parts — matches UCFunctionTool + semantic_validation.
FQN_RE = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")

# Unity Catalog information_schema ``data_type`` -> our coarse param type.
# Unmapped scalars default to "string" (bound as STRING, implicit-cast).
UC_TYPE_MAP: dict[str, str] = {
    "STRING": "string",
    "VARCHAR": "string",
    "CHAR": "string",
    "INT": "integer",
    "INTEGER": "integer",
    "BIGINT": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "LONG": "integer",
    "DOUBLE": "number",
    "FLOAT": "number",
    "REAL": "number",
    "DECIMAL": "number",
    "NUMERIC": "number",
    "BOOLEAN": "boolean",
    "BOOL": "boolean",
    "DATE": "string",
    "TIMESTAMP": "string",
    "TIMESTAMP_NTZ": "string",
    "INTERVAL": "string",
    "BINARY": "string",
}
_COMPLEX_PREFIXES = ("ARRAY", "MAP", "STRUCT")


def is_complex_type(data_type: str, full_data_type: str) -> bool:
    """True for array/map/struct params — rejected by v1 scalar-only signatures."""
    return data_type in ("ARRAY", "MAP", "STRUCT") or full_data_type.upper().startswith(
        _COMPLEX_PREFIXES
    )


def uc_type_to_param_type(data_type: str) -> str:
    """Map a UC ``data_type`` to a coarse param type; unknown scalars -> string."""
    return UC_TYPE_MAP.get(data_type.upper(), "string")


def _param_item(name: str, value: str) -> StatementParameterListItem:
    from databricks.sdk.service.sql import StatementParameterListItem

    return StatementParameterListItem(name=name, value=value, type="STRING")


def _valid_ident(value: str) -> bool:
    return bool(IDENT_RE.fullmatch(value))


# ---------------------------------------------------------------------------
# Browse: catalog -> schema -> function
# ---------------------------------------------------------------------------


def list_catalogs(
    sql_executor: SqlExecutor, *, name_prefix: str = "", user_token: str = ""
) -> list[dict[str, str]]:
    """List catalogs the caller can see (``SHOW CATALOGS``), simple names only.

    ``SHOW CATALOGS`` does not accept bound parameters, so ``name_prefix`` is
    applied client-side (the catalog count is small).
    """
    rows = sql_executor("SHOW CATALOGS", [], user_token)
    prefix = name_prefix.lower()
    out: list[dict[str, str]] = []
    for row in rows:
        raw = str(row.get("catalog") or row.get("catalog_name") or "").strip().lower()
        if not raw or not _valid_ident(raw):
            continue
        if prefix and not raw.startswith(prefix):
            continue
        out.append({"name": raw, "full_name": raw})
    out.sort(key=lambda d: d["name"])
    return out


def list_schemas(
    sql_executor: SqlExecutor,
    catalog: str,
    *,
    name_prefix: str = "",
    user_token: str = "",
) -> list[dict[str, str]]:
    """List schemas in ``catalog`` via its ``information_schema.schemata``."""
    catalog = catalog.strip().lower()
    if not _valid_ident(catalog):
        raise ValueError(f"unsupported catalog identifier: {catalog!r}")
    sql = f"SELECT schema_name FROM `{catalog}`.information_schema.schemata"
    params: list[StatementParameterListItem] = []
    if name_prefix:
        sql += " WHERE schema_name LIKE :q"
        params.append(_param_item("q", f"{name_prefix.lower()}%"))
    sql += " ORDER BY schema_name"
    rows = sql_executor(sql, params, user_token)
    out: list[dict[str, str]] = []
    for row in rows:
        name = str(row.get("schema_name") or "").strip().lower()
        if not name or not _valid_ident(name):
            continue
        out.append({"name": name, "full_name": f"{catalog}.{name}"})
    return out


def list_functions(
    sql_executor: SqlExecutor,
    catalog: str,
    schema: str,
    *,
    name_prefix: str = "",
    user_token: str = "",
) -> list[dict[str, str]]:
    """List FUNCTION routines in ``catalog.schema`` (always parent-scoped).

    Returns ``{name, full_name, description}`` — ``full_name`` is the 3-part FQN
    the picker stores in ``config.function``; ``description`` carries the return
    type as a lightweight hint.
    """
    catalog = catalog.strip().lower()
    schema = schema.strip().lower()
    if not (_valid_ident(catalog) and _valid_ident(schema)):
        raise ValueError(f"unsupported identifier: {catalog!r}.{schema!r}")
    sql = (
        f"SELECT routine_name, data_type FROM `{catalog}`.information_schema.routines "
        "WHERE routine_schema = :schema AND routine_type = 'FUNCTION'"
    )
    params: list[StatementParameterListItem] = [_param_item("schema", schema)]
    if name_prefix:
        sql += " AND routine_name LIKE :q"
        params.append(_param_item("q", f"{name_prefix.lower()}%"))
    sql += " ORDER BY routine_name"
    rows = sql_executor(sql, params, user_token)
    out: list[dict[str, str]] = []
    for row in rows:
        name = str(row.get("routine_name") or "").strip().lower()
        if not name or not _valid_ident(name):
            continue
        return_type = str(row.get("data_type") or "").strip()
        out.append(
            {
                "name": name,
                "full_name": f"{catalog}.{schema}.{name}",
                "description": f"returns {return_type}" if return_type else "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Signature (parameters) — shared by save-time introspection and the live route
# ---------------------------------------------------------------------------


def run_parameters_query(
    sql_executor: SqlExecutor,
    catalog: str,
    schema: str,
    fn_names: list[str],
    *,
    user_token: str = "",
) -> list[dict[str, Any]]:
    """Query ``information_schema.parameters`` for one or more functions.

    ``catalog``/``schema``/``fn_names`` are expected pre-lowercased by callers
    (information_schema stores identifiers lowercase). Raises on empty
    ``fn_names`` to avoid an ``IN ()`` syntax error.
    """
    if not fn_names:
        raise ValueError("run_parameters_query requires at least one function name")
    markers = ", ".join(f":fn{i}" for i in range(len(fn_names)))
    sql = (
        "SELECT specific_name, parameter_name, data_type, full_data_type, "
        "ordinal_position, parameter_default "
        f"FROM `{catalog}`.information_schema.parameters "
        "WHERE specific_schema = :schema "
        f"AND specific_name IN ({markers}) "
        "AND parameter_name IS NOT NULL "
        "ORDER BY specific_name, ordinal_position"
    )
    params: list[StatementParameterListItem] = [_param_item("schema", schema)]
    for i, fn in enumerate(fn_names):
        params.append(_param_item(f"fn{i}", fn))
    return sql_executor(sql, params, user_token)


def parse_parameter_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]] | None]:
    """Group ``information_schema.parameters`` rows into per-function param lists.

    Returns ``specific_name`` (lowercased) -> ordered ``[{name, type, required}]``,
    or ``None`` when the function has any non-scalar (array/map/struct) parameter
    (rejected in v1). Functions with no parameter rows are absent from the map —
    callers treat absence as "no-arg" (empty list). Warning text is left to the
    caller so this stays a pure parser.
    """
    raw: dict[str, list[dict[str, Any]]] = {}
    complex_fns: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        specific = str(row.get("specific_name") or "").lower()
        pname = row.get("parameter_name")
        if not specific or pname is None:
            continue
        seen.add(specific)
        data_type = str(row.get("data_type") or "").upper()
        full_data_type = str(row.get("full_data_type") or data_type)
        if is_complex_type(data_type, full_data_type):
            complex_fns.add(specific)
            continue
        raw.setdefault(specific, []).append(
            {
                "name": str(pname),
                "type": uc_type_to_param_type(data_type),
                "required": row.get("parameter_default") is None,
            }
        )
    result: dict[str, list[dict[str, Any]] | None] = {}
    for fn in seen:
        result[fn] = None if fn in complex_fns else raw.get(fn, [])
    return result


def get_signature(
    sql_executor: SqlExecutor, fqn: str, *, user_token: str = ""
) -> dict[str, Any]:
    """Introspect a single function's signature for the live picker.

    Returns ``{"function": fqn, "params": [...], "scalar": bool}``. ``scalar`` is
    ``False`` when the function has a non-scalar parameter (params then empty and
    the UI cannot auto-map). A no-arg / unknown function yields ``params=[]`` with
    ``scalar=True`` (runtime passes any provided args through untyped). Raises
    ``ValueError`` on a malformed FQN; query errors propagate to the caller, which
    is responsible for fail-soft handling.
    """
    if not FQN_RE.fullmatch(fqn):
        raise ValueError(
            f"function {fqn!r} must be 'catalog.schema.function' of [A-Za-z0-9_] parts"
        )
    catalog, schema, fn = (part.lower() for part in fqn.split("."))
    rows = run_parameters_query(
        sql_executor, catalog, schema, [fn], user_token=user_token
    )
    by_fn = parse_parameter_rows(rows)
    # Absent => no parameter rows (no-arg / unknown): scalar with empty params.
    # Present-and-None => a non-scalar (array/map/struct) param: cannot auto-map.
    params = by_fn[fn] if fn in by_fn else []
    scalar = params is not None
    return {
        "function": f"{catalog}.{schema}.{fn}",
        "params": params if scalar else [],
        "scalar": scalar,
    }
