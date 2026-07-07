"""Shared OBO-SQL Unity Catalog metadata reads for Designer tool authoring.

Browse (catalog -> schema -> function) and signature introspection go through
the ``text_table`` :data:`SqlExecutor` (OBO, ``sql`` scope). They use ``SHOW``
and ``DESCRIBE FUNCTION`` — which require only the ``BROWSE`` privilege — rather
than ``information_schema`` (which requires ``USE CATALOG`` and so returned
``[INSUFFICIENT_PERMISSIONS]`` for catalogs a user can see but not use, e.g. most
workshop/demo catalogs). ``check_use_catalog`` is the one place we deliberately
probe ``information_schema`` — as a litmus for *run*-readiness (invoking a
function needs USE CATALOG + USE SCHEMA + EXECUTE, unlike listing it).

Injection posture: only identifiers validated against :data:`IDENT_RE` are ever
interpolated into SQL text (backtick-quoted). The session catalog/schema context
needed for ``SHOW USER FUNCTIONS`` (a cross-catalog ``IN cat.schema`` reference
is unsupported) is set on the *executor* as a bound kwarg, never interpolated.
Identifiers are lowercased — Unity Catalog resolves names case-insensitively and
the lowercased FQN the picker emits is what ``UCFunctionTool`` invokes at runtime.

This module is the single home for these reads: ``uc_function_introspect``
(save-time param fill) and the ``/resources`` browse + signature routes (live,
authoring-time) both build on the helpers here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databricks_deep_research.tools.builtins.text_table.tools._common import (
        SqlExecutor,
    )

# A single Databricks-legal identifier segment (letters / digits / underscore).
# Matches ``UCFunctionTool``'s per-part rule: hyphenated catalogs are a v1 limit,
# so they are filtered from listings to keep "what you can pick" == "what you can
# run" (no drill-in dead ends). Also the interpolation guard for FROM/IN clauses.
IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Strict 3-part FQN of IDENT_RE parts — matches UCFunctionTool + semantic_validation.
FQN_RE = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")

# Unity Catalog ``data_type`` -> our coarse param type. Unmapped scalars default
# to "string" (bound as STRING, implicit-cast).
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


def _valid_ident(value: str) -> bool:
    return bool(IDENT_RE.fullmatch(value))


def _first_col(row: dict[str, Any]) -> Any:
    """First column value of a result row. ``SHOW`` commands use varied column
    names (``catalog`` / ``databaseName`` / ``function``); we only ever read the
    single/first column, so read positionally to stay name-agnostic."""
    for value in row.values():
        return value
    return None


# ---------------------------------------------------------------------------
# Error classification (shared by browse error surfacing + run-readiness probe)
# ---------------------------------------------------------------------------


def classify_sql_error(message: str) -> str:
    """Classify a Statement Execution error message: permission / not_found / other."""
    upper = message.upper()
    if (
        "INSUFFICIENT_PERMISSIONS" in upper
        or "INSUFFICIENT PRIVILEGES" in upper
        or "USE CATALOG" in upper
        or "PERMISSION_DENIED" in upper
        or "42501" in upper
    ):
        return "permission"
    if (
        "TABLE_OR_VIEW_NOT_FOUND" in upper
        or "SCHEMA_NOT_FOUND" in upper
        or "CATALOG_NOT_FOUND" in upper
        or "42P01" in upper
    ):
        return "not_found"
    return "other"


def _is_permission_error(message: str) -> bool:
    return classify_sql_error(message) == "permission"


# ---------------------------------------------------------------------------
# Browse: catalog -> schema -> function (SHOW commands; BROWSE-sufficient)
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
        raw = str(_first_col(row) or "").strip().lower()
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
    """List schemas in ``catalog`` via ``SHOW SCHEMAS IN`` (needs only BROWSE)."""
    catalog = catalog.strip().lower()
    if not _valid_ident(catalog):
        raise ValueError(f"unsupported catalog identifier: {catalog!r}")
    rows = sql_executor(f"SHOW SCHEMAS IN `{catalog}`", [], user_token)
    prefix = name_prefix.lower()
    out: list[dict[str, str]] = []
    for row in rows:
        name = str(_first_col(row) or "").strip().lower()
        if not name or not _valid_ident(name):
            continue
        if prefix and not name.startswith(prefix):
            continue
        out.append({"name": name, "full_name": f"{catalog}.{name}"})
    out.sort(key=lambda d: d["name"])
    return out


def list_functions(
    sql_executor: SqlExecutor,
    catalog: str,
    schema: str,
    *,
    name_prefix: str = "",
    user_token: str = "",
) -> list[dict[str, str]]:
    """List user (UDF) functions in ``catalog.schema`` via ``SHOW USER FUNCTIONS``.

    The ``sql_executor`` MUST be bound to ``catalog``/``schema`` session context
    (a cross-catalog ``IN cat.schema`` reference is unsupported); the caller
    (``discovery._list_uc_resources``) builds such a context-scoped executor.
    ``SHOW USER FUNCTIONS`` returns 3-part FQNs (col ``function``) and needs only
    BROWSE. Returns ``{name, full_name}`` — ``full_name`` is the 3-part FQN the
    picker stores in ``config.function``.
    """
    catalog = catalog.strip().lower()
    schema = schema.strip().lower()
    if not (_valid_ident(catalog) and _valid_ident(schema)):
        raise ValueError(f"unsupported identifier: {catalog!r}.{schema!r}")
    rows = sql_executor("SHOW USER FUNCTIONS", [], user_token)
    prefix = name_prefix.lower()
    qualifier = f"{catalog}.{schema}."
    out: list[dict[str, str]] = []
    for row in rows:
        raw = str(_first_col(row) or "").strip().lower()
        if not raw:
            continue
        # SHOW USER FUNCTIONS yields fully-qualified names in the bound context;
        # keep only this schema's, and take the bare function name.
        if raw.startswith(qualifier):
            name = raw[len(qualifier):]
        elif "." not in raw:
            name = raw
        else:
            continue
        if not _valid_ident(name):
            continue
        if prefix and not name.startswith(prefix):
            continue
        out.append({"name": name, "full_name": f"{catalog}.{schema}.{name}"})
    out.sort(key=lambda d: d["name"])
    return out


# ---------------------------------------------------------------------------
# Signature — DESCRIBE FUNCTION EXTENDED (BROWSE-sufficient)
# ---------------------------------------------------------------------------

# A ``DESCRIBE FUNCTION EXTENDED`` output row that starts a labelled section,
# e.g. ``Function:``, ``Type:``, ``Input:``, ``Returns:``, ``Comment:``.
_DESCRIBE_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z ]*):\s*(.*)$")


def _parse_input_param(raw: str) -> dict[str, Any] | None:
    """Parse one DESCRIBE ``Input`` line.

    Format: ``<name> <TYPE> [COLLATE ..] [DEFAULT <expr>] ['comment']``. Returns
    a param dict (with a private ``_complex`` flag for array/map/struct) or
    ``None`` for a no-arg marker / unparseable line.
    """
    text = raw.strip()
    if not text or text in ("()", "(none)", "N/A"):
        return None
    tokens = text.split()
    if len(tokens) < 2:
        return None
    name = tokens[0]
    if not _valid_ident(name):
        return None
    type_token = tokens[1]
    # Strip parameterization (DECIMAL(10,2), ARRAY<STRING>) for the base type.
    type_base = re.split(r"[(<]", type_token, maxsplit=1)[0].upper()
    complex_ = is_complex_type(type_base, type_token.upper())
    has_default = " DEFAULT " in f" {text.upper()} "
    return {
        "name": name,
        "type": uc_type_to_param_type(type_base),
        "required": not has_default,
        "_complex": complex_,
    }


def _parse_describe_signature(fqn: str, lines: list[str]) -> dict[str, Any]:
    """Parse ``DESCRIBE FUNCTION EXTENDED`` output lines into a signature dict.

    Reads the ``Type:`` (SCALAR/TABLE) and ``Input:`` sections. The Input
    section spans the ``Input:`` line plus following *unlabelled* continuation
    lines until the next ``Label:`` row — each is one parameter.
    """
    returns_table = False
    input_lines: list[str] = []
    in_input = False
    for line in lines:
        match = _DESCRIBE_LABEL_RE.match(line)
        if match:
            label = match.group(1).strip().lower()
            rest = match.group(2).strip()
            if label == "type":
                returns_table = rest.upper() == "TABLE"
                in_input = False
            elif label == "input":
                in_input = True
                if rest:
                    input_lines.append(rest)
            else:
                in_input = False
        elif in_input and line.strip():
            input_lines.append(line.strip())

    params: list[dict[str, Any]] = []
    non_scalar = False
    for raw in input_lines:
        parsed = _parse_input_param(raw)
        if parsed is None:
            continue
        if parsed.pop("_complex"):
            non_scalar = True
            continue
        params.append(parsed)
    scalar = not non_scalar
    return {
        "function": fqn,
        "params": params if scalar else [],
        "scalar": scalar,
        "returns_table": returns_table,
    }


def get_signature(
    sql_executor: SqlExecutor, fqn: str, *, user_token: str = ""
) -> dict[str, Any]:
    """Introspect a function's signature via ``DESCRIBE FUNCTION EXTENDED``.

    Works under BROWSE (unlike ``information_schema.parameters``, which needs
    USE CATALOG). Returns ``{"function", "params", "scalar", "returns_table"}``.
    ``scalar`` means the params are simple scalars (auto-mappable); a non-scalar
    (array/map/struct) param yields ``scalar=False`` + empty ``params``.
    ``returns_table`` is ``True`` for a table-valued function (``Type: TABLE``),
    which the runtime invokes ``SELECT * FROM fn(args)``. A no-arg / unknown
    function yields ``params=[]``. Raises ``ValueError`` on a malformed FQN;
    query errors propagate to the caller (which is responsible for fail-soft).
    """
    if not FQN_RE.fullmatch(fqn):
        raise ValueError(
            f"function {fqn!r} must be 'catalog.schema.function' of [A-Za-z0-9_] parts"
        )
    catalog, schema, fn = (part.lower() for part in fqn.split("."))
    quoted = f"`{catalog}`.`{schema}`.`{fn}`"
    rows = sql_executor(f"DESCRIBE FUNCTION EXTENDED {quoted}", [], user_token)
    lines = [str(_first_col(row) or "") for row in rows]
    return _parse_describe_signature(f"{catalog}.{schema}.{fn}", lines)


# ---------------------------------------------------------------------------
# Run-readiness: USE CATALOG litmus (browse != run)
# ---------------------------------------------------------------------------


def check_use_catalog(
    sql_executor: SqlExecutor, catalog: str, *, user_token: str = ""
) -> bool:
    """Whether the caller has ``USE CATALOG`` on ``catalog``.

    Listing/introspecting a function needs only BROWSE, but *invoking* it needs
    USE CATALOG + USE SCHEMA + EXECUTE. This runs a trivial ``information_schema``
    read (which requires USE CATALOG) as the litmus: ``True`` on success,
    ``False`` on an insufficient-privileges error. Any other error propagates.
    """
    catalog = catalog.strip().lower()
    if not _valid_ident(catalog):
        return False
    sql = f"SELECT 1 FROM `{catalog}`.information_schema.schemata LIMIT 1"
    try:
        sql_executor(sql, [], user_token)
    except Exception as exc:  # noqa: BLE001 - classify permission vs re-raise
        if _is_permission_error(str(exc)):
            return False
        raise
    return True
