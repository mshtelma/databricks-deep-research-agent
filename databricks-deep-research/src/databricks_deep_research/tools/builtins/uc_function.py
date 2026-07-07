"""Deterministic ``uc_function`` tool: invoke a Unity Catalog function via SQL.

A ``uc_function`` names an existing UC *scalar or table-valued* function
(``catalog.schema.fn``) and invokes it through the framework's OBO SQL executor
(:class:`~databricks_deep_research.tools.builtins.text_table.runtime_wiring.StatementExecutionTableSQL`),
so it runs under the *caller's* identity (on-behalf-of-user) with the ``sql``
scope — the same path the ``table_*`` tools use, and the one scope proven to
work under Databricks Apps OBO (managed-functions MCP 403s under OBO).

It is a plain :class:`ResearchTool`, so one declaration is callable by agents
mid-ReAct AND by deterministic ``tool`` nodes.

Injection posture: values are always bound via parameter markers (``:name``);
only the regex-validated FQN and ASCII-validated argument *names* are ever
interpolated into the SQL text.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from databricks_deep_research.tools.builtins.python_function import (
    compile_params_schema,
)
from databricks_deep_research.tools.builtins.text_table.tools._common import (
    get_user_token,
)
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

if TYPE_CHECKING:
    from databricks.sdk.service.sql import StatementParameterListItem

    from databricks_deep_research.tools.builtins.text_table.tools._common import (
        SqlExecutor,
    )

logger = logging.getLogger(__name__)

__all__ = ["UCFunctionTool"]

# Strict 3-part FQN: only [A-Za-z0-9_] per part. Forbids backticks, spaces,
# hyphens, and dots-within-parts, so backtick-quoting the parts cannot be
# escaped (defense in depth) and hyphenated catalogs are rejected (v1 limit).
_FQN_RE = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
# ASCII identifier for argument names. ``str.isidentifier()`` accepts unicode
# and is NOT safe for interpolation into SQL text.
_ARG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Declared param type -> Databricks SQL type for StatementParameterListItem.
# ``integer`` -> ``BIGINT`` (JSON ints can exceed INT's 2^31). Types absent
# from this map (incl. array/map/struct, which save-time introspection rejects)
# bind untyped so the engine implicit-casts — verified on AIS.
_SQL_TYPE_MAP: dict[str, str] = {
    "string": "STRING",
    "str": "STRING",
    "integer": "BIGINT",
    "int": "BIGINT",
    "number": "DOUBLE",
    "float": "DOUBLE",
    "boolean": "BOOLEAN",
    "bool": "BOOLEAN",
}

# Cap the rows rendered into a table-function result so a large result set never
# blows up the tool-call content / citation snippet.
_MAX_PREVIEW_ROWS = 50


def _format_rows(rows: list[dict[str, Any]]) -> str:
    """Render table-function rows as a compact markdown table (row-capped)."""
    if not rows:
        return "(no rows)"
    columns = list(rows[0].keys())
    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    for row in rows[:_MAX_PREVIEW_ROWS]:
        lines.append(" | ".join(str(row.get(col, "")) for col in columns))
    if len(rows) > _MAX_PREVIEW_ROWS:
        lines.append(f"… ({len(rows) - _MAX_PREVIEW_ROWS} more row(s))")
    return "\n".join(lines)


class UCFunctionTool:
    """Invoke a Unity Catalog function via the OBO SQL executor.

    Scalar functions are called ``SELECT fn(args) AS result`` (a single value);
    table-valued functions (``returns_table=True``) are called
    ``SELECT * FROM fn(args)`` and return all rows. ``returns_table`` is derived
    at authoring time from ``DESCRIBE FUNCTION`` (``Type: TABLE``); it defaults
    to ``False`` so an un-introspected declaration keeps the scalar behavior.
    """

    def __init__(
        self,
        *,
        name: str,
        function_name: str,
        sql_executor: SqlExecutor,
        params: Sequence[dict[str, Any]] = (),
        description: str = "",
        citeable: bool = True,
        returns_table: bool = False,
    ) -> None:
        if not _FQN_RE.match(function_name):
            raise ValueError(
                f"uc_function '{name}': function {function_name!r} must be a "
                "three-part 'catalog.schema.function' of [A-Za-z0-9_] parts "
                "(hyphenated catalog names are unsupported in v1)"
            )
        self._name = name
        self._function_name = function_name
        self._quoted_fqn = ".".join(f"`{p}`" for p in function_name.split("."))
        self._sql_executor = sql_executor
        self._params = [dict(p) for p in params]
        self._schema = compile_params_schema(self._params)
        self._description = description
        self._citeable = citeable
        self._returns_table = returns_table
        # Declared name -> SQL binding type (None = untyped, implicit-cast).
        self._sql_types: dict[str, str | None] = {
            str(p.get("name")): _SQL_TYPE_MAP.get(
                str(p.get("type", "string")).lower()
            )
            for p in self._params
        }

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description
            or f"Unity Catalog function '{self._function_name}'",
            parameters=self._schema,
            source_type="uc_function",
            source_kind="sql_analytics" if self._citeable else "builtin",
            metadata={"uc_function": self._function_name},
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments or {})
        if not self._params:
            # Introspection not run / failed: pass ALL provided args through
            # untyped (implicit-cast). Filtering to a declared set here would
            # drop every arg and emit ``SELECT fn()`` -> opaque arity error.
            return args
        missing: list[str] = []
        for param in self._params:
            name = str(param.get("name"))
            if name in args:
                continue
            if "default" in param:
                args[name] = param["default"]
            elif param.get("required"):
                missing.append(name)
        if missing:
            raise ValueError(
                f"uc_function '{self._name}' missing required argument(s): "
                f"{missing}"
            )
        declared = {str(p.get("name")) for p in self._params}
        return {k: v for k, v in args.items() if k in declared}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        try:
            sql, sdk_params = self._build_sql(arguments)
        except ValueError as exc:
            return ToolResult(
                content=f"{self._function_name} argument error: {exc}",
                success=False,
                error=str(exc),
            )
        user_token = get_user_token(context.extras)
        try:
            # Run the SYNC executor off the event loop. Never let an exception
            # escape — a bare TimeoutError from the executor's 30s cap would be
            # caught upstream and mislabeled RESEARCH_TIMEOUT.
            rows = await asyncio.to_thread(
                self._sql_executor, sql, sdk_params, user_token
            )
        except Exception as exc:  # noqa: BLE001 - fail-soft by contract
            logger.warning(
                "UC_FUNCTION_SQL_FAILED fn=%s error=%s",
                self._function_name,
                str(exc)[:200],
            )
            return ToolResult(
                content=f"{self._function_name} execution failed: {exc}",
                success=False,
                error=str(exc),
            )
        return self._to_tool_result(arguments, rows)

    # -- helpers --------------------------------------------------------------

    def _build_sql(
        self, arguments: dict[str, Any]
    ) -> tuple[str, list[StatementParameterListItem]]:
        from databricks.sdk.service.sql import StatementParameterListItem

        clauses: list[str] = []
        sdk_params: list[StatementParameterListItem] = []
        for name, value in arguments.items():
            if not _ARG_NAME_RE.match(name):
                raise ValueError(f"invalid argument name {name!r}")
            clauses.append(f"{name} => :{name}")
            sdk_params.append(
                StatementParameterListItem(
                    name=name,
                    value=str(value) if value is not None else None,
                    type=self._sql_types.get(name),
                )
            )
        args_sql = ", ".join(clauses)
        if self._returns_table:
            # Table-valued function: SELECT * FROM fn(args) yields all rows.
            sql = f"SELECT * FROM {self._quoted_fqn}({args_sql})"
        else:
            sql = f"SELECT {self._quoted_fqn}({args_sql}) AS result"
        return sql, sdk_params

    def _to_tool_result(
        self,
        arguments: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> ToolResult:
        arg_repr = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        if self._returns_table:
            value: Any = None
            content = f"{self._function_name}({arg_repr}) ->\n{_format_rows(rows)}"
        else:
            value = rows[0].get("result") if rows else None
            content = f"{self._function_name}({arg_repr}) -> {value}"
        sources: list[SourceInfo] = []
        if self._citeable:
            arg_hash = hashlib.sha256(
                repr(sorted(arguments.items())).encode("utf-8")
            ).hexdigest()[:12]
            sources.append(
                SourceInfo(
                    url=f"uc-function://{self._function_name}/{arg_hash}",
                    title=self._description or self._function_name,
                    snippet=content[:800],
                    content=content,
                    source_type="uc_function",
                    source_kind="sql_analytics",
                )
            )
        return ToolResult(
            content=content,
            success=True,
            sources=sources,
            data={"result": value, "rows": rows},
        )
