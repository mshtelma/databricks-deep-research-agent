"""Runtime wiring for framework text-table tools.

The ``table_*`` tools are framework implementations, but they need runtime
dependencies: a binding registry, a schema cache, and a SQL executor.  This
module provides the shared Databricks Statement Execution wiring so host
applications do not duplicate that adapter code.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from databricks_deep_research.tools.factory import ToolFactoryContext

from .registry import TableBindingRegistry
from .schema_cache import Schema, SchemaCache, SchemaColumn

logger = logging.getLogger(__name__)


def _quote_fqn(fqn: str) -> str:
    parts = fqn.split(".")
    if len(parts) != 3 or any(not p for p in parts) or any("`" in p for p in parts):
        raise ValueError(f"invalid table FQN {fqn!r}; expected catalog.schema.table")
    return ".".join(f"`{p}`" for p in parts)


def _statement_state(response: Any) -> str:
    status = getattr(response, "status", None)
    state = getattr(status, "state", None)
    if state is None:
        return "UNKNOWN"
    return str(getattr(state, "value", None) or getattr(state, "name", None) or state)


def _statement_error(response: Any) -> str:
    status = getattr(response, "status", None)
    err = getattr(status, "error", None)
    code = getattr(err, "error_code", None)
    message = getattr(err, "message", None) or "statement execution failed"
    return f"{code or 'SQL_ERROR'}: {message}"


def _statement_rows(response: Any) -> list[dict[str, Any]]:
    columns: list[str] = []
    manifest = getattr(response, "manifest", None)
    schema = getattr(manifest, "schema", None) if manifest is not None else None
    for col in getattr(schema, "columns", None) or ():
        name = getattr(col, "name", None)
        if name is not None:
            columns.append(str(name))

    result = getattr(response, "result", None)
    rows: list[dict[str, Any]] = []
    for row in getattr(result, "data_array", None) or ():
        if columns:
            rows.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
        else:
            rows.append({str(i): value for i, value in enumerate(row)})
    return rows


class StatementExecutionTableSQL:
    """Sync adapter matching the framework text-table SQL executor protocol."""

    def __init__(
        self,
        *,
        workspace_client: Any,
        warehouse_id: str,
        timeout_sec: float = 30.0,
        poll_interval_sec: float = 1.0,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        self._workspace_client = workspace_client
        self._warehouse_id = warehouse_id
        self._timeout_sec = timeout_sec
        self._poll_interval_sec = poll_interval_sec
        # Optional session context. When set, statements run as if
        # ``USE CATALOG <catalog>; USE SCHEMA <schema>`` were issued first — the
        # only way to run ``SHOW USER FUNCTIONS`` cross-catalog (a 3-part ``IN``
        # reference is unsupported). Default ``None`` = no context (table tools
        # pass fully-qualified names and are unaffected).
        self._catalog = catalog
        self._schema = schema

    def __call__(
        self,
        sql: str,
        params: list[Any],
        user_token: str,
    ) -> list[dict[str, Any]]:
        # Auth is baked into the WorkspaceClient at construction. The host
        # resolves OBO-vs-SP per request (see build_databricks_workflow_runner
        # / resolve_workspace_client) and bakes the chosen client here, so the
        # per-call user_token is an intentional no-op for this executor.
        del user_token
        initial_wait = min(max(int(self._timeout_sec), 1), 50)
        response = self._workspace_client.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=self._warehouse_id,
            parameters=params or None,
            wait_timeout=f"{initial_wait}s",
            catalog=self._catalog,
            schema=self._schema,
        )
        started = time.monotonic()
        while True:
            state = _statement_state(response).upper()
            if state == "SUCCEEDED":
                return _statement_rows(response)
            if state in {"FAILED", "CANCELED", "CLOSED"}:
                raise RuntimeError(_statement_error(response))
            if time.monotonic() - started >= self._timeout_sec:
                statement_id = getattr(response, "statement_id", None)
                if statement_id:
                    try:
                        self._workspace_client.statement_execution.cancel_execution(
                            statement_id=statement_id
                        )
                    except Exception:  # noqa: BLE001 - best-effort cancel
                        logger.debug("TEXT_TABLE_SQL_CANCEL_FAILED", exc_info=True)
                raise TimeoutError(
                    f"text-table SQL statement timed out after {self._timeout_sec}s"
                )
            time.sleep(self._poll_interval_sec)
            statement_id = response.statement_id
            if not statement_id:
                raise RuntimeError("Statement Execution response missing statement_id")
            response = self._workspace_client.statement_execution.get_statement(
                statement_id
            )


def schema_from_describe_rows(fqn: str, rows: list[dict[str, Any]]) -> Schema:
    columns: list[SchemaColumn] = []
    for row in rows:
        name = row.get("col_name") or row.get("column_name") or row.get("name")
        data_type = row.get("data_type") or row.get("type")
        if not name or not data_type:
            continue
        name_s = str(name)
        if name_s.startswith("#"):
            continue
        columns.append(SchemaColumn(name=name_s, data_type=str(data_type)))
    if not columns:
        raise RuntimeError(f"DESCRIBE TABLE returned no columns for {fqn}")
    return Schema(fqn=fqn, columns=tuple(columns))


def _resolved_warehouse_id(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    value = os.environ.get("TABLE_TOOLS_WAREHOUSE_ID") or os.environ.get(
        "STORAGE_WAREHOUSE_ID"
    )
    return value.strip() if value else None


def wire_statement_execution_text_table_context(
    ctx: ToolFactoryContext,
    *,
    warehouse_id: str | None = None,
    table_discovery_provider: Any | None = None,
) -> ToolFactoryContext:
    """Populate ``ToolFactoryContext`` dependencies for text-table tools.

    A fresh ``TableBindingRegistry`` is installed for each wired context so
    bound/discovered table names never leak across workflow runs.  If a
    Databricks ``workspace_client`` and SQL warehouse ID are available, the
    helper also wires a Statement Execution-backed ``sql_executor`` plus
    ``SchemaCache``.
    """
    ctx.table_registry = TableBindingRegistry()
    ctx.table_discovery_provider = table_discovery_provider

    resolved_warehouse_id = _resolved_warehouse_id(warehouse_id)
    if ctx.workspace_client is not None and resolved_warehouse_id:
        sql_executor = StatementExecutionTableSQL(
            workspace_client=ctx.workspace_client,
            warehouse_id=resolved_warehouse_id,
        )
        ctx.sql_executor = sql_executor

        def _fetch_schema(fqn: str, token: str) -> Schema:
            rows = sql_executor(f"DESCRIBE TABLE {_quote_fqn(fqn)}", [], token)
            return schema_from_describe_rows(fqn, rows)

        ctx.schema_cache = SchemaCache(fetcher=_fetch_schema)
    elif ctx.workspace_client is not None and not resolved_warehouse_id:
        logger.warning(
            "TEXT_TABLE_WIRING_INCOMPLETE warehouse_id=MISSING "
            "reason=STORAGE_WAREHOUSE_ID and TABLE_TOOLS_WAREHOUSE_ID are unset. "
            "Workflows declaring table_search/table_read/table_neighbors/table_load/"
            "table_aggregate/uc_function will fail strict tool resolution."
        )

    return ctx
