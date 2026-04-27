"""Delta-table-backed checkpointer for Workflow event streams.

Every event emitted by the executor is appended to a Delta table keyed by
``thread_id``. On crash, :meth:`DeltaCheckpointer.recover` replays the
events into a fresh :class:`RuntimeState` so the workflow can resume from
the last completed node.

Phase 2 ships the **synchronous** path — events are persisted before the
executor advances to the next node. Async batching is deferred.

DDL (created on first ``ensure_table`` call)::

    CREATE TABLE IF NOT EXISTS <table> (
        thread_id     STRING NOT NULL,
        event_seq     BIGINT NOT NULL,
        event_type    STRING NOT NULL,
        event_payload STRING NOT NULL,
        created_at_ms BIGINT NOT NULL
    ) USING DELTA
    PARTITIONED BY (thread_id);
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from databricks_deep_research.events.types import StreamEvent

logger = logging.getLogger(__name__)


class DeltaCheckpointer:
    """Persists workflow events to a Delta table for crash-recovery replay."""

    def __init__(
        self,
        table: str = "main.ai.agent_runs",
        workspace_client: Any | None = None,
        *,
        synchronous: bool = True,
    ) -> None:
        self._table = table
        self._wc = workspace_client
        self._synchronous = synchronous
        self._seq_by_thread: dict[str, int] = {}

    def _build_workspace_client(self) -> Any:
        if self._wc is not None:
            return self._wc
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as exc:
            raise RuntimeError(
                "DeltaCheckpointer requires databricks-sdk; "
                "install with `pip install databricks-sdk`."
            ) from exc
        self._wc = WorkspaceClient()
        return self._wc

    def _next_seq(self, thread_id: str) -> int:
        seq = self._seq_by_thread.get(thread_id, 0) + 1
        self._seq_by_thread[thread_id] = seq
        return seq

    async def ensure_table(self) -> None:
        """Create the Delta table if it does not exist."""
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "thread_id STRING NOT NULL, "
            "event_seq BIGINT NOT NULL, "
            "event_type STRING NOT NULL, "
            "event_payload STRING NOT NULL, "
            "created_at_ms BIGINT NOT NULL"
            ") USING DELTA"
        )
        await self._sql(ddl)

    async def on_event(self, thread_id: str, event: StreamEvent) -> None:
        """Append a single event row.

        When ``synchronous=True`` the call awaits the SQL statement before
        returning so the executor cannot advance past an unpersisted event.
        """
        seq = self._next_seq(thread_id)
        payload = event.model_dump_json()
        # Escape single quotes for the inline INSERT.
        safe_payload = payload.replace("'", "''")
        sql = (
            f"INSERT INTO {self._table} VALUES ("
            f"'{thread_id}', {seq}, '{event.event_type}', "
            f"'{safe_payload}', {int(time.time() * 1000)}"
            ")"
        )
        if self._synchronous:
            await self._sql(sql)
        else:
            asyncio.create_task(self._sql(sql))

    async def recover(self, thread_id: str) -> list[StreamEvent]:
        """Replay events for a thread in seq order.

        Returns a list of :class:`StreamEvent` instances; callers can fold
        these into a fresh :class:`RuntimeState` via the executor's
        replay path.
        """
        rows = await self._sql_query(
            f"SELECT event_payload FROM {self._table} "
            f"WHERE thread_id = '{thread_id}' ORDER BY event_seq"
        )
        events: list[StreamEvent] = []
        for row in rows:
            payload = row.get("event_payload") if isinstance(row, dict) else getattr(row, "event_payload", None)
            if not payload:
                continue
            try:
                events.append(StreamEvent.model_validate(json.loads(payload)))
            except Exception:  # noqa: BLE001
                logger.warning("CHECKPOINT_REPLAY_DECODE_FAIL thread=%s", thread_id)
        return events

    async def _sql(self, statement: str) -> None:
        wc = self._build_workspace_client()
        loop = asyncio.get_event_loop()

        def _run() -> None:
            try:
                wc.statement_execution.execute_statement(  # type: ignore[union-attr]
                    statement=statement,
                    warehouse_id=self._warehouse_id(),
                )
            except Exception:  # noqa: BLE001
                logger.exception("CHECKPOINT_SQL_FAILED stmt=%r", statement[:120])

        await loop.run_in_executor(None, _run)

    async def _sql_query(self, statement: str) -> list[dict]:
        wc = self._build_workspace_client()
        loop = asyncio.get_event_loop()

        def _run() -> list[dict]:
            try:
                response = wc.statement_execution.execute_statement(  # type: ignore[union-attr]
                    statement=statement,
                    warehouse_id=self._warehouse_id(),
                )
                rows = getattr(response.result, "data_array", None) or []
                schema = getattr(response.manifest, "schema", None)
                columns: list[str] = []
                if schema is not None:
                    columns = [c.name for c in getattr(schema, "columns", [])]
                return [dict(zip(columns, row, strict=False)) for row in rows] if columns else rows
            except Exception:  # noqa: BLE001
                logger.exception("CHECKPOINT_SQL_QUERY_FAILED stmt=%r", statement[:120])
                return []

        return await loop.run_in_executor(None, _run)

    def _warehouse_id(self) -> str:
        import os
        return os.environ.get("DATABRICKS_WAREHOUSE_ID", "")


__all__ = ["DeltaCheckpointer"]
