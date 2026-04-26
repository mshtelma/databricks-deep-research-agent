"""Cache-backed `IResearchEventService` — writes via the `WriteQueue` event
buffer, reads via the backend's list/chunk APIs.

Write pattern: `save_event` / `save_events_batch` enqueue into the
`research_events` append buffer. Zero synchronous round-trip. Flushed as a
single multi-row INSERT on the next queue tick (bounded by
`storage_flush_interval_sec` and `storage_flush_size`).

Read pattern: `get_events_for_session` / `get_events_since_sequence` /
`get_event_count` hit the backend directly via `list_rows`. These are not
latency-critical (event timeline replay / reconnection) and aren't cached
— the data is append-only with high cardinality, so a cold cache would
churn.

Schema contract: the `research_events` table on BOTH backends has exactly
four columns — `(session_id, sequence_number, ts, event)`. See
`storage/lakebase_ddl.sql:109-115` and `storage/sql_warehouse_ddl.sql:100-105`.
`event_type`, `payload`, and a client-side `id` live inside the `event`
JSON document (jsonb on Lakebase, JSON-serialized string on Warehouse).

Returned shape: lightweight `SimpleNamespace` instances mirroring the legacy
`ResearchEvent` ORM attribute surface (`event_type`, `payload`, `timestamp`,
`sequence_number`, `research_session_id`, `id`). Callers relying on SQLAlchemy-
specific behavior must port to the namespace form.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IResearchEventService

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)


class CachedResearchEventService(_CachedServiceBase, IResearchEventService):
    """`IResearchEventService` implementation over `StorageStack`."""

    _EVENTS_TABLE = "research_events"

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)
        # Per-session monotonic counters so `sequence_number` defaults match
        # the legacy behavior when callers don't supply their own. Reset on
        # process restart; reconnection clients rely on the client-supplied
        # sequence_number in that case.
        self._seq_counters: dict[UUID, int] = {}

    # -- Writes ---------------------------------------------------------

    async def save_event(
        self,
        research_session_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> SimpleNamespace:
        ts = timestamp or datetime.now(UTC)
        seq = self._next_seq(research_session_id)
        event_id = uuid4()
        event_doc = {
            "id": str(event_id),
            "event_type": event_type,
            "payload": payload,
        }
        self._append_event(
            self._EVENTS_TABLE,
            {
                "session_id": research_session_id,
                "sequence_number": seq,
                "ts": ts,
                "event": event_doc,
            },
        )
        return _legacy_event(
            event_id=event_id,
            session_id=research_session_id,
            event_type=event_type,
            payload=payload,
            ts=ts,
            sequence_number=seq,
        )

    async def save_events_batch(
        self,
        research_session_id: UUID,
        events: "Sequence[dict[str, Any]]",
    ) -> int:
        count = 0
        for e in events:
            event_type = e.get("event_type", "unknown")
            payload = e.get("payload", {})
            ts = e.get("timestamp", datetime.now(UTC))
            seq = e.get("sequence_number")
            if seq is None:
                seq = self._next_seq(research_session_id)
            event_id = e.get("id") or uuid4()
            event_doc = {
                "id": str(event_id),
                "event_type": event_type,
                "payload": payload,
            }
            self._append_event(
                self._EVENTS_TABLE,
                {
                    "session_id": research_session_id,
                    "sequence_number": seq,
                    "ts": ts,
                    "event": event_doc,
                },
            )
            count += 1
        return count

    # -- Reads ----------------------------------------------------------

    async def get_events_for_session(
        self,
        research_session_id: UUID,
        *,
        event_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SimpleNamespace]:
        # Pull all rows for the session, then filter by event_type in-process
        # because `event_type` is embedded in the `event` JSON payload — the
        # backend `list_rows` surface supports only equality on top-level
        # columns. Event timelines are bounded (few hundred to a few
        # thousand), so the scan is acceptable.
        rows = await self._stack.backend.list_rows(
            self._EVENTS_TABLE,
            where={"session_id": research_session_id},
            order_by="sequence_number",
            limit=None if event_types else limit,
        )
        events = [
            _row_to_event(r, fallback_session_id=research_session_id) for r in rows
        ]
        if event_types:
            allowed = set(event_types)
            events = [e for e in events if e.event_type in allowed]
            if limit is not None:
                events = events[:limit]
        return events

    async def get_event_count(self, research_session_id: UUID) -> int:
        rows = await self._stack.backend.list_rows(
            self._EVENTS_TABLE,
            where={"session_id": research_session_id},
        )
        return len(rows)

    async def get_events_since_sequence(
        self,
        research_session_id: UUID,
        since_sequence: int,
        limit: int = 100,
    ) -> list[SimpleNamespace]:
        # backend.list_rows only supports equality predicates; do the filter
        # in Python. This method is used for reconnection (rare), so the
        # extra row-scan is acceptable.
        rows = await self._stack.backend.list_rows(
            self._EVENTS_TABLE,
            where={"session_id": research_session_id},
            order_by="sequence_number",
        )
        events = [
            _row_to_event(r, fallback_session_id=research_session_id) for r in rows
        ]
        filtered = [
            e for e in events
            if e.sequence_number is not None
            and int(e.sequence_number) > since_sequence
        ]
        return filtered[:limit]

    # -- Serialization helpers (match the legacy API surface) ----------

    @staticmethod
    def event_to_dict(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            return dict(event)
        # SimpleNamespace / ORM instance — pull the known fields and coerce
        # UUID/datetime to JSON-serializable strings. Shape matches the legacy
        # `ResearchEventService.event_to_dict` contract (camelCase keys) so
        # the SSE payload at `api/v1/jobs.py` and frontend TS union types line
        # up regardless of which storage impl is active.
        event_id = getattr(event, "id", None)
        timestamp = getattr(event, "timestamp", None)
        return {
            "id": str(event_id) if event_id is not None else None,
            "eventType": getattr(event, "event_type", None),
            "timestamp": timestamp.isoformat() if timestamp is not None else None,
            "sequenceNumber": getattr(event, "sequence_number", None),
            "payload": getattr(event, "payload", None),
        }

    @staticmethod
    def events_to_list(events: list[Any]) -> list[dict[str, Any]]:
        return [CachedResearchEventService.event_to_dict(e) for e in events]

    # -- Internal -------------------------------------------------------

    def _next_seq(self, session_id: UUID) -> int:
        current = self._seq_counters.get(session_id, 0)
        next_value = current + 1
        self._seq_counters[session_id] = next_value
        return next_value


# -- Row <-> legacy-shaped namespace --------------------------------------


def _legacy_event(
    *,
    event_id: UUID | str | None,
    session_id: UUID,
    event_type: str,
    payload: dict[str, Any],
    ts: datetime,
    sequence_number: int,
) -> SimpleNamespace:
    """Build a SimpleNamespace mirroring the legacy `ResearchEvent` ORM."""
    return SimpleNamespace(
        id=event_id,
        research_session_id=session_id,
        session_id=session_id,
        event_type=event_type,
        payload=payload,
        timestamp=ts,
        ts=ts,
        sequence_number=sequence_number,
    )


def _row_to_event(
    row: dict[str, Any],
    *,
    fallback_session_id: UUID | None = None,
) -> SimpleNamespace:
    """Decode a backend row (4 columns) into a legacy-shaped namespace.

    Tolerates both Lakebase (JSONB → dict) and Warehouse (STRING → str) event
    representations, plus the pre-fix on-disk rows if any slipped through.
    """
    event_doc = row.get("event")
    if isinstance(event_doc, (bytes, bytearray)):
        try:
            event_doc = event_doc.decode("utf-8")
        except UnicodeDecodeError:
            event_doc = None
    if isinstance(event_doc, str):
        try:
            event_doc = json.loads(event_doc)
        except (json.JSONDecodeError, ValueError):
            event_doc = None
    if not isinstance(event_doc, dict):
        # Legacy row (pre-fix) may have set the columns directly — fall back.
        event_doc = {
            "id": row.get("id"),
            "event_type": row.get("event_type", "unknown"),
            "payload": row.get("payload", {}),
        }

    session_id = row.get("session_id") or fallback_session_id
    if isinstance(session_id, str):
        try:
            session_id = UUID(session_id)
        except (ValueError, TypeError):
            pass

    event_id_raw = event_doc.get("id")
    event_id: UUID | str | None = None
    if isinstance(event_id_raw, UUID):
        event_id = event_id_raw
    elif isinstance(event_id_raw, str):
        try:
            event_id = UUID(event_id_raw)
        except ValueError:
            event_id = event_id_raw

    ts = row.get("ts") or row.get("timestamp")

    return SimpleNamespace(
        id=event_id,
        research_session_id=session_id,
        session_id=session_id,
        event_type=event_doc.get("event_type", "unknown"),
        payload=event_doc.get("payload") or {},
        timestamp=ts,
        ts=ts,
        sequence_number=row.get("sequence_number"),
    )
