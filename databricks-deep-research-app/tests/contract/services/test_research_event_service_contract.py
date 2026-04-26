"""Contract tests for `CachedResearchEventService` (F-RE).

Exercises the 4-column schema contract across every enabled backend
(`FakeBackend`, optionally `LakebaseBackend` and `SQLWarehouseBackend`).
Pre-fix the cached impl wrote 9 keys into a 4-column DDL; these tests pin
the fix and detect any regression that re-adds stale column names.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from deep_research.services.cached.research_event import (
    CachedResearchEventService,
    _row_to_event,
)


ALLOWED_COLS: frozenset[str] = frozenset({
    "session_id", "sequence_number", "ts", "event",
})
FORBIDDEN_COLS: frozenset[str] = frozenset({
    "id", "research_session_id", "event_type", "payload", "timestamp",
})


async def _flush_queue(stack) -> None:
    """Force a queue tick so appends actually persist."""
    # The queue's flush_interval_sec is 0.05 in the conftest fixture — one
    # short sleep is enough; a direct flush helper isn't exposed.
    await asyncio.sleep(0.15)


class TestCachedResearchEventServiceContract:
    """Schema-drift regression + attribute-shape compatibility tests."""

    @pytest.mark.asyncio
    async def test_save_event_writes_only_allowed_columns(self, stack) -> None:
        """After the write flushes, backend.list_rows must surface only the
        four DDL columns — any stale `id`/`event_type`/etc. means the
        schema-drift regression is back.
        """
        svc = CachedResearchEventService(stack)
        session_id = uuid4()

        await svc.save_event(
            research_session_id=session_id,
            event_type="tool_call",
            payload={"tool": "web_fetch", "url": "https://x"},
        )
        await _flush_queue(stack)

        rows = await stack.backend.list_rows(
            "research_events",
            where={"session_id": session_id},
        )
        assert len(rows) == 1, rows
        row_keys = set(rows[0].keys())
        assert row_keys.issubset(ALLOWED_COLS | FORBIDDEN_COLS), row_keys
        # Strict: any column the caller wrote MUST be in the allowed set.
        # The backend may still surface extras if the DDL has them (it
        # doesn't), but none of the forbidden-at-write columns may carry
        # non-null values.
        for forbidden in FORBIDDEN_COLS:
            if forbidden in rows[0]:
                assert rows[0][forbidden] in (None, ""), (
                    f"column {forbidden!r} leaked into persisted row: "
                    f"{rows[0][forbidden]!r}"
                )

    @pytest.mark.asyncio
    async def test_event_doc_carries_nested_fields(self, stack) -> None:
        """`event_type`, `payload`, and a client-side `id` must round-trip
        through the `event` JSON column, not separate columns.
        """
        svc = CachedResearchEventService(stack)
        session_id = uuid4()

        await svc.save_event(
            research_session_id=session_id,
            event_type="claim_verified",
            payload={"claim": "foo", "confidence": 0.9},
        )
        await _flush_queue(stack)

        events = await svc.get_events_for_session(session_id)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "claim_verified"
        assert ev.payload == {"claim": "foo", "confidence": 0.9}
        assert ev.sequence_number == 1
        assert ev.session_id == session_id
        assert ev.research_session_id == session_id
        assert ev.timestamp is not None

    @pytest.mark.asyncio
    async def test_batch_preserves_client_sequence_numbers(self, stack) -> None:
        svc = CachedResearchEventService(stack)
        session_id = uuid4()
        events = [
            {
                "event_type": "progress",
                "payload": {"step": i},
                "timestamp": datetime.now(UTC),
                "sequence_number": 10 + i,
            }
            for i in range(5)
        ]
        count = await svc.save_events_batch(session_id, events)
        assert count == 5
        await _flush_queue(stack)

        read_back = await svc.get_events_for_session(session_id)
        assert len(read_back) == 5
        seqs = [e.sequence_number for e in read_back]
        assert seqs == [10, 11, 12, 13, 14]
        assert [e.payload["step"] for e in read_back] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, stack) -> None:
        svc = CachedResearchEventService(stack)
        session_id = uuid4()
        for t, p in [
            ("tool_call", {}),
            ("observation", {}),
            ("tool_call", {}),
            ("synthesis", {}),
        ]:
            await svc.save_event(
                research_session_id=session_id, event_type=t, payload=p,
            )
        await _flush_queue(stack)

        tool_calls = await svc.get_events_for_session(
            session_id, event_types=["tool_call"],
        )
        assert len(tool_calls) == 2
        assert all(e.event_type == "tool_call" for e in tool_calls)

    @pytest.mark.asyncio
    async def test_get_events_since_sequence(self, stack) -> None:
        svc = CachedResearchEventService(stack)
        session_id = uuid4()
        for i in range(10):
            await svc.save_event(
                research_session_id=session_id,
                event_type="progress",
                payload={"i": i},
            )
        await _flush_queue(stack)

        tail = await svc.get_events_since_sequence(
            session_id, since_sequence=5, limit=100,
        )
        assert [e.sequence_number for e in tail] == [6, 7, 8, 9, 10]

    @pytest.mark.asyncio
    async def test_event_count(self, stack) -> None:
        svc = CachedResearchEventService(stack)
        session_id = uuid4()
        for _ in range(7):
            await svc.save_event(
                research_session_id=session_id,
                event_type="progress",
                payload={},
            )
        await _flush_queue(stack)

        assert await svc.get_event_count(session_id) == 7


class TestRowToEventAdapter:
    """Pure unit tests for `_row_to_event` — no backend required."""

    def test_decodes_event_json_string_warehouse_shape(self) -> None:
        """SQL Warehouse stores `event` as a JSON string. Adapter must
        parse it transparently.
        """
        session_id = uuid4()
        ts = datetime.now(UTC)
        row = {
            "session_id": str(session_id),
            "sequence_number": 42,
            "ts": ts,
            "event": (
                '{"id": "11111111-2222-3333-4444-555555555555", '
                '"event_type": "progress", "payload": {"k": 1}}'
            ),
        }
        e = _row_to_event(row, fallback_session_id=session_id)
        assert e.event_type == "progress"
        assert e.payload == {"k": 1}
        assert e.sequence_number == 42
        assert e.session_id == session_id

    def test_decodes_event_dict_lakebase_shape(self) -> None:
        """Lakebase stores `event` as JSONB → dict. Adapter handles directly."""
        session_id = uuid4()
        row = {
            "session_id": session_id,
            "sequence_number": 7,
            "ts": datetime.now(UTC),
            "event": {
                "id": str(uuid4()),
                "event_type": "tool_call",
                "payload": {"tool": "search"},
            },
        }
        e = _row_to_event(row)
        assert e.event_type == "tool_call"
        assert e.payload == {"tool": "search"}
        assert e.sequence_number == 7

    def test_tolerates_legacy_flat_row(self) -> None:
        """A pre-F-RE row on disk (flat columns) still decodes."""
        session_id = uuid4()
        row = {
            "session_id": session_id,
            "sequence_number": 1,
            "ts": datetime.now(UTC),
            "event": None,
            # Legacy flat fields:
            "event_type": "observation",
            "payload": {"note": "legacy"},
        }
        e = _row_to_event(row)
        assert e.event_type == "observation"
        assert e.payload == {"note": "legacy"}

    def test_unknown_event_type_defaults_gracefully(self) -> None:
        """Malformed event JSON → fallback event_type 'unknown', payload {}."""
        session_id = uuid4()
        row = {
            "session_id": session_id,
            "sequence_number": 1,
            "ts": datetime.now(UTC),
            "event": "not-json-at-all",
        }
        e = _row_to_event(row)
        assert e.event_type == "unknown"
        assert e.payload == {}
