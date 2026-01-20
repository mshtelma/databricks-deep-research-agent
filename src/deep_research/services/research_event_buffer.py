"""Research event buffer for batch-persisting SSE events to Lakebase.

This module provides buffered event persistence to reduce database round-trips
during research streaming. Events are batched and flushed periodically or
when the buffer is full.

Key Design:
- Uses independent DB sessions to survive request cancellation
- Thread-safe with asyncio.Lock
- Configurable buffer size (default 10 events)
- Best-effort persistence (failures logged but don't stop research)
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from deep_research.schemas.streaming import StreamEvent

logger = logging.getLogger(__name__)


class EventBuffer:
    """Buffer SSE events and batch-persist to Lakebase.

    Reduces DB round-trips by batching events. Flushes automatically
    when buffer is full or explicitly via flush(). Uses independent
    DB session to survive request cancellation.

    Usage:
        buffer = EventBuffer(research_session_id)
        async for event in orchestrate_research(...):
            yield event
            await buffer.add_event(event)
        await buffer.flush()  # Final flush

    Args:
        research_session_id: UUID of the research session (FK in research_events).
        buffer_size: Number of events to buffer before auto-flushing. Default 10.
    """

    def __init__(self, research_session_id: UUID, buffer_size: int = 10) -> None:
        """Initialize event buffer.

        Args:
            research_session_id: FK to research_sessions table.
            buffer_size: Auto-flush when buffer reaches this size.
        """
        self._session_id = research_session_id
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = buffer_size
        # Start sequence at 1 so that since_sequence=0 (default) returns all events
        # The query uses "sequence_number > since_sequence", so sequence 0 would be skipped
        self._sequence = 1
        self._lock = asyncio.Lock()
        self._total_flushed = 0

    async def add_event(self, event: StreamEvent) -> None:
        """Add event to buffer, flush if buffer is full.

        Args:
            event: StreamEvent to buffer for persistence.
        """
        async with self._lock:
            self._buffer.append({
                "event_type": event.event_type,
                # Store full payload for frontend replay (camelCase keys)
                # Use mode="json" to serialize datetime objects to ISO strings
                "payload": event.model_dump(by_alias=True, mode="json"),
                "timestamp": datetime.now(UTC),
                "sequence_number": self._sequence,
            })
            self._sequence += 1

            if len(self._buffer) >= self._buffer_size:
                await self._flush_unlocked()

    async def flush(self) -> None:
        """Flush all buffered events to database.

        Call this at the end of research to ensure all events are persisted.
        """
        async with self._lock:
            await self._flush_unlocked()

    async def _flush_unlocked(self) -> None:
        """Internal flush without lock (caller must hold lock).

        Uses independent DB session that survives request cancellation.
        Failures are logged but don't raise - best-effort persistence.
        """
        if not self._buffer:
            return

        try:
            # Import here to avoid circular imports
            from deep_research.db.session import get_session_maker
            from deep_research.services.research_event_service import ResearchEventService

            session_maker = get_session_maker()
            async with session_maker() as db:
                service = ResearchEventService(db)
                count = await service.save_events_batch(self._session_id, self._buffer)
                await db.commit()

            self._total_flushed += len(self._buffer)
            logger.debug(
                "EVENT_BUFFER_FLUSHED count=%d total=%d session=%s",
                len(self._buffer),
                self._total_flushed,
                str(self._session_id)[:8],
            )
        except Exception as e:
            # Best-effort: log warning but don't raise
            # Research continues even if event persistence fails
            logger.warning(
                "EVENT_BUFFER_FLUSH_FAILED count=%d session=%s error=%s",
                len(self._buffer),
                str(self._session_id)[:8],
                str(e)[:200],
            )
        finally:
            self._buffer.clear()

    @property
    def pending_count(self) -> int:
        """Number of events currently buffered (not yet flushed)."""
        return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        """Total number of events successfully flushed to DB."""
        return self._total_flushed

    @property
    def current_sequence(self) -> int:
        """Current sequence number (next event will get this number)."""
        return self._sequence
