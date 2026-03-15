"""Research event service for activity event persistence."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.logging_utils import get_logger
from deep_research.models.research_event import ResearchEvent

logger = get_logger(__name__)


class ResearchEventService:
    """Service for managing research activity events.

    Handles persistence and retrieval of SSE events for accordion display
    and event history replay.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize research event service.

        Args:
            session: Database session.
        """
        self._session = session

    async def save_event(
        self,
        research_session_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> ResearchEvent:
        """Save a single research event.

        Args:
            research_session_id: ID of the research session.
            event_type: Type of event (e.g., "claim_verified", "tool_call").
            payload: Event-specific data.
            timestamp: When the event occurred. Defaults to now.

        Returns:
            Created ResearchEvent instance.
        """
        event = ResearchEvent(
            research_session_id=research_session_id,
            event_type=event_type,
            timestamp=timestamp or datetime.now(UTC),
            payload=payload,
        )
        self._session.add(event)
        await self._session.flush()
        await self._session.refresh(event)

        logger.debug(
            "EVENT_SAVED",
            event_type=event_type,
            session_id=str(research_session_id)[:8],
        )

        return event

    async def save_events_batch(
        self,
        research_session_id: UUID,
        events: list[dict[str, Any]],
    ) -> int:
        """Save multiple research events in a batch.

        Optimized for bulk inserts to reduce database round trips.
        Supports sequence_number for guaranteed event ordering.

        Args:
            research_session_id: ID of the research session.
            events: List of event dicts with 'event_type', 'payload',
                   optional 'timestamp', and optional 'sequence_number'.

        Returns:
            Number of events saved.
        """
        count = 0
        for event_dict in events:
            event = ResearchEvent(
                research_session_id=research_session_id,
                event_type=event_dict.get("event_type", "unknown"),
                timestamp=event_dict.get("timestamp", datetime.now(UTC)),
                payload=event_dict.get("payload", {}),
                sequence_number=event_dict.get("sequence_number"),
            )
            self._session.add(event)
            count += 1

        await self._session.flush()

        logger.info(
            "EVENTS_BATCH_SAVED",
            count=count,
            session_id=str(research_session_id)[:8],
        )

        return count

    async def get_events_for_session(
        self,
        research_session_id: UUID,
        event_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ResearchEvent]:
        """Get events for a research session.

        Args:
            research_session_id: ID of the research session.
            event_types: Optional list of event types to filter by.
            limit: Optional maximum number of events to return.

        Returns:
            List of ResearchEvent instances ordered by timestamp.
        """
        query = (
            select(ResearchEvent)
            .where(ResearchEvent.research_session_id == research_session_id)
            .order_by(ResearchEvent.timestamp)
        )

        if event_types:
            query = query.where(ResearchEvent.event_type.in_(event_types))

        if limit:
            query = query.limit(limit)

        result = await self._session.execute(query)
        events = list(result.scalars().all())

        logger.debug(
            "EVENTS_RETRIEVED",
            count=len(events),
            session_id=str(research_session_id)[:8],
        )

        return events

    async def get_event_count(self, research_session_id: UUID) -> int:
        """Get the count of events for a session.

        Args:
            research_session_id: ID of the research session.

        Returns:
            Number of events.
        """
        from sqlalchemy import func

        query = select(func.count()).select_from(ResearchEvent).where(
            ResearchEvent.research_session_id == research_session_id
        )
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def get_events_since_sequence(
        self,
        research_session_id: UUID,
        since_sequence: int,
        limit: int = 100,
    ) -> list[ResearchEvent]:
        """Get events with sequence_number greater than since_sequence.

        Used for reconnection: fetch all events the client missed.

        Args:
            research_session_id: ID of the research session.
            since_sequence: Return events with sequence_number > this value.
            limit: Maximum number of events to return.

        Returns:
            List of ResearchEvent instances ordered by sequence_number.
        """
        query = (
            select(ResearchEvent)
            .where(ResearchEvent.research_session_id == research_session_id)
            .where(ResearchEvent.sequence_number > since_sequence)
            .order_by(ResearchEvent.sequence_number)
            .limit(limit)
        )

        result = await self._session.execute(query)
        events = list(result.scalars().all())

        logger.debug(
            "EVENTS_SINCE_SEQUENCE_RETRIEVED",
            count=len(events),
            since_sequence=since_sequence,
            session_id=str(research_session_id)[:8],
        )

        return events

    def event_to_dict(self, event: ResearchEvent) -> dict[str, Any]:
        """Convert a ResearchEvent to dictionary for API responses.

        Args:
            event: ResearchEvent instance.

        Returns:
            Dictionary representation with camelCase keys for frontend.
        """
        return {
            "id": str(event.id),
            "eventType": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "sequenceNumber": event.sequence_number,
            "payload": event.payload,
        }

    def events_to_list(self, events: list[ResearchEvent]) -> list[dict[str, Any]]:
        """Convert multiple events to list of dictionaries.

        Args:
            events: List of ResearchEvent instances.

        Returns:
            List of dictionary representations.
        """
        return [self.event_to_dict(event) for event in events]
