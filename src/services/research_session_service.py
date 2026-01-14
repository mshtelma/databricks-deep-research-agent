"""Research session service - manages research session lifecycle."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.research_session import ResearchSession, ResearchSessionStatus, ResearchStatus

logger = logging.getLogger(__name__)


class ResearchSessionService:
    """Service for managing research sessions."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize research session service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create(
        self,
        message_id: UUID,
        query: str,
        research_depth: str = "auto",
        query_classification: dict[str, Any] | None = None,
    ) -> ResearchSession:
        """Create a new research session.

        Args:
            message_id: Associated message ID.
            query: Research query.
            research_depth: Depth level (auto, light, medium, extended).
            query_classification: Optional classification data.

        Returns:
            Created research session.
        """
        research_session = ResearchSession(
            message_id=message_id,
            query=query,
            research_depth=research_depth,
            query_classification=query_classification,
        )
        self._session.add(research_session)
        await self._session.flush()
        await self._session.refresh(research_session)
        logger.info(f"Created research session {research_session.id}")
        return research_session

    async def get(self, session_id: UUID) -> ResearchSession | None:
        """Get a research session by ID.

        Args:
            session_id: Research session ID.

        Returns:
            Research session if found, None otherwise.
        """
        result = await self._session.execute(
            select(ResearchSession).where(ResearchSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_by_message(self, message_id: UUID) -> ResearchSession | None:
        """Get research session by message ID.

        Args:
            message_id: Message ID.

        Returns:
            Research session if found, None otherwise.
        """
        result = await self._session.execute(
            select(ResearchSession).where(ResearchSession.message_id == message_id)
        )
        return result.scalar_one_or_none()

    async def get_active_session_by_chat(
        self,
        chat_id: UUID,
        user_id: str,
    ) -> ResearchSession | None:
        """Get active (in_progress) research session for a chat.

        Used to prevent duplicate research sessions when SSE reconnects.
        This is a critical guard against the duplicate research bug.

        Args:
            chat_id: Chat to check.
            user_id: User ID for security verification.

        Returns:
            Active ResearchSession if one exists, None otherwise.
        """
        stmt = (
            select(ResearchSession)
            .where(ResearchSession.chat_id == chat_id)
            .where(ResearchSession.user_id == user_id)
            .where(ResearchSession.status == ResearchStatus.IN_PROGRESS)
            .order_by(ResearchSession.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_plan(
        self,
        session_id: UUID,
        plan: dict[str, Any],
    ) -> ResearchSession | None:
        """Update research session with plan.

        Args:
            session_id: Research session ID.
            plan: Plan data.

        Returns:
            Updated session or None if not found.
        """
        session = await self.get(session_id)
        if not session:
            return None

        session.plan = plan
        await self._session.flush()
        await self._session.refresh(session)
        logger.info(f"Updated plan for session {session_id}")
        return session

    async def add_observation(
        self,
        session_id: UUID,
        observation: str,
        step_index: int,
    ) -> ResearchSession | None:
        """Add an observation to the session.

        Args:
            session_id: Research session ID.
            observation: Observation text.
            step_index: Step index.

        Returns:
            Updated session or None if not found.
        """
        session = await self.get(session_id)
        if not session:
            return None

        observations = session.observations or []
        observations.append({
            "step_index": step_index,
            "observation": observation,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        session.observations = observations
        await self._session.flush()
        await self._session.refresh(session)
        return session

    async def add_reasoning_step(
        self,
        session_id: UUID,
        step_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchSession | None:
        """Add a reasoning step to the session.

        Args:
            session_id: Research session ID.
            step_type: Type of step (search, reflect, etc.).
            content: Step content.
            metadata: Optional metadata.

        Returns:
            Updated session or None if not found.
        """
        session = await self.get(session_id)
        if not session:
            return None

        steps = session.reasoning_steps or []
        steps.append({
            "type": step_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat(),
        })
        session.reasoning_steps = steps
        await self._session.flush()
        await self._session.refresh(session)
        return session

    async def complete(
        self,
        session_id: UUID,
        error_message: str | None = None,
    ) -> ResearchSession | None:
        """Mark research session as complete.

        Args:
            session_id: Research session ID.
            error_message: Optional error message if failed.

        Returns:
            Updated session or None if not found.
        """
        session = await self.get(session_id)
        if not session:
            return None

        if error_message:
            session.status = ResearchSessionStatus.FAILED
            session.error_message = error_message
        else:
            session.status = ResearchSessionStatus.COMPLETED

        session.completed_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(session)
        logger.info(f"Completed research session {session_id}")
        return session

    async def cancel(self, session_id: UUID) -> ResearchSession | None:
        """Cancel a research session.

        Args:
            session_id: Research session ID.

        Returns:
            Updated session or None if not found.
        """
        session = await self.get(session_id)
        if not session:
            return None

        if session.status != ResearchSessionStatus.IN_PROGRESS:
            return session

        session.status = ResearchSessionStatus.CANCELLED
        session.completed_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(session)
        logger.info(f"Cancelled research session {session_id}")
        return session

    async def update_classification(
        self,
        session_id: UUID,
        classification: dict[str, Any],
    ) -> ResearchSession | None:
        """Update query classification.

        Args:
            session_id: Research session ID.
            classification: Classification data.

        Returns:
            Updated session or None if not found.
        """
        session = await self.get(session_id)
        if not session:
            return None

        session.query_classification = classification
        await self._session.flush()
        await self._session.refresh(session)
        return session
