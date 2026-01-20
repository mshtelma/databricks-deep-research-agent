"""Feedback service for managing user feedback on agent responses."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import mlflow
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.logging_utils import get_logger
from deep_research.models.message import Message
from deep_research.models.message_feedback import FeedbackRating, MessageFeedback

logger = get_logger(__name__)


class FeedbackService:
    """Service for managing feedback on agent messages."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize feedback service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create_feedback(
        self,
        message_id: UUID,
        user_id: str,
        rating: str,
        feedback_text: str | None = None,
        feedback_category: str | None = None,
    ) -> MessageFeedback:
        """Create feedback for a message.

        Args:
            message_id: Message ID to provide feedback for.
            user_id: User providing feedback.
            rating: Feedback rating (positive/negative).
            feedback_text: Optional detailed feedback text.
            feedback_category: Optional category (factual_error, incomplete, etc.).

        Returns:
            Created MessageFeedback record.

        Raises:
            ValueError: If message not found or already has feedback from user.
        """
        # Verify message exists
        result = await self._session.execute(
            select(Message).where(Message.id == message_id)
        )
        message = result.scalar_one_or_none()
        if not message:
            raise ValueError(f"Message {message_id} not found")

        # Check for existing feedback from this user
        existing = await self._session.execute(
            select(MessageFeedback).where(
                MessageFeedback.message_id == message_id,
                MessageFeedback.user_id == user_id,
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Feedback already exists for this message")

        # Parse rating
        try:
            rating_enum = FeedbackRating(rating)
        except ValueError as e:
            raise ValueError(f"Invalid rating: {rating}") from e

        # Create feedback
        feedback = MessageFeedback(
            message_id=message_id,
            user_id=user_id,
            rating=rating_enum,
            feedback_text=feedback_text,
            feedback_category=feedback_category,
        )

        self._session.add(feedback)
        await self._session.flush()
        await self._session.refresh(feedback)

        # Log to MLflow for trace correlation
        self._log_feedback_to_mlflow(message_id, rating, feedback_category)

        logger.info(
            "FEEDBACK_CREATED",
            message_id=str(message_id),
            rating=rating,
            category=feedback_category,
        )

        return feedback

    async def get_feedback(
        self,
        message_id: UUID,
        user_id: str,
    ) -> MessageFeedback | None:
        """Get feedback for a message from a user.

        Args:
            message_id: Message ID.
            user_id: User ID.

        Returns:
            MessageFeedback if found, None otherwise.
        """
        result = await self._session.execute(
            select(MessageFeedback).where(
                MessageFeedback.message_id == message_id,
                MessageFeedback.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def update_feedback(
        self,
        feedback_id: UUID,
        user_id: str,
        rating: str | None = None,
        feedback_text: str | None = None,
        feedback_category: str | None = None,
    ) -> MessageFeedback | None:
        """Update existing feedback.

        Args:
            feedback_id: Feedback ID.
            user_id: User ID (for ownership verification).
            rating: New rating (optional).
            feedback_text: New feedback text (optional).
            feedback_category: New category (optional).

        Returns:
            Updated feedback or None if not found.
        """
        result = await self._session.execute(
            select(MessageFeedback).where(
                MessageFeedback.id == feedback_id,
                MessageFeedback.user_id == user_id,
            )
        )
        feedback = result.scalar_one_or_none()
        if not feedback:
            return None

        if rating:
            feedback.rating = FeedbackRating(rating)
        if feedback_text is not None:
            feedback.feedback_text = feedback_text
        if feedback_category is not None:
            feedback.feedback_category = feedback_category

        # Note: MessageFeedback doesn't have updated_at, rely on flush to persist changes
        await self._session.flush()
        await self._session.refresh(feedback)

        logger.info(
            "FEEDBACK_UPDATED",
            feedback_id=str(feedback_id),
        )

        return feedback

    async def delete_feedback(
        self,
        feedback_id: UUID,
        user_id: str,
    ) -> bool:
        """Delete feedback.

        Args:
            feedback_id: Feedback ID.
            user_id: User ID (for ownership verification).

        Returns:
            True if deleted, False if not found.
        """
        result = await self._session.execute(
            select(MessageFeedback).where(
                MessageFeedback.id == feedback_id,
                MessageFeedback.user_id == user_id,
            )
        )
        feedback = result.scalar_one_or_none()
        if not feedback:
            return False

        # Note: session.delete() is synchronous in SQLAlchemy, not async
        self._session.delete(feedback)
        await self._session.flush()

        logger.info(
            "FEEDBACK_DELETED",
            feedback_id=str(feedback_id),
        )

        return True

    async def get_message_feedback_stats(
        self,
        message_id: UUID,
    ) -> dict[str, Any]:
        """Get feedback statistics for a message.

        Args:
            message_id: Message ID.

        Returns:
            Dictionary with positive_count, negative_count, total.
        """
        from sqlalchemy import func

        result = await self._session.execute(
            select(
                func.count().filter(MessageFeedback.rating == FeedbackRating.POSITIVE),
                func.count().filter(MessageFeedback.rating == FeedbackRating.NEGATIVE),
            ).where(MessageFeedback.message_id == message_id)
        )
        row = result.one()
        positive, negative = row

        return {
            "positive_count": positive,
            "negative_count": negative,
            "total": positive + negative,
        }

    def _log_feedback_to_mlflow(
        self,
        message_id: UUID,
        rating: str,
        category: str | None,
    ) -> None:
        """Log feedback to MLflow for trace correlation.

        Args:
            message_id: Message ID.
            rating: Feedback rating.
            category: Feedback category.
        """
        try:
            with mlflow.start_span(name="user_feedback", span_type="UNKNOWN") as span:
                span.set_attributes({
                    "feedback.message_id": str(message_id),
                    "feedback.rating": rating,
                    "feedback.category": category or "none",
                    "feedback.timestamp": datetime.now(UTC).isoformat(),
                })
        except Exception as e:
            # Don't fail the feedback creation if MLflow logging fails
            logger.warning(
                "MLFLOW_FEEDBACK_LOG_FAILED",
                error=str(e),
            )
