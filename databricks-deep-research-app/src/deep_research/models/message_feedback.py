"""MessageFeedback SQLAlchemy model."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from deep_research.models.message import Message


class FeedbackRating(str, Enum):
    """Feedback rating values."""

    NEGATIVE = "negative"
    POSITIVE = "positive"


class MessageFeedback(Base, UUIDMixin):
    """Message feedback model.

    Stores user-provided feedback on agent responses.
    """

    __tablename__ = "message_feedback"

    # Foreign key to message
    message_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # User who provided feedback
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    # Rating: 'positive' or 'negative' (matches migration String(20))
    rating: Mapped[FeedbackRating] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    # Optional feedback text (matches migration column name)
    feedback_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Feedback category
    feedback_category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Trace ID for observability
    trace_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship
    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="feedback",
    )

    @property
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        return self.rating == FeedbackRating.POSITIVE

    @property
    def is_negative(self) -> bool:
        """Check if feedback is negative."""
        return self.rating == FeedbackRating.NEGATIVE
