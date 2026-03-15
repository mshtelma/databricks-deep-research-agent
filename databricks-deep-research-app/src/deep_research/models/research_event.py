"""ResearchEvent SQLAlchemy model for activity event persistence."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import Base

if TYPE_CHECKING:
    from deep_research.models.research_session import ResearchSession


class ResearchEvent(Base):
    """Research event model for activity event persistence.

    Stores individual SSE events during research for accordion display.
    Events are cascade deleted when the research session is deleted.

    Note: Inherits from Base (not BaseModel) because:
    - The table was created without updated_at column (migration 007)
    - This model explicitly defines its own id and created_at columns
    - No need for UUIDMixin or TimestampMixin
    """

    __tablename__ = "research_events"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Foreign key to research session
    research_session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Event type (e.g., "claim_verified", "tool_call", "step_started")
    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # When the event occurred
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # Event-specific payload (JSONB for flexibility)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # Sequence number for guaranteed ordering (timestamps may collide)
    sequence_number: Mapped[int | None] = mapped_column(
        nullable=True,
        comment="Monotonically increasing sequence number per session",
    )

    # Record creation time (may differ from event timestamp)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # Relationship back to research session
    research_session: Mapped["ResearchSession"] = relationship(
        "ResearchSession",
        back_populates="events",
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<ResearchEvent {self.event_type} @ {self.timestamp}>"
