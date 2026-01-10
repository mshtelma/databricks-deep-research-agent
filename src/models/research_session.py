"""ResearchSession SQLAlchemy model."""

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import BaseModel

if TYPE_CHECKING:
    from src.models.message import Message
    from src.models.research_event import ResearchEvent
    from src.models.source import Source


class ResearchDepth(str, Enum):
    """Research depth levels."""

    AUTO = "auto"
    LIGHT = "light"  # 1-2 search iterations
    MEDIUM = "medium"  # 3-5 search iterations
    EXTENDED = "extended"  # 6-10 search iterations


class ResearchStatus(str, Enum):
    """Research session status.

    Lifecycle states:
    - IN_PROGRESS: Research is actively running (set at START for crash resilience)
    - COMPLETED: Research finished successfully
    - CANCELLED: Research was cancelled by user
    - FAILED: Research failed due to error

    Phase states (legacy, for more granular tracking):
    - PENDING, CLASSIFYING, CLARIFYING, PLANNING, RESEARCHING, REFLECTING, SYNTHESIZING
    """

    # Lifecycle states (used for crash resilience)
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    # Phase states (legacy, for granular tracking)
    PENDING = "pending"
    CLASSIFYING = "classifying"
    CLARIFYING = "clarifying"
    PLANNING = "planning"
    RESEARCHING = "researching"
    REFLECTING = "reflecting"
    SYNTHESIZING = "synthesizing"


class ResearchSessionStatus(str, Enum):
    """Research session status (for service layer)."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ResearchSession(BaseModel):
    """Research session model.

    Represents the execution context for a research query.
    """

    __tablename__ = "research_sessions"

    # Foreign key to message (agent response)
    message_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Research query
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Observations from research steps
    observations: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Query classification (stored as JSONB)
    query_classification: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Research configuration
    research_depth: Mapped[ResearchDepth] = mapped_column(
        String(20),
        default=ResearchDepth.AUTO,
        nullable=False,
    )

    # Reasoning steps (stored as JSONB array)
    reasoning_steps: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    # Research plan (stored as JSONB)
    plan: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Current execution state
    current_step_index: Mapped[int | None] = mapped_column(
        nullable=True,
    )
    plan_iterations: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
    )

    # Status tracking
    status: Mapped[ResearchStatus] = mapped_column(
        String(20),
        default=ResearchStatus.PENDING,
        nullable=False,
    )
    current_agent: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Query mode (simple, web_search, deep_research)
    query_mode: Mapped[str] = mapped_column(
        String(20),
        default="deep_research",
        nullable=False,
    )

    # Relationships
    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="research_session",
    )
    sources: Mapped[list["Source"]] = relationship(
        "Source",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    events: Mapped[list["ResearchEvent"]] = relationship(
        "ResearchEvent",
        back_populates="research_session",
        cascade="all, delete-orphan",
        order_by="ResearchEvent.timestamp",
    )

    @property
    def is_complete(self) -> bool:
        """Check if research is complete."""
        return self.status in (
            ResearchStatus.COMPLETED,
            ResearchStatus.CANCELLED,
            ResearchStatus.FAILED,
        )

    @property
    def is_running(self) -> bool:
        """Check if research is currently running."""
        return self.status not in (
            ResearchStatus.PENDING,
            ResearchStatus.COMPLETED,
            ResearchStatus.CANCELLED,
            ResearchStatus.FAILED,
        )

    def complete(self) -> None:
        """Mark research as completed."""
        self.status = ResearchStatus.COMPLETED
        self.completed_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Mark research as cancelled."""
        self.status = ResearchStatus.CANCELLED
        self.completed_at = datetime.now(UTC)

    def fail(self, error_message: str) -> None:
        """Mark research as failed."""
        self.status = ResearchStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(UTC)
