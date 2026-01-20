"""UserPreferences SQLAlchemy model."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import Base, UUIDMixin
from deep_research.models.enums import QueryMode
from deep_research.models.research_session import ResearchDepth


class UserPreferences(Base, UUIDMixin):
    """User preferences model.

    Stores persistent user settings. One record per user.
    """

    __tablename__ = "user_preferences"

    # User ID (Databricks workspace user ID) - unique but not primary key
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )

    # Default research depth (matches migration column name)
    default_research_depth: Mapped[ResearchDepth] = mapped_column(
        String(20),
        default=ResearchDepth.AUTO,
        server_default="auto",
        nullable=False,
    )

    # System instructions for all chats
    system_instructions: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Default query mode (simple, web_search, deep_research)
    default_query_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="simple",
        default="simple",
    )

    # UI preferences (separate columns matching migration)
    theme: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="system",
    )
    notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="true",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def update_instructions(self, instructions: str | None) -> None:
        """Update system instructions."""
        if instructions and len(instructions) > 10000:
            raise ValueError("System instructions must be at most 10,000 characters")
        self.system_instructions = instructions

    def update_depth(self, depth: ResearchDepth) -> None:
        """Update default research depth."""
        self.default_research_depth = depth

    def update_theme(self, theme: str) -> None:
        """Update UI theme."""
        self.theme = theme

    def update_notifications(self, enabled: bool) -> None:
        """Update notifications setting."""
        self.notifications_enabled = enabled

    def update_query_mode(self, mode: QueryMode) -> None:
        """Update default query mode."""
        self.default_query_mode = mode.value
