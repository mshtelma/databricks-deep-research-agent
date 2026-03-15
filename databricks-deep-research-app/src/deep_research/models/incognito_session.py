"""Incognito session model for ephemeral chat storage."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import BaseModel

if TYPE_CHECKING:
    from deep_research.models.chat import Chat


# Session constants
SESSION_TTL_HOURS = 1
MAX_INCOGNITO_CHATS = 5


class IncognitoSession(BaseModel):
    """Server-side session for incognito chat lifecycle.

    Tracks browser sessions for incognito chats. Sessions have a 1-hour
    idle timeout and are automatically cleaned up when expired, which
    cascade-deletes all associated incognito chats.

    Attributes:
        user_id: The Databricks user ID owning this session.
        session_token: Unique token stored in httpOnly cookie.
        last_activity: Last interaction timestamp.
        expires_at: When the session expires (last_activity + 1 hour).
    """

    __tablename__ = "incognito_sessions"

    # User identification (index defined in __table_args__)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Session token for cookie identification (index defined in __table_args__)
    session_token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # Activity tracking
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Expiration timestamp (index defined in __table_args__)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Relationship to chats (cascade delete on session expiry)
    chats: Mapped[list["Chat"]] = relationship(
        "Chat",
        back_populates="incognito_session",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_incognito_sessions_token", "session_token"),
        Index("idx_incognito_sessions_expires", "expires_at"),
        Index("idx_incognito_sessions_user", "user_id"),
    )

    def touch(self) -> None:
        """Update last_activity and extend expiration by 1 hour."""
        now = datetime.now(UTC)
        self.last_activity = now
        self.expires_at = now + timedelta(hours=SESSION_TTL_HOURS)

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now(UTC) > self.expires_at

    @property
    def chat_count(self) -> int:
        """Get the number of active incognito chats in this session."""
        return len([c for c in self.chats if not c.is_deleted])

    @property
    def can_create_chat(self) -> bool:
        """Check if a new incognito chat can be created in this session."""
        return self.chat_count < MAX_INCOGNITO_CHATS
