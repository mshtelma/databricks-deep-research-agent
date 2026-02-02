"""Session service - manages incognito chat sessions."""

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import TypedDict
from uuid import UUID

from sqlalchemy import delete, func, select


class SessionStatusResponse(TypedDict):
    """Typed response for session status."""

    has_session: bool
    chat_count: int
    max_chats: int
    expires_at: str | None

from deep_research.models.chat import Chat, ChatType
from deep_research.models.incognito_session import (
    MAX_INCOGNITO_CHATS,
    SESSION_TTL_HOURS,
    IncognitoSession,
)
from deep_research.services.base import BaseRepository

logger = logging.getLogger(__name__)


class SessionService(BaseRepository[IncognitoSession]):
    """Service for managing incognito chat sessions.

    Handles:
    - Session creation and retrieval
    - Activity tracking and TTL extension
    - Chat quota enforcement (max 5 concurrent chats)
    - Expired session cleanup

    Sessions are linked to browser sessions via httpOnly cookies.
    When a session expires, all associated incognito chats are
    cascade-deleted.
    """

    model = IncognitoSession

    async def get_by_token(self, session_token: str) -> IncognitoSession | None:
        """Get session by token (internal use only).

        WARNING: This method does NOT verify user ownership.
        For endpoints, use get_by_token_for_user() instead.

        Args:
            session_token: The session token from the cookie.

        Returns:
            IncognitoSession if found and not expired, None otherwise.
        """
        result = await self._session.execute(
            select(IncognitoSession).where(
                IncognitoSession.session_token == session_token,
                IncognitoSession.expires_at > datetime.now(UTC),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_token_for_user(
        self, session_token: str, user_id: str
    ) -> IncognitoSession | None:
        """Get session by token with user ownership verification.

        This is the SECURE method that should be used in all endpoints.
        It prevents cross-user session token reuse attacks by validating
        that the session belongs to the authenticated user.

        Args:
            session_token: The session token from the cookie.
            user_id: The authenticated user's ID to verify ownership.

        Returns:
            IncognitoSession if found, not expired, AND owned by user_id.
            None if session not found, expired, or belongs to different user.
        """
        result = await self._session.execute(
            select(IncognitoSession).where(
                IncognitoSession.session_token == session_token,
                IncognitoSession.user_id == user_id,
                IncognitoSession.expires_at > datetime.now(UTC),
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create_session(
        self, user_id: str, session_token: str | None = None
    ) -> tuple[IncognitoSession, str, bool]:
        """Get existing session or create a new one.

        If a session_token is provided and valid, returns that session.
        Otherwise, creates a new session with a generated token.

        Args:
            user_id: The Databricks user ID.
            session_token: Optional existing session token from cookie.

        Returns:
            Tuple of (session, token, is_new) where is_new indicates
            if a new session was created.
        """
        # Try to find existing session with user ownership verification
        if session_token:
            existing = await self.get_by_token_for_user(session_token, user_id)
            if existing:
                # Touch to extend TTL
                existing.touch()
                await self.update(existing)
                return existing, session_token, False

        # Create new session
        new_token = secrets.token_urlsafe(32)
        now = datetime.now(UTC)

        session = IncognitoSession(
            user_id=user_id,
            session_token=new_token,
            last_activity=now,
            expires_at=now + timedelta(hours=SESSION_TTL_HOURS),
        )
        session = await self.add(session)
        logger.info(f"Created new incognito session for user {user_id}")

        return session, new_token, True

    async def touch_session(self, session_id: UUID) -> IncognitoSession | None:
        """Update session activity and extend TTL.

        Args:
            session_id: The session UUID.

        Returns:
            Updated session, or None if not found.
        """
        session = await self.get(session_id)
        if session:
            session.touch()
            await self.update(session)
        return session

    async def count_incognito_chats(self, session_id: UUID) -> int:
        """Count active incognito chats for a session.

        Args:
            session_id: The session UUID.

        Returns:
            Number of non-deleted incognito chats.
        """
        result = await self._session.execute(
            select(func.count(Chat.id)).where(
                Chat.incognito_session_id == session_id,
                Chat.chat_type == ChatType.INCOGNITO,
                Chat.deleted_at.is_(None),
            )
        )
        return result.scalar() or 0

    async def can_create_chat(self, session_id: UUID) -> bool:
        """Check if a new incognito chat can be created.

        Args:
            session_id: The session UUID.

        Returns:
            True if under the quota limit.
        """
        count = await self.count_incognito_chats(session_id)
        return count < MAX_INCOGNITO_CHATS

    async def cleanup_expired(self) -> int:
        """Delete expired sessions and their chats.

        Due to CASCADE delete on the foreign key, deleting an
        IncognitoSession will automatically delete all associated
        incognito chats.

        Returns:
            Number of expired sessions deleted.
        """
        now = datetime.now(UTC)

        # Find expired sessions
        result = await self._session.execute(
            select(IncognitoSession).where(IncognitoSession.expires_at < now)
        )
        expired_sessions = result.scalars().all()

        if not expired_sessions:
            return 0

        # Delete expired sessions (cascade deletes chats)
        count = len(expired_sessions)
        for session in expired_sessions:
            await self._session.delete(session)

        await self._session.flush()
        logger.info(f"Cleaned up {count} expired incognito sessions")

        return count

    async def get_session_status(
        self, session_token: str | None, user_id: str
    ) -> SessionStatusResponse:
        """Get status information for a session with user ownership verification.

        Args:
            session_token: The session token from cookie.
            user_id: The authenticated user's ID to verify ownership.

        Returns:
            SessionStatusResponse with session status:
            - has_session: Whether a valid session exists for this user
            - chat_count: Number of active incognito chats
            - max_chats: Maximum allowed chats
            - expires_at: Session expiry time (or None)
        """
        if not session_token:
            return SessionStatusResponse(
                has_session=False,
                chat_count=0,
                max_chats=MAX_INCOGNITO_CHATS,
                expires_at=None,
            )

        # Use secure method that verifies user ownership
        session = await self.get_by_token_for_user(session_token, user_id)
        if not session:
            return SessionStatusResponse(
                has_session=False,
                chat_count=0,
                max_chats=MAX_INCOGNITO_CHATS,
                expires_at=None,
            )

        chat_count = await self.count_incognito_chats(session.id)
        return SessionStatusResponse(
            has_session=True,
            chat_count=chat_count,
            max_chats=MAX_INCOGNITO_CHATS,
            expires_at=session.expires_at.isoformat(),
        )
