"""Integration tests for research session access control.

Verifies that users cannot cancel or access other users' research sessions.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.core.exceptions import NotFoundError


@pytest.mark.integration
class TestResearchSessionAccessControl:
    """Tests for research session access control via API."""

    @pytest.mark.asyncio
    async def test_cancel_research_returns_404_for_unauthorized_user(self) -> None:
        """cancel_research returns NotFoundError (404) for unauthorized access.

        This prevents information leakage by returning 404 instead of 403.
        """
        from deep_research.api.v1.research import cancel_research
        from deep_research.core.auth import UserIdentity
        from deep_research.models.chat import Chat
        from deep_research.models.message import Message
        from deep_research.models.research_session import ResearchSession

        # Setup mock database session
        mock_db = AsyncMock()

        # Create mock objects
        session_id = uuid4()
        message_id = uuid4()
        chat_id = uuid4()

        # Session exists
        mock_session = MagicMock(spec=ResearchSession)
        mock_session.id = session_id
        mock_session.message_id = message_id

        # Message exists
        mock_message = MagicMock(spec=Message)
        mock_message.id = message_id
        mock_message.chat_id = chat_id

        # Chat belongs to User A
        mock_chat = MagicMock(spec=Chat)
        mock_chat.id = chat_id
        mock_chat.user_id = "user-a-id"

        # Mock db.get to return session and message
        async def mock_get(model: type, id: str) -> MagicMock | None:
            if model.__name__ == "ResearchSession" and id == session_id:
                return mock_session
            if model.__name__ == "Message" and id == message_id:
                return mock_message
            return None

        mock_db.get = mock_get

        # User B attempts to cancel User A's research
        user_b = UserIdentity(
            user_id="user-b-id",
            email="userb@company.com",
            display_name="User B",
        )

        with patch(
            "deep_research.api.v1.research.ChatService"
        ) as MockChatService:
            mock_chat_service = AsyncMock()
            mock_chat_service.get_by_id.return_value = mock_chat
            MockChatService.return_value = mock_chat_service

            # Execute - should raise NotFoundError (not AuthorizationError)
            with pytest.raises(NotFoundError) as exc_info:
                await cancel_research(
                    session_id=session_id,
                    user=user_b,
                    db=mock_db,
                )

            # Verify 404-style error (not 403)
            assert "ResearchSession" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_research_succeeds_for_owner(self) -> None:
        """cancel_research succeeds when user owns the research session."""
        from deep_research.api.v1.research import cancel_research
        from deep_research.core.auth import UserIdentity
        from deep_research.models.chat import Chat
        from deep_research.models.message import Message
        from deep_research.models.research_session import ResearchSession

        mock_db = AsyncMock()

        session_id = uuid4()
        message_id = uuid4()
        chat_id = uuid4()

        # Session exists
        mock_session = MagicMock(spec=ResearchSession)
        mock_session.id = session_id
        mock_session.message_id = message_id
        mock_session.observations = None

        # Message exists
        mock_message = MagicMock(spec=Message)
        mock_message.id = message_id
        mock_message.chat_id = chat_id

        # Chat belongs to User A (the requester)
        mock_chat = MagicMock(spec=Chat)
        mock_chat.id = chat_id
        mock_chat.user_id = "user-a-id"

        async def mock_get(model: type, id: str) -> MagicMock | None:
            if model.__name__ == "ResearchSession" and id == session_id:
                return mock_session
            if model.__name__ == "Message" and id == message_id:
                return mock_message
            return None

        mock_db.get = mock_get
        mock_db.commit = AsyncMock()

        # User A cancels their own research
        user_a = UserIdentity(
            user_id="user-a-id",
            email="usera@company.com",
            display_name="User A",
        )

        with (
            patch("deep_research.api.v1.research.ChatService") as MockChatService,
            patch(
                "deep_research.api.v1.research.ResearchSessionService"
            ) as MockResearchService,
        ):
            mock_chat_service = AsyncMock()
            mock_chat_service.get_by_id.return_value = mock_chat
            MockChatService.return_value = mock_chat_service

            mock_research_service = AsyncMock()
            mock_research_service.cancel.return_value = mock_session
            MockResearchService.return_value = mock_research_service

            # Execute - should succeed
            result = await cancel_research(
                session_id=session_id,
                user=user_a,
                db=mock_db,
            )

            assert result.session_id == session_id
            assert result.status == "cancelled"
            mock_research_service.cancel.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_cancel_research_returns_404_for_nonexistent_session(self) -> None:
        """cancel_research returns NotFoundError for nonexistent session.

        Same error type as unauthorized access to prevent enumeration.
        """
        from deep_research.api.v1.research import cancel_research
        from deep_research.core.auth import UserIdentity

        mock_db = AsyncMock()
        nonexistent_session_id = uuid4()

        # Session doesn't exist
        mock_db.get = AsyncMock(return_value=None)

        user = UserIdentity(
            user_id="any-user",
            email="user@company.com",
            display_name="Any User",
        )

        with pytest.raises(NotFoundError) as exc_info:
            await cancel_research(
                session_id=nonexistent_session_id,
                user=user,
                db=mock_db,
            )

        # Same error type as unauthorized
        assert "ResearchSession" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_research_events_returns_403_for_unauthorized_user(
        self,
    ) -> None:
        """get_research_events raises AuthorizationError for unauthorized access."""
        from deep_research.api.v1.research import get_research_events
        from deep_research.core.auth import UserIdentity
        from deep_research.core.exceptions import AuthorizationError
        from deep_research.models.chat import Chat
        from deep_research.models.message import Message
        from deep_research.models.research_session import ResearchSession

        mock_db = AsyncMock()

        session_id = uuid4()
        message_id = uuid4()
        chat_id = uuid4()

        # Session exists
        mock_session = MagicMock(spec=ResearchSession)
        mock_session.id = session_id
        mock_session.message_id = message_id
        mock_session.status = "in_progress"

        # Message exists with correct chat_id
        mock_message = MagicMock(spec=Message)
        mock_message.id = message_id
        mock_message.chat_id = chat_id

        # Chat belongs to User A
        mock_chat = MagicMock(spec=Chat)
        mock_chat.id = chat_id
        mock_chat.user_id = "user-a-id"

        async def mock_get(model: type, id: str) -> MagicMock | None:
            if model.__name__ == "ResearchSession" and id == session_id:
                return mock_session
            if model.__name__ == "Message" and id == message_id:
                return mock_message
            return None

        mock_db.get = mock_get

        # User B attempts to access
        user_b = UserIdentity(
            user_id="user-b-id",
            email="userb@company.com",
            display_name="User B",
        )

        with patch(
            "deep_research.api.v1.research.ChatService"
        ) as MockChatService:
            mock_chat_service = AsyncMock()
            mock_chat_service.get_by_id.return_value = mock_chat
            MockChatService.return_value = mock_chat_service

            with pytest.raises(AuthorizationError):
                await get_research_events(
                    chat_id=chat_id,
                    session_id=session_id,
                    user=user_b,
                    since_sequence=0,
                    limit=100,
                    db=mock_db,
                )
