"""Integration tests for chat access control.

Verifies that users cannot access other users' chats via direct URL or API.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.api.v1.utils.authorization import verify_chat_ownership
from deep_research.core.exceptions import NotFoundError


@pytest.mark.integration
class TestChatAccessControl:
    """Tests for chat access control via direct URL/API access."""

    @pytest.mark.asyncio
    async def test_verify_chat_ownership_returns_404_for_unauthorized_access(
        self,
    ) -> None:
        """verify_chat_ownership returns NotFoundError (404) for unauthorized access.

        This prevents information leakage by returning the same error as 'not found'.
        """
        from unittest.mock import AsyncMock

        from deep_research.services.chat_service import ChatService

        # Create mock DB session
        mock_db = AsyncMock()

        # Setup: chat exists but belongs to different user
        mock_chat = MagicMock()
        mock_chat.id = uuid4()
        mock_chat.user_id = "user-a-id"  # Chat belongs to User A

        # Mock ChatService to return None for get_for_user (ownership check fails)
        with patch.object(ChatService, "get_for_user", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None  # User B can't access User A's chat

            # User B tries to access User A's chat
            with pytest.raises(NotFoundError) as exc_info:
                await verify_chat_ownership(
                    chat_id=mock_chat.id,
                    user_id="user-b-id",  # User B trying to access
                    db=mock_db,
                )

            # Verify 404-style error (not 403)
            assert "Chat" in str(exc_info.value)
            mock_get.assert_called_once_with(mock_chat.id, "user-b-id")

    @pytest.mark.asyncio
    async def test_verify_chat_ownership_succeeds_for_owner(self) -> None:
        """verify_chat_ownership succeeds when user owns the chat."""
        from unittest.mock import AsyncMock

        from deep_research.services.chat_service import ChatService

        mock_db = AsyncMock()
        chat_id = uuid4()

        # Setup: chat exists and belongs to requesting user
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.user_id = "user-a-id"

        with patch.object(ChatService, "get_for_user", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_chat  # Owner can access

            # User A accesses their own chat
            result = await verify_chat_ownership(
                chat_id=chat_id,
                user_id="user-a-id",
                db=mock_db,
            )

            assert result == mock_chat
            mock_get.assert_called_once_with(chat_id, "user-a-id")

    @pytest.mark.asyncio
    async def test_verify_chat_ownership_returns_404_for_nonexistent_chat(self) -> None:
        """verify_chat_ownership returns NotFoundError for nonexistent chat.

        Same error type as unauthorized access to prevent enumeration.
        """
        from unittest.mock import AsyncMock

        from deep_research.services.chat_service import ChatService

        mock_db = AsyncMock()
        nonexistent_chat_id = uuid4()

        with patch.object(ChatService, "get_for_user", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None  # Chat doesn't exist

            with pytest.raises(NotFoundError) as exc_info:
                await verify_chat_ownership(
                    chat_id=nonexistent_chat_id,
                    user_id="any-user",
                    db=mock_db,
                )

            # Same error type as unauthorized
            assert "Chat" in str(exc_info.value)
