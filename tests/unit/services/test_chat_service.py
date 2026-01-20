"""Unit tests for ChatService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.models.chat import Chat, ChatStatus
from deep_research.services.chat_service import ChatService


class TestChatServiceCreate:
    """Tests for ChatService.create method."""

    @pytest.mark.asyncio
    async def test_create_chat_with_title(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test creating a chat with a title."""
        # Arrange
        user_id = "test-user-123"
        title = "My Research Chat"
        expected_chat = chat_factory(user_id=user_id, title=title)

        # Configure mock to set ID on add
        def side_effect_add(obj):
            if isinstance(obj, Chat):
                obj.id = expected_chat.id

        mock_db_session.add.side_effect = side_effect_add
        mock_db_session.refresh = AsyncMock()

        service = ChatService(mock_db_session)

        # Act
        result = await service.create(user_id=user_id, title=title)

        # Assert
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_awaited_once()
        assert result.user_id == user_id
        assert result.title == title

    @pytest.mark.asyncio
    async def test_create_chat_without_title(self, mock_db_session: AsyncMock):
        """Test creating a chat without a title."""
        # Arrange
        user_id = "test-user-456"
        service = ChatService(mock_db_session)

        # Act
        result = await service.create(user_id=user_id, title=None)

        # Assert
        mock_db_session.add.assert_called_once()
        assert result.user_id == user_id
        assert result.title is None


class TestChatServiceGetForUser:
    """Tests for ChatService.get_for_user method."""

    @pytest.mark.asyncio
    async def test_get_existing_chat(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test getting an existing chat."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user-123"
        expected_chat = chat_factory(id=chat_id, user_id=user_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_chat
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.get_for_user(chat_id, user_id)

        # Assert
        assert result == expected_chat
        mock_db_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_nonexistent_chat(self, mock_db_session: AsyncMock):
        """Test getting a chat that doesn't exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.get_for_user(uuid4(), "test-user-123")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_chat_wrong_user(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test that get_for_user returns None for wrong user."""
        # Arrange
        chat_id = uuid4()
        # Simulate query returning None because user_id doesn't match
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.get_for_user(chat_id, "wrong-user")

        # Assert
        assert result is None


class TestChatServiceList:
    """Tests for ChatService.list method."""

    @pytest.mark.asyncio
    async def test_list_chats_empty(self, mock_db_session: AsyncMock):
        """Test listing chats when none exist."""
        # Arrange
        # First call: count query
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        # Second call: list query
        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = ChatService(mock_db_session)

        # Act
        chats, total = await service.list(user_id="test-user")

        # Assert
        assert chats == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_chats_with_results(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test listing chats with results."""
        # Arrange
        user_id = "test-user-123"
        chat1 = chat_factory(user_id=user_id, title="Chat 1")
        chat2 = chat_factory(user_id=user_id, title="Chat 2")

        count_result = MagicMock()
        count_result.scalar.return_value = 2

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [chat1, chat2]

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = ChatService(mock_db_session)

        # Act
        chats, total = await service.list(user_id=user_id)

        # Assert
        assert len(chats) == 2
        assert total == 2

    @pytest.mark.asyncio
    async def test_list_chats_with_status_filter(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test listing chats with status filter."""
        # Arrange
        user_id = "test-user-123"
        archived_chat = chat_factory(
            user_id=user_id, title="Archived", status=ChatStatus.ARCHIVED
        )

        count_result = MagicMock()
        count_result.scalar.return_value = 1

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [archived_chat]

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = ChatService(mock_db_session)

        # Act
        chats, total = await service.list(
            user_id=user_id, status=ChatStatus.ARCHIVED
        )

        # Assert
        assert len(chats) == 1
        assert chats[0].status == ChatStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_list_chats_with_search(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test listing chats with search filter."""
        # Arrange
        user_id = "test-user-123"
        matching_chat = chat_factory(
            user_id=user_id, title="Quantum Computing Research"
        )

        count_result = MagicMock()
        count_result.scalar.return_value = 1

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [matching_chat]

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = ChatService(mock_db_session)

        # Act
        chats, total = await service.list(user_id=user_id, search="quantum")

        # Assert
        assert len(chats) == 1


class TestChatServiceUpdateChat:
    """Tests for ChatService.update_chat method."""

    @pytest.mark.asyncio
    async def test_update_chat_title(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test updating chat title."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user-123"
        existing_chat = chat_factory(id=chat_id, user_id=user_id, title="Old Title")

        # Mock get_for_user() to return existing chat
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_chat
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.update_chat(
            chat_id=chat_id, user_id=user_id, title="New Title"
        )

        # Assert
        assert result is not None
        assert result.title == "New Title"
        mock_db_session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_chat_status(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test updating chat status to archived."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user-123"
        existing_chat = chat_factory(id=chat_id, user_id=user_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_chat
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.update_chat(
            chat_id=chat_id, user_id=user_id, status=ChatStatus.ARCHIVED
        )

        # Assert
        assert result is not None
        assert result.status == ChatStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_update_nonexistent_chat(self, mock_db_session: AsyncMock):
        """Test updating a chat that doesn't exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.update_chat(
            chat_id=uuid4(), user_id="test-user", title="New Title"
        )

        # Assert
        assert result is None


class TestChatServiceSoftDelete:
    """Tests for ChatService.soft_delete method."""

    @pytest.mark.asyncio
    async def test_soft_delete_existing_chat(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test soft deleting an existing chat."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user-123"
        existing_chat = chat_factory(id=chat_id, user_id=user_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_chat
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.soft_delete(chat_id, user_id)

        # Assert
        assert result is True
        assert existing_chat.status == ChatStatus.DELETED
        assert existing_chat.deleted_at is not None

    @pytest.mark.asyncio
    async def test_soft_delete_nonexistent_chat(self, mock_db_session: AsyncMock):
        """Test soft deleting a chat that doesn't exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.soft_delete(uuid4(), "test-user")

        # Assert
        assert result is False


class TestChatServiceRestore:
    """Tests for ChatService.restore method."""

    @pytest.mark.asyncio
    async def test_restore_deleted_chat(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test restoring a soft-deleted chat."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user-123"
        deleted_chat = chat_factory(
            id=chat_id,
            user_id=user_id,
            status=ChatStatus.DELETED,
            deleted_at=datetime.now(UTC),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = deleted_chat
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.restore(chat_id, user_id)

        # Assert
        assert result is not None
        assert result.status == ChatStatus.ACTIVE
        assert result.deleted_at is None

    @pytest.mark.asyncio
    async def test_restore_nonexistent_chat(self, mock_db_session: AsyncMock):
        """Test restoring a chat that doesn't exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        result = await service.restore(uuid4(), "test-user")

        # Assert
        assert result is None


class TestChatServiceUpdateTitleFromMessage:
    """Tests for ChatService.update_title_from_message method."""

    @pytest.mark.asyncio
    async def test_update_title_when_no_title(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test that title is set from first message when chat has no title."""
        # Arrange
        chat_id = uuid4()
        chat_without_title = chat_factory(id=chat_id, title=None)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = chat_without_title
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        await service.update_title_from_message(
            chat_id, "What are the latest developments in quantum computing?"
        )

        # Assert
        assert chat_without_title.title is not None
        assert "quantum" in chat_without_title.title.lower()

    @pytest.mark.asyncio
    async def test_no_update_when_title_exists(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test that existing title is not overwritten."""
        # Arrange
        chat_id = uuid4()
        existing_title = "My Existing Title"
        chat_with_title = chat_factory(id=chat_id, title=existing_title)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = chat_with_title
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)

        # Act
        await service.update_title_from_message(chat_id, "New message content")

        # Assert
        assert chat_with_title.title == existing_title

    @pytest.mark.asyncio
    async def test_title_truncation_for_long_messages(
        self, mock_db_session: AsyncMock, chat_factory
    ):
        """Test that long messages are truncated for title."""
        # Arrange
        chat_id = uuid4()
        chat_without_title = chat_factory(id=chat_id, title=None)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = chat_without_title
        mock_db_session.execute.return_value = mock_result

        service = ChatService(mock_db_session)
        long_message = "x " * 200  # 400 chars

        # Act
        await service.update_title_from_message(chat_id, long_message)

        # Assert
        assert chat_without_title.title is not None
        assert len(chat_without_title.title) <= 103  # 100 + "..."
        assert chat_without_title.title.endswith("...")
