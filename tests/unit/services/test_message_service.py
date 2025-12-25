"""Unit tests for MessageService."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.message import Message, MessageRole
from src.services.message_service import MessageService


class TestMessageServiceCreate:
    """Tests for MessageService.create method."""

    @pytest.mark.asyncio
    async def test_create_user_message(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test creating a user message."""
        # Arrange
        chat_id = uuid4()
        content = "What are the latest quantum computing developments?"

        service = MessageService(mock_db_session)

        # Act
        result = await service.create(
            chat_id=chat_id,
            role=MessageRole.USER,
            content=content,
        )

        # Assert
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_awaited_once()
        assert result.chat_id == chat_id
        assert result.role == MessageRole.USER
        assert result.content == content
        # is_edited has a default of False in the model, but may be None before DB commit
        assert result.is_edited in (False, None)

    @pytest.mark.asyncio
    async def test_create_agent_message(
        self, mock_db_session: AsyncMock
    ):
        """Test creating an agent message."""
        # Arrange
        chat_id = uuid4()
        content = "Based on my research, quantum computing has..."

        service = MessageService(mock_db_session)

        # Act
        result = await service.create(
            chat_id=chat_id,
            role=MessageRole.AGENT,
            content=content,
        )

        # Assert
        assert result.role == MessageRole.AGENT
        assert result.content == content


class TestMessageServiceGet:
    """Tests for MessageService.get and get_with_chat methods."""

    @pytest.mark.asyncio
    async def test_get_existing_message(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test getting an existing message by ID."""
        # Arrange
        message_id = uuid4()
        expected_message = message_factory(id=message_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_message
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.get(message_id)

        # Assert
        assert result == expected_message

    @pytest.mark.asyncio
    async def test_get_nonexistent_message(self, mock_db_session: AsyncMock):
        """Test getting a message that doesn't exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.get(uuid4())

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_chat(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test getting a message by ID and chat ID."""
        # Arrange
        message_id = uuid4()
        chat_id = uuid4()
        expected_message = message_factory(id=message_id, chat_id=chat_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_message
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.get_with_chat(message_id, chat_id)

        # Assert
        assert result == expected_message

    @pytest.mark.asyncio
    async def test_get_with_chat_wrong_chat_id(
        self, mock_db_session: AsyncMock
    ):
        """Test that get_with_chat returns None for wrong chat_id."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.get_with_chat(uuid4(), uuid4())

        # Assert
        assert result is None


class TestMessageServiceListMessages:
    """Tests for MessageService.list_messages method."""

    @pytest.mark.asyncio
    async def test_list_messages_empty(self, mock_db_session: AsyncMock):
        """Test listing messages when none exist."""
        # Arrange
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = MessageService(mock_db_session)

        # Act
        messages, total = await service.list_messages(chat_id=uuid4())

        # Assert
        assert messages == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_messages_with_results(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test listing messages with results."""
        # Arrange
        chat_id = uuid4()
        msg1 = message_factory(chat_id=chat_id, content="Hello")
        msg2 = message_factory(
            chat_id=chat_id, content="Hi!", role=MessageRole.AGENT
        )

        count_result = MagicMock()
        count_result.scalar.return_value = 2

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [msg1, msg2]

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = MessageService(mock_db_session)

        # Act
        messages, total = await service.list_messages(chat_id=chat_id)

        # Assert
        assert len(messages) == 2
        assert total == 2

    @pytest.mark.asyncio
    async def test_list_messages_with_pagination(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test listing messages with limit and offset."""
        # Arrange
        chat_id = uuid4()
        msg = message_factory(chat_id=chat_id)

        count_result = MagicMock()
        count_result.scalar.return_value = 10  # Total is 10

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [msg]  # But only 1 returned

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = MessageService(mock_db_session)

        # Act
        messages, total = await service.list_messages(
            chat_id=chat_id, limit=1, offset=5
        )

        # Assert
        assert len(messages) == 1
        assert total == 10

    @pytest.mark.asyncio
    async def test_list_messages_before_timestamp(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test listing messages before a specific timestamp."""
        # Arrange
        chat_id = uuid4()
        before = datetime.now(UTC)
        msg = message_factory(
            chat_id=chat_id,
            created_at=before - timedelta(hours=1),
        )

        count_result = MagicMock()
        count_result.scalar.return_value = 1

        list_result = MagicMock()
        list_result.scalars.return_value.all.return_value = [msg]

        mock_db_session.execute.side_effect = [count_result, list_result]

        service = MessageService(mock_db_session)

        # Act
        messages, total = await service.list_messages(
            chat_id=chat_id, before=before
        )

        # Assert
        assert len(messages) == 1


class TestMessageServiceUpdateContent:
    """Tests for MessageService.update_content method."""

    @pytest.mark.asyncio
    async def test_update_content_success(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test updating message content."""
        # Arrange
        message_id = uuid4()
        existing_message = message_factory(
            id=message_id, content="Original content"
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_message
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.update_content(message_id, "Updated content")

        # Assert
        assert result is not None
        assert result.content == "Updated content"
        assert result.is_edited is True
        mock_db_session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_content_nonexistent(self, mock_db_session: AsyncMock):
        """Test updating content of nonexistent message."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.update_content(uuid4(), "New content")

        # Assert
        assert result is None


class TestMessageServiceDeleteSubsequent:
    """Tests for MessageService.delete_subsequent method."""

    @pytest.mark.asyncio
    async def test_delete_subsequent_messages(self, mock_db_session: AsyncMock):
        """Test deleting messages after a certain time."""
        # Arrange
        chat_id = uuid4()
        after_time = datetime.now(UTC) - timedelta(hours=1)

        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        count = await service.delete_subsequent(chat_id, after_time)

        # Assert
        assert count == 3
        mock_db_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_subsequent_no_messages(self, mock_db_session: AsyncMock):
        """Test deleting when no subsequent messages exist."""
        # Arrange
        chat_id = uuid4()
        after_time = datetime.now(UTC)

        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        count = await service.delete_subsequent(chat_id, after_time)

        # Assert
        assert count == 0


class TestMessageServiceSetResearchSession:
    """Tests for MessageService.set_research_session method."""

    @pytest.mark.asyncio
    async def test_set_research_session_success(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test associating a message with a research session."""
        # Arrange
        message_id = uuid4()
        research_session_id = uuid4()
        existing_message = message_factory(id=message_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_message
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.set_research_session(
            message_id, research_session_id
        )

        # Assert
        assert result is not None
        assert result.research_session_id == research_session_id

    @pytest.mark.asyncio
    async def test_set_research_session_nonexistent(
        self, mock_db_session: AsyncMock
    ):
        """Test setting research session on nonexistent message."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        result = await service.set_research_session(uuid4(), uuid4())

        # Assert
        assert result is None


class TestMessageServiceGetConversationHistory:
    """Tests for MessageService.get_conversation_history method."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(
        self, mock_db_session: AsyncMock
    ):
        """Test getting history when no messages exist."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        history = await service.get_conversation_history(uuid4())

        # Assert
        assert history == []

    @pytest.mark.asyncio
    async def test_get_conversation_history_with_messages(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test getting history with messages."""
        # Arrange
        chat_id = uuid4()
        # Messages come back in desc order from query, then reversed
        msg1 = message_factory(
            chat_id=chat_id, role=MessageRole.USER, content="Hello"
        )
        msg2 = message_factory(
            chat_id=chat_id, role=MessageRole.AGENT, content="Hi there!"
        )

        mock_result = MagicMock()
        # Query returns desc order
        mock_result.scalars.return_value.all.return_value = [msg2, msg1]
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        history = await service.get_conversation_history(chat_id)

        # Assert
        # Should be in chronological order after reversal
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "agent"
        assert history[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_get_conversation_history_respects_limit(
        self, mock_db_session: AsyncMock, message_factory
    ):
        """Test that history respects limit parameter."""
        # Arrange
        chat_id = uuid4()
        messages = [
            message_factory(chat_id=chat_id, content=f"Message {i}")
            for i in range(5)
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = messages[:3]  # Limited
        mock_db_session.execute.return_value = mock_result

        service = MessageService(mock_db_session)

        # Act
        history = await service.get_conversation_history(chat_id, limit=3)

        # Assert
        assert len(history) == 3
