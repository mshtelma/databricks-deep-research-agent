"""Unit tests for ExportService.

Tests the batch source loading optimization (N+1 query fix).
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


# Use delayed import to avoid circular import issues
@pytest.fixture
def export_service_class():
    """Import ExportService class lazily to avoid circular imports."""
    from deep_research.services.export_service import ExportService
    return ExportService


@pytest.fixture
def message_role():
    """Import MessageRole enum lazily."""
    from deep_research.models.message import MessageRole
    return MessageRole


class TestExportServiceBatchLoading:
    """Tests for batch source loading optimization."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def export_service(self, mock_session, export_service_class):
        """Create export service with mocked dependencies."""
        return export_service_class(session=mock_session)

    @pytest.mark.asyncio
    async def test_get_sources_for_messages_empty_list(self, export_service):
        """Test batch loading with empty message list."""
        result = await export_service._get_sources_for_messages([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_sources_for_messages_returns_grouped_sources(
        self, mock_session, export_service
    ):
        """Test that sources are correctly grouped by message_id."""
        # Arrange
        msg1_id = uuid4()
        msg2_id = uuid4()

        # Create mock Source objects
        source1 = MagicMock()
        source1.title = "Source 1"
        source1.url = "https://example.com/1"

        source2 = MagicMock()
        source2.title = "Source 2"
        source2.url = "https://example.com/2"

        source3 = MagicMock()
        source3.title = None  # Test fallback to URL
        source3.url = "https://example.com/3"

        # Mock query result - returns (Source, message_id) tuples
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (source1, msg1_id),
            (source2, msg1_id),  # Two sources for msg1
            (source3, msg2_id),  # One source for msg2
        ]
        mock_session.execute.return_value = mock_result

        # Act
        result = await export_service._get_sources_for_messages([msg1_id, msg2_id])

        # Assert
        assert len(result) == 2
        assert len(result[msg1_id]) == 2
        assert len(result[msg2_id]) == 1
        assert result[msg1_id][0]["title"] == "Source 1"
        assert result[msg1_id][0]["url"] == "https://example.com/1"
        # Test title fallback to URL when title is None
        assert result[msg2_id][0]["title"] == "https://example.com/3"

    @pytest.mark.asyncio
    async def test_get_sources_for_messages_single_query(
        self, mock_session, export_service
    ):
        """Test that only one query is executed for batch loading."""
        # Arrange
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Act
        await export_service._get_sources_for_messages([uuid4(), uuid4(), uuid4()])

        # Assert - only ONE query should be executed
        assert mock_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_export_markdown_uses_batch_loading(
        self, mock_session, message_role, export_service_class
    ):
        """Test that export_markdown uses batch loading instead of N+1 queries."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user"

        # Mock chat
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.title = "Test Chat"
        mock_chat.created_at = None

        # Mock messages - 3 agent messages
        messages = []
        for i in range(3):
            msg = MagicMock()
            msg.id = uuid4()
            msg.role = message_role.AGENT
            msg.content = f"Agent response {i}"
            messages.append(msg)

        # Add a user message
        user_msg = MagicMock()
        user_msg.id = uuid4()
        user_msg.role = message_role.USER
        user_msg.content = "User question"
        messages.insert(0, user_msg)

        # Mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        # Mock message service
        mock_message_service = AsyncMock()
        mock_message_service.list_messages = AsyncMock(
            return_value=(messages, len(messages))
        )

        # Mock batch source loading - return empty for all
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Create service with mocks
        service = export_service_class(
            session=mock_session,
            chat_service=mock_chat_service,
            message_service=mock_message_service,
        )

        # Act
        await service.export_markdown(chat_id, user_id, include_sources=True)

        # Assert - Should only execute ONE query for batch source loading
        # Not 2 * 3 = 6 queries (one for session + one for sources per agent message)
        assert mock_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_export_markdown_skips_batch_when_sources_disabled(
        self, mock_session, message_role, export_service_class
    ):
        """Test that batch loading is skipped when include_sources=False."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user"

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.title = "Test Chat"
        mock_chat.created_at = None

        mock_msg = MagicMock()
        mock_msg.id = uuid4()
        mock_msg.role = message_role.USER
        mock_msg.content = "User question"

        mock_chat_service = AsyncMock()
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        mock_message_service = AsyncMock()
        mock_message_service.list_messages = AsyncMock(return_value=([mock_msg], 1))

        service = export_service_class(
            session=mock_session,
            chat_service=mock_chat_service,
            message_service=mock_message_service,
        )

        # Act
        await service.export_markdown(chat_id, user_id, include_sources=False)

        # Assert - No source queries should be executed
        mock_session.execute.assert_not_called()


class TestExportServiceSourcesInOutput:
    """Tests for source inclusion in markdown output."""

    @pytest.mark.asyncio
    async def test_sources_included_in_markdown_output(
        self, export_service_class, message_role
    ):
        """Test that sources appear in the markdown output."""
        # Arrange
        chat_id = uuid4()
        user_id = "test-user"
        msg_id = uuid4()

        mock_session = AsyncMock()

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.title = "Test Research"
        mock_chat.created_at = None

        mock_msg = MagicMock()
        mock_msg.id = msg_id
        mock_msg.role = message_role.AGENT
        mock_msg.content = "Research findings."

        mock_chat_service = AsyncMock()
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        mock_message_service = AsyncMock()
        mock_message_service.list_messages = AsyncMock(return_value=([mock_msg], 1))

        # Mock sources returned by batch loading
        source = MagicMock()
        source.title = "Important Paper"
        source.url = "https://arxiv.org/paper"

        mock_result = MagicMock()
        mock_result.all.return_value = [(source, msg_id)]
        mock_session.execute.return_value = mock_result

        service = export_service_class(
            session=mock_session,
            chat_service=mock_chat_service,
            message_service=mock_message_service,
        )

        # Act
        markdown = await service.export_markdown(chat_id, user_id, include_sources=True)

        # Assert
        assert "#### Sources" in markdown
        assert "[Important Paper](https://arxiv.org/paper)" in markdown
