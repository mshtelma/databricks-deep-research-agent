"""Unit tests for Message API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.core.auth import UserIdentity
from src.db.session import get_db
from src.main import app
from src.middleware.auth import get_current_user_identity
from src.models.message import Message, MessageRole


@pytest.fixture
def mock_user() -> UserIdentity:
    """Create a test user identity."""
    return UserIdentity(
        user_id="test-user-123",
        email="test@example.com",
        display_name="Test User",
    )


@pytest.fixture
def mock_message() -> Message:
    """Create a mock message for testing."""
    from datetime import UTC, datetime

    msg = Message(
        chat_id=uuid4(),
        role=MessageRole.USER,
        content="Test message content",
    )
    msg.id = uuid4()
    msg.created_at = datetime.now(UTC)
    msg.is_edited = False
    return msg


@pytest.fixture
def mock_agent_message() -> Message:
    """Create a mock agent message for testing."""
    from datetime import UTC, datetime

    msg = Message(
        chat_id=uuid4(),
        role=MessageRole.AGENT,
        content="Agent response content",
    )
    msg.id = uuid4()
    msg.created_at = datetime.now(UTC)
    msg.is_edited = False
    return msg


@pytest.fixture
def client(mock_user: UserIdentity) -> TestClient:
    """Create a test client with mocked dependencies."""

    async def override_get_db():
        mock_session = AsyncMock()
        yield mock_session

    async def override_get_current_user_identity():
        return mock_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_identity] = (
        override_get_current_user_identity
    )

    yield TestClient(app)

    # Clear overrides after test
    app.dependency_overrides.clear()


class TestListMessages:
    """Tests for GET /api/v1/chats/{chat_id}/messages endpoint."""

    def test_list_messages_empty(self, client: TestClient):
        """Test listing messages when none exist."""
        chat_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.list_messages = AsyncMock(return_value=([], 0))
            MockService.return_value = mock_service

            response = client.get(f"/api/v1/chats/{chat_id}/messages")

            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0

    def test_list_messages_with_results(
        self, client: TestClient, mock_message: Message
    ):
        """Test listing messages with results."""
        chat_id = mock_message.chat_id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.list_messages = AsyncMock(
                return_value=([mock_message], 1)
            )
            MockService.return_value = mock_service

            response = client.get(f"/api/v1/chats/{chat_id}/messages")

            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 1
            assert data["total"] == 1

    def test_list_messages_with_pagination(self, client: TestClient):
        """Test listing messages with pagination parameters."""
        chat_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.list_messages = AsyncMock(return_value=([], 0))
            MockService.return_value = mock_service

            response = client.get(
                f"/api/v1/chats/{chat_id}/messages?limit=10&offset=5"
            )

            assert response.status_code == 200
            mock_service.list_messages.assert_awaited_once()


class TestSendMessage:
    """Tests for POST /api/v1/chats/{chat_id}/messages endpoint."""

    def test_send_message_success(
        self, client: TestClient, mock_message: Message
    ):
        """Test sending a message successfully."""
        chat_id = mock_message.chat_id

        # Create a mock Chat for ownership verification
        mock_chat = MagicMock()
        mock_chat.id = chat_id

        with (
            patch("src.api.v1.messages.MessageService") as MockMessageService,
            patch("src.api.v1.messages.ChatService") as MockChatService,
        ):
            mock_message_service = MagicMock()
            mock_message_service.create = AsyncMock(return_value=mock_message)
            MockMessageService.return_value = mock_message_service

            mock_chat_service = MagicMock()
            mock_chat_service.get = AsyncMock(return_value=mock_chat)
            mock_chat_service.update_title_from_message = AsyncMock(return_value=None)
            MockChatService.return_value = mock_chat_service

            response = client.post(
                f"/api/v1/chats/{chat_id}/messages",
                json={"content": "What is quantum computing?"},
            )

            assert response.status_code == 201
            data = response.json()
            assert "user_message" in data
            assert "research_session_id" in data

    def test_send_message_empty_content(self, client: TestClient):
        """Test sending a message with empty content is rejected."""
        chat_id = uuid4()

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"content": ""},
        )

        # FastAPI validation should reject empty content
        assert response.status_code == 422


class TestGetMessage:
    """Tests for GET /api/v1/chats/{chat_id}/messages/{message_id} endpoint."""

    def test_get_existing_message(
        self, client: TestClient, mock_message: Message
    ):
        """Test getting an existing message."""
        chat_id = mock_message.chat_id
        message_id = mock_message.id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=mock_message)
            MockService.return_value = mock_service

            response = client.get(
                f"/api/v1/chats/{chat_id}/messages/{message_id}"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(message_id)
            assert data["content"] == mock_message.content

    def test_get_nonexistent_message(self, client: TestClient):
        """Test getting a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.get(
                f"/api/v1/chats/{chat_id}/messages/{message_id}"
            )

            assert response.status_code == 404


class TestEditMessage:
    """Tests for PATCH /api/v1/chats/{chat_id}/messages/{message_id} endpoint."""

    def test_edit_message_success(
        self, client: TestClient, mock_message: Message
    ):
        """Test editing a message successfully."""
        chat_id = mock_message.chat_id
        message_id = mock_message.id
        mock_message.is_edited = True
        mock_message.content = "Updated content"

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=mock_message)
            mock_service.delete_subsequent = AsyncMock(return_value=2)
            mock_service.update_content = AsyncMock(return_value=mock_message)
            MockService.return_value = mock_service

            response = client.patch(
                f"/api/v1/chats/{chat_id}/messages/{message_id}",
                json={"content": "Updated content"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["message"]["is_edited"] is True
            assert data["removed_message_count"] == 2

    def test_edit_nonexistent_message(self, client: TestClient):
        """Test editing a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.patch(
                f"/api/v1/chats/{chat_id}/messages/{message_id}",
                json={"content": "Updated content"},
            )

            assert response.status_code == 404

    def test_edit_agent_message_rejected(
        self, client: TestClient, mock_agent_message: Message
    ):
        """Test that editing an agent message is rejected."""
        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(
                return_value=mock_agent_message
            )
            MockService.return_value = mock_service

            response = client.patch(
                f"/api/v1/chats/{chat_id}/messages/{message_id}",
                json={"content": "Try to edit agent message"},
            )

            # Should return 404 as only user messages can be edited
            assert response.status_code == 404


class TestRegenerateMessage:
    """Tests for POST /api/v1/chats/{chat_id}/messages/{message_id}/regenerate endpoint."""

    def test_regenerate_message_success(
        self, client: TestClient, mock_agent_message: Message
    ):
        """Test regenerating an agent message."""
        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(
                return_value=mock_agent_message
            )
            mock_service.delete_subsequent = AsyncMock(return_value=1)
            MockService.return_value = mock_service

            response = client.post(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/regenerate"
            )

            assert response.status_code == 201
            data = response.json()
            assert "new_message_id" in data
            assert "research_session_id" in data

    def test_regenerate_nonexistent_message(self, client: TestClient):
        """Test regenerating a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.post(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/regenerate"
            )

            assert response.status_code == 404


class TestSubmitFeedback:
    """Tests for POST /api/v1/chats/{chat_id}/messages/{message_id}/feedback endpoint."""

    def test_submit_positive_feedback(
        self, client: TestClient, mock_agent_message: Message
    ):
        """Test submitting positive feedback."""
        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(
                return_value=mock_agent_message
            )
            MockService.return_value = mock_service

            response = client.post(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/feedback",
                json={"rating": "positive"},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["rating"] == "positive"
            assert data["message_id"] == str(message_id)

    def test_submit_negative_feedback_with_report(
        self, client: TestClient, mock_agent_message: Message
    ):
        """Test submitting negative feedback with error report."""
        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(
                return_value=mock_agent_message
            )
            MockService.return_value = mock_service

            response = client.post(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/feedback",
                json={
                    "rating": "negative",
                    "feedback_text": "The dates mentioned are incorrect",
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert data["rating"] == "negative"
            assert data["feedback_text"] is not None

    def test_submit_feedback_nonexistent_message(self, client: TestClient):
        """Test submitting feedback for a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.post(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/feedback",
                json={"rating": "positive"},
            )

            assert response.status_code == 404


class TestGetMessageContent:
    """Tests for GET /api/v1/chats/{chat_id}/messages/{message_id}/copy endpoint."""

    def test_get_message_content(
        self, client: TestClient, mock_message: Message
    ):
        """Test getting message content for clipboard."""
        chat_id = mock_message.chat_id
        message_id = mock_message.id

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=mock_message)
            MockService.return_value = mock_service

            response = client.get(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/copy"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["content"] == mock_message.content

    def test_get_message_content_nonexistent(self, client: TestClient):
        """Test getting content for a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()

        with patch("src.api.v1.messages.MessageService") as MockService:
            mock_service = MagicMock()
            mock_service.get_with_chat = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.get(
                f"/api/v1/chats/{chat_id}/messages/{message_id}/copy"
            )

            assert response.status_code == 404
