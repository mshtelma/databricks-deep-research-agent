"""Unit tests for Message API endpoints."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.core.deps import (
    get_chat_service,
    get_feedback_service,
    get_message_service,
)
from deep_research.core.exceptions import NotFoundError
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity
from deep_research.models.message import Message, MessageRole


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
def mock_chat_service(mock_user: UserIdentity) -> MagicMock:
    """Create a reusable mock chat service that passes ownership checks."""
    svc = MagicMock()
    # get_for_user returns a chat-like object owned by the test user
    mock_chat = MagicMock()
    mock_chat.user_id = mock_user.user_id
    svc.get_for_user = AsyncMock(return_value=mock_chat)
    svc.update_title_from_message = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def mock_message_service() -> MagicMock:
    """Create a reusable mock message service."""
    svc = MagicMock()
    svc.list_messages = AsyncMock(return_value=([], 0))
    svc.get_with_chat = AsyncMock(return_value=None)
    svc.create = AsyncMock(return_value=None)
    svc.update_content = AsyncMock(return_value=None)
    svc.delete_subsequent = AsyncMock(return_value=0)
    return svc


@pytest.fixture
def mock_feedback_service() -> MagicMock:
    """Create a reusable mock feedback service."""
    svc = MagicMock()
    svc.create_feedback = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def client(
    mock_user: UserIdentity,
    mock_chat_service: MagicMock,
    mock_message_service: MagicMock,
    mock_feedback_service: MagicMock,
) -> TestClient:
    """Create a test client with mocked dependencies."""

    async def override_get_current_user_identity():
        return mock_user

    app.dependency_overrides[get_current_user_identity] = (
        override_get_current_user_identity
    )
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    app.dependency_overrides[get_message_service] = lambda: mock_message_service
    app.dependency_overrides[get_feedback_service] = lambda: mock_feedback_service

    yield TestClient(app)

    # Clear overrides after test
    app.dependency_overrides.clear()


class TestListMessages:
    """Tests for GET /api/v1/chats/{chat_id}/messages endpoint."""

    def test_list_messages_empty(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test listing messages when none exist."""
        chat_id = uuid4()
        mock_message_service.list_messages = AsyncMock(return_value=([], 0))

        response = client.get(f"/api/v1/chats/{chat_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_messages_with_results(
        self,
        client: TestClient,
        mock_message: Message,
        mock_message_service: MagicMock,
        mock_chat_service: MagicMock,
    ):
        """Test listing messages with results."""
        chat_id = mock_message.chat_id
        mock_message_service.list_messages = AsyncMock(
            return_value=([mock_message], 1)
        )

        response = client.get(f"/api/v1/chats/{chat_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        mock_chat_service.get_for_user.assert_awaited_once_with(
            chat_id, "test-user-123"
        )
        mock_message_service.list_messages.assert_awaited_once_with(
            chat_id=chat_id,
            limit=20,
            offset=0,
        )

    def test_list_messages_with_pagination(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test listing messages with pagination parameters."""
        chat_id = uuid4()
        mock_message_service.list_messages = AsyncMock(return_value=([], 0))

        response = client.get(
            f"/api/v1/chats/{chat_id}/messages?limit=10&offset=5"
        )

        assert response.status_code == 200
        mock_message_service.list_messages.assert_awaited_once()

    def test_list_messages_missing_chat_returns_404(
        self,
        client: TestClient,
        mock_chat_service: MagicMock,
    ):
        """Chat not found / not owned → 404."""
        chat_id = uuid4()
        mock_chat_service.get_for_user = AsyncMock(return_value=None)

        response = client.get(f"/api/v1/chats/{chat_id}/messages")

        assert response.status_code == 404


class TestSendMessage:
    """Tests for POST /api/v1/chats/{chat_id}/messages endpoint."""

    def test_send_message_success(
        self,
        client: TestClient,
        mock_message: Message,
        mock_chat_service: MagicMock,
        mock_message_service: MagicMock,
    ):
        """Test sending a message successfully."""
        chat_id = mock_message.chat_id
        mock_message_service.create = AsyncMock(return_value=mock_message)

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"content": "What is quantum computing?"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "userMessage" in data
        assert "researchSessionId" in data
        mock_chat_service.get_for_user.assert_awaited_once_with(
            chat_id, "test-user-123"
        )
        mock_message_service.create.assert_awaited_once_with(
            chat_id=chat_id,
            role=MessageRole.USER,
            content="What is quantum computing?",
        )
        mock_chat_service.update_title_from_message.assert_awaited_once_with(
            chat_id,
            "What is quantum computing?",
        )

    def test_send_message_empty_content(self, client: TestClient):
        """Test sending a message with empty content is rejected."""
        chat_id = uuid4()

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"content": ""},
        )

        # FastAPI validation should reject empty content
        assert response.status_code == 422

    def test_send_message_invalid_chat_id_returns_422(self, client: TestClient):
        """Invalid chat IDs should be rejected before authorization/service wiring."""
        response = client.post(
            "/api/v1/chats/not-a-uuid/messages",
            json={"content": "What is quantum computing?"},
        )

        assert response.status_code == 422

    def test_send_message_missing_chat_returns_404(
        self,
        client: TestClient,
        mock_chat_service: MagicMock,
    ):
        """Ownership verification failures should map to 404."""
        chat_id = uuid4()
        mock_chat_service.get_for_user = AsyncMock(
            side_effect=NotFoundError("Chat", str(chat_id))
        )

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"content": "What is quantum computing?"},
        )

        assert response.status_code == 404


class TestGetMessage:
    """Tests for GET /api/v1/chats/{chat_id}/messages/{message_id} endpoint."""

    def test_get_existing_message(
        self,
        client: TestClient,
        mock_message: Message,
        mock_message_service: MagicMock,
        mock_chat_service: MagicMock,
    ):
        """Test getting an existing message."""
        chat_id = mock_message.chat_id
        message_id = mock_message.id
        mock_message_service.get_with_chat = AsyncMock(return_value=mock_message)

        response = client.get(
            f"/api/v1/chats/{chat_id}/messages/{message_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(message_id)
        assert data["content"] == mock_message.content
        mock_chat_service.get_for_user.assert_awaited_once_with(
            chat_id, "test-user-123"
        )
        mock_message_service.get_with_chat.assert_awaited_once_with(message_id, chat_id)

    def test_get_nonexistent_message(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test getting a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()
        mock_message_service.get_with_chat = AsyncMock(return_value=None)

        response = client.get(
            f"/api/v1/chats/{chat_id}/messages/{message_id}"
        )

        assert response.status_code == 404


class TestEditMessage:
    """Tests for PATCH /api/v1/chats/{chat_id}/messages/{message_id} endpoint."""

    def test_edit_message_success(
        self,
        client: TestClient,
        mock_message: Message,
        mock_message_service: MagicMock,
        mock_chat_service: MagicMock,
    ):
        """Test editing a message successfully."""
        chat_id = mock_message.chat_id
        message_id = mock_message.id
        mock_message.is_edited = True
        mock_message.content = "Updated content"

        mock_message_service.get_with_chat = AsyncMock(return_value=mock_message)
        mock_message_service.delete_subsequent = AsyncMock(return_value=2)
        mock_message_service.update_content = AsyncMock(return_value=mock_message)

        response = client.patch(
            f"/api/v1/chats/{chat_id}/messages/{message_id}",
            json={"content": "Updated content"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"]["isEdited"] is True
        assert data["removedMessageCount"] == 2
        mock_chat_service.get_for_user.assert_awaited_once_with(
            chat_id, "test-user-123"
        )
        mock_message_service.delete_subsequent.assert_awaited_once_with(
            chat_id,
            mock_message.created_at,
        )
        mock_message_service.update_content.assert_awaited_once_with(
            message_id,
            "Updated content",
            chat_id=chat_id,
        )

    def test_edit_nonexistent_message(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test editing a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()
        mock_message_service.get_with_chat = AsyncMock(return_value=None)

        response = client.patch(
            f"/api/v1/chats/{chat_id}/messages/{message_id}",
            json={"content": "Updated content"},
        )

        assert response.status_code == 404

    def test_edit_agent_message_rejected(
        self,
        client: TestClient,
        mock_agent_message: Message,
        mock_message_service: MagicMock,
    ):
        """Test that editing an agent message is rejected."""
        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id
        mock_message_service.get_with_chat = AsyncMock(return_value=mock_agent_message)

        response = client.patch(
            f"/api/v1/chats/{chat_id}/messages/{message_id}",
            json={"content": "Try to edit agent message"},
        )

        # Should return 404 as only user messages can be edited
        assert response.status_code == 404


class TestRegenerateMessage:
    """Tests for POST /api/v1/chats/{chat_id}/messages/{message_id}/regenerate endpoint."""

    def test_regenerate_message_success(
        self,
        client: TestClient,
        mock_agent_message: Message,
        mock_message_service: MagicMock,
        mock_chat_service: MagicMock,
    ):
        """Test regenerating an agent message."""
        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id
        mock_message_service.get_with_chat = AsyncMock(return_value=mock_agent_message)
        mock_message_service.delete_subsequent = AsyncMock(return_value=1)

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/regenerate"
        )

        assert response.status_code == 201
        data = response.json()
        assert "newMessageId" in data
        assert "researchSessionId" in data
        mock_chat_service.get_for_user.assert_awaited_once_with(
            chat_id, "test-user-123"
        )
        mock_message_service.delete_subsequent.assert_awaited_once_with(
            chat_id,
            mock_agent_message.created_at,
        )

    def test_regenerate_nonexistent_message(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test regenerating a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()
        mock_message_service.get_with_chat = AsyncMock(return_value=None)

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/regenerate"
        )

        assert response.status_code == 404


class TestSubmitFeedback:
    """Tests for POST /api/v1/chats/{chat_id}/messages/{message_id}/feedback endpoint."""

    def test_submit_positive_feedback(
        self,
        client: TestClient,
        mock_agent_message: Message,
        mock_message_service: MagicMock,
        mock_feedback_service: MagicMock,
    ):
        """Test submitting positive feedback."""
        from datetime import UTC, datetime

        from deep_research.models.message_feedback import FeedbackRating, MessageFeedback

        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id

        # Create mock feedback
        mock_feedback = MagicMock(spec=MessageFeedback)
        mock_feedback.id = uuid4()
        mock_feedback.message_id = message_id
        mock_feedback.rating = FeedbackRating.POSITIVE
        mock_feedback.feedback_text = None
        mock_feedback.feedback_category = None
        mock_feedback.created_at = datetime.now(UTC)

        mock_message_service.get_with_chat = AsyncMock(return_value=mock_agent_message)
        mock_feedback_service.create_feedback = AsyncMock(return_value=mock_feedback)

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/feedback",
            json={"rating": "positive"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["rating"] == "positive"
        assert data["messageId"] == str(message_id)

    def test_submit_negative_feedback_with_report(
        self,
        client: TestClient,
        mock_agent_message: Message,
        mock_message_service: MagicMock,
        mock_feedback_service: MagicMock,
    ):
        """Test submitting negative feedback with error report."""
        from datetime import UTC, datetime

        from deep_research.models.message_feedback import FeedbackRating, MessageFeedback

        chat_id = mock_agent_message.chat_id
        message_id = mock_agent_message.id

        # Create mock feedback
        mock_feedback = MagicMock(spec=MessageFeedback)
        mock_feedback.id = uuid4()
        mock_feedback.message_id = message_id
        mock_feedback.rating = FeedbackRating.NEGATIVE
        mock_feedback.feedback_text = "The dates mentioned are incorrect"
        mock_feedback.feedback_category = None
        mock_feedback.created_at = datetime.now(UTC)

        mock_message_service.get_with_chat = AsyncMock(return_value=mock_agent_message)
        mock_feedback_service.create_feedback = AsyncMock(return_value=mock_feedback)

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
        assert data["feedbackText"] is not None

    def test_submit_feedback_nonexistent_message(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test submitting feedback for a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()
        mock_message_service.get_with_chat = AsyncMock(return_value=None)

        response = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/feedback",
            json={"rating": "positive"},
        )

        assert response.status_code == 404


class TestGetMessageContent:
    """Tests for GET /api/v1/chats/{chat_id}/messages/{message_id}/copy endpoint."""

    def test_get_message_content(
        self,
        client: TestClient,
        mock_message: Message,
        mock_message_service: MagicMock,
    ):
        """Test getting message content for clipboard."""
        chat_id = mock_message.chat_id
        message_id = mock_message.id
        mock_message_service.get_with_chat = AsyncMock(return_value=mock_message)

        response = client.get(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/copy"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == mock_message.content

    def test_get_message_content_nonexistent(
        self,
        client: TestClient,
        mock_message_service: MagicMock,
    ):
        """Test getting content for a message that doesn't exist."""
        chat_id = uuid4()
        message_id = uuid4()
        mock_message_service.get_with_chat = AsyncMock(return_value=None)

        response = client.get(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/copy"
        )

        assert response.status_code == 404
