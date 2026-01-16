"""Unit tests for Chat API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.core.auth import UserIdentity
from src.db.session import get_db
from src.main import app
from src.middleware.auth import get_current_user_identity
from src.models.chat import Chat, ChatStatus


@pytest.fixture
def mock_user() -> UserIdentity:
    """Create a test user identity."""
    return UserIdentity(
        user_id="test-user-123",
        email="test@example.com",
        display_name="Test User",
    )


@pytest.fixture
def mock_chat() -> Chat:
    """Create a mock chat for testing."""
    from datetime import UTC, datetime

    chat = Chat(user_id="test-user-123", title="Test Chat")
    chat.id = uuid4()
    chat.status = ChatStatus.ACTIVE
    chat.created_at = datetime.now(UTC)
    chat.updated_at = datetime.now(UTC)
    return chat


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


class TestListChats:
    """Tests for GET /api/v1/chats endpoint."""

    def test_list_chats_empty(self, client: TestClient):
        """Test listing chats when none exist."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.list = AsyncMock(return_value=([], 0))
            MockService.return_value = mock_service

            response = client.get("/api/v1/chats")

            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0

    def test_list_chats_with_results(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test listing chats with results."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.list = AsyncMock(return_value=([mock_chat], 1))
            MockService.return_value = mock_service

            response = client.get("/api/v1/chats")

            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 1
            assert data["total"] == 1

    def test_list_chats_with_pagination(self, client: TestClient):
        """Test listing chats with pagination parameters."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.list = AsyncMock(return_value=([], 0))
            MockService.return_value = mock_service

            response = client.get("/api/v1/chats?limit=10&offset=5")

            assert response.status_code == 200
            mock_service.list.assert_awaited_once()
            call_kwargs = mock_service.list.call_args.kwargs
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 5

    def test_list_chats_with_status_filter(self, client: TestClient):
        """Test listing chats with status filter."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.list = AsyncMock(return_value=([], 0))
            MockService.return_value = mock_service

            response = client.get("/api/v1/chats?status=archived")

            assert response.status_code == 200
            mock_service.list.assert_awaited_once()
            call_kwargs = mock_service.list.call_args.kwargs
            assert call_kwargs["status"] == ChatStatus.ARCHIVED

    def test_list_chats_with_search(self, client: TestClient):
        """Test listing chats with search parameter."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.list = AsyncMock(return_value=([], 0))
            MockService.return_value = mock_service

            response = client.get("/api/v1/chats?search=quantum")

            assert response.status_code == 200
            mock_service.list.assert_awaited_once()
            call_kwargs = mock_service.list.call_args.kwargs
            assert call_kwargs["search"] == "quantum"


class TestCreateChat:
    """Tests for POST /api/v1/chats endpoint."""

    def test_create_chat_with_title(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test creating a chat with a title."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.create = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            response = client.post(
                "/api/v1/chats",
                json={"title": "My Research Chat"},
            )

            assert response.status_code == 201
            data = response.json()
            assert "id" in data
            mock_service.create.assert_awaited_once()

    def test_create_chat_without_title(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test creating a chat without a title."""
        mock_chat.title = None

        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.create = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            response = client.post("/api/v1/chats", json={})

            assert response.status_code == 201

    def test_create_chat_empty_body(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test creating a chat with empty body."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.create = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            # No body at all
            response = client.post("/api/v1/chats")

            assert response.status_code == 201


class TestGetChat:
    """Tests for GET /api/v1/chats/{chat_id} endpoint."""

    def test_get_existing_chat(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test getting an existing chat."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.get_for_user = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            response = client.get(f"/api/v1/chats/{mock_chat.id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(mock_chat.id)

    def test_get_nonexistent_chat(self, client: TestClient):
        """Test getting a chat that doesn't exist."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.get_for_user = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.get(f"/api/v1/chats/{uuid4()}")

            assert response.status_code == 404


class TestUpdateChat:
    """Tests for PATCH /api/v1/chats/{chat_id} endpoint."""

    def test_update_chat_title(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test updating a chat's title."""
        mock_chat.title = "Updated Title"

        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.update_chat = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            response = client.patch(
                f"/api/v1/chats/{mock_chat.id}",
                json={"title": "Updated Title"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Updated Title"

    def test_update_chat_status(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test updating a chat's status."""
        mock_chat.status = ChatStatus.ARCHIVED

        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.update_chat = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            response = client.patch(
                f"/api/v1/chats/{mock_chat.id}",
                json={"status": "archived"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "archived"

    def test_update_nonexistent_chat(self, client: TestClient):
        """Test updating a chat that doesn't exist."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.update_chat = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.patch(
                f"/api/v1/chats/{uuid4()}",
                json={"title": "New Title"},
            )

            assert response.status_code == 404


class TestDeleteChat:
    """Tests for DELETE /api/v1/chats/{chat_id} endpoint."""

    def test_delete_existing_chat(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test soft deleting an existing chat."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.soft_delete = AsyncMock(return_value=True)
            MockService.return_value = mock_service

            response = client.delete(f"/api/v1/chats/{mock_chat.id}")

            assert response.status_code == 204

    def test_delete_nonexistent_chat(self, client: TestClient):
        """Test deleting a chat that doesn't exist."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.soft_delete = AsyncMock(return_value=False)
            MockService.return_value = mock_service

            response = client.delete(f"/api/v1/chats/{uuid4()}")

            assert response.status_code == 404


class TestRestoreChat:
    """Tests for POST /api/v1/chats/{chat_id}/restore endpoint."""

    def test_restore_deleted_chat(
        self, client: TestClient, mock_chat: Chat
    ):
        """Test restoring a soft-deleted chat."""
        mock_chat.status = ChatStatus.ACTIVE

        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.restore = AsyncMock(return_value=mock_chat)
            MockService.return_value = mock_service

            response = client.post(f"/api/v1/chats/{mock_chat.id}/restore")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "active"

    def test_restore_nonexistent_chat(self, client: TestClient):
        """Test restoring a chat that doesn't exist."""
        with patch("src.api.v1.chats.ChatService") as MockService:
            mock_service = MagicMock()
            mock_service.restore = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            response = client.post(f"/api/v1/chats/{uuid4()}/restore")

            assert response.status_code == 404
