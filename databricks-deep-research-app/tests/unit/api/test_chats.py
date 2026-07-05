"""Unit tests for Chat API endpoints."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.core.deps import get_chat_service
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity
from deep_research.models.chat import Chat, ChatStatus


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
def mock_chat_service() -> MagicMock:
    """Create a reusable mock chat service."""
    svc = MagicMock()
    svc.list = AsyncMock(return_value=([], 0))
    svc.create = AsyncMock()
    svc.get_for_user = AsyncMock(return_value=None)
    svc.get_by_id = AsyncMock(return_value=None)
    svc.update_chat = AsyncMock(return_value=None)
    svc.soft_delete = AsyncMock(return_value=False)
    svc.restore = AsyncMock(return_value=None)
    # ``commit()`` is awaited at the end of every write endpoint after
    # the legacy hasattr(db, "commit") cleanup was replaced by an
    # explicit service-level commit (PR-1.3). Mocks must satisfy it.
    svc.commit = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def client(mock_user: UserIdentity, mock_chat_service: MagicMock) -> TestClient:
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
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    yield TestClient(app)

    # Clear overrides after test
    app.dependency_overrides.clear()


class TestListChats:
    """Tests for GET /api/v1/chats endpoint."""

    def test_list_chats_empty(self, client: TestClient, mock_chat_service: MagicMock):
        """Test listing chats when none exist."""
        mock_chat_service.list = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/chats")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_chats_with_results(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test listing chats with results."""
        mock_chat_service.list = AsyncMock(return_value=([mock_chat], 1))

        response = client.get("/api/v1/chats")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["items"][0]["id"] == str(mock_chat.id)
        assert "createdAt" in data["items"][0]
        assert "updatedAt" in data["items"][0]
        mock_chat_service.list.assert_awaited_once()
        call_kwargs = mock_chat_service.list.call_args.kwargs
        assert call_kwargs["user_id"] == "test-user-123"
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 0
        assert call_kwargs["status"] is None
        assert call_kwargs["search"] is None

    def test_list_chats_with_pagination(self, client: TestClient, mock_chat_service: MagicMock):
        """Test listing chats with pagination parameters."""
        mock_chat_service.list = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/chats?limit=10&offset=5")

        assert response.status_code == 200
        mock_chat_service.list.assert_awaited_once()
        call_kwargs = mock_chat_service.list.call_args.kwargs
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 5

    def test_list_chats_with_status_filter(self, client: TestClient, mock_chat_service: MagicMock):
        """Test listing chats with status filter."""
        mock_chat_service.list = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/chats?status=archived")

        assert response.status_code == 200
        mock_chat_service.list.assert_awaited_once()
        call_kwargs = mock_chat_service.list.call_args.kwargs
        assert call_kwargs["status"] == ChatStatus.ARCHIVED

    def test_list_chats_with_search(self, client: TestClient, mock_chat_service: MagicMock):
        """Test listing chats with search parameter."""
        mock_chat_service.list = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/chats?search=quantum")

        assert response.status_code == 200
        mock_chat_service.list.assert_awaited_once()
        call_kwargs = mock_chat_service.list.call_args.kwargs
        assert call_kwargs["search"] == "quantum"


class TestCreateChat:
    """Tests for POST /api/v1/chats endpoint."""

    def test_create_chat_with_title(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test creating a chat with a title."""
        mock_chat_service.create = AsyncMock(return_value=mock_chat)

        response = client.post(
            "/api/v1/chats",
            json={"title": "My Research Chat"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        mock_chat_service.create.assert_awaited_once()
        call_kwargs = mock_chat_service.create.call_args.kwargs
        assert call_kwargs["user_id"] == "test-user-123"
        assert call_kwargs["title"] == "My Research Chat"

    def test_create_chat_without_title(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test creating a chat without a title."""
        mock_chat.title = None
        mock_chat_service.create = AsyncMock(return_value=mock_chat)

        response = client.post("/api/v1/chats", json={})

        assert response.status_code == 201

    def test_create_chat_empty_body(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test creating a chat with empty body."""
        mock_chat_service.create = AsyncMock(return_value=mock_chat)

        # No body at all
        response = client.post("/api/v1/chats")

        assert response.status_code == 201


class TestGetChat:
    """Tests for GET /api/v1/chats/{chat_id} endpoint."""

    def test_get_existing_chat(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test getting an existing chat."""
        mock_chat_service.get_for_user = AsyncMock(return_value=mock_chat)

        response = client.get(f"/api/v1/chats/{mock_chat.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_chat.id)
        mock_chat_service.get_for_user.assert_awaited_once_with(mock_chat.id, "test-user-123")

    def test_get_chat_invalid_uuid_returns_422(self, client: TestClient):
        """Invalid chat IDs should fail request validation before hitting the service."""
        response = client.get("/api/v1/chats/not-a-uuid")

        assert response.status_code == 422

    def test_get_nonexistent_chat(self, client: TestClient, mock_chat_service: MagicMock):
        """Test getting a chat that doesn't exist."""
        mock_chat_service.get_for_user = AsyncMock(return_value=None)

        response = client.get(f"/api/v1/chats/{uuid4()}")

        assert response.status_code == 404


class TestUpdateChat:
    """Tests for PATCH /api/v1/chats/{chat_id} endpoint."""

    def test_update_chat_title(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test updating a chat's title."""
        mock_chat.title = "Updated Title"
        mock_chat_service.update_chat = AsyncMock(return_value=mock_chat)

        response = client.patch(
            f"/api/v1/chats/{mock_chat.id}",
            json={"title": "Updated Title"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        call_kwargs = mock_chat_service.update_chat.call_args.kwargs
        assert call_kwargs["chat_id"] == mock_chat.id
        assert call_kwargs["user_id"] == "test-user-123"
        assert call_kwargs["title"] == "Updated Title"

    def test_update_chat_status(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test updating a chat's status."""
        mock_chat.status = ChatStatus.ARCHIVED
        mock_chat_service.update_chat = AsyncMock(return_value=mock_chat)

        response = client.patch(
            f"/api/v1/chats/{mock_chat.id}",
            json={"status": "archived"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "archived"

    def test_update_nonexistent_chat(self, client: TestClient, mock_chat_service: MagicMock):
        """Test updating a chat that doesn't exist."""
        mock_chat_service.update_chat = AsyncMock(return_value=None)

        response = client.patch(
            f"/api/v1/chats/{uuid4()}",
            json={"title": "New Title"},
        )

        assert response.status_code == 404


class TestDeleteChat:
    """Tests for DELETE /api/v1/chats/{chat_id} endpoint."""

    def test_delete_existing_chat(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test soft deleting an existing chat."""
        mock_chat_service.soft_delete = AsyncMock(return_value=True)

        response = client.delete(f"/api/v1/chats/{mock_chat.id}")

        assert response.status_code == 204
        mock_chat_service.soft_delete.assert_awaited_once_with(mock_chat.id, "test-user-123")

    def test_delete_nonexistent_chat(self, client: TestClient, mock_chat_service: MagicMock):
        """Test deleting a chat that doesn't exist."""
        mock_chat_service.soft_delete = AsyncMock(return_value=False)

        response = client.delete(f"/api/v1/chats/{uuid4()}")

        assert response.status_code == 404


class TestRestoreChat:
    """Tests for POST /api/v1/chats/{chat_id}/restore endpoint."""

    def test_restore_deleted_chat(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """Test restoring a soft-deleted chat."""
        mock_chat.status = ChatStatus.ACTIVE
        mock_chat_service.restore = AsyncMock(return_value=mock_chat)

        response = client.post(f"/api/v1/chats/{mock_chat.id}/restore")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        mock_chat_service.restore.assert_awaited_once_with(mock_chat.id, "test-user-123")

    def test_restore_nonexistent_chat(self, client: TestClient, mock_chat_service: MagicMock):
        """Test restoring a chat that doesn't exist."""
        mock_chat_service.restore = AsyncMock(return_value=None)

        response = client.post(f"/api/v1/chats/{uuid4()}/restore")

        assert response.status_code == 404


class TestResearchSessionInlineFromCachedView:
    """Guard the /chats/{id}/full inline-session builder against the cached
    session view's shape.

    Regression: once messages are linked to their research session, get_chat_full
    builds a ResearchSessionInline from the CACHED view (_state_session_to_view),
    which emits query_classification={}, plan={}, plan_iterations=None. Passing
    those straight to the typed sub-models raised a pydantic ValidationError that
    500'd the whole endpoint -> "nothing renders". The builder must coerce them
    and never crash the render.
    """

    @staticmethod
    def _cached_session_view(**overrides):
        """Mirror services/cached/chat._state_session_to_view's output shape."""
        from datetime import UTC, datetime
        from types import SimpleNamespace

        base = {
            "id": uuid4(),
            "message_id": uuid4(),
            "status": "completed",
            "plan": {},
            "observations": {},
            "query_classification": {},
            "verification_data": {"claims": [{"citation_key": "1"}]},
            "current_step_index": 0,
            "plan_iterations": None,
            "started_at": datetime.now(UTC),
            "completed_at": datetime.now(UTC),
            "sources": [],
            "reasoning_steps": [],
            "current_agent": None,
            "research_depth": "auto",
            "query": None,
            "query_mode": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_empty_dicts_coerce_to_none_and_session_survives(self) -> None:
        from deep_research.api.v1.chats import _research_session_inline

        rs = self._cached_session_view()
        result = _research_session_inline(rs, [])

        assert result is not None, "empty plan/classification must not nuke the session"
        assert result.query_classification is None
        assert result.plan is None
        assert result.plan_iterations == 0

    def test_malformed_nonempty_plan_falls_back_but_keeps_session(self) -> None:
        from deep_research.api.v1.chats import _research_session_inline

        # A non-empty but schema-invalid plan/classification must not crash the
        # endpoint; the builder retries without them so the session (and the
        # frontend citation gate keyed on !!researchSession) survives.
        rs = self._cached_session_view(
            plan={"unexpected": "shape"},
            query_classification={"garbage": True},
        )
        result = _research_session_inline(rs, [])

        assert result is not None
        assert result.plan is None
        assert result.query_classification is None


class TestUpdateSurfaceState:
    """Tests for PUT /api/v1/chats/{chat_id}/surface-state."""

    def test_put_surface_state_happy_path(
        self, client: TestClient, mock_chat: Chat, mock_chat_service: MagicMock
    ):
        """A valid surface_state patch returns 200 and {"ok": true}."""
        mock_chat_service.update_chat = AsyncMock(return_value=mock_chat)

        payload = {
            "surface_state": {
                "agent-1": {
                    "data_model": {"counter": 1},
                    "surface_etag": "etag-1",
                }
            }
        }
        response = client.put(f"/api/v1/chats/{mock_chat.id}/surface-state", json=payload)

        assert response.status_code == 200
        assert response.json() == {"ok": True}

        # Verify service was called with the right patch
        call_kwargs = mock_chat_service.update_chat.call_args.kwargs
        assert call_kwargs["chat_id"] == mock_chat.id
        assert call_kwargs["user_id"] == "test-user-123"
        assert call_kwargs["surface_state_patch"] == payload["surface_state"]

    def test_put_surface_state_not_found(
        self, client: TestClient, mock_chat_service: MagicMock
    ):
        """When service returns None the endpoint returns 404."""
        mock_chat_service.update_chat = AsyncMock(return_value=None)

        payload = {"surface_state": {"agent-1": {"data_model": {}}}}
        response = client.put(f"/api/v1/chats/{uuid4()}/surface-state", json=payload)

        assert response.status_code == 404

    def test_put_surface_state_extra_fields_rejected(
        self, client: TestClient, mock_chat_service: MagicMock
    ):
        """Extra top-level fields in the body (extra='forbid') should 422."""
        payload = {
            "surface_state": {"agent-1": {}},
            "unknown_field": "should_fail",
        }
        response = client.put(f"/api/v1/chats/{uuid4()}/surface-state", json=payload)

        assert response.status_code == 422

    def test_put_surface_state_non_dict_value_rejected(
        self, client: TestClient, mock_chat_service: MagicMock
    ):
        """Agent entry that is not a dict must fail validator (422)."""
        payload = {"surface_state": {"agent-1": "not-a-dict"}}
        response = client.put(f"/api/v1/chats/{uuid4()}/surface-state", json=payload)

        assert response.status_code == 422


class TestGetChatFullSurfaceState:
    """Tests that GET /chats/{id}/full returns surface_state from metadata."""

    @staticmethod
    def _make_chat_with_metadata(chat_id, surface_state_val=None):
        """Build a Chat ORM-like object with metadata_ containing surface_state."""
        from datetime import UTC, datetime
        from types import SimpleNamespace

        return SimpleNamespace(
            id=chat_id,
            user_id="test-user-123",
            title="Chat With Surface",
            status=ChatStatus.ACTIVE,
            chat_type=None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            deleted_at=None,
            incognito_session_id=None,
            messages=[],
            metadata_={"surface_state": surface_state_val} if surface_state_val else {},
        )

    def test_get_full_returns_surface_state(
        self, client: TestClient, mock_chat_service: MagicMock
    ):
        """surface_state in metadata_ is included in the /full response."""
        chat_id = uuid4()
        surface = {"agent-1": {"data_model": {"v": 42}, "surface_etag": "etag-1"}}
        fake_chat = self._make_chat_with_metadata(chat_id, surface_state_val=surface)
        mock_chat_service.get_full = AsyncMock(return_value=fake_chat)

        response = client.get(f"/api/v1/chats/{chat_id}/full")

        assert response.status_code == 200
        data = response.json()
        assert data["surfaceState"] == {
            "agent-1": {"data_model": {"v": 42}, "surface_etag": "etag-1"}
        }

    def test_get_full_returns_null_surface_state_when_absent(
        self, client: TestClient, mock_chat_service: MagicMock
    ):
        """When no surface_state key in metadata_, surfaceState is null."""
        chat_id = uuid4()
        fake_chat = self._make_chat_with_metadata(chat_id, surface_state_val=None)
        mock_chat_service.get_full = AsyncMock(return_value=fake_chat)

        response = client.get(f"/api/v1/chats/{chat_id}/full")

        assert response.status_code == 200
        data = response.json()
        assert data["surfaceState"] is None
