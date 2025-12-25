"""Unit test fixtures for mocking external dependencies.

These fixtures provide mocked versions of:
- AsyncSession (SQLAlchemy database sessions)
- LLMClient (Databricks LLM endpoints)
- BraveSearchClient (Web search)
- WebCrawler (HTTP content fetching)
- UserIdentity (Authentication)
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import UserIdentity
from src.models.chat import Chat, ChatStatus
from src.models.message import Message, MessageRole

# ---------------------------------------------------------------------------
# Database Session Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mocked AsyncSession for unit tests.

    Returns:
        Mocked AsyncSession with common methods configured.
    """
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_scalar_result() -> MagicMock:
    """Create a mock result that returns a scalar value.

    Returns:
        Mock result object with scalar methods.
    """
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    result.scalar = MagicMock(return_value=0)
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return result


# ---------------------------------------------------------------------------
# Model Factories
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_factory():
    """Factory for creating Chat model instances.

    Returns:
        Function that creates Chat instances with optional overrides.
    """

    def _create_chat(
        id: UUID | None = None,
        user_id: str = "test-user-123",
        title: str | None = "Test Chat",
        status: ChatStatus = ChatStatus.ACTIVE,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        deleted_at: datetime | None = None,
    ) -> Chat:
        chat = Chat(
            user_id=user_id,
            title=title,
            status=status,
        )
        # Override auto-generated fields
        if id is not None:
            chat.id = id
        if created_at is not None:
            chat.created_at = created_at
        if updated_at is not None:
            chat.updated_at = updated_at
        if deleted_at is not None:
            chat.deleted_at = deleted_at
        return chat

    return _create_chat


@pytest.fixture
def message_factory():
    """Factory for creating Message model instances.

    Returns:
        Function that creates Message instances with optional overrides.

    Note:
        The Message model uses a relationship to ResearchSession, not a
        direct foreign key column. Research session association is done
        via the set_research_session service method.
    """

    def _create_message(
        id: UUID | None = None,
        chat_id: UUID | None = None,
        role: MessageRole = MessageRole.USER,
        content: str = "Test message content",
        is_edited: bool = False,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> Message:
        msg = Message(
            chat_id=chat_id or uuid4(),
            role=role,
            content=content,
        )
        # Override auto-generated fields
        if id is not None:
            msg.id = id
        msg.is_edited = is_edited
        if created_at is not None:
            msg.created_at = created_at
        if updated_at is not None:
            msg.updated_at = updated_at
        return msg

    return _create_message


# ---------------------------------------------------------------------------
# Authentication Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user() -> UserIdentity:
    """Create a mock user identity for testing.

    Returns:
        UserIdentity instance for test user.
    """
    return UserIdentity(
        user_id="test-user-123",
        email="test@example.com",
        display_name="Test User",
    )


@pytest.fixture
def mock_anonymous_user() -> UserIdentity:
    """Create an anonymous user identity for testing.

    Returns:
        Anonymous UserIdentity instance.
    """
    return UserIdentity.anonymous()


# ---------------------------------------------------------------------------
# LLM Client Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mocked LLMClient for unit tests.

    Returns:
        Mocked LLMClient with generate and close methods.
    """
    client = AsyncMock()
    client.generate = AsyncMock(return_value="Mocked LLM response")
    client.generate_structured = AsyncMock(return_value={"key": "value"})
    client.stream = AsyncMock()
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Search and Crawler Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_brave_client() -> AsyncMock:
    """Create a mocked BraveSearchClient for unit tests.

    Returns:
        Mocked BraveSearchClient with search method.
    """
    client = AsyncMock()
    client.search = AsyncMock(
        return_value={
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com/test",
                        "description": "A test search result",
                    }
                ]
            }
        }
    )
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_web_crawler() -> AsyncMock:
    """Create a mocked WebCrawler for unit tests.

    Returns:
        Mocked WebCrawler with fetch method.
    """
    crawler = AsyncMock()
    crawler.fetch = AsyncMock(
        return_value={
            "url": "https://example.com/test",
            "title": "Test Page",
            "content": "This is the page content for testing.",
            "success": True,
        }
    )
    crawler.close = AsyncMock()
    return crawler


# ---------------------------------------------------------------------------
# HTTP Client Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response.

    Returns:
        Mock response object with common attributes.
    """
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(return_value={"data": "test"})
    response.text = "Test response body"
    response.headers = {"Content-Type": "application/json"}
    return response


@pytest.fixture
def mock_httpx_client(mock_http_response: MagicMock) -> AsyncMock:
    """Create a mocked httpx.AsyncClient for unit tests.

    Args:
        mock_http_response: Mock response to return.

    Returns:
        Mocked AsyncClient with get/post methods.
    """
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_http_response)
    client.post = AsyncMock(return_value=mock_http_response)
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# FastAPI TestClient Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def override_get_db(mock_db_session: AsyncMock):
    """Create a dependency override for get_db.

    Args:
        mock_db_session: Mocked database session.

    Returns:
        Function that returns the mock session.
    """

    async def _override():
        yield mock_db_session

    return _override


@pytest.fixture
def override_get_current_user(mock_user: UserIdentity):
    """Create a dependency override for get_current_user_identity.

    Args:
        mock_user: Mock user identity.

    Returns:
        Function that returns the mock user.
    """

    async def _override():
        return mock_user

    return _override
