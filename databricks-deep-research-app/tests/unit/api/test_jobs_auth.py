"""IDOR regression tests for Jobs API authorization gate.

Verifies that submit_job enforces chat ownership via verify_chat_access
before proceeding with any business logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.core.exceptions import NotFoundError
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity


@pytest.fixture
def mock_user() -> UserIdentity:
    return UserIdentity(user_id="user-A", email="a@test.com", display_name="User A")


@pytest.fixture
def client(mock_user: UserIdentity) -> TestClient:
    async def override_get_db():
        yield AsyncMock()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_identity] = lambda: mock_user

    # submit_job accesses app.state attributes before verify_chat_access
    app.state.llm_client = MagicMock()
    app.state.brave_client = MagicMock()
    app.state.web_crawler = MagicMock()

    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _mock_job_manager():
    """Patch get_job_manager so it doesn't require real infrastructure."""
    with patch(
        "deep_research.api.v1.jobs.get_job_manager",
        return_value=MagicMock(),
    ):
        yield


class TestSubmitJobIDOR:
    """Verify that submit_job enforces chat ownership before proceeding."""

    def test_rejects_other_users_chat(self, client: TestClient) -> None:
        """Submitting a job for another user's chat must return 404."""
        chat_id = uuid4()
        with patch(
            "deep_research.api.v1.jobs.verify_chat_access",
            new_callable=AsyncMock,
            side_effect=NotFoundError("Chat", str(chat_id)),
        ):
            resp = client.post(
                "/api/v1/research/jobs",
                json={"chat_id": str(chat_id), "query": "test"},
            )
        assert resp.status_code == 404

    def test_verify_called_with_correct_args(self, client: TestClient) -> None:
        """verify_chat_access must be called with (chat_id, user_id)."""
        chat_id = uuid4()
        with patch(
            "deep_research.api.v1.jobs.verify_chat_access",
            new_callable=AsyncMock,
            side_effect=NotFoundError("Chat", str(chat_id)),
        ) as mock_verify:
            client.post(
                "/api/v1/research/jobs",
                json={"chat_id": str(chat_id), "query": "test"},
            )
            mock_verify.assert_awaited_once()
            call_args = mock_verify.await_args
            assert call_args.args[0] == chat_id
            assert call_args.args[1] == "user-A"
