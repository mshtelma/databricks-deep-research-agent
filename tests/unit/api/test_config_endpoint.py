"""Unit tests for GET /api/v1/config/model-catalog endpoint (T016).

Verifies the endpoint returns categories and endpoints dicts matching
the loaded AppConfig.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity


@pytest.fixture
def mock_user() -> UserIdentity:
    return UserIdentity(
        user_id="test-user-123",
        email="test@example.com",
        display_name="Test User",
    )


@pytest.fixture
def mock_app_config() -> MagicMock:
    """Build a mock AppConfig with 2 tiers and 3 endpoints."""
    config = MagicMock()

    # Endpoints
    ep_haiku = MagicMock()
    ep_haiku.endpoint_identifier = "haiku-ep-id"
    ep_haiku.max_context_window = 200000
    ep_haiku.supports_structured_output = True

    ep_opus = MagicMock()
    ep_opus.endpoint_identifier = "opus-ep-id"
    ep_opus.max_context_window = 200000
    ep_opus.supports_structured_output = True

    ep_llama = MagicMock()
    ep_llama.endpoint_identifier = "llama-70b-ep-id"
    ep_llama.max_context_window = 128000
    ep_llama.supports_structured_output = False

    config.endpoints = {
        "databricks-haiku": ep_haiku,
        "databricks-opus": ep_opus,
        "databricks-llama-70b": ep_llama,
    }

    # Model roles (tiers)
    tier_analytical = MagicMock()
    tier_analytical.endpoints = ["databricks-llama-70b", "databricks-haiku"]
    tier_analytical.temperature = 0.7
    tier_analytical.max_tokens = 4096

    tier_complex = MagicMock()
    tier_complex.endpoints = ["databricks-opus"]
    tier_complex.temperature = 0.5
    tier_complex.max_tokens = 8192

    config.models = {
        "analytical": tier_analytical,
        "complex": tier_complex,
    }

    return config


@pytest.fixture
def client(mock_user: UserIdentity) -> TestClient:
    from unittest.mock import AsyncMock

    async def override_get_db():
        mock_session = AsyncMock()
        yield mock_session

    async def override_get_current_user_identity():
        return mock_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_identity] = override_get_current_user_identity

    yield TestClient(app)

    app.dependency_overrides.clear()


class TestModelCatalogEndpoint:
    """Tests for GET /api/v1/config/model-catalog."""

    @patch("deep_research.api.v1.config.get_app_config")
    def test_returns_categories_and_endpoints(
        self,
        mock_get_config: MagicMock,
        mock_app_config: MagicMock,
        client: TestClient,
    ) -> None:
        """Should return categories and endpoints matching AppConfig."""
        mock_get_config.return_value = mock_app_config

        response = client.get("/api/v1/config/model-catalog")

        assert response.status_code == 200
        data = response.json()

        # Check top-level keys
        assert "categories" in data
        assert "endpoints" in data

        # Verify categories (camelCase serialization from BaseSchema)
        assert len(data["categories"]) == 2
        assert "analytical" in data["categories"]
        assert "complex" in data["categories"]

        analytical = data["categories"]["analytical"]
        assert analytical["name"] == "analytical"
        assert analytical["defaultEndpoints"] == ["databricks-llama-70b", "databricks-haiku"]
        assert analytical["temperature"] == 0.7
        assert analytical["maxTokens"] == 4096

        complex_cat = data["categories"]["complex"]
        assert complex_cat["name"] == "complex"
        assert complex_cat["defaultEndpoints"] == ["databricks-opus"]

        # Verify endpoints (camelCase serialization)
        assert len(data["endpoints"]) == 3
        assert "databricks-haiku" in data["endpoints"]
        assert "databricks-opus" in data["endpoints"]
        assert "databricks-llama-70b" in data["endpoints"]

        haiku = data["endpoints"]["databricks-haiku"]
        assert haiku["name"] == "databricks-haiku"
        assert haiku["endpointIdentifier"] == "haiku-ep-id"
        assert haiku["maxContextWindow"] == 200000
        assert haiku["supportsStructuredOutput"] is True

        llama = data["endpoints"]["databricks-llama-70b"]
        assert llama["supportsStructuredOutput"] is False

    @patch("deep_research.api.v1.config.get_app_config")
    def test_empty_config(
        self,
        mock_get_config: MagicMock,
        client: TestClient,
    ) -> None:
        """Should handle empty config gracefully."""
        mock_config = MagicMock()
        mock_config.models = {}
        mock_config.endpoints = {}
        mock_get_config.return_value = mock_config

        response = client.get("/api/v1/config/model-catalog")

        assert response.status_code == 200
        data = response.json()
        assert data["categories"] == {}
        assert data["endpoints"] == {}
