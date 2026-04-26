"""Unit tests for Data Source API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.core.deps import get_data_source_service
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity
from deep_research.models.data_source import (
    DataSourceType,
    DataSourceValidationStatus,
    DataSourceVisibility,
    UserDataSource,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user() -> UserIdentity:
    """Create a test user identity."""
    return UserIdentity(
        user_id="test-user-123",
        email="test@example.com",
        display_name="Test User",
    )


def _make_source(
    *,
    owner_id: str = "test-user-123",
    source_type: str = DataSourceType.VECTOR_SEARCH.value,
    name: str = "Test VS Source",
    endpoint_identifier: str = "catalog.schema.index",
    config: dict | None = None,
    visibility: str = DataSourceVisibility.PRIVATE.value,
    validation_status: str = DataSourceValidationStatus.VALID.value,
) -> UserDataSource:
    """Create a mock UserDataSource for testing."""
    source = UserDataSource(
        owner_id=owner_id,
        type=source_type,
        name=name,
        description="Test description",
        endpoint_identifier=endpoint_identifier,
        config=config or {
            "endpoint_name": "ep-1",
            "index_name": "catalog.schema.index",
            "columns": ["col1", "col2"],
            "columns_to_rerank": ["col1"],
            "enable_hybrid": True,
            "enable_reranking": True,
            "num_results": 10,
        },
        visibility=visibility,
        validation_status=validation_status,
    )
    source.id = uuid4()
    source.created_at = datetime.now(UTC)
    source.updated_at = datetime.now(UTC)
    source.last_validated_at = datetime.now(UTC)
    return source


@pytest.fixture
def mock_source() -> UserDataSource:
    """Default mock Vector Search source."""
    return _make_source()


@pytest.fixture
def mock_data_source_service() -> MagicMock:
    """Mock `IDataSourceService` injected via FastAPI dependency override.

    F-DS migrated the route handlers from `DataSourceService(db)` to
    `Depends(get_data_source_service)`, so tests no longer patch the class
    directly — they override the dep.
    """
    return MagicMock()


@pytest.fixture
def client(
    mock_user: UserIdentity,
    mock_data_source_service: MagicMock,
) -> TestClient:
    """Create test client with mocked dependencies."""

    async def override_get_db():
        mock_session = AsyncMock()
        yield mock_session

    async def override_get_current_user_identity():
        return mock_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_identity] = (
        override_get_current_user_identity
    )
    app.dependency_overrides[get_data_source_service] = (
        lambda: mock_data_source_service
    )

    yield TestClient(app)

    app.dependency_overrides.clear()


# =========================================================================
# TestHelpers
# =========================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_infer_capabilities_vector_search(self) -> None:
        """Vector Search returns semantic + keyword + metadata filtering."""
        from deep_research.api.v1.data_sources import _infer_capabilities
        from deep_research.schemas.data_source import DataSourceCapability

        caps = _infer_capabilities(DataSourceType.VECTOR_SEARCH)

        assert DataSourceCapability.SEMANTIC_SEARCH in caps
        assert DataSourceCapability.KEYWORD_SEARCH in caps
        assert DataSourceCapability.METADATA_FILTERING in caps

    def test_infer_capabilities_genie(self) -> None:
        """Genie returns SQL analytics + aggregations + follow-up."""
        from deep_research.api.v1.data_sources import _infer_capabilities
        from deep_research.schemas.data_source import DataSourceCapability

        caps = _infer_capabilities(DataSourceType.GENIE)

        assert DataSourceCapability.SQL_ANALYTICS in caps
        assert DataSourceCapability.AGGREGATIONS in caps
        assert DataSourceCapability.FOLLOW_UP in caps

    def test_infer_capabilities_knowledge_assistant(self) -> None:
        """Knowledge Assistant returns domain expertise."""
        from deep_research.api.v1.data_sources import _infer_capabilities
        from deep_research.schemas.data_source import DataSourceCapability

        caps = _infer_capabilities(DataSourceType.KNOWLEDGE_ASSISTANT)

        assert DataSourceCapability.DOMAIN_EXPERTISE in caps


# =========================================================================
# TestCreateEndpoints
# =========================================================================


class TestCreateEndpoints:
    """Tests for POST /api/v1/data-sources/* creation endpoints."""

    def test_create_vs_missing_obo_token_returns_403(
        self, client: TestClient
    ) -> None:
        """Missing OBO token returns 403 PermissionDenied."""
        with patch(
            "deep_research.api.v1.data_sources._get_obo_token", return_value=None
        ):
            response = client.post(
                "/api/v1/data-sources/vector-search",
                json={
                    "name": "Test VS",
                    "endpoint_name": "ep-1",
                    "index_name": "catalog.schema.index",
                },
            )

        assert response.status_code == 403

    def test_create_vs_obo_validation_failure_returns_400(
        self, client: TestClient, mock_data_source_service: MagicMock,
    ) -> None:
        """OBO validation failure returns 400 (ValidationError)."""
        mock_data_source_service.create_vector_search_source = AsyncMock(
            return_value=(None, "Index not accessible")
        )
        with (
            patch(
                "deep_research.api.v1.data_sources._get_obo_token",
                return_value="test-obo-token",
            ),
            patch("deep_research.api.v1.data_sources.OBODatabricksClient"),
        ):
            response = client.post(
                "/api/v1/data-sources/vector-search",
                json={
                    "name": "Test VS",
                    "endpoint_name": "ep-1",
                    "index_name": "catalog.schema.index",
                },
            )

        assert response.status_code == 400

    def test_create_vs_success_returns_201(
        self,
        client: TestClient,
        mock_source: UserDataSource,
        mock_data_source_service: MagicMock,
    ) -> None:
        """Successful creation returns 201 with source data."""
        mock_data_source_service.create_vector_search_source = AsyncMock(
            return_value=(mock_source, None)
        )
        with (
            patch(
                "deep_research.api.v1.data_sources._get_obo_token",
                return_value="test-obo-token",
            ),
            patch("deep_research.api.v1.data_sources.OBODatabricksClient"),
        ):
            response = client.post(
                "/api/v1/data-sources/vector-search",
                json={
                    "name": "Test VS",
                    "endpoint_name": "ep-1",
                    "index_name": "catalog.schema.index",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == mock_source.name
        assert data["type"] == "vector_search"
        assert "id" in data
        mock_data_source_service.create_vector_search_source.assert_awaited_once()
        call_kwargs = (
            mock_data_source_service.create_vector_search_source.call_args.kwargs
        )
        assert call_kwargs["owner_id"] == "test-user-123"
        assert call_kwargs["user_token"] == "test-obo-token"
        assert call_kwargs["name"] == "Test VS"
        assert call_kwargs["endpoint_name"] == "ep-1"
        assert call_kwargs["index_name"] == "catalog.schema.index"

    def test_create_genie_missing_obo_returns_403(
        self, client: TestClient
    ) -> None:
        """Missing OBO token on Genie create returns 403."""
        with patch(
            "deep_research.api.v1.data_sources._get_obo_token", return_value=None
        ):
            response = client.post(
                "/api/v1/data-sources/genie",
                json={"name": "Test Genie", "space_id": "space-123"},
            )

        assert response.status_code == 403


# =========================================================================
# TestReadEndpoints
# =========================================================================


class TestReadEndpoints:
    """Tests for GET /api/v1/data-sources endpoints."""

    def test_list_returns_sources_with_counts(
        self,
        client: TestClient,
        mock_source: UserDataSource,
        mock_data_source_service: MagicMock,
    ) -> None:
        """List endpoint returns sources with user/workspace counts."""
        mock_data_source_service.get_accessible_sources = AsyncMock(
            return_value=([mock_source], 1)
        )

        response = client.get("/api/v1/data-sources")

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 1
        assert data["total"] == 1
        assert "user_sources" in data
        assert "workspace_sources" in data
        assert data["sources"][0]["id"] == str(mock_source.id)
        assert data["sources"][0]["type"] == "vector_search"
        mock_data_source_service.get_accessible_sources.assert_awaited_once_with(
            user_id="test-user-123",
            source_type=None,
            only_valid=True,
            limit=100,
            offset=0,
        )

    def test_get_not_found_returns_404(
        self, client: TestClient, mock_data_source_service: MagicMock,
    ) -> None:
        """Getting nonexistent source returns 404."""
        mock_data_source_service.get_accessible = AsyncMock(return_value=None)

        response = client.get(f"/api/v1/data-sources/{uuid4()}")

        assert response.status_code == 404

    def test_get_success_returns_source(
        self,
        client: TestClient,
        mock_source: UserDataSource,
        mock_data_source_service: MagicMock,
    ) -> None:
        """Getting accessible source returns 200 with all fields."""
        mock_data_source_service.get_accessible = AsyncMock(return_value=mock_source)

        response = client.get(f"/api/v1/data-sources/{mock_source.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_source.id)
        assert data["name"] == mock_source.name
        assert data["type"] == "vector_search"
        assert "config" in data
        assert "capabilities" in data
        mock_data_source_service.get_accessible.assert_awaited_once_with(
            mock_source.id,
            "test-user-123",
        )

    def test_get_invalid_uuid_returns_422(self, client: TestClient) -> None:
        """Invalid source IDs should be rejected before service wiring."""
        response = client.get("/api/v1/data-sources/not-a-uuid")

        assert response.status_code == 422


# =========================================================================
# TestMutationEndpoints
# =========================================================================


class TestMutationEndpoints:
    """Tests for PATCH and DELETE endpoints."""

    def test_update_not_owner_returns_404(
        self, client: TestClient, mock_data_source_service: MagicMock,
    ) -> None:
        """Updating a source you don't own returns 404."""
        mock_data_source_service.get_for_user = AsyncMock(return_value=None)

        response = client.patch(
            f"/api/v1/data-sources/{uuid4()}",
            json={"name": "New Name"},
        )

        assert response.status_code == 404

    def test_delete_not_owner_returns_404(
        self, client: TestClient, mock_data_source_service: MagicMock,
    ) -> None:
        """Deleting a source you don't own returns 404."""
        mock_data_source_service.get_for_user = AsyncMock(return_value=None)

        response = client.delete(f"/api/v1/data-sources/{uuid4()}")

        assert response.status_code == 404

    def test_update_success_partial_fields(
        self,
        client: TestClient,
        mock_source: UserDataSource,
        mock_data_source_service: MagicMock,
    ) -> None:
        """Partial update applies only provided fields."""
        mock_data_source_service.get_for_user = AsyncMock(return_value=mock_source)
        mock_data_source_service.update = AsyncMock(return_value=mock_source)

        response = client.patch(
            f"/api/v1/data-sources/{mock_source.id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        # The source name should have been updated in the mock
        assert mock_source.name == "Updated Name"
        mock_data_source_service.get_for_user.assert_awaited_once_with(
            mock_source.id,
            "test-user-123",
        )
        mock_data_source_service.update.assert_awaited_once_with(mock_source)


# =========================================================================
# TestQueryConfigEndpoints
# =========================================================================


class TestQueryConfigEndpoints:
    """Tests for query config GET/PUT endpoints."""

    def test_get_query_config_non_vs_returns_400(
        self, client: TestClient, mock_data_source_service: MagicMock,
    ) -> None:
        """Query config on non-VS source returns 400 (ValidationError)."""
        genie_source = _make_source(
            source_type=DataSourceType.GENIE.value,
            config={"space_id": "space-123", "example_questions": []},
        )

        mock_data_source_service.get_accessible = AsyncMock(return_value=genie_source)

        response = client.get(
            f"/api/v1/data-sources/{genie_source.id}/query-config"
        )

        assert response.status_code == 400
        data = response.json()
        assert "Vector Search" in data["message"]

    def test_get_query_config_success(
        self,
        client: TestClient,
        mock_source: UserDataSource,
        mock_data_source_service: MagicMock,
    ) -> None:
        """Successfully gets query config for VS source."""
        mock_data_source_service.get_accessible = AsyncMock(return_value=mock_source)

        response = client.get(
            f"/api/v1/data-sources/{mock_source.id}/query-config"
        )

        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert data["source_id"] == str(mock_source.id)
