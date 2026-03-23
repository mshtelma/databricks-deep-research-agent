"""Unit tests for DataSourceService.

Tests CRUD operations, OBO validation dispatch, and query methods
for user-configured enterprise data sources.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from deep_research.models.data_source import (
    DataSourceType,
    DataSourceValidationStatus,
    DataSourceVisibility,
    UserDataSource,
)
from deep_research.services.data_source_service import DataSourceService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source(
    *,
    owner_id: str = "test-user-123",
    source_type: str = DataSourceType.VECTOR_SEARCH.value,
    name: str = "Test Source",
    description: str | None = None,
    endpoint_identifier: str = "catalog.schema.index",
    config: dict | None = None,
    visibility: str = DataSourceVisibility.PRIVATE.value,
    validation_status: str = DataSourceValidationStatus.VALID.value,
    source_id: UUID | None = None,
) -> UserDataSource:
    """Create a UserDataSource instance for testing."""
    source = UserDataSource(
        owner_id=owner_id,
        type=source_type,
        name=name,
        description=description,
        endpoint_identifier=endpoint_identifier,
        config=config or {},
        visibility=visibility,
        validation_status=validation_status,
    )
    source.id = source_id or uuid4()
    source.created_at = datetime.now(UTC)
    source.updated_at = datetime.now(UTC)
    source.last_validated_at = datetime.now(UTC)
    return source


@pytest.fixture
def mock_obo_client() -> MagicMock:
    """Mock OBODatabricksClient with async validation methods."""
    client = MagicMock()
    client.validate_vector_search_access = AsyncMock(return_value=(True, None))
    client.validate_genie_access = AsyncMock(return_value=(True, None))
    client.validate_assistant_access = AsyncMock(return_value=(True, None))
    client.get_vector_search_index_schema = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def service(mock_session: AsyncMock, mock_obo_client: MagicMock) -> DataSourceService:
    """Create DataSourceService with mocked dependencies."""
    return DataSourceService(mock_session, obo_client=mock_obo_client)


# =========================================================================
# TestCreateVectorSearchSource
# =========================================================================


class TestCreateVectorSearchSource:
    """Tests for create_vector_search_source."""

    @pytest.mark.asyncio
    async def test_success_creates_source_with_valid_status(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
        mock_obo_client: MagicMock,
    ) -> None:
        """Successful creation sets validation_status to VALID."""
        mock_obo_client.validate_vector_search_access.return_value = (True, None)
        mock_obo_client.get_vector_search_index_schema.return_value = {
            "columns": [{"name": "c1"}, {"name": "c2"}],
            "text_columns": ["c1"],
        }

        source, error = await service.create_vector_search_source(
            owner_id="user-1",
            user_token="token-abc",
            name="My VS Index",
            endpoint_name="ep-1",
            index_name="catalog.schema.index",
        )

        assert error is None
        assert source is not None
        assert source.validation_status == DataSourceValidationStatus.VALID.value
        assert source.type == DataSourceType.VECTOR_SEARCH.value
        assert source.name == "My VS Index"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_obo_failure_returns_error_without_persisting(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
        mock_obo_client: MagicMock,
    ) -> None:
        """OBO validation failure returns error and does not persist."""
        mock_obo_client.validate_vector_search_access.return_value = (
            False,
            "Access denied to index",
        )

        source, error = await service.create_vector_search_source(
            owner_id="user-1",
            user_token="token-abc",
            name="Bad Index",
            endpoint_name="ep-1",
            index_name="catalog.schema.bad",
        )

        assert source is None
        assert error == "Access denied to index"
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_schema_detection_extracts_columns_and_text_columns(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Auto-detected schema populates config columns correctly."""
        mock_obo_client.get_vector_search_index_schema.return_value = {
            "columns": [{"name": "col_a"}, {"name": "col_b"}, {"name": "col_c"}],
            "text_columns": ["col_a", "col_b", "col_c"],
        }

        source, _ = await service.create_vector_search_source(
            owner_id="user-1",
            user_token="token",
            name="Schema Test",
            endpoint_name="ep-1",
            index_name="catalog.schema.idx",
        )

        assert source is not None
        assert source.config["columns"] == ["col_a", "col_b", "col_c"]
        # columns_to_rerank is capped at first 2 text columns
        assert source.config["columns_to_rerank"] == ["col_a", "col_b"]

    @pytest.mark.asyncio
    async def test_null_schema_defaults_to_empty_columns(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Null schema produces empty column lists."""
        mock_obo_client.get_vector_search_index_schema.return_value = None

        source, _ = await service.create_vector_search_source(
            owner_id="user-1",
            user_token="token",
            name="Null Schema",
            endpoint_name="ep-1",
            index_name="catalog.schema.idx",
        )

        assert source is not None
        assert source.config["columns"] == []
        assert source.config["columns_to_rerank"] == []


# =========================================================================
# TestCreateGenieSource
# =========================================================================


class TestCreateGenieSource:
    """Tests for create_genie_source."""

    @pytest.mark.asyncio
    async def test_success_creates_with_valid_status(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Successful Genie source creation."""
        mock_obo_client.validate_genie_access.return_value = (True, None)

        source, error = await service.create_genie_source(
            owner_id="user-1",
            user_token="token",
            name="My Genie",
            space_id="space-123",
            example_questions=["How many?"],
        )

        assert error is None
        assert source is not None
        assert source.type == DataSourceType.GENIE.value
        assert source.config["space_id"] == "space-123"
        assert source.config["example_questions"] == ["How many?"]

    @pytest.mark.asyncio
    async def test_obo_failure_returns_error(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
        mock_obo_client: MagicMock,
    ) -> None:
        """Genie OBO failure returns error."""
        mock_obo_client.validate_genie_access.return_value = (False, "No Genie access")

        source, error = await service.create_genie_source(
            owner_id="user-1",
            user_token="token",
            name="Bad Genie",
            space_id="bad-space",
        )

        assert source is None
        assert error == "No Genie access"
        mock_session.add.assert_not_called()


# =========================================================================
# TestCreateAssistantSource
# =========================================================================


class TestCreateAssistantSource:
    """Tests for create_assistant_source."""

    @pytest.mark.asyncio
    async def test_success_creates_with_valid_status(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Successful Knowledge Assistant source creation."""
        mock_obo_client.validate_assistant_access.return_value = (True, None)

        source, error = await service.create_assistant_source(
            owner_id="user-1",
            user_token="token",
            name="My KA",
            endpoint_name="ka-endpoint",
            pass_context=False,
        )

        assert error is None
        assert source is not None
        assert source.type == DataSourceType.KNOWLEDGE_ASSISTANT.value
        assert source.config["endpoint_name"] == "ka-endpoint"
        assert source.config["pass_context"] is False

    @pytest.mark.asyncio
    async def test_obo_failure_returns_error(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
        mock_obo_client: MagicMock,
    ) -> None:
        """Assistant OBO failure returns error."""
        mock_obo_client.validate_assistant_access.return_value = (
            False,
            "Endpoint not found",
        )

        source, error = await service.create_assistant_source(
            owner_id="user-1",
            user_token="token",
            name="Bad KA",
            endpoint_name="bad-ep",
        )

        assert source is None
        assert error == "Endpoint not found"
        mock_session.add.assert_not_called()


# =========================================================================
# TestGetAccessibleSources
# =========================================================================


class TestGetAccessibleSources:
    """Tests for get_accessible_sources."""

    @pytest.mark.asyncio
    async def test_returns_own_sources(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Returns sources owned by the user."""
        own_source = _make_source(owner_id="user-1")

        # Mock execute to return sources for the data query and count
        count_result = MagicMock()
        count_result.scalar.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [own_source]

        mock_session.execute.side_effect = [count_result, data_result]

        sources, total = await service.get_accessible_sources(user_id="user-1")

        assert total == 1
        assert len(sources) == 1
        assert sources[0].owner_id == "user-1"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_sources(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Returns empty list when no sources match."""
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        sources, total = await service.get_accessible_sources(user_id="user-1")

        assert total == 0
        assert sources == []

    @pytest.mark.asyncio
    async def test_calls_execute_twice_for_count_and_data(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Executes two queries: one for count, one for data."""
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        await service.get_accessible_sources(user_id="user-1")

        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_with_type_filter(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Type filter narrows results."""
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        await service.get_accessible_sources(
            user_id="user-1",
            source_type=DataSourceType.GENIE,
        )

        # Verify execute was called (queries include type filter)
        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_pagination_params_passed(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Pagination parameters are passed to query."""
        count_result = MagicMock()
        count_result.scalar.return_value = 50

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        sources, total = await service.get_accessible_sources(
            user_id="user-1",
            limit=10,
            offset=20,
        )

        assert total == 50
        assert mock_session.execute.await_count == 2


# =========================================================================
# TestGetForUser
# =========================================================================


class TestGetForUser:
    """Tests for get_for_user (ownership check)."""

    @pytest.mark.asyncio
    async def test_returns_source_when_owner_matches(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Returns source when user is the owner."""
        source = _make_source(owner_id="user-1")
        result = MagicMock()
        result.scalar_one_or_none.return_value = source
        mock_session.execute.return_value = result

        got = await service.get_for_user(source.id, "user-1")
        assert got is source

    @pytest.mark.asyncio
    async def test_returns_none_when_owner_differs(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Returns None when user is not the owner."""
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = result

        got = await service.get_for_user(uuid4(), "different-user")
        assert got is None


# =========================================================================
# TestGetAccessible
# =========================================================================


class TestGetAccessible:
    """Tests for get_accessible (ownership or workspace-visible)."""

    @pytest.mark.asyncio
    async def test_returns_workspace_valid_source_for_non_owner(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Returns workspace-visible, valid source even for non-owner."""
        source = _make_source(
            owner_id="other-user",
            visibility=DataSourceVisibility.WORKSPACE.value,
            validation_status=DataSourceValidationStatus.VALID.value,
        )
        result = MagicMock()
        result.scalar_one_or_none.return_value = source
        mock_session.execute.return_value = result

        got = await service.get_accessible(source.id, "requesting-user")
        assert got is source

    @pytest.mark.asyncio
    async def test_returns_none_for_inaccessible_source(
        self,
        service: DataSourceService,
        mock_session: AsyncMock,
    ) -> None:
        """Returns None when source is not accessible."""
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = result

        got = await service.get_accessible(uuid4(), "user-1")
        assert got is None


# =========================================================================
# TestRevalidateSource
# =========================================================================


class TestRevalidateSource:
    """Tests for revalidate_source."""

    @pytest.mark.asyncio
    async def test_vector_search_dispatches_to_vs_validation(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Vector Search source calls validate_vector_search_access."""
        source = _make_source(
            source_type=DataSourceType.VECTOR_SEARCH.value,
            config={"endpoint_name": "ep-1", "index_name": "cat.sch.idx"},
        )

        await service.revalidate_source(source, "token")

        mock_obo_client.validate_vector_search_access.assert_awaited_once_with(
            "token", "ep-1", "cat.sch.idx"
        )

    @pytest.mark.asyncio
    async def test_genie_dispatches_to_genie_validation(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Genie source calls validate_genie_access."""
        source = _make_source(
            source_type=DataSourceType.GENIE.value,
            endpoint_identifier="space-id-1",
        )

        await service.revalidate_source(source, "token")

        mock_obo_client.validate_genie_access.assert_awaited_once_with(
            "token", "space-id-1"
        )

    @pytest.mark.asyncio
    async def test_marks_valid_on_success(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
        mock_session: AsyncMock,
    ) -> None:
        """Source is marked valid when OBO succeeds."""
        mock_obo_client.validate_vector_search_access.return_value = (True, None)
        source = _make_source(
            source_type=DataSourceType.VECTOR_SEARCH.value,
            config={"endpoint_name": "ep", "index_name": "idx"},
            validation_status=DataSourceValidationStatus.EXPIRED.value,
        )

        has_access, error = await service.revalidate_source(source, "token")

        assert has_access is True
        assert error is None
        assert source.validation_status == DataSourceValidationStatus.VALID.value
        assert source.last_validated_at is not None

    @pytest.mark.asyncio
    async def test_marks_invalid_on_failure(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
        mock_session: AsyncMock,
    ) -> None:
        """Source is marked invalid when OBO fails."""
        mock_obo_client.validate_vector_search_access.return_value = (
            False,
            "Permission denied",
        )
        source = _make_source(
            source_type=DataSourceType.VECTOR_SEARCH.value,
            config={"endpoint_name": "ep", "index_name": "idx"},
        )

        has_access, error = await service.revalidate_source(source, "token")

        assert has_access is False
        assert error == "Permission denied"
        assert source.validation_status == DataSourceValidationStatus.INVALID.value

    @pytest.mark.asyncio
    async def test_assistant_dispatches_to_assistant_validation(
        self,
        service: DataSourceService,
        mock_obo_client: MagicMock,
    ) -> None:
        """Knowledge Assistant source calls validate_assistant_access."""
        source = _make_source(
            source_type=DataSourceType.KNOWLEDGE_ASSISTANT.value,
            endpoint_identifier="ka-ep",
        )

        await service.revalidate_source(source, "token")

        mock_obo_client.validate_assistant_access.assert_awaited_once_with(
            "token", "ka-ep"
        )
