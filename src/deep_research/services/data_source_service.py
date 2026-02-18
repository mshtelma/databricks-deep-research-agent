"""DataSourceService - CRUD operations for user data sources.

Manages user-configured enterprise data sources (Vector Search indexes,
Genie spaces, Knowledge Assistants) with OBO validation.

Part of 007-enterprise-data-sources feature (T014).

Security Model (T110-T111):
============================

1. OWNERSHIP:
   - Each data source has an owner_id (Databricks workspace user ID)
   - Only the owner can delete or update their sources
   - Owner is set at creation time and cannot be changed

2. VISIBILITY:
   - PRIVATE: Only the owner can access
   - WORKSPACE: Anyone in the workspace can use (if source is VALID)
   - Workspace visibility requires the source to be validated

3. ACCESS CONTROL:
   - get_for_user(): Returns source only if owned by user (strict)
   - get_accessible(): Returns source if owned OR workspace-visible
   - get_accessible_sources(): Lists sources user can access

4. OBO VALIDATION:
   - All data sources are validated against user's OBO token at creation
   - Re-validation can be triggered to check continued access
   - Invalid sources cannot be used until re-validated

5. TOKEN HANDLING:
   - User tokens are passed to OBODatabricksClient for validation
   - Tokens are NOT stored in the database
   - See obo_client.py for token security details
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select

from deep_research.models.data_source import (
    DataSourceType,
    DataSourceValidationStatus,
    DataSourceVisibility,
    UserDataSource,
)
from deep_research.services.base import BaseRepository
from deep_research.services.obo_client import OBODatabricksClient

logger = logging.getLogger(__name__)


class DataSourceService(BaseRepository[UserDataSource]):
    """Service for managing user data sources.

    Extends BaseRepository[UserDataSource] for standard CRUD operations.
    Provides specialized methods for:
    - Creating sources with schema auto-detection
    - OBO access validation
    - Listing accessible sources (own + workspace)
    """

    model = UserDataSource

    def __init__(self, session: Any, obo_client: OBODatabricksClient | None = None) -> None:
        """Initialize service with database session.

        Args:
            session: Async SQLAlchemy session.
            obo_client: Optional OBO client for validation. If not provided,
                        validation methods will use a default client.
        """
        super().__init__(session)
        self._obo_client = obo_client or OBODatabricksClient()

    # =========================================================================
    # Vector Search Sources (US1)
    # =========================================================================

    async def create_vector_search_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        endpoint_name: str,
        index_name: str,
        description: str | None = None,
        visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE,
        enable_hybrid: bool = True,
        enable_reranking: bool = True,
        num_results: int = 10,
    ) -> tuple[UserDataSource, str | None]:
        """Create a Vector Search data source with schema auto-detection.

        Validates OBO access and auto-detects columns from index metadata.

        Args:
            owner_id: Databricks workspace user ID.
            user_token: User's OAuth token for OBO validation.
            name: Display name for the source.
            endpoint_name: Vector Search endpoint name.
            index_name: Fully qualified index name (catalog.schema.index).
            description: Optional description.
            visibility: Visibility level.
            enable_hybrid: Enable hybrid search (BM25 + vectors).
            enable_reranking: Enable reranking for improved relevance.
            num_results: Default number of results to return.

        Returns:
            Tuple of (created source, error message or None).
        """
        # Validate OBO access
        has_access, error = await self._obo_client.validate_vector_search_access(
            user_token, endpoint_name, index_name
        )
        if not has_access:
            return None, error  # type: ignore

        # Auto-detect schema from index
        schema = await self._obo_client.get_vector_search_index_schema(
            user_token, endpoint_name, index_name
        )

        # Build config with auto-detected columns
        columns = [col["name"] for col in (schema or {}).get("columns", [])]
        text_columns = (schema or {}).get("text_columns", [])

        config = {
            "endpoint_name": endpoint_name,
            "index_name": index_name,
            "columns": columns,
            "columns_to_rerank": text_columns[:2] if text_columns else [],  # First 2 text columns
            "enable_hybrid": enable_hybrid,
            "enable_reranking": enable_reranking,
            "num_results": num_results,
        }

        # Create source
        source = UserDataSource(
            owner_id=owner_id,
            type=DataSourceType.VECTOR_SEARCH.value,
            name=name,
            description=description,
            endpoint_identifier=index_name,
            config=config,
            visibility=visibility.value,
            validation_status=DataSourceValidationStatus.VALID.value,
            last_validated_at=datetime.now(UTC),
        )

        source = await self.add(source)
        logger.info(
            "Created Vector Search source",
            extra={
                "source_id": str(source.id),
                "owner_id": owner_id,
                "index_name": index_name,
            },
        )
        return source, None

    # =========================================================================
    # Genie Sources (US2)
    # =========================================================================

    async def create_genie_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        space_id: str,
        description: str | None = None,
        example_questions: list[str] | None = None,
        visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE,
    ) -> tuple[UserDataSource, str | None]:
        """Create a Genie data source.

        Validates OBO access to the Genie space.

        Args:
            owner_id: Databricks workspace user ID.
            user_token: User's OAuth token for OBO validation.
            name: Display name for the source.
            space_id: Genie space ID.
            description: Optional description.
            example_questions: Example questions to show in UI.
            visibility: Visibility level.

        Returns:
            Tuple of (created source, error message or None).
        """
        # Validate OBO access
        has_access, error = await self._obo_client.validate_genie_access(
            user_token, space_id
        )
        if not has_access:
            return None, error  # type: ignore

        config = {
            "space_id": space_id,
            "example_questions": example_questions or [],
        }

        source = UserDataSource(
            owner_id=owner_id,
            type=DataSourceType.GENIE.value,
            name=name,
            description=description,
            endpoint_identifier=space_id,
            config=config,
            visibility=visibility.value,
            validation_status=DataSourceValidationStatus.VALID.value,
            last_validated_at=datetime.now(UTC),
        )

        source = await self.add(source)
        logger.info(
            "Created Genie source",
            extra={
                "source_id": str(source.id),
                "owner_id": owner_id,
                "space_id": space_id,
            },
        )
        return source, None

    # =========================================================================
    # Knowledge Assistant Sources (US3)
    # =========================================================================

    async def create_assistant_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        endpoint_name: str,
        description: str | None = None,
        pass_context: bool = True,
        visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE,
    ) -> tuple[UserDataSource, str | None]:
        """Create a Knowledge Assistant data source.

        Validates OBO access to the serving endpoint.

        Args:
            owner_id: Databricks workspace user ID.
            user_token: User's OAuth token for OBO validation.
            name: Display name for the source.
            endpoint_name: Serving endpoint name.
            description: Optional description.
            pass_context: Whether to pass research context to the assistant.
            visibility: Visibility level.

        Returns:
            Tuple of (created source, error message or None).
        """
        # Validate OBO access
        has_access, error = await self._obo_client.validate_assistant_access(
            user_token, endpoint_name
        )
        if not has_access:
            return None, error  # type: ignore

        config = {
            "endpoint_name": endpoint_name,
            "pass_context": pass_context,
        }

        source = UserDataSource(
            owner_id=owner_id,
            type=DataSourceType.KNOWLEDGE_ASSISTANT.value,
            name=name,
            description=description,
            endpoint_identifier=endpoint_name,
            config=config,
            visibility=visibility.value,
            validation_status=DataSourceValidationStatus.VALID.value,
            last_validated_at=datetime.now(UTC),
        )

        source = await self.add(source)
        logger.info(
            "Created Knowledge Assistant source",
            extra={
                "source_id": str(source.id),
                "owner_id": owner_id,
                "endpoint_name": endpoint_name,
            },
        )
        return source, None

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_accessible_sources(
        self,
        user_id: str,
        source_type: DataSourceType | None = None,
        only_valid: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[UserDataSource], int]:
        """Get data sources accessible to a user.

        Returns sources that are:
        - Owned by the user (any visibility)
        - OR workspace-visible AND valid

        Args:
            user_id: Databricks workspace user ID.
            source_type: Optional filter by source type.
            only_valid: If True, only return sources with valid OBO access.
            limit: Maximum number of sources.
            offset: Number of sources to skip.

        Returns:
            Tuple of (sources, total_count).
        """
        # Build conditions: own sources OR (workspace visible AND valid)
        access_conditions = or_(
            UserDataSource.owner_id == user_id,
            and_(
                UserDataSource.visibility == DataSourceVisibility.WORKSPACE.value,
                UserDataSource.validation_status == DataSourceValidationStatus.VALID.value,
            ) if only_valid else UserDataSource.visibility == DataSourceVisibility.WORKSPACE.value,
        )

        conditions = [access_conditions]

        if source_type:
            conditions.append(UserDataSource.type == source_type.value)

        if only_valid:
            # For own sources, also filter by valid status
            conditions.append(
                or_(
                    # Workspace sources already filtered above
                    UserDataSource.owner_id != user_id,
                    UserDataSource.validation_status == DataSourceValidationStatus.VALID.value,
                )
            )

        # Get total count
        count_query = select(func.count(UserDataSource.id)).where(and_(*conditions))
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        # Get sources
        query = (
            select(UserDataSource)
            .where(and_(*conditions))
            .order_by(UserDataSource.name)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(query)
        sources = list(result.scalars().all())

        return sources, total

    async def get_for_user(self, source_id: UUID, user_id: str) -> UserDataSource | None:
        """Get a source by ID with user ownership check.

        Args:
            source_id: Source ID.
            user_id: User ID (for ownership check).

        Returns:
            Source if found and owned by user, None otherwise.
        """
        result = await self._session.execute(
            select(UserDataSource).where(
                and_(
                    UserDataSource.id == source_id,
                    UserDataSource.owner_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_accessible(self, source_id: UUID, user_id: str) -> UserDataSource | None:
        """Get a source by ID if accessible to user.

        Source is accessible if:
        - Owned by the user
        - OR workspace-visible and valid

        Args:
            source_id: Source ID.
            user_id: User ID.

        Returns:
            Source if accessible, None otherwise.
        """
        result = await self._session.execute(
            select(UserDataSource).where(
                and_(
                    UserDataSource.id == source_id,
                    or_(
                        UserDataSource.owner_id == user_id,
                        and_(
                            UserDataSource.visibility == DataSourceVisibility.WORKSPACE.value,
                            UserDataSource.validation_status == DataSourceValidationStatus.VALID.value,
                        ),
                    ),
                )
            )
        )
        return result.scalar_one_or_none()

    # =========================================================================
    # Validation Methods
    # =========================================================================

    async def revalidate_source(
        self,
        source: UserDataSource,
        user_token: str,
    ) -> tuple[bool, str | None]:
        """Re-validate OBO access for a data source.

        Args:
            source: The source to validate.
            user_token: User's OAuth token.

        Returns:
            Tuple of (is_valid, error_message or None).
        """
        source_type = DataSourceType(source.type)
        config = source.config

        if source_type == DataSourceType.VECTOR_SEARCH:
            has_access, error = await self._obo_client.validate_vector_search_access(
                user_token,
                config.get("endpoint_name", ""),
                config.get("index_name", source.endpoint_identifier),
            )
        elif source_type == DataSourceType.GENIE:
            has_access, error = await self._obo_client.validate_genie_access(
                user_token,
                source.endpoint_identifier,
            )
        elif source_type == DataSourceType.KNOWLEDGE_ASSISTANT:
            has_access, error = await self._obo_client.validate_assistant_access(
                user_token,
                source.endpoint_identifier,
            )
        else:
            # Custom sources: assume valid
            has_access, error = True, None

        if has_access:
            source.mark_valid()
        else:
            source.mark_invalid()

        await self.update(source)
        return has_access, error

    async def mark_sources_expired(self, owner_id: str) -> int:
        """Mark all sources for an owner as expired.

        Used when user's OBO access might have changed (e.g., role change).

        Args:
            owner_id: Owner user ID.

        Returns:
            Number of sources updated.
        """
        result = await self._session.execute(
            select(UserDataSource).where(
                and_(
                    UserDataSource.owner_id == owner_id,
                    UserDataSource.validation_status == DataSourceValidationStatus.VALID.value,
                )
            )
        )
        sources = list(result.scalars().all())

        for source in sources:
            source.mark_expired()

        await self._session.flush()
        return len(sources)
