"""Data Source API endpoints.

Provides CRUD operations for user-configured enterprise data sources
(Vector Search indexes, Genie spaces, Knowledge Assistants).

Part of 007-enterprise-data-sources feature (T016).
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.exceptions import NotFoundError, PermissionDeniedError, ValidationError
from deep_research.db.session import get_db
from deep_research.middleware.auth import AuthenticatedUser, CurrentUser
from deep_research.models.data_source import (
    DataSourceType,
    DataSourceValidationStatus,
)
from deep_research.models.data_source import (
    DataSourceVisibility as ModelDataSourceVisibility,
)
from deep_research.schemas.data_source import (
    CreateGenieSourceRequest,
    CreateKnowledgeAssistantSourceRequest,
    CreateVectorSearchSourceRequest,
    DataSourceCapability,
    DataSourceConfig,
    DataSourceListResponse,
    DataSourceResponse,
    DataSourceValidationResponse,
    DataSourceVisibility,
    UpdateDataSourceRequest,
)
from deep_research.schemas.query_config import (
    FilterExpression,
    QueryConfigResponse,
    QueryConfigValidationResult,
    QueryType,
    UpdateQueryConfigRequest,
    VectorSearchQueryConfig,
    validate_query_config,
)
from deep_research.services.data_source_service import DataSourceService
from deep_research.services.obo_client import OBODatabricksClient

router = APIRouter(prefix="/data-sources", tags=["Data Sources"])


def _get_obo_token(request: Request) -> str | None:
    """Extract OBO token from request state (set by middleware).

    Args:
        request: FastAPI request object.

    Returns:
        OBO token or None if not available.
    """
    return getattr(request.state, "obo_token", None)


def _infer_capabilities(source_type: DataSourceType) -> list[DataSourceCapability]:
    """Infer capabilities from source type.

    Args:
        source_type: Type of data source.

    Returns:
        List of capabilities.
    """
    capabilities_map = {
        DataSourceType.VECTOR_SEARCH: [
            DataSourceCapability.SEMANTIC_SEARCH,
            DataSourceCapability.KEYWORD_SEARCH,
            DataSourceCapability.METADATA_FILTERING,
        ],
        DataSourceType.GENIE: [
            DataSourceCapability.SQL_ANALYTICS,
            DataSourceCapability.AGGREGATIONS,
            DataSourceCapability.FOLLOW_UP,
        ],
        DataSourceType.KNOWLEDGE_ASSISTANT: [
            DataSourceCapability.DOMAIN_EXPERTISE,
        ],
    }
    return capabilities_map.get(source_type, [])


def _source_to_response(source: Any) -> DataSourceResponse:
    """Convert UserDataSource model to response schema.

    Args:
        source: UserDataSource model instance.

    Returns:
        DataSourceResponse schema.
    """
    source_type = DataSourceType(source.type)
    config = source.config or {}

    return DataSourceResponse(
        id=source.id,
        owner_id=source.owner_id,
        type=source_type,
        name=source.name,
        description=source.description,
        endpoint_identifier=source.endpoint_identifier,
        config=DataSourceConfig(
            endpoint_name=config.get("endpoint_name"),
            index_name=config.get("index_name"),
            columns=config.get("columns"),
            columns_to_rerank=config.get("columns_to_rerank"),
            enable_hybrid=config.get("enable_hybrid"),
            enable_reranking=config.get("enable_reranking"),
            num_results=config.get("num_results"),
            space_id=config.get("space_id"),
            example_questions=config.get("example_questions"),
            pass_context=config.get("pass_context"),
        ),
        visibility=DataSourceVisibility(source.visibility),
        validation_status=DataSourceValidationStatus(source.validation_status),
        last_validated_at=source.last_validated_at,
        created_at=source.created_at,
        updated_at=source.updated_at,
        capabilities=_infer_capabilities(source_type),
        source_origin="user",
    )


# =============================================================================
# List and Get Endpoints
# =============================================================================


@router.get("", response_model=DataSourceListResponse)
async def list_data_sources(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    source_type: DataSourceType | None = Query(None, description="Filter by source type"),
    only_valid: bool = Query(True, description="Only return sources with valid OBO access"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DataSourceListResponse:
    """List data sources accessible to the current user.

    Returns both user-owned sources and workspace-visible sources with valid access.
    """
    service = DataSourceService(db)
    sources, total = await service.get_accessible_sources(
        user_id=user.user_id,
        source_type=source_type,
        only_valid=only_valid,
        limit=limit,
        offset=offset,
    )

    # Count user vs workspace sources
    user_sources = sum(1 for s in sources if s.owner_id == user.user_id)
    workspace_sources = total - user_sources

    return DataSourceListResponse(
        sources=[_source_to_response(s) for s in sources],
        total=total,
        user_sources=user_sources,
        workspace_sources=workspace_sources,
    )


@router.get("/{source_id}", response_model=DataSourceResponse)
async def get_data_source(
    source_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> DataSourceResponse:
    """Get details of a specific data source.

    Returns the source if owned by user or workspace-visible with valid access.
    """
    service = DataSourceService(db)
    source = await service.get_accessible(source_id, user.user_id)

    if not source:
        raise NotFoundError("Data source", str(source_id))

    return _source_to_response(source)


# =============================================================================
# Create Endpoints
# =============================================================================


@router.post("/vector-search", response_model=DataSourceResponse, status_code=201)
async def create_vector_search_source(
    request_body: CreateVectorSearchSourceRequest,
    request: Request,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> DataSourceResponse:
    """Create a new Vector Search data source.

    Validates OBO access to the index and auto-detects column schema.
    Requires authenticated user with workspace access.
    """
    obo_token = _get_obo_token(request)
    if not obo_token:
        raise PermissionDeniedError(
            "OBO authentication required. Please sign in through Databricks Apps."
        )

    service = DataSourceService(db, OBODatabricksClient())
    source, error = await service.create_vector_search_source(
        owner_id=user.user_id,
        user_token=obo_token,
        name=request_body.name,
        endpoint_name=request_body.endpoint_name,
        index_name=request_body.index_name,
        description=request_body.description,
        visibility=ModelDataSourceVisibility(request_body.visibility.value),
        enable_hybrid=request_body.enable_hybrid,
        enable_reranking=request_body.enable_reranking,
        num_results=request_body.num_results,
    )

    if error:
        raise ValidationError(error)

    await db.commit()
    return _source_to_response(source)


@router.post("/genie", response_model=DataSourceResponse, status_code=201)
async def create_genie_source(
    request_body: CreateGenieSourceRequest,
    request: Request,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> DataSourceResponse:
    """Create a new Genie data source.

    Validates OBO access to the Genie space.
    Requires authenticated user with workspace access.
    """
    obo_token = _get_obo_token(request)
    if not obo_token:
        raise PermissionDeniedError(
            "OBO authentication required. Please sign in through Databricks Apps."
        )

    service = DataSourceService(db, OBODatabricksClient())
    source, error = await service.create_genie_source(
        owner_id=user.user_id,
        user_token=obo_token,
        name=request_body.name,
        space_id=request_body.space_id,
        description=request_body.description,
        example_questions=request_body.example_questions,
        visibility=ModelDataSourceVisibility(request_body.visibility.value),
    )

    if error:
        raise ValidationError(error)

    await db.commit()
    return _source_to_response(source)


@router.post("/knowledge-assistant", response_model=DataSourceResponse, status_code=201)
async def create_knowledge_assistant_source(
    request_body: CreateKnowledgeAssistantSourceRequest,
    request: Request,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> DataSourceResponse:
    """Create a new Knowledge Assistant data source.

    Validates OBO access to the serving endpoint.
    Requires authenticated user with workspace access.
    """
    obo_token = _get_obo_token(request)
    if not obo_token:
        raise PermissionDeniedError(
            "OBO authentication required. Please sign in through Databricks Apps."
        )

    service = DataSourceService(db, OBODatabricksClient())
    source, error = await service.create_assistant_source(
        owner_id=user.user_id,
        user_token=obo_token,
        name=request_body.name,
        endpoint_name=request_body.endpoint_name,
        description=request_body.description,
        pass_context=request_body.pass_context,
        visibility=ModelDataSourceVisibility(request_body.visibility.value),
    )

    if error:
        raise ValidationError(error)

    await db.commit()
    return _source_to_response(source)


# =============================================================================
# Update and Delete Endpoints
# =============================================================================


@router.patch("/{source_id}", response_model=DataSourceResponse)
async def update_data_source(
    source_id: UUID,
    request_body: UpdateDataSourceRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> DataSourceResponse:
    """Update a data source.

    Only the source owner can update. Updates name, description, visibility,
    and type-specific configuration.
    """
    service = DataSourceService(db)
    source = await service.get_for_user(source_id, user.user_id)

    if not source:
        raise NotFoundError("Data source", str(source_id))

    # Update fields
    if request_body.name is not None:
        source.name = request_body.name
    if request_body.description is not None:
        source.description = request_body.description
    if request_body.visibility is not None:
        source.visibility = request_body.visibility.value

    # Update type-specific config
    config = source.config or {}

    source_type = DataSourceType(source.type)

    if source_type == DataSourceType.VECTOR_SEARCH:
        if request_body.enable_hybrid is not None:
            config["enable_hybrid"] = request_body.enable_hybrid
        if request_body.enable_reranking is not None:
            config["enable_reranking"] = request_body.enable_reranking
        if request_body.num_results is not None:
            config["num_results"] = request_body.num_results

    elif source_type == DataSourceType.GENIE:
        if request_body.example_questions is not None:
            config["example_questions"] = request_body.example_questions

    elif source_type == DataSourceType.KNOWLEDGE_ASSISTANT:
        if request_body.pass_context is not None:
            config["pass_context"] = request_body.pass_context

    source.config = config
    source.updated_at = datetime.now(UTC)

    await service.update(source)
    await db.commit()

    return _source_to_response(source)


@router.delete("/{source_id}", status_code=204)
async def delete_data_source(
    source_id: UUID,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a data source.

    Only the source owner can delete.
    """
    service = DataSourceService(db)
    source = await service.get_for_user(source_id, user.user_id)

    if not source:
        raise NotFoundError("Data source", str(source_id))

    await service.delete(source)
    await db.commit()


# =============================================================================
# Validation Endpoints
# =============================================================================


@router.post("/{source_id}/validate", response_model=DataSourceValidationResponse)
async def validate_data_source(
    source_id: UUID,
    request: Request,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> DataSourceValidationResponse:
    """Re-validate OBO access for a data source.

    Checks if the user still has access to the underlying resource.
    Updates the source's validation status.
    """
    obo_token = _get_obo_token(request)
    if not obo_token:
        raise PermissionDeniedError(
            "OBO authentication required. Please sign in through Databricks Apps."
        )

    service = DataSourceService(db, OBODatabricksClient())
    source = await service.get_for_user(source_id, user.user_id)

    if not source:
        raise NotFoundError("Data source", str(source_id))

    has_access, error = await service.revalidate_source(source, obo_token)
    await db.commit()

    return DataSourceValidationResponse(
        source_id=source.id,
        has_access=has_access,
        error_message=error,
        validated_at=source.last_validated_at or datetime.now(UTC),
        # For VS sources, we could return detected columns here
        detected_columns=source.config.get("columns") if source.config else None,
        detected_text_columns=source.config.get("columns_to_rerank") if source.config else None,
    )


@router.post("/validate-connection", response_model=DataSourceValidationResponse)
async def validate_connection(
    request: Request,
    user: AuthenticatedUser,  # noqa: ARG001 - Required for auth
    source_type: DataSourceType = Query(..., description="Type of source to validate"),
    endpoint_name: str | None = Query(None, description="Vector Search endpoint or assistant endpoint"),
    index_name: str | None = Query(None, description="Vector Search index name"),
    space_id: str | None = Query(None, description="Genie space ID"),
    db: AsyncSession = Depends(get_db),  # noqa: ARG001 - Required for session
) -> DataSourceValidationResponse:
    """Validate connection to an enterprise resource before creating a source.

    Tests OBO access without creating a data source.
    Useful for UI connection testing dialogs.
    """
    obo_token = _get_obo_token(request)
    if not obo_token:
        raise PermissionDeniedError(
            "OBO authentication required. Please sign in through Databricks Apps."
        )

    obo_client = OBODatabricksClient()
    has_access = False
    error = None
    detected_columns = None
    detected_text_columns = None

    if source_type == DataSourceType.VECTOR_SEARCH:
        if not endpoint_name or not index_name:
            raise ValidationError(
                "endpoint_name and index_name are required for Vector Search validation"
            )
        has_access, error = await obo_client.validate_vector_search_access(
            obo_token, endpoint_name, index_name
        )
        if has_access:
            # Get schema info
            schema = await obo_client.get_vector_search_index_schema(
                obo_token, endpoint_name, index_name
            )
            if schema:
                detected_columns = [col["name"] for col in schema.get("columns", [])]
                detected_text_columns = schema.get("text_columns", [])

    elif source_type == DataSourceType.GENIE:
        if not space_id:
            raise ValidationError("space_id is required for Genie validation")
        has_access, error = await obo_client.validate_genie_access(obo_token, space_id)

    elif source_type == DataSourceType.KNOWLEDGE_ASSISTANT:
        if not endpoint_name:
            raise ValidationError("endpoint_name is required for Knowledge Assistant validation")
        has_access, error = await obo_client.validate_assistant_access(obo_token, endpoint_name)

    else:
        raise ValidationError(f"Unsupported source type for validation: {source_type}")

    # Return a placeholder UUID for validation responses (not persisted)
    from uuid import uuid4

    return DataSourceValidationResponse(
        source_id=uuid4(),  # Placeholder
        has_access=has_access,
        error_message=error,
        validated_at=datetime.now(UTC),
        detected_columns=detected_columns,
        detected_text_columns=detected_text_columns,
    )


# =============================================================================
# Query Configuration Endpoints (US9b - T010s, T010t)
# =============================================================================


@router.get("/{source_id}/query-config", response_model=QueryConfigResponse)
async def get_query_config(
    source_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    validate: bool = Query(False, description="Validate config against source capabilities"),
) -> QueryConfigResponse:
    """Get query configuration for a Vector Search data source.

    Returns the current query configuration including query type, filters,
    and result settings. Optionally validates against index capabilities.

    Only applicable to Vector Search sources.
    """
    service = DataSourceService(db)
    source = await service.get_accessible(source_id, user.user_id)

    if not source:
        raise NotFoundError("Data source", str(source_id))

    source_type = DataSourceType(source.type)
    if source_type != DataSourceType.VECTOR_SEARCH:
        raise ValidationError(
            f"Query configuration only supported for Vector Search sources (got {source_type.value})"
        )

    # Extract query_config from source config
    config = source.config or {}
    query_config_data = config.get("query_config", {})

    # Build VectorSearchQueryConfig from stored data
    filters: list[FilterExpression] = []
    if "filters" in query_config_data:
        for f in query_config_data["filters"]:
            filters.append(FilterExpression(**f))

    query_config = VectorSearchQueryConfig(
        query_type=QueryType(query_config_data.get("query_type", "ANN")),
        num_results=query_config_data.get("num_results", 10),
        score_threshold=query_config_data.get("score_threshold"),
        columns=query_config_data.get("columns"),
        enable_reranking=query_config_data.get("enable_reranking", False),
        columns_to_rerank=query_config_data.get("columns_to_rerank"),
        filters=filters,
        filter_syntax=query_config_data.get("filter_syntax", "sql"),
    )

    # Optionally validate against index capabilities
    validation: QueryConfigValidationResult | None = None
    if validate:
        # Get available filter columns and supported query types from source config
        available_columns = config.get("columns", [])
        text_columns = config.get("columns_to_rerank", [])

        # Determine supported query types (ANN always, HYBRID if text columns)
        supported_types = [QueryType.ANN]
        if text_columns:
            supported_types.append(QueryType.HYBRID)
            supported_types.append(QueryType.FULL_TEXT)

        validation = validate_query_config(
            config=query_config,
            supported_query_types=supported_types,
            filter_columns=available_columns,
            supports_reranking=bool(text_columns),
        )

    return QueryConfigResponse(
        source_id=str(source_id),
        config=query_config,
        validation=validation,
    )


@router.put("/{source_id}/query-config", response_model=QueryConfigResponse)
async def update_query_config(
    source_id: UUID,
    request_body: UpdateQueryConfigRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    validate: bool = Query(True, description="Validate config before saving"),
) -> QueryConfigResponse:
    """Update query configuration for a Vector Search data source.

    Updates query type, filters, result settings, and reranking configuration.
    By default, validates the configuration against index capabilities.

    Only applicable to Vector Search sources.
    """
    service = DataSourceService(db)
    source = await service.get_for_user(source_id, user.user_id)

    if not source:
        raise NotFoundError("Data source", str(source_id))

    source_type = DataSourceType(source.type)
    if source_type != DataSourceType.VECTOR_SEARCH:
        raise ValidationError(
            f"Query configuration only supported for Vector Search sources (got {source_type.value})"
        )

    # Get current config
    config = source.config or {}
    current_query_config = config.get("query_config", {})

    # Merge updates into current config (partial update)
    if request_body.query_type is not None:
        current_query_config["query_type"] = request_body.query_type.value
    if request_body.num_results is not None:
        current_query_config["num_results"] = request_body.num_results
    if request_body.score_threshold is not None:
        current_query_config["score_threshold"] = request_body.score_threshold
    if request_body.columns is not None:
        current_query_config["columns"] = request_body.columns
    if request_body.enable_reranking is not None:
        current_query_config["enable_reranking"] = request_body.enable_reranking
    if request_body.columns_to_rerank is not None:
        current_query_config["columns_to_rerank"] = request_body.columns_to_rerank
    if request_body.filters is not None:
        current_query_config["filters"] = [f.model_dump() for f in request_body.filters]
    if request_body.filter_syntax is not None:
        current_query_config["filter_syntax"] = request_body.filter_syntax.value

    # Build the full query config for validation
    filters: list[FilterExpression] = []
    if "filters" in current_query_config:
        for f in current_query_config["filters"]:
            filters.append(FilterExpression(**f))

    query_config = VectorSearchQueryConfig(
        query_type=QueryType(current_query_config.get("query_type", "ANN")),
        num_results=current_query_config.get("num_results", 10),
        score_threshold=current_query_config.get("score_threshold"),
        columns=current_query_config.get("columns"),
        enable_reranking=current_query_config.get("enable_reranking", False),
        columns_to_rerank=current_query_config.get("columns_to_rerank"),
        filters=filters,
        filter_syntax=current_query_config.get("filter_syntax", "sql"),
    )

    # Validate if requested
    validation: QueryConfigValidationResult | None = None
    if validate:
        available_columns = config.get("columns", [])
        text_columns = config.get("columns_to_rerank", [])

        supported_types = [QueryType.ANN]
        if text_columns:
            supported_types.append(QueryType.HYBRID)
            supported_types.append(QueryType.FULL_TEXT)

        validation = validate_query_config(
            config=query_config,
            supported_query_types=supported_types,
            filter_columns=available_columns,
            supports_reranking=bool(text_columns),
        )

        # If validation failed with errors, reject the update
        if not validation.is_valid:
            raise ValidationError(
                f"Invalid query configuration: {'; '.join(validation.errors)}"
            )

    # Save the updated config
    config["query_config"] = current_query_config
    source.config = config
    source.updated_at = datetime.now(UTC)

    await service.update(source)
    await db.commit()

    return QueryConfigResponse(
        source_id=str(source_id),
        config=query_config,
        validation=validation,
    )
