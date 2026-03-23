"""Discovery API endpoints for data source auto-discovery.

This module provides API endpoints for discovering data sources available
to the authenticated user via OBO (On-Behalf-Of) authentication.

Endpoints:
- GET /api/v1/discovery/sources - Discover all available data sources
- GET /api/v1/discovery/sources/{source_id}/metadata - Get detailed source metadata
- POST /api/v1/discovery/refresh - Force cache refresh

Contract: /specs/007-enterprise-data-sources/contracts/discovery.yaml
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status

from deep_research.core.logging_utils import get_logger
from deep_research.middleware.auth import CurrentUser
from deep_research.schemas.data_source import DataSourceType
from deep_research.schemas.discovery import (
    DiscoveryResponse,
    RefreshDiscoveryRequest,
    SourceMetadataResponse,
)
from deep_research.services.discovery_cache import get_discovery_cache
from deep_research.services.discovery_service import get_discovery_service

logger = get_logger(__name__)

router = APIRouter(prefix="/discovery")


def _get_obo_token(request: Request) -> str | None:
    """Extract OBO token from request state (set by middleware).

    Args:
        request: FastAPI request object.

    Returns:
        OBO token or None if not available.
    """
    return getattr(request.state, "obo_token", None)


@router.get(
    "/sources",
    response_model=DiscoveryResponse,
    summary="Discover all available data sources",
    description="""
    Returns all data sources the authenticated user has access to.

    In Databricks Apps: Uses OBO (On-Behalf-Of) authentication.
    In local development: Uses workspace client with user's profile credentials.

    Results are cached for 5 minutes per user.

    Discovered source types:
    - **vector_search**: Databricks Vector Search indexes
    - **genie**: Databricks Genie spaces (AI/BI)
    - **knowledge_assistant**: Serving endpoints identified as Knowledge Assistants

    Use the `refresh` query parameter to force cache refresh.
    Use `include_all_endpoints=true` to include all serving endpoints, not just detected Knowledge Assistants.
    """,
    responses={
        200: {"description": "Successfully retrieved discovered sources"},
        401: {"description": "Not authenticated"},
        503: {"description": "Discovery service unavailable"},
    },
)
async def discover_sources(
    request: Request,
    user: CurrentUser,
    source_type: Annotated[
        DataSourceType | None,
        Query(description="Filter by source type"),
    ] = None,
    refresh: Annotated[
        bool,
        Query(description="Force cache refresh"),
    ] = False,
    include_all_endpoints: Annotated[
        bool,
        Query(description="Include all serving endpoints, not just detected Knowledge Assistants"),
    ] = False,
) -> DiscoveryResponse:
    """Discover all available data sources.

    Args:
        request: FastAPI request object.
        user: Current authenticated user.
        source_type: Optional filter by source type.
        refresh: If True, bypass cache and re-discover.
        include_all_endpoints: If True, include all serving endpoints.

    Returns:
        DiscoveryResponse with all discovered sources.
    """
    # Get OBO token (only available in Databricks Apps)
    obo_token = _get_obo_token(request)

    # Comprehensive logging for debugging
    logger.info(
        "DISCOVERY_AUTH_STATE",
        has_obo_token=bool(obo_token),
        user_id=user.user_id[:8] if user.user_id and len(user.user_id) > 8 else user.user_id,
        user_email=user.email,
        is_anonymous=user.user_id == "anonymous",
    )

    # Require authenticated user (not anonymous)
    if user.user_id == "anonymous":
        logger.warning("DISCOVERY_REJECTED", reason="Anonymous user not allowed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for discovery. Please configure Databricks authentication.",
        )

    logger.info(
        "DISCOVERY_REQUEST",
        user_id=user.user_id[:8] if len(user.user_id) > 8 else user.user_id,
        source_type=source_type.value if source_type else "all",
        refresh=refresh,
        include_all_endpoints=include_all_endpoints,
        auth_method="obo" if obo_token else "workspace_client",
    )

    service = get_discovery_service()

    try:
        # Filter by type if specified
        source_types = [source_type] if source_type else None

        response = await service.discover_all(
            user_id=user.user_id,  # Always pass user_id for caching
            user_token=obo_token,  # Optional - None in local dev is OK
            force_refresh=refresh,
            source_types=source_types,
            include_all_endpoints=include_all_endpoints,
        )

        logger.info(
            "DISCOVERY_RESPONSE",
            total_count=response.total_count,
            cached=response.cached,
            error_count=len(response.errors) if response.errors else 0,
        )

        return response

    except Exception as e:
        logger.error("DISCOVERY_ERROR", error=str(e)[:200], user_id=user.user_id[:8])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Discovery service error: {str(e)[:200]}",
        ) from e


@router.get(
    "/sources/{source_id}/metadata",
    response_model=SourceMetadataResponse,
    summary="Get detailed metadata for a source",
    description="""
    Returns expanded metadata for a specific discovered source.

    The source_id format varies by type:
    - Vector Search: `vs:catalog.schema.index`
    - Genie: `genie:space_id`
    - Knowledge Assistant: `assistant:endpoint_name`
    """,
    responses={
        200: {"description": "Successfully retrieved source metadata"},
        401: {"description": "Not authenticated"},
        404: {"description": "Source not found"},
    },
)
async def get_source_metadata(
    source_id: str,
    request: Request,
    user: CurrentUser,
) -> SourceMetadataResponse:
    """Get detailed metadata for a specific source.

    Args:
        source_id: Source identifier (e.g., 'vs:catalog.schema.index').
        request: FastAPI request object.
        user: Current authenticated user.

    Returns:
        SourceMetadataResponse with detailed metadata.
    """
    # Require authenticated user (not anonymous)
    if user.user_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for discovery.",
        )

    obo_token = _get_obo_token(request)

    logger.info(
        "METADATA_REQUEST",
        source_id=source_id,
        user_id=user.user_id[:8] if len(user.user_id) > 8 else user.user_id,
    )

    service = get_discovery_service()

    try:
        response = await service.get_source_metadata(
            user_id=user.user_id,
            user_token=obo_token,
            source_id=source_id,
        )

        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source not found: {source_id}",
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("METADATA_ERROR", source_id=source_id, error=str(e)[:200])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get source metadata: {str(e)[:200]}",
        ) from e


@router.post(
    "/refresh",
    response_model=DiscoveryResponse,
    summary="Force refresh discovery cache",
    description="""
    Invalidates the discovery cache and re-queries Databricks APIs.

    Optionally specify source_types to refresh only specific types.
    """,
    responses={
        200: {"description": "Cache refreshed successfully"},
        401: {"description": "Not authenticated"},
    },
)
async def refresh_discovery(
    request: Request,
    user: CurrentUser,
    body: RefreshDiscoveryRequest | None = None,
) -> DiscoveryResponse:
    """Force refresh discovery cache.

    Args:
        request: FastAPI request object.
        user: Current authenticated user.
        body: Optional request body with source_types to refresh.

    Returns:
        Fresh DiscoveryResponse.
    """
    # Require authenticated user (not anonymous)
    if user.user_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for discovery.",
        )

    obo_token = _get_obo_token(request)
    source_types = body.source_types if body else None

    logger.info(
        "REFRESH_REQUEST",
        user_id=user.user_id[:8] if len(user.user_id) > 8 else user.user_id,
        source_types=[t.value for t in source_types] if source_types else "all",
        auth_method="obo" if obo_token else "workspace_client",
    )

    service = get_discovery_service()

    try:
        response = await service.refresh(
            user_id=user.user_id,
            user_token=obo_token,
            source_types=source_types,
        )

        logger.info(
            "REFRESH_RESPONSE",
            total_count=response.total_count,
            error_count=len(response.errors) if response.errors else 0,
        )

        return response

    except Exception as e:
        logger.error("REFRESH_ERROR", error=str(e)[:200])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Refresh failed: {str(e)[:200]}",
        ) from e


@router.get(
    "/stats",
    summary="Get discovery cache statistics",
    description="Returns cache statistics for monitoring.",
    responses={
        200: {"description": "Cache stats retrieved"},
    },
)
async def get_cache_stats(
    user: CurrentUser,  # noqa: ARG001 - Required for auth
) -> dict[str, int | dict[str, int] | float]:
    """Get discovery cache statistics.

    Args:
        user: Current authenticated user.

    Returns:
        Dict with cache statistics.
    """
    cache = get_discovery_cache()
    return await cache.get_stats()
