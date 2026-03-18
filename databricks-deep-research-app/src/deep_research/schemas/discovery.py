"""Discovery schemas for automatic data source discovery.

This module defines Pydantic models for the data source discovery API:
- DiscoveryStatus: Status of discovered sources
- DiscoveredSource: Auto-discovered data source from Databricks workspace
- VectorSearchMetadata: Detailed metadata for Vector Search indexes
- GenieSpaceMetadata: Detailed metadata for Genie spaces
- ServingEndpointMetadata: Detailed metadata for serving endpoints (Knowledge Assistants)
- DiscoveryResponse: API response for discovery endpoints
- DiscoveryError: Partial failure information

API Contract: /specs/007-enterprise-data-sources/contracts/discovery.yaml
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research.schemas.data_source import DataSourceType


class DiscoveryStatus(str, Enum):
    """Status of a discovered data source."""

    READY = "ready"
    """Source is available and ready for queries."""

    SYNCING = "syncing"
    """Source is being synchronized (VS index syncing, etc.)."""

    UNAVAILABLE = "unavailable"
    """Source is temporarily unavailable."""

    ERROR = "error"
    """Source encountered an error during discovery."""


class QueryType(str, Enum):
    """Supported query types for Vector Search."""

    ANN = "ANN"
    """Approximate Nearest Neighbor - fast vector search."""

    HYBRID = "HYBRID"
    """Combines vector + keyword search (max 200 results)."""

    FULL_TEXT = "FULL_TEXT"
    """Keyword-only search, no vectors (max 200 results, beta)."""


# =============================================================================
# Type-Specific Metadata Models
# =============================================================================


class FilterColumnInfo(BaseModel):
    """Information about a filterable column in a Vector Search index."""

    name: str
    """Column name."""

    data_type: Literal["string", "integer", "float", "timestamp", "boolean"]
    """Data type of the column."""

    operators: list[str] = Field(default_factory=lambda: ["=", "!="])
    """Supported filter operators for this column type."""


class VectorSearchMetadata(BaseModel):
    """Metadata specific to a Vector Search index.

    Extracted via:
    - w.vector_search_indexes.get_index(index_name)
    - Delta table schema for filter columns
    """

    index_name: str
    """Fully qualified index name (catalog.schema.index)."""

    endpoint_name: str
    """Vector Search endpoint name."""

    primary_key: str
    """Primary key column name."""

    index_type: Literal["DELTA_SYNC", "DIRECT_ACCESS"]
    """Index type - affects supported features."""

    # Embedding configuration
    embedding_columns: list[str] = Field(default_factory=list)
    """Columns containing embeddings."""

    embedding_dimension: int | None = None
    """Dimension of embedding vectors."""

    embedding_model: str | None = None
    """Model endpoint used for embeddings."""

    queryable_columns: list[str] = Field(default_factory=list)
    """Column names available for query_index().
    Populated during discovery enrichment or on-demand metadata fetch.
    Empty means columns are unknown (tool discovers at query time)."""

    # Query capabilities
    filter_columns: list[FilterColumnInfo] = Field(default_factory=list)
    """Available columns for filtering."""

    supported_query_types: list[QueryType] = Field(default_factory=lambda: [QueryType.ANN])
    """Supported query types (ANN, HYBRID, FULL_TEXT)."""

    supports_reranking: bool = False
    """Whether reranking is available for this index."""

    # Status
    row_count: int | None = None
    """Approximate number of indexed rows."""

    is_ready: bool = True
    """Whether index is ready for queries."""


class GenieSpaceMetadata(BaseModel):
    """Metadata specific to a Genie space.

    Extracted via:
    - w.genie.list_spaces()
    - w.genie.get_space(space_id)
    """

    space_id: str
    """Genie space ID."""

    title: str
    """Space title/name."""

    description: str | None = None
    """Space description."""

    warehouse_id: str | None = None
    """Connected SQL warehouse ID."""

    owner: str | None = None
    """Creator/owner username."""

    created_at: datetime | None = None
    """When the space was created."""

    # Capabilities (always same for Genie)
    capabilities: list[str] = Field(default_factory=lambda: ["sql", "conversation", "follow_up"])
    """Capabilities available for Genie spaces."""


class ServingEndpointMetadata(BaseModel):
    """Metadata specific to a serving endpoint (Knowledge Assistant).

    Extracted via:
    - w.serving_endpoints.list()
    - w.serving_endpoints.get(name)
    """

    endpoint_name: str
    """Serving endpoint name."""

    endpoint_type: str
    """Endpoint type (CUSTOM, EXTERNAL_MODEL, etc.)."""

    state: Literal["READY", "NOT_READY", "PENDING"]
    """Current endpoint state."""

    tags: dict[str, str] = Field(default_factory=dict)
    """Custom tags on the endpoint."""

    is_knowledge_assistant: bool = True
    """Whether this endpoint is identified as a Knowledge Assistant."""

    assistant_type: str | None = None
    """Type of assistant (domain_expert, supervisor, etc.)."""

    creator: str | None = None
    """Creator username."""


# =============================================================================
# Core Discovery Models
# =============================================================================


class DiscoveredSource(BaseModel):
    """A data source discovered via Databricks SDK APIs.

    Represents an auto-discovered source that the user has OBO access to.
    Not persisted - cached in memory with TTL.
    """

    source_id: str
    """Unique identifier (e.g., 'vs:catalog.schema.index', 'genie:space_id')."""

    source_type: DataSourceType
    """Type of data source."""

    name: str
    """Display name for the source."""

    endpoint_name: str
    """Databricks endpoint/space/index identifier."""

    description: str | None = None
    """Optional description."""

    status: DiscoveryStatus = DiscoveryStatus.READY
    """Current status of the source."""

    capabilities: list[str] = Field(default_factory=list)
    """Supported query capabilities (e.g., ['ann', 'hybrid', 'reranking'])."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Type-specific metadata (serialized VectorSearchMetadata, etc.)."""

    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    """When the source was discovered."""

    cached_until: datetime | None = None
    """When the cache entry expires."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class DiscoveryError(BaseModel):
    """Error encountered during discovery of a specific source type."""

    source_type: DataSourceType
    """Which source type failed to discover."""

    error_code: str
    """Error code (e.g., 'PERMISSION_DENIED', 'SERVICE_UNAVAILABLE')."""

    error_message: str
    """Human-readable error message."""

    retryable: bool = True
    """Whether discovery should be retried."""


class DiscoveryResponse(BaseModel):
    """Response from the discovery API.

    GET /api/v1/discovery/sources
    """

    sources: list[DiscoveredSource]
    """All discovered sources."""

    total_count: int
    """Total number of discovered sources."""

    by_type: dict[str, list[DiscoveredSource]] = Field(default_factory=dict)
    """Sources grouped by type for UI convenience."""

    discovered_at: datetime
    """When discovery was performed."""

    cached: bool
    """Whether results came from cache."""

    cache_expires_at: datetime | None = None
    """When cache expires (if cached)."""

    errors: list[DiscoveryError] | None = None
    """Errors during discovery (partial success)."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class SourceMetadataResponse(BaseModel):
    """Detailed metadata response for a specific source.

    GET /api/v1/discovery/sources/{source_id}/metadata
    """

    source: DiscoveredSource
    """The discovered source."""

    # Type-specific metadata (populated based on source type)
    vector_search: VectorSearchMetadata | None = None
    """Vector Search index metadata."""

    genie: GenieSpaceMetadata | None = None
    """Genie space metadata."""

    serving_endpoint: ServingEndpointMetadata | None = None
    """Serving endpoint metadata."""

    # Saved configuration (if user has customized)
    saved_config: dict[str, Any] | None = None
    """User's saved query configuration for this source."""


class RefreshDiscoveryRequest(BaseModel):
    """Request to refresh discovery cache.

    POST /api/v1/discovery/refresh
    """

    source_types: list[DataSourceType] | None = None
    """Source types to refresh. None = refresh all."""
