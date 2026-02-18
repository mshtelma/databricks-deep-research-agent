"""Data source schemas for enterprise data source integration.

This module defines Pydantic models for:
- DataSourceType: Types of queryable data sources
- DataSourceDefinition: Definition of a data source's capabilities
- SourceConstraints: Constraints on which sources can be used
- Request/Response schemas for data source API endpoints
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DataSourceType(str, Enum):
    """Types of queryable data sources.

    Each type has different capabilities and query patterns:
    - VECTOR_SEARCH: Semantic similarity search with optional filtering
    - GENIE: Natural language to SQL for analytics
    - KNOWLEDGE_ASSISTANT: Domain expert Q&A
    - WEB_SEARCH: External web search (Brave)
    - UPLOADED_FILE: User-uploaded documents
    - CUSTOM: Plugin-provided custom sources
    """

    VECTOR_SEARCH = "vector_search"
    GENIE = "genie"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"
    WEB_SEARCH = "web_search"
    UPLOADED_FILE = "uploaded_file"
    CUSTOM = "custom"


class DataSourceCapability(str, Enum):
    """Capabilities that data sources can have."""

    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    METADATA_FILTERING = "metadata_filtering"
    SQL_ANALYTICS = "sql_analytics"
    AGGREGATIONS = "aggregations"
    FOLLOW_UP = "follow_up"
    DOMAIN_EXPERTISE = "domain_expertise"
    CURRENT_EVENTS = "current_events"
    DOCUMENT_SEARCH = "document_search"


class DataSourceVisibility(str, Enum):
    """Visibility levels for user-configured data sources."""

    PRIVATE = "private"  # Only creator can see/use
    WORKSPACE = "workspace"  # All workspace users (with OBO access)


class DataSourceValidationStatus(str, Enum):
    """Validation status for user data sources."""

    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"


# =============================================================================
# Core Definition Models
# =============================================================================


class DataSourceDefinition(BaseModel):
    """Definition of a queryable data source.

    Used to describe data sources in the source browser and
    for tool generation.
    """

    type: DataSourceType
    """Type of data source."""

    name: str
    """Unique name for the source (used in tool naming)."""

    description: str
    """Human-readable description for UI and LLM."""

    endpoint_identifier: str
    """Identifier for the endpoint (VS endpoint, Genie space ID, etc.)."""

    capabilities: list[DataSourceCapability] = Field(default_factory=list)
    """List of capabilities this source supports."""

    filter_schema: dict[str, Any] | None = None
    """JSON Schema for metadata filters (for VS sources)."""

    example_queries: list[str] = Field(default_factory=list)
    """Example queries to show in UI and include in tool descriptions."""

    source: str = "system"
    """Origin of the source: 'system', 'plugin:{name}', 'user:{id}'."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class SourceConstraints(BaseModel):
    """Constraints on which sources can be used.

    Applied at step level to control source routing.
    """

    allowed_types: set[DataSourceType] | None = None
    """If set, only these source types are allowed. None = all allowed."""

    allowed_sources: list[str] | None = None
    """If set, only these specific sources (by name) are allowed."""

    required_sources: list[str] = Field(default_factory=list)
    """Sources that MUST be consulted (step incomplete without them)."""

    excluded_sources: list[str] = Field(default_factory=list)
    """Sources that cannot be used."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


# =============================================================================
# API Request Schemas
# =============================================================================


class CreateVectorSearchSourceRequest(BaseModel):
    """Request to create a Vector Search data source."""

    name: str = Field(..., min_length=1, max_length=255)
    """Display name for the source."""

    description: str | None = None
    """Optional description."""

    endpoint_name: str = Field(..., min_length=1, max_length=255)
    """Databricks Vector Search endpoint name."""

    index_name: str = Field(..., min_length=1, max_length=500)
    """Fully qualified index name (catalog.schema.index)."""

    visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE
    """Visibility level."""

    # Optional configuration overrides (auto-detected from index if not provided)
    enable_hybrid: bool = True
    """Enable hybrid search (BM25 + vectors)."""

    enable_reranking: bool = True
    """Enable reranking for improved relevance."""

    num_results: int = Field(default=10, ge=1, le=100)
    """Default number of results to return."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class CreateGenieSourceRequest(BaseModel):
    """Request to create a Genie data source."""

    name: str = Field(..., min_length=1, max_length=255)
    """Display name for the source."""

    description: str | None = None
    """Optional description of what data is available."""

    space_id: str = Field(..., min_length=1, max_length=255)
    """Genie space ID."""

    example_questions: list[str] = Field(default_factory=list, max_length=10)
    """Example questions to show in UI."""

    visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE
    """Visibility level."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class CreateKnowledgeAssistantSourceRequest(BaseModel):
    """Request to create a Knowledge Assistant data source."""

    name: str = Field(..., min_length=1, max_length=255)
    """Display name for the source."""

    description: str | None = None
    """Description of the assistant's expertise."""

    endpoint_name: str = Field(..., min_length=1, max_length=255)
    """Serving endpoint name for the assistant."""

    pass_context: bool = True
    """Whether to pass research context to the assistant."""

    visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE
    """Visibility level."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class UpdateDataSourceRequest(BaseModel):
    """Request to update a data source."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    visibility: DataSourceVisibility | None = None

    # Vector Search specific
    enable_hybrid: bool | None = None
    enable_reranking: bool | None = None
    num_results: int | None = Field(None, ge=1, le=100)

    # Genie specific
    example_questions: list[str] | None = Field(None, max_length=10)

    # Knowledge Assistant specific
    pass_context: bool | None = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


# =============================================================================
# API Response Schemas
# =============================================================================


class DataSourceConfig(BaseModel):
    """Type-specific configuration for a data source (from JSONB)."""

    # Vector Search
    endpoint_name: str | None = None
    index_name: str | None = None
    columns: list[str] | None = None
    columns_to_rerank: list[str] | None = None
    enable_hybrid: bool | None = None
    enable_reranking: bool | None = None
    num_results: int | None = None

    # Genie
    space_id: str | None = None
    example_questions: list[str] | None = None

    # Knowledge Assistant
    pass_context: bool | None = None


class DataSourceResponse(BaseModel):
    """Response schema for a single data source."""

    id: UUID
    owner_id: str
    type: DataSourceType
    name: str
    description: str | None
    endpoint_identifier: str
    config: DataSourceConfig
    visibility: DataSourceVisibility
    validation_status: DataSourceValidationStatus
    last_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime

    # Derived fields
    capabilities: list[DataSourceCapability] = Field(default_factory=list)
    source_origin: str = "user"  # 'system', 'plugin', 'user'

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        use_enum_values = True


class DataSourceListResponse(BaseModel):
    """Response schema for listing data sources."""

    sources: list[DataSourceResponse]
    total: int
    user_sources: int
    workspace_sources: int

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class DataSourceValidationResponse(BaseModel):
    """Response from data source validation."""

    source_id: UUID
    has_access: bool
    error_message: str | None = None
    validated_at: datetime

    # Schema info (for VS sources)
    detected_columns: list[str] | None = None
    detected_text_columns: list[str] | None = None
