# Data Model: Enterprise Data Sources & Discovery

**Date**: 2026-02-04
**Feature**: 007-enterprise-data-sources

---

## Overview

This document defines the data model for enterprise data source discovery, configuration, and query settings. It extends the existing `UserDataSource` model with discovery and query configuration capabilities.

---

## 1. Core Entities

### 1.1 DiscoveredSource (Schema - Transient)

Represents a data source discovered via Databricks SDK APIs. Not persisted - cached in memory with TTL.

```python
class DiscoveredSource(BaseModel):
    """Auto-discovered data source from Databricks workspace."""

    # Identity
    source_id: str                      # Unique identifier (e.g., "vs:index_name" or "genie:space_id")
    source_type: DataSourceType         # vector_search, genie, knowledge_assistant
    name: str                           # Display name
    endpoint_name: str                  # VS endpoint, Genie space_id, or serving endpoint name

    # Metadata
    description: str | None = None
    status: DiscoveryStatus             # ready, syncing, unavailable, error
    capabilities: list[str]             # ["ann", "hybrid", "full_text", "reranking"]

    # Type-specific metadata (polymorphic)
    metadata: VectorSearchMetadata | GenieSpaceMetadata | ServingEndpointMetadata

    # Discovery metadata
    discovered_at: datetime
    cached_until: datetime

class DiscoveryStatus(str, Enum):
    READY = "ready"
    SYNCING = "syncing"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
```

### 1.2 VectorSearchMetadata (Schema)

Metadata specific to Vector Search indexes.

```python
class VectorSearchMetadata(BaseModel):
    """Metadata for a Vector Search index."""

    # Identity
    index_name: str                     # Full name: catalog.schema.index
    endpoint_name: str                  # VS endpoint name

    # Schema
    primary_key: str                    # Primary key column
    index_type: str                     # "DELTA_SYNC" or "DIRECT_ACCESS"

    # Embedding configuration
    embedding_columns: list[str]        # Columns with embeddings
    embedding_dimension: int | None     # Vector dimension
    embedding_model: str | None         # Model endpoint name

    # Query capabilities
    filter_columns: list[FilterColumnInfo]  # Available filter columns
    supported_query_types: list[QueryType]  # ["ANN", "HYBRID", "FULL_TEXT"]
    supports_reranking: bool            # Whether reranking is available

    # Status
    row_count: int | None               # Number of indexed rows
    is_ready: bool                      # Ready for queries

class FilterColumnInfo(BaseModel):
    """Information about a filterable column."""
    name: str
    data_type: str                      # "string", "integer", "float", "timestamp", "boolean"
    operators: list[str]                # ["=", "!=", "<", ">", "LIKE", "IN"]
```

### 1.3 GenieSpaceMetadata (Schema)

Metadata specific to Genie spaces.

```python
class GenieSpaceMetadata(BaseModel):
    """Metadata for a Genie space."""

    # Identity
    space_id: str                       # Genie space ID
    title: str                          # Space title

    # Configuration
    description: str | None
    warehouse_id: str | None            # Connected SQL warehouse

    # Access
    owner: str | None                   # Creator username
    created_at: datetime | None

    # Capabilities (always same for Genie)
    capabilities: list[str] = ["sql", "conversation", "follow_up"]
```

### 1.4 ServingEndpointMetadata (Schema)

Metadata specific to serving endpoints (Knowledge Assistants).

```python
class ServingEndpointMetadata(BaseModel):
    """Metadata for a serving endpoint (Knowledge Assistant)."""

    # Identity
    endpoint_name: str
    endpoint_type: str                  # "CUSTOM", "EXTERNAL_MODEL", etc.

    # Status
    state: str                          # "READY", "NOT_READY", "PENDING"

    # Classification
    tags: dict[str, str]                # Custom tags
    is_knowledge_assistant: bool        # Manual or heuristic classification
    assistant_type: str | None          # "domain_expert", "supervisor", etc.

    # Access
    creator: str | None
```

---

## 2. Configuration Entities

### 2.1 VectorSearchQueryConfig (Schema - Persisted)

Per-source query configuration for Vector Search indexes.

```python
class VectorSearchQueryConfig(BaseModel):
    """Query configuration for a Vector Search index."""

    # Query type
    query_type: QueryType = QueryType.ANN  # ANN, HYBRID, FULL_TEXT

    # Results
    num_results: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    columns: list[str] | None = None    # Columns to return (None = all)

    # Reranking
    enable_reranking: bool = False
    columns_to_rerank: list[str] | None = None

    # Filters
    filters: list[FilterExpression] = []
    filter_syntax: FilterSyntax = FilterSyntax.SQL  # SQL or DICT

class QueryType(str, Enum):
    ANN = "ANN"                         # Approximate nearest neighbor
    HYBRID = "HYBRID"                   # Vector + keyword (max 200 results)
    FULL_TEXT = "FULL_TEXT"             # Keyword only (max 200 results, beta)

class FilterSyntax(str, Enum):
    SQL = "sql"                         # SQL-like: "col = 'val' AND col2 > 10"
    DICT = "dict"                       # Dictionary: {"col": "val", "col2 >": 10}
```

### 2.2 FilterExpression (Schema)

Individual filter expression for Vector Search queries.

```python
class FilterExpression(BaseModel):
    """A single filter expression for Vector Search."""

    column: str                         # Column name
    operator: FilterOperator            # Comparison operator
    value: str | int | float | list[str | int | float]  # Filter value

    def to_sql(self) -> str:
        """Convert to SQL-like filter string."""
        if self.operator == FilterOperator.IN:
            values = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in self.value)
            return f"{self.column} IN ({values})"
        elif self.operator == FilterOperator.LIKE:
            return f"{self.column} LIKE '{self.value}'"
        else:
            val = f"'{self.value}'" if isinstance(self.value, str) else self.value
            return f"{self.column} {self.operator.value} {val}"

    def to_dict(self) -> dict:
        """Convert to dictionary filter format."""
        if self.operator == FilterOperator.EQ:
            return {self.column: self.value}
        return {f"{self.column} {self.operator.value}": self.value}

class FilterOperator(str, Enum):
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
```

---

## 3. Storage Models

### 3.1 UserDataSource Extensions (Model)

Extend existing `UserDataSource` model to store query configuration.

```python
class UserDataSource(BaseModel):
    """User-configured data source connection."""

    # Existing fields
    id: UUID
    owner_id: str
    type: DataSourceType
    name: str
    description: str | None
    endpoint_identifier: str            # Index name, space ID, etc.
    visibility: Visibility
    validation_status: ValidationStatus
    last_validated_at: datetime | None

    # EXTENDED: Store query configuration
    config: dict[str, Any]              # Type-specific config (JSONB)

    # For Vector Search, config includes:
    # {
    #     "endpoint_name": "vs-endpoint-prod",
    #     "index_name": "catalog.schema.index",
    #     "columns": [...],
    #     "columns_to_rerank": [...],
    #     "query_config": {
    #         "query_type": "ANN",
    #         "num_results": 10,
    #         "filters": [...],
    #         ...
    #     }
    # }
```

### 3.2 DiscoveryCache (Service - In-Memory)

Cache for discovered sources with TTL management.

```python
@dataclass
class DiscoveryCache:
    """In-memory cache for discovered data sources."""

    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    default_ttl: timedelta = timedelta(minutes=5)

    @dataclass
    class CacheEntry:
        sources: list[DiscoveredSource]
        created_at: datetime
        expires_at: datetime

    async def get(self, user_id: str, source_type: DataSourceType | None = None) -> list[DiscoveredSource] | None:
        """Get cached sources for user, optionally filtered by type."""
        key = self._make_key(user_id, source_type)
        async with self._lock:
            entry = self._cache.get(key)
            if entry and entry.expires_at > datetime.utcnow():
                return entry.sources
            return None

    async def set(self, user_id: str, sources: list[DiscoveredSource], source_type: DataSourceType | None = None) -> None:
        """Cache discovered sources for user."""
        key = self._make_key(user_id, source_type)
        now = datetime.utcnow()
        async with self._lock:
            self._cache[key] = self.CacheEntry(
                sources=sources,
                created_at=now,
                expires_at=now + self.default_ttl,
            )

    def _make_key(self, user_id: str, source_type: DataSourceType | None) -> str:
        type_suffix = f":{source_type.value}" if source_type else ":all"
        return f"discovery:{user_id}{type_suffix}"
```

---

## 4. API Response Types

### 4.1 Discovery Response

```python
class DiscoveryResponse(BaseModel):
    """Response for GET /api/v1/discovery/sources."""

    sources: list[DiscoveredSource]
    total_count: int

    # Grouped by type for UI convenience
    by_type: dict[DataSourceType, list[DiscoveredSource]]

    # Discovery metadata
    discovered_at: datetime
    cached: bool
    cache_expires_at: datetime | None

    # Errors (partial success)
    errors: list[DiscoveryError] | None

class DiscoveryError(BaseModel):
    """Error during discovery of a specific source type."""
    source_type: DataSourceType
    error_code: str
    error_message: str
    retryable: bool
```

### 4.2 Source Metadata Response

```python
class SourceMetadataResponse(BaseModel):
    """Response for GET /api/v1/discovery/sources/{id}/metadata."""

    source: DiscoveredSource

    # Expanded metadata based on type
    vector_search: VectorSearchMetadata | None
    genie: GenieSpaceMetadata | None
    serving_endpoint: ServingEndpointMetadata | None

    # Configuration if saved
    saved_config: VectorSearchQueryConfig | None
```

---

## 5. State Extensions

### 5.1 ResearchState Extensions

Add discovery state tracking to `ResearchState`.

```python
@dataclass
class ResearchState:
    """Extended with discovery state."""

    # Existing fields...

    # NEW: Discovery state
    discovered_sources: list[DiscoveredSource] | None = None
    discovery_completed_at: datetime | None = None

    # NEW: Per-source query configurations (session-scoped)
    query_configs: dict[str, VectorSearchQueryConfig] = field(default_factory=dict)

    def get_query_config(self, source_id: str) -> VectorSearchQueryConfig:
        """Get query config for a source, with defaults."""
        return self.query_configs.get(source_id, VectorSearchQueryConfig())

    def set_query_config(self, source_id: str, config: VectorSearchQueryConfig) -> None:
        """Set query config for a source."""
        self.query_configs[source_id] = config
```

### 5.2 ResearchContext Extensions

Add discovery-related context for tool execution.

```python
@dataclass
class ResearchContext:
    """Extended with discovery context."""

    # Existing fields...

    # NEW: Discovered sources available for this research
    available_sources: list[DiscoveredSource] | None = None

    # NEW: Query configurations (merged from session + user defaults)
    query_configs: dict[str, VectorSearchQueryConfig] = field(default_factory=dict)
```

---

## 6. Entity Relationships

```
┌─────────────────────┐
│   DiscoveryCache    │  (In-memory, TTL-managed)
│   user_id → sources │
└─────────┬───────────┘
          │ populates
          ▼
┌─────────────────────┐     ┌─────────────────────────┐
│  DiscoveredSource   │────▶│  *Metadata (polymorphic) │
│  - source_id        │     │  - VectorSearchMetadata  │
│  - source_type      │     │  - GenieSpaceMetadata    │
│  - capabilities     │     │  - ServingEndpointMeta.  │
└─────────┬───────────┘     └─────────────────────────┘
          │
          │ user selects
          ▼
┌─────────────────────┐
│   UserDataSource    │  (Persisted when user saves)
│   - endpoint_id     │
│   - config (JSONB)  │──────▶ VectorSearchQueryConfig
└─────────────────────┘        - query_type
                               - filters[]
                               - num_results
```

---

## 7. Validation Rules

### 7.1 Filter Expression Validation

```python
def validate_filter_expression(expr: FilterExpression, column_info: FilterColumnInfo) -> list[str]:
    """Validate a filter expression against column metadata."""
    errors = []

    # Check operator is valid for column type
    if expr.operator not in column_info.operators:
        errors.append(f"Operator {expr.operator} not supported for column {expr.column} (type: {column_info.data_type})")

    # Check value type matches column type
    if column_info.data_type == "integer" and not isinstance(expr.value, (int, list)):
        errors.append(f"Column {expr.column} requires integer value")

    # Check IN value count
    if expr.operator == FilterOperator.IN and isinstance(expr.value, list):
        if len(expr.value) > 1024:
            errors.append(f"IN filter exceeds 1024 ID limit (got {len(expr.value)})")

    return errors
```

### 7.2 Query Config Validation

```python
def validate_query_config(config: VectorSearchQueryConfig, metadata: VectorSearchMetadata) -> list[str]:
    """Validate query configuration against index capabilities."""
    errors = []

    # Check query type is supported
    if config.query_type not in metadata.supported_query_types:
        errors.append(f"Query type {config.query_type} not supported by index")

    # Check reranking columns exist
    if config.enable_reranking and config.columns_to_rerank:
        available = {col.name for col in metadata.filter_columns}
        missing = set(config.columns_to_rerank) - available
        if missing:
            errors.append(f"Reranking columns not found: {missing}")

    # Check result limit for hybrid/full-text
    if config.query_type in [QueryType.HYBRID, QueryType.FULL_TEXT] and config.num_results > 200:
        errors.append(f"{config.query_type} limited to 200 results (requested: {config.num_results})")

    return errors
```

---

## 8. Migration Notes

No database migrations required for discovery (transient data). Query config stored in existing `UserDataSource.config` JSONB column.

For frontend state, add to session storage:
```typescript
interface SessionState {
  discoveredSources: DiscoveredSource[];
  queryConfigs: Record<string, VectorSearchQueryConfig>;
}
```
