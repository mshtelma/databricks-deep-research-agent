# Implementation Plan: Enterprise Data Sources & Data Source Discovery

**Branch**: `007-enterprise-data-sources` | **Date**: 2026-02-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/007-enterprise-data-sources/spec.md`

## Summary

This plan covers the implementation of enterprise data source integrations (Vector Search, Genie, Knowledge Assistants) with automatic discovery, per-source query configuration (ANN/Hybrid/Full-Text), and UI for source selection. The implementation follows existing tool patterns (ResearchTool protocol) with OBO authentication via `ModelServingUserCredentials`.

**Key Technical Decisions:**
- Use `WorkspaceClient` with `ModelServingUserCredentials()` for OBO authentication
- Extend existing `ToolRegistry` and `ResearchTool` protocol for new data sources
- Leverage existing `SourceScopeConfig` and `ManualStepDefinition` schemas
- Implement discovery service with caching (5-minute TTL)
- Frontend uses TanStack Query for data fetching with SSE for streaming

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**:
- Backend: FastAPI, SQLAlchemy (async), Pydantic, databricks-sdk, databricks-ai-bridge
- Frontend: React 18, TanStack Query, Tailwind CSS
**Storage**: PostgreSQL (Databricks Lakebase)
**Testing**: pytest (unit, integration, complex markers), Vitest (frontend)
**Target Platform**: Databricks Apps (containerized deployment)
**Project Type**: Web application (backend + frontend)
**Performance Goals**:
- Discovery: < 3 seconds for 50 sources
- Cache hit rate: > 80%
- Parallel tool execution: 30-50% latency reduction
**Constraints**:
- OBO token refresh: 1-hour lifetime
- Genie rate limit: 5 queries/minute/workspace (preview)
- Vector Search filter limit: 1,024 IDs per clause
**Scale/Scope**: 10-100 data sources per workspace, 100+ concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Clients & Workspace Integration | ✅ PASS | All Databricks API calls use `WorkspaceClient` with OBO credentials |
| II. Typing-First Python | ✅ PASS | All new code uses full type annotations (Pydantic models, typed dicts) |
| III. Avoid Runtime Introspection | ✅ PASS | Uses `typing.Protocol` for tool interfaces, Pydantic for validation |
| IV. Linting & Static Type Enforcement | ✅ PASS | mypy strict + ruff required before merge |

## Project Structure

### Documentation (this feature)

```text
specs/007-enterprise-data-sources/
├── plan.md              # This file
├── research.md          # Phase 0 output - API research findings
├── data-model.md        # Phase 1 output - Entity definitions
├── quickstart.md        # Phase 1 output - Developer guide
├── contracts/           # Phase 1 output - OpenAPI schemas
│   ├── data-sources.yaml
│   ├── discovery.yaml
│   └── query-config.yaml
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
# Backend (Python)
src/deep_research/
├── agent/
│   ├── tools/
│   │   ├── base.py                    # Existing ResearchTool protocol
│   │   ├── registry.py                # Existing ToolRegistry
│   │   ├── factory.py                 # EXTEND: Discovery-based tool creation
│   │   ├── vector_search.py           # EXTEND: Query type configuration
│   │   ├── user_vector_search.py      # EXTEND: Filter expressions
│   │   ├── genie.py                   # Existing GenieTool
│   │   ├── knowledge_assistant.py     # Existing KnowledgeAssistantTool
│   │   └── discovery.py               # NEW: Data source discovery service
│   ├── state.py                       # EXTEND: Discovery state tracking
│   └── nodes/
│       ├── source_routing.py          # EXTEND: Query type handling
│       └── background.py              # EXTEND: Pre-planning discovery
├── api/v1/
│   ├── data_sources.py                # EXTEND: Discovery endpoints
│   └── discovery.py                   # NEW: Real-time discovery API
├── models/
│   └── data_source.py                 # EXTEND: Query configuration storage
├── schemas/
│   ├── data_source.py                 # EXTEND: Discovery response types
│   ├── query_config.py                # NEW: Query type configuration
│   └── discovery.py                   # NEW: Discovery result schemas
└── services/
    ├── data_source_service.py         # EXTEND: Discovery caching
    ├── discovery_service.py           # NEW: Centralized discovery logic
    └── obo_client.py                  # EXTEND: OBO token management

# Frontend (TypeScript/React)
frontend/src/
├── api/
│   └── dataSources.ts                 # EXTEND: Discovery API client
├── components/
│   └── sources/
│       ├── DataSourceSelector.tsx     # NEW: Main dropdown component
│       ├── SourceList.tsx             # NEW: Grouped source list
│       ├── SourceCard.tsx             # NEW: Individual source display
│       ├── QueryConfigPanel.tsx       # NEW: ANN/Hybrid/Filter config
│       └── FilterBuilder.tsx          # NEW: Filter expression builder
├── hooks/
│   ├── useDataSources.ts              # EXTEND: Discovery hooks
│   └── useQueryConfig.ts              # NEW: Per-source config hooks
└── types/
    └── dataSources.ts                 # EXTEND: Discovery types

# Tests
tests/
├── unit/
│   ├── agent/tools/test_discovery.py  # NEW: Discovery service tests
│   └── services/test_discovery_service.py  # NEW: Caching tests
├── integration/
│   └── agent/tools/test_enterprise_sources.py  # NEW: Live API tests
└── complex/
    └── test_multi_source_research.py  # NEW: End-to-end with discovery

frontend/src/__tests__/
└── components/sources/                # NEW: Component tests
```

**Structure Decision**: Web application pattern with existing backend/frontend separation. All new code follows established patterns in existing files.

## Complexity Tracking

No constitution violations requiring justification.

## Implementation Phases

### Phase 1: Discovery Service & Backend APIs (F13)

**Goal**: Implement data source discovery via Databricks SDK with caching.

**Components**:
1. `DiscoveryService` class with async methods:
   - `discover_vector_search_sources()` - List endpoints → list indexes per endpoint → get metadata
   - `discover_genie_spaces()` - List spaces → get space details
   - `discover_serving_endpoints()` - List endpoints → filter by type/tags
   - `discover_all()` - Parallel execution of all discovery methods

2. `DiscoveryCache` with TTL management:
   - 5-minute default TTL
   - Per-user cache keys (different OBO tokens)
   - Background refresh on near-expiry

3. API endpoints:
   - `GET /api/v1/discovery/sources` - Returns all discoverable sources
   - `GET /api/v1/discovery/sources/{source_id}/metadata` - Detailed metadata
   - `POST /api/v1/discovery/refresh` - Force cache refresh

**Key Types**:
```python
class DiscoveredSource(BaseModel):
    source_type: DataSourceType  # vector_search, genie, knowledge_assistant
    name: str
    endpoint_name: str
    description: str | None
    status: str  # ready, syncing, unavailable
    capabilities: list[str]  # ["ann", "hybrid", "full_text"]
    metadata: dict[str, Any]  # Type-specific metadata

class VectorSearchMetadata(BaseModel):
    index_name: str
    endpoint_name: str
    primary_key: str
    index_type: str  # delta_sync, direct_access
    embedding_columns: list[str]
    filter_columns: list[str]
    supported_query_types: list[str]  # ["ANN", "HYBRID", "FULL_TEXT"]

class GenieSpaceMetadata(BaseModel):
    space_id: str
    title: str
    description: str | None
    warehouse_id: str
    owner: str | None

class ServingEndpointMetadata(BaseModel):
    endpoint_name: str
    endpoint_type: str
    state: str
    tags: dict[str, str]
    creator: str | None
```

### Phase 2: Query Type Configuration (F14)

**Goal**: Enable per-source query configuration with UI.

**Components**:
1. `QueryConfig` model for storage:
   - Associated with `UserDataSource` or session-scoped
   - Stores query_type, default_filters, num_results, score_threshold

2. Filter expression validation:
   - SQL-like syntax parser for storage-optimized endpoints
   - Dictionary syntax validation for standard endpoints
   - Column type → operator mapping

3. API endpoints:
   - `PUT /api/v1/data-sources/{id}/query-config` - Save configuration
   - `GET /api/v1/data-sources/{id}/query-config` - Retrieve configuration

**Key Types**:
```python
class QueryTypeConfig(BaseModel):
    query_type: Literal["ANN", "HYBRID", "FULL_TEXT"] = "ANN"
    num_results: int = 10
    score_threshold: float | None = None
    columns: list[str] | None = None  # Columns to return

class FilterExpression(BaseModel):
    column: str
    operator: str  # =, !=, <, >, <=, >=, LIKE, IN
    value: str | int | float | list[str]

class VectorSearchQueryConfig(BaseModel):
    query_type: QueryTypeConfig
    filters: list[FilterExpression]
    filter_syntax: Literal["sql", "dict"] = "sql"
```

### Phase 3: Frontend Data Source Selector

**Goal**: Build UI for source discovery, selection, and configuration.

**Components**:
1. `DataSourceSelector` - Main dropdown component
   - Grouped by source type
   - Search/filter functionality
   - Multi-select support

2. `SourceCard` - Individual source display
   - Name, type icon, description
   - Status indicator (available/syncing/unavailable)
   - Expandable metadata panel

3. `QueryConfigPanel` - Configuration UI
   - Query type radio buttons (ANN/Hybrid/Full-Text)
   - Filter builder integration
   - Save as default toggle

4. `FilterBuilder` - Filter expression builder
   - Column dropdown (populated from metadata)
   - Operator dropdown (based on column type)
   - Value input (type-aware)

### Phase 4: Tool Integration

**Goal**: Connect discovery to tool creation and execution.

**Components**:
1. Extend `factory.py` to use discovered sources:
   - Accept `DiscoveredSource` objects
   - Apply query configuration from session/user prefs
   - Handle discovery failures gracefully

2. Extend tools with query type support:
   - `UserVectorSearchTool.execute()` respects `query_type` setting
   - Filter expressions passed to SDK `query_index()` call

3. Source routing integration:
   - `source_routing.py` filters tools by discovery status
   - Unavailable sources excluded from tool calls

---

## Databricks SDK Integration Notes

### Vector Search Discovery Pattern

```python
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials

async def discover_vector_search(user_token: str | None) -> list[DiscoveredSource]:
    w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

    sources = []
    # 1. List all VS endpoints
    for endpoint in w.vector_search_endpoints.list_endpoints():
        # 2. List indexes per endpoint
        for mini_index in w.vector_search_indexes.list_indexes(endpoint.name):
            # 3. Get full index metadata
            index = w.vector_search_indexes.get_index(mini_index.name)

            sources.append(DiscoveredSource(
                source_type=DataSourceType.VECTOR_SEARCH,
                name=index.name,
                endpoint_name=endpoint.name,
                status="ready" if index.status.ready else "syncing",
                capabilities=_determine_capabilities(index),
                metadata=VectorSearchMetadata(
                    index_name=index.name,
                    endpoint_name=endpoint.name,
                    primary_key=index.primary_key,
                    index_type=index.index_type.value,
                    embedding_columns=_extract_embedding_columns(index),
                    filter_columns=_extract_filter_columns(index),
                    supported_query_types=_get_supported_query_types(index),
                ).model_dump(),
            ))
    return sources
```

### Genie Discovery Pattern

```python
async def discover_genie_spaces(user_token: str | None) -> list[DiscoveredSource]:
    w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

    sources = []
    # 1. List all Genie spaces
    response = w.genie.list_spaces()
    for space_summary in response.spaces:
        # 2. Get space details
        space = w.genie.get_space(space_summary.id)

        sources.append(DiscoveredSource(
            source_type=DataSourceType.GENIE,
            name=space.title or space_summary.id,
            endpoint_name=space_summary.id,  # space_id used as identifier
            description=space.description,
            status="ready",
            capabilities=["sql", "conversation"],
            metadata=GenieSpaceMetadata(
                space_id=space_summary.id,
                title=space.title,
                description=space.description,
                warehouse_id=space.warehouse_id,
                owner=space.creator,
            ).model_dump(),
        ))
    return sources
```

### Serving Endpoints Discovery (Knowledge Assistants)

```python
async def discover_serving_endpoints(user_token: str | None) -> list[DiscoveredSource]:
    w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

    sources = []
    # List all serving endpoints
    for endpoint in w.serving_endpoints.list():
        # Filter: only include likely Knowledge Assistants
        # Heuristics: tags, name patterns, endpoint type
        if not _is_knowledge_assistant(endpoint):
            continue

        sources.append(DiscoveredSource(
            source_type=DataSourceType.KNOWLEDGE_ASSISTANT,
            name=endpoint.name,
            endpoint_name=endpoint.name,
            status=endpoint.state.value if endpoint.state else "unknown",
            capabilities=["chat", "context"],
            metadata=ServingEndpointMetadata(
                endpoint_name=endpoint.name,
                endpoint_type=str(endpoint.endpoint_type) if endpoint.endpoint_type else "unknown",
                state=endpoint.state.value if endpoint.state else "unknown",
                tags=dict(endpoint.tags) if endpoint.tags else {},
                creator=endpoint.creator,
            ).model_dump(),
        ))
    return sources
```

### Query Execution with Configuration

```python
# In UserVectorSearchTool.execute()
async def execute(self, arguments: dict, context: ResearchContext) -> ToolResult:
    query = arguments["query"]
    config = context.query_config.get(self.source_name, QueryTypeConfig())

    # Build query parameters based on configuration
    query_params = {
        "index_name": self.index_name,
        "query_text": query,
        "columns": config.columns or self.default_columns,
        "num_results": config.num_results,
        "query_type": config.query_type,  # "ANN", "HYBRID", or "FULL_TEXT"
    }

    # Apply filters if configured
    if config.filters:
        if config.filter_syntax == "sql":
            query_params["filters_json"] = _build_sql_filter(config.filters)
        else:
            query_params["filters"] = _build_dict_filter(config.filters)

    # Execute via WorkspaceClient
    response = await self._client.query_index(**query_params)

    return ToolResult(
        content=_format_results(response),
        success=True,
        sources=[_to_source_info(r) for r in response.result.data_array],
    )
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OBO token expiry during long research | Refresh token before each tool execution batch |
| Discovery API rate limits | 5-minute cache with background refresh |
| Large number of sources (>50) | Pagination in API, virtual scrolling in UI |
| Genie rate limit (5 QPM) | Queue management with backpressure |
| Filter syntax errors | Real-time validation with helpful error messages |

---

## Dependencies

- **databricks-sdk**: Vector Search, Genie, Serving Endpoints APIs
- **databricks-ai-bridge**: `ModelServingUserCredentials` for OBO
- **Existing infrastructure**: ToolRegistry, ResearchTool protocol, SourceScopeConfig

---

## Next Steps

1. Run `/speckit.tasks` to generate implementation tasks
2. Implement Phase 1 (Discovery Service) first - foundation for all other phases
3. Write integration tests with live Databricks APIs early
