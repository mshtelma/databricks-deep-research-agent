# Tasks: Enterprise Data Sources & Custom Research Workflows

**Input**: Design documents from `/specs/007-enterprise-data-sources/`
**Prerequisites**: plan.md (completed), spec.md (completed), research.md, data-model.md, contracts/discovery.yaml, quickstart.md
**Feature Branch**: `007-enterprise-data-sources`

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. User stories are ordered by priority (P1 first, then P2, then P3).

**New in this version**: Added US9a (Data Source Discovery) and US9b (Query Configuration) from spec.md updates.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US10)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `src/deep_research/`
- **Frontend**: `frontend/src/`
- **Tests**: `tests/`
- **Migrations**: `src/deep_research/db/versions/`

---

## Phase 1: Foundation (Shared Infrastructure)

**Purpose**: Establish core infrastructure that ALL user stories depend on. Must complete before any story work.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### 1.1 OBO Client Service (All Enterprise Sources Depend on This)

- [x] T001 [US1,US2,US10] Create OBO Databricks client service in `src/deep_research/services/obo_client.py`
  - Implement `OBODatabricksClient` class with `get_client(user_token)` method
  - Add token exchange using existing `get_user_workspace_client()`
  - Add access validation methods: `validate_vector_search_access()`, `validate_genie_access()`, `validate_assistant_access()`
  - Cache validated access for session duration

- [x] T002 [US1,US2,US10] Fix OBO token preservation in middleware `src/deep_research/core/middleware/auth.py`
  - Store OBO token in `request.state.obo_token`
  - Store user workspace client in `request.state.user_workspace_client`

- [x] T003 [US1,US2,US10] Add `user_token` field to `ResearchContext` in `src/deep_research/agent/state.py`
  - Add `user_token: str | None = None` field
  - Update all context creation points to pass through OBO token

### 1.2 New Plugin Protocols

- [x] T004 [P] [US8] Create `DataSourceProvider` protocol in `src/deep_research/plugins/base.py`
  - Define `get_data_sources()` and `get_source_constraints()` methods

- [x] T005 [P] [US8] Create `TemplateProvider` protocol in `src/deep_research/plugins/base.py`
  - Define `get_templates()` and `resolve_variable()` methods

- [x] T006 [P] [US8] Create `CustomAgentProvider` protocol in `src/deep_research/plugins/base.py`
  - Define `get_custom_agents()` method

- [x] T007 [P] [US8] Create `FileProcessorProvider` protocol in `src/deep_research/plugins/base.py`
  - Define `get_supported_extensions()` and `process_file()` methods

### 1.3 Data Source Definition Models

- [x] T008 [US1,US2,US9] Create data source schemas in `src/deep_research/schemas/data_source.py`
  - Define `DataSourceType` enum (VECTOR_SEARCH, GENIE, KNOWLEDGE_ASSISTANT, WEB_SEARCH, UPLOADED_FILE, CUSTOM)
  - Define `DataSourceDefinition` Pydantic model
  - Define `SourceConstraints` Pydantic model

### 1.4 Lifecycle Events

- [x] T009 [P] [US8] Add new lifecycle events in `src/deep_research/plugins/lifecycle/events.py`
  - Add `DataSourceQueryEvent` dataclass
  - Add `TemplateAppliedEvent` dataclass
  - Add `CustomAgentSelectedEvent` dataclass

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 2: User Story 9a - Data Source Discovery (Priority: P1) 🎯 MVP

**Goal**: Users can see ALL data sources they have access to without manual configuration via auto-discovery.

**Independent Test**: Open data source selector and verify it shows all Vector Search indexes, Genie spaces, and serving endpoints the user has OBO access to.

### 2.1 Discovery Service Infrastructure

- [x] T010a [US9a] Create discovery cache in `src/deep_research/services/discovery_cache.py`
  - Implement `DiscoveryCache` class with TTL management (5-minute default)
  - Per-user cache keys (different OBO tokens see different sources)
  - Background refresh on near-expiry
  - Thread-safe async operations with `asyncio.Lock`

- [x] T010b [US9a] Create discovery service in `src/deep_research/services/discovery_service.py`
  - Implement `DiscoveryService` class with `WorkspaceClient` factory using `ModelServingUserCredentials()`
  - Implement `discover_all()` with parallel execution via `asyncio.gather()`

### 2.2 Vector Search Discovery

- [x] T010c [US9a] Implement `discover_vector_search_sources()` in discovery service
  - Use `w.vector_search_endpoints.list_endpoints()` → Iterator[EndpointInfo]
  - Use `w.vector_search_indexes.list_indexes(endpoint_name)` → Iterator[MiniVectorIndex]
  - Use `w.vector_search_indexes.get_index(index_name)` → VectorIndex for full metadata
  - Extract: index_name, endpoint_name, primary_key, index_type, status, embedding_columns
  - Determine filter columns from Delta table schema
  - Detect supported query types: ANN (always), HYBRID (if text columns), FULL_TEXT (if enabled)

### 2.3 Genie Discovery

- [x] T010d [US9a] Implement `discover_genie_spaces()` in discovery service
  - Use `w.genie.list_spaces()` → GenieListSpacesResponse
  - Use `w.genie.get_space(space_id)` for details
  - Extract: space_id, title, description, warehouse_id, owner

### 2.4 Knowledge Assistant Discovery

- [x] T010e [US9a] Implement `discover_serving_endpoints()` in discovery service
  - Use `w.serving_endpoints.list()` → Iterator[ServingEndpoint]
  - Filter by endpoint type/tags to identify Knowledge Assistants
  - Use heuristics: endpoint name patterns, model type, custom tags
  - Extract: endpoint_name, endpoint_type, state, tags, creator

### 2.5 Discovery Schemas

- [x] T010f [P] [US9a] Create discovery schemas in `src/deep_research/schemas/discovery.py`
  - `DiscoveryStatus` enum (ready, syncing, unavailable, error)
  - `DiscoveredSource` model with source_id, source_type, name, endpoint_name, description, status, capabilities, metadata
  - `VectorSearchMetadata` model with index details, filter_columns, supported_query_types
  - `GenieSpaceMetadata` model with space details
  - `ServingEndpointMetadata` model with endpoint details
  - `DiscoveryResponse` model with sources list, by_type grouping, cached status, errors
  - `DiscoveryError` model for partial failures

### 2.6 Discovery API Endpoints

- [x] T010g [US9a] Create discovery API router in `src/deep_research/api/v1/discovery.py`
  - `GET /api/v1/discovery/sources` - Returns all discoverable sources (cached)
  - `GET /api/v1/discovery/sources/{source_id}/metadata` - Detailed metadata for specific source
  - `POST /api/v1/discovery/refresh` - Force cache invalidation and re-discovery

- [x] T010h [US9a] Register discovery router in `src/deep_research/api/v1/__init__.py`

### 2.7 Frontend Discovery Components

- [x] T010i [P] [US9a] Create TypeScript types in `frontend/src/types/discovery.ts`
  - `DiscoveredSource`, `VectorSearchMetadata`, `GenieSpaceMetadata`, `ServingEndpointMetadata`
  - `DiscoveryResponse`, `DiscoveryError`

- [x] T010j [P] [US9a] Create discovery API client in `frontend/src/api/discovery.ts`
  - `discoverSources()`, `getSourceMetadata()`, `refreshDiscovery()`

- [x] T010k [US9a] Create `useDiscoveredSources` hook in `frontend/src/hooks/useDiscoveredSources.ts`
  - TanStack Query hooks with 5-minute stale time matching backend cache

- [x] T010l [US9a] Create `DataSourceSelector` component in `frontend/src/components/sources/DataSourceSelector.tsx`
  - Grouped dropdown by source type (Vector Search, Genie, Assistants)
  - Search/filter by source name
  - Loading state during initial discovery
  - Refresh button for manual cache invalidation

- [x] T010m [US9a] Create `SourceList` component in `frontend/src/components/sources/SourceList.tsx`
  - Type grouping with collapsible sections
  - Source count per type
  - Status indicators (available/syncing/unavailable)

- [x] T010n [US9a] Create `DiscoveredSourceCard` component in `frontend/src/components/sources/DiscoveredSourceCard.tsx`
  - Display name, type icon, description (truncated)
  - Status indicator badge
  - Expandable metadata panel showing query types, filter columns, endpoint details

### 2.8 Discovery Tests

- [x] T010o [P] [US9a] Write unit tests in `tests/unit/services/test_discovery_service.py`
  - Mock WorkspaceClient responses
  - Test parallel discovery
  - Test cache TTL and invalidation
  - Test error handling for partial failures

- [x] T010p [P] [US9a] Write unit tests in `tests/unit/services/test_discovery_cache.py`
  - Test cache key generation per user
  - Test TTL expiration
  - Test concurrent access

**Checkpoint**: US9a complete - Users can discover all available data sources automatically

---

## Phase 2b: User Story 9b - Query Configuration (Priority: P2)

**Goal**: Users can configure query settings (ANN/Hybrid/Full-Text, filters) per Vector Search data source.

**Independent Test**: Select a Vector Search index, configure Hybrid search with filters, verify queries use those settings.

### 2b.1 Query Config Schemas

- [x] T010q [P] [US9b] Create query config schemas in `src/deep_research/schemas/query_config.py`
  - `QueryType` enum (ANN, HYBRID, FULL_TEXT)
  - `FilterOperator` enum (=, !=, <, <=, >, >=, LIKE, NOT_LIKE, IN)
  - `FilterSyntax` enum (SQL, DICT)
  - `FilterExpression` model with column, operator, value, to_sql(), to_dict() methods
  - `VectorSearchQueryConfig` model with query_type, num_results, score_threshold, columns, enable_reranking, filters, filter_syntax

### 2b.2 Query Config Validation

- [x] T010r [US9b] Implement filter expression validation in `src/deep_research/schemas/query_config.py`
  - Validate operator compatibility with column data type
  - Validate IN filter ID limit (1,024 per clause)
  - Validate query type against index capabilities
  - Validate reranking columns exist

### 2b.3 Query Config API

- [x] T010s [US9b] Add query config endpoints to `src/deep_research/api/v1/data_sources.py`
  - `PUT /api/v1/data-sources/{id}/query-config` - Save configuration
  - `GET /api/v1/data-sources/{id}/query-config` - Retrieve configuration

- [x] T010t [US9b] Persist query config in `UserDataSource.config` JSONB column
  - Store under `query_config` key within existing config structure

### 2b.4 Tool Integration

- [x] T010u [US9b] Extend `UserVectorSearchTool` to use query config
  - Read `query_config` from context or user defaults
  - Apply query_type to SDK `query_index()` call
  - Build filter expressions using configured syntax
  - Pass columns_to_rerank when reranking enabled

### 2b.5 Frontend Query Config Components

- [x] T010v [P] [US9b] Create `QueryConfigPanel` component in `frontend/src/components/discovery/QueryConfigPanel.tsx`
  - Query type radio buttons (ANN/Hybrid/Full-Text) with descriptions
  - Disable unsupported query types based on index capabilities
  - Num results slider (1-100, default 10)
  - Score threshold input (optional, 0.0-1.0)
  - Reranking toggle with column selector

- [x] T010w [P] [US9b] Create `FilterBuilder` component in `frontend/src/components/discovery/FilterBuilder.tsx`
  - Add filter button
  - Column dropdown (populated from index metadata)
  - Operator dropdown (filtered by column data type)
  - Value input (type-aware: text, number, date, list for IN)
  - Remove filter button
  - Real-time validation with error messages

- [x] T010x [US9b] Create `useQueryConfig` hook in `frontend/src/hooks/useQueryConfig.ts`
  - Per-source config state management
  - Save/load from API
  - Validation before save

- [x] T010y [US9b] Integrate QueryConfigPanel into DataSourceSelector
  - Show config panel when source is expanded
  - Save as default toggle

### 2b.6 Query Config Tests

- [x] T010z [P] [US9b] Write unit tests in `tests/unit/schemas/test_query_config.py`
  - Test FilterExpression.to_sql() and to_dict()
  - Test validation rules for operators and limits
  - Test query config validation against index capabilities

**Checkpoint**: US9b complete - Users can configure query settings per data source

---

## Phase 3: User Story 1 - Researcher Queries Enterprise Knowledge Base (Priority: P1)

**Goal**: Enable researchers to query Databricks Vector Search indexes alongside web sources with proper attribution.

**Independent Test**: Configure a Vector Search endpoint and submit a research query requiring enterprise data. Verify VS results appear in Research Panel.

**Depends on**: US9a (Discovery) - tools should use discovered sources

### 3.1 User Data Source Model

- [x] T011 [US1,US9] Create `UserDataSource` SQLAlchemy model in `src/deep_research/models/data_source.py`
  - id, owner_id, type, name, description, endpoint_identifier
  - config (JSONB for type-specific settings with AUTO-DETECTED schema)
  - visibility (private/workspace), validation_status, last_validated_at
  - Add indexes on owner_id and visibility

- [x] T012 [US1,US9] Create migration for `user_data_sources` table in `src/deep_research/db/versions/`

### 3.2 Vector Search Tool Implementation

- [x] T013 [US1] Enhance Vector Search tool in `src/deep_research/agent/tools/user_vector_search.py`
  - Accept `UserDataSource` or `DiscoveredSource` as constructor parameter
  - Implement OBO authentication via `OBODatabricksClient`
  - Support query types from US9b (ANN, HYBRID, FULL_TEXT)
  - Support reranking via `DatabricksReranker` (requires databricks-vectorsearch >= 0.57)
  - Support metadata filtering from `arguments["filters"]` or query config
  - Implement deduplication by content hash
  - Format results as `SourceInfo` with type="vector_search"

- [x] T014 [US1] Create dynamic tool definition generation in `UserVectorSearchTool.definition` property
  - Include filterable columns in parameters schema (from discovery metadata)
  - Generate tool name from index name: `search_{index_name.replace('.', '_')}`

### 3.3 Data Source Service

- [x] T015 [US1,US9] Create `DataSourceService` in `src/deep_research/services/data_source_service.py`
  - Extend `BaseRepository[UserDataSource]`
  - Implement `create_vector_search_source()` with schema auto-detection from index metadata
  - Implement `get_accessible_sources()` for own + valid workspace sources
  - Implement `_extract_column_schema()` from VS index info
  - Add OBO access validation and caching

### 3.4 Tool Factory

- [x] T016 [US1,US2,US3] Create tool factory in `src/deep_research/agent/tools/factory.py`
  - Implement `create_tools_from_discovered_sources()` using discovery results
  - Implement `create_tools_from_user_sources()` async function
  - Dynamically create tools from `DiscoveredSource` or `UserDataSource` config
  - Apply query configuration from US9b when available
  - Support VectorSearchTool, GenieTool, KnowledgeAssistantTool

### 3.5 API Endpoints

- [x] T017 [US1,US9] Create data source API endpoints in `src/deep_research/api/v1/data_sources.py`
  - POST `/data-sources` - Create new data source (with OBO validation)
  - GET `/data-sources` - List accessible sources (combines discovered + user-added)
  - GET `/data-sources/{id}` - Get source details
  - DELETE `/data-sources/{id}` - Delete source
  - POST `/data-sources/{id}/validate` - Re-validate OBO access

- [x] T018 [P] [US1,US9] Create data source request/response schemas in `src/deep_research/schemas/data_source.py`
  - `CreateVectorSearchSourceRequest`, `CreateGenieSourceRequest`, etc.
  - `DataSourceResponse`, `DataSourceListResponse`

### 3.6 Frontend Components

- [x] T019 [P] [US1] Create `SourceCard.tsx` in `frontend/src/components/sources/`
  - Display source name, type, description, capabilities
  - Show validation status indicator
  - Add enable/disable toggle

- [x] T020 [P] [US1] Create `SourceConfigModal.tsx` in `frontend/src/components/sources/`
  - Form for adding Vector Search source (endpoint, index name)
  - Form for adding Genie source (space ID, description)
  - OBO access validation feedback

- [x] T021 [US1] Create API client in `frontend/src/api/dataSources.ts`
  - CRUD operations for data sources
  - Type definitions in `frontend/src/types/dataSources.ts`

- [x] T022 [US1] Create `useDataSources` hook in `frontend/src/hooks/useDataSources.ts`
  - TanStack Query hooks for data source operations

**Checkpoint**: User Story 1 complete - VS queries work with OBO authentication and discovered sources

---

## Phase 3: User Story 2 - Researcher Queries Relational Data via Genie (Priority: P1)

**Goal**: Enable natural language queries against enterprise databases via Genie, returning tabular results with narrative summaries.

**Independent Test**: Configure a Genie space and submit a query requiring data analysis. Verify SQL and results appear.

### 3.1 Genie Tool Implementation

- [x] T022 [US2] Create Genie tool in `src/deep_research/agent/tools/genie.py`
  - Implement `GenieTool` class with OBO authentication
  - Support conversation context for follow-up queries (store `_conversation_id`)
  - Implement `_start_conversation()` and `_continue_conversation()` methods
  - Format tabular results with `_format_genie_result()`
  - Truncate large result sets (configurable, default 100 rows)
  - Return generated SQL for transparency
  - Generate narrative summary via LLM if available

- [x] T023 [US2] Create dynamic tool definition in `GenieTool.definition` property
  - Include `is_follow_up` boolean parameter
  - Generate tool name from space ID: `query_genie_{space_id}`

### 3.2 Data Source Service Extension

- [x] T024 [US2] Add `create_genie_source()` to `DataSourceService`
  - Validate Genie space access via OBO
  - Store space_id, description, example_questions in config

### 3.3 Frontend Extensions

- [x] T025 [P] [US2] Add Genie source form to `SourceConfigModal.tsx`
  - Space ID input
  - Description and example questions
  - Connection test button

- [x] T026 [P] [US2] Create Genie result display component for Research Panel
  - Show tabular results with truncation
  - Display generated SQL in collapsible section
  - Show narrative summary

**Checkpoint**: User Story 2 complete - Genie queries work with follow-up context

---

## Phase 4: User Story 10 - System Discovers Available Data Before Planning (Priority: P1)

**Goal**: Before planning, explore ALL data sources to build a DataLandscape that informs intelligent source routing.

**Independent Test**: Configure multiple data sources, submit query. Verify discovery queries all sources in parallel (<5s).

### 4.1 Data Landscape Models

- [x] T027 [US10,US11] Create data landscape schemas in `src/deep_research/schemas/data_landscape.py`
  - `SourceDiscoveryResult` dataclass with relevance_score, sample_results, available_filters, suggested_queries
  - `DataLandscape` dataclass aggregating all discovery results
  - Implement `to_planner_summary()` method for planner consumption

### 4.2 Source Scope Models

- [x] T028 [US10,US11,US12] Create source scope schemas in `src/deep_research/schemas/source_scope.py`
  - `SourceScope` enum (ENTERPRISE_ONLY, WEB_ONLY, ALL)
  - `SourceScopeConfig` Pydantic model with `filter_sources()` method

### 4.3 Background Discovery Implementation

- [x] T029 [US10] Extend background investigator in `src/deep_research/agent/nodes/background.py`
  - Implement `run_background_discovery()` async function
  - Generate exploratory queries dynamically from user prompt
  - Query ALL enabled sources in parallel with `asyncio.gather()`
  - Handle failures gracefully (create low-relevance result for failed sources)
  - Complete within 5 seconds timeout (FR-088)

- [x] T030 [US10] Implement `generate_exploratory_queries()` function
  - Generate 3 query variants covering different aspects
  - Use LLM to decompose complex queries

- [x] T031 [P] [US10] Implement `explore_vector_search()` function
  - Lightweight query with num_results=3
  - Return relevance score from top result
  - Extract available filter columns

- [x] T032 [P] [US10] Implement `explore_genie()` function (FR-089)
  - Try metadata query first ("What data do you have related to...")
  - Fall back to sample query if ambiguous

- [x] T033 [P] [US10] Implement `explore_web_source()` function
  - Use existing Brave search with limited results
  - Return relevance indicators

- [x] T034 [US10] Implement `build_data_landscape()` function
  - Aggregate discovery results
  - Rank sources by relevance
  - Build capabilities map per source

### 4.4 State Extensions

- [x] T035 [US10,US11] Extend `ResearchState` in `src/deep_research/agent/state.py`
  - Add `source_scope_config: SourceScopeConfig | None`
  - Add `data_landscape: DataLandscape | None`
  - Add `source_query_counts: dict[str, int]`
  - Add `source_results: dict[str, list[SourceInfo]]`
  - Implement `get_source_budget()` and `record_source_queries()` methods

**Checkpoint**: User Story 10 complete - Discovery builds DataLandscape before planning

---

## Phase 5: User Story 11 - Planner Generates Source-Aware Steps (Priority: P1)

**Goal**: Planner receives DataLandscape and outputs per-step source hints for intelligent routing.

**Independent Test**: Submit query requiring both enterprise and web data. Verify plan steps include appropriate source hints.

### 5.1 Plan Schema Extensions

- [x] T036 [US11] Create step source hint schemas in `src/deep_research/schemas/plan.py`
  - `StepSourceHint` model with source_name, source_type, priority (1-3), query_hint, filters
  - `PlanStepWithSources` extending PlanStep with source_hints, exclude_sources, require_all_sources

### 5.2 Source-Aware Planner

- [x] T037 [US11] Extend planner in `src/deep_research/agent/nodes/planner.py`
  - Create `PLANNER_SYSTEM_PROMPT_WITH_SOURCES` template (in prompts/source_aware_planner.py)
  - Implement `run_source_aware_planner()` that receives DataLandscape
  - Generate steps with source_hints based on capabilities
  - Store source hints in state.phase_results for researcher access

### 5.3 Per-Step Tool Filtering

- [x] T038 [US11] Implement source-aware tool filtering in `src/deep_research/agent/nodes/source_routing.py`
  - Create `execute_step_with_source_routing()` function
  - Implement `filter_tools_for_step()` based on source hints
  - Inject query hints into tool definitions
  - Respect source budgets per type

### 5.4 Configuration

- [x] T039 [US11] Add source routing configuration to `config/app.yaml`
  - `background.discovery` settings (enabled, max_sources, timeout)
  - `source_routing` settings (enabled, default_scope, source_budgets, type_priorities)
  - `source_routing.discovery` query templates

**Checkpoint**: User Story 11 complete - Planner outputs source-routed steps

---

## Phase 6: User Story 12 - User Controls Source Scope (Priority: P1)

**Goal**: Users can control which source categories are available and optionally review/edit plans before execution.

**Independent Test**: Select "Enterprise Only" scope and verify no web searches during research.

### 6.1 Plan Review Implementation

- [x] T040 [US12] Implement plan review in orchestrator `src/deep_research/agent/orchestrator.py`
  - Create `execute_with_plan_review()` async generator
  - Yield `PlanReviewEvent` when review enabled
  - Wait for user response via SSE
  - Implement `apply_user_edits()` function
  - Support timeout with auto-proceed

### 6.2 API Extensions

- [x] T041 [US12] Extend research request schema in `src/deep_research/schemas/research_request.py`
  - Add `source_scope: SourceScope`
  - Add `enabled_sources: list[str] | None`
  - Add `disabled_sources: list[str]`
  - Add `enable_plan_review: bool`
  - Add `require_plan_approval: bool`

### 6.3 Frontend Components

- [x] T042 [P] [US12] Create `SourceScopeSelector.tsx` in `frontend/src/components/research/`
  - Toggle group: Enterprise Only | Web Only | All
  - Expandable per-source toggle section
  - Show source descriptions and relevance hints

- [x] T043 [P] [US12] Create `PlanReviewModal.tsx` in `frontend/src/components/research/`
  - Display plan steps with source hints
  - Drag-and-drop reordering
  - Per-step source selector
  - Add/remove step buttons
  - Countdown timer for timeout
  - Approve/Edit/Cancel buttons

- [x] T044 [P] [US12] Create `PlanEditor.tsx` in `frontend/src/components/research/`
  - Editable step list with drag-and-drop
  - Per-step source selection (multi-select)
  - Source priority adjustment (1/2/3)
  - Query hint input per source

- [x] T045 [US12] Create `usePlanReview` hook in `frontend/src/hooks/usePlanReview.ts`
  - SSE hook for plan review events
  - Handle edit submission and approval

**Checkpoint**: User Story 12 complete - Users control source scope and can review plans

---

## Phase 7: User Story 3 - Researcher Consults Domain Expert Assistants (Priority: P2)

**Goal**: Query Knowledge Assistants for authoritative answers with research context.

**Independent Test**: Configure Knowledge Assistant endpoint and submit domain-specific question.

### 7.1 Knowledge Assistant Tool

- [x] T046 [US3] Enhance Knowledge Assistant tool in `src/deep_research/agent/tools/knowledge_assistant.py`
  - Implement OBO authentication
  - Support context passing (`include_context` parameter)
  - Implement `_add_research_context()` helper
  - Return confidence level and internal references in source

### 7.2 Data Source Service Extension

- [x] T047 [US3] Add `create_assistant_source()` to `DataSourceService` (already done in T024)
  - Validate endpoint access via OBO
  - Store endpoint_name, description, pass_context flag

### 7.3 Frontend Extensions

- [x] T048 [P] [US3] Add Knowledge Assistant source form to `SourceConfigModal.tsx`
- [x] T049 [P] [US3] Create Expert source display component for Research Panel

**Checkpoint**: User Story 3 complete - Knowledge Assistant queries work

---

## Phase 8: User Story 4 - User Defines Manual Research Steps (Priority: P2)

**Goal**: Power users can define precise research workflows with specific sources per step.

**Independent Test**: Create manual steps with specific sources, execute in Manual mode.

### 8.1 Manual Step Models

- [x] T050 [US4] Create manual step schemas in `src/deep_research/schemas/manual_step.py`
  - `SourceConstraint` model (allowed_types, allowed_sources, required_sources, excluded_sources)
  - `StepSourceAttachment` model (source_name, type, custom_prompt, filters)
  - `ManualStepDefinition` model (id, title, objective, sources, constraints, order)

### 8.2 Workflow Mode Support

- [x] T051 [US4] Extend `ResearchState` with workflow mode support
  - Add `WorkflowMode` enum (PLANNER, MANUAL, HYBRID)
  - Add `workflow_mode`, `manual_steps`, `step_source_constraints` fields
  - Add `get_source_constraint()`, `get_manual_step()` methods

### 8.3 Orchestrator Extensions

- [x] T052 [US4] Extend orchestrator in `src/deep_research/agent/orchestrator.py`
  - Support `WorkflowMode.MANUAL` (bypass planner, use manual steps)
  - Support `WorkflowMode.HYBRID` (manual steps first, then planner)
  - Implement `_convert_manual_steps_to_plan()` helper

### 8.4 Researcher Source Constraints

- [x] T053 [US4] Implement source constraints in `src/deep_research/agent/tools/source_routing.py`
  - Create `filter_tools_by_constraint()` function
  - Validate required sources were consulted
  - Implement `prompt_for_required_sources()` for missing requirements

### 8.5 Frontend Components

- [x] T054 [P] [US4] Create `SourceBrowser.tsx` in `frontend/src/components/sources/`
  - Group sources by category (Web, VS, Genie, Assistants, Files)
  - Show descriptions, capabilities, example queries

- [x] T055 [P] [US4] Create `ManualStepEditor.tsx` in `frontend/src/components/steps/`
  - Step title and objective inputs
  - Source selection with custom prompts
  - Optional filter configuration for VS sources

- [x] T056 [P] [US4] Create `StepSourcePicker.tsx` in `frontend/src/components/steps/`
  - Multi-select source picker
  - Custom prompt input per source

- [x] T057 [P] [US4] Create `StepReorderList.tsx` in `frontend/src/components/steps/`
  - Drag-and-drop step reordering
  - Add/edit/remove step buttons

- [x] T058 [US4] Create workflow mode selector in research input UI
  - Planner | Manual | Hybrid toggle

**Checkpoint**: User Story 4 complete - Manual step definition works

---

## Phase 9: User Story 9 - User Adds Custom Data Source Connection (Priority: P2)

**Goal**: Users can add their own Databricks data sources via UI.

**Independent Test**: User adds a Vector Search index they have access to and uses it in research.

Note: Most infrastructure is already built in US1. This phase adds the user-facing UI flow.

### 9.1 Frontend Data Source Management

- [x] T059 [US9] Create data source management page `frontend/src/pages/DataSourcesPage.tsx`
  - List user's data sources with status
  - Add new source button → SourceConfigModal
  - Edit/delete source actions
  - Visibility toggle (private/workspace)

- [x] T060 [US9] Add data sources to navigation/settings menu

**Checkpoint**: User Story 9 complete - Self-service data source configuration

---

## Phase 10: User Story 8 - Plugin Registers Custom Data Sources (Priority: P2)

**Goal**: Plugins can register data sources, templates, and agents that appear in the UI.

**Independent Test**: Create plugin that registers a Vector Search endpoint and verify it appears in source browser.

### 10.1 Plugin Discovery Integration

- [x] T061 [US8] Integrate plugin protocols with source browser
  - Query `DataSourceProvider.get_data_sources()` during source list
  - Apply `get_source_constraints()` when active
  - Show plugin attribution in UI

- [x] T062 [US8] Integrate plugin templates with template library
  - Query `TemplateProvider.get_templates()` during template list
  - Show plugin attribution

- [x] T063 [US8] Integrate plugin agents with agent selector
  - Query `CustomAgentProvider.get_custom_agents()` during agent list
  - Show plugin attribution

### 10.2 Lifecycle Event Emission

- [x] T064 [US8] Emit lifecycle events in tool execution
  - Emit `DataSourceQueryEvent` after each source query
  - Emit `TemplateAppliedEvent` when template used
  - Emit `CustomAgentSelectedEvent` when agent selected

**Checkpoint**: User Story 8 complete - Plugin extensibility works

---

## Phase 11: User Story 5 - User Creates and Uses Prompt Templates (Priority: P3)

**Goal**: Users can create reusable prompt templates with variables.

**Independent Test**: Create synthesis template, use in research, verify output follows template.

### 11.1 Template Models

- [x] T065 [US5] Create `PromptTemplate` SQLAlchemy model in `src/deep_research/models/prompt_template.py`
  - id, owner_id, name, type (system/step/synthesis/query)
  - content, variables (JSONB with metadata), tags
  - visibility (private/workspace), is_default

- [x] T066 [US5] Create migration for `prompt_templates` table

### 11.2 Template Service

- [x] T067 [US5] Create `TemplateService` in `src/deep_research/services/template_service.py`
  - CRUD operations extending BaseRepository
  - Variable validation and rendering
  - Default template management

### 11.3 Template API

- [x] T068 [US5] Create template API endpoints in `src/deep_research/api/v1/templates.py`
  - CRUD endpoints for templates
  - Template rendering endpoint

- [x] T069 [P] [US5] Create template schemas in `src/deep_research/schemas/template.py`

### 11.4 Frontend Components

- [x] T070 [P] [US5] Create `TemplateEditor.tsx` in `frontend/src/components/templates/`
  - Template content editor with `{{variable}}` syntax highlighting
  - Variable metadata editor (name, type, required, default)
  - Preview with sample values

- [x] T071 [P] [US5] Create `TemplateLibrary.tsx` in `frontend/src/components/templates/`
  - Filter by type and search by name/tags
  - Show system, plugin, and user templates

- [x] T072 [P] [US5] Create `VariableInput.tsx` in `frontend/src/components/templates/`
  - Dynamic form for template variables
  - Validation for required variables

- [x] T073 [US5] Create `useTemplates` hook in `frontend/src/hooks/useTemplates.ts`

**Checkpoint**: User Story 5 complete - Template system works

---

## Phase 12: User Story 6 - User Creates and Uses Custom Research Agent (Priority: P3)

**Goal**: Users can create GPT-like specialized research assistants combining prompts, sources, and workflows.

**Independent Test**: Create custom agent with preset steps and specific sources, use for research.

### 12.1 Custom Agent Models

- [x] T074 [US6] Create `CustomAgent` SQLAlchemy model in `src/deep_research/models/custom_agent.py`
  - id, owner_id, name, description, avatar_url
  - system_prompt_template_id, synthesis_template_id (FK to templates)
  - source_scope, enabled_sources, disabled_sources (JSONB)
  - use_planner, default_depth, default_mode, enable_clarification
  - output_format, output_schema (JSONB)
  - visibility (private/workspace/system)

- [x] T075 [US6] Create `AgentPresetStep` SQLAlchemy model
  - id, agent_id (FK), title, description, order, is_required
  - source_hints (JSONB), source_scope

- [x] T076 [US6] Create migration for `custom_agents` and `agent_preset_steps` tables

### 12.2 Custom Agent Service

- [x] T077 [US6] Create `CustomAgentService` in `src/deep_research/services/custom_agent_service.py`
  - CRUD operations with cascade delete for preset steps
  - Agent resolution for research requests
  - Default agent management

### 12.3 Custom Agent API

- [x] T078 [US6] Create custom agent API endpoints in `src/deep_research/api/v1/custom_agents.py`
  - CRUD endpoints for agents
  - Preset step management endpoints

- [x] T079 [P] [US6] Create custom agent schemas in `src/deep_research/schemas/custom_agent.py`

### 12.4 Orchestrator Integration

- [x] T080 [US6] Integrate custom agents with orchestrator
  - Apply agent's source_scope to research request
  - Use agent's templates for prompts and synthesis
  - Execute preset steps when use_planner is false
  - Support per-query overrides

### 12.5 Frontend Components

- [x] T081 [P] [US6] Create `AgentBuilder.tsx` in `frontend/src/components/agents/`
  - Identity section (name, description, avatar)
  - Prompts section (template selection or inline)
  - Sources section (scope, enabled/disabled sources)
  - Workflow section (use planner, preset steps, default depth)
  - Output section (format, schema)

- [x] T082 [P] [US6] Create `AgentSelector.tsx` in `frontend/src/components/agents/`
  - Browse available agents (own, workspace, system)
  - Show agent descriptions and capabilities

- [x] T083 [P] [US6] Create `AgentPresetSteps.tsx` in `frontend/src/components/agents/`
  - Manage preset steps with drag-and-drop
  - Per-step source configuration

- [x] T084 [US6] Create `useCustomAgents` hook in `frontend/src/hooks/useCustomAgents.ts`

**Checkpoint**: User Story 6 complete - Custom agents work

---

## Phase 13: User Story 7 - User Uploads Files for Research Context (Priority: P3)

**Goal**: Users can upload documents that become searchable sources for their research.

**Independent Test**: Upload document, start research, verify content appears in sources with citations.

### 13.1 File Models

- [x] T085 [US7] Create `UploadedFile` SQLAlchemy model in `src/deep_research/models/uploaded_file.py`
  - id, owner_id, session_id, filename, file_type, file_size
  - storage_path, processing_status, chunk_count
  - expires_at, created_at

- [x] T086 [US7] Create `FileChunk` SQLAlchemy model
  - id, file_id (FK), chunk_index, content, metadata (JSONB)

- [x] T087 [US7] Create migration for `uploaded_files` and `file_chunks` tables

### 13.2 File Upload Service

- [x] T088 [US7] Create `FileUploadService` in `src/deep_research/services/file_upload_service.py`
  - File validation (type, size, count limits)
  - Chunking with paragraph/page boundaries (naive chunking)
  - Storage to Databricks Volumes
  - Expiration management
  - No OCR support (FR-059)

### 13.3 File Search Tool

- [x] T089 [US7] Create file search tool in `src/deep_research/agent/tools/file_search.py`
  - Keyword search across file chunks
  - Return citations with filename and chunk location

### 13.4 File API

- [x] T090 [US7] Create file upload API endpoints in `src/deep_research/api/v1/files.py`
  - POST `/files/upload` - Upload file(s)
  - GET `/files` - List session files
  - GET `/files/{id}/preview` - Preview file content
  - DELETE `/files/{id}` - Remove file

- [x] T091 [P] [US7] Create file upload schemas in `src/deep_research/schemas/file_upload.py`

### 13.5 Frontend Components

- [x] T092 [P] [US7] Create `FileUploadZone.tsx` in `frontend/src/components/files/`
  - Drag-and-drop upload area
  - File type and size validation
  - Upload progress indicator

- [x] T093 [P] [US7] Create `UploadedFileList.tsx` in `frontend/src/components/files/`
  - Show uploaded files with status
  - Preview and remove actions

- [x] T094 [US7] Create `useFileUpload` hook in `frontend/src/hooks/useFileUpload.ts`

**Checkpoint**: User Story 7 complete - File upload and search works

---

## Phase 14: Parallel Tool Execution (F12) - Performance Enhancement

**Purpose**: Execute different source types concurrently for 20-40% latency reduction.

**Note**: This implements functional requirements FR-107 through FR-125 from the spec.

### 14.1 State Async Safety ✅ COMPLETED

- [x] T095 [P] Add asyncio.Lock to ResearchState in `src/deep_research/agent/state.py`
- [x] T096 [P] Add asyncio.Lock to ReactResearchState in `src/deep_research/agent/nodes/react_researcher.py`

### 14.2 Parallel Execution Implementation ✅ COMPLETED

- [x] T097 Implement parallel tool execution in `src/deep_research/agent/nodes/react_researcher.py`
- [x] T098 Add parallel execution configuration to `src/deep_research/core/app_config.py`
- [x] T099 Add parallel execution to `config/app.yaml`

### 14.3 Testing ✅ COMPLETED

- [x] T100 Add unit tests for parallel tool execution in `tests/unit/agent/test_parallel_tools.py`

**Checkpoint**: F12 complete - Parallel tool execution operational

---

## Phase 15: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements affecting multiple user stories.

### 15.1 Documentation

- [x] T101 [P] Update API documentation with new endpoints
- [x] T102 [P] Add user guide for data source configuration
- [x] T103 [P] Add developer guide for plugin development

### 15.2 Error Handling & Validation

- [x] T104 Comprehensive error messages for OBO permission failures
- [x] T105 Validation for conflicting source constraints
- [x] T106 Graceful degradation when sources unavailable

### 15.3 Performance & Observability

- [x] T107 Add MLflow tracing for all enterprise source queries
- [x] T108 Add metrics for source query latencies
- [x] T109 Add dashboards for data source usage

### 15.4 Security Review

- [x] T110 Review OBO token handling for security
- [x] T111 Ensure proper permission checks on all endpoints
- [x] T112 Validate file upload security (path traversal, etc.)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Foundation → BLOCKS ALL user stories
  ↓
Phase 2: US9a (Discovery) → MVP - Auto-discover all data sources
  ↓
Phase 2b: US9b (Query Config) → Configure ANN/Hybrid/Filters per source
  ↓
Phases 3-6 (P1 stories): US1, US2, US10, US11, US12
  ↓ (can run in parallel after US9a)
Phases 7-10 (P2 stories): US3, US4, US9, US8
  ↓ (can run in parallel)
Phases 11-13 (P3 stories): US5, US6, US7
  ↓
Phase 14: Parallel Execution (already complete)
  ↓
Phase 15: Polish
```

### Critical Path (Updated)

1. **Foundation** (T001-T009) - 1 sprint
2. **US9a: Data Source Discovery** (T010a-T010p) - 1 sprint - **MVP increment** 🎯
3. **US9b: Query Configuration** (T010q-T010z) - 0.5 sprint (can overlap with US1)
4. **US1: Vector Search** (T011-T022) - 1 sprint
5. **US2: Genie** (T023-T027) - 0.5 sprint
6. **US10: Background Discovery** (T028-T036) - 1 sprint
7. **US11: Source-Aware Planning** (T037-T040) - 0.5 sprint
8. **US12: User Control** (T041-T046) - 0.5 sprint

### Parallel Opportunities

After Foundation:
- **US9a (Discovery) is the new MVP** - must complete before other stories
- US9b can start in parallel with US1 after US9a schemas are done
- US1, US2, US3 can develop in parallel after US9a (different tools)
- US10, US11, US12 should be sequential (background discovery → planning → control)
- US4, US5, US6, US7 can develop in parallel (independent features)
- US8, US9 can develop in parallel with above

### Within Each User Story

1. Models/schemas before services
2. Services before API endpoints
3. API before frontend components
4. Backend complete before frontend integration

---

## Implementation Strategy

### MVP First (P1 Stories Only)

1. Complete Foundation (T001-T009)
2. Complete **US9a: Data Source Discovery** (T010a-T010p) → **Deploy/Demo - Users can see all available sources** 🎯
3. Complete US9b: Query Configuration (T010q-T010z) → **Deploy/Demo - Users can configure query settings**
4. Complete US1: Vector Search (T011-T022) → **Deploy/Demo - VS queries work**
5. Complete US2: Genie (T023-T027) → **Deploy/Demo - Genie queries work**
6. Complete US10: Background Discovery (T028-T036) → **Deploy/Demo - Intelligent data landscape**
7. Complete US11: Planning (T037-T040) → **Deploy/Demo - Source-aware plans**
8. Complete US12: User Control (T041-T046) → **Deploy/Demo - Full user control**

Each P1 story adds incremental value and can be demonstrated independently.

### Full Feature

After P1 MVP:
- Add P2 stories in priority order (US3, US4, US9, US8)
- Add P3 stories for power users (US5, US6, US7)
- Polish phase for production readiness

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- F12 (Parallel Execution) is already implemented - marked as complete

## Key Changes in This Version

**US9a (Data Source Discovery)** and **US9b (Query Configuration)** added as new P1/P2 priorities based on spec.md updates:

1. **US9a - Data Source Discovery (P1)**: Auto-discover all Vector Search indexes, Genie spaces, and Knowledge Assistants via Databricks SDK. This is now the **MVP foundation** - users see all available sources without manual configuration.

2. **US9b - Query Configuration (P2)**: Configure query type (ANN/Hybrid/Full-Text) and filter expressions per Vector Search index. Enables power users to optimize query strategies.

**New Technical Components**:
- `DiscoveryService` with parallel discovery using `asyncio.gather()`
- `DiscoveryCache` with 5-minute TTL per user
- Discovery API endpoints (`/api/v1/discovery/sources`)
- Frontend `DataSourceSelector` with grouped dropdown
- `VectorSearchQueryConfig` with filter expression builder

**SDK Integration** (from research.md):
- `w.vector_search_endpoints.list_endpoints()` → discover endpoints
- `w.vector_search_indexes.list_indexes(endpoint_name)` → discover indexes
- `w.genie.list_spaces()` → discover Genie spaces
- `w.serving_endpoints.list()` → discover Knowledge Assistants
- All use `ModelServingUserCredentials()` for OBO authentication
