# Feature Specification: Enterprise Data Sources & Custom Research Workflows

**Feature Branch**: `007-enterprise-data-sources`
**Created**: 2026-02-04
**Status**: Draft
**Input**: User description: "Enterprise data source integrations (Databricks Vector Search, Genie, Knowledge Assistants), per-step source constraints, manual step definition via UI, prompt templates, custom agents/GPTs, and file upload for synthesis"

---

## Overview

This feature extends the Deep Research Agent with enterprise data source integrations and user-customizable research workflows. It enables researchers to query proprietary data through Databricks Vector Search, Genie (AI/BI), and Knowledge Assistants, while giving users precise control over research methodology through manual steps, prompt templates, and custom agent definitions.

### Feature Components

| ID | Component | Description |
|----|-----------|-------------|
| F1 | Vector Search Integration | Query Databricks Vector Search indexes with filtering |
| F2 | Genie Agent Integration | Query relational data via natural language |
| F3 | Knowledge Assistant Integration | Consult domain-expert AI assistants |
| F4 | Per-Step Source Constraints | Control which sources each step can use |
| F5 | Manual Step Definition UI | User-defined research steps with source selection |
| F6 | Prompt Template System | Reusable customizable prompt templates |
| F7 | Custom Agent Definition | GPT-like custom research agents |
| F8 | File Upload for Synthesis | Upload documents as research sources |
| F9 | Multi-Source Background Discovery | Explore ALL data sources before planning to build data landscape |
| F10 | Source-Aware Planning | Planner outputs per-step source hints based on data landscape |
| F11 | Source Scope & Plan Review | User controls source scope and can review/edit plan before execution |
| F12 | Parallel Tool Execution | Execute same-type tools concurrently for 30-50% latency reduction |
| F13 | Data Source Discovery API | Auto-discover available Vector Search, Genie, and Assistant sources via Databricks SDK |
| F14 | Query Type & Filter Configuration | Configure ANN/Hybrid/Full-Text query type and filter expressions per Vector Search index |

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Researcher Queries Enterprise Knowledge Base (Priority: P1)

A researcher needs to include proprietary company documentation in their research. They submit a query, and the system automatically searches relevant Vector Search indexes alongside web sources, returning enterprise content with proper attribution.

**Why this priority**: Core value proposition - enables research over proprietary data that web search cannot access. Without this, the system cannot differentiate from standard web-only research tools.

**Independent Test**: Can be fully tested by configuring a Vector Search endpoint and submitting a research query that requires enterprise data. Delivers immediate value by surfacing internal documentation in research results.

**Acceptance Scenarios**:

1. **Given** a configured Vector Search endpoint with product documentation, **When** a researcher asks "What are our product's security features?", **Then** the system queries the Vector Search index and includes relevant chunks as sources in the Research Panel
2. **Given** multiple Vector Search indexes (product-docs, support-tickets), **When** the researcher's query spans both domains, **Then** the system queries both indexes and aggregates results
3. **Given** a Vector Search index with metadata filters available, **When** the researcher needs recent documentation, **Then** the system applies appropriate filters (e.g., date range) to narrow results
4. **Given** Vector Search results are retrieved, **When** displayed in the Research Panel, **Then** sources show index name, document origin, and relevance score

---

### User Story 2 - Researcher Queries Relational Data via Genie (Priority: P1)

A researcher needs quantitative data (metrics, trends, aggregations) to support their research. They ask questions in natural language, and Genie translates them to SQL queries against enterprise databases, returning tabular results with narrative summaries.

**Why this priority**: Enables data-driven research with actual business metrics - critical for enterprise use cases where decisions require quantitative evidence.

**Independent Test**: Can be tested by configuring a Genie space and submitting a query requiring data analysis. Delivers value by providing data-backed insights in research output.

**Acceptance Scenarios**:

1. **Given** a configured Genie space for revenue analytics, **When** a researcher asks "What was Q3 revenue by region?", **Then** Genie generates SQL, executes it, and returns a tabular result with narrative summary
2. **Given** an initial Genie query was executed, **When** the researcher asks a follow-up question "Break that down by product line", **Then** Genie maintains conversation context and refines the query
3. **Given** a Genie query returns large results, **When** displayed in the Research Panel, **Then** results are truncated with row count indication, and the generated SQL is visible for transparency
4. **Given** multiple Genie spaces are configured, **When** the researcher needs data from a specific domain, **Then** the system selects the appropriate space based on the data need

---

### User Story 3 - Researcher Consults Domain Expert Assistants (Priority: P2)

A researcher needs authoritative answers on specialized topics (security, compliance, ML best practices). They query Knowledge Assistants that have curated access to specific knowledge bases, receiving expert-level responses integrated into their research.

**Why this priority**: Provides access to validated, domain-expert knowledge that supplements general research. Important but builds on core data source capability.

**Independent Test**: Can be tested by configuring a Knowledge Assistant endpoint and submitting a domain-specific question. Delivers value by providing authoritative expert input.

**Acceptance Scenarios**:

1. **Given** a configured Security Expert assistant, **When** a researcher asks about compliance requirements, **Then** the assistant provides an authoritative answer based on its curated knowledge base
2. **Given** a question spans multiple domains, **When** the Supervisor Agent is queried, **Then** it coordinates across relevant assistants to provide a comprehensive answer
3. **Given** research context exists, **When** querying an assistant with context enabled, **Then** the assistant receives relevant context and provides more targeted answers
4. **Given** an assistant response is received, **When** displayed in the Research Panel, **Then** it appears as an "Expert" source with assistant name, confidence level, and internal references

---

### User Story 4 - User Defines Manual Research Steps with Source Selection (Priority: P2)

A power user wants precise control over their research methodology. They open a source browser, select specific data sources for each step, provide custom search prompts, and define their research workflow before execution.

**Why this priority**: Enables expert users to exercise fine-grained control. Builds on data source integrations to provide workflow customization.

**Independent Test**: Can be tested by opening the manual step editor, creating steps with specific sources, and executing. Delivers value by enabling controlled, repeatable research workflows.

**Acceptance Scenarios**:

1. **Given** a user opens the source browser, **When** viewing available sources, **Then** they see all sources grouped by category (Web Search, Vector Search, Genie, Assistants, Files) with descriptions and capabilities
2. **Given** a user creates a manual step, **When** selecting sources, **Then** they can select multiple sources and provide custom prompts describing what to search for in each
3. **Given** a user has defined multiple manual steps, **When** reordering via drag-and-drop, **Then** step order updates and is reflected in execution
4. **Given** "Manual Mode" is selected, **When** research executes, **Then** only user-defined steps run and the Planner is bypassed
5. **Given** "Hybrid Mode" is selected, **When** research executes, **Then** user-defined steps run first, followed by Planner-generated steps based on findings

---

### User Story 5 - User Creates and Uses Prompt Templates (Priority: P3)

A user wants to customize how the research agents behave and structure their output. They create prompt templates with variables, save them for reuse, and select them when starting new research queries.

**Why this priority**: Enables personalization and efficiency for repeated research patterns. Enhances user experience but not critical for core functionality.

**Independent Test**: Can be tested by creating a synthesis template, using it in a research query, and verifying output follows the template structure.

**Acceptance Scenarios**:

1. **Given** a user opens the template editor, **When** creating a new synthesis template, **Then** they can write template content with `{{variable}}` placeholders and define variable metadata
2. **Given** a template with required variables exists, **When** a user selects it for research, **Then** they are prompted to fill in variable values before proceeding
3. **Given** a user saves a template, **When** setting visibility to "workspace public", **Then** other workspace users can see and use the template
4. **Given** templates exist, **When** browsing the template library, **Then** templates can be filtered by type (system, step, synthesis, query) and searched by name/tags

---

### User Story 6 - User Creates and Uses Custom Research Agent (Priority: P3)

A user wants to create a specialized research assistant (like OpenAI's GPTs) that combines specific prompts, data sources, and workflow configuration. They define the agent once and select it for future research queries.

**Why this priority**: Advanced customization feature that combines all other features into reusable configurations. Valuable for power users and teams.

**Independent Test**: Can be tested by creating a custom agent with preset steps and specific sources, then using it for research.

**Acceptance Scenarios**:

1. **Given** a user opens the agent builder, **When** configuring a new agent, **Then** they can set identity (name, description, avatar), prompts, data sources, workflow, and output format
2. **Given** a custom agent has preset steps defined, **When** "Use Planner" is disabled, **Then** only the preset steps execute in order
3. **Given** a custom agent exists with workspace visibility, **When** other users open the agent selector, **Then** they can see and use the shared agent
4. **Given** a user selects a custom agent, **When** starting research, **Then** the agent's configuration (sources, depth, templates) applies automatically
5. **Given** a custom agent is selected, **When** the user wants to override settings, **Then** they can override research depth and source selection per-query

---

### User Story 7 - User Uploads Files for Research Context (Priority: P3)

A user has documents they want incorporated into their research (company templates, reference materials, previous reports). They upload files that become searchable sources available during research and synthesis.

**Why this priority**: Enables incorporation of user-provided context. Useful but not core to enterprise data source integration.

**Independent Test**: Can be tested by uploading a document, starting research, and verifying the document content appears in sources and can be cited.

**Acceptance Scenarios**:

1. **Given** a user drags files into the upload zone, **When** files are valid types and sizes, **Then** upload progress shows and files appear in the uploaded files list
2. **Given** files are uploaded, **When** viewing the source browser, **Then** uploaded files appear under "Uploaded Files" category and can be selected for manual steps
3. **Given** an uploaded file is selected as a source, **When** the Researcher executes, **Then** it can search the file content and include relevant chunks as sources
4. **Given** research is complete, **When** synthesis references uploaded file content, **Then** citations reference the filename and chunk location
5. **Given** file limits are configured, **When** a user exceeds max files or file size, **Then** they receive a clear error message indicating the limit

---

### User Story 8 - Plugin Registers Custom Data Sources (Priority: P2)

A plugin developer wants to extend the system with domain-specific data sources (e.g., CRM data via Vector Search, Sales Analytics via Genie). They register sources through the plugin system, which appear in the source browser with plugin attribution.

**Why this priority**: Critical for extensibility - enables domain-specific plugins to provide tailored data access.

**Independent Test**: Can be tested by creating a plugin that registers a Vector Search endpoint and verifying it appears in source browser.

**Acceptance Scenarios**:

1. **Given** a plugin implements DataSourceProvider, **When** it returns Vector Search endpoint definitions, **Then** those endpoints appear in the source browser with plugin attribution
2. **Given** a plugin defines source constraints, **When** the plugin is active, **Then** its required sources must be consulted and restricted sources are unavailable
3. **Given** a plugin provides a custom agent, **When** users view the agent selector, **Then** the plugin's agent appears with plugin attribution
4. **Given** a plugin provides prompt templates, **When** users view the template library, **Then** plugin templates appear with plugin attribution
5. **Given** a data source is queried, **When** lifecycle hooks are enabled, **Then** the plugin receives DataSourceQueryEvent with query details

---

### User Story 9 - User Adds Custom Data Source Connection (Priority: P2)

A user wants to connect their own Databricks data sources (Vector Search index, Genie room, or Knowledge Assistant) that aren't pre-configured by the system. They add the connection via UI, and it becomes available for their research.

**Why this priority**: Enables self-service data source configuration, reducing admin bottleneck and allowing users to leverage their own Databricks resources.

**Independent Test**: Can be tested by a user adding a Vector Search index they have access to and using it in research.

**Acceptance Scenarios**:

1. **Given** a user has access to a Vector Search index in their workspace, **When** they add it via the data source configuration UI, **Then** the system validates OBO access and saves the connection
2. **Given** a user adds a Genie room, **When** they provide the space ID and description, **Then** the source appears in their source browser
3. **Given** a user sets a data source to "Workspace" visibility, **When** other users browse sources, **Then** they see the shared source (if they have OBO access)
4. **Given** a user tries to add a data source they don't have access to, **When** OBO validation fails, **Then** they receive a clear error message about insufficient permissions

---

### User Story 9a - User Discovers and Selects Available Data Sources (Priority: P1)

A user wants to see ALL data sources they have access to in the workspace without manual configuration. They open a data source selector dropdown/list that automatically queries Databricks APIs to discover available Vector Search indexes, Genie spaces, and Knowledge Assistants, displaying them with metadata to help the user choose which sources to use for their research.

**Why this priority**: Critical for usability - users shouldn't need to manually configure every data source. Automatic discovery reduces friction and ensures users can leverage all available enterprise data.

**Independent Test**: Can be tested by a user opening the data source selector and verifying it shows all Vector Search indexes, Genie spaces, and serving endpoints they have OBO access to.

**Acceptance Scenarios**:

1. **Given** a user opens the data source selector, **When** the system queries Databricks APIs, **Then** it discovers all Vector Search endpoints and their indexes via `w.vector_search_endpoints.list_endpoints()` and `w.vector_search_indexes.list_indexes(endpoint_name)`
2. **Given** a user opens the data source selector, **When** the system queries Genie API, **Then** it discovers all accessible Genie spaces via `w.genie.list_spaces()` with space names and descriptions
3. **Given** a user opens the data source selector, **When** the system queries serving endpoints, **Then** it discovers Knowledge Assistants via `w.serving_endpoints.list()` filtered by endpoint type/tags
4. **Given** discovered data sources are displayed, **When** viewing the list, **Then** each source shows: name, type, description, and capabilities (query types supported, filter fields available)
5. **Given** a Vector Search index is discovered, **When** viewing its metadata, **Then** the user sees: supported query types (ANN, Hybrid, Full-Text), primary key, available filter columns (from Delta table schema), and embedding model
6. **Given** a Genie space is discovered, **When** viewing its metadata, **Then** the user sees: space name, description, connected warehouse, and example queries if available
7. **Given** data sources are discovered, **When** the user selects sources for research, **Then** selected sources persist and are used for source scope and manual step configuration
8. **Given** discovery returns many sources, **When** viewing the selector, **Then** sources are grouped by type (Vector Search, Genie, Assistants) and searchable/filterable

---

### User Story 9b - User Configures Query Settings Per Data Source (Priority: P2)

After discovering available data sources, a user can configure query settings specific to each source type - including query type selection (ANN vs Hybrid) for Vector Search and metadata filters for targeted queries.

**Why this priority**: Enables power users to optimize query performance and relevance by choosing appropriate query strategies per source.

**Independent Test**: Can be tested by selecting a Vector Search index and configuring it to use Hybrid search with specific filters.

**Acceptance Scenarios**:

1. **Given** a user selects a Vector Search index, **When** configuring query settings, **Then** they can choose query type: ANN (default, fastest), Hybrid (keyword + semantic), or Full-Text (keyword only)
2. **Given** a Vector Search index has filter columns, **When** configuring the source, **Then** the user can define filter expressions using available columns (SQL-like syntax or dictionary)
3. **Given** filter expressions are configured, **When** viewing the filter builder, **Then** the UI shows available operators per column type: comparison (<, >, =, !=), LIKE for strings, IN for lists
4. **Given** a user configures default query type as "Hybrid", **When** research executes, **Then** all queries to that index use hybrid keyword-similarity search
5. **Given** query settings are configured per source, **When** the settings are saved, **Then** they persist for the user's session and can be saved as defaults
6. **Given** a source is used without explicit configuration, **When** research executes, **Then** system uses sensible defaults: ANN for Vector Search, standard conversation for Genie

---

### User Story 10 - System Discovers Available Data Before Planning (Priority: P1)

When a user submits a research query, the system first explores ALL configured data sources (web, Vector Search, Genie, Knowledge Assistants, uploaded files) to understand what data is available. This "data landscape" informs the Planner's decisions about which sources to use for each step.

**Why this priority**: Critical for intelligent source routing - without discovery, planner would guess blindly about which sources have relevant data. This is the foundation of multi-source research.

**Independent Test**: Can be tested by configuring multiple data sources and verifying discovery queries all sources in parallel before planning begins.

**Acceptance Scenarios**:

1. **Given** a user has configured Vector Search indexes and Genie spaces, **When** they submit a research query, **Then** the system queries all sources with exploratory queries BEFORE generating the plan
2. **Given** discovery finds relevant data in Vector Search but not Genie, **When** the Planner generates steps, **Then** it prioritizes Vector Search and deprioritizes Genie
3. **Given** a data source returns no relevant results during discovery, **When** viewing the data landscape, **Then** that source is marked as "low relevance" but still available for planner to use
4. **Given** discovery runs across 5+ sources, **When** sources are queried, **Then** queries run in parallel to minimize latency (< 5 seconds total)

---

### User Story 11 - Planner Generates Source-Aware Steps (Priority: P1)

After background discovery, the Planner receives a summary of the data landscape and generates steps that include source routing hints. Each step specifies which sources should be consulted, in what priority order, with optional query hints.

**Why this priority**: Core improvement to planning quality - enables intelligent routing instead of querying all sources for every step.

**Independent Test**: Can be tested by submitting a query that requires both enterprise and web data, then verifying generated steps include appropriate source hints.

**Acceptance Scenarios**:

1. **Given** the data landscape shows Vector Search has product docs, **When** a step needs product information, **Then** the Planner includes Vector Search as a required source for that step
2. **Given** a step needs recent market data, **When** the Planner generates the step, **Then** it routes to Web Search (not enterprise sources) with appropriate query hints
3. **Given** a step could benefit from multiple sources, **When** the Planner generates source hints, **Then** it includes multiple sources with priority ordering (1=required, 2=recommended, 3=optional)
4. **Given** the Planner outputs a plan, **When** viewing the plan, **Then** each step shows its source routing (which sources, what priority)

---

### User Story 12 - User Controls Source Scope (Priority: P1)

Users can control which categories of sources are available for their research via simple toggles. They can also enable/disable individual sources and optionally review/edit the generated plan before execution.

**Why this priority**: Essential for user control - some research should only use enterprise data, some should only use web, some should use both.

**Independent Test**: Can be tested by selecting "Enterprise Only" scope and verifying no web searches are performed during research.

**Acceptance Scenarios**:

1. **Given** a user views the research input form, **When** they see source scope options, **Then** they can choose: "Enterprise Only", "Web Only", or "All Sources" (default)
2. **Given** "Enterprise Only" is selected, **When** research executes, **Then** only Vector Search, Genie, Knowledge Assistants, and Files are queried (no web search)
3. **Given** a user wants fine-grained control, **When** they expand source options, **Then** they can enable/disable individual sources (e.g., disable a specific Vector Search index)
4. **Given** plan review is enabled, **When** the Planner generates a plan, **Then** the user sees the plan with source assignments and can edit steps, reorder, or change sources before execution
5. **Given** plan review is disabled (default), **When** the Planner generates a plan, **Then** research proceeds automatically without user intervention

---

### Edge Cases

- What happens when a Vector Search endpoint is unavailable or returns errors?
  - System logs the error, continues with other sources, and indicates the unavailable source in the Research Panel
- How does the system handle Genie query timeouts?
  - Query is cancelled after configurable timeout, partial results returned if available, user notified of timeout
- What happens when a user exceeds file upload limits mid-session?
  - Clear error message shown, existing files preserved, user can remove files to make room
- How does the system handle conflicting source constraints (allowlist vs required)?
  - Required sources must be in allowlist; if conflict detected, system warns and prioritizes required sources
- What happens when a custom agent references a template that was deleted?
  - System falls back to default prompts and notifies user of missing template
- How does the system handle assistant responses with low confidence?
  - Response is still included but marked with confidence indicator; user can filter by confidence if desired
- What happens when user lacks permission to access a data source via OBO?
  - Source query returns permission error, system continues with other accessible sources, user sees which sources were inaccessible due to permissions
- What happens when a workspace-shared agent/template is deleted (owner removed)?
  - Users who had selected it see "resource no longer available" message and fall back to default agent/no template
- What happens when one tool in a parallel batch fails while others succeed?
  - System returns error message for the failed tool, successful results are preserved, research continues with partial results
- What happens when concurrent state mutations attempt to add duplicate sources?
  - asyncio.Lock ensures atomic check-then-act, only one source added, no duplicates
- How does the system handle tool dependencies when LLM generates web_crawl before web_search?
  - Topological sort ensures web_search executes first regardless of LLM ordering, web_crawl waits in subsequent batch
- What happens when parallel execution exceeds rate limits?
  - Existing per-client rate limiters (already async-safe) queue excess requests; same-source parallelism is intentionally limited
- What happens when parallel_tool_calls is disabled for a model tier?
  - System falls back to sequential execution, performance is slower but correctness preserved
- What happens when a tool times out during parallel execution?
  - Tool receives timeout error message, other tools in batch continue, batch completes with partial results
- What happens when parallel results exceed context window budget?
  - System tracks token count, truncates lowest-relevance results, logs warning about context pressure
- What happens when discovery API calls fail for one source type but succeed for others?
  - System displays available sources from successful calls, shows error indicator for failed type, offers retry button
- What happens when a Vector Search index is discovered but has status "not ready"?
  - Index is shown in selector with "syncing" status indicator, greyed out and unselectable until ready
- What happens when discovery cache expires during a research session?
  - Background refresh triggered, existing selections preserved, user notified if selected source becomes unavailable
- What happens when user selects Hybrid query type but index lacks text columns?
  - System falls back to ANN with warning message explaining Hybrid requires text metadata columns
- What happens when filter expression has syntax errors?
  - Real-time validation shows error, prevents save, offers suggested corrections based on column type
- What happens when filter expression exceeds 1,024 ID limit?
  - System warns user, suggests batching strategy, or auto-splits into OR clauses if under 2,048 total
- What happens when Genie space has no description or example queries?
  - Space is still shown in selector with name only, metadata section shows "No description available"
- What happens when serving endpoint tags don't clearly identify it as a Knowledge Assistant?
  - System uses heuristics (endpoint name patterns, model type) and allows user to manually classify endpoints

---

## Requirements *(mandatory)*

### Functional Requirements - Data Sources (F1-F3)

- **FR-001**: System MUST execute semantic similarity searches against configured Databricks Vector Search endpoints
- **FR-002**: System MUST support natural language queries that are automatically embedded using the endpoint's embedding model
- **FR-003**: System MUST apply metadata filters to Vector Search queries when filters are configured and applicable
- **FR-004**: System MUST support querying multiple Vector Search indexes within a single research session
- **FR-005**: System MUST treat Vector Search results as sources equivalent to crawled web pages, including content, source reference, relevance score, and metadata
- **FR-006**: System MUST deduplicate Vector Search results when the same content appears across multiple queries
- **FR-007**: System MUST execute natural language queries against configured Databricks Genie spaces, translating to SQL
- **FR-008**: System MUST support multiple Genie spaces, each with description and example questions
- **FR-009**: System MUST maintain Genie conversation context within a research session for follow-up queries
- **FR-010**: System MUST automatically summarize Genie tabular results into narrative insights
- **FR-011**: System MUST truncate large Genie result sets with total row count indication
- **FR-012**: System MUST display generated SQL for Genie queries for transparency
- **FR-013**: System MUST send natural language questions to configured Knowledge Assistants
- **FR-014**: System MUST support querying the Supervisor Agent for questions spanning multiple domains
- **FR-015**: System MUST optionally pass current research context to assistants for more targeted answers
- **FR-016**: System MUST preserve assistant confidence levels and internal knowledge base references
- **FR-016a**: System MUST use On-Behalf-Of (OBO) authentication for all enterprise data source queries, acting under the authenticated user's identity so that access follows their existing workspace permissions
- **FR-016b**: Users MUST be able to add their own data source connections (Vector Search indexes, Genie rooms, Knowledge Assistants) via UI
- **FR-016c**: When a user adds a data source, system MUST validate the user has access to it via OBO before saving
- **FR-016d**: User-added data sources MUST support visibility levels: Private (only creator) or Workspace (all users who have OBO access)
- **FR-016e**: User-added data sources MUST appear in the source browser alongside system-configured and plugin-registered sources

### Functional Requirements - Data Source Discovery (F1a - US9a/US9b)

**Discovery APIs (via OBO authentication)**:
- **FR-126**: System MUST discover Vector Search endpoints using `WorkspaceClient.vector_search_endpoints.list_endpoints()` → Iterator[EndpointInfo]
- **FR-127**: System MUST discover Vector Search indexes per endpoint using `WorkspaceClient.vector_search_indexes.list_indexes(endpoint_name)` → Iterator[MiniVectorIndex]
- **FR-128**: System MUST retrieve index metadata using `WorkspaceClient.vector_search_indexes.get_index(index_name)` → VectorIndex (includes schema, primary_key, index_type, status)
- **FR-129**: System MUST discover Genie spaces using `WorkspaceClient.genie.list_spaces()` → GenieListSpacesResponse
- **FR-130**: System MUST retrieve Genie space details using `WorkspaceClient.genie.get_space(space_id)` → includes title, description, warehouse_id
- **FR-131**: System MUST discover serving endpoints (Knowledge Assistants) using `WorkspaceClient.serving_endpoints.list()` → Iterator[ServingEndpoint]
- **FR-132**: System MUST filter serving endpoints to identify Knowledge Assistants by endpoint type, tags, or naming convention

**Metadata Extraction**:
- **FR-133**: For Vector Search indexes, system MUST extract: index_name, endpoint_name, primary_key, index_type (DELTA_SYNC or DIRECT_ACCESS), status (ready/not ready), embedding_source_columns, embedding_dimensions
- **FR-134**: For Vector Search indexes, system MUST determine available filter columns from the Delta table schema (any column can be filtered)
- **FR-135**: For Vector Search indexes, system MUST indicate supported query types: ANN (always), HYBRID (if text columns exist), FULL_TEXT (if enabled)
- **FR-136**: For Genie spaces, system MUST extract: space_id, title, description, warehouse_id, owner
- **FR-137**: For serving endpoints, system MUST extract: endpoint_name, endpoint_type, state, tags, creator

**Query Type Configuration**:
- **FR-138**: System MUST support configuring query_type per Vector Search index: "ANN" (default), "HYBRID", or "FULL_TEXT"
- **FR-139**: ANN query type MUST use approximate nearest neighbor search with HNSW algorithm and L2 distance metric
- **FR-140**: HYBRID query type MUST combine vector embedding search with keyword matching (max 200 results)
- **FR-141**: FULL_TEXT query type (beta) MUST use keyword-based retrieval without embeddings (max 200 results)
- **FR-142**: System MUST recommend HYBRID when domain-specific keywords are critical, ANN otherwise for best performance

**Filter Configuration**:
- **FR-143**: System MUST support filter expressions for Vector Search queries based on any Delta table column
- **FR-144**: System MUST support SQL-like filter syntax for storage-optimized endpoints (e.g., `"category = 'docs' AND date > '2024-01-01'"`)
- **FR-145**: System MUST support dictionary filter syntax for standard endpoints (e.g., `{"category": "docs", "date >": "2024-01-01"}`)
- **FR-146**: System MUST enforce filter ID limit of 1,024 per clause, with guidance to batch using OR when exceeding
- **FR-147**: System MUST support filter operators: comparison (<, <=, >, >=, =, !=), LIKE (pattern matching), NOT LIKE, IN (list membership)

**UI Requirements**:
- **FR-148**: System MUST provide a data source selector dropdown/list accessible from the research input form
- **FR-149**: Data source selector MUST group sources by type: Vector Search, Genie, Knowledge Assistants, Files
- **FR-150**: Data source selector MUST support search/filter by source name
- **FR-151**: Each source in selector MUST display: name, type icon, description (truncated), status indicator (available/unavailable)
- **FR-152**: Clicking a source MUST expand to show full metadata: query types, filter columns, endpoint details
- **FR-153**: System MUST cache discovery results for configurable duration (default 5 minutes) to avoid repeated API calls
- **FR-154**: System MUST show loading state during initial discovery and refresh button for manual refresh
- **FR-155**: System MUST gracefully handle discovery failures (show error, allow retry, don't block other sources)

### Functional Requirements - Source Constraints (F4)

- **FR-017**: System MUST allow specifying allowlist constraints defining which source types/endpoints a step CAN use
- **FR-018**: System MUST allow specifying required constraints defining source types/endpoints that MUST be consulted
- **FR-019**: System MUST consider a step incomplete if required sources were not consulted
- **FR-020**: System MUST prompt the Researcher to use required sources during step execution
- **FR-021**: System MUST support specifying multiple Vector Search indexes as allowed sources for a single step
- **FR-022**: When Planner generates steps, it MUST be able to include recommended source constraints
- **FR-023**: Users MUST be able to override Planner-generated constraints when using manual steps

### Functional Requirements - Manual Steps (F5)

- **FR-024**: System MUST provide a source browser UI showing all available data sources grouped by category
- **FR-025**: Source browser MUST display: Web Search, Vector Search Indexes, Genie Spaces, Knowledge Assistants, Uploaded Files
- **FR-026**: Each source in browser MUST show name, description, example queries, and capabilities
- **FR-027**: Users MUST be able to select one or more sources for each manual step
- **FR-028**: For each selected source, users MUST be able to provide custom prompts describing what to search for
- **FR-029**: Users MUST be able to provide optional filters for sources that support filtering
- **FR-030**: Users MUST be able to define step title and objective
- **FR-031**: Users MUST be able to reorder steps via drag-and-drop
- **FR-032**: Users MUST be able to add, edit, or remove steps before submitting
- **FR-033**: System MUST support three workflow modes: Planner (default), Manual, and Hybrid
- **FR-034**: In Manual mode, system MUST bypass Planner and execute only user-defined steps
- **FR-035**: In Hybrid mode, system MUST execute user-defined steps first, then allow Planner to add additional steps

### Functional Requirements - Templates (F6)

- **FR-036**: System MUST support four template types: System Prompt, Step Prompt, Synthesis, and Query templates
- **FR-037**: Templates MUST support `{{variable}}` placeholder syntax
- **FR-038**: Template variables MUST have configurable metadata: name, type, required flag, default value
- **FR-039**: Users MUST be able to create, edit, duplicate, and delete templates
- **FR-040**: Users MUST be able to organize templates with tags and categories
- **FR-041**: Templates MUST be private by default (only creator can see)
- **FR-042**: Users MUST be able to mark templates as "workspace public" for sharing
- **FR-043**: System MUST provide default templates for common use cases
- **FR-044**: When selecting a template with required variables, system MUST prompt user to fill values before proceeding

### Functional Requirements - Custom Agents (F7)

- **FR-045**: Users MUST be able to define custom agents with: name, description, avatar
- **FR-046**: Custom agents MUST support system prompt template configuration (reference or inline)
- **FR-047**: Custom agents MUST support synthesis template configuration (reference or inline)
- **FR-048**: Custom agents MUST support data source configuration: enabled sources, disabled sources, constraints
- **FR-049**: Custom agents MUST support workflow configuration: Use Planner toggle, preset steps, default depth, default mode, clarification toggle
- **FR-050**: Custom agents MUST support output configuration: default format (markdown/JSON), output schema
- **FR-051**: When Planner is disabled, system MUST execute only preset steps in defined order
- **FR-052**: Each preset step MUST have: title, description, attached sources, order, required/optional flag
- **FR-053**: Custom agents MUST support three visibility levels: Private, Workspace, System
- **FR-054**: Users MUST be able to browse available agents (own, workspace, system) via agent selector
- **FR-055**: Users MUST be able to override agent settings (depth, sources) per-query
- **FR-056**: System MUST provide a "Default" agent representing standard deep research behavior
- **FR-056a**: System MUST delete all templates and custom agents owned by a user when that user is removed from the workspace

### Functional Requirements - File Upload (F8)

- **FR-057**: System MUST support uploading text files (.txt, .md), documents (.pdf, .docx), spreadsheets (.csv, .xlsx), code files, and data files (.json, .yaml)
- **FR-058**: System MUST enforce configurable limits: max files per session, max file size, max total size
- **FR-059**: System MUST NOT perform OCR on scanned PDFs or images
- **FR-060**: System MUST split uploaded files into chunks using configurable chunk size and overlap
- **FR-061**: Chunking MUST use paragraph/page boundaries where possible, otherwise by size (naive chunking)
- **FR-062**: Uploaded files MUST become searchable sources for the research session duration
- **FR-063**: Files MUST be automatically deleted after configurable retention period
- **FR-064**: Users MUST be able to preview uploaded file content before research
- **FR-065**: Users MUST be able to remove uploaded files before or during research
- **FR-066**: Citations to uploaded files MUST reference filename and chunk location

### Functional Requirements - Plugin Integration

- **FR-067**: Plugins MUST be able to register Vector Search endpoints via DataSourceProvider protocol
- **FR-068**: Plugins MUST be able to register Genie spaces via DataSourceProvider protocol
- **FR-069**: Plugins MUST be able to register Knowledge Assistants via DataSourceProvider protocol
- **FR-070**: Plugins MUST be able to register custom data source types via DataSourceProvider protocol
- **FR-071**: Plugin-registered sources MUST appear in source browser with plugin attribution
- **FR-072**: Plugins MUST be able to define default source constraints for their domain
- **FR-073**: Plugins MUST be able to require their sources be consulted when active
- **FR-074**: Plugins MUST be able to restrict available sources when active
- **FR-075**: Plugins MUST be able to provide prompt templates via TemplateProvider protocol
- **FR-076**: Plugins MUST be able to resolve template variables dynamically based on context
- **FR-077**: Plugins MUST be able to define complete custom agents via CustomAgentProvider protocol
- **FR-078**: Plugin agents MUST be able to reference plugin sources, templates, and steps
- **FR-079**: Plugins MUST be able to add supported file types via FileProcessorProvider protocol
- **FR-080**: Plugins MUST be able to provide custom file processing
- **FR-081**: Plugins MUST be able to configure UI elements: visibility, defaults, locked selections, pre-selections
- **FR-082**: Plugins MUST receive lifecycle events: data source queries, template applications, agent selections, file uploads

### Functional Requirements - Multi-Source Background Discovery (F9)

- **FR-083**: System MUST run background discovery across ALL enabled sources before planning
- **FR-084**: Discovery MUST generate exploratory queries dynamically based on user's research prompt
- **FR-085**: Discovery MUST query sources in parallel to minimize latency
- **FR-086**: Discovery MUST collect: relevance score, sample results, available filters, suggested queries per source
- **FR-087**: Discovery results MUST be summarized into a DataLandscape structure for the Planner
- **FR-088**: Discovery MUST complete within 5 seconds for up to 10 sources
- **FR-089**: For Genie sources, discovery MUST first try metadata query, then sample query if ambiguous
- **FR-090**: Discovery MUST respect source scope settings (skip disabled sources)

### Functional Requirements - Source-Aware Planning (F10)

- **FR-091**: Planner MUST receive DataLandscape as input alongside the user query
- **FR-092**: Planner MUST output `source_hints` for each step specifying which sources to use
- **FR-093**: Each source hint MUST include: source name, type, priority (1-3), optional query hint, optional filters
- **FR-094**: Planner MAY route to ANY available source, not only those with high discovery scores
- **FR-095**: Planner MUST consider source capabilities when routing (Vector Search for semantic, Genie for SQL/metrics)
- **FR-096**: Planner MUST output `exclude_sources` list for steps where certain sources should NOT be used
- **FR-097**: Researcher MUST filter available tools per-step based on source hints
- **FR-098**: Researcher MUST track per-source query counts and respect source budgets

### Functional Requirements - Source Scope & Plan Review (F11)

- **FR-099**: System MUST provide source scope selector with options: Enterprise Only, Web Only, All (default)
- **FR-100**: System MUST allow users to enable/disable individual sources via UI
- **FR-101**: Source scope and individual settings MUST persist per user/session
- **FR-102**: System MUST support optional plan review before execution (configurable)
- **FR-103**: Plan review UI MUST allow: editing step titles/descriptions, reordering steps, changing source assignments
- **FR-104**: Plan review MUST support approval timeout (auto-proceed after configurable duration)
- **FR-105**: Custom agents MUST be able to specify source scope and per-step source overrides
- **FR-106**: Research request API MUST accept: source_scope, enabled_sources, disabled_sources parameters

### Functional Requirements - Parallel Tool Execution (F12)

**State Safety (Prerequisites)**
- **FR-107**: System MUST implement async-safe state mutations using `asyncio.Lock` (NOT threading.RLock) for ResearchState methods
- **FR-108**: System MUST use granular per-collection locks (sources, claims, evidence) to minimize contention
- **FR-109**: System MUST provide sync-compatible wrappers that detect async context and schedule appropriately

**Tool Execution**
- **FR-110**: System MUST support executing different SOURCE TYPES concurrently (Web + Vector Search + Genie simultaneously)
- **FR-111**: System MUST respect tool dependencies when executing (web_crawl depends on web_search URL registry)
- **FR-112**: System MUST group tool calls by SOURCE TYPE (not just tool name) for optimal cross-source parallelism
- **FR-113**: System MUST return parallel execution results in original tool call order for correct message history
- **FR-114**: System MUST handle partial failures gracefully (individual tool errors don't fail the entire batch)

**Timeout & Budget Management**
- **FR-115**: System MUST apply per-tool timeout using `asyncio.wait_for()` to prevent hanging
- **FR-116**: System MUST support atomic budget reservation before starting parallel batches
- **FR-117**: System MUST track context window token usage and truncate results if needed

**Observability & Configuration**
- **FR-118**: System MUST log parallel execution metrics (source groups, elapsed time, success/error counts)
- **FR-119**: System MUST create MLflow spans for parallel batches with child spans per tool
- **FR-120**: System MUST support per-model-tier configuration of `parallel_tool_calls` API parameter
- **FR-121**: System MUST support disabling parallel execution via configuration fallback to sequential mode
- **FR-122**: System MUST yield events as tools complete (not batched) for real-time UI feedback

**Rate Limiting Awareness**
- **FR-123**: System MUST respect existing per-client rate limiters (BraveSearchClient, WebCrawler)
- **FR-124**: System MUST NOT parallelize same-source tools beyond what rate limiters allow (limited benefit)
- **FR-125**: System SHOULD prioritize cross-source parallelism over same-source parallelism for maximum benefit

### Key Entities

- **DataSource**: Represents a queryable data source (type, endpoint identifier, description, capabilities, filter schema, example queries); can be system-configured, plugin-registered, or user-added
- **UserDataSource**: User-configured data source connection (owner, type, endpoint identifier, description, visibility, validation status)
- **SourceConstraint**: Defines allowed, required, and excluded sources for a step (allowlist, required list, exclusion list)
- **ManualStep**: User-defined research step (title, objective, attached sources with prompts/filters, order)
- **PromptTemplate**: Reusable prompt configuration (type, content with placeholders, variable definitions, visibility)
- **CustomAgent**: Complete agent definition (identity, prompts, sources, workflow config, output config, visibility)
- **UploadedFile**: User-uploaded document (filename, type, size, chunks, session reference, expiry)
- **FileChunk**: Searchable portion of uploaded file (content, source file reference, chunk index, metadata)
- **DataLandscape**: Complete view of available data for a query (web sources, enterprise sources, uploaded files, recommended sources, source capabilities)
- **SourceDiscoveryResult**: Result of exploring a single source (source name, type, has_relevant_data, relevance_score, sample_results, available_filters, suggested_queries)
- **StepSourceHint**: Source routing hint for a plan step (source_name, source_type, priority, query_hint, filters)
- **SourceScope**: Enum defining source categories (ENTERPRISE_ONLY, WEB_ONLY, ALL)
- **ParallelToolConfig**: Configuration for parallel tool execution (enabled, max_parallel_per_batch, timeout_per_tool_seconds)
- **ToolDependency**: Defines execution order constraints between tool types (e.g., web_crawl depends on web_search)
- **DiscoveredVectorSearchIndex**: Auto-discovered VS index (index_name, endpoint_name, primary_key, index_type, status, embedding_columns, filter_columns, supported_query_types)
- **DiscoveredGenieSpace**: Auto-discovered Genie space (space_id, title, description, warehouse_id, owner)
- **DiscoveredServingEndpoint**: Auto-discovered serving endpoint/assistant (endpoint_name, endpoint_type, state, tags, creator)
- **VectorSearchQueryConfig**: Per-index query configuration (query_type: ANN|HYBRID|FULL_TEXT, default_filters, num_results, score_threshold)
- **FilterExpression**: Configured filter for Vector Search (column, operator, value, syntax_type: SQL|DICT)

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can access enterprise data (Vector Search, Genie, Assistants) in at least 90% of research sessions where relevant enterprise data exists
- **SC-002**: Users can create and execute manual research steps in under 5 minutes of setup time
- **SC-003**: Users can create a custom agent with preset steps in under 10 minutes
- **SC-004**: 80% of users who create custom agents reuse them for subsequent research queries
- **SC-005**: File uploads complete processing (chunking, indexing) within 30 seconds for files under 5MB
- **SC-006**: Plugin-registered data sources appear in source browser within 2 seconds of plugin activation
- **SC-007**: Research queries using enterprise sources return results with relevance comparable to web sources (measured by user satisfaction ratings)
- **SC-008**: 70% of power users adopt manual step definition or custom agents within 30 days of feature availability
- **SC-009**: System supports at least 10 concurrent Vector Search queries without degradation
- **SC-010**: Template creation to first use takes under 3 minutes for typical use cases
- **SC-011**: Background discovery completes in < 5 seconds for up to 10 sources
- **SC-012**: 90% of research queries with enterprise sources correctly route internal-data steps to enterprise sources
- **SC-013**: Users who enable plan review edit at least one source assignment in 30% of sessions (indicating value of control)
- **SC-014**: Research using source scope "Enterprise Only" queries zero web sources
- **SC-015**: Cross-source parallel execution achieves 30-50% latency reduction when querying 2+ different source types (Web + VS, Web + Genie, etc.)
- **SC-016**: Same-source parallel execution achieves 10-20% improvement (limited by rate limiters)
- **SC-017**: No duplicate sources appear in research results when parallel execution is enabled
- **SC-018**: System maintains correct message ordering in LLM conversation history during parallel execution
- **SC-019**: No tool execution exceeds the configured timeout (default 30s)
- **SC-020**: Data source discovery completes in < 3 seconds for workspaces with up to 50 sources
- **SC-021**: 95% of users can find and select relevant data sources without manual configuration
- **SC-022**: Discovery cache reduces API calls by 80% during typical research sessions
- **SC-023**: Users can configure query type (ANN/Hybrid) and see impact on result quality within 2 interactions
- **SC-024**: Filter configuration UI shows all available columns with appropriate operators within 1 second of index selection

---

## Assumptions

- Databricks Vector Search endpoints are pre-configured and accessible via workspace credentials
- Databricks Genie spaces are pre-configured with appropriate data access permissions
- Knowledge Assistants are deployed as serving endpoints accessible via workspace credentials
- Users have appropriate permissions to query configured data sources
- File storage for uploads uses Databricks Volumes or equivalent persistent storage
- Plugin system infrastructure (discovery, registration, lifecycle hooks) exists as documented
- Frontend framework supports drag-and-drop and complex form interactions
- Existing source display patterns in Research Panel can be extended for new source types

---

## Scope Boundaries

### In Scope

- Integration with Databricks Vector Search, Genie, and Knowledge Assistants
- Per-step source constraints (allowlist, required, excluded)
- Manual step definition UI with source browser
- Prompt template creation, management, and sharing
- Custom agent definition with preset steps and source configuration
- File upload with naive chunking (paragraph/page boundaries, size-based fallback)
- Plugin extension points for all features

### Out of Scope

- OCR for scanned documents or images
- Sophisticated document parsing (tables, complex formatting preservation)
- Semantic chunking or intelligent document structure analysis
- Real-time collaborative editing of templates or agents
- Version control or history for templates and agents
- Cross-workspace sharing of templates or agents
- Integration with non-Databricks vector databases
- Integration with non-Databricks BI tools

---

## Clarifications

### Session 2026-02-04

- Q: Authorization model for enterprise data sources? → A: On-Behalf-Of (OBO) - system acts using the authenticated user's identity; source access follows their existing workspace permissions
- Q: Custom agent/template lifecycle on owner removal? → A: Auto-delete - all user's templates and custom agents are deleted when user is removed
- Q: Data source configuration location? → A: User-configurable - users can add their own Genie rooms, Vector Search indexes, and other Databricks data sources via UI (access validated via OBO)

---

## Dependencies

- Databricks SDK for Vector Search, Genie, and Assistant APIs
- Existing plugin architecture (discovery, protocols, lifecycle hooks)
- Existing tool registry for researcher tool management
- Existing Research Panel UI for source display
- Frontend component library for forms, drag-and-drop, file upload
- Database infrastructure for storing templates, agents, and file metadata
