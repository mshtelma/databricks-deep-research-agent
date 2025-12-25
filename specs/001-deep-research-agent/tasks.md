# Tasks: Deep Research Agent

**Input**: Design documents from `/specs/001-deep-research-agent/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/openapi.yaml

**Tests**: Tests are NOT explicitly requested in this specification. Test tasks are omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project directory structure per plan.md (backend/src/, frontend/src/)
- [X] T002 [P] Initialize Python backend with uv, pyproject.toml, and dependencies (FastAPI, httpx, pydantic, sqlalchemy, mlflow)
- [X] T003 [P] Initialize React frontend with Vite, TypeScript, Tailwind CSS, shadcn/ui
- [X] T004 [P] Configure ruff for Python linting/formatting in `backend/pyproject.toml`
- [X] T005 [P] Configure ESLint and Prettier for TypeScript in `frontend/`
- [X] T006 [P] Create `.env.example` with required environment variables (DATABRICKS_HOST, BRAVE_API_KEY, etc.)
- [X] T007 [P] Create `backend/src/core/config.py` with Pydantic settings from environment
- [X] T008 [P] Create `docker-compose.yaml` for local PostgreSQL development
- [X] T009 Create `backend/scripts/quickstart.sh` for local development setup
- [X] T010 [P] Create `.gitignore` for Python and Node.js artifacts
- [X] T010a [P] Create `config/models.yaml` with model endpoint configuration template

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Database & Models

- [X] T011 Create database schema migrations framework with Alembic in `backend/src/db/`
- [X] T012 [P] Create SQLAlchemy base model with UUID primary key mixin in `backend/src/db/base.py`
- [X] T013 Create `chats` table migration in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T014 Create `messages` table migration with FTS index in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T015 Create `research_sessions` table migration in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T016 Create `sources` table migration in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T017 [P] Create `user_preferences` table migration in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T018 [P] Create `message_feedback` table migration in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T019 [P] Create `audit_logs` table migration in `backend/src/db/migrations/` (in 001_initial_schema.py)
- [X] T020 [P] Create Chat SQLAlchemy model in `backend/src/models/chat.py`
- [X] T021 [P] Create Message SQLAlchemy model in `backend/src/models/message.py`
- [X] T022 [P] Create ResearchSession SQLAlchemy model in `backend/src/models/research_session.py`
- [X] T023 [P] Create Source SQLAlchemy model in `backend/src/models/source.py`
- [X] T024 [P] Create UserPreferences SQLAlchemy model in `backend/src/models/user_preferences.py`
- [X] T025 [P] Create MessageFeedback SQLAlchemy model in `backend/src/models/message_feedback.py`
- [X] T026 [P] Create AuditLog SQLAlchemy model in `backend/src/models/audit_log.py`

### Authentication & Middleware

- [X] T027 Create Databricks authentication middleware using WorkspaceClient in `backend/src/middleware/auth.py`
- [X] T028 Create user identity extraction from OBO token in `backend/src/core/auth.py`
- [X] T029 [P] Create audit logging middleware in `backend/src/middleware/audit.py`
- [X] T030 [P] Create request/response logging middleware in `backend/src/middleware/logging.py`

### API Infrastructure

- [X] T031 Create FastAPI application factory with CORS in `backend/src/main.py`
- [X] T032 [P] Create Pydantic schemas for API requests/responses in `backend/src/schemas/`
- [X] T033 [P] Create pagination helpers in `backend/src/core/pagination.py`
- [X] T034 [P] Create error handling with custom exceptions in `backend/src/core/exceptions.py`
- [X] T035 Create database session dependency in `backend/src/db/session.py`

### LLM Service Layer

- [X] T036 Create LLMService interface with complete(), stream(), structured() in `backend/src/services/llm/service.py`
- [X] T037 Create ModelRouter with role-based endpoint selection in `backend/src/services/llm/router.py`
- [X] T038 [P] Create EndpointHealth class for tracking availability in `backend/src/services/llm/health.py`
- [X] T039 [P] Create TokenBucketRateLimiter for per-endpoint rate limiting in `backend/src/services/llm/rate_limiter.py`
- [X] T040 Create DatabricksLLMClient with retry logic in `backend/src/services/llm/client.py`
- [X] T041 [P] Create model configuration loader from YAML in `backend/src/services/llm/config.py`
- [X] T042 [P] JSON repair utility integrated in `src/services/llm/client.py` via json-repair library

### Web Tools

- [X] T043 Create BraveSearchClient with rate limiting in `backend/src/services/search/brave.py`
- [X] T043a Create web_search async wrapper using BraveSearchClient in `backend/src/agent/tools/web_search.py`
- [X] T044 Create WebCrawler with async HTTP and HTML parsing in `backend/src/agent/tools/web_crawler.py`
- [X] T045 [P] Content extraction/cleaning integrated in `src/agent/tools/web_crawler.py` via trafilatura

### MLflow Tracing Infrastructure

- [X] T046 Create setup_tracing() with mlflow.openai.autolog() in `backend/src/core/tracing.py`
- [X] T047 [P] Enable async logging via mlflow.config.enable_async_logging(True) in `backend/src/core/tracing.py`
- [X] T048 [P] Configure MLflow experiment for trace grouping in `backend/src/core/tracing.py`

### Frontend Infrastructure

- [X] T049 Create React Router configuration in `frontend/src/App.tsx`
- [X] T050 [P] Create TanStack Query client provider in `frontend/src/main.tsx`
- [X] T051 [P] Generate OpenAPI client from contracts/openapi.yaml in `frontend/src/api/`
- [X] T052 [P] Create cn() utility for Tailwind class merging in `frontend/src/lib/utils.ts`
- [X] T053 [P] Install shadcn/ui base components (Button, Card, Input, etc.) in `frontend/src/components/ui/`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 2.5: Frontend-Backend Consolidation

**Purpose**: Unify frontend and backend into single deployment for Databricks Apps

- [X] T_CONS_1 [P] Create `backend/src/static_files.py` for SPA serving with fallback routing
- [X] T_CONS_2 Modify `backend/src/main.py` to call setup_static_files() at end of create_app()
- [X] T_CONS_3 [P] Update `frontend/vite.config.ts` outDir to ../backend/static
- [X] T_CONS_4 [P] Create root `Makefile` with build automation (dev, build, prod targets)
- [X] T_CONS_5 [P] Update `.gitignore` to exclude backend/static/
- [X] T_CONS_6 [P] Update `backend/app.yaml` with SERVE_STATIC=true environment variable
- [X] T_CONS_7 Update CLAUDE.md with new Makefile commands
- [X] T_CONS_8 Update quickstart.md with unified deployment instructions

**Checkpoint**: Unified deployment ready - single container serves both API and frontend

---

## Phase 3: User Story 1 - Conduct Deep Research via Chat (Priority: P1) 🎯 MVP

**Goal**: Users can submit research queries and receive comprehensive answers with citations

**Independent Test**: Submit a research query and verify the agent searches web, fetches pages, and returns synthesized answer with sources

### Multi-Agent State Management

- [X] T054 [US1] Create ResearchState dataclass with all agent phases in `backend/src/agent/state.py`
- [X] T055 [P] [US1] Create Plan and PlanStep Pydantic models in `backend/src/agent/models/plan.py`
- [X] T056 [P] [US1] Create ReflectionResult and ReflectionDecision models in `backend/src/agent/models/reflection.py`
- [X] T057 [P] [US1] Create StreamEvent types (AgentEvent, StepEvent, ContentEvent, etc.) in `backend/src/agent/models/events.py`

### Agent Nodes (5-Agent Architecture)

- [X] T058 [P] [US1] Implement Coordinator agent (query classification, simple query handling) in `backend/src/agent/nodes/coordinator.py`
- [X] T059 [US1] Add @mlflow.trace(span_type=AGENT, agent_role="coordinator") to coordinator
- [X] T060 [P] [US1] Implement BackgroundInvestigator agent (quick web search for context) in `backend/src/agent/nodes/background.py`
- [X] T061 [US1] Add @mlflow.trace(span_type=AGENT, agent_role="background") to background investigator
- [X] T062 [P] [US1] Implement Planner agent (create structured research plan) in `backend/src/agent/nodes/planner.py`
- [X] T063 [US1] Add @mlflow.trace(span_type=AGENT, agent_role="planner") to planner
- [X] T064 [P] [US1] Implement Researcher agent (execute single plan step) in `backend/src/agent/nodes/researcher.py`
- [X] T065 [US1] Add @mlflow.trace(span_type=AGENT, agent_role="researcher") to researcher
- [X] T066 [P] [US1] Implement Reflector agent (CONTINUE/ADJUST/COMPLETE decisions) in `backend/src/agent/nodes/reflector.py`
- [X] T067 [US1] Add @mlflow.trace(span_type=AGENT, agent_role="reflector") to reflector
- [X] T068 [P] [US1] Implement Synthesizer agent (generate final report with citations) in `backend/src/agent/nodes/synthesizer.py`
- [X] T069 [US1] Add @mlflow.trace(span_type=AGENT, agent_role="synthesizer") to synthesizer

### Tool Tracing

- [X] T070 [US1] Add @mlflow.trace(span_type=TOOL, tool_name="brave_search") to web_search in `backend/src/agent/tools/web_search.py`
- [X] T071 [US1] Add @mlflow.trace(span_type=TOOL, tool_name="web_crawler") to web_crawler in `backend/src/agent/tools/web_crawler.py`

### Research Orchestrator

- [X] T072 [US1] Create run_research() async generator orchestrating all agents in `backend/src/agent/orchestrator.py`
- [X] T073 [US1] Add @mlflow.trace(span_type=AGENT, pipeline="deep_research") to run_research()
- [X] T074 [US1] Add research_loop span with span_type=CHAIN in orchestrator
- [X] T075 [US1] Implement plan iteration limit (max 3) enforcement in orchestrator
- [X] T076 [US1] Add session_id, user_id, query attributes to root span

### Prompt Templates

- [X] T077 [P] [US1] Create coordinator prompts (classify, simple_answer) in `backend/src/agent/prompts/coordinator.py`
- [X] T078 [P] [US1] Create planner prompts (create_plan, replan) in `backend/src/agent/prompts/planner.py`
- [X] T079 [P] [US1] Create researcher prompts (execute_step) in `backend/src/agent/prompts/researcher.py`
- [X] T080 [P] [US1] Create reflector prompts (evaluate_step) in `backend/src/agent/prompts/reflector.py`
- [X] T081 [P] [US1] Create synthesizer prompts (generate_report) in `backend/src/agent/prompts/synthesizer.py`

### Services

- [X] T082 [US1] Create ChatService with CRUD operations in `backend/src/services/chat_service.py`
- [X] T083 [US1] Create MessageService with create, list operations in `backend/src/services/message_service.py`
- [X] T084 [US1] Create ResearchSessionService for session management in `backend/src/services/research_session_service.py`
- [X] T085 [US1] Create SourceService for storing/retrieving sources in `backend/src/services/source_service.py`

### API Endpoints

- [X] T086 [US1] Implement POST /api/v1/chats endpoint in `backend/src/api/v1/chats.py`
- [X] T087 [US1] Implement GET /api/v1/chats/{chatId} endpoint in `backend/src/api/v1/chats.py`
- [X] T088 [US1] Implement POST /api/v1/chats/{chatId}/messages endpoint in `backend/src/api/v1/messages.py`
- [X] T089 [US1] Implement GET /api/v1/chats/{chatId}/stream SSE endpoint in `backend/src/api/v1/research.py`
- [X] T090 [US1] Implement POST /api/v1/agent/query endpoint (Databricks pattern) in `backend/src/api/v1/agent.py`

### Databricks Agent Server

- [X] T091 [US1] Create agent_server package init in `src/agent_server/__init__.py`
- [X] T092 [US1] Implement @invoke handler wrapping run_research() in `src/agent_server/agent.py`
- [X] T093 [US1] Implement @stream handler wrapping run_research() in `src/agent_server/agent.py`
- [X] T094 [US1] Implement event transformation (StreamEvent → ResponsesAgentStreamEvent) in `src/agent_server/utils.py`
- [X] T095 [P] [US1] Implement OBO token authentication helper in `src/agent_server/utils.py`
- [X] T096 [US1] Create AgentServer entry point in `src/agent_server/start_server.py`
- [X] T097 [P] [US1] Create app.yaml for Databricks Apps deployment in `app.yaml`

### LLM Service Tracing

- [X] T098 [US1] Add mlflow.start_span(span_type=CHAT_MODEL) to LLMClient.complete() in `src/services/llm/client.py`
- [X] T099 [US1] Add mlflow.start_span(span_type=CHAT_MODEL) to LLMClient.structured() (via complete()) in `src/services/llm/client.py`
- [X] T100 [US1] Add endpoint_id, model_role, message_count attributes to LLM spans

### Frontend - Chat Interface

- [X] T101 [US1] Create ChatPage layout with sidebar in `frontend/src/pages/ChatPage.tsx`
- [X] T102 [US1] Create ChatSidebar with chat list in `frontend/src/components/chat/ChatSidebar.tsx`
- [X] T103 [US1] Create MessageList component in `frontend/src/components/chat/MessageList.tsx`
- [X] T104 [US1] Create MessageInput component in `frontend/src/components/chat/MessageInput.tsx`
- [X] T105 [US1] Create UserMessage component in `frontend/src/components/chat/UserMessage.tsx`
- [X] T106 [US1] Create AgentMessage component with citations in `frontend/src/components/chat/AgentMessage.tsx`
- [X] T107 [US1] Create useStreamingQuery hook for SSE in `frontend/src/hooks/useStreamingQuery.ts`
- [X] T108 [US1] Streaming messages handled in ChatPage via useStreamingQuery hook

### Frontend - Research Progress

- [X] T109 [US1] ReasoningPanel functionality in ChatPage activity log
- [X] T110 [US1] Create PlanProgress (research plan visualization) in `frontend/src/components/research/PlanProgress.tsx`
- [X] T111 [US1] SourcesList integrated in AgentMessage citations
- [X] T112 [US1] Create AgentStatusIndicator component in `frontend/src/components/research/AgentStatusIndicator.tsx`

**Checkpoint**: User Story 1 complete - users can conduct deep research via chat

---

## Phase 4: User Story 2 - Intelligent Query Understanding (Priority: P1.5)

**Goal**: System intelligently classifies queries and adapts research depth

**Independent Test**: Submit queries of varying complexity and verify correct classification and appropriate research depth

### Query Classification

- [X] T113 [US2] Create QueryClassification Pydantic model in `src/agent/state.py`
- [X] T114 [US2] Enhance Coordinator with complexity estimation (simple/moderate/complex) in `src/agent/nodes/coordinator.py`
- [X] T115 [US2] Implement follow-up type detection (clarification/complex_follow_up/new_topic) in `src/agent/nodes/coordinator.py`
- [X] T116 [US2] Implement ambiguity detection and clarifying question generation in `src/agent/nodes/coordinator.py`

### Research Depth Control

- [X] T117 [US2] Implement auto research depth selection based on complexity in `src/agent/state.py`
- [X] T118 [US2] Add research_depth parameter to run_research() in `src/agent/orchestrator.py`
- [X] T119 [US2] Map depth levels to iteration limits (light=3, medium=6, extended=10) in `src/agent/state.py`

### Clarification Flow

- [X] T120 [US2] Implement clarification round tracking (max 3) in ResearchState in `src/agent/state.py`
- [ ] T121 [US2] Add clarification_needed StreamEvent type in `src/schemas/streaming.py`
- [X] T122 [US2] Handle clarification responses in Coordinator in `src/agent/nodes/coordinator.py`

### Simple Query Fast Path

- [X] T123 [US2] Implement direct response for simple queries (skip research) in `src/agent/nodes/coordinator.py`
- [ ] T124 [US2] Implement context-only responses for clarification follow-ups in `src/agent/nodes/coordinator.py`

### API Updates

- [X] T125 [US2] Add research_depth parameter to stream endpoint in `src/api/v1/research.py`
- [X] T126 [US2] Return query_classification in research state (captured in ResearchSession)

### Frontend - Depth Control

- [X] T127 [US2] Create ResearchDepthSelector component in `frontend/src/components/chat/ResearchDepthSelector.tsx`
- [X] T128 [US2] Add depth selector to MessageInput in `frontend/src/components/chat/MessageInput.tsx`
- [X] T129 [US2] Display query classification reasoning in UI in `frontend/src/components/chat/ClassificationBadge.tsx`
- [X] T130 [US2] Create ClarificationDialog for answering clarifying questions in `frontend/src/components/chat/ClarificationDialog.tsx`

**Checkpoint**: User Story 2 complete - system intelligently classifies queries

---

## Phase 5: User Story 3 - Multi-User Chat with Persistence (Priority: P2)

**Goal**: Multiple users have isolated, persistent chat histories

**Independent Test**: Create multiple users, have each create chats, verify isolation and persistence

### Chat Management

- [X] T131 [US3] Implement GET /api/v1/chats (list with pagination) in `src/api/v1/chats.py`
- [X] T132 [US3] Implement user isolation in ChatService queries in `src/services/chat_service.py`
- [X] T133 [US3] Implement chat title auto-generation from first message in `src/services/chat_service.py`
- [X] T134 [US3] Implement GET /api/v1/chats/{chatId}/messages (list with pagination) in `src/api/v1/messages.py`

### Soft Delete

- [X] T135 [US3] Implement DELETE /api/v1/chats/{chatId} (soft delete) in `src/api/v1/chats.py`
- [X] T136 [US3] Implement POST /api/v1/chats/{chatId}/restore in `src/api/v1/chats.py`
- [X] T137 [US3] Create background job for permanent purge after 30 days in `scripts/purge_deleted_chats.py`

### Frontend - Chat List

- [X] T138 [US3] Create useChatList hook with React Query in `frontend/src/hooks/useChats.ts`
- [X] T139 [US3] Create ChatListItem component in `frontend/src/components/chat/ChatSidebar.tsx`
- [X] T140 [US3] Implement chat switching in ChatSidebar in `frontend/src/components/chat/ChatSidebar.tsx`
- [X] T141 [US3] Create NewChatButton component in `frontend/src/components/chat/ChatSidebar.tsx`
- [X] T142 [US3] Add loading states and empty state for chat list in `frontend/src/components/chat/ChatSidebar.tsx`
- [X] T143 [US3] Implement delete confirmation dialog in `frontend/src/components/chat/DeleteChatDialog.tsx`

**Checkpoint**: User Story 3 complete - multi-user persistence works

---

## Phase 6: User Story 4 - Message Control & Refinement (Priority: P2.5)

**Goal**: Users can stop, edit, and regenerate research operations

**Independent Test**: Start research, stop it, edit a message, regenerate a response

### Cancel Operation

- [X] T144 [US4] Implement research cancellation signal in ResearchState in `src/agent/state.py`
- [X] T145 [US4] Add cancellation check points in orchestrator loop in `src/agent/orchestrator.py`
- [X] T146 [US4] Implement POST /api/v1/research/{sessionId}/cancel in `src/api/v1/research.py`
- [X] T147 [US4] Preserve partial results on cancellation in `src/services/research_session_service.py`

### Message Editing

- [X] T148 [US4] Implement PATCH /api/v1/chats/{chatId}/messages/{messageId} in `src/api/v1/messages.py`
- [X] T149 [US4] Implement cascade delete of subsequent messages on edit in `src/services/message_service.py`
- [X] T150 [US4] Mark edited messages with is_edited flag in `src/services/message_service.py`

### Regeneration

- [X] T151 [US4] Implement POST /api/v1/chats/{chatId}/messages/{messageId}/regenerate in `src/api/v1/messages.py`
- [X] T152 [US4] Create new research session for regeneration in `src/services/message_service.py`

### Frontend - Message Controls

- [X] T153 [US4] Create StopButton component in `frontend/src/components/chat/StopButton.tsx`
- [X] T154 [US4] Create MessageActions dropdown (edit, copy, regenerate) in `frontend/src/components/chat/MessageActions.tsx`
- [X] T155 [US4] Create EditMessageModal in `frontend/src/components/chat/EditMessageModal.tsx`
- [X] T156 [US4] Add regenerate button to AgentMessage in `frontend/src/components/chat/MessageActions.tsx`
- [X] T157 [US4] Handle message invalidation in UI state in `frontend/src/hooks/useChat.ts`

**Checkpoint**: User Story 4 complete - message control works

---

## Phase 7: User Story 5 - Resilient Model Access (Priority: P3)

**Goal**: System automatically fails over between model endpoints

**Independent Test**: Simulate endpoint failure, verify automatic failover without user intervention

### Endpoint Selection

- [ ] T158 [US5] Implement round-robin endpoint selection in ModelRouter in `backend/src/services/llm/router.py`
- [ ] T159 [US5] Implement priority-based endpoint selection in ModelRouter in `backend/src/services/llm/router.py`
- [ ] T160 [US5] Implement health-aware endpoint filtering in `backend/src/services/llm/router.py`

### Failover Logic

- [ ] T161 [US5] Implement automatic failover on 429/5xx errors in `backend/src/services/llm/client.py`
- [ ] T162 [US5] Implement exponential backoff with jitter in `backend/src/services/llm/client.py`
- [ ] T163 [US5] Track consecutive errors and mark endpoints unhealthy in `backend/src/services/llm/health.py`

### Context Truncation

- [ ] T164 [US5] Implement context truncation for smaller context windows in `backend/src/services/llm/truncation.py`
- [ ] T165 [US5] Preserve system prompt and recent messages during truncation in `backend/src/services/llm/truncation.py`

### Rate Limiting

- [ ] T166 [US5] Implement per-endpoint tokens_per_minute tracking in `backend/src/services/llm/rate_limiter.py`
- [ ] T167 [US5] Implement proactive rate limit checking before request in `backend/src/services/llm/router.py`

**Checkpoint**: User Story 5 complete - resilient model access works

---

## Phase 8: User Story 6 - Chat Organization & Discovery (Priority: P3.5)

**Goal**: Users can rename, search, archive, and export chats

**Independent Test**: Create chats, rename them, search for content, archive, export

### Chat Rename

- [X] T168 [US6] Implement PATCH /api/v1/chats/{chatId} (rename, archive) in `backend/src/api/v1/chats.py`

### Chat Search

- [ ] T169 [US6] Implement full-text search using PostgreSQL FTS in `backend/src/services/chat_service.py`
- [X] T170 [US6] Add search parameter to GET /api/v1/chats in `backend/src/api/v1/chats.py`
- [ ] T171 [US6] Return matching message snippets in search results in `backend/src/schemas/chat.py`

### Chat Archive

- [ ] T172 [US6] Implement archive/unarchive status transitions in `backend/src/services/chat_service.py`
- [X] T173 [US6] Add status filter (active/archived/all) to chat list in `backend/src/api/v1/chats.py`

### Chat Export

- [X] T174 [US6] Implement GET /api/v1/chats/{chatId}/export?format=markdown in `backend/src/api/v1/chats.py`
- [X] T175 [US6] Create Markdown export formatter with citations in `backend/src/services/export_service.py`
- [ ] T176 [P] [US6] Create PDF export using client-side rendering in `frontend/src/utils/exportPdf.ts`

### Frontend - Organization

- [ ] T177 [US6] Create ChatSearchInput component in `frontend/src/components/chat/ChatSearchInput.tsx`
- [ ] T178 [US6] Create RenameChatDialog component in `frontend/src/components/chat/RenameChatDialog.tsx`
- [ ] T179 [US6] Add archive/unarchive to chat context menu in `frontend/src/components/chat/ChatListItem.tsx`
- [ ] T180 [US6] Add status filter tabs (Active, Archived) in `frontend/src/components/chat/ChatSidebar.tsx`
- [ ] T181 [US6] Create ExportChatDialog component in `frontend/src/components/chat/ExportChatDialog.tsx`

**Checkpoint**: User Story 6 complete - chat organization works

---

## Phase 9: User Story 7 - View Agent Reasoning Process (Priority: P4)

**Goal**: Users can observe the agent's step-by-step reasoning process

**Independent Test**: Submit research query and verify reasoning steps are visible during and after research

### Reasoning Display

- [ ] T182 [US7] Store reasoning steps in ResearchSession in `backend/src/services/research_session_service.py`
- [ ] T183 [US7] Include reasoning_steps in GET /api/v1/chats/{chatId}/messages/{messageId} in `backend/src/api/v1/messages.py`

### Frontend - Reasoning UI

- [ ] T184 [US7] Enhance ReasoningPanel with step details in `frontend/src/components/research/ReasoningPanel.tsx`
- [ ] T185 [US7] Create ReasoningStep component (search, fetch, reflect icons) in `frontend/src/components/research/ReasoningStep.tsx`
- [ ] T186 [US7] Create expandable reasoning section in AgentMessage in `frontend/src/components/chat/AgentMessage.tsx`
- [ ] T187 [US7] Show real-time reasoning updates during streaming in `frontend/src/components/research/LiveReasoningPanel.tsx`
- [ ] T188 [US7] Display fetched URLs and search queries in reasoning in `frontend/src/components/research/ReasoningStep.tsx`

**Checkpoint**: User Story 7 complete - reasoning visibility works

---

## Phase 10: User Story 8 - REST API Access (Priority: P5)

**Goal**: External applications can use the research agent via REST API

**Independent Test**: Make API calls to submit queries and receive responses without UI

### API Polish

- [ ] T189 [US8] Add comprehensive OpenAPI documentation to all endpoints in `backend/src/api/v1/`
- [ ] T190 [US8] Implement API versioning middleware in `backend/src/middleware/versioning.py`
- [ ] T191 [US8] Add rate limiting per user for API endpoints in `backend/src/middleware/rate_limit.py`

### MLflow Evaluation

- [ ] T192 [P] [US8] Create MLflow evaluation setup with scorers in `backend/src/agent_server/evaluate.py`
- [ ] T193 [P] [US8] Create evaluation dataset for research quality in `backend/src/agent_server/eval_dataset.py`

**Checkpoint**: User Story 8 complete - REST API ready for external integration

---

## Phase 11: User Story 9 - E2E Testing with Playwright (Priority: P2)

**Goal**: Automated browser-based tests that validate complete user journeys

**Independent Test**: Run `npx playwright test` and verify all tests pass

### Setup & Infrastructure

- [X] T_E2E_01 Install Playwright and dependencies via `npm install -D @playwright/test` in frontend/package.json
- [X] T_E2E_02 Run `npx playwright install` to download browser binaries
- [X] T_E2E_03 Create Playwright configuration in frontend/e2e/playwright.config.ts
- [X] T_E2E_04 [P] [US9] Add npm scripts for e2e tests in frontend/package.json (`test:e2e`, `test:e2e:ui`, `test:e2e:debug`)
- [X] T_E2E_05 [P] [US9] Add Makefile targets for e2e tests (`make test-e2e`, `make test-e2e-ui`)
- [X] T_E2E_06 [US9] Create directory structure: frontend/e2e/{fixtures,pages,tests,utils}/

### Page Objects

- [X] T_E2E_07 [P] [US9] Create ChatPage class in frontend/e2e/pages/chat.page.ts with selectors and methods
- [X] T_E2E_08 [P] [US9] Create SidebarPage class in frontend/e2e/pages/sidebar.page.ts
- [X] T_E2E_09 [P] [US9] Create ResearchPage class in frontend/e2e/pages/research.page.ts for reasoning panel

### Fixtures & Utils

- [X] T_E2E_10 [US9] Create app fixture in frontend/e2e/fixtures/app.fixture.ts (base URL, auth setup)
- [X] T_E2E_11 [US9] Create chat fixture in frontend/e2e/fixtures/chat.fixture.ts (extends app fixture, provides chatPage)
- [X] T_E2E_12 [US9] Create combined test fixture in frontend/e2e/fixtures/index.ts (re-exports all fixtures)
- [X] T_E2E_13 [P] [US9] Create SSE wait helpers in frontend/e2e/utils/wait-helpers.ts
- [X] T_E2E_14 [P] [US9] Create test data generators in frontend/e2e/utils/test-data.ts

### Frontend data-testid Attributes

- [X] T_E2E_15 [P] [US9] Add data-testid="message-input" to message input in frontend/src/components/chat/MessageInput.tsx
- [X] T_E2E_16 [P] [US9] Add data-testid="send-button" to send button in frontend/src/components/chat/MessageInput.tsx
- [X] T_E2E_17 [P] [US9] Add data-testid="stop-button" to stop button in frontend/src/components/chat/MessageInput.tsx
- [X] T_E2E_18 [P] [US9] Add data-testid="loading-indicator" to loading indicator in frontend/src/components/chat/MessageList.tsx
- [X] T_E2E_19 [P] [US9] Add data-testid="streaming-indicator" to streaming indicator in frontend/src/components/chat/MessageList.tsx
- [X] T_E2E_20 [P] [US9] Add data-testid="message-list" to message list container in frontend/src/components/chat/MessageList.tsx
- [X] T_E2E_21 [P] [US9] Add data-testid="user-message" to user message components in frontend/src/components/chat/UserMessage.tsx
- [X] T_E2E_22 [P] [US9] Add data-testid="agent-response" to agent response components in frontend/src/components/chat/AgentMessage.tsx
- [X] T_E2E_23 [P] [US9] Add data-testid="citation" to citation links in frontend/src/components/chat/AgentMessage.tsx
- [X] T_E2E_24 [P] [US9] Add data-testid="reasoning-panel" to reasoning panel in frontend/src/components/research/PlanProgress.tsx
- [X] T_E2E_25 [P] [US9] Add data-testid="regenerate-response" to regenerate button in frontend/src/components/chat/AgentMessage.tsx
- [X] T_E2E_26 [P] [US9] Add data-testid="new-chat-button" to new chat button in frontend/src/components/sidebar/ChatSidebar.tsx

### Test Scenarios (US9.1-US9.6)

- [X] T_E2E_27 [US9.6] Create smoke.spec.ts in frontend/e2e/tests/smoke.spec.ts
- [X] T_E2E_28 [US9.6] Implement "app loads successfully" test - verify title and message input visible
- [X] T_E2E_29 [US9.6] Implement "can create new chat" test - click new chat, verify input empty
- [X] T_E2E_30 [US9.6] Implement "can send simple message" test - send "Hello", verify response within 15s
- [X] T_E2E_31 [US9.1] Create research-flow.spec.ts in frontend/e2e/tests/research-flow.spec.ts
- [X] T_E2E_32 [US9.1] Implement "submits query and receives response" test per e2e-test-patterns.md Scenario 1
- [X] T_E2E_33 [US9.1] Implement streaming indicator verification (visible during research)
- [X] T_E2E_34 [US9.1] Implement citation count verification (minimum 1 citation)
- [X] T_E2E_35 [US9.2] Create follow-up.spec.ts in frontend/e2e/tests/follow-up.spec.ts
- [X] T_E2E_36 [US9.2] Implement follow-up question test with context verification
- [X] T_E2E_37 [US9.3] Create stop-cancel.spec.ts in frontend/e2e/tests/stop-cancel.spec.ts
- [X] T_E2E_38 [US9.3] Implement "stops operation within 2 seconds" test
- [X] T_E2E_39 [US9.4] Create edit-message.spec.ts in frontend/e2e/tests/edit-message.spec.ts
- [X] T_E2E_40 [US9.4] Add editMessage method to ChatPage in frontend/e2e/pages/chat.page.ts
- [X] T_E2E_41 [US9.4] Implement "invalidates subsequent messages on edit" test
- [X] T_E2E_42 [US9.5] Create regenerate.spec.ts in frontend/e2e/tests/regenerate.spec.ts
- [X] T_E2E_43 [US9.5] Add regenerateButton locator to ChatPage in frontend/e2e/pages/chat.page.ts
- [X] T_E2E_44 [US9.5] Implement "generates new response for same query" test

### CI Integration

- [X] T_E2E_45 [P] [US9] Create GitHub Actions workflow in .github/workflows/e2e.yml
- [X] T_E2E_46 [P] [US9] Configure artifact upload for playwright-report/ in CI workflow
- [X] T_E2E_47 [P] [US9] Add error handling tests in frontend/e2e/tests/error-handling.spec.ts
- [X] T_E2E_48 [US9] Update quickstart.md with e2e test instructions
- [X] T_E2E_49 [US9] Add E2E_BASE_URL environment variable documentation in .env.example

**Checkpoint**: User Story 9 complete - automated e2e tests validate all core user journeys

---

## Phase 12: Cross-Cutting Concerns

**Purpose**: Features that span multiple user stories

### Feedback System (FR-033 to FR-035)

- [ ] T194 Implement POST /api/v1/chats/{chatId}/messages/{messageId}/feedback in `backend/src/api/v1/messages.py`
- [X] T195 Create FeedbackService with MLflow trace logging in `backend/src/services/feedback_service.py`
- [ ] T196 Create FeedbackButtons component (thumbs up/down) in `frontend/src/components/chat/FeedbackButtons.tsx`
- [ ] T197 [P] Create ErrorReportDialog for detailed feedback in `frontend/src/components/chat/ErrorReportDialog.tsx`

### User Preferences (FR-040)

- [X] T198 Implement GET /api/v1/preferences in `backend/src/api/v1/preferences.py`
- [X] T199 Implement PUT /api/v1/preferences in `backend/src/api/v1/preferences.py`
- [X] T200 Create PreferencesService in `backend/src/services/preferences_service.py`
- [X] T201 Apply system_instructions to all research prompts in `backend/src/agent/orchestrator.py`
- [ ] T202 Create SettingsPage in `frontend/src/pages/SettingsPage.tsx`
- [ ] T203 Create SystemInstructionsEditor component in `frontend/src/components/settings/SystemInstructionsEditor.tsx`

### Keyboard Shortcuts (FR-042)

- [ ] T204 Create useKeyboardShortcuts hook in `frontend/src/hooks/useKeyboardShortcuts.ts`
- [ ] T205 Implement Cmd/Ctrl+N for new chat, Cmd/Ctrl+Enter for send in `frontend/src/hooks/useKeyboardShortcuts.ts`
- [ ] T206 [P] Create KeyboardShortcutsHelp component in `frontend/src/components/common/KeyboardShortcutsHelp.tsx`

### Clipboard (FR-041)

- [ ] T207 Create CopyButton component with clipboard API in `frontend/src/components/chat/CopyButton.tsx`

### Accessibility (FR-044)

- [ ] T209 Add ARIA labels and roles to all interactive components in `frontend/src/components/`
- [ ] T210 [P] Ensure sufficient color contrast (WCAG AA) in `frontend/src/styles/`
- [ ] T211 [P] Add skip-to-content link in `frontend/src/components/layout/Layout.tsx`
- [ ] T212 [P] Test keyboard navigation flow across all pages

### Audit Logging (FR-043)

- [ ] T213 Implement audit log creation in all CRUD operations in `backend/src/services/`
- [ ] T214 [P] Add IP address and user agent capture to audit logs in `backend/src/middleware/audit.py`

---

## Phase 12: Observability & Polish

**Purpose**: Final validation, performance, error handling

### Error Handling & Edge Cases

- [X] T215 Handle Brave Search API rate limits with queue and backoff in `backend/src/services/search/brave.py`
- [X] T216 Handle web page fetch failures (403, 404, timeout) gracefully in `backend/src/agent/tools/web_crawler.py`
- [X] T217 Handle LLM unexpectedly long responses with truncation in `backend/src/services/llm/client.py`
- [X] T218 Handle empty plans (go directly to Synthesizer) in `backend/src/agent/orchestrator.py`

### Performance

- [ ] T219 [P] Ensure <200ms p95 for non-LLM operations (profile and optimize) in `backend/src/`
- [ ] T220 [P] Ensure <3s chat history load time in `frontend/src/pages/ChatPage.tsx`

### Documentation

- [ ] T221 [P] Create quickstart.md with setup instructions in `docs/quickstart.md`
- [ ] T222 [P] Add inline API documentation (docstrings) in `backend/src/api/v1/`

### Final Validation

- [X] T223 Run mypy --strict on backend, fix any errors in `backend/src/`
- [X] T224 Run ruff check and format on backend in `backend/src/`
- [ ] T225 Run TypeScript typecheck on frontend in `frontend/src/`
- [ ] T226 Run ESLint on frontend in `frontend/src/`
- [ ] T227 [P] Run axe-core accessibility audit, fix critical violations in `frontend/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-10)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P1.5 → P2 → ...)
- **Cross-Cutting (Phase 11)**: Can start after US1 is complete
- **Polish (Phase 12)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational - **No dependencies on other stories**
- **US2 (P1.5)**: Can start after Foundational - Enhances US1 but independently testable
- **US3 (P2)**: Can start after Foundational - No dependencies
- **US4 (P2.5)**: Can start after US1 - Depends on research flow existing
- **US5 (P3)**: Can start after Foundational - LLM service enhancement
- **US6 (P3.5)**: Can start after US3 - Depends on chat list existing
- **US7 (P4)**: Can start after US1 - Depends on research flow existing
- **US8 (P5)**: Can start after US1 - API polish

### Within Each User Story

- Models before services
- Services before endpoints
- Backend before frontend
- Core implementation before integration

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel
- Once Foundational phase completes, all user stories can start in parallel
- Within each story, tasks marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch agent node implementations in parallel (different files):
Task: "Implement Coordinator agent in backend/src/agent/nodes/coordinator.py"
Task: "Implement Planner agent in backend/src/agent/nodes/planner.py"
Task: "Implement Researcher agent in backend/src/agent/nodes/researcher.py"
Task: "Implement Reflector agent in backend/src/agent/nodes/reflector.py"
Task: "Implement Synthesizer agent in backend/src/agent/nodes/synthesizer.py"

# Launch prompt templates in parallel (different files):
Task: "Create coordinator prompts in backend/src/agent/prompts/coordinator.py"
Task: "Create planner prompts in backend/src/agent/prompts/planner.py"
Task: "Create researcher prompts in backend/src/agent/prompts/researcher.py"
Task: "Create reflector prompts in backend/src/agent/prompts/reflector.py"
Task: "Create synthesizer prompts in backend/src/agent/prompts/synthesizer.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test deep research via chat end-to-end
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 (Deep Research) → Test → Deploy/Demo (MVP!)
3. Add US2 (Query Intelligence) → Test → Enhances MVP
4. Add US3 (Persistence) → Test → Multi-user ready
5. Add US4-US8 → Test each → Full feature set
6. Polish phase → Production ready

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (backend)
   - Developer B: User Story 1 (frontend)
   - Developer C: User Story 3 (persistence) + US5 (model routing)
3. Stories integrate and deliver independently

---

## Task Summary

| Phase | Task Count | Story |
|-------|------------|-------|
| Phase 1: Setup | 10 | - |
| Phase 2: Foundational | 43 | - |
| Phase 2.5: Frontend-Backend Consolidation | 8 | - |
| Phase 3: US1 (Deep Research) | 59 | P1 |
| Phase 4: US2 (Query Intelligence) | 18 | P1.5 |
| Phase 5: US3 (Multi-User) | 13 | P2 |
| Phase 6: US4 (Message Control) | 14 | P2.5 |
| Phase 7: US5 (Model Failover) | 10 | P3 |
| Phase 8: US6 (Organization) | 14 | P3.5 |
| Phase 9: US7 (Reasoning) | 7 | P4 |
| Phase 10: US8 (REST API) | 5 | P5 |
| Phase 11: US9 (E2E Testing) | 49 | P2 |
| Phase 12: Cross-Cutting | 21 | - |
| Phase 13: Observability & Polish | 13 | - |
| Phase 14: Central YAML Config (FR-081-090) | 33 | INFRA |
| **TOTAL** | **317** | |

### MVP Scope (Recommended)

For minimal viable product, complete:
- Phase 1: Setup (10 tasks)
- Phase 2: Foundational (43 tasks)
- Phase 2.5: Frontend-Backend Consolidation (8 tasks) ✅ Complete
- Phase 3: US1 - Deep Research (59 tasks)

**MVP Total: 120 tasks**

This delivers core deep research via chat with unified deployment - the essential value proposition.

---

## Phase 14: Central YAML Configuration (FR-081 to FR-090)

**Purpose**: Consolidate all model, agent, and search configuration into a central YAML file

**Goal**: Replace scattered hardcoded values with centralized, validated YAML configuration

**Added**: 2025-12-24

### Setup Tasks

- [X] T285 Create config directory at repository root `config/`
- [X] T286 [P] Create YAML loader with env var interpolation in `src/core/yaml_loader.py`
- [X] T287 [P] Create Pydantic configuration models in `src/core/app_config.py`

### Configuration Files

- [X] T288 [P] Create default configuration file `config/app.yaml` with model endpoints and roles
- [X] T289 [P] Create documented example configuration `config/app.example.yaml`

### LLM Service Integration

- [X] T290 [P] [INFRA] Update `src/services/llm/config.py` to load ModelConfig from AppConfig
- [X] T291 [INFRA] Update `src/services/llm/client.py` to use centralized config
- [X] T292 [INFRA] Verify existing LLM tests pass with new config

### Agent Node Integration

- [X] T293 [P] [INFRA] Create agent config accessors in `src/agent/config.py`
- [X] T294 [P] [INFRA] Update `src/agent/nodes/researcher.py` to use config accessors for limits
- [X] T295 [P] [INFRA] Update `src/agent/nodes/planner.py` to use config accessors for max_plan_iterations
- [X] T296 [P] [INFRA] Update `src/agent/nodes/coordinator.py` to use config accessors for max_clarification_rounds
- [X] T297 [P] [INFRA] Update `src/agent/nodes/synthesizer.py` to use config accessors
- [X] T298 [INFRA] Verify existing agent tests pass with new config

### Search Service Integration

- [X] T299 [INFRA] Update `src/services/search/brave.py` to use config for rate limiting and freshness
- [X] T300 [INFRA] Verify existing search tests pass with new config

### Startup Validation

- [X] T301 [INFRA] Add startup event to validate config in `src/main.py`
- [X] T302 [INFRA] Update `src/core/__init__.py` to export config functions (get_app_config, etc.)
- [X] T303 [INFRA] Test startup behavior with missing config file (should use defaults)
- [X] T304 [INFRA] Test startup behavior with invalid config file (should fail fast with clear error)

### Unit Tests

- [X] T305 [P] [INFRA] Create env var interpolation tests in `tests/unit/core/test_yaml_loader.py`
- [X] T306 [P] [INFRA] Create config loading and default fallback tests in `tests/unit/core/test_app_config.py`
- [X] T307 [P] [INFRA] Create endpoint reference validation tests in `tests/unit/core/test_app_config.py`
- [X] T308 [P] [INFRA] Create agent config accessor tests in `tests/unit/agent/test_config.py`
- [X] T309 [INFRA] Run full test suite to verify no regressions (112+ tests)

### Documentation

- [X] T310 [P] [INFRA] Add configuration section to `CLAUDE.md` with YAML examples
- [X] T311 [P] [INFRA] Add configuration quick start to `README.md`
- [X] T312 [INFRA] Verify `specs/001-deep-research-agent/data-model.md` has AppConfig entity

### Verification

- [X] T313 Run `uv run pytest tests/unit/ -v` to verify all unit tests pass
- [X] T314 Run `uv run mypy src --strict` to verify type checking passes
- [X] T315 Run `uv run ruff check src` to verify linting passes
- [X] T316 Verify SC-029: System starts with defaults when config file absent
- [X] T317 Verify SC-030: Config errors reported within 5 seconds of startup

**Checkpoint**: All model, agent, and search configuration loaded from central YAML file

---

## Phase 14 Parallel Opportunities

```bash
# Launch configuration infrastructure in parallel:
Task T286: "Create YAML loader in src/core/yaml_loader.py"
Task T287: "Create Pydantic models in src/core/app_config.py"

# Launch config files in parallel:
Task T288: "Create config/app.yaml"
Task T289: "Create config/app.example.yaml"

# Launch agent node updates in parallel (after T293):
Task T294: "Update researcher.py"
Task T295: "Update planner.py"
Task T296: "Update coordinator.py"
Task T297: "Update synthesizer.py"

# Launch unit tests in parallel:
Task T305: "Create tests/unit/core/test_yaml_loader.py"
Task T306: "Create tests/unit/core/test_app_config.py"
Task T308: "Create tests/unit/agent/test_config.py"
```

---

## Notes

- [P] tasks = different files, no dependencies
- [INFRA] label = infrastructure task supporting all user stories
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
