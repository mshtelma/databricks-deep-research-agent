# Tasks: Custom Agent Configuration & Selection

**Input**: Design documents from `/specs/009-custom-agent-config/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/api.md, quickstart.md

**Tests**: Included. The spec explicitly references testable acceptance scenarios and the plan includes test file paths.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `src/deep_research/` at repository root
- **Frontend**: `frontend/src/`
- **Tests**: `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Database migration and shared schema/model extensions needed by multiple user stories

- [x] T001 Create database migration adding `model_overrides` (JSONB, nullable), `domain_filter_mode` (VARCHAR(20), nullable), `include_domains` (JSONB, nullable), `exclude_domains` (JSONB, nullable) columns to `custom_agents` table in `src/deep_research/db/migrations/versions/017_custom_agent_model_overrides.py`
- [x] T002 Add four new columns (`model_overrides: Mapped[dict[str, str] | None]`, `domain_filter_mode: Mapped[str | None]`, `include_domains: Mapped[list[str] | None]`, `exclude_domains: Mapped[list[str] | None]`) to `CustomAgent` SQLAlchemy model in `src/deep_research/models/custom_agent.py`, following the same JSONB pattern as `enabled_sources`/`disabled_sources`
- [x] T003 Add `model_overrides`, `domain_filter_mode`, `include_domains`, `exclude_domains` fields to `CreateCustomAgentRequest`, `UpdateCustomAgentRequest`, `CustomAgentResponse` in `src/deep_research/schemas/custom_agent.py`. Add `has_model_overrides: bool`, `has_domain_filter: bool`, `has_source_config: bool` computed fields to `CustomAgentSummary`. Import `DomainFilterMode` from `core/app_config.py` for validation.
- [x] T004 [P] Add `model_overrides: dict[str, str] | None = None` and `domain_filter: DomainFilterConfig | None = None` fields to `OrchestrationConfig` dataclass in `src/deep_research/agent/orchestrator.py` (after the existing `agent_id` field, ~line 272)
- [x] T005 [P] Add `model_overrides: dict[str, str] | None = None` field to `ResearchState` dataclass in `src/deep_research/agent/state.py`

**Checkpoint**: Database schema updated, all model/schema/config dataclass extensions in place. No behavioral changes yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core backend wiring that MUST be complete before any user story's backend can function

**CRITICAL**: This phase activates the entire agent configuration pipeline. Before this phase, `apply_custom_agent_to_config()` is never called and no agent settings are applied.

- [x] T006 Wire `apply_custom_agent_to_config()` call in `_run_job()` in `src/deep_research/services/job_manager.py`: After constructing `OrchestrationConfig` (~line 540) and before calling `stream_research()`, add a block that: (1) if `agent_id` is not None, fetches the `CustomAgent` from DB using `CustomAgentService.get_accessible(agent_id, user_id)`, (2) calls `apply_custom_agent_to_config(config, agent)`, (3) if agent has `system_prompt_template` relationship loaded with content, sets `config.system_instructions = agent.system_prompt_template.content`. Use an independent DB session (`get_session_maker()`) to avoid lifecycle issues. Log at INFO level with key `JOB_AGENT_CONFIG_APPLIED`.
- [x] T007 Update CRUD handlers in `src/deep_research/api/v1/custom_agents.py` to pass through the four new fields: In `create_custom_agent()`, pass `model_overrides`, `domain_filter_mode`, `include_domains`, `exclude_domains` to `service.create_agent()`. In `update_custom_agent()`, add `if request_body.X is not None: agent.X = request_body.X` for each new field. In `_agent_to_response()`, include the four new fields. In `_agent_to_summary()`, add computed `has_model_overrides`, `has_domain_filter`, `has_source_config` booleans.
- [x] T008 Extend `CustomAgentService.create_agent()` in `src/deep_research/services/custom_agent_service.py` to accept and persist the four new keyword arguments: `model_overrides`, `domain_filter_mode`, `include_domains`, `exclude_domains`.
- [x] T009 Wire `model_overrides` from `OrchestrationConfig` to `ResearchState` in `stream_research()` in `src/deep_research/agent/orchestrator.py`: After the existing `state.agent_id = config.agent_id` line (~1401), add `if config.model_overrides: state.model_overrides = config.model_overrides`.
- [x] T010 Write unit tests for `apply_custom_agent_to_config()` in `tests/unit/agent/test_agent_config_apply.py`: Test that (1) source_scope from agent overrides config, (2) model_overrides from agent are set on config, (3) domain_filter from agent columns constructs DomainFilterConfig on config, (4) system_instructions from template relationship are set on config, (5) query_overrides take precedence over agent, (6) null/empty agent fields leave config defaults unchanged, (7) preset steps are converted in manual/hybrid mode.

**Checkpoint**: When a user selects an agent and submits a query, the backend now fetches the agent from DB and applies its existing config fields (source_scope, workflow_mode, depth, etc.) to the research pipeline. Model overrides and domain filters are threaded through but not yet consumed by LLM client or search client.

---

## Phase 3: User Story 1 — Select a Custom Agent from the Main Chat Screen (Priority: P1)

**Goal**: Users can select a custom agent from the main chat screen dropdown and have its configuration applied to research queries. The agent selector persists across sessions.

**Independent Test**: Create a custom agent with `source_scope=enterprise_only` via API. Select it from the main screen dropdown. Submit a query. Verify in activity log that no web searches occurred. Switch to "Default" and verify web searches resume.

### Tests for User Story 1

- [x] T011 [P] [US1] Write unit test in `tests/unit/services/test_job_manager_agent.py` verifying that `_run_job()` calls `apply_custom_agent_to_config()` when `agent_id` is provided, and does not call it when `agent_id` is None. Mock `CustomAgentService` and `stream_research`. Verify config fields are set.
- [x] T012 [P] [US1] Write unit test in `tests/unit/agent/test_agent_config_apply.py` (extend T010) verifying the full agent → config → state wiring: create a mock agent with source_scope="enterprise_only", call `apply_custom_agent_to_config`, verify config.source_scope is set, then verify `stream_research()` wires it to `state.source_scope_config`.

### Implementation for User Story 1

- [x] T013 [US1] Verify agent selection persistence in frontend `frontend/src/components/chat/MessageInput.tsx`: Confirm the existing agent picker dropdown reads from and writes to `localStorage` key `deep-research-selected-agent`. If not already persisted (FR-005), add `useEffect` to save selected agentId to localStorage on change and read on mount. Ensure "Default" option clears the key.
- [x] T014 [US1] Verify agent selector groups agents by ownership in `frontend/src/components/chat/MessageInput.tsx`: Confirm the existing dropdown groups user agents separately from workspace agents (FR-002). If not, add group headers "My Agents" and "Workspace" to the dropdown options using the `owner_id` field from `CustomAgentSummary`.
- [x] T015 [US1] Add `model_override_warnings` computed field to `CustomAgentResponse` in `src/deep_research/api/v1/custom_agents.py`: In `_agent_to_response()`, after building the base response, iterate `agent.model_overrides` (if non-null), check each endpoint name against `get_app_config().endpoints`, and build a warnings list for missing endpoints. Add `model_override_warnings: list[dict[str, str]]` to `CustomAgentResponse` schema in `src/deep_research/schemas/custom_agent.py`.

**Checkpoint**: User Story 1 is complete. Users can select agents, agent config is applied to queries, selection persists across sessions, and stale endpoint warnings are shown.

---

## Phase 4: User Story 2 — Configure Per-Agent Model Overrides (Priority: P2)

**Goal**: Users can configure per-tier model endpoint overrides in the agent editor. A new backend API serves the live endpoint catalog. The LLM client respects model overrides at query time.

**Independent Test**: Create an agent with `model_overrides: {"complex": "databricks-haiku"}` via API. Select it, run a research query. Check backend logs for `AGENT_MODEL_OVERRIDE_APPLIED` confirming the complex tier used `databricks-haiku`.

### Tests for User Story 2

- [x] T016 [P] [US2] Write unit test in `tests/unit/api/test_config_endpoint.py` verifying `GET /api/v1/config/model-catalog` returns `categories` and `endpoints` dicts matching the loaded `AppConfig`. Mock `get_app_config()` with 2 tiers and 3 endpoints. Verify response schema matches `EndpointCatalogResponse`.
- [x] T017 [P] [US2] Write unit test in `tests/unit/agent/test_agent_config_apply.py` (extend) verifying that `apply_custom_agent_to_config()` with `agent.model_overrides = {"complex": "ep-opus"}` sets `config.model_overrides = {"complex": "ep-opus"}`, and that an override referencing a non-existent endpoint is dropped with a log warning.

### Implementation for User Story 2

- [x] T018 [P] [US2] Create Pydantic response schemas `EndpointInfo`, `ModelCategoryInfo`, `EndpointCatalogResponse` in `src/deep_research/schemas/config.py`. `EndpointInfo`: name (str), endpoint_identifier (str), max_context_window (int), supports_structured_output (bool). `ModelCategoryInfo`: name (str), default_endpoints (list[str]), temperature (float), max_tokens (int). `EndpointCatalogResponse`: categories (dict[str, ModelCategoryInfo]), endpoints (dict[str, EndpointInfo]).
- [x] T019 [P] [US2] Create `GET /api/v1/config/model-catalog` endpoint in `src/deep_research/api/v1/config.py`: New `APIRouter(prefix="/config", tags=["Config"])`. Single endpoint that reads `get_app_config().models` and `get_app_config().endpoints`, transforms them into `EndpointCatalogResponse`. Return 200.
- [x] T020 [US2] Register config router in `src/deep_research/api/v1/__init__.py`: Import `config` module, add `router.include_router(config.router, tags=["Config"])` after the existing router registrations.
- [x] T021 [US2] Extend `apply_custom_agent_to_config()` in `src/deep_research/agent/orchestrator.py` to handle model overrides: After existing output_schema handling (~line 457), add a block: if `agent.model_overrides` is a non-empty dict, validate each entry against `get_app_config().endpoints` (skip entries with missing endpoints, log warning with key `AGENT_MODEL_OVERRIDE_SKIPPED`), then set `config.model_overrides = validated_overrides`. Log `AGENT_MODEL_OVERRIDE_APPLIED` with override count.
- [x] T022 [US2] Add `endpoint_override: str | None = None` parameter to `LLMClient.complete()` and `LLMClient.stream()` in `src/deep_research/services/llm/client.py`: In `_complete_impl()`, before calling `self._select_endpoint(role, estimated_tokens)`, check if `endpoint_override` is set. If so, look up the endpoint in `self._config.endpoints[endpoint_override]` and use it directly (skip rotation). If the override endpoint is not found, log warning and fall back to normal selection. Apply the same pattern to `_stream_impl()`.
- [x] T023 [US2] Thread `model_overrides` from `ResearchState` to LLM calls in agent nodes: In `src/deep_research/agent/nodes/planner.py`, `researcher.py`, `react_researcher.py`, `reflector.py`, `synthesizer.py`, `citation_synthesizer.py`: where `llm.complete(tier=ModelTier.X)` or `llm.stream(tier=ModelTier.X)` is called, pass `endpoint_override=state.model_overrides.get(tier.value) if state.model_overrides else None`. This can be done by adding a helper function `_get_endpoint_override(state: ResearchState, tier: ModelTier) -> str | None` in `src/deep_research/agent/orchestrator.py` and importing it in each node.
- [x] T024 [P] [US2] Create frontend API client for endpoint catalog in `frontend/src/api/config.ts`: Export `async function getModelCatalog(): Promise<EndpointCatalogResponse>` that fetches `GET /api/v1/config/model-catalog`. Define `EndpointCatalogResponse`, `EndpointInfo`, `ModelCategoryInfo` TypeScript interfaces.
- [x] T025 [P] [US2] Create `useModelCatalog` TanStack Query hook in `frontend/src/hooks/useModelCatalog.ts`: Uses `useQuery` with key `["model-catalog"]` and `queryFn: getModelCatalog`. Exports `categories`, `endpoints`, `isLoading`, `error`. Cache for 5 minutes (staleTime).
- [x] T026 [US2] Create `ModelConfigSection` component in `frontend/src/components/agents/ModelConfigSection.tsx`: Accepts `modelOverrides: Record<string, string> | null`, `onChange: (overrides: Record<string, string> | null) => void`. Uses `useModelCatalog` to populate dropdowns. Renders one row per model category (from `categories`), each with a `<select>` dropdown listing "Use Default" + all `endpoints` by name. On change, updates the overrides dict (removing entries set to "Use Default"). Shows warning icons for overrides referencing unknown endpoints.
- [x] T027 [US2] Integrate `ModelConfigSection` into the existing agent editor form: Find the agent create/edit form component in `frontend/src/components/agents/` (or `frontend/src/pages/AgentsPage.tsx`), add `<ModelConfigSection>` after the existing fields. Wire its `modelOverrides` state to the agent form data. Include `model_overrides` in the create/update API request body.
- [x] T028 [US2] Add `model_overrides` field to frontend `CustomAgent` and `CreateCustomAgentRequest` types in `frontend/src/types/customAgents.ts`: `modelOverrides?: Record<string, string> | null`.

**Checkpoint**: User Story 2 is complete. Users can override model tiers per agent via the editor UI. Backend validates overrides and threads them to LLM calls. Stale endpoint warnings display.

---

## Phase 5: User Story 3 — Define Enterprise and Web Source Scope per Agent (Priority: P3)

**Goal**: Users can configure source scope (enterprise_only, web_only, all) and toggle individual enterprise sources in the agent editor. When the agent is selected, the per-query source scope selector is hidden.

**Independent Test**: Create agent with `source_scope=enterprise_only`, `enabled_sources=["Product Knowledge"]`. Select it, run a query. Verify no web searches in activity log and only "Product Knowledge" enterprise source is queried. Verify the per-query source scope selector is hidden in the UI.

### Implementation for User Story 3

- [x] T029 [US3] Verify that `apply_custom_agent_to_config()` correctly wires `source_scope`, `enabled_sources`, `disabled_sources` from agent to config in `src/deep_research/agent/orchestrator.py`. This code already exists at lines 422-438 — write a unit test in `tests/unit/agent/test_agent_config_apply.py` confirming it: create agent with `source_scope="enterprise_only"`, `enabled_sources=["vs_1"]`, call apply, verify `config.source_scope == "enterprise_only"` and `config.enabled_sources == ["vs_1"]`.
- [x] T030 [US3] Hide per-query source scope selector when agent defines sources (FR-015a): In `frontend/src/components/chat/MessageInput.tsx`, check if the selected agent has `source_scope != "all"` or has non-null `enabled_sources`. If so, hide the `SourceScopeSelector` component (pass `hidden={true}` or conditionally render). The agent data is already available from `useCustomAgents` hook. When "Default" is selected (no agent), always show the selector.
- [x] T031 [P] [US3] Add `hidden` or `disabled` prop to `frontend/src/components/research/SourceScopeSelector.tsx` (or `frontend/src/components/chat/SourceScopeSelector.tsx` — find the correct file): When `hidden` is true, render nothing (return null). This provides clean conditional rendering from parent.

**Checkpoint**: User Story 3 is complete. Agent-level source scope is enforced. Per-query selector hides when agent defines sources.

---

## Phase 6: User Story 4 — Define Web Domain Whitelist/Blacklist per Agent (Priority: P4)

**Goal**: Users can configure domain filtering (include/exclude/both) with wildcard patterns in the agent editor. At query time, agent domain filters override system-wide YAML filters.

**Independent Test**: Create agent with `domain_filter_mode="include"`, `include_domains=["*.gov", "*.edu"]`. Select it, run a web research query. Verify in backend logs that search results are filtered to only .gov and .edu domains.

### Tests for User Story 4

- [x] T032 [P] [US4] Write unit test in `tests/unit/agent/test_agent_config_apply.py` verifying domain filter wiring: create agent with `domain_filter_mode="include"`, `include_domains=["*.gov"]`, call `apply_custom_agent_to_config`, verify `config.domain_filter` is a `DomainFilterConfig` with `mode=DomainFilterMode.INCLUDE` and `include_domains=["*.gov"]`. Also test that null `domain_filter_mode` leaves `config.domain_filter` as None.

### Implementation for User Story 4

- [x] T033 [US4] Extend `apply_custom_agent_to_config()` in `src/deep_research/agent/orchestrator.py` to construct `DomainFilterConfig` from agent columns: After model overrides handling, add a block: if `agent.domain_filter_mode` is not None, construct `DomainFilterConfig(mode=DomainFilterMode(agent.domain_filter_mode), include_domains=agent.include_domains or [], exclude_domains=agent.exclude_domains or [])` and set `config.domain_filter = domain_filter`. Log with key `AGENT_DOMAIN_FILTER_APPLIED`.
- [x] T034 [US4] Wire `config.domain_filter` to search client in `stream_research()` in `src/deep_research/agent/orchestrator.py`: After state initialization, if `config.domain_filter` is not None, store it on state as `state.domain_filter = config.domain_filter`. Then in the researcher nodes where `BraveSearchClient` is called, pass the override filter. Alternatively, wire through `ResearchState.domain_filter` to the web_search tool.
- [x] T035 [US4] Add `domain_filter: DomainFilterConfig | None = None` field to `ResearchState` in `src/deep_research/agent/state.py`. In the web_search tool (`src/deep_research/agent/tools/web_search.py`) and in `background.py` where `brave_client.search()` is called, check `state.domain_filter` and if non-null, use it instead of the system-wide `get_app_config().search.domain_filter`. This requires the search methods to accept an optional `domain_filter` parameter.
- [x] T036 [US4] Add domain filter validation to `CreateCustomAgentRequest` and `UpdateCustomAgentRequest` in `src/deep_research/schemas/custom_agent.py`: Add a `@model_validator(mode="after")` that checks: if `domain_filter_mode` is "include" or "both", `include_domains` must be non-empty; if "exclude" or "both", `exclude_domains` must be non-empty. Domain patterns must match regex `^[a-zA-Z0-9.*-]+$`.
- [x] T037 [P] [US4] Create `DomainFilterSection` component in `frontend/src/components/agents/DomainFilterSection.tsx`: Accepts `domainFilterMode: string | null`, `includeDomains: string[] | null`, `excludeDomains: string[] | null`, `onChange: (mode, include, exclude) => void`. Renders: mode selector (None, Include, Exclude, Both), textarea for include patterns (one per line), textarea for exclude patterns (one per line). Shows validation errors for invalid patterns. Only show relevant textarea based on mode.
- [x] T038 [US4] Integrate `DomainFilterSection` into agent editor: Add `<DomainFilterSection>` below `ModelConfigSection` in the agent create/edit form. Wire state. Include `domain_filter_mode`, `include_domains`, `exclude_domains` in API requests.
- [x] T039 [US4] Add `domainFilterMode`, `includeDomains`, `excludeDomains` fields to frontend `CustomAgent` and `CreateCustomAgentRequest` types in `frontend/src/types/customAgents.ts`.

**Checkpoint**: User Story 4 is complete. Per-agent domain filters are configurable and enforced at search time.

---

## Phase 7: User Story 5 — Per-Step Source Configuration (Priority: P5)

**Goal**: Preset steps can have individual source scope overrides that take precedence over agent-level scope during execution.

**Independent Test**: Create agent with 2 preset steps: step 1 `source_scope=enterprise_only`, step 2 `source_scope=web_only`. Run a manual-mode query. Verify step 1 only hits enterprise sources and step 2 only does web search.

### Implementation for User Story 5

- [x] T040 [US5] Verify per-step source scope override wiring in researcher nodes: In `src/deep_research/agent/nodes/researcher.py` and `src/deep_research/agent/nodes/react_researcher.py`, when executing a step, check if the current plan step has a `source_scope` override (from `ManualStepDefinition.sources` or similar). If so, temporarily override `state.source_scope_config` for that step's execution and restore it afterward. This may require extending `_convert_preset_steps_to_manual_steps()` in `src/deep_research/agent/orchestrator.py` to carry the `source_scope` field from `AgentPresetStep` to `ManualStepDefinition`.
- [x] T041 [US5] Extend `ManualStepDefinition` (or the equivalent step config passed to researchers) to include `source_scope: str | None` field. In `_convert_preset_steps_to_manual_steps()` in `src/deep_research/agent/orchestrator.py`, propagate `step.source_scope` to the `ManualStepDefinition`.
- [x] T042 [US5] In the frontend step editor within the agent editor (find in `frontend/src/components/agents/` or create inline), add a source scope dropdown per step with options: "Use Agent Default", "Enterprise Only", "Web Only", "All". Wire the selected value to `source_scope` in the `CreatePresetStepRequest`.
- [x] T043 [P] [US5] Write unit test in `tests/unit/agent/test_agent_config_apply.py` verifying that preset steps with `source_scope` overrides are correctly converted: create agent with 2 steps (step1: source_scope="enterprise_only", step2: source_scope=None), call `apply_custom_agent_to_config` in manual mode, verify `config.manual_steps[0]` has source_scope set and `config.manual_steps[1]` inherits None.

**Checkpoint**: User Story 5 is complete. Per-step source scoping works in manual/hybrid workflows.

---

## Phase 8: User Story 6 — Select and Create Simple Prompt Templates (Priority: P6)

**Goal**: Agent editor provides template dropdowns with inline "Create New..." flow for simple plain-text templates.

**Independent Test**: Open agent editor, click system prompt dropdown, select "Create New...", enter name "Be Formal" and text "Always respond in a formal academic tone.", save. Verify the new template appears selected. Save the agent. Run a query. Verify the system prompt was applied (check backend logs for system_instructions).

### Implementation for User Story 6

- [x] T044 [US6] Verify template content is wired to `config.system_instructions` in the agent resolution path: In T006's implementation (in `_run_job()`), after fetching the agent, if `agent.system_prompt_template` relationship is loaded and has `.content`, set `config.system_instructions = agent.system_prompt_template.content`. If `agent.synthesis_template` is loaded, set `config.structured_system_prompt = agent.synthesis_template.content`. Write a unit test confirming this.
- [x] T045 [P] [US6] Create `TemplatePickerDropdown` component in `frontend/src/components/agents/TemplatePickerDropdown.tsx`: Accepts `selectedTemplateId: string | null`, `onChange: (id: string | null) => void`, `templateType: "system" | "synthesis"`. Fetches templates using existing template API/hooks. Lists: "None" option, user's templates, workspace templates (grouped), "Create New..." option. When "Create New..." is selected, renders an inline form (name input + textarea + save/cancel buttons). On save, calls the existing template creation API, then auto-selects the new template.
- [x] T046 [US6] Integrate `TemplatePickerDropdown` into agent editor: Replace the existing template_id input fields (if any) with two `<TemplatePickerDropdown>` instances — one for system prompt, one for synthesis prompt. Wire `selectedTemplateId` to `systemPromptTemplateId` and `synthesisTemplateId` in the agent form state.
- [x] T047 [P] [US6] Write unit test in `tests/unit/agent/test_agent_config_apply.py` verifying template content wiring: create mock agent with `system_prompt_template` relationship containing `content="Be formal"`, call `apply_custom_agent_to_config`, verify `config.system_instructions == "Be formal"`.

**Checkpoint**: User Story 6 is complete. Users can select and create templates inline, and template content is applied to research queries.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Quality assurance, error handling, and integration validation across all stories

- [x] T048 Add OBO permission error handling (FR-012a) in `src/deep_research/services/job_manager.py`: In the `_run_job()` agent resolution block, wrap the agent fetch and config apply in a try/except. If an OBO-related permission error occurs (e.g., accessing a model endpoint the user lacks access to), emit a `StreamErrorEvent` with `error_code="AGENT_RESOURCE_ACCESS_DENIED"` and a message identifying the specific resource. Format: "You do not have access to [resource_type] '[resource_name]'. Contact the agent owner or your workspace admin."
- [x] T049 [P] Add stale endpoint warning display in the frontend agent selector: In `frontend/src/components/chat/MessageInput.tsx` (or wherever the agent picker renders), when the selected agent has `model_override_warnings` (non-empty), show a small warning icon/badge next to the agent name in the dropdown.
- [x] T050 [P] Add edge case handling for deleted sources in `apply_custom_agent_to_config()` in `src/deep_research/agent/orchestrator.py`: If agent's `enabled_sources` references sources that no longer exist (can check against loaded enterprise tools), log a warning with key `AGENT_SOURCE_NOT_FOUND` but do not error — let the agent proceed with available sources.
- [x] T051 Run `make typecheck` and fix any mypy errors introduced by the new fields and parameters across all modified files
- [x] T052 Run `make lint` and fix any ruff errors introduced across all modified files
- [x] T053 Run `uv run pytest tests/unit/ -v` and verify all new and existing tests pass
- [x] T054 [P] Verify quickstart.md scenarios: Follow the quickstart.md development tips to manually test the full flow end-to-end: create agent with model overrides + domain filters + template via API, select from UI, submit query, verify logs show correct config application

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Phase 2 — MVP target
- **User Story 2 (Phase 4)**: Depends on Phase 2 — can run in parallel with US1 (different files)
- **User Story 3 (Phase 5)**: Depends on Phase 2 — can run in parallel with US1/US2
- **User Story 4 (Phase 6)**: Depends on Phase 2 and T021 (apply_custom_agent_to_config extension)
- **User Story 5 (Phase 7)**: Depends on Phase 2 — independent of US3/US4
- **User Story 6 (Phase 8)**: Depends on Phase 2 — independent of all other stories
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

```
Phase 1 (Setup)
  └── Phase 2 (Foundational) ← CRITICAL GATE
        ├── US1 (P1) — Agent Selection (MVP)
        ├── US2 (P2) — Model Overrides
        │     └── requires: T018-T019 (catalog API) before T026-T027 (frontend)
        ├── US3 (P3) — Source Scope
        ├── US4 (P4) — Domain Filters
        │     └── requires: T033 (backend) before T037-T038 (frontend)
        ├── US5 (P5) — Per-Step Sources
        └── US6 (P6) — Templates
              └── Phase 9 (Polish)
```

### Within Each User Story

- Tests (when present) can be written before or alongside implementation
- Backend wiring before frontend UI
- Schema/model changes before API endpoint changes
- API endpoints before frontend components
- Components before integration into existing views

### Parallel Opportunities

**Phase 1**: T004 and T005 can run in parallel (different files: orchestrator.py vs state.py)

**Phase 2**: T006 is sequential (critical path), but T007 and T008 can start once T003 (schemas) is done

**After Phase 2 completes, these user stories can run in parallel**:
- US1 (T011-T015): Frontend agent picker + persistence
- US2 (T016-T028): Endpoint catalog + model override UI
- US3 (T029-T031): Source scope hide
- US6 (T044-T047): Template picker

**Within US2**: T018, T019, T024, T025 can all run in parallel (different files)

**Within US4**: T032, T037 can run in parallel (test + frontend component)

---

## Parallel Example: User Story 2

```bash
# Wave 1 — All parallel (different files):
Task T018: "Create EndpointCatalogResponse schemas in src/deep_research/schemas/config.py"
Task T019: "Create GET /config/model-catalog endpoint in src/deep_research/api/v1/config.py"
Task T024: "Create frontend API client in frontend/src/api/config.ts"
Task T025: "Create useModelCatalog hook in frontend/src/hooks/useModelCatalog.ts"
Task T016: "Write unit test for catalog endpoint in tests/unit/api/test_config_endpoint.py"
Task T017: "Write unit test for model override apply in tests/unit/agent/test_agent_config_apply.py"

# Wave 2 — Depends on Wave 1:
Task T020: "Register config router in src/deep_research/api/v1/__init__.py"
Task T021: "Extend apply_custom_agent_to_config for model overrides in orchestrator.py"
Task T022: "Add endpoint_override to LLMClient in services/llm/client.py"
Task T028: "Add model_overrides to frontend types in frontend/src/types/customAgents.ts"

# Wave 3 — Depends on Wave 2:
Task T023: "Thread model_overrides to LLM calls in agent nodes"
Task T026: "Create ModelConfigSection component"

# Wave 4 — Depends on Wave 3:
Task T027: "Integrate ModelConfigSection into agent editor"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T010)
3. Complete Phase 3: User Story 1 (T011-T015)
4. **STOP and VALIDATE**: Test that selecting an agent applies its existing config fields (source scope, workflow mode, depth)
5. This delivers immediate value: agents that were configurable but non-functional now actually work

### Incremental Delivery

1. Setup + Foundational → Agent pipeline activated (MVP!)
2. Add US1 → Agent selection works → Demo
3. Add US2 → Model overrides configurable → Demo (high business impact)
4. Add US3 → Source scoping per-agent → Demo
5. Add US4 → Domain filters → Demo
6. Add US5+US6 → Fine-grained control → Feature complete
7. Polish → Production ready

### Parallel Team Strategy

With 2-3 developers after Phase 2 completes:

- **Developer A**: US1 (frontend agent picker polish) + US3 (source scope hide) + US6 (template picker)
- **Developer B**: US2 (model overrides — most complex, largest surface area)
- **Developer C**: US4 (domain filters) + US5 (per-step sources)

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- The **single most important unlock** is T006 (wiring `apply_custom_agent_to_config` in `_run_job()`). Without it, nothing works. With it, all existing agent fields immediately become functional.
- Total task count: 54
- DB is not live — migrations can be recreated freely
- Frontend agent picker already exists and threads `agentId` — no new frontend wiring needed for the basic flow
- LLM client uses `tier` parameter (ModelTier enum), not `role` string — the endpoint override parameter should match
- Domain filter override **replaces** system-wide filter (no merge)
- Template content is plain text — no variable substitution
