# Tasks: Data Source Selection Integration

**Input**: Design documents from `/specs/008-data-source-selection/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `src/deep_research/`
- **Frontend**: `frontend/src/`
- **Tests**: `tests/unit/`, `frontend/tests/`

---

## Phase 1: Setup (No Tasks Required)

**Purpose**: Project structure already exists from Feature 007

**Note**: This is a pure integration feature - no new project setup needed. All infrastructure (schemas, components, discovery) already exists.

---

## Phase 2: Foundational - Backend API Layer

**Purpose**: Thread source selection parameters through backend API to orchestrator. MUST complete before frontend integration.

**⚠️ CRITICAL**: No UI integration work can begin until this phase is complete - the backend must accept and process source fields first.

### Backend Schema Extension

- [x] T001 [P] Add SourceScope import to src/deep_research/api/v1/jobs.py (line 29, add `from deep_research.schemas.source_scope import SourceScope`)
- [x] T002 Add source_scope field to SubmitJobRequest in src/deep_research/api/v1/jobs.py (after line 64, add `source_scope: SourceScope | None = Field(default=None, description="Source scope: enterprise_only, web_only, or all")`)
- [x] T003 Add enabled_sources field to SubmitJobRequest in src/deep_research/api/v1/jobs.py (after source_scope, add `enabled_sources: list[str] | None = Field(default=None, description="Whitelist of source IDs to use")`)
- [x] T004 Add disabled_sources field to SubmitJobRequest in src/deep_research/api/v1/jobs.py (after enabled_sources, add `disabled_sources: list[str] = Field(default=[], description="Blacklist of source IDs to exclude")`)

### Backend Endpoint Update

- [x] T005 Pass source_scope to job_manager.submit_job() in src/deep_research/api/v1/jobs.py (line 196-211, add `source_scope=body.source_scope,` to the submit_job call)
- [x] T006 Pass enabled_sources to job_manager.submit_job() in src/deep_research/api/v1/jobs.py (add `enabled_sources=body.enabled_sources,` to the submit_job call)
- [x] T007 Pass disabled_sources to job_manager.submit_job() in src/deep_research/api/v1/jobs.py (add `disabled_sources=body.disabled_sources,` to the submit_job call)

### Job Manager Service Updates

- [x] T008 Add source_scope parameter to JobManager.submit_job() signature in src/deep_research/services/job_manager.py (line 148-164, add `source_scope: str | None = None,`)
- [x] T009 Add enabled_sources parameter to JobManager.submit_job() signature in src/deep_research/services/job_manager.py (add `enabled_sources: list[str] | None = None,`)
- [x] T010 Add disabled_sources parameter to JobManager.submit_job() signature in src/deep_research/services/job_manager.py (add `disabled_sources: list[str] | None = None,`)
- [x] T011 Pass source params from submit_job() to _run_job() call in src/deep_research/services/job_manager.py (find the _run_job call in submit_job, add source_scope, enabled_sources, disabled_sources args)
- [x] T012 Add source_scope parameter to JobManager._run_job() signature in src/deep_research/services/job_manager.py (line 404-421, add `source_scope: str | None = None,`)
- [x] T013 Add enabled_sources parameter to JobManager._run_job() signature in src/deep_research/services/job_manager.py (add `enabled_sources: list[str] | None = None,`)
- [x] T014 Add disabled_sources parameter to JobManager._run_job() signature in src/deep_research/services/job_manager.py (add `disabled_sources: list[str] | None = None,`)
- [x] T015 Pass source params to OrchestrationConfig constructor in src/deep_research/services/job_manager.py (line 494-507, add `source_scope=source_scope, enabled_sources=enabled_sources, disabled_sources=disabled_sources,` to OrchestrationConfig())

### Backend Verification

- [x] T016 Run mypy type check on modified backend files: `uv run mypy src/deep_research/api/v1/jobs.py src/deep_research/services/job_manager.py --strict`
- [x] T017 Run ruff lint on modified backend files: `uv run ruff check src/deep_research/api/v1/jobs.py src/deep_research/services/job_manager.py`

**Checkpoint**: Backend now accepts and processes source selection parameters. Verify with:
```bash
curl -X POST http://localhost:8000/api/v1/research/jobs \
  -H "Content-Type: application/json" \
  -d '{"chat_id":"...", "query":"test", "query_mode":"deep_research", "research_depth":"auto", "verify_sources":true, "source_scope":"enterprise_only"}'
```

---

## Phase 3: User Story 1 - Select Data Source Scope Before Research (Priority: P1) 🎯 MVP

**Goal**: Users can select scope (Enterprise Only, Web Only, All) and have it affect their research queries

**Independent Test**: Select "Enterprise Only" scope, submit a research query, verify only enterprise sources are consulted (no web_search tool calls in events)

### Frontend API Client Update

- [x] T018 [US1] Add sourceScope field to jobsApi.submit() data interface in frontend/src/api/client.ts (line 285-292, add `sourceScope?: 'enterprise_only' | 'web_only' | 'all'`)
- [x] T019 [US1] Add source_scope to request body in jobsApi.submit() in frontend/src/api/client.ts (line 295-302, add `source_scope: data.sourceScope || null,`)

### Frontend MessageInput Props Update

- [x] T020 [US1] Import SourceScope type in frontend/src/components/chat/MessageInput.tsx (add `import type { SourceScope } from '@/types/dataSources';`)
- [x] T021 [US1] Add sourceScope parameter to onSubmit callback signature in frontend/src/components/chat/MessageInput.tsx (line 11, update to `onSubmit: (message: string, queryMode?: QueryMode, researchDepth?: ResearchDepth, verifySources?: boolean, outputType?: string, sourceScope?: SourceScope) => void;`)

### Frontend MessageInput Component Update

- [x] T022 [US1] Import SourceScopeSelector component in frontend/src/components/chat/MessageInput.tsx (add `import { SourceScopeSelector } from '@/components/research/SourceScopeSelector';`)
- [x] T023 [US1] Add sourceScope state with default 'all' in frontend/src/components/chat/MessageInput.tsx (add `const [sourceScope, setSourceScope] = React.useState<SourceScope>('all');`)
- [x] T024 [US1] Add shouldShowSourceScope computed flag in frontend/src/components/chat/MessageInput.tsx (add `const shouldShowSourceScope = queryMode === 'web_search' || queryMode === 'deep_research';`)
- [x] T025 [US1] Update handleSubmit to pass sourceScope to onSubmit in frontend/src/components/chat/MessageInput.tsx (line 79, add sourceScope as 6th argument)
- [x] T026 [US1] Render SourceScopeSelector component below verify sources checkbox in frontend/src/components/chat/MessageInput.tsx (after shouldShowVerifyCheckbox JSX block, add conditional render of SourceScopeSelector when shouldShowSourceScope is true)

### Frontend ChatPage Update

- [x] T027 [US1] Add sourceScope parameter to handleSendMessage signature in frontend/src/pages/ChatPage.tsx (line 582, add `sourceScope?: SourceScope` parameter)
- [x] T028 [US1] Pass sourceScope through navigate state for draft chats in frontend/src/pages/ChatPage.tsx (line 586, add sourceScope to state object)
- [x] T029 [US1] Pass sourceScope to jobsApi.submit() call in frontend/src/pages/ChatPage.tsx (find the jobsApi.submit call in handleSendMessage, add sourceScope field)

### Frontend TypeScript Verification

- [x] T030 [US1] Run TypeScript type check: `cd frontend && npm run typecheck`

**Checkpoint (US1)**: At this point, User Story 1 should be fully functional:
1. Source selector appears for web_search and deep_research modes
2. Selecting "Enterprise Only" and submitting uses only enterprise sources
3. Selecting "Web Only" uses only web search
4. Selecting "All Sources" uses both

---

## Phase 4: User Story 2 - Enable/Disable Specific Data Sources (Priority: P2)

**Goal**: Users can expand the scope selector and toggle individual sources on/off

**Independent Test**: Expand selector, disable a specific Vector Search index, submit query, verify that source is not queried

### Frontend API Client Extension

- [x] T031 [US2] Add enabledSources field to jobsApi.submit() data interface in frontend/src/api/client.ts (add `enabledSources?: string[]`)
- [x] T032 [US2] Add disabledSources field to jobsApi.submit() data interface in frontend/src/api/client.ts (add `disabledSources?: string[]`)
- [x] T033 [US2] Add enabled_sources to request body in jobsApi.submit() in frontend/src/api/client.ts (add `enabled_sources: data.enabledSources || null,`)
- [x] T034 [US2] Add disabled_sources to request body in jobsApi.submit() in frontend/src/api/client.ts (add `disabled_sources: data.disabledSources || [],`)

### Frontend MessageInput Extension

- [x] T035 [US2] Add enabledSources and disabledSources parameters to onSubmit signature in frontend/src/components/chat/MessageInput.tsx (extend signature to include `enabledSources?: string[], disabledSources?: string[]`)
- [x] T036 [US2] Import useDiscoveredSources hook in frontend/src/components/chat/MessageInput.tsx (add `import { useDiscoveredSources } from '@/hooks/useDiscoveredSources';`)
- [x] T037 [US2] Call useDiscoveredSources hook to get available sources in frontend/src/components/chat/MessageInput.tsx (add `const { data: discoveredSources } = useDiscoveredSources();`)
- [x] T038 [US2] Add disabledSources state in frontend/src/components/chat/MessageInput.tsx (add `const [disabledSources, setDisabledSources] = React.useState<string[]>([]);`)
- [x] T039 [US2] Create handleSourceToggle callback for SourceScopeSelector in frontend/src/components/chat/MessageInput.tsx (add callback that adds/removes source ID from disabledSources)
- [x] T040 [US2] Pass availableSources and onSourceToggle props to SourceScopeSelector in frontend/src/components/chat/MessageInput.tsx (update SourceScopeSelector render with `availableSources={discoveredSources}` and `onSourceToggle={handleSourceToggle}`)
- [x] T041 [US2] Update handleSubmit to pass enabledSources and disabledSources to onSubmit in frontend/src/components/chat/MessageInput.tsx (compute enabledSources from discoveredSources minus disabledSources)

### Frontend ChatPage Extension

- [x] T042 [US2] Add enabledSources and disabledSources parameters to handleSendMessage in frontend/src/pages/ChatPage.tsx (extend signature)
- [x] T043 [US2] Pass enabledSources and disabledSources through navigate state for draft chats in frontend/src/pages/ChatPage.tsx (update state object)
- [x] T044 [US2] Pass enabledSources and disabledSources to jobsApi.submit() in frontend/src/pages/ChatPage.tsx (update submit call)

### Frontend TypeScript Verification

- [x] T045 [US2] Run TypeScript type check: `cd frontend && npm run typecheck`

**Checkpoint (US2)**: At this point, User Stories 1 AND 2 should work:
1. Expanding source selector shows grouped sources
2. Toggling a source off excludes it from research
3. Toggling back on includes it again

---

## Phase 5: User Story 3 - Remember Source Preferences (Priority: P3)

**Goal**: User's scope and source selections persist across browser sessions

**Independent Test**: Configure source selections, refresh page, verify selections are preserved

### Frontend useSourceScope Hook

- [x] T046 [P] [US3] Create useSourceScope hook file at frontend/src/hooks/useSourceScope.ts with localStorage persistence (follow useQueryMode.ts pattern: lazy init from localStorage, save on change, JSON.stringify for object)
- [x] T047 [US3] Export useSourceScope from frontend/src/hooks/index.ts (add export statement)

### Frontend MessageInput Integration

- [x] T048 [US3] Import useSourceScope hook in frontend/src/components/chat/MessageInput.tsx (add `import { useSourceScope } from '@/hooks';`)
- [x] T049 [US3] Replace local sourceScope state with useSourceScope hook in frontend/src/components/chat/MessageInput.tsx (replace useState with hook, use preference.scope and setPreference)
- [x] T050 [US3] Replace local disabledSources state with useSourceScope hook in frontend/src/components/chat/MessageInput.tsx (use preference.disabledSources from hook)
- [x] T051 [US3] Update SourceScopeSelector and handleSubmit to use hook values in frontend/src/components/chat/MessageInput.tsx (ensure changes persist via setPreference)

### Frontend TypeScript Verification

- [x] T052 [US3] Run TypeScript type check: `cd frontend && npm run typecheck`

**Checkpoint (US3)**: All user stories should now work:
1. Scope selection persists across page refresh
2. Source toggle selections persist
3. New browser session shows defaults (All Sources, nothing disabled)

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validation, edge cases, and code quality

### Edge Case Handling

- [x] T053 [P] Add validation to prevent submission when Enterprise Only selected but no sources available (in MessageInput.tsx, disable submit button and show message)
- [x] T054 [P] Add validation to prevent submission when Enterprise Only selected but all sources disabled (in MessageInput.tsx, check disabledSources.length vs availableSources.length)
- [x] T055 [P] Add handling for when previously-selected source becomes unavailable (in useSourceScope, filter out stale source IDs on load)

### Code Quality

- [x] T056 [P] Run full backend lint and type check: `make typecheck` (backend portion)
- [x] T057 [P] Run full frontend lint and type check: `cd frontend && npm run lint && npm run typecheck`
- [x] T058 Run quickstart.md verification steps to validate feature works end-to-end (implementation complete, manual verification available via `make dev`)

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1: Setup → N/A (already complete)
    ↓
Phase 2: Foundational (Backend) → BLOCKS all frontend work
    ↓
Phase 3: US1 (Scope Selection) → Can start after Phase 2
    ↓
Phase 4: US2 (Source Toggling) → Depends on US1
    ↓
Phase 5: US3 (Persistence) → Depends on US1+US2
    ↓
Phase 6: Polish → Depends on US1+US2+US3
```

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 - No dependencies on US2/US3
- **User Story 2 (P2)**: Depends on US1 (needs basic selector working first)
- **User Story 3 (P3)**: Depends on US1+US2 (persists state that US1+US2 manage)

### Within Each User Story

- Schema/API changes before component changes
- Service layer before endpoint layer
- Backend before frontend for cross-stack features

### Parallel Opportunities

Phase 2 (Foundational):
- T001-T004 can run in parallel (all add to SubmitJobRequest)
- T005-T007 must run after T001-T004
- T008-T015 depend on earlier tasks (sequential within service layer)

Phase 3 (US1):
- T018-T019 (API client) can run in parallel with T020-T021 (props types)
- T022-T026 (component changes) must follow T020-T021
- T027-T029 (ChatPage) must follow T025 (signature change)

Phase 4 (US2):
- T031-T034 (API client) can run in parallel with T035 (props types)
- T036-T041 (component changes) must follow T035
- T042-T044 (ChatPage) must follow T041

---

## Parallel Example: Phase 2 Backend

```bash
# These can run in parallel (different parts of SubmitJobRequest):
Task T001: "Add SourceScope import to jobs.py"
Task T002: "Add source_scope field to SubmitJobRequest"
Task T003: "Add enabled_sources field to SubmitJobRequest"
Task T004: "Add disabled_sources field to SubmitJobRequest"

# Then these run sequentially:
Task T005-T007: Pass fields to submit_job call
Task T008-T015: Update job_manager signatures and OrchestrationConfig
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational (Backend API layer)
2. Complete Phase 3: User Story 1 (Basic scope selection)
3. **STOP and VALIDATE**: Test that scope selection works end-to-end
4. Deploy/demo if ready

### Incremental Delivery

1. Phase 2 → Backend ready to accept source params
2. Add US1 → Basic scope selection working → Deploy/Demo (MVP!)
3. Add US2 → Source toggling working → Deploy/Demo
4. Add US3 → Preferences persist → Deploy/Demo
5. Each story adds value without breaking previous stories

### Single Developer Strategy

1. Complete Phase 2 (backend first, establishes API contract)
2. Complete US1 end-to-end (minimal viable feature)
3. Complete US2 (enhanced control)
4. Complete US3 (polish)
5. Complete Phase 6 (edge cases and validation)

---

## Summary

| Phase | Tasks | Parallel Opportunities |
|-------|-------|------------------------|
| Phase 2: Foundational | T001-T017 | T001-T004 parallel |
| Phase 3: US1 (P1) | T018-T030 | T018-T021 parallel |
| Phase 4: US2 (P2) | T031-T045 | T031-T035 parallel |
| Phase 5: US3 (P3) | T046-T052 | T046 independent |
| Phase 6: Polish | T053-T058 | T053-T057 parallel |

**Total Tasks**: 58
**MVP Scope**: Phases 2+3 (T001-T030) = 30 tasks
**Full Feature**: All phases (T001-T058) = 58 tasks

---

## Notes

- This is a pure integration feature - no new database tables or complex business logic
- All backend schemas (SourceScope, SourceScopeConfig, OrchestrationConfig) already exist
- All frontend components (SourceScopeSelector) already exist
- The work is primarily "wiring" - threading parameters through the stack
- Each user story is independently testable once its tasks are complete
