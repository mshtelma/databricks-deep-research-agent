# Implementation Plan: Data Source Selection Integration

**Branch**: `008-data-source-selection` | **Date**: 2026-02-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-data-source-selection/spec.md`

## Summary

Connect existing SourceScopeSelector UI component to research submission flow, enabling users to select which enterprise data sources (Vector Search, Genie, Knowledge Assistant) to use. This is a **pure integration/wiring feature** - all schemas, components, and business logic exist from Feature 007; this feature threads source selection parameters through the HTTP request pipeline from frontend to orchestrator.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI 0.109+, React 18, TanStack Query 5.x, Pydantic 2.x
**Storage**: Databricks Lakebase (PostgreSQL) - existing, localStorage for preferences
**Testing**: pytest (backend unit/integration), Playwright (E2E)
**Target Platform**: Databricks Apps (Linux container), modern browsers
**Project Type**: Web application (backend + frontend)
**Performance Goals**: Source selector renders in <2s (SC-005), submission in <10s (SC-001)
**Constraints**: Must work with OBO authentication, no breaking changes to existing API
**Scale/Scope**: Single user preferences, ~10-50 data sources per user

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Clients and Workspace Integration
- ✅ **PASS**: No new LLM calls - this is pure data flow integration
- ✅ **PASS**: OBO authentication already implemented in Feature 007

### Principle II: Typing-First Python
- ✅ **PASS**: All new fields use existing typed Pydantic models (`SourceScope`, `SourceScopeConfig`)
- ✅ **PASS**: Method signatures will have full type annotations

### Principle III: Avoid Runtime Introspection
- ✅ **PASS**: Uses explicit Pydantic models with discriminated unions
- ✅ **PASS**: No hasattr/isinstance checks planned

### Principle IV: Linting and Static Type Enforcement
- ✅ **PASS**: Will require mypy strict + ruff pass before merge
- ✅ **PASS**: TypeScript strict mode for frontend changes

**Gate Status**: PASSED - No violations detected

## Project Structure

### Documentation (this feature)

```text
specs/008-data-source-selection/
├── spec.md              # Feature specification (COMPLETE)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
backend/
├── src/deep_research/
│   ├── api/v1/
│   │   └── jobs.py              # SubmitJobRequest schema updates
│   ├── services/
│   │   └── job_manager.py       # submit_job(), _run_job() parameter threading
│   ├── schemas/
│   │   ├── source_scope.py      # ✅ EXISTS - SourceScope, SourceScopeConfig
│   │   └── research_request.py  # ✅ EXISTS - ResearchRequest with source fields
│   └── agent/
│       └── orchestrator.py      # ✅ EXISTS - OrchestrationConfig has source fields
└── tests/
    └── unit/api/test_jobs.py    # New tests for source scope fields

frontend/
├── src/
│   ├── api/
│   │   └── client.ts            # jobsApi.submit() parameter additions
│   ├── components/
│   │   ├── chat/
│   │   │   └── MessageInput.tsx # onSubmit signature, SourceScopeSelector rendering
│   │   └── research/
│   │       └── SourceScopeSelector.tsx  # ✅ EXISTS - 385 lines, fully built
│   ├── hooks/
│   │   ├── useDiscoveredSources.ts      # ✅ EXISTS - discovery hooks
│   │   └── useSourceScope.ts            # NEW - localStorage persistence
│   └── pages/
│       └── ChatPage.tsx         # handleSendMessage source param threading
└── tests/
    └── components/              # MessageInput tests
```

**Structure Decision**: Web application with existing backend/frontend separation. Feature 008 only modifies existing files - no new directories needed except `contracts/` for API schema documentation.

## Complexity Tracking

> No constitution violations detected - this section is empty.

---

## Phase 0: Outline & Research

### Research Tasks

| Unknown | Research Task | Priority |
|---------|---------------|----------|
| Source scope field propagation | Verify existing schema fields work end-to-end | P0 |
| Frontend state persistence | Best practices for localStorage with TypeScript | P1 |
| API backward compatibility | Verify optional fields don't break existing clients | P1 |

### Consolidation Plan

Research will verify:
1. `SourceScope` enum values match frontend/backend
2. `OrchestrationConfig` correctly receives source fields from `JobManager`
3. Existing planner/researcher nodes read source scope from state
4. localStorage patterns used elsewhere in the codebase

---

## Phase 1: Design & Contracts

### Data Flow Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  MessageInput                                                                │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐   │
│  │SourceScopeSelector │→│ onSubmit(msg,scope,...)│→│ ChatPage.handleSend│   │
│  │ [Enterprise|Web|All]│    │                      │    │                  │   │
│  └─────────────────┘    └──────────────────────┘    └──────────────────┘   │
│                                                              ↓               │
│                                                    ┌──────────────────┐      │
│                                                    │ jobsApi.submit() │      │
│                                                    │ POST /research/jobs│    │
│                                                    └──────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKEND                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐    ┌───────────────────┐    ┌───────────────────┐  │
│  │ SubmitJobRequest   │→│ JobManager.submit_job│→│ JobManager._run_job│  │
│  │ + source_scope     │    │ + source_scope     │    │ + source_scope   │  │
│  │ + enabled_sources  │    │ + enabled_sources  │    │ + enabled_sources│  │
│  │ + disabled_sources │    │ + disabled_sources │    │ + disabled...    │  │
│  └────────────────────┘    └───────────────────┘    └───────────────────┘  │
│                                                              ↓               │
│                                                    ┌───────────────────┐     │
│                                                    │OrchestrationConfig│     │
│                                                    │ source_scope ✅   │     │
│                                                    │ enabled_sources ✅│     │
│                                                    │ disabled_sources✅│     │
│                                                    └───────────────────┘     │
│                                                              ↓               │
│                                                    ┌───────────────────┐     │
│                                                    │ Planner/Researcher│     │
│                                                    │ (reads from state)│     │
│                                                    └───────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### API Contract Changes

**Endpoint**: `POST /api/v1/research/jobs`

**Request Body Changes** (additions to existing `SubmitJobRequest`):

```python
# NEW OPTIONAL FIELDS (all with defaults for backward compatibility)
source_scope: SourceScope | None = None  # "enterprise_only" | "web_only" | "all"
enabled_sources: list[str] | None = None  # Whitelist of source IDs
disabled_sources: list[str] = []  # Blacklist of source IDs
```

**Response**: No changes - existing `JobResponse` remains unchanged.

### Frontend Type Changes

```typescript
// In client.ts jobsApi.submit() data parameter
interface SubmitJobData {
  chatId: string
  query: string
  queryMode?: string
  researchDepth?: string
  verifySources?: boolean
  outputType?: string
  // NEW FIELDS:
  sourceScope?: 'enterprise_only' | 'web_only' | 'all'
  enabledSources?: string[]
  disabledSources?: string[]
}
```

### State Persistence Design

```typescript
// useSourceScope.ts hook
const STORAGE_KEY = 'deep-research-source-scope'

interface SourceScopePreference {
  scope: SourceScope
  enabledSources: string[]
  disabledSources: string[]
  updatedAt: string  // ISO timestamp
}

// Behavior:
// - Load from localStorage on mount
// - Save to localStorage on change (debounced)
// - Default: { scope: 'all', enabledSources: [], disabledSources: [] }
```

---

## Integration Points Summary

### Files to Modify (12 total)

| Layer | File | Change Type | Lines Affected |
|-------|------|-------------|----------------|
| Backend Schema | `api/v1/jobs.py` | Add fields to SubmitJobRequest | 42-64 |
| Backend Endpoint | `api/v1/jobs.py` | Pass fields in submit_job call | 196-211 |
| Backend Service | `services/job_manager.py` | Add params to submit_job() | 148-164 |
| Backend Service | `services/job_manager.py` | Add params to _run_job() | 404-421 |
| Backend Service | `services/job_manager.py` | Pass to OrchestrationConfig | 494-507 |
| Frontend API | `api/client.ts` | Add fields to submit() | 283-303 |
| Frontend Component | `components/chat/MessageInput.tsx` | Update onSubmit signature | 10-11 |
| Frontend Component | `components/chat/MessageInput.tsx` | Render SourceScopeSelector | ~107-133 |
| Frontend Component | `components/chat/MessageInput.tsx` | Pass params in handleSubmit | ~79 |
| Frontend Page | `pages/ChatPage.tsx` | Thread params in handleSendMessage | TBD |
| Frontend Hook | `hooks/useSourceScope.ts` | NEW - localStorage persistence | NEW FILE |
| Backend Test | `tests/unit/api/test_jobs.py` | Test new fields | NEW TESTS |

### Existing Components to Reuse (No Modifications)

| Component | File | Status |
|-----------|------|--------|
| SourceScopeSelector | `frontend/src/components/research/SourceScopeSelector.tsx` | ✅ Ready (385 lines) |
| SourceScope enum | `src/deep_research/schemas/source_scope.py` | ✅ Ready |
| SourceScopeConfig | `src/deep_research/schemas/source_scope.py` | ✅ Ready |
| useDiscoveredSources | `frontend/src/hooks/useDiscoveredSources.ts` | ✅ Ready (226 lines) |
| ResearchRequest | `src/deep_research/schemas/research_request.py` | ✅ Ready |
| OrchestrationConfig | `src/deep_research/agent/orchestrator.py` | ✅ Ready |

---

## Verification Plan

### Manual Testing
1. Start app with `make dev`
2. Navigate to chat page
3. Verify SourceScopeSelector appears below mode selector (for deep_research/web_search modes)
4. Select "Enterprise Only" scope
5. Submit a research query
6. Check backend logs for `source_scope=enterprise_only`
7. Verify no web_search tool calls in research events
8. Refresh page - verify scope selection persists

### Automated Tests
- Unit: `SubmitJobRequest` accepts source fields
- Unit: `JobManager.submit_job()` passes source params
- Integration: Full request flow with source scope
- E2E: Select scope → submit query → verify events

### TypeScript Validation
- `cd frontend && npm run typecheck` must pass

---

## Post-Design Constitution Re-Check

### Principle I: Clients and Workspace Integration
- ✅ **PASS**: No new client patterns introduced

### Principle II: Typing-First Python
- ✅ **PASS**: All new parameters have type annotations
- ✅ **PASS**: Uses existing Pydantic models

### Principle III: Avoid Runtime Introspection
- ✅ **PASS**: No isinstance/hasattr checks

### Principle IV: Linting and Static Type Enforcement
- ✅ **PASS**: Plan requires mypy strict + ruff pass

**Final Gate Status**: PASSED

---

## Next Steps

1. Run `/speckit.tasks` to generate tasks.md with ordered implementation tasks
2. Implement P1 tasks (backend schema + endpoint updates)
3. Implement P2 tasks (frontend API client updates)
4. Implement P3 tasks (MessageInput integration + useSourceScope hook)
5. Run verification plan
6. Create PR
