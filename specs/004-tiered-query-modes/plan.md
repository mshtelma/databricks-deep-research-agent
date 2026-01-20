# Implementation Plan: Tiered Query Modes

**Branch**: `004-tiered-query-modes` | **Date**: 2026-01-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-tiered-query-modes/spec.md`

## Summary

Implement three query modes (Simple, Web Search, Deep Research) with progressive disclosure UI, plus enhanced activity display with centered panel during research, collapsible accordion after completion, enhanced event labels, and comprehensive visited sources tracking.

**Technical Approach**:
- Add `QueryMode` enum to backend and frontend type systems
- Modify `OrchestrationConfig` and `ResearchState` to carry query mode through pipeline
- Create mode-specific routing logic in coordinator and orchestrator
- Add `QueryModeSelector` component to frontend with progressive disclosure for depth options
- Extend database schema with query_mode columns and new ResearchEvent/VisitedSource tables
- Refactor activity panel from sidebar to centered position during research

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI, Pydantic v2, React 18, TanStack Query, Tailwind CSS
**Storage**: Databricks Lakebase (PostgreSQL) via asyncpg, existing schema extensions
**Testing**: pytest (unit/integration/complex tiers), Vitest (frontend), Playwright (E2E)
**Target Platform**: Web application (Linux server backend, modern browsers frontend)
**Project Type**: Web application (backend + frontend)
**Performance Goals**: Simple mode <3s TTFB, Web Search <15s total, Deep Research existing perf
**Constraints**: Animation transitions <300ms, mode switch <100ms, no layout shift
**Scale/Scope**: Existing user base, 9 user stories, 45 functional requirements

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Clients & Workspace Integration** | PASS | LLM calls via AsyncOpenAI client, all existing patterns preserved |
| **II. Typing-First Python** | PASS | All new models use Pydantic v2 with full type annotations, enums as str subclasses |
| **III. Avoid Runtime Introspection** | PASS | Explicit QueryMode enum instead of string checks, Pydantic validation at boundaries |
| **IV. Linting & Static Type Enforcement** | PASS | mypy strict mode, ruff linting, TypeScript strict mode on frontend |

**Gate Status**: PASSED - All 4 constitution principles satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/004-tiered-query-modes/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (OpenAPI additions)
│   └── openapi-patch.yaml
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/                                    # Python backend
├── agent/
│   ├── orchestrator.py                 # Add query_mode routing logic (~50 lines)
│   ├── state.py                        # Add query_mode to ResearchState
│   ├── config.py                       # Add get_query_mode_config()
│   └── nodes/
│       ├── coordinator.py              # Unchanged (existing is_simple_query path reused)
│       ├── researcher.py               # Unchanged (mode=classic with limits reused)
│       └── synthesizer.py              # Unchanged (generation_mode=natural reused)
├── api/
│   └── v1/
│       ├── research.py                 # Add query_mode parameter
│       └── preferences.py              # Expose default_query_mode
├── models/
│   ├── enums.py                        # Add QueryMode enum
│   ├── research_session.py             # Add query_mode column
│   ├── user_preferences.py             # Add default_query_mode field
│   └── research_event.py               # NEW: ResearchEvent model
├── schemas/
│   └── streaming.py                    # Add mode-specific event filtering
├── services/
│   ├── preferences_service.py          # Add query_mode methods
│   └── research_event_service.py       # NEW: Event persistence service
├── core/
│   └── app_config.py                   # Add QueryModeConfig models
└── db/
    └── migrations/versions/
        └── 005_tiered_query_modes.py   # NEW: Migration for schema changes

frontend/                               # React frontend
├── src/
│   ├── components/
│   │   ├── chat/
│   │   │   ├── QueryModeSelector.tsx   # NEW: Mode selector with progressive disclosure
│   │   │   ├── MessageInput.tsx        # Add mode selector integration
│   │   │   └── AgentMessage.tsx        # Mode-aware rendering
│   │   ├── research/
│   │   │   ├── CenteredActivityPanel.tsx # NEW: Centered activity display
│   │   │   ├── ActivityAccordion.tsx   # NEW: Collapsible accordion for completed research
│   │   │   ├── EnhancedEventLabel.tsx  # NEW: Rich event labels with context
│   │   │   └── VisitedSourcesPanel.tsx # NEW: All visited sources display
│   │   └── common/
│   │       └── VirtualizedList.tsx     # NEW: For >50 events virtualization
│   ├── hooks/
│   │   ├── useStreamingQuery.ts        # Add queryMode to sendQuery signature
│   │   ├── useQueryMode.ts             # NEW: Mode state management hook
│   │   └── useResearchEvents.ts        # NEW: Event accumulation hook
│   ├── types/
│   │   └── index.ts                    # Add QueryMode type
│   └── utils/
│       └── activityLabels.ts           # Extend for enhanced event formatting
└── tests/
    └── components/
        └── QueryModeSelector.test.tsx  # NEW: Component tests

tests/                                  # Python backend tests
├── unit/
│   ├── agent/
│   │   └── test_query_mode_routing.py  # NEW: Mode routing unit tests
│   └── services/
│       └── test_research_event_service.py # NEW: Event service tests
├── integration/
│   └── test_tiered_modes.py            # NEW: Integration tests for all modes
└── complex/
    └── test_mode_transitions.py        # NEW: Complex mode switching tests

e2e/                                    # Playwright E2E tests
├── tests/
│   └── tiered-modes.spec.ts            # NEW: E2E tests for mode selection
└── pages/
    └── ChatPage.ts                     # Extend with mode selector methods

config/
└── app.yaml                            # Add query_modes section
```

**Structure Decision**: Existing web application structure (src/ for backend, frontend/ for React app) with new files added per component. Follows existing patterns for models, services, components, and hooks.

## Complexity Tracking

No constitution violations. All new code follows established patterns. Maximum code reuse achieved.

| Aspect | Justification |
|--------|---------------|
| New ResearchEvent table | Required for accordion display persistence (FR-029, FR-034) |
| New VisitedSource table | Required for comprehensive source tracking (FR-037, FR-045) |
| Web Search Mode Routing | ~50 lines in orchestrator reusing researcher+synthesizer (FR-006, FR-007) |
| Centered Activity Panel | UX requirement for visibility during research (FR-017, FR-018) |

**Reuse Summary**:
| Component | Web Search Mode Usage |
|-----------|----------------------|
| `is_simple_query` path | Simple mode (100% existing code) |
| `run_researcher()` | `mode=classic`, 2 queries, 3 crawls |
| `stream_synthesis()` | `generation_mode=natural` for [1],[2] citations |
| Full pipeline | Deep Research mode (100% existing code) |

## Phase 0: Research Summary

**Key Technical Decisions**:

1. **QueryMode Enum**: Following `ResearchDepth` pattern - `class QueryMode(str, Enum)` with SIMPLE, WEB_SEARCH, DEEP_RESEARCH values

2. **Maximum Code Reuse Strategy**: Instead of creating a new `WebSearchAgent`, we reuse existing components:
   - **Simple Mode**: Reuse existing `is_simple_query` path in orchestrator (100% existing code)
   - **Web Search Mode**: Single-step researcher with `mode=classic`, synthesizer with `generation_mode=natural`
   - **Deep Research Mode**: Unchanged full pipeline (100% existing code)

3. **Web Search Mode Pipeline** (reusing existing components):
   - Create minimal 1-step plan programmatically: "Answer the query"
   - Run researcher with `mode=classic`, 2 search queries, 3 crawls max
   - Skip reflector (always COMPLETE after 1 step)
   - Run synthesizer with `generation_mode=natural` for [1], [2] citations
   - Creates lightweight ResearchSession (sources only, no claims/steps)

4. **Activity Panel Positioning**:
   - During research: Centered panel below user message with sticky positioning
   - After completion: Transition to collapsed accordion in message card
   - Use CSS transforms for smooth transitions

5. **Event Persistence**: Store ResearchEvent records linked to research_session_id with:
   - event_type, timestamp, payload (JSONB for event-specific data)
   - Cascade delete when chat is deleted

6. **Visited Sources Tracking**: Extend tool_result event handling to capture all crawled URLs with:
   - step_reference to link to research step
   - crawl_status (success/failed) and error_reason
   - is_cited flag to distinguish from cited sources

## Phase 1: Design Artifacts

See companion files:
- [data-model.md](./data-model.md) - Entity definitions and relationships
- [contracts/openapi-patch.yaml](./contracts/openapi-patch.yaml) - API endpoint additions
- [quickstart.md](./quickstart.md) - Developer setup guide

## Implementation Strategy

### Layer 1: Backend Foundation
1. Add QueryMode enum to models/enums.py
2. Create database migration for new tables and columns
3. Update ResearchState and OrchestrationConfig with query_mode field
4. Add query_mode parameter to research API endpoint
5. Add `query_modes` section to config/app.yaml

### Layer 2: Mode Routing Logic (~50 lines in orchestrator)
1. Add mode routing switch in `stream_research()`:
   - Simple: Skip to direct LLM response (existing `is_simple_query` path)
   - Web Search: Create 1-step plan, run researcher (classic mode), run synthesizer (natural mode)
   - Deep Research: Full pipeline unchanged
2. Configure researcher limits for Web Search: `max_search_queries=2`, `max_urls_to_crawl=3`
3. Configure synthesizer for Web Search: `generation_mode=natural`, skip verification stages
4. Create research_event_service for event persistence

### Layer 3: Frontend Mode Selection
1. Create QueryModeSelector component
2. Integrate with MessageInput and useStreamingQuery
3. Add QueryMode type to frontend types
4. Persist mode selection in user preferences

### Layer 4: Enhanced Activity Display
1. Create CenteredActivityPanel component
2. Implement ActivityAccordion for post-completion display
3. Create EnhancedEventLabel with context extraction
4. Add VirtualizedList for large event sets

### Layer 5: Visited Sources
1. Extend Source model with new fields (is_cited, step_index, crawl_status)
2. Extend tool_result handling to track all URLs
3. Create VisitedSourcesPanel component
4. Integrate with AgentMessage display

### Layer 6: Polish & Testing
1. Add animations for transitions (200ms events, 300ms accordion)
2. Implement sticky positioning for activity panel
3. Write comprehensive unit/integration/E2E tests
4. Performance optimization for <3s Simple mode TTFB

## Verify Sources Toggle & Snippet Fallback

**Implemented**: 2026-01-06 | **Related FRs**: FR-046 through FR-055

### Overview

User-controllable "Verify sources" checkbox that appears when Web Search or Deep Research mode is selected. When enabled, runs the full citation verification pipeline. When disabled, uses classical synthesis with `[Title](url)` style citations.

Additionally, Brave Search snippets serve as fallback evidence when web page fetching fails or was not performed.

### Implementation Details

#### Backend Parameter Plumbing

| File | Change |
|------|--------|
| `src/api/v1/research.py` | Added `verify_sources: bool = Query(default=True)` parameter to both GET/POST endpoints |
| `src/agent/orchestrator.py` | Added `verify_sources: bool = True` to `OrchestrationConfig` dataclass |
| `src/agent/orchestrator.py` | Pass `config.verify_sources` to `ResearchState.enable_citation_verification` |

```python
# API endpoint (research.py)
verify_sources: bool = Query(default=True, description="Enable citation verification pipeline"),

# OrchestrationConfig (orchestrator.py)
verify_sources: bool = True  # When False, use classical synthesis

# State initialization (orchestrator.py)
enable_citation_verification=config.verify_sources,
```

#### Frontend Checkbox UI

| File | Change |
|------|--------|
| `frontend/src/components/chat/MessageInput.tsx` | Added checkbox state, useEffect for mode-based defaults, conditional UI |
| `frontend/src/hooks/useStreamingQuery.ts` | Added `verifySources` param, append to URL as `&verify_sources=` |
| `frontend/src/pages/ChatPage.tsx` | Thread `verifySources` through sendQuery chain and router state |

```tsx
// MessageInput.tsx - State management
const [verifySources, setVerifySources] = React.useState<boolean>(false);

// Reset default when mode changes
React.useEffect(() => {
  setVerifySources(queryMode === 'deep_research');
}, [queryMode]);

// Show checkbox only for web_search or deep_research
const shouldShowVerifyCheckbox = queryMode === 'web_search' || queryMode === 'deep_research';
```

#### Snippet Fallback Support

| File | Change |
|------|--------|
| `src/services/citation/evidence_selector.py` | Use `snippet` as fallback when `content` unavailable; add `is_snippet_based: bool` to RankedEvidence |
| `src/services/citation/pipeline.py` | Include `snippet` in source_dicts; accept snippet-only sources in quality filtering |

```python
# evidence_selector.py - Snippet fallback with lower confidence
if not source_content and source_snippet:
    evidence = RankedEvidence(
        source_id=source_id,
        source_url=source_url,
        source_title=source_title,
        quote_text=source_snippet,
        relevance_score=0.5,  # Lower confidence for snippet-based evidence
        is_snippet_based=True,
    )
    all_evidence.append(evidence)
    continue

# pipeline.py - Include snippet-only sources
source_dicts = [
    {"url": source.url, "title": source.title, "content": source.content, "snippet": source.snippet}
    for source in sources
    if source.content or source.snippet  # Include snippet-only sources
]
```

### Behavior Matrix

| Mode | verify_sources | Synthesis Type | Evidence Source |
|------|----------------|----------------|-----------------|
| Simple | N/A (hidden) | Direct LLM | None |
| Web Search | false (default) | Classical `[Title](url)` | Full content (snippet fallback if fetch fails) |
| Web Search | true | Verification pipeline | Full content (snippet fallback if fetch fails) |
| Deep Research | true (default) | Verification pipeline | Full content (snippet fallback if fetch fails) |
| Deep Research | false | Classical `[Title](url)` | Full content (snippet fallback if fetch fails) |

**Note**: The researcher always attempts to fetch full page content regardless of `verify_sources`. The flag only controls whether the citation verification pipeline runs during synthesis. Snippets are used as fallback only when web fetch fails (blocked, timeout, parsing error) or returns empty content.

### Design Decisions

1. **Default values**: `false` for web_search (prioritize speed), `true` for deep_research (prioritize accuracy)
2. **Snippet confidence**: 0.5 relevance score for snippet-based evidence (vs calculated score for full content)
3. **No persistence**: `verify_sources` is per-query, not saved to user preferences
4. **Graceful fallback**: Snippets always available as evidence backup when web fetch fails

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Simple mode still slow | Precompute common queries, optimize LLM client initialization |
| Activity panel jank | Use CSS transforms, will-change hints, test on low-end devices |
| Event storage bloat | Implement event retention policy (delete with chat) |
| Web Search timeout | Aggressive 10s timeout with fallback to Simple mode |
| Migration complexity | Test migration on staging Lakebase before production |
| Snippet-only citations lack context | Lower confidence score (0.5), mark with `is_snippet_based` flag |

## Post-Design Constitution Check

*Re-verified after Phase 1 design completion.*

| Principle | Status | Post-Design Evidence |
|-----------|--------|---------------------|
| **I. Clients & Workspace Integration** | PASS | WebSearchAgent reuses existing LLM client pattern; no new direct API calls |
| **II. Typing-First Python** | PASS | data-model.md defines all new types with full annotations; QueryMode(str, Enum) pattern |
| **III. Avoid Runtime Introspection** | PASS | Mode routing uses explicit enum matching, not hasattr/isinstance; Pydantic validates payloads |
| **IV. Linting & Static Type Enforcement** | PASS | All new models follow existing mypy-compliant patterns; TypeScript interfaces defined |

**Final Gate Status**: PASSED - Ready for Phase 2 task generation via `/speckit.tasks`

## Deployment Architecture (2026-01-11)

### Databricks Apps + Lakebase Integration

The application deploys to Databricks Apps with Lakebase (managed PostgreSQL) for persistence.

#### Deployment Flow (8 Steps)

```bash
make deploy TARGET=dev BRAVE_SCOPE=msh
```

| Step | Description | Key Command |
|------|-------------|-------------|
| 1 | Build frontend | `npm run build` |
| 2 | Generate requirements.txt | `uv pip compile pyproject.toml` |
| 3 | Deploy with postgres DB (bootstrap) | `databricks bundle deploy --var lakebase_database=postgres` |
| 4 | Wait for Lakebase | `./scripts/wait-for-lakebase.sh` |
| 5 | Create deep_research database | `./scripts/create-database.sh` |
| 6 | Re-deploy with deep_research DB | `databricks bundle deploy --var lakebase_database=deep_research` |
| 7 | Run migrations | `uv run alembic upgrade head` |
| 8 | Grant table permissions | `./scripts/grant-app-permissions.sh` |
| 9 | Start app | `databricks bundle run` |

#### Two-Phase Deployment Rationale

**Problem**: Chicken-and-egg with Lakebase
- App config needs `LAKEBASE_DATABASE` pointing to `deep_research`
- But `deep_research` doesn't exist until Lakebase instance is ready
- Lakebase instance is created by bundle deploy

**Solution**: Bootstrap with `postgres` (always exists), then switch to `deep_research`

#### Permission Model

**Problem**: Table ownership mismatch
- Developer runs migrations → tables owned by developer's identity
- App's service principal has `CAN_CONNECT_AND_CREATE` on database
- `CAN_CONNECT_AND_CREATE` doesn't grant SELECT/INSERT/UPDATE/DELETE on tables

**Solution**: Post-migration GRANT statements via `src/db/grant_permissions.py`

```sql
GRANT ALL ON ALL TABLES IN SCHEMA public TO <app_service_principal>;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO <app_service_principal>;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO <app_service_principal>;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO <app_service_principal>;
```

#### Lakebase Authentication

**OAuth Token Flow**:
1. Developer authenticates via `DATABRICKS_CONFIG_PROFILE`
2. `WorkspaceClient.database.generate_database_credential()` returns OAuth token
3. Connect to Lakebase with `user="token"`, `password=<oauth_token>`
4. Tokens auto-refresh (1-hour lifetime, 5-minute buffer)

**Host Resolution**: `{LAKEBASE_INSTANCE_NAME}.database.cloud.databricks.com`

#### Key Deployment Files

| File | Purpose |
|------|---------|
| `databricks.yml` | DAB bundle configuration |
| `app.yaml` | Databricks Apps manifest |
| `Makefile` | Deploy target with 8-step pipeline |
| `src/db/grant_permissions.py` | Python module to grant table permissions |
| `src/db/lakebase_auth.py` | OAuth credential provider |
| `scripts/grant-app-permissions.sh` | Shell wrapper for permission grants |
| `scripts/create-database.sh` | Create deep_research database |
| `scripts/wait-for-lakebase.sh` | Wait for instance to be connectable |

#### Operations

```bash
# View logs (via /logz/batch REST API)
make logs TARGET=dev                         # Fetch logs once
make logs TARGET=dev FOLLOW=-f               # Follow logs in real-time
make logs TARGET=dev SEARCH="--search ERROR" # Filter by search term

# Restart app
databricks bundle run -t dev deep_research_agent

# Check status
databricks bundle summary -t dev

# Manual migrations
make db-migrate-remote TARGET=dev
```

#### Workspace Mapping

| TARGET | Profile | App Name | Lakebase Instance |
|--------|---------|----------|-------------------|
| dev | e2-demo-west | deep-research-agent-dre-dev | deep-research-lakebase-dre-dev |
| ais | ais | deep-research-agent-dre-ais | deep-research-lakebase-dre-ais |
