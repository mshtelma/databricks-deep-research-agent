# Implementation Plan: Custom Agent Configuration & Selection

**Branch**: `009-custom-agent-config` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/009-custom-agent-config/spec.md`

## Summary

Extend the existing custom agent system with per-agent model overrides (mapping tier names to live endpoints), per-agent web domain whitelists/blacklists, and wire the already-existing but uncalled `apply_custom_agent_to_config()` function so that selecting an agent actually applies its configuration. Add an endpoint catalog API for frontend model dropdowns, simplified inline prompt template creation, and hide the per-query source scope selector when the agent defines sources.

The critical insight from research: the agent_id is already threaded from frontend → API → job_manager → orchestrator → state, but `apply_custom_agent_to_config()` is **never called**. The primary backend task is wiring this call in `_run_job()` and extending it with model override + domain filter logic. Most frontend infrastructure (agent picker, agent editor) already exists.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI 0.109+, SQLAlchemy (async), React 18, TanStack Query 5.x, Pydantic 2.x
**Storage**: Databricks Lakebase (PostgreSQL) — not yet live, schema freely modifiable
**Testing**: pytest (unit), Playwright (e2e), Vitest (frontend)
**Target Platform**: Databricks Apps (Linux server), browser frontend
**Project Type**: Web application (Python backend + React frontend)
**Performance Goals**: Agent selector opens in <2s (SC-001), model catalog loads in <3s (SC-002)
**Constraints**: All LLM calls via OpenAI client through WorkspaceClient; mypy strict + ruff pass
**Scale/Scope**: Single workspace deployment; ~50 concurrent users; <100 custom agents per workspace

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Clients and Workspace Integration | PASS | LLM calls remain through OpenAI client via WorkspaceClient. No new direct API calls. |
| II. Typing-First Python | PASS | All new functions will have full type annotations. Model overrides typed as `dict[str, str]`. JSONB shapes validated via Pydantic. |
| III. Avoid Runtime Introspection | PASS | No `hasattr`/`isinstance` for type safety. Agent config validated through Pydantic models. Endpoint existence checked via dict lookup. |
| IV. Linting and Static Type Enforcement | PASS | mypy strict + ruff must pass. New schemas use Pydantic BaseModel. |

**Post-Phase-1 re-check**: No violations identified. The design uses:
- Pydantic schemas for all API boundaries (Principle III)
- Typed dataclass fields on `OrchestrationConfig` (Principle II)
- Standard SQLAlchemy columns with proper type annotations (Principle II)
- No new `WorkspaceClient` calls needed — model catalog reads from already-loaded YAML config (Principle I)

## Project Structure

### Documentation (this feature)

```text
specs/009-custom-agent-config/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research findings
├── data-model.md        # Phase 1 entity + schema design
├── quickstart.md        # Phase 1 development guide
├── contracts/
│   └── api.md           # Phase 1 API contract definitions
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/deep_research/
├── agent/
│   ├── orchestrator.py          # Extend OrchestrationConfig + apply_custom_agent_to_config()
│   └── state.py                 # Add model_overrides field
├── api/v1/
│   ├── config.py                # NEW — Endpoint catalog route
│   └── custom_agents.py         # Extend CRUD with new fields
├── models/
│   └── custom_agent.py          # Add 4 new columns
├── schemas/
│   ├── custom_agent.py          # Extend request/response schemas
│   └── config.py                # NEW — Endpoint catalog response schemas
├── services/
│   ├── job_manager.py           # Wire apply_custom_agent_to_config() in _run_job()
│   └── llm/client.py            # Accept endpoint_override parameter
├── db/migrations/versions/
│   └── 017_custom_agent_model_overrides.py  # NEW — Migration
└── main.py                      # Register config router

frontend/src/
├── api/
│   └── config.ts                # NEW — Endpoint catalog API client
├── hooks/
│   └── useModelCatalog.ts       # NEW — Hook for endpoint catalog
├── components/
│   ├── agents/
│   │   ├── ModelConfigSection.tsx    # NEW — Model override editor
│   │   └── DomainFilterSection.tsx   # NEW — Domain filter editor
│   └── chat/
│       ├── MessageInput.tsx          # Hide source selector when agent defines sources
│       └── SourceScopeSelector.tsx   # Accept hidden prop
└── types/
    └── customAgents.ts               # Add model_overrides, domain_filter types

tests/
├── unit/
│   ├── agent/
│   │   └── test_agent_config_apply.py    # NEW — apply_custom_agent_to_config tests
│   ├── api/
│   │   └── test_config_endpoint.py       # NEW — Endpoint catalog tests
│   └── schemas/
│       └── test_custom_agent_schema.py   # Extend with new fields
└── integration/
    └── test_agent_model_override.py      # NEW — End-to-end model override test
```

**Structure Decision**: Web application pattern — Python backend under `src/deep_research/`, React frontend under `frontend/src/`, tests under `tests/`. This matches the existing repository structure exactly.

## Complexity Tracking

No constitution violations to justify. The design stays within existing patterns:
- JSONB columns follow the same pattern as `enabled_sources`/`disabled_sources`
- New API endpoint follows the same pattern as other `api/v1/` routers
- `OrchestrationConfig` extension follows the same dataclass field pattern
- Frontend components follow the existing agent editor component structure

## Phase Summary

| Phase | Deliverable | Depends On |
|-------|------------|------------|
| Phase 0 | [research.md](./research.md) — Integration point analysis | Spec |
| Phase 1a | [data-model.md](./data-model.md) — Entity extensions | Research |
| Phase 1b | [contracts/api.md](./contracts/api.md) — API contracts | Research |
| Phase 1c | [quickstart.md](./quickstart.md) — Dev guide | Data model + Contracts |
| Phase 2 | tasks.md — Implementation tasks | All Phase 1 |

## Critical Path

```
DB Migration (T1)
  → Model + Schema Extensions (T2)
    → Agent Resolution in _run_job (T3)     ← THIS IS THE UNLOCK
      → Model Override Wiring (T4)
        → Domain Filter Wiring (T5)
          → Frontend Model Config UI (T6)
            → Frontend Domain Filter UI (T7)

Parallel:
  Endpoint Catalog API (T8) — independent, feeds T6
  Source Scope Selector Hide (T9) — independent, frontend only
  Inline Template Creation (T10) — independent, frontend only
```

**T3 (Agent Resolution) is the critical unlock**: Once `apply_custom_agent_to_config()` is called in `_run_job()`, all existing agent config fields (source_scope, enabled_sources, workflow_mode, depth, output_format, preset_steps) become functional. Everything before T3 is schema/model work; everything after adds the new capabilities (model overrides, domain filters, UI).

---

## Adversarial Review & Corrections

**Date**: 2026-02-10 | **Status**: Validated against actual codebase

This section documents bugs and gaps found by adversarial validation of the original tasks.md against actual code. Each finding includes severity, what the original plan assumed, what the codebase actually shows, and the corrected approach.

### BUG 1 (HIGH): Domain filter wiring underestimated — 4 layers of plumbing needed

**Original plan** (T034-T035): "wire config.domain_filter to search client" — implies simple pass-through.

**Reality**: `BraveSearchClient` creates `DomainFilter` in its **constructor** (`brave.py:78`). The `search()` method has NO `domain_filter` parameter. The client is a singleton passed through `_run_job()` → `stream_research()` → researcher nodes. There are 5 `web_search()` call sites across 4 node files.

**Fix**: Must add `domain_filter: DomainFilter | None = None` to:
1. `BraveSearchClient.search()` — use override instead of `self._domain_filter` when provided
2. `web_search()` function — pass through to `brave_client.search()`
3. Each of 5 call sites in researcher/background nodes — pass `state.domain_filter`

### BUG 2 (HIGH): LLM client needs override in 3 methods, not 2

**Original plan** (T022): Add `endpoint_override` to `complete()` and `stream()`.

**Reality**: `stream_with_tools()` at `client.py:1267` also calls `_select_endpoint()` at line 1377. ReAct researcher and coordinator use this method. Total: **3 public methods + 3 `_impl` methods** need the parameter.

**Fix**: Add `endpoint_override: str | None = None` to `complete()`, `stream()`, `stream_with_tools()` and their corresponding `_*_impl()` methods. In each `_impl`, bypass `_select_endpoint()` when override is set using existing `ModelConfig.get_endpoint(id)` at `config.py:71`.

### BUG 3 (MEDIUM): Model override threading hits 15+ call sites

**Original plan** (T023): "Thread model_overrides to LLM calls in 6 node files."

**Reality**: There are 15+ `llm.complete()`/`llm.stream()`/`llm.stream_with_tools()` call sites across 10 files (background, planner, researcher, react_researcher, reflector, synthesizer, citation_synthesizer, coordinator, react_synthesizer, custom_phase_executor).

**Fix**: Use helper `get_endpoint_override(state, tier) -> str | None` in `agent/config.py`. Each call site adds one kwarg. Critical path is 10 sites in 6 files (planner, researcher, react_researcher, reflector, synthesizer, citation_synthesizer).

### BUG 4 (HIGH): Agent resolution needs separate DB session with correct timing

**Original plan** (T006): "Use an independent DB session" — correct idea but ambiguous about timing.

**Reality**: In `_run_job()`, line 554 opens a session for `stream_research()`. Agent resolution must happen AFTER config construction (line 551) but BEFORE `stream_research()` (line 556). The agent session must be opened and **closed** independently.

**Fix**: Open/close a separate session for agent loading between lines 551-554:
```python
if agent_id:
    async with get_session_maker()() as agent_db:
        agent_svc = CustomAgentService(agent_db)
        agent = await agent_svc.get_accessible(UUID(agent_id), user_id)
        if agent:
            config = apply_custom_agent_to_config(config, agent)
```

### BUG 5 (LOW): `apply_custom_agent_to_config` takes `agent: Any`

**Reality**: Function signature at `orchestrator.py:389` uses `agent: Any`. Violates Constitution Principle II (typing-first).

**Fix**: Change to `agent: "CustomAgent"` with `TYPE_CHECKING` import guard.

### BUG 6 (LOW): SourceScopeSelector location confirmed, no `hidden` prop needed

**Original plan** (T031): Suggests adding `hidden` prop to SourceScopeSelector.

**Reality**: Component is at `frontend/src/components/research/SourceScopeSelector.tsx`. `MessageInput.tsx` already uses conditional rendering via `shouldShowSourceScope` flag at line 93. Just modify the flag.

**Fix**: Delete T031. Modify `shouldShowSourceScope` in `MessageInput.tsx` to check `selectedAgent.has_source_config`.

### BUG 7 (MEDIUM): Frontend `CustomAgentSummary` type doesn't include source info

**Original plan** (T030): "check if selected agent has `source_scope != 'all'`".

**Reality**: Frontend `CustomAgentSummary` type (`customAgents.ts:77-87`) has `capabilities: AgentCapability[]` but NO `source_scope` or `has_source_config`. Backend schema already has `source_scope` on summary. Types are misaligned.

**Fix**: Add `has_source_config?: boolean` to frontend `CustomAgentSummary` type. Backend computes: `bool(agent.source_scope and agent.source_scope != "all") or bool(agent.enabled_sources)`.

### BUG 8 (NON-ISSUE): Endpoint lookup already exists

**Original concern**: `_select_endpoint` might not support name-based lookup.

**Reality**: `ModelConfig.get_endpoint(endpoint_id)` at `config.py:71` returns `ModelEndpoint` by string ID. Works for the override mechanism.

---

## Corrected Implementation Details

### Change A: Agent Resolution in `_run_job()` (T006, corrected)

**File**: `src/deep_research/services/job_manager.py` (after line 551, before line 554)

```python
# --- Agent resolution (separate DB session) ---
if agent_id:
    from deep_research.services.custom_agent_service import CustomAgentService
    agent_session_maker = get_session_maker()
    async with agent_session_maker() as agent_db:
        agent_service = CustomAgentService(agent_db)
        agent = await agent_service.get_accessible(UUID(agent_id), user_id)
        if agent:
            config = apply_custom_agent_to_config(config, agent)
            # Wire template content to config
            if agent.system_prompt_template and agent.system_prompt_template.content:
                config.system_instructions = agent.system_prompt_template.content
            if agent.synthesis_template and agent.synthesis_template.content:
                config.structured_system_prompt = agent.synthesis_template.content
            logger.info("JOB_AGENT_CONFIG_APPLIED", agent_id=agent_id, agent_name=agent.name)
        else:
            logger.warning("JOB_AGENT_NOT_FOUND", agent_id=agent_id, user_id=user_id)
# Session is closed here, before stream_research opens its own
```

Import `apply_custom_agent_to_config` from `orchestrator.py`. The agent session MUST close before `stream_research()` starts its own session at line 555.

### Change B: LLM Client `endpoint_override` (T022, corrected)

**File**: `src/deep_research/services/llm/client.py`

Add `endpoint_override: str | None = None` to these 6 methods:
- `complete()` → pass to `_complete_impl()`
- `_complete_impl()` → override logic before `_select_endpoint()`
- `stream()` → pass to `_stream_impl()`
- `_stream_impl()` → override logic
- `stream_with_tools()` → pass to `_stream_with_tools_impl()`
- `_stream_with_tools_impl()` → override logic

Override logic in each `_impl` (replaces `_select_endpoint` call):
```python
if endpoint_override:
    try:
        endpoint = self._config.get_endpoint(endpoint_override)
        health = self._get_health(endpoint.id)
    except (ValueError, KeyError):
        logger.warning("ENDPOINT_OVERRIDE_NOT_FOUND", override=endpoint_override, tier=tier.value)
        endpoint, health = self._select_endpoint(role, estimated_tokens)
else:
    endpoint, health = self._select_endpoint(role, estimated_tokens)
```

On rate limit: retry loop retries the SAME override endpoint (correct — user chose it deliberately).

### Change C: Model Override Helper + Node Threading (T023, corrected)

**File**: `src/deep_research/agent/config.py` (new helper)

```python
def get_endpoint_override(state: ResearchState, tier: ModelTier) -> str | None:
    """Get endpoint override for a tier from state's model_overrides."""
    if not state.model_overrides:
        return None
    return state.model_overrides.get(tier.value)
```

Then at each LLM call site, add the kwarg. Example:
```python
# Before:
response = await llm.complete(messages=messages, tier=ModelTier.ANALYTICAL)
# After:
response = await llm.complete(
    messages=messages, tier=ModelTier.ANALYTICAL,
    endpoint_override=get_endpoint_override(state, ModelTier.ANALYTICAL),
)
```

**Critical call sites** (10 in 6 files):

| File | Tier | Method |
|------|------|--------|
| `planner.py` | ANALYTICAL | complete (2 sites) |
| `researcher.py` | ANALYTICAL, FAST | complete (2 sites) |
| `react_researcher.py` | ANALYTICAL | stream_with_tools (1 site) |
| `reflector.py` | BULK_ANALYSIS | complete (1 site) |
| `synthesizer.py` | COMPLEX | complete, stream (3 sites) |
| `citation_synthesizer.py` | COMPLEX | stream (1 site) |

### Change D: Domain Filter Plumbing (T034-T035, corrected)

**4-layer change:**

**Layer 1**: `BraveSearchClient.search()` at `brave.py:100` — add `domain_filter: DomainFilter | None = None` param. Use `domain_filter if domain_filter is not None else self._domain_filter` when applying filtering.

**Layer 2**: `web_search()` function at `web_search.py:55` — add `domain_filter: DomainFilter | None = None` param, pass through to `client.search()`.

**Layer 3**: `ResearchState` at `state.py` — add `domain_filter: DomainFilterConfig | None = None` field.

**Layer 4**: 5 call sites construct `DomainFilter(state.domain_filter)` once per function and pass to `web_search()`:
- `researcher.py:334`
- `react_researcher.py:702`
- `background.py:168`
- `background.py:661`
- `custom_phase_executor.py:139`

### Change E: Source Scope Selector Hiding (T030, corrected)

**File**: `frontend/src/components/chat/MessageInput.tsx`

Modify `shouldShowSourceScope` at line 93:
```typescript
const agentDefinesSources = selectedAgent?.has_source_config === true;
const shouldShowSourceScope =
  (queryMode === 'deep_research' || queryMode === 'web_search') && !agentDefinesSources;
```

Requires: `has_source_config?: boolean` on frontend `CustomAgentSummary` type (added in T003/T028).

**Delete T031** — no `hidden` prop needed.

---

## Complete Files Modified List

| File | Change |
|------|--------|
| `src/deep_research/db/migrations/versions/017_*.py` | **NEW** — 4 columns |
| `src/deep_research/models/custom_agent.py` | 4 new columns |
| `src/deep_research/schemas/custom_agent.py` | New fields on all schemas + `has_source_config` |
| `src/deep_research/schemas/config.py` | **NEW** — EndpointCatalogResponse |
| `src/deep_research/agent/orchestrator.py` | OrchestrationConfig fields, fix `agent: Any` → `"CustomAgent"`, extend `apply_custom_agent_to_config()` |
| `src/deep_research/agent/state.py` | `model_overrides` + `domain_filter` fields |
| `src/deep_research/agent/config.py` | `get_endpoint_override()` helper |
| `src/deep_research/services/job_manager.py` | Wire agent resolution in `_run_job()` |
| `src/deep_research/services/llm/client.py` | `endpoint_override` on 6 methods |
| `src/deep_research/services/search/brave.py` | `domain_filter` param on `search()` |
| `src/deep_research/agent/tools/web_search.py` | `domain_filter` param on `web_search()` |
| `src/deep_research/agent/nodes/planner.py` | endpoint_override kwarg (2 sites) |
| `src/deep_research/agent/nodes/researcher.py` | endpoint_override + domain_filter (3 sites) |
| `src/deep_research/agent/nodes/react_researcher.py` | endpoint_override + domain_filter (2 sites) |
| `src/deep_research/agent/nodes/reflector.py` | endpoint_override kwarg (1 site) |
| `src/deep_research/agent/nodes/synthesizer.py` | endpoint_override kwarg (3 sites) |
| `src/deep_research/agent/nodes/citation_synthesizer.py` | endpoint_override kwarg (1 site) |
| `src/deep_research/agent/nodes/background.py` | domain_filter (2 sites) |
| `src/deep_research/agent/nodes/custom_phase_executor.py` | domain_filter (1 site) |
| `src/deep_research/api/v1/config.py` | **NEW** — model-catalog endpoint |
| `src/deep_research/api/v1/custom_agents.py` | CRUD for new fields |
| `src/deep_research/api/v1/__init__.py` | Register config router |
| `src/deep_research/services/custom_agent_service.py` | New field persistence |
| `src/deep_research/main.py` | Register config router |
| `frontend/src/types/customAgents.ts` | model_overrides, domain_filter, has_source_config |
| `frontend/src/components/chat/MessageInput.tsx` | Source scope hiding logic |
| `frontend/src/api/config.ts` | **NEW** — Catalog API client |
| `frontend/src/hooks/useModelCatalog.ts` | **NEW** — Catalog hook |
| `frontend/src/components/agents/ModelConfigSection.tsx` | **NEW** — Model override editor |
| `frontend/src/components/agents/DomainFilterSection.tsx` | **NEW** — Domain filter editor |
| `tests/unit/agent/test_agent_config_apply.py` | **NEW** — apply_custom_agent_to_config tests |
| `tests/unit/api/test_config_endpoint.py` | **NEW** — Catalog endpoint tests |

---

## Revised Execution Strategy

### MVP (Phase 1-2): Wire the pipeline — immediate value

Complete tasks T001-T010. This activates ALL existing agent config fields (source_scope, workflow_mode, depth, output_format, preset_steps) that have been configurable but non-functional. Zero UI changes needed.

### Then parallel tracks:
- **US2 (Model Overrides)**: T022 is highest-risk (LLM client changes). Do this next.
- **US3 (Source Scope Hide)**: Simple frontend change, can be quick win.
- **US4 (Domain Filters)**: T034 is second-highest risk (BraveSearchClient changes).

### Risk ordering:
1. **T022 (LLM client)** — affects every LLM call. Must be backward-compatible. 6 method signatures change.
2. **T034 (domain filter)** — 5 call sites plus BraveSearchClient. Must be backward-compatible.
3. **T006 (agent resolution)** — separate DB session timing. Pattern is established.
4. **T023 (model override threading)** — mechanical, tests catch misses. 10+ sites.

---

## Verification Plan

```bash
# Phase 1-2: Core wiring
uv run pytest tests/unit/agent/test_agent_config_apply.py -v
uv run pytest tests/unit/services/test_job_manager_agent.py -v

# Phase 4: Model overrides
uv run pytest tests/unit/ -v  # Full regression (LLM client changes affect everything)

# Phase 6: Domain filters
uv run pytest tests/unit/agent/ -v

# All phases: Type safety
make typecheck
make lint

# End-to-end: Create agent via API → select → run query → check logs
# Verify: JOB_AGENT_CONFIG_APPLIED, AGENT_MODEL_OVERRIDE_APPLIED, AGENT_DOMAIN_FILTER_APPLIED
```

### Acceptance Criteria
- [ ] Selecting an agent with `source_scope=enterprise_only` prevents web searches
- [ ] Agent with `model_overrides: {"complex": "databricks-haiku"}` uses haiku for synthesis (check logs)
- [ ] Agent with `domain_filter_mode=include, include_domains=["*.gov"]` filters search results
- [ ] Source scope selector hides when agent defines sources
- [ ] `GET /api/v1/config/model-catalog` returns categories + endpoints
- [ ] All existing tests pass (backward compatibility)
- [ ] `make typecheck && make lint` pass
