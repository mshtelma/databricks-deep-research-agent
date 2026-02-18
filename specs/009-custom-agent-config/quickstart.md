# Quickstart: Custom Agent Configuration & Selection

**Feature**: 009-custom-agent-config
**Date**: 2026-02-09

## Prerequisites

```bash
# Install dependencies
make install

# Start dev environment (backend + frontend)
make dev

# Run database migration for new columns
make db-migrate
```

## Implementation Order

The feature has 6 user stories with clear priority ordering. Implement in this sequence:

### Phase 1: Backend Wiring (P1 + P2 critical path)

1. **DB Migration** — Add `model_overrides`, `domain_filter_mode`, `include_domains`, `exclude_domains` to `custom_agents`
2. **Model + Schema** — Extend `CustomAgent` model and Pydantic schemas with new fields
3. **Endpoint Catalog API** — New `GET /api/v1/config/model-catalog` endpoint
4. **Agent Resolution** — Wire `apply_custom_agent_to_config()` call in `_run_job()`
5. **Model Overrides** — Extend `apply_custom_agent_to_config()` to handle model overrides + wire to LLM client

### Phase 2: Domain Filtering + Source Scope (P3 + P4)

6. **Domain Filter Wiring** — Extend `apply_custom_agent_to_config()` to handle domain filters
7. **Source Scope Hide** — Frontend hides per-query source selector when agent defines sources (FR-015a)

### Phase 3: Frontend Enhancements (P5 + P6)

8. **Model Config UI** — Agent editor "Model Configuration" section with endpoint dropdowns
9. **Domain Filter UI** — Agent editor "Web Domain Filtering" section
10. **Template Inline Creation** — "Create New..." flow in template dropdowns

## Key Files to Modify

### Backend

| File | What to change |
|------|---------------|
| `src/deep_research/models/custom_agent.py` | Add 4 new columns |
| `src/deep_research/schemas/custom_agent.py` | Add fields to request/response schemas |
| `src/deep_research/api/v1/custom_agents.py` | Handle new fields in CRUD endpoints |
| `src/deep_research/api/v1/config.py` | **NEW** — Endpoint catalog route |
| `src/deep_research/agent/orchestrator.py` | Extend `OrchestrationConfig`, `apply_custom_agent_to_config()` |
| `src/deep_research/agent/state.py` | Add `model_overrides` field |
| `src/deep_research/services/job_manager.py` | Call `apply_custom_agent_to_config()` in `_run_job()` |
| `src/deep_research/services/llm/client.py` | Accept `endpoint_override` parameter |
| `src/deep_research/main.py` | Register config router |
| `db/migrations/versions/017_*.py` | **NEW** — Migration for new columns |

### Frontend

| File | What to change |
|------|---------------|
| `frontend/src/types/customAgents.ts` | Add model_overrides, domain_filter types |
| `frontend/src/api/customAgents.ts` | Update API types |
| `frontend/src/api/config.ts` | **NEW** — Endpoint catalog API client |
| `frontend/src/hooks/useModelCatalog.ts` | **NEW** — Hook for endpoint catalog |
| `frontend/src/components/agents/ModelConfigSection.tsx` | **NEW** — Model override editor |
| `frontend/src/components/agents/DomainFilterSection.tsx` | **NEW** — Domain filter editor |
| `frontend/src/components/chat/MessageInput.tsx` | Hide source scope selector when agent defines sources |
| `frontend/src/components/chat/SourceScopeSelector.tsx` | Accept `hidden` prop |

## Running Tests

```bash
# Unit tests (fast, mocked)
uv run pytest tests/unit/ -v

# Specific test file
uv run pytest tests/unit/agent/test_enterprise_tools_wiring.py -v

# Type checking
make typecheck

# Lint
make lint
```

## Development Tips

1. **Start with `_run_job()`**: The single most important change is calling `apply_custom_agent_to_config()` in `_run_job()`. This activates the entire existing agent → config pipeline.

2. **Test model overrides with logging**: Before building UI, verify model overrides work by:
   - Creating an agent via API with `model_overrides: {"complex": "different-endpoint"}`
   - Running a query with that agent
   - Checking logs for the overridden endpoint being used

3. **Domain filters use existing infra**: `DomainFilterConfig` already exists in `app_config.py`. The agent version is the same structure, just sourced from DB instead of YAML.

4. **Frontend agent picker already works**: The dropdown in `MessageInput.tsx` already sends `agentId`. Focus on the agent editor enhancements, not the picker.

5. **Constitution compliance**: All new Python code needs type annotations. Use Pydantic for validation at boundaries. No `hasattr`/`isinstance` for type checks.
