# Phase 0 Research: Custom Agent Configuration & Selection

**Feature**: 009-custom-agent-config
**Date**: 2026-02-09
**Status**: Complete

## Research Questions & Findings

### RQ-1: Where does the agent_id get resolved to a full config?

**Finding**: The `agent_id` is already threaded from frontend through to the orchestrator but is **not resolved**. The full data flow:

1. `frontend/src/types/querySubmission.ts` — `QuerySubmission.agentId?: string`
2. `frontend/src/hooks/useStreamingQuery.ts` — passes `agentId` to `jobsApi.submit()`
3. `src/deep_research/api/v1/jobs.py:83` — `SubmitJobRequest.agent_id: str | None`
4. `src/deep_research/services/job_manager.py:169` — `submit_job(agent_id=)` → `_run_job(agent_id=)`
5. `_run_job()` creates `OrchestrationConfig(agent_id=agent_id)` (line ~540)
6. `stream_research()` at line 1400: `state.agent_id = config.agent_id` — stores ID on state, **does nothing with it**

**Gap**: `apply_custom_agent_to_config()` exists at `orchestrator.py:387-481` but is **never called** anywhere. It needs to be invoked in `_run_job()` after constructing `OrchestrationConfig` but **before** calling `stream_research()`. The function must:
- Fetch the `CustomAgent` from DB using agent_id
- Call `apply_custom_agent_to_config(config, agent)` to merge agent settings
- Handle model overrides (new) and domain filters (new)

**Resolution**: Call `apply_custom_agent_to_config()` in `_run_job()` between config construction and `stream_research()` invocation. The function already handles source scope, workflow mode, depth, output format, and preset steps.

---

### RQ-2: How are model overrides applied in the LLM client?

**Finding**: The LLM model selection happens via `app_config.py` model tiers:

- `AppConfig.models: dict[str, ModelRoleConfig]` — keys are tier names: `"simple"`, `"analytical"`, `"complex"`
- `ModelRoleConfig.endpoints: list[str]` — references keys in `AppConfig.endpoints`
- `EndpointConfig.endpoint_identifier: str` — the actual Databricks serving endpoint name
- Agent code calls tiers via `get_model_config("analytical")` which returns the `ModelRoleConfig`
- The LLM client's `call_llm()` selects an endpoint from the tier's endpoint list using `rotation_strategy`

**Model override approach**: Agent model overrides should be a `dict[str, str]` mapping tier name to endpoint name (e.g., `{"complex": "databricks-opus", "simple": "databricks-haiku"}`). At runtime, when the agent is applied:
1. For each tier with an override, create a temporary `ModelRoleConfig` with the single overridden endpoint
2. Store these in `OrchestrationConfig` as `model_overrides: dict[str, str] | None`
3. In `stream_research()`, if model overrides exist, patch the LLM client's model resolution

**Simpler approach (recommended)**: Since `OrchestrationConfig` is passed to `stream_research()` which creates `ResearchState`, add `model_overrides` to both. The LLM client already accepts an `endpoint_name` override parameter — we just need to thread the override map through to each `call_llm()` site.

Actually, examining the code more carefully: the LLM client's `call_llm(role=)` parameter selects the model tier. The cleanest approach is to:
1. Store `model_overrides: dict[str, str] | None` on `OrchestrationConfig`
2. Wire to `ResearchState.model_overrides`
3. In `apply_custom_agent_to_config()`, read agent's model_overrides JSONB and set on config
4. In `stream_research()`, if overrides exist, create a patched app config or pass overrides to LLM calls

**Decision**: Use `OrchestrationConfig.model_overrides` → `ResearchState.model_overrides`. The LLM client needs a single new parameter `endpoint_override: str | None` that bypasses tier-based selection when set.

---

### RQ-3: How does domain filtering currently work?

**Finding**: Domain filtering is system-wide via YAML:

```yaml
search:
  domain_filter:
    mode: exclude          # include | exclude | both
    include_domains: []
    exclude_domains: ["*.ru", "*.cn"]
```

- `DomainFilterConfig` at `app_config.py:225-245` — mode, include_domains, exclude_domains
- `SearchConfig.domain_filter` holds the config
- Filtering is applied in `BraveSearchClient` at search time
- The `BraveSearchConfig` has a `freshness` field but domain filtering is in `DomainFilterConfig`

**Agent override approach**:
1. Add `domain_filter_mode`, `include_domains`, `exclude_domains` columns to `CustomAgent` model
2. Wire through `OrchestrationConfig.domain_filter: DomainFilterConfig | None`
3. In `stream_research()`, if agent domain filter is set, override the system-wide filter
4. The override is applied when constructing search queries — pass the filter config through to `BraveSearchClient`

**Important**: Per the spec, agent domain filters **replace** (not merge with) system-wide filters.

---

### RQ-4: What frontend components exist for agent selection?

**Finding**: The agent selection UI already exists in `MessageInput.tsx`:

- Lines 156-168: `useCustomAgents()` hook fetches agents
- Lines 409-469: Agent picker dropdown with "Default" + user agents + workspace agents
- `useStreamingQuery.ts`: Already extracts `agentId` from `QuerySubmission` and passes to API

**Gaps**:
1. No "Model Configuration" section in the agent editor
2. No "Domain Filtering" section in the agent editor
3. No endpoint catalog API for populating model dropdowns
4. Source scope selector doesn't hide when agent defines sources (FR-015a)
5. No inline template creation from dropdowns

**Existing components to extend**:
- `frontend/src/components/agents/` — agent editor components
- `frontend/src/hooks/useCustomAgents.ts` — agent CRUD hooks
- `frontend/src/types/customAgents.ts` — TypeScript types

---

### RQ-5: How should the endpoint catalog API work?

**Finding**: `AppConfig.endpoints` and `AppConfig.models` already contain all needed data:

```python
# app_config.py
class AppConfig:
    endpoints: dict[str, EndpointConfig]  # endpoint_name -> config
    models: dict[str, ModelRoleConfig]    # tier_name -> config (tier.endpoints = [endpoint_names])
```

The catalog API should return:
```json
{
  "model_categories": {
    "simple": {
      "default_endpoints": ["databricks-haiku"],
      "description": "Fast, simple tasks"
    },
    "analytical": {
      "default_endpoints": ["databricks-sonnet"],
      "description": "Balanced quality"
    },
    "complex": {
      "default_endpoints": ["databricks-opus"],
      "description": "High-quality reasoning"
    }
  },
  "available_endpoints": {
    "databricks-haiku": { "endpoint_identifier": "databricks-meta-llama-3-1-8b", "max_context_window": 128000 },
    "databricks-sonnet": { "endpoint_identifier": "databricks-claude-sonnet", "max_context_window": 200000 },
    "databricks-opus": { "endpoint_identifier": "databricks-claude-opus", "max_context_window": 200000 }
  }
}
```

This is a read-only endpoint that derives from the loaded YAML config at runtime. No database involvement.

---

### RQ-6: What's the migration strategy for new DB columns?

**Finding**: Per user clarification, the application is not yet live. Database schema changes can be applied via migration recreation — no backward-compatible migration constraints.

**New columns on `custom_agents` table**:
- `model_overrides: JSONB | null` — `{"tier_name": "endpoint_name", ...}`
- `domain_filter_mode: VARCHAR(20) | null` — `"include"`, `"exclude"`, `"both"`, or null (no filter)
- `include_domains: JSONB | null` — `["*.gov", "reuters.com"]`
- `exclude_domains: JSONB | null` — `["*.ru"]`

**Approach**: Create migration `017_custom_agent_model_overrides.py` that adds these four columns.

---

### RQ-7: How does `apply_custom_agent_to_config()` need to be extended?

**Finding**: The existing function at `orchestrator.py:387-481` handles:
- source_scope, enabled_sources, disabled_sources
- research_depth, workflow_mode, enable_clarification
- output_format, output_schema
- Preset steps (manual/hybrid mode)
- System instructions (noted as "handled at a higher level")

**Missing from `apply_custom_agent_to_config()`**:
1. Model overrides — read `agent.model_overrides` JSONB, validate against live endpoints, set on config
2. Domain filters — read `agent.domain_filter_mode` + `agent.include_domains` + `agent.exclude_domains`, construct `DomainFilterConfig`, set on config
3. Template rendering — read template content from `agent.system_prompt_template` relationship, set as `config.system_instructions`

**Validation needed at application time**:
- For model overrides: check each referenced endpoint exists in `AppConfig.endpoints`. If missing, log warning and skip that override (use system default for that tier).
- For domain filters: construct `DomainFilterConfig` from agent columns. If mode is null, no override.
- For OBO errors: if workspace agent references inaccessible resources, the OBO token will fail at tool execution time — the error message needs to identify the specific resource.

---

## Unknowns Resolved

| Unknown | Resolution |
|---------|-----------|
| JSONB shape for model_overrides | `{"tier_name": "endpoint_name"}` — flat dict, validated at apply time |
| Where to call apply_custom_agent_to_config | In `_run_job()` after config construction, before `stream_research()` |
| How model overrides propagate to LLM calls | Via `OrchestrationConfig.model_overrides` → `ResearchState.model_overrides` → passed to LLM client per-call |
| Domain filter merge semantics | Agent replaces system-wide (no merge). Null agent filter = use system default |
| Endpoint catalog source | Read-only derivation from `AppConfig.endpoints` + `AppConfig.models` |
| Migration strategy | Recreate migration (not live yet) |
| Frontend agent selector behavior | Already exists in `MessageInput.tsx`, needs extension |

## Open Decisions

None — all unknowns resolved during research.
