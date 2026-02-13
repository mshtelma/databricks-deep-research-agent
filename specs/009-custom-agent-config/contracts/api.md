# API Contracts: Custom Agent Configuration & Selection

**Feature**: 009-custom-agent-config
**Date**: 2026-02-09

## New Endpoints

### GET /api/v1/config/model-catalog

**Purpose**: Return the live endpoint catalog derived from YAML config (FR-010).

**Authentication**: Any authenticated user.

**Response** `200 OK`:

```json
{
  "categories": {
    "simple": {
      "name": "simple",
      "default_endpoints": ["databricks-haiku"],
      "temperature": 0.3,
      "max_tokens": 4000
    },
    "analytical": {
      "name": "analytical",
      "default_endpoints": ["databricks-sonnet", "databricks-llama-70b"],
      "temperature": 0.7,
      "max_tokens": 8000
    },
    "complex": {
      "name": "complex",
      "default_endpoints": ["databricks-opus"],
      "temperature": 0.7,
      "max_tokens": 16000
    }
  },
  "endpoints": {
    "databricks-haiku": {
      "name": "databricks-haiku",
      "endpoint_identifier": "databricks-meta-llama-3-1-8b-instruct",
      "max_context_window": 128000,
      "supports_structured_output": false
    },
    "databricks-sonnet": {
      "name": "databricks-sonnet",
      "endpoint_identifier": "databricks-claude-sonnet",
      "max_context_window": 200000,
      "supports_structured_output": true
    },
    "databricks-opus": {
      "name": "databricks-opus",
      "endpoint_identifier": "databricks-claude-opus",
      "max_context_window": 200000,
      "supports_structured_output": true
    }
  }
}
```

**Implementation notes**:
- Read-only, no DB involvement
- Derive from `get_app_config().models` and `get_app_config().endpoints`
- Cache at module level (config is immutable after load)
- Route: `src/deep_research/api/v1/config.py` (new router, prefix `/config`)

---

## Extended Endpoints

### POST /api/v1/custom-agents (extended request body)

**New fields** on `CreateCustomAgentRequest`:

```json
{
  "name": "Legal Research",
  "description": "...",
  "model_overrides": {
    "complex": "databricks-opus",
    "simple": "databricks-haiku"
  },
  "domain_filter_mode": "include",
  "include_domains": ["*.gov", "*.edu", "reuters.com"],
  "exclude_domains": null,
  "source_scope": "web_only",
  "system_prompt_template_id": "uuid-here",
  "synthesis_template_id": "uuid-here"
}
```

All new fields are optional. Omitting them preserves existing behavior.

**Validation**:
- `model_overrides`: keys must be valid tier names, values must be valid endpoint names (validated against live config)
- `domain_filter_mode`: if set, appropriate domain list must be non-empty
- Domain patterns: non-empty strings, valid domain characters + wildcards

---

### PATCH /api/v1/custom-agents/{agent_id} (extended request body)

Same new fields as POST, all optional. Setting `model_overrides` to `null` clears overrides. Setting `domain_filter_mode` to `null` clears domain filter.

---

### GET /api/v1/custom-agents/{agent_id} (extended response)

**New fields** on `CustomAgentResponse`:

```json
{
  "id": "uuid",
  "name": "Legal Research",
  "model_overrides": {
    "complex": "databricks-opus"
  },
  "domain_filter_mode": "include",
  "include_domains": ["*.gov", "*.edu"],
  "exclude_domains": null,
  "model_override_warnings": [
    {
      "tier": "complex",
      "endpoint": "databricks-opus-deprecated",
      "message": "Endpoint no longer available in system configuration"
    }
  ]
}
```

`model_override_warnings` is computed at response time by checking overrides against live `AppConfig.endpoints`. This array is empty when all overrides reference valid endpoints.

---

### GET /api/v1/custom-agents (extended summary)

**New fields** on `CustomAgentSummary`:

```json
{
  "id": "uuid",
  "name": "Legal Research",
  "has_model_overrides": true,
  "has_domain_filter": true,
  "has_source_config": true
}
```

Boolean flags for the listing UI to show configuration badges.

---

## Unchanged Endpoints

The following endpoints are NOT modified:

- `POST /api/v1/research/jobs` — already accepts `agent_id`. The backend resolution of agent config happens in `_run_job()`.
- `GET /api/v1/custom-agents/{agent_id}/steps` — preset steps already support `source_scope` field
- `POST/PATCH/DELETE /api/v1/custom-agents/{agent_id}/steps` — already support `source_scope`
- Template CRUD endpoints — no changes needed; the simplified inline creation uses the same API

---

## Frontend → Backend Contract

### Query Submission (unchanged)

```typescript
interface QuerySubmission {
  query: string;
  chatId: string;
  agentId?: string;       // Already exists
  // ... other fields
}
```

The frontend sends **only** `agentId`. The backend resolves the full agent config from DB at query time (per clarification: "Frontend sends agent_id only; backend resolves from DB").

### Agent Selection Persistence

The selected `agentId` is stored in `localStorage` under key `deep-research-selected-agent`. This is per-user, not per-chat (per spec assumption).

---

## Error Responses

### OBO Access Denied (FR-012a)

When a workspace agent references resources the user cannot access:

```json
{
  "error_code": "AGENT_RESOURCE_ACCESS_DENIED",
  "error_message": "You do not have access to endpoint 'databricks-opus'. Contact the agent owner or your workspace admin.",
  "details": {
    "agent_id": "uuid",
    "agent_name": "Company Legal Agent",
    "resource_type": "endpoint",
    "resource_name": "databricks-opus"
  }
}
```

This error surfaces at query execution time (when `call_llm()` or tool execution fails with a permission error), not at agent selection time.

### Stale Endpoint Warning (FR-011)

Not an error — included in `GET /custom-agents/{id}` response as `model_override_warnings` array. Frontend displays warning badges.

### Invalid Domain Pattern (FR-020)

```json
{
  "error_code": "VALIDATION_ERROR",
  "error_message": "Invalid domain pattern: '' (empty pattern not allowed)"
}
```

Standard 422 validation error on save.
