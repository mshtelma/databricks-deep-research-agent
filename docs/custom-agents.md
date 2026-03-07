# Custom Agents

## Overview

Custom agents are specialized research profiles that bundle model, source, prompt, and workflow configuration into a reusable preset. Instead of adjusting settings before every query, users create an agent once and select it from a dropdown — all configured preferences apply automatically.

A custom agent can control:

- **Model routing** — override which LLM endpoints power each tier (simple/analytical/complex)
- **Source scope** — restrict research to web only, enterprise only, or both
- **Domain filtering** — whitelist or blacklist web domains for search
- **Workflow mode** — use the AI planner, define manual steps, or combine both
- **Prompt templates** — attach custom system and synthesis prompts
- **Research depth** — set default depth (light/medium/extended)

## Quick Start

### 1. Create an Agent

```http
POST /v1/custom-agents
Content-Type: application/json

{
  "name": "Security Researcher",
  "description": "Focused on cybersecurity topics with .gov and .edu sources",
  "default_depth": "extended",
  "source_scope": "web_only",
  "domain_filter_mode": "include",
  "include_domains": ["*.gov", "*.edu", "*.mil", "cve.mitre.org"]
}
```

### 2. Select the Agent

In the chat UI, open the agent selector dropdown and choose "Security Researcher". The selection persists across sessions.

### 3. Run a Query

Submit any research query. The agent's configuration is applied automatically — all searches are restricted to `.gov`, `.edu`, `.mil`, and `cve.mitre.org` domains at extended depth.

## Agent Configuration

### Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Display name (unique per owner, max 255 chars) |
| `description` | string | null | Human-readable description (max 5000 chars) |
| `avatar_url` | string | null | URL for agent avatar image |
| `visibility` | enum | `private` | Who can see/use this agent |
| `default_depth` | enum | `medium` | Default research depth |
| `default_mode` | enum | `planner` | Workflow mode |
| `use_planner` | bool | true | Whether to use AI planner for step generation |
| `enable_clarification` | bool | true | Whether to ask clarifying questions |
| `output_format` | enum | `markdown` | Default output format (`markdown` or `json`) |
| `output_schema` | object | null | JSON Schema for structured output (when format is `json`) |

### Source Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_scope` | enum | `all` | Source scope: `all`, `enterprise_only`, `web_only` |
| `enabled_sources` | string[] | null | Explicit list of source names to enable |
| `disabled_sources` | string[] | `[]` | List of source names to disable |
| `source_query_configs` | object | null | Per-source query configuration overrides |

### Model Override Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_overrides` | object | null | Mapping of tier name to endpoint identifier |

### Domain Filtering Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domain_filter_mode` | string | null | Filter mode: `include`, `exclude`, or `both` |
| `include_domains` | string[] | null | Domain whitelist patterns |
| `exclude_domains` | string[] | null | Domain blacklist patterns |

### Template Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt_template_id` | UUID | null | Template for system instructions |
| `synthesis_template_id` | UUID | null | Template for synthesis prompt |

## Visibility Levels

| Level | Access Rules |
|-------|-------------|
| `private` | Only the creator can see and use the agent |
| `workspace` | All workspace users can see and use the agent |
| `system` | System-provided agents visible to everyone (admin-managed) |

When listing agents, the response includes counts by category:

```json
{
  "agents": [...],
  "total": 12,
  "user_agents": 5,
  "workspace_agents": 4,
  "system_agents": 3
}
```

## Model Overrides

Model overrides let an agent route specific tiers to different endpoints. The override maps a tier name (`simple`, `analytical`, `complex`) to a serving endpoint identifier.

```json
{
  "model_overrides": {
    "complex": "databricks-claude-sonnet-er",
    "analytical": "databricks-gemini-flash"
  }
}
```

### Stale Endpoint Warnings

When an agent references an endpoint that no longer exists in the system configuration, the response includes warnings:

```json
{
  "model_override_warnings": [
    {
      "tier": "complex",
      "endpoint": "removed-endpoint",
      "message": "Endpoint 'removed-endpoint' not found in current configuration"
    }
  ]
}
```

The system falls back to the default endpoint for that tier when the override is stale.

### Discovering Available Endpoints

Use the Configuration API to list live endpoints:

```http
GET /v1/config/model-catalog
```

Returns model categories with their assigned endpoints, letting the UI populate dropdowns.

```http
GET /v1/config/serving-endpoints
```

Returns all workspace serving endpoints (cached for 2 minutes).

## Source Scope Configuration

Three source scopes control where research data comes from:

| Scope | Behavior |
|-------|----------|
| `all` | Search both web (Brave) and enterprise sources |
| `enterprise_only` | Only enterprise sources (Vector Search, Genie, Knowledge Assistants) |
| `web_only` | Only web search via Brave API |

### Per-Agent Source Enable/Disable

Beyond the broad scope, you can fine-tune which sources are active:

```json
{
  "source_scope": "all",
  "enabled_sources": ["product_docs", "support_kb"],
  "disabled_sources": ["legacy_index"]
}
```

### Override of Per-Query Selector

When a custom agent with source configuration is selected, the per-query source scope selector in the UI is hidden. The agent's source scope becomes the authoritative configuration. When no agent is selected (or the agent has no source config), the per-query selector remains visible.

## Web Domain Filtering

Domain filtering restricts which web domains Brave Search results come from.

### Filter Modes

| Mode | Behavior |
|------|----------|
| `include` | Only allow results from listed domains |
| `exclude` | Block results from listed domains |
| `both` | Apply both include and exclude lists |

### Pattern Syntax

Patterns support wildcard matching consistent with the YAML configuration:

```
*.gov         — all .gov domains
*.edu         — all .edu domains
news.*        — news.* (any TLD)
cve.mitre.org — exact domain
```

### Override Semantics

Agent-level domain filters **replace** (not merge with) the system-wide domain filter from `config/app.yaml`. When no domain filter is configured on the agent, the system-wide filter applies unchanged.

```json
{
  "domain_filter_mode": "include",
  "include_domains": ["*.gov", "*.edu"],
  "exclude_domains": null
}
```

### Validation

Domain filter configuration is validated on save:
- Patterns must be non-empty strings
- Patterns must contain valid domain characters
- If `domain_filter_mode` is set, at least one pattern list must be provided

## Preset Steps

Preset steps define a fixed research workflow. Instead of the AI planner generating steps, you define them explicitly.

### Workflow Modes

| Mode | Behavior |
|------|----------|
| `planner` | AI generates the research plan (default) |
| `manual` | Only preset steps are executed, no AI planning |
| `hybrid` | Preset steps execute first, then AI can add more |

### Step Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | string | required | Short title (max 255 chars) |
| `description` | string | null | What this step should accomplish |
| `order` | int | 1 | Execution order (1-based) |
| `is_required` | bool | true | Whether this step must be executed |
| `source_hints` | object | null | Hints for source selection |
| `source_scope` | enum | null | Optional source scope override for this step |

### Per-Step Source Overrides

Each preset step can override the agent-level source scope. Steps without a scope override inherit the agent-level configuration.

```json
{
  "preset_steps": [
    {
      "title": "Search internal docs",
      "description": "Check enterprise knowledge base first",
      "order": 1,
      "source_scope": "enterprise_only",
      "source_hints": {
        "preferred_sources": ["product_docs"],
        "search_queries": ["installation guide"]
      }
    },
    {
      "title": "Search public resources",
      "description": "Find community solutions",
      "order": 2,
      "source_scope": "web_only"
    }
  ]
}
```

### Source Hints

Source hints guide the researcher without strict enforcement:

```json
{
  "source_hints": {
    "preferred_sources": ["product_docs", "support_kb"],
    "search_queries": ["how to configure X", "troubleshooting X"],
    "filters": {"category": "docs"}
  }
}
```

### Step Reordering

Reorder steps without individual updates:

```http
POST /v1/custom-agents/{agent_id}/steps/reorder
Content-Type: application/json

["step-uuid-3", "step-uuid-1", "step-uuid-2"]
```

## Prompt Templates

Custom agents can reference two prompt templates:

| Slot | Purpose |
|------|---------|
| `system_prompt_template_id` | System instructions injected into the agent's system prompt |
| `synthesis_template_id` | Instructions for the synthesis (report generation) phase |

### Template Types

Templates are plain text content with optional `{{variable}}` placeholders. For the current iteration, templates used via custom agents are plain text only — no variable substitution is performed.

### Inline Creation

Templates can be created separately via the Templates API:

```http
POST /v1/templates
Content-Type: application/json

{
  "name": "Security Analysis System Prompt",
  "content": "Focus on CVE details, CVSS scores, and mitigation steps. Always cite NIST NVD entries.",
  "type": "system"
}
```

Then reference the template ID when creating or updating the agent:

```json
{
  "system_prompt_template_id": "template-uuid"
}
```

### Deleted Templates

If a referenced template is deleted, the agent gracefully clears its reference — no errors occur, the field reverts to null.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/custom-agents` | List accessible agents (own + workspace + system) |
| `POST` | `/v1/custom-agents` | Create a new agent |
| `GET` | `/v1/custom-agents/{id}` | Get agent with full config |
| `PATCH` | `/v1/custom-agents/{id}` | Update agent fields |
| `DELETE` | `/v1/custom-agents/{id}` | Delete agent (owner only) |
| `GET` | `/v1/custom-agents/{id}/steps` | List preset steps |
| `POST` | `/v1/custom-agents/{id}/steps` | Create a preset step |
| `PATCH` | `/v1/custom-agents/{id}/steps/{sid}` | Update a preset step |
| `DELETE` | `/v1/custom-agents/{id}/steps/{sid}` | Delete a preset step |
| `POST` | `/v1/custom-agents/{id}/steps/reorder` | Reorder preset steps |

For full request/response schemas, see [API Reference](./api.md).

## How Config Is Applied at Runtime

When a user submits a research query with a custom agent selected:

1. **Frontend** sends only the `agent_id` with the query submission
2. **Backend** resolves the full agent config from the database via `CustomAgentService.resolve_agent_for_request()`
3. Agent configuration is applied to `OrchestrationConfig`:
   - `model_overrides` — override tier-to-endpoint mapping
   - `source_scope` — set source scope on the research state
   - `domain_filter_mode` + `include_domains` / `exclude_domains` — override web search filters
   - `system_prompt_template_id` / `synthesis_template_id` — inject prompt templates
   - `default_depth`, `default_mode`, `use_planner` — configure orchestration behavior
4. If the agent has preset steps, they are loaded and used according to the `default_mode`:
   - `manual`: only preset steps execute
   - `hybrid`: preset steps execute first, AI can add more
   - `planner`: preset steps are ignored, AI plans from scratch
5. The research pipeline executes with the applied configuration

```
Frontend (agent_id) → API → resolve from DB → apply to OrchestrationConfig → Pipeline
```

## Edge Cases

### Stale Endpoints

If a `model_overrides` entry references an endpoint that was removed from `config/app.yaml`, the system:
- Returns a warning in the `model_override_warnings` field
- Falls back to the default endpoint for that tier at query time

### Deleted Sources

If `enabled_sources` references a source that no longer exists, the source is silently skipped. Research continues with remaining available sources.

### Duplicate Names

Agent names are unique per owner. Attempting to create an agent with a duplicate name returns a `409 Conflict` error.

### System Agents

System agents (`visibility: system`) are read-only for regular users. They appear in the agent list but cannot be edited or deleted.

### OBO Access Errors

When a workspace agent references endpoints or enterprise sources the current user lacks OBO (On-Behalf-Of) access to, the system returns a clear error identifying the inaccessible resource rather than silently falling back.

### Mid-Conversation Agent Switch

Switching agents mid-conversation applies the new configuration to subsequent queries only. Previous messages retain their original configuration context.

## See Also

- [Configuration](./configuration.md) — YAML configuration and runtime config
- [API Reference](./api.md) — Full endpoint documentation
- [Data Models](./data-models.md) — Entity definitions
- [Data Source Configuration](./data-source-config.md) — Enterprise source setup
