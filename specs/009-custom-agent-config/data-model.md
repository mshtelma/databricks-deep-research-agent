# Data Model: Custom Agent Configuration & Selection

**Feature**: 009-custom-agent-config
**Date**: 2026-02-09

## Entity Changes

### CustomAgent (extended)

**Table**: `custom_agents`

#### New Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `model_overrides` | `JSONB` | yes | `null` | Per-tier endpoint overrides: `{"simple": "ep-name", "analytical": "ep-name"}` |
| `domain_filter_mode` | `VARCHAR(20)` | yes | `null` | Domain filtering mode: `"include"`, `"exclude"`, `"both"`, or null (use system default) |
| `include_domains` | `JSONB` | yes | `null` | Array of include domain patterns: `["*.gov", "reuters.com"]` |
| `exclude_domains` | `JSONB` | yes | `null` | Array of exclude domain patterns: `["*.ru", "*.cn"]` |

#### JSONB Shape: `model_overrides`

```json
{
  "simple": "databricks-haiku",
  "complex": "databricks-opus"
}
```

- Keys are model tier names (must exist in `AppConfig.models`)
- Values are endpoint names (must exist in `AppConfig.endpoints`)
- Omitted tiers use system defaults
- Validated at query time, not at save time (endpoints may change between config and execution)

#### JSONB Shape: `include_domains` / `exclude_domains`

```json
["*.gov", "*.edu", "reuters.com", "news.*"]
```

- Standard domain wildcard patterns matching `DomainFilterConfig` in `app_config.py`
- Stored as JSON arrays of strings
- `domain_filter_mode` must be non-null for these to take effect

#### Existing Columns (no changes)

| Column | Type | Used for |
|--------|------|----------|
| `source_scope` | `VARCHAR(50)` | Source scope (all, enterprise_only, web_only) — already exists |
| `enabled_sources` | `JSONB` | Whitelist of source names — already exists |
| `disabled_sources` | `JSONB` | Blacklist of source names — already exists |
| `system_prompt_template_id` | `UUID FK` | System prompt template — already exists |
| `synthesis_template_id` | `UUID FK` | Synthesis template — already exists |

### AgentPresetStep (no changes)

The `source_scope` column already exists on `agent_preset_steps` table. No additional columns needed for per-step source overrides (FR-016).

### PromptTemplate (no changes)

The existing `prompt_templates` table already has: `name`, `content`, `type`, `visibility`, `owner_id`, `variables`, `tags`. The simplified template creation flow (FR-025) uses existing columns — `variables` is simply left empty/null.

### Endpoint Catalog (virtual — no table)

Not persisted. Derived at runtime from `AppConfig.endpoints` and `AppConfig.models`.

```python
@dataclass
class EndpointCatalogEntry:
    name: str                    # Endpoint config key (e.g., "databricks-opus")
    endpoint_identifier: str     # Actual serving endpoint name
    max_context_window: int
    supports_structured_output: bool
    supports_temperature: bool
    supports_prompt_caching: bool

@dataclass
class ModelCategoryEntry:
    name: str                    # Tier name (e.g., "analytical")
    default_endpoints: list[str] # Endpoint names assigned to this tier
    temperature: float
    max_tokens: int

@dataclass
class EndpointCatalog:
    categories: dict[str, ModelCategoryEntry]
    endpoints: dict[str, EndpointCatalogEntry]
```

---

## OrchestrationConfig Extensions

| Field | Type | Default | Wire to |
|-------|------|---------|---------|
| `model_overrides` | `dict[str, str] \| None` | `None` | `ResearchState.model_overrides` |
| `domain_filter` | `DomainFilterConfig \| None` | `None` | Search client at query time |

Both already-existing fields (`source_scope`, `enabled_sources`, `disabled_sources`, `agent_id`) continue unchanged.

---

## ResearchState Extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_overrides` | `dict[str, str] \| None` | `None` | Per-tier endpoint name overrides |

The `agent_id` field already exists on `ResearchState`.

---

## Migration

**File**: `src/deep_research/db/migrations/versions/017_custom_agent_model_overrides.py`

```python
"""Add model overrides and domain filter columns to custom_agents.

Revision ID: 017
"""

def upgrade() -> None:
    op.add_column("custom_agents", sa.Column("model_overrides", JSONB, nullable=True))
    op.add_column("custom_agents", sa.Column("domain_filter_mode", sa.String(20), nullable=True))
    op.add_column("custom_agents", sa.Column("include_domains", JSONB, nullable=True))
    op.add_column("custom_agents", sa.Column("exclude_domains", JSONB, nullable=True))

def downgrade() -> None:
    op.drop_column("custom_agents", "exclude_domains")
    op.drop_column("custom_agents", "include_domains")
    op.drop_column("custom_agents", "domain_filter_mode")
    op.drop_column("custom_agents", "model_overrides")
```

Since the application is not live, this migration can also be implemented by recreating the existing migration file if preferred.

---

## Schema Extensions (Pydantic)

### CreateCustomAgentRequest (extended)

```python
# New fields added to schemas/custom_agent.py
model_overrides: dict[str, str] | None = Field(
    None,
    description="Per-tier model endpoint overrides: {tier_name: endpoint_name}",
)
domain_filter_mode: DomainFilterMode | None = Field(
    None,
    description="Domain filtering mode for web search",
)
include_domains: list[str] | None = Field(
    None,
    description="Domain patterns to include (whitelist)",
)
exclude_domains: list[str] | None = Field(
    None,
    description="Domain patterns to exclude (blacklist)",
)
```

### UpdateCustomAgentRequest (extended)

Same four fields, all optional (consistent with existing PATCH semantics).

### CustomAgentResponse (extended)

Same four fields, reflecting persisted state.

### CustomAgentSummary (extended)

Add `has_model_overrides: bool` and `has_domain_filter: bool` flags for listing UI.

### EndpointCatalogResponse (new)

```python
class EndpointInfo(BaseSchema):
    name: str
    endpoint_identifier: str
    max_context_window: int
    supports_structured_output: bool

class ModelCategoryInfo(BaseSchema):
    name: str
    default_endpoints: list[str]
    temperature: float
    max_tokens: int

class EndpointCatalogResponse(BaseSchema):
    categories: dict[str, ModelCategoryInfo]
    endpoints: dict[str, EndpointInfo]
```

---

## Validation Rules

### Model Overrides (at query time in `apply_custom_agent_to_config`)

1. For each entry in `model_overrides`:
   - Tier name must exist in `AppConfig.models` — skip with warning if not
   - Endpoint name must exist in `AppConfig.endpoints` — skip with warning if not
2. Invalid entries are silently dropped (system defaults used)
3. Warning badge shown in frontend if agent references stale endpoints

### Domain Filters (at save time in API)

1. `domain_filter_mode` required if `include_domains` or `exclude_domains` is non-empty
2. Patterns must be non-empty strings containing valid domain characters (`[a-zA-Z0-9.*-]`)
3. `include_domains` required when mode is `"include"` or `"both"`
4. `exclude_domains` required when mode is `"exclude"` or `"both"`

### Template References

1. FK with `ondelete="SET NULL"` already handles deleted templates (FR-028)
2. No additional validation needed — null template = no override
