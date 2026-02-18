# Data Model: Data Source Selection Integration

**Feature**: 008-data-source-selection
**Date**: 2026-02-05

## Overview

This feature is a **pure integration layer** - no new database entities are created. All data models already exist from Feature 007. This document describes the existing models being connected and the new client-side state model.

## Existing Backend Models (No Changes)

### SourceScope Enum

**File**: `src/deep_research/schemas/source_scope.py`

```python
class SourceScope(str, Enum):
    ENTERPRISE_ONLY = "enterprise_only"  # Only enterprise data sources
    WEB_ONLY = "web_only"                 # Only web search
    ALL = "all"                           # Both enterprise and web
```

**Usage**: Discriminates which category of sources to query.

### SourceScopeConfig Model

**File**: `src/deep_research/schemas/source_scope.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scope` | `SourceScope` | `ALL` | High-level scope selection |
| `enabled_sources` | `list[str] \| None` | `None` | Whitelist of source IDs (None = no whitelist) |
| `disabled_sources` | `list[str]` | `[]` | Blacklist of source IDs |
| `enable_vector_search` | `bool` | `True` | Allow Vector Search sources |
| `enable_genie` | `bool` | `True` | Allow Genie spaces |
| `enable_knowledge_assistant` | `bool` | `True` | Allow Knowledge Assistants |
| `enable_web_search` | `bool` | `True` | Allow web searches |
| `enable_uploaded_files` | `bool` | `True` | Allow user-uploaded files |

**Methods**:
- `is_type_enabled(source_type: str) -> bool`
- `is_source_enabled(source_id: str) -> bool`
- `filter_sources(sources: list[DiscoveredSource]) -> list[DiscoveredSource]`

### ResearchRequest Model

**File**: `src/deep_research/schemas/research_request.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | Required | Research query text |
| `research_depth` | `str` | `"standard"` | Research thoroughness |
| `verify_sources` | `bool` | `True` | Enable citation verification |
| `source_scope` | `SourceScope \| None` | `None` | Scope selector value |
| `enabled_sources` | `list[str] \| None` | `None` | Whitelist |
| `disabled_sources` | `list[str]` | `[]` | Blacklist |

### OrchestrationConfig Dataclass

**File**: `src/deep_research/agent/orchestrator.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_scope` | `str \| None` | `None` | String value of scope |
| `enabled_sources` | `list[str] \| None` | `None` | Whitelist |
| `disabled_sources` | `list[str] \| None` | `None` | Blacklist |

**Note**: Uses `str` instead of `SourceScope` enum for serialization flexibility.

---

## New API Model (Backend Addition)

### SubmitJobRequest Extension

**File**: `src/deep_research/api/v1/jobs.py`

**Current fields** (unchanged):
| Field | Type | Default |
|-------|------|---------|
| `chat_id` | `UUID` | Required |
| `query` | `str` | Required |
| `query_mode` | `str` | Required |
| `research_depth` | `str` | Required |
| `verify_sources` | `bool` | Required |
| `output_type` | `str \| None` | `None` |

**New fields** (to be added):
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_scope` | `SourceScope \| None` | `None` | Scope selection |
| `enabled_sources` | `list[str] \| None` | `None` | Whitelist |
| `disabled_sources` | `list[str]` | `[]` | Blacklist |

**Validation rules**:
- All new fields are optional for backward compatibility
- `source_scope=None` treated as "all"
- Empty `disabled_sources` means no exclusions
- `enabled_sources=None` means no whitelist (use all available)

---

## New Client-Side Model (Frontend Addition)

### SourceScopePreference

**File**: `frontend/src/hooks/useSourceScope.ts` (new)

```typescript
interface SourceScopePreference {
  scope: SourceScope           // 'enterprise_only' | 'web_only' | 'all'
  enabledSources: string[]     // IDs of explicitly enabled sources
  disabledSources: string[]    // IDs of explicitly disabled sources
}
```

**Storage**: localStorage under key `deep-research-source-scope`

**Default value**:
```typescript
{
  scope: 'all',
  enabledSources: [],
  disabledSources: []
}
```

**State transitions**:
1. **Initialize**: Load from localStorage or use default
2. **Scope change**: User selects new scope → update state + persist
3. **Source toggle**: User toggles source → move between enabled/disabled → persist
4. **Page refresh**: Restore from localStorage

---

## Type Mapping (Frontend ↔ Backend)

| Frontend (TypeScript) | Backend (Python) | JSON Wire Format |
|-----------------------|------------------|------------------|
| `SourceScope` | `SourceScope` | `"enterprise_only"` / `"web_only"` / `"all"` |
| `string[]` | `list[str]` | `["id1", "id2"]` |
| `null` / `undefined` | `None` | `null` |

**Serialization rules**:
- Frontend uses camelCase in TypeScript code
- API requests use snake_case in JSON body
- API client transforms between them

---

## Entity Relationships

```text
┌─────────────────────────────────────────────────────────────────┐
│                     REQUEST FLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SourceScopePreference (localStorage)                           │
│          │                                                       │
│          ▼                                                       │
│  SubmitJobRequest (HTTP POST body)                              │
│          │                                                       │
│          ▼                                                       │
│  JobManager.submit_job() params                                 │
│          │                                                       │
│          ▼                                                       │
│  OrchestrationConfig                                            │
│          │                                                       │
│          ▼                                                       │
│  ResearchState.source_scope_config                              │
│          │                                                       │
│          ▼                                                       │
│  Tool execution (web_search, vector_search, etc.)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## No Database Schema Changes

This feature does not modify the database schema. Source preferences are:
1. **Ephemeral per-request**: Sent with each job submission
2. **Persisted client-side**: In browser localStorage

Future consideration: If per-user server-side persistence is needed, a new table would be required:
```sql
-- NOT IMPLEMENTED IN THIS FEATURE (out of scope)
CREATE TABLE user_source_preferences (
    user_id VARCHAR PRIMARY KEY,
    source_scope VARCHAR,
    enabled_sources JSONB,
    disabled_sources JSONB,
    updated_at TIMESTAMP
);
```
