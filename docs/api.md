# API Reference

## Overview

The Deep Research Agent exposes a REST API via FastAPI with Server-Sent Events (SSE) for real-time streaming. All endpoints require authentication via Databricks workspace identity.

## Base URL

```
Development: http://localhost:8000/v1
Production: https://<app-name>.<workspace>.databricks.com/v1
```

## Authentication

All requests must include Databricks authentication headers. The middleware extracts user identity from the request context.

## REST Endpoints

### Research

#### Start Research

```http
POST /v1/research
Content-Type: application/json

{
  "query": "What are the latest developments in AI?",
  "chat_id": "uuid-optional",
  "query_mode": "deep_research",
  "research_depth": "medium",
  "verify_sources": true
}
```

**Response**: Server-Sent Events stream

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Research query |
| `chat_id` | UUID | auto | Chat to add message to |
| `query_mode` | string | `deep_research` | `simple`, `web_search`, `deep_research` |
| `research_depth` | string | `auto` | `auto`, `light`, `medium`, `extended` |
| `verify_sources` | bool | varies | Enable citation verification |

### Chats

#### List Chats

```http
GET /v1/chats
```

**Response**:
```json
{
  "chats": [
    {
      "id": "uuid",
      "title": "AI Research",
      "createdAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-01-15T11:45:00Z"
    }
  ]
}
```

#### Create Chat

```http
POST /v1/chats
Content-Type: application/json

{
  "title": "New Research Chat"
}
```

**Response**:
```json
{
  "id": "uuid",
  "title": "New Research Chat",
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-15T10:30:00Z"
}
```

#### Get Chat

```http
GET /v1/chats/{chat_id}
```

**Response**:
```json
{
  "id": "uuid",
  "title": "AI Research",
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-15T11:45:00Z",
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "What are the latest developments in AI?",
      "queryMode": "deep_research",
      "createdAt": "2024-01-15T10:30:00Z"
    },
    {
      "id": "uuid",
      "role": "agent",
      "content": "Based on my research...",
      "queryMode": "deep_research",
      "createdAt": "2024-01-15T10:31:00Z"
    }
  ]
}
```

#### Delete Chat

```http
DELETE /v1/chats/{chat_id}
```

**Response**: `204 No Content`

Note: Soft delete with 30-day recovery window.

### Messages

#### Export Report

```http
GET /v1/messages/{message_id}/report
```

**Response**:
```markdown
# AI Research Report

**Generated**: 2024-01-15 10:31:00 UTC
**Query**: What are the latest developments in AI?
**Mode**: Deep Research (Medium)

## Report

Based on my research...

## Sources

1. [Source Title](https://example.com/article)
2. [Another Source](https://example.com/other)
```

#### Get Provenance

```http
GET /v1/messages/{message_id}/provenance?format=json
```

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `json` | `json` or `markdown` |

**Response (JSON)**:
```json
{
  "claims": [
    {
      "id": "uuid",
      "claimText": "AI models have improved significantly.",
      "claimType": "general",
      "positionStart": 0,
      "positionEnd": 40,
      "citationKey": "[Arxiv]",
      "verdict": "supported",
      "confidence": 0.95,
      "citations": [
        {
          "id": "uuid",
          "evidenceSpan": {
            "quote": "Recent AI models show 50% improvement...",
            "sourceUrl": "https://arxiv.org/...",
            "sourceTitle": "AI Progress Report 2024"
          },
          "confidence": 0.95
        }
      ]
    }
  ],
  "verificationSummary": {
    "totalClaims": 15,
    "supported": 12,
    "partial": 2,
    "unsupported": 1,
    "contradicted": 0
  }
}
```

**Response (Markdown)**:
```markdown
# Verification Report

## Summary
- Total Claims: 15
- Supported: 12
- Partial: 2
- Unsupported: 1

## Claims

### Claim 1 (SUPPORTED)
**Text**: AI models have improved significantly.
**Evidence**: "Recent AI models show 50% improvement..."
**Source**: [AI Progress Report 2024](https://arxiv.org/...)
```

#### Submit Feedback

```http
POST /v1/messages/{message_id}/feedback
Content-Type: application/json

{
  "rating": "positive",
  "feedbackText": "Very helpful research!"
}
```

**Response**:
```json
{
  "id": "uuid",
  "messageId": "uuid",
  "rating": "positive",
  "feedbackText": "Very helpful research!",
  "createdAt": "2024-01-15T10:35:00Z"
}
```

### User Preferences

#### Get Preferences

```http
GET /v1/preferences
```

**Response**:
```json
{
  "systemInstructions": "Always cite academic sources.",
  "defaultQueryMode": "deep_research",
  "defaultResearchDepth": "medium"
}
```

#### Update Preferences

```http
PUT /v1/preferences
Content-Type: application/json

{
  "systemInstructions": "Focus on recent publications.",
  "defaultQueryMode": "web_search",
  "defaultResearchDepth": "light"
}
```

### Health

#### Health Check

```http
GET /v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "llm": "available"
}
```

## Server-Sent Events (SSE)

The `/v1/research` endpoint returns an SSE stream with real-time updates.

### Event Format

```
event: <event_type>
data: <json_payload>

```

### Event Types

#### research_started

Emitted when research begins.

```json
{
  "eventType": "research_started",
  "sessionId": "uuid",
  "messageId": "uuid",
  "chatId": "uuid"
}
```

#### step_started

Emitted when a research step begins.

```json
{
  "eventType": "step_started",
  "stepIndex": 0,
  "stepTitle": "Research AI developments",
  "stepDescription": "Search for recent AI papers and news"
}
```

#### tool_call

Emitted when a tool is invoked.

```json
{
  "eventType": "tool_call",
  "tool": "web_search",
  "args": {
    "query": "AI developments 2024"
  },
  "callNumber": 1
}
```

#### tool_result

Emitted when a tool completes.

```json
{
  "eventType": "tool_result",
  "tool": "web_search",
  "resultPreview": "Found 5 results for AI developments...",
  "sourcesCrawled": 3
}
```

#### step_completed

Emitted when a research step completes.

```json
{
  "eventType": "step_completed",
  "stepIndex": 0,
  "stepTitle": "Research AI developments",
  "observation": "Found several recent papers on transformer improvements...",
  "sourcesFound": 5
}
```

#### reflection_decision

Emitted when reflector makes a decision.

```json
{
  "eventType": "reflection_decision",
  "decision": "continue",
  "reasoning": "Need more information about specific applications",
  "suggestedChanges": null
}
```

#### synthesis_progress

Emitted during report generation (streaming).

```json
{
  "eventType": "synthesis_progress",
  "contentChunk": "Based on my research, "
}
```

#### claim_verified

Emitted when a claim is verified.

```json
{
  "eventType": "claim_verified",
  "claimText": "GPT-4 achieved 90% on the bar exam",
  "verdict": "supported",
  "confidence": 0.95,
  "citationKey": "[OpenAI]"
}
```

#### numeric_claim_detected

Emitted when a numeric claim is found.

```json
{
  "eventType": "numeric_claim_detected",
  "claimText": "Revenue increased by $3.2 billion",
  "value": "3.2",
  "unit": "billion USD",
  "entity": "Revenue"
}
```

#### verification_summary

Emitted after all claims are verified.

```json
{
  "eventType": "verification_summary",
  "totalClaims": 15,
  "supported": 12,
  "partial": 2,
  "unsupported": 1,
  "contradicted": 0
}
```

#### research_complete

Emitted when research finishes successfully.

```json
{
  "eventType": "research_complete",
  "sessionId": "uuid",
  "messageId": "uuid",
  "totalSources": 12,
  "totalClaims": 15,
  "durationSeconds": 45.3
}
```

#### error

Emitted when an error occurs.

```json
{
  "eventType": "error",
  "errorCode": "RATE_LIMIT_EXCEEDED",
  "errorMessage": "LLM rate limit exceeded, retrying...",
  "recoverable": true
}
```

### SSE Client Example

```typescript
const eventSource = new EventSource('/v1/research?query=AI+developments');

eventSource.addEventListener('research_started', (e) => {
  const data = JSON.parse(e.data);
  console.log('Research started:', data.sessionId);
});

eventSource.addEventListener('synthesis_progress', (e) => {
  const data = JSON.parse(e.data);
  // Append to streaming content
  content += data.contentChunk;
});

eventSource.addEventListener('research_complete', (e) => {
  const data = JSON.parse(e.data);
  console.log('Research complete:', data.totalSources, 'sources');
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  if (!data.recoverable) {
    console.error('Fatal error:', data.errorMessage);
    eventSource.close();
  }
});
```

## Error Responses

### Standard Error Format

```json
{
  "detail": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid query mode",
    "field": "query_mode"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `UNAUTHORIZED` | 401 | Missing/invalid auth |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/v1/research` | 10 req/min |
| `/v1/chats` | 60 req/min |
| `/v1/messages/*` | 60 req/min |

## OpenAPI Specification

Full OpenAPI specification available at:
- `specs/001-deep-research-agent/contracts/openapi.yaml`
- `specs/004-tiered-query-modes/contracts/openapi-patch.yaml`

## Enterprise Data Sources API

The Enterprise Data Sources feature provides endpoints for auto-discovering and configuring enterprise data sources including Vector Search indexes, Genie spaces, and Knowledge Assistants.

### Discovery

#### Discover All Sources

```http
GET /v1/discovery/sources
```

Auto-discovers all data sources the user has access to via OBO (On-Behalf-Of) authentication. Results are cached for 5 minutes per user.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_type` | string | all | Filter by type: `vector_search`, `genie`, `knowledge_assistant` |
| `refresh` | bool | false | Force cache refresh |

**Response**:
```json
{
  "sources": [
    {
      "source_id": "vs:catalog.schema.product_docs",
      "source_type": "vector_search",
      "name": "product_docs",
      "endpoint_name": "docs-endpoint",
      "description": "Product documentation embeddings",
      "status": "ready",
      "capabilities": {
        "query_types": ["ANN", "HYBRID"],
        "supports_filters": true,
        "supports_reranking": true
      },
      "metadata": {
        "index_type": "DELTA_SYNC",
        "embedding_dimension": 1024,
        "filter_columns": ["category", "date", "version"]
      }
    }
  ],
  "by_type": {
    "vector_search": 5,
    "genie": 2,
    "knowledge_assistant": 1
  },
  "total_count": 8,
  "cached": true,
  "cache_expires_at": "2024-01-15T10:35:00Z",
  "errors": []
}
```

#### Get Source Metadata

```http
GET /v1/discovery/sources/{source_id}/metadata
```

Returns detailed metadata for a specific source.

**Source ID Formats**:
- Vector Search: `vs:catalog.schema.index`
- Genie: `genie:space_id`
- Knowledge Assistant: `assistant:endpoint_name`

**Response (Vector Search)**:
```json
{
  "source_id": "vs:catalog.schema.product_docs",
  "source_type": "vector_search",
  "metadata": {
    "endpoint_name": "docs-endpoint",
    "index_name": "catalog.schema.product_docs",
    "primary_key": "id",
    "index_type": "DELTA_SYNC",
    "status": "ONLINE",
    "embedding_dimension": 1024,
    "embedding_column": "embedding",
    "filter_columns": [
      {"name": "category", "data_type": "string"},
      {"name": "date", "data_type": "timestamp"},
      {"name": "version", "data_type": "integer"}
    ],
    "supported_query_types": ["ANN", "HYBRID", "FULL_TEXT"],
    "row_count": 150000
  }
}
```

#### Force Discovery Refresh

```http
POST /v1/discovery/refresh
Content-Type: application/json

{
  "source_types": ["vector_search", "genie"]
}
```

Invalidates the discovery cache and re-queries Databricks APIs.

**Response**: Same as `GET /v1/discovery/sources`

#### Get Cache Statistics

```http
GET /v1/discovery/stats
```

Returns cache statistics for monitoring.

**Response**:
```json
{
  "total_entries": 25,
  "hit_count": 150,
  "miss_count": 10,
  "hit_rate": 0.9375
}
```

### Data Sources

#### List User Data Sources

```http
GET /v1/data-sources
```

Lists all data sources accessible to the user (discovered + user-added).

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | all | Filter by source type |
| `include_workspace` | bool | true | Include workspace-visible sources |

**Response**:
```json
{
  "sources": [
    {
      "id": "uuid",
      "owner_id": "user-uuid",
      "type": "vector_search",
      "name": "My Product Docs",
      "description": "Custom product documentation index",
      "endpoint_identifier": "catalog.schema.my_docs",
      "visibility": "private",
      "validation_status": "valid",
      "last_validated_at": "2024-01-15T10:30:00Z",
      "config": {
        "query_config": {
          "query_type": "HYBRID",
          "num_results": 20,
          "score_threshold": 0.7,
          "filters": []
        }
      }
    }
  ],
  "total": 5
}
```

#### Create Data Source

```http
POST /v1/data-sources
Content-Type: application/json

{
  "type": "vector_search",
  "name": "My Product Docs",
  "description": "Custom product documentation index",
  "endpoint_identifier": "catalog.schema.my_docs",
  "visibility": "private"
}
```

**Response**:
```json
{
  "id": "uuid",
  "owner_id": "user-uuid",
  "type": "vector_search",
  "name": "My Product Docs",
  "validation_status": "valid",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Data Source

```http
GET /v1/data-sources/{id}
```

#### Delete Data Source

```http
DELETE /v1/data-sources/{id}
```

**Response**: `204 No Content`

#### Validate Data Source

```http
POST /v1/data-sources/{id}/validate
```

Re-validates OBO access to the data source.

**Response**:
```json
{
  "id": "uuid",
  "validation_status": "valid",
  "last_validated_at": "2024-01-15T10:35:00Z"
}
```

### Query Configuration

#### Get Query Configuration

```http
GET /v1/data-sources/{id}/query-config?validate=true
```

Retrieves the query configuration for a Vector Search data source.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validate` | bool | false | Validate config against index capabilities |

**Response**:
```json
{
  "source_id": "uuid",
  "config": {
    "query_type": "HYBRID",
    "num_results": 20,
    "score_threshold": 0.7,
    "columns": ["title", "content", "metadata"],
    "enable_reranking": true,
    "columns_to_rerank": ["content"],
    "filters": [
      {"column": "category", "operator": "=", "value": "docs"}
    ],
    "filter_syntax": "SQL"
  },
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": ["Filter column 'category' may not exist in index"]
  }
}
```

#### Update Query Configuration

```http
PUT /v1/data-sources/{id}/query-config
Content-Type: application/json

{
  "query_type": "HYBRID",
  "num_results": 20,
  "score_threshold": 0.7,
  "enable_reranking": true,
  "columns_to_rerank": ["content"],
  "filters": [
    {"column": "category", "operator": "=", "value": "docs"},
    {"column": "date", "operator": ">", "value": "2024-01-01"}
  ],
  "filter_syntax": "SQL"
}
```

**Filter Operators**:
| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equals | `category = 'docs'` |
| `!=` | Not equals | `status != 'archived'` |
| `<` | Less than | `date < '2024-01-01'` |
| `<=` | Less or equal | `score <= 100` |
| `>` | Greater than | `count > 0` |
| `>=` | Greater or equal | `price >= 10.0` |
| `LIKE` | Contains | `title LIKE '%python%'` |
| `NOT LIKE` | Not contains | `content NOT LIKE '%draft%'` |
| `IN` | In list | `id IN (1, 2, 3)` |

**Validation**:
- IN clause limited to 1,024 values
- Query type validated against index capabilities
- Reranking columns must exist in index

### Source Metrics

#### Get Metrics Summary

```http
GET /v1/metrics/sources
```

Returns query metrics for all data sources.

**Response**:
```json
{
  "total_queries": 1500,
  "total_errors": 15,
  "overall_error_rate": 0.01,
  "by_type": {
    "vector_search": {
      "query_count": 1000,
      "success_count": 990,
      "error_count": 10,
      "error_rate": 0.01,
      "avg_latency_ms": 125.5,
      "min_latency_ms": 45.2,
      "max_latency_ms": 890.1
    },
    "genie": {
      "query_count": 400,
      "success_count": 397,
      "error_count": 3,
      "error_rate": 0.0075,
      "avg_latency_ms": 2500.0
    }
  },
  "by_name": {
    "product_docs": {
      "query_count": 500,
      "avg_latency_ms": 110.2
    }
  }
}
```

#### Get Latency Percentiles

```http
GET /v1/metrics/sources/{source_type}/percentiles
```

**Response**:
```json
{
  "source_type": "vector_search",
  "percentiles": {
    "p50": 95.2,
    "p90": 180.5,
    "p95": 250.1,
    "p99": 450.8
  }
}
```

## Custom Agents API

Manage reusable research profiles that bundle model, source, prompt, and workflow configuration. See [Custom Agents](./custom-agents.md) for a comprehensive guide.

### List Agents

```http
GET /v1/custom-agents?visibility=private&limit=100&offset=0
```

Returns agents accessible to the current user (own + workspace + system).

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `visibility` | string | all | Filter by visibility: `private`, `workspace`, `system` |
| `source_scope` | string | all | Filter by source scope: `all`, `enterprise_only`, `web_only` |
| `limit` | int | 100 | Max results (1-500) |
| `offset` | int | 0 | Pagination offset |

**Response**:
```json
{
  "agents": [
    {
      "id": "uuid",
      "owner_id": "user-id",
      "name": "Security Researcher",
      "description": "Focused on cybersecurity",
      "visibility": "private",
      "source_scope": "web_only",
      "default_mode": "planner",
      "default_depth": "extended",
      "preset_step_count": 3,
      "has_model_overrides": true,
      "has_domain_filter": true,
      "has_source_config": true,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 12,
  "user_agents": 5,
  "workspace_agents": 4,
  "system_agents": 3
}
```

### Create Agent

```http
POST /v1/custom-agents
Content-Type: application/json

{
  "name": "My Agent",
  "description": "Optional description",
  "default_depth": "medium",
  "source_scope": "all",
  "model_overrides": {"complex": "databricks-claude-sonnet-er"},
  "preset_steps": [
    {"title": "Step 1", "order": 1}
  ]
}
```

**Response**: `201` with full `CustomAgentResponse`

### Get Agent

```http
GET /v1/custom-agents/{agent_id}
```

**Response**: Full agent configuration including preset steps and model override warnings.

### Update Agent

```http
PATCH /v1/custom-agents/{agent_id}
Content-Type: application/json

{
  "name": "Updated Name",
  "model_overrides": {"complex": "new-endpoint"}
}
```

Only provided fields are updated. **Response**: Full `CustomAgentResponse`.

### Delete Agent

```http
DELETE /v1/custom-agents/{agent_id}
```

**Response**: `204 No Content`. Only the owner can delete an agent.

### Preset Steps

```http
GET    /v1/custom-agents/{agent_id}/steps
POST   /v1/custom-agents/{agent_id}/steps
PATCH  /v1/custom-agents/{agent_id}/steps/{step_id}
DELETE /v1/custom-agents/{agent_id}/steps/{step_id}
POST   /v1/custom-agents/{agent_id}/steps/reorder
```

The reorder endpoint accepts a JSON array of step UUIDs in the desired order.

## Prompt Templates API

Manage reusable prompt templates for system instructions, synthesis prompts, and custom steps.

### List Templates

```http
GET /v1/templates
```

Returns templates accessible to the current user (own + workspace).

### Create Template

```http
POST /v1/templates
Content-Type: application/json

{
  "name": "Security System Prompt",
  "content": "Focus on CVE details and CVSS scores.",
  "type": "system"
}
```

**Response**: `201` with created template including extracted variables.

### Get Template

```http
GET /v1/templates/{template_id}
```

### Update Template

```http
PATCH /v1/templates/{template_id}
Content-Type: application/json

{
  "content": "Updated template content."
}
```

### Delete Template

```http
DELETE /v1/templates/{template_id}
```

**Response**: `204 No Content`

### Render Template

```http
POST /v1/templates/{template_id}/render
Content-Type: application/json

{
  "variables": {"topic": "quantum computing"}
}
```

Renders the template with variable substitution and returns the result.

### Default Templates

```http
GET  /v1/templates/defaults/{type}
POST /v1/templates/{template_id}/set-default
```

Get or set the default template for a given type (`system`, `synthesis`, `step`, `query`).

## File Upload API

Upload documents (PDF, TXT, MD, DOCX) for use as research sources. Files are chunked for content-based search.

### Upload Files

```http
POST /v1/files/upload
Content-Type: multipart/form-data

file: <binary>
session_id: "optional-uuid"
```

**Limits**: 10 MB per file, 50 MB per session.

**Response**: `201` with file metadata including processing status.

### List Files

```http
GET /v1/files?session_id=uuid
```

Returns files owned by the current user, optionally filtered by session.

### Get File

```http
GET /v1/files/{file_id}
```

### Preview File Content

```http
GET /v1/files/{file_id}/preview
```

Returns the first chunk of the file for quick preview.

### Delete File

```http
DELETE /v1/files/{file_id}
```

**Response**: `204 No Content`. Deletes the file and all its chunks.

## Configuration API

Read-only endpoints exposing the current system configuration for the agent editor UI.

### Model Catalog

```http
GET /v1/config/model-catalog
```

Returns model categories (tiers) and their assigned endpoints. Used to populate model override dropdowns.

**Response**:
```json
{
  "categories": {
    "simple": {
      "endpoints": ["databricks-gemini-flash"],
      "temperature": 0.3
    },
    "analytical": {
      "endpoints": ["databricks-claude-sonnet", "databricks-gemini-flash"],
      "temperature": 0.7
    },
    "complex": {
      "endpoints": ["databricks-claude-sonnet-er", "databricks-claude-sonnet"],
      "temperature": 0.7
    }
  }
}
```

### Serving Endpoints

```http
GET /v1/config/serving-endpoints
```

Returns all workspace serving endpoints. Results are cached for 2 minutes.

## Key Files

| File | Purpose |
|------|---------|
| `src/api/v1/research.py` | Research endpoints |
| `src/api/v1/chats.py` | Chat management |
| `src/api/v1/citations.py` | Citation endpoints |
| `src/api/v1/health.py` | Health checks |
| `src/api/v1/discovery.py` | Data source discovery |
| `src/api/v1/data_sources.py` | Data source management |
| `src/api/v1/custom_agents.py` | Custom agent CRUD + preset steps |
| `src/api/v1/templates.py` | Prompt template CRUD + rendering |
| `src/api/v1/files.py` | File upload + preview |
| `src/api/v1/config.py` | Model catalog + serving endpoints |
| `src/schemas/streaming.py` | SSE event schemas |
| `src/schemas/discovery.py` | Discovery response schemas |
| `src/schemas/query_config.py` | Query configuration schemas |
| `src/schemas/custom_agent.py` | Custom agent request/response schemas |

## See Also

- [Architecture](./architecture.md) - System overview
- [Data Models](./data-models.md) - Entity definitions
- [Deployment](./deployment.md) - Running the API
- [Custom Agents](./custom-agents.md) - Custom agent configuration guide
- [Data Source Configuration](./data-source-config.md) - User guide for data sources
- [Plugin Development](./plugin-development.md) - Developer guide for plugins
