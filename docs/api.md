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

## Key Files

| File | Purpose |
|------|---------|
| `src/api/v1/research.py` | Research endpoints |
| `src/api/v1/chats.py` | Chat management |
| `src/api/v1/citations.py` | Citation endpoints |
| `src/api/v1/health.py` | Health checks |
| `src/schemas/streaming.py` | SSE event schemas |

## See Also

- [Architecture](./architecture.md) - System overview
- [Data Models](./data-models.md) - Entity definitions
- [Deployment](./deployment.md) - Running the API
