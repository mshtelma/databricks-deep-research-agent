# Quickstart: Data Source Selection Integration

**Feature**: 008-data-source-selection
**Date**: 2026-02-05

## Prerequisites

- Feature 007 (Enterprise Data Sources) deployed and functional
- At least one discovered enterprise data source (Vector Search, Genie, or Knowledge Assistant)
- Development environment set up (`make install`)

## Quick Verification

After implementation, verify the feature works:

### 1. Start Development Server

```bash
make dev
```

### 2. Open Chat Interface

Navigate to `http://localhost:5173` and select or create a chat.

### 3. Verify Source Selector Appears

1. Set query mode to "Deep Research" or "Web Search"
2. Look for the Source Scope selector (three toggle buttons: Enterprise Only, Web Only, All Sources)
3. The selector should NOT appear when query mode is "Simple"

### 4. Test Enterprise Only Mode

1. Select "Enterprise Only" scope
2. Type a query and submit
3. Check browser DevTools Network tab:
   - Request body should include `"source_scope": "enterprise_only"`
4. Watch research events - no `web_search` tool calls should appear

### 5. Test Web Only Mode

1. Select "Web Only" scope
2. Submit a query
3. Verify research only uses web search (no enterprise source queries)

### 6. Test Preference Persistence

1. Select "Enterprise Only" scope
2. Refresh the page
3. Verify scope selection is preserved

## Development Commands

```bash
# Run backend tests
uv run pytest tests/unit/api/test_jobs.py -v

# Run frontend type check
cd frontend && npm run typecheck

# Run full test suite
make test

# Check linting
make lint
```

## Key Files to Review

| File | What to Look For |
|------|------------------|
| `src/deep_research/api/v1/jobs.py` | `source_scope`, `enabled_sources`, `disabled_sources` in SubmitJobRequest |
| `src/deep_research/services/job_manager.py` | Source params in `submit_job()` and `_run_job()` signatures |
| `frontend/src/api/client.ts` | Source fields in `jobsApi.submit()` data parameter |
| `frontend/src/components/chat/MessageInput.tsx` | SourceScopeSelector rendering and onSubmit integration |
| `frontend/src/hooks/useSourceScope.ts` | localStorage persistence hook |

## API Usage Examples

### Submit with Enterprise Sources Only

```bash
curl -X POST http://localhost:8000/api/v1/research/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "chat_id": "YOUR_CHAT_ID",
    "query": "What is our company policy on expenses?",
    "query_mode": "deep_research",
    "research_depth": "standard",
    "verify_sources": true,
    "source_scope": "enterprise_only"
  }'
```

### Submit with Specific Sources Enabled

```bash
curl -X POST http://localhost:8000/api/v1/research/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "chat_id": "YOUR_CHAT_ID",
    "query": "Q1 revenue projections",
    "query_mode": "deep_research",
    "research_depth": "extended",
    "verify_sources": true,
    "source_scope": "all",
    "enabled_sources": ["catalog.schema.financial_docs"],
    "disabled_sources": ["catalog.schema.legacy_data"]
  }'
```

### Submit with Web Only (backward compatible)

```bash
curl -X POST http://localhost:8000/api/v1/research/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "chat_id": "YOUR_CHAT_ID",
    "query": "Latest news on AI regulation",
    "query_mode": "web_search",
    "research_depth": "light",
    "verify_sources": false,
    "source_scope": "web_only"
  }'
```

## Troubleshooting

### Source Selector Not Appearing

1. Check query mode is "deep_research" or "web_search" (not "simple")
2. Check browser console for JavaScript errors
3. Verify `SourceScopeSelector` is imported in MessageInput.tsx

### Source Selection Not Persisted

1. Check browser localStorage for key `deep-research-source-scope`
2. Verify no localStorage quota errors in console
3. Check `useSourceScope` hook is correctly initialized

### Backend Ignoring Source Scope

1. Check network request body includes `source_scope` field
2. Verify `SubmitJobRequest` validation passes (check server logs)
3. Confirm `JobManager.submit_job()` receives the parameters
4. Trace through to `OrchestrationConfig` creation

### Enterprise Only Returns No Results

1. Verify enterprise sources are discovered (`/api/v1/discovery/sources`)
2. Check source status is "available" not "unavailable"
3. Confirm OBO token has access to the data sources

## Success Metrics

After implementation, verify these success criteria:

- [ ] SC-001: Source scope selection in <10 seconds
- [ ] SC-002: Enterprise Only queries don't trigger web searches
- [ ] SC-003: Preferences persist across page refresh
- [ ] SC-004: Single click toggles individual sources
- [ ] SC-005: Sources display within 2 seconds of page load
