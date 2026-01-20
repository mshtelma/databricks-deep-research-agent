# Quickstart: Tiered Query Modes

**Feature**: 004-tiered-query-modes
**Date**: 2026-01-04

## Overview

This guide helps developers implement the tiered query modes feature. Follow these steps to set up your development environment and understand the key integration points.

## Prerequisites

- Python 3.11+ with uv package manager
- Node.js 18+ with npm
- PostgreSQL 14+ (or Databricks Lakebase connection)
- Access to Brave Search API
- Databricks workspace (for LLM endpoints)

## Quick Setup

### 1. Run Existing Setup

```bash
# From repository root
make dev  # Starts both backend and frontend
```

### 2. Run Database Migration

After implementing the migration file:

```bash
# Apply the new migration
uv run alembic upgrade head

# Verify new tables exist
uv run python -c "
from src.db.session import engine
from sqlalchemy import inspect
inspector = inspect(engine)
print('Tables:', inspector.get_table_names())
print('research_events columns:', [c['name'] for c in inspector.get_columns('research_events')])
"
```

### 3. Verify Configuration

Add to `config/app.yaml`:

```yaml
query_modes:
  simple:
    # Uses existing is_simple_query path
    model_role: simple
    emit_events: false
    create_session: false

  web_search:
    model_role: analytical
    timeout_seconds: 15
    # Reuse researcher with minimal config
    researcher:
      mode: classic
      max_search_queries: 2
      max_urls_to_crawl: 3
    # Skip planning stages
    skip_coordinator: true
    skip_planner: true
    skip_reflector: true
    # Synthesis config - Natural mode for [1], [2] citations
    citation_verification:
      enabled: true
      generation_mode: natural
      enable_numeric_qa_verification: false
      enable_verification_retrieval: false
    create_session: true  # Lightweight session for sources

  deep_research:
    # Uses existing research_types config
    model_role: complex
    use_research_types: true  # Inherits light/medium/extended
```

## Key Files to Implement

### Backend (Maximum Code Reuse - No New Agent Class!)

| File | Purpose | Priority |
|------|---------|----------|
| `src/models/enums.py` | Add QueryMode enum | 1 |
| `src/db/migrations/versions/005_tiered_query_modes.py` | Schema changes | 1 |
| `src/agent/state.py` | Add query_mode to ResearchState | 2 |
| `src/agent/orchestrator.py` | Add ~50 lines mode routing logic | 2 |
| `src/api/v1/research.py` | Add query_mode parameter | 2 |
| `src/models/research_event.py` | New ResearchEvent model | 3 |
| `src/services/research_event_service.py` | Event persistence | 3 |
| `config/app.yaml` | Add query_modes section | 2 |

**Note**: No new agent class needed! Web Search mode reuses existing `researcher.py` (mode=classic) and `synthesizer.py` (generation_mode=natural).

### Frontend

| File | Purpose | Priority |
|------|---------|----------|
| `frontend/src/types/index.ts` | Add QueryMode type | 1 |
| `frontend/src/components/chat/QueryModeSelector.tsx` | Mode selector UI | 1 |
| `frontend/src/components/chat/MessageInput.tsx` | Integrate selector | 2 |
| `frontend/src/hooks/useQueryMode.ts` | Mode state management | 2 |
| `frontend/src/hooks/useStreamingQuery.ts` | Add queryMode param | 2 |
| `frontend/src/components/research/CenteredActivityPanel.tsx` | Centered display | 3 |
| `frontend/src/components/research/ActivityAccordion.tsx` | Post-completion view | 3 |

## Implementation Patterns

### Adding QueryMode Enum

```python
# src/models/enums.py
from enum import Enum

class QueryMode(str, Enum):
    SIMPLE = "simple"
    WEB_SEARCH = "web_search"
    DEEP_RESEARCH = "deep_research"
```

### Updating ResearchState

```python
# src/agent/state.py
@dataclass
class ResearchState:
    # ... existing fields ...
    query_mode: str = "deep_research"  # Default for backward compatibility
```

### Mode Routing in Orchestrator (Reuse Approach)

```python
# src/agent/orchestrator.py - ~50 lines of routing logic
async def stream_research(..., query_mode: str = "deep_research"):
    state = ResearchState(query=query, query_mode=query_mode, ...)
    mode_config = get_query_mode_config(query_mode)

    if query_mode == QueryMode.SIMPLE:
        # Existing path - direct LLM response (100% existing code)
        async for chunk in stream_simple_response(state, llm):
            yield SynthesisProgressEvent(content=chunk)
        return

    elif query_mode == QueryMode.WEB_SEARCH:
        # Reuse existing components with minimal configuration
        # 1. Create minimal 1-step plan programmatically
        state.current_plan = Plan(
            title="Quick Web Search",
            thought="Answering query with web search",
            steps=[PlanStep(
                title="Search and answer",
                description=f"Find information about: {query}",
                needs_search=True
            )]
        )
        # 2. Reuse researcher (mode=classic, 2 queries, 3 crawls)
        state = await run_researcher(state, llm, crawler, brave_client)

        # 3. Skip reflector - always complete after 1 step
        state.mark_step_complete(0, observation=state.last_observation)

        # 4. Reuse synthesizer (generation_mode=natural for [1],[2] citations)
        async for event in stream_synthesis_with_citations(state, llm):
            yield event

        # 5. Persist lightweight session
        await persist_web_search_session(state)
        return

    else:  # DEEP_RESEARCH
        # Existing full pipeline - unchanged
        async for event in _stream_deep_research(state):
            yield event
```

### Frontend QueryModeSelector

```tsx
// frontend/src/components/chat/QueryModeSelector.tsx
import { QueryMode } from '@/types';

interface QueryModeSelectorProps {
  value: QueryMode;
  onChange: (mode: QueryMode) => void;
  disabled?: boolean;
}

export function QueryModeSelector({ value, onChange, disabled }: QueryModeSelectorProps) {
  return (
    <div className="flex gap-1 p-1 bg-muted rounded-lg">
      {(['simple', 'web_search', 'deep_research'] as QueryMode[]).map((mode) => (
        <button
          key={mode}
          onClick={() => onChange(mode)}
          disabled={disabled}
          className={cn(
            'px-3 py-1.5 text-sm rounded-md transition-colors',
            value === mode
              ? 'bg-background shadow-sm'
              : 'hover:bg-background/50'
          )}
        >
          {formatModeLabel(mode)}
        </button>
      ))}
    </div>
  );
}

function formatModeLabel(mode: QueryMode): string {
  switch (mode) {
    case 'simple': return 'Simple';
    case 'web_search': return 'Web Search';
    case 'deep_research': return 'Deep Research';
  }
}
```

### Progressive Disclosure

```tsx
// In MessageInput.tsx
const [queryMode, setQueryMode] = useState<QueryMode>('simple');

return (
  <form>
    <QueryModeSelector value={queryMode} onChange={setQueryMode} />

    {/* Show depth selector only for deep research */}
    {queryMode === 'deep_research' && (
      <ResearchDepthSelector
        value={researchDepth}
        onChange={setResearchDepth}
      />
    )}

    <textarea ... />
    <button type="submit">Send</button>
  </form>
);
```

## Testing

### Unit Tests

```bash
# Run mode routing tests
uv run pytest tests/unit/agent/test_query_mode_routing.py -v

# Run event service tests
uv run pytest tests/unit/services/test_research_event_service.py -v
```

### Integration Tests

```bash
# Test all modes end-to-end
uv run pytest tests/integration/test_tiered_modes.py -v
```

### Frontend Tests

```bash
cd frontend
npm run test -- QueryModeSelector
```

### E2E Tests

```bash
make e2e  # Runs all E2E tests including tiered-modes.spec.ts
```

## Debugging Tips

### Check Query Mode Flow

```python
# Add to orchestrator for debugging
import logging
logger = logging.getLogger(__name__)

async def stream_research(..., query_mode: str):
    logger.info(f"Processing query with mode: {query_mode}")
    # ...
```

### Verify Event Persistence

```sql
-- Check events stored for a session
SELECT event_type, timestamp, payload
FROM research_events
WHERE research_session_id = 'your-session-id'
ORDER BY timestamp;
```

### Test Mode Selector in Browser

1. Open DevTools Console
2. Run: `localStorage.setItem('queryMode', 'web_search')`
3. Refresh page - mode should be restored

## Common Issues

### Issue: Depth selector shows for non-Deep Research modes

**Fix**: Ensure conditional rendering in MessageInput:
```tsx
{queryMode === 'deep_research' && <ResearchDepthSelector ... />}
```

### Issue: Simple mode still creates research session

**Fix**: Check orchestrator routing - Simple mode should skip session creation:
```python
if query_mode == QueryMode.SIMPLE:
    # Direct LLM call, no session (reuses existing is_simple_query path)
    async for chunk in stream_simple_response(state, llm):
        yield SynthesisProgressEvent(content=chunk)
    return
```

### Issue: Web Search mode too slow (>15s)

**Fix**: Verify researcher config limits are applied:
```python
# Ensure mode=classic with limited searches
researcher_config = ResearcherConfig(
    mode=ResearcherMode.CLASSIC,
    max_search_queries=2,
    max_urls_to_crawl=3,
)
```

### Issue: Web Search uses wrong citation style

**Fix**: Verify synthesizer is using natural mode:
```python
# Should output [1], [2] style citations
citation_config = CitationVerificationConfig(
    generation_mode=GenerationMode.NATURAL,
    enable_numeric_qa_verification=False,
    enable_verification_retrieval=False,
)
```

### Issue: Events not showing in accordion

**Fix**: Verify events are being stored and loaded:
1. Check `research_event_service.save_events()` is called
2. Check `include_events=true` query param on messages endpoint
3. Check frontend passes events to ActivityAccordion component

## Next Steps

After implementing core functionality:

1. **Layer 4**: Enhanced Activity Display
   - CenteredActivityPanel component
   - EnhancedEventLabel with context extraction
   - Smooth transitions between centered and accordion views

2. **Layer 5**: Visited Sources Tracking
   - Extend Source model with new fields
   - Track all crawled URLs in tool_result handler
   - Create VisitedSourcesPanel component

3. **Layer 6**: Polish
   - Animation timing (200ms events, 300ms accordion)
   - Performance optimization for Simple mode
   - Comprehensive test coverage
