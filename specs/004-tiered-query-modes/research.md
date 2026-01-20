# Research: Tiered Query Modes

**Feature**: 004-tiered-query-modes
**Date**: 2026-01-04
**Status**: Complete

## Research Questions & Decisions

### 1. QueryMode Enum Design

**Question**: How should the QueryMode enum be structured to follow existing patterns?

**Decision**: Follow the `ResearchDepth` pattern using `str` subclass enum.

**Rationale**:
- Consistent with existing `ResearchDepth(str, Enum)` pattern in `src/models/research_session.py`
- Enables direct string serialization in Pydantic models and API responses
- Works seamlessly with SQLAlchemy string columns
- Frontend can use string literals directly

**Implementation**:
```python
class QueryMode(str, Enum):
    SIMPLE = "simple"
    WEB_SEARCH = "web_search"
    DEEP_RESEARCH = "deep_research"
```

**Alternatives Considered**:
- IntEnum: Rejected - less readable in logs/responses
- Plain strings: Rejected - no type safety
- Separate TypedDict: Rejected - overkill for 3 options

---

### 2. Web Search Mode Architecture

**Question**: How should Web Search mode be implemented without duplicating Deep Research complexity?

**Decision**: Reuse existing researcher (classic mode) and synthesizer (natural mode) with minimal configuration, avoiding a new agent class.

**Rationale**:
- Web Search mode needs to complete in <15 seconds (FR-007)
- Existing `researcher.py` has `mode=classic` for single-pass execution
- Existing `synthesizer.py` has `generation_mode=natural` for [1], [2] citations
- Maximum code reuse: ~50 lines of routing logic vs ~500 lines for new agent
- Battle-tested components with known performance characteristics

**Implementation** (reuse existing components):
```python
# In orchestrator.py - ~50 lines of routing logic
async def stream_research(query, query_mode="deep_research", ...):
    if query_mode == QueryMode.SIMPLE:
        # Existing path - direct LLM response
        async for chunk in stream_simple_response(state, llm):
            yield SynthesisProgressEvent(content=chunk)
        return

    elif query_mode == QueryMode.WEB_SEARCH:
        # Create minimal 1-step plan
        state.current_plan = Plan(
            title="Quick Web Search",
            steps=[PlanStep(title="Search and answer", needs_search=True)]
        )
        # Reuse researcher with classic mode, limited searches
        state = await run_researcher(state, llm, crawler, brave_client)
        # Reuse synthesizer with natural mode ([1], [2] citations)
        async for event in stream_synthesis_with_citations(state, llm):
            yield event
        return

    else:  # DEEP_RESEARCH - full pipeline unchanged
        async for event in _stream_full_research(state, config, ...):
            yield event
```

**Data Flow**:
1. User submits query with `query_mode=web_search`
2. Orchestrator creates 1-step plan programmatically (skips planner LLM call)
3. Orchestrator calls `run_researcher()` with `mode=classic`, 2 queries, 3 crawls max
4. Orchestrator skips reflector (always COMPLETE after 1 step)
5. Orchestrator calls `stream_synthesis()` with `generation_mode=natural`
6. Lightweight ResearchSession created with sources only

**Configuration**:
```yaml
# config/app.yaml
query_modes:
  web_search:
    researcher:
      mode: classic
      max_search_queries: 2
      max_urls_to_crawl: 3
    citation_verification:
      enabled: true
      generation_mode: natural  # Outputs [1], [2] style
      enable_numeric_qa_verification: false
      enable_verification_retrieval: false
```

**Alternatives Considered**:
- Create new `WebSearchAgent` class: Rejected - duplicates existing logic, ~500 lines new code
- Create separate API endpoint: Rejected - complicates frontend logic
- Use coordinator's simple_query path: Rejected - that's for LLM-only responses (no sources)

---

### 3. Activity Panel Positioning Strategy

**Question**: How to implement centered activity panel during research with smooth transition to accordion?

**Decision**: Use React portal with CSS transforms for positioning, transition to inline accordion on completion.

**Rationale**:
- Portals allow positioning relative to viewport without DOM restructuring
- CSS transforms enable GPU-accelerated animations
- Inline accordion in message card follows existing AgentMessage structure
- Sticky positioning via `position: sticky` for scroll behavior

**Implementation**:
```tsx
// During research: CenteredActivityPanel.tsx
<div className="fixed inset-x-0 top-1/3 mx-auto max-w-xl z-50
                transition-all duration-300 ease-out">
  <ActivityPanel events={events} isLive={true} />
</div>

// After completion: ActivityAccordion.tsx (inside AgentMessage)
<Accordion defaultOpen={false}>
  <AccordionTrigger>Research Activity ({events.length} events)</AccordionTrigger>
  <AccordionContent>
    <EventList events={events} />
  </AccordionContent>
</Accordion>
```

**Transition Logic**:
1. `isStreaming=true` → Show `CenteredActivityPanel`
2. `research_completed` event received → Animate panel exit (scale/fade)
3. Message renders with `ActivityAccordion` containing same events
4. No duplicate events - state passes from hook to both components

**Alternatives Considered**:
- Replace sidebar entirely: Rejected - breaks existing layout for Deep Research
- Modal overlay: Rejected - blocks content, poor UX
- Bottom sheet: Rejected - too mobile-specific

---

### 4. Event Persistence Schema

**Question**: How to store ResearchEvent records for accordion display after page reload?

**Decision**: New `research_events` table linked to `research_session_id` with JSONB payload.

**Rationale**:
- Events need to persist for accordion display on page reload (FR-034)
- JSONB payload allows flexible event-specific data without schema changes per event type
- Cascade delete when chat deleted matches retention policy
- Index on research_session_id for fast retrieval

**Schema**:
```sql
CREATE TABLE research_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    research_session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_research_events_session ON research_events(research_session_id);
CREATE INDEX ix_research_events_timestamp ON research_events(research_session_id, timestamp);
```

**Payload Examples**:
```json
// claim_verified
{"claim_id": "uuid", "claim_text": "Revenue was $50B", "verdict": "supported", "confidence": "high"}

// tool_call (web_search)
{"tool_name": "web_search", "query": "Apple 2024 revenue"}

// step_started
{"step_index": 2, "step_title": "Financial Performance", "step_description": "..."}
```

**Alternatives Considered**:
- Store in research_session.metadata JSONB: Rejected - unbounded growth, harder to query
- Separate table per event type: Rejected - too many tables, complex joins
- Store in message.metadata: Rejected - no research_session for Simple mode

---

### 5. Visited Sources Tracking

**Question**: How to distinguish between all visited URLs and cited sources?

**Decision**: Extend existing `sources` table with `is_cited` flag and `step_reference` field.

**Rationale**:
- Sources table already captures URL, title, snippet, content
- Adding `is_cited` flag simpler than new table with duplicate columns
- `step_reference` enables grouping by research step (FR-039)
- Existing cascade delete on research_session preserves retention policy

**Schema Changes**:
```sql
ALTER TABLE sources ADD COLUMN is_cited BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE sources ADD COLUMN step_index INTEGER;
ALTER TABLE sources ADD COLUMN step_title VARCHAR(255);
ALTER TABLE sources ADD COLUMN crawl_status VARCHAR(20) DEFAULT 'success';
ALTER TABLE sources ADD COLUMN error_reason TEXT;
```

**Display Logic**:
1. "Cited Sources" = sources WHERE is_cited = true (existing behavior)
2. "All Visited Sources" = all sources, grouped by step_index
3. Cross-reference: Show "(cited)" badge for is_cited=true in visited list

**Alternatives Considered**:
- New visited_sources table: Rejected - duplicates sources schema
- Track in research_session.metadata: Rejected - hard to query/display
- Frontend-only tracking: Rejected - doesn't persist on reload (FR-045)

---

### 6. Mode-Specific Event Emission

**Question**: Which SSE events should be emitted for each query mode?

**Decision**: Define event profiles per mode, filter in orchestrator.

| Event Type | Simple | Web Search | Deep Research |
|------------|--------|------------|---------------|
| agent_started | No | Yes (synthesizer only) | Yes (all agents) |
| agent_completed | No | Yes (synthesizer only) | Yes (all agents) |
| plan_created | No | No | Yes |
| step_started | No | No | Yes |
| step_completed | No | No | Yes |
| tool_call | No | Yes (search only) | Yes |
| tool_result | No | Yes (search only) | Yes |
| reflection_decision | No | No | Yes |
| synthesis_started | No | Yes | Yes |
| synthesis_progress | Yes (streaming) | Yes | Yes |
| claim_verified | No | No | Yes |
| verification_summary | No | No | Yes |
| research_completed | No | Yes | Yes |

**Rationale**:
- Simple mode: Minimal events, just stream the response
- Web Search: Show search activity, skip pipeline events
- Deep Research: Full event set for transparency

**Implementation**:
```python
def should_emit_event(event: StreamEvent, mode: QueryMode) -> bool:
    if mode == QueryMode.SIMPLE:
        return event.event_type == "synthesis_progress"
    elif mode == QueryMode.WEB_SEARCH:
        return event.event_type in WEB_SEARCH_EVENTS
    return True  # Deep Research emits all
```

---

### 7. Frontend State Management

**Question**: How to manage query mode state across components?

**Decision**: Create `useQueryMode` hook with localStorage persistence for session, user preferences for defaults.

**Rationale**:
- Mode selection needs to persist within session (FR-011)
- Default mode comes from user preferences (FR-014)
- React Context would add complexity for simple state
- localStorage provides immediate persistence without API call

**Implementation**:
```typescript
export function useQueryMode() {
  const { preferences } = useUserPreferences();
  const [mode, setMode] = useState<QueryMode>(() => {
    const saved = localStorage.getItem('queryMode');
    return saved as QueryMode || preferences?.defaultQueryMode || 'simple';
  });

  const setQueryMode = useCallback((newMode: QueryMode) => {
    setMode(newMode);
    localStorage.setItem('queryMode', newMode);
  }, []);

  return { mode, setQueryMode, isDeepResearch: mode === 'deep_research' };
}
```

**Integration**:
- `MessageInput` uses hook to show/hide depth selector
- `useStreamingQuery.sendQuery(query, researchDepth, queryMode)` passes mode
- `ChatPage` conditionally renders centered panel based on mode

---

### 8. Enhanced Event Labels

**Question**: How to format event labels with contextual data?

**Decision**: Extend `activityLabels.ts` with event-specific formatters that extract payload data.

**Rationale**:
- Existing `formatActivityLabel()` handles basic labels
- Need to extract claim_text, query, step_title from event payload
- Keep formatting logic centralized for consistency
- Support truncation (60 chars for claims per FR-024)

**Implementation**:
```typescript
export function formatEnhancedEventLabel(event: StreamEvent): EnhancedLabel {
  switch (event.event_type) {
    case 'claim_verified':
      return {
        icon: getVerdictIcon(event.verdict),
        text: `Claim Verified: ${truncate(event.claimText, 60)}`,
        badge: { text: event.verdict, color: getVerdictColor(event.verdict) }
      };
    case 'numeric_claim_detected':
      return {
        icon: HashIcon,
        text: `Numeric: ${event.rawValue} ${event.unit || ''}`,
        badge: event.qaVerified ? { text: 'QA Verified', color: 'green' } : null
      };
    case 'tool_call':
      if (event.toolName === 'web_search') {
        return { icon: SearchIcon, text: `Searching: ${event.query}` };
      }
      return { icon: GlobeIcon, text: 'Crawling page...' };
    // ... other cases
  }
}
```

---

### 9. Accordion Virtualization

**Question**: How to handle accordion with hundreds of events (FR-036)?

**Decision**: Use `react-window` for virtualized list when events > 50.

**Rationale**:
- 50 events threshold from FR-036
- `react-window` is lightweight (3KB gzipped), well-tested
- Pairs well with TanStack Query already in use
- "Load more" pagination via `InfiniteLoader`

**Implementation**:
```typescript
import { FixedSizeList } from 'react-window';

function VirtualizedEventList({ events }: { events: StreamEvent[] }) {
  const [visibleCount, setVisibleCount] = useState(50);
  const visibleEvents = events.slice(0, visibleCount);

  if (events.length <= 50) {
    return <SimpleEventList events={events} />;
  }

  return (
    <>
      <FixedSizeList
        height={400}
        itemCount={visibleEvents.length}
        itemSize={40}
        width="100%"
      >
        {({ index, style }) => (
          <div style={style}>
            <EnhancedEventLabel event={visibleEvents[index]} />
          </div>
        )}
      </FixedSizeList>
      {visibleCount < events.length && (
        <button onClick={() => setVisibleCount(c => c + 50)}>
          Load more ({events.length - visibleCount} remaining)
        </button>
      )}
    </>
  );
}
```

---

### 10. Configuration Structure

**Question**: How to configure mode-specific settings in app.yaml?

**Decision**: Add `query_modes` section with per-mode configuration.

**Rationale**:
- Follows existing `research_types` pattern
- Centralizes all mode-specific settings
- Enables runtime configuration without code changes

**Implementation**:
```yaml
query_modes:
  simple:
    enabled: true
    model_role: simple  # Uses fast model tier
    timeout_seconds: 10
    emit_events: false

  web_search:
    enabled: true
    model_role: analytical
    timeout_seconds: 15
    max_searches: 3
    max_crawls_per_search: 2
    emit_search_events: true
    create_session: true  # Lightweight session with sources only

  deep_research:
    enabled: true
    model_role: complex
    # Uses existing research_types config for depth-specific settings
    emit_all_events: true
```

**Accessor**:
```python
def get_query_mode_config(mode: QueryMode) -> QueryModeConfig:
    app_config = get_app_config()
    return app_config.query_modes.get(mode.value, DEFAULT_MODE_CONFIG)
```

---

## Dependencies Identified

| Dependency | Purpose | Version |
|------------|---------|---------|
| react-window | Virtualized event list | ^1.8.10 |
| @radix-ui/react-accordion | Collapsible accordion | ^1.2.0 |

**Note**: Both packages are commonly used with shadcn/ui which is already in the project.

## Open Questions (None)

All clarifications resolved during spec phase. No blocking unknowns remain.

## Next Steps

1. Proceed to Phase 1: Generate data-model.md
2. Generate API contracts (openapi-patch.yaml)
3. Generate quickstart.md
4. Update agent context
