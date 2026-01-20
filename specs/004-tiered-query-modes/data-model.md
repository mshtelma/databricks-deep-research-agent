# Data Model: Tiered Query Modes

**Feature**: 004-tiered-query-modes
**Date**: 2026-01-04
**Status**: Complete

## Entity Overview

This feature introduces:
1. **QueryMode** - New enum for query mode classification
2. **ResearchEvent** - New entity for activity event persistence
3. Schema extensions to existing entities (ResearchSession, UserPreferences, Source)

## Entity Definitions

### QueryMode (Enum)

New enumeration for query mode classification.

| Value | Description | Creates Session | Pipeline |
|-------|-------------|-----------------|----------|
| `simple` | LLM-only response | No | Direct LLM call |
| `web_search` | Quick sourced answer | Yes (lightweight) | WebSearchAgent |
| `deep_research` | Full research pipeline | Yes (full) | Full orchestrator |

**Backend (Python)**:
```python
class QueryMode(str, Enum):
    SIMPLE = "simple"
    WEB_SEARCH = "web_search"
    DEEP_RESEARCH = "deep_research"
```

**Frontend (TypeScript)**:
```typescript
export type QueryMode = 'simple' | 'web_search' | 'deep_research';
```

---

### ResearchEvent (New Entity)

Stores individual activity events for accordion display persistence.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, auto-generated | Unique identifier |
| `research_session_id` | UUID | FK → research_sessions, NOT NULL | Parent session |
| `event_type` | VARCHAR(50) | NOT NULL | Event type (e.g., "claim_verified") |
| `timestamp` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | When event occurred |
| `payload` | JSONB | NOT NULL, DEFAULT '{}' | Event-specific data |
| `created_at` | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Record creation time |

**Relationships**:
- Many-to-one with ResearchSession (cascade delete)

**Indexes**:
- `ix_research_events_session` on (research_session_id)
- `ix_research_events_timestamp` on (research_session_id, timestamp)

**Payload Schema by Event Type**:

```typescript
interface ClaimVerifiedPayload {
  claim_id: string;
  claim_text: string;
  verdict: 'supported' | 'partial' | 'unsupported' | 'contradicted';
  confidence: 'high' | 'medium' | 'low';
  evidence_preview?: string;
}

interface NumericClaimPayload {
  claim_id: string;
  raw_value: string;
  normalized_value?: number;
  unit?: string;
  derivation_type: 'direct' | 'calculated' | 'inferred';
  qa_verified: boolean;
}

interface ToolCallPayload {
  tool_name: 'web_search' | 'web_crawl';
  query?: string;  // For web_search
  url?: string;    // For web_crawl
}

interface StepPayload {
  step_index: number;
  step_title: string;
  step_description?: string;
}

interface VerificationSummaryPayload {
  total_claims: number;
  supported: number;
  partial: number;
  unsupported: number;
  contradicted: number;
}
```

**SQLAlchemy Model**:
```python
class ResearchEvent(Base):
    __tablename__ = "research_events"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    research_session_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC)
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC)
    )

    # Relationships
    research_session: Mapped["ResearchSession"] = relationship(
        back_populates="events"
    )
```

---

### ResearchSession (Extended)

Add `query_mode` field to existing entity.

| New Field | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `query_mode` | VARCHAR(20) | NOT NULL, DEFAULT 'deep_research' | Mode used for this session |

**Schema Change**:
```sql
ALTER TABLE research_sessions
ADD COLUMN query_mode VARCHAR(20) NOT NULL DEFAULT 'deep_research';
```

**Updated Relationships**:
- One-to-many with ResearchEvent (new)

**SQLAlchemy Update**:
```python
class ResearchSession(Base):
    # ... existing fields ...

    query_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="deep_research"
    )

    # New relationship
    events: Mapped[list["ResearchEvent"]] = relationship(
        back_populates="research_session",
        cascade="all, delete-orphan",
        order_by="ResearchEvent.timestamp"
    )
```

---

### UserPreferences (Extended)

Add `default_query_mode` field for mode persistence.

| New Field | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `default_query_mode` | VARCHAR(20) | NOT NULL, DEFAULT 'simple' | User's default mode |

**Schema Change**:
```sql
ALTER TABLE user_preferences
ADD COLUMN default_query_mode VARCHAR(20) NOT NULL DEFAULT 'simple';
```

**SQLAlchemy Update**:
```python
class UserPreferences(Base):
    # ... existing fields ...

    default_query_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="simple"
    )
```

---

### Source (Extended)

Add fields for visited sources tracking.

| New Field | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `is_cited` | BOOLEAN | NOT NULL, DEFAULT FALSE | Whether source is cited in report |
| `step_index` | INTEGER | NULL | Research step that visited this source |
| `step_title` | VARCHAR(255) | NULL | Title of research step |
| `crawl_status` | VARCHAR(20) | NOT NULL, DEFAULT 'success' | Status of crawl attempt |
| `error_reason` | TEXT | NULL | Error message if crawl failed |

**Schema Changes**:
```sql
ALTER TABLE sources ADD COLUMN is_cited BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE sources ADD COLUMN step_index INTEGER;
ALTER TABLE sources ADD COLUMN step_title VARCHAR(255);
ALTER TABLE sources ADD COLUMN crawl_status VARCHAR(20) NOT NULL DEFAULT 'success';
ALTER TABLE sources ADD COLUMN error_reason TEXT;

CREATE INDEX ix_sources_is_cited ON sources(research_session_id, is_cited);
CREATE INDEX ix_sources_step ON sources(research_session_id, step_index);
```

**SQLAlchemy Update**:
```python
class Source(Base):
    # ... existing fields ...

    is_cited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    step_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    step_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    crawl_status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    error_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
```

---

## Entity Relationships Diagram

```
┌─────────────────┐
│      Chat       │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐
│     Message     │
└────────┬────────┘
         │ 1:1
         ▼
┌─────────────────────────────────────┐
│         ResearchSession             │
│  + query_mode (NEW)                 │
└────────┬────────────────┬───────────┘
         │ 1:N            │ 1:N
         ▼                ▼
┌─────────────────┐  ┌─────────────────────────────────┐
│ ResearchEvent   │  │           Source                │
│    (NEW)        │  │  + is_cited (NEW)               │
│                 │  │  + step_index (NEW)             │
│ - event_type    │  │  + step_title (NEW)             │
│ - timestamp     │  │  + crawl_status (NEW)           │
│ - payload       │  │  + error_reason (NEW)           │
└─────────────────┘  └─────────────────────────────────┘

┌─────────────────────────────────┐
│       UserPreferences           │
│  + default_query_mode (NEW)     │
└─────────────────────────────────┘
```

---

## State Transitions

### QueryMode Selection Flow

```
User Opens Chat
    │
    ▼
┌─────────────────────────────────────┐
│ Load default_query_mode from:       │
│ 1. localStorage (session override)  │
│ 2. UserPreferences (user default)   │
│ 3. "simple" (system default)        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Display QueryModeSelector           │
│ [Simple] [Web Search] [Deep ▼]      │
└─────────────────┬───────────────────┘
                  │ User selects mode
                  ▼
┌─────────────────────────────────────┐
│ If "Deep Research":                 │
│   Show ResearchDepthSelector        │
│   [Auto] [Light] [Medium] [Extended]│
└─────────────────┬───────────────────┘
                  │ User submits query
                  ▼
┌─────────────────────────────────────┐
│ API Request:                        │
│ GET /chats/{id}/stream              │
│   ?query=...                        │
│   &query_mode=web_search            │
│   &research_depth=medium (if DR)    │
└─────────────────────────────────────┘
```

### ResearchSession Lifecycle by Mode

```
Simple Mode (reuses existing is_simple_query path):
    Query → Direct LLM Response → Done (no session)

Web Search Mode (reuses researcher + synthesizer):
    Query → Create Lightweight Session
        → Create 1-step plan programmatically
        → Researcher (mode=classic, 2 queries, 3 crawls)
        → Skip Reflector (always COMPLETE after 1 step)
        → Synthesizer (generation_mode=natural for [1],[2] citations)
        → Store Sources (is_cited=true for used)
        → Store Events (tool_call, synthesis_progress)
        → Complete Session

Deep Research Mode (unchanged full pipeline):
    Query → Create Full Session
        → Coordinator
        → Background Investigator
        → Planner
        → Research Loop (Researcher ↔ Reflector)
        → Synthesizer
        → Citation Verification (7 stages)
        → Store All Sources (is_cited based on claims)
        → Store All Events
        → Complete Session
```

---

## Validation Rules

### QueryMode
- Must be one of: `simple`, `web_search`, `deep_research`
- Cannot be null when submitting a query

### ResearchEvent
- `event_type` must be a valid SSE event type
- `payload` must be valid JSON matching event type schema
- `timestamp` must not be in the future

### Source (extended fields)
- `crawl_status` must be one of: `success`, `failed`, `timeout`, `blocked`
- `error_reason` required when `crawl_status != 'success'`
- `step_index` must be >= 0 when present

### UserPreferences (extended)
- `default_query_mode` must be valid QueryMode value

---

## Migration Strategy

**Migration File**: `005_tiered_query_modes.py`

**Operations**:
1. Add `query_mode` column to `research_sessions` table
2. Add `default_query_mode` column to `user_preferences` table
3. Add visited sources columns to `sources` table
4. Create `research_events` table
5. Create indexes

**Rollback Strategy**:
1. Drop `research_events` table
2. Drop new columns from `sources`, `research_sessions`, `user_preferences`
3. Drop indexes

**Data Migration**:
- Existing research_sessions: Set `query_mode = 'deep_research'` (default)
- Existing user_preferences: Set `default_query_mode = 'simple'` (default)
- Existing sources: Set `is_cited = true`, `crawl_status = 'success'`

---

## Frontend Types

```typescript
// types/index.ts additions

export type QueryMode = 'simple' | 'web_search' | 'deep_research';

export interface ResearchEvent {
  id: string;
  researchSessionId: string;
  eventType: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface VisitedSource extends Source {
  isCited: boolean;
  stepIndex: number | null;
  stepTitle: string | null;
  crawlStatus: 'success' | 'failed' | 'timeout' | 'blocked';
  errorReason: string | null;
}

// Extended ResearchSession
export interface ResearchSession {
  // ... existing fields ...
  queryMode: QueryMode;
  events?: ResearchEvent[];
}

// Extended UserPreferences
export interface UserPreferences {
  // ... existing fields ...
  defaultQueryMode: QueryMode;
}
```

---

## API Response Shapes

### GET /chats/{chat_id}/stream (query params)

```typescript
interface StreamQueryParams {
  query: string;
  query_mode: QueryMode;  // NEW
  research_depth?: ResearchDepth;  // Only for deep_research mode
}
```

### GET /chats/{chat_id}/messages (response)

```typescript
interface MessageResponse {
  // ... existing fields ...
  research_session?: {
    // ... existing fields ...
    query_mode: QueryMode;  // NEW
    events?: ResearchEvent[];  // NEW - for accordion display
  };
}
```

### GET /users/me/preferences (response)

```typescript
interface UserPreferencesResponse {
  // ... existing fields ...
  default_query_mode: QueryMode;  // NEW
}
```

### PUT /users/me/preferences (request)

```typescript
interface UpdatePreferencesRequest {
  // ... existing fields ...
  default_query_mode?: QueryMode;  // NEW
}
```
