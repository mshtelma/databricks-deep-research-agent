# Data Models

## Overview

The Deep Research Agent uses SQLAlchemy models for database persistence and Pydantic models for API schemas. This document describes the entity relationships and key fields.

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA MODEL                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User (Databricks workspace identity)                                        │
│    │                                                                         │
│    ├── Chat (conversation thread)                                            │
│    │     │                                                                   │
│    │     ├── Message (user or agent message)                                 │
│    │     │     │                                                             │
│    │     │     ├── Claim (atomic factual assertion)                          │
│    │     │     │     │                                                       │
│    │     │     │     └── Citation (claim-to-evidence mapping)                │
│    │     │     │                                                             │
│    │     │     └── MessageFeedback (user ratings)                            │
│    │     │                                                                   │
│    │     └── ResearchSession (execution context)                             │
│    │           │                                                             │
│    │           ├── Source (web resource)                                     │
│    │           │     │                                                       │
│    │           │     └── EvidenceSpan (minimal supporting quote)             │
│    │           │                                                             │
│    │           └── ResearchEvent (activity log)                              │
│    │                                                                         │
│    └── UserPreferences (system instructions, defaults)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Entities

### Chat

**File**: `src/models/chat.py`

Represents a conversation thread.

```python
class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(500))
    is_deleted: Mapped[bool] = mapped_column(default=False)
    deleted_at: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    messages: Mapped[list["Message"]] = relationship(back_populates="chat")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | str | Databricks workspace user ID |
| `title` | str | Chat title (auto-generated or user-set) |
| `is_deleted` | bool | Soft delete flag |
| `deleted_at` | datetime | When soft deleted (30-day recovery) |

### Message

**File**: `src/models/chat.py`

Represents a user or agent message.

```python
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(ForeignKey("chats.id"), nullable=False)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    query_mode: Mapped[QueryMode | None] = mapped_column(Enum(QueryMode))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    chat: Mapped["Chat"] = relationship(back_populates="messages")
    research_session: Mapped["ResearchSession | None"] = relationship(back_populates="message")
    claims: Mapped[list["Claim"]] = relationship(back_populates="message")
    feedback: Mapped[list["MessageFeedback"]] = relationship(back_populates="message")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `chat_id` | UUID | Foreign key to chat |
| `role` | MessageRole | "user" or "agent" |
| `content` | str | Message text (NULL for agent placeholder) |
| `query_mode` | QueryMode | "simple", "web_search", "deep_research" |

**Enums**:
```python
class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"

class QueryMode(str, Enum):
    SIMPLE = "simple"
    WEB_SEARCH = "web_search"
    DEEP_RESEARCH = "deep_research"
```

### ResearchSession

**File**: `src/models/research_session.py`

Represents a research execution context.

```python
class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), nullable=False)
    query_mode: Mapped[QueryMode] = mapped_column(Enum(QueryMode), nullable=False)
    research_depth: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[SessionStatus] = mapped_column(Enum(SessionStatus), nullable=False)
    classification: Mapped[dict] = mapped_column(JSONB, default=dict)
    plan: Mapped[dict | None] = mapped_column(JSONB)
    verification_summary: Mapped[dict | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column()

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="research_session")
    sources: Mapped[list["Source"]] = relationship(back_populates="research_session")
    events: Mapped[list["ResearchEvent"]] = relationship(back_populates="research_session")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `message_id` | UUID | Foreign key to agent message |
| `query_mode` | QueryMode | Mode used for this research |
| `research_depth` | str | "auto", "light", "medium", "extended" |
| `status` | SessionStatus | IN_PROGRESS, COMPLETED, FAILED |
| `classification` | dict | Query classification JSON |
| `plan` | dict | Research plan JSON |
| `verification_summary` | dict | Citation verification stats |

**Session Status**:
```python
class SessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Source

**File**: `src/models/research_session.py`

Represents a web resource found during research.

```python
class Source(Base):
    __tablename__ = "sources"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    research_session_id: Mapped[UUID] = mapped_column(
        ForeignKey("research_sessions.id"), nullable=False
    )
    chat_id: Mapped[UUID | None] = mapped_column(ForeignKey("chats.id"))
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    is_cited: Mapped[bool] = mapped_column(default=False)
    step_index: Mapped[int | None] = mapped_column()
    step_title: Mapped[str | None] = mapped_column(String(500))
    crawl_status: Mapped[str] = mapped_column(String(50), default="success")
    error_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    research_session: Mapped["ResearchSession"] = relationship(back_populates="sources")
    evidence_spans: Mapped[list["EvidenceSpan"]] = relationship(back_populates="source")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("research_session_id", "url", name="uq_source_session_url"),
    )
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `research_session_id` | UUID | Foreign key to session |
| `url` | str | Source URL |
| `title` | str | Page title |
| `snippet` | str | Brave Search snippet |
| `content` | str | Full page content (if crawled) |
| `is_cited` | bool | Whether source has citations |
| `crawl_status` | str | "success", "failed", "timeout", "blocked" |

### EvidenceSpan

**File**: `src/models/research_session.py`

Represents a minimal supporting quote from a source.

```python
class EvidenceSpan(Base):
    __tablename__ = "evidence_spans"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("sources.id"), nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    start_offset: Mapped[int | None] = mapped_column()
    end_offset: Mapped[int | None] = mapped_column()
    section_heading: Mapped[str | None] = mapped_column(String(500))
    relevance_score: Mapped[float] = mapped_column(default=0.0)
    has_numeric_content: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    source: Mapped["Source"] = relationship(back_populates="evidence_spans")
    citations: Mapped[list["Citation"]] = relationship(back_populates="evidence_span")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `source_id` | UUID | Foreign key to source |
| `quote` | str | The evidence text |
| `relevance_score` | float | 0.0-1.0 relevance to query |
| `has_numeric_content` | bool | Contains statistics |

## Citation Entities

### Claim

**File**: `src/models/research_session.py`

Represents an atomic factual assertion in the report.

```python
class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(50), nullable=False)
    position_start: Mapped[int] = mapped_column(nullable=False)
    position_end: Mapped[int] = mapped_column(nullable=False)
    citation_key: Mapped[str | None] = mapped_column(String(100))
    verdict: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(default=0.0)
    verification_reasoning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="claims")
    citations: Mapped[list["Citation"]] = relationship(back_populates="claim")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `message_id` | UUID | Foreign key to agent message |
| `claim_text` | str | The claim text |
| `claim_type` | str | "general" or "numeric" |
| `position_start` | int | Start position in report |
| `position_end` | int | End position in report |
| `citation_key` | str | Human-readable key `[Arxiv]` |
| `verdict` | str | SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED |

### Citation

**File**: `src/models/research_session.py`

Represents a claim-to-evidence mapping.

```python
class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    claim_id: Mapped[UUID] = mapped_column(ForeignKey("claims.id"), nullable=False)
    evidence_span_id: Mapped[UUID] = mapped_column(
        ForeignKey("evidence_spans.id"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(default=0.0)
    is_primary: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    claim: Mapped["Claim"] = relationship(back_populates="citations")
    evidence_span: Mapped["EvidenceSpan"] = relationship(back_populates="citations")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `claim_id` | UUID | Foreign key to claim |
| `evidence_span_id` | UUID | Foreign key to evidence |
| `confidence` | float | 0.0-1.0 confidence score |
| `is_primary` | bool | Primary vs alternate citation |

## User Entities

### UserPreferences

**File**: `src/models/chat.py`

Stores user preferences and settings.

```python
class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    system_instructions: Mapped[str | None] = mapped_column(Text)
    default_query_mode: Mapped[QueryMode | None] = mapped_column(Enum(QueryMode))
    default_research_depth: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC)
    )
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | str | Databricks user ID (unique) |
| `system_instructions` | str | Custom instructions for agents |
| `default_query_mode` | QueryMode | Default mode for new queries |
| `default_research_depth` | str | Default depth for deep research |

### MessageFeedback

**File**: `src/models/chat.py`

Stores user feedback on agent messages.

```python
class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    rating: Mapped[str] = mapped_column(String(50), nullable=False)  # positive/negative
    feedback_text: Mapped[str | None] = mapped_column(Text)
    mlflow_trace_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="feedback")
```

**Key Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `message_id` | UUID | Foreign key to message |
| `user_id` | str | User who gave feedback |
| `rating` | str | "positive" or "negative" |
| `feedback_text` | str | Optional text feedback |
| `mlflow_trace_id` | str | Correlation to MLflow trace |

## Activity Tracking

### ResearchEvent

**File**: `src/models/research_event.py`

Stores research activity events for display.

```python
class ResearchEvent(Base):
    __tablename__ = "research_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    research_session_id: Mapped[UUID] = mapped_column(
        ForeignKey("research_sessions.id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    timestamp: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # Relationships
    research_session: Mapped["ResearchSession"] = relationship(back_populates="events")
```

**Event Types**:
| Event Type | Description |
|------------|-------------|
| `step_started` | Research step beginning |
| `tool_call` | Tool invocation |
| `tool_result` | Tool completion |
| `step_completed` | Research step finished |
| `reflection_decision` | CONTINUE/ADJUST/COMPLETE |
| `claim_verified` | Claim verification result |
| `verification_summary` | Overall verification stats |

## Pydantic Schemas

API request/response schemas are in `src/schemas/`:

### Common Schemas

**File**: `src/schemas/common.py`

```python
class BaseSchema(BaseModel):
    """Base schema with camelCase aliases."""
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )

class ChatResponse(BaseSchema):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

class MessageResponse(BaseSchema):
    id: UUID
    role: str
    content: str | None
    query_mode: str | None
    created_at: datetime
```

### Streaming Schemas

**File**: `src/schemas/streaming.py`

```python
class ResearchStartedEvent(BaseSchema):
    event_type: Literal["research_started"] = "research_started"
    session_id: UUID
    message_id: UUID

class StepStartedEvent(BaseSchema):
    event_type: Literal["step_started"] = "step_started"
    step_index: int
    step_title: str

class SynthesisProgressEvent(BaseSchema):
    event_type: Literal["synthesis_progress"] = "synthesis_progress"
    content_chunk: str

class ClaimVerifiedEvent(BaseSchema):
    event_type: Literal["claim_verified"] = "claim_verified"
    claim_text: str
    verdict: str
    confidence: float
```

## Database Migrations

Migrations are managed with Alembic in `src/db/migrations/`:

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

**Key Migrations**:
| Version | Description |
|---------|-------------|
| `001_initial` | Core tables (chat, message) |
| `002_research_session` | Research sessions and sources |
| `003_citations` | Claims, citations, evidence spans |
| `004_user_preferences` | User preferences and feedback |
| `005_add_source_url_unique` | Unique constraint on source URL |
| `006_add_chat_id_to_sources` | Chat-level source pool |
| `007_tiered_query_modes` | Query mode fields |
| `008_research_session_lifecycle` | Session status tracking |
| `009_background_research_jobs` | Background job support |

## See Also

- [API Reference](./api.md) - REST endpoints
- [Architecture](./architecture.md) - System overview
- [Citation Pipeline](./citation-pipeline.md) - Verification process
