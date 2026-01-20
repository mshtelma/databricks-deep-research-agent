# System Architecture

## Overview

The Deep Research Agent is a production-grade multi-agent system built on Databricks infrastructure. It combines intelligent web research with claim-level citation verification using a 5-agent orchestration architecture and 7-stage verification pipeline.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DEEP RESEARCH AGENT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Frontend  │    │                 5-Agent Orchestrator                │ │
│  │  React 18   │◀──▶│                                                     │ │
│  │  TanStack   │SSE │  Coordinator → Planner → Researcher → Reflector    │ │
│  │  Tailwind   │    │                              ↓                      │ │
│  └─────────────┘    │                         Synthesizer                 │ │
│                     │                              ↓                      │ │
│  ┌─────────────┐    │                   7-Stage Citation Pipeline         │ │
│  │  FastAPI    │    │                                                     │ │
│  │  REST API   │◀──▶│                                                     │ │
│  │  /v1/...    │    └─────────────────────────────────────────────────────┘ │
│  └─────────────┘                                                            │
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │ Lakebase    │◀──▶│     Databricks Foundation Model Endpoints           │ │
│  │ PostgreSQL  │    │  Simple (Gemini) │ Analytical (Claude) │ Complex   │ │
│  └─────────────┘    └─────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │ Brave Search│    │              MLflow Tracing (3.8+)                  │ │
│  │     API     │    └─────────────────────────────────────────────────────┘ │
│  └─────────────┘                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Backend Language** | Python | 3.11+ | Async orchestration |
| **Frontend Language** | TypeScript | 5.x | Type-safe UI |
| **Backend Framework** | FastAPI | 0.109+ | REST API + SSE streaming |
| **Frontend Framework** | React | 18.x | Streaming UI |
| **Database** | Databricks Lakebase | Preview | PostgreSQL persistence |
| **LLM Client** | AsyncOpenAI | 1.10+ | Custom base_url to Databricks |
| **HTTP Client** | httpx | 0.26+ | Async web requests |
| **Observability** | MLflow | 3.8+ | Distributed tracing |
| **Web Scraping** | Trafilatura | 2.0+ | Content extraction |
| **Search** | Brave Search API | - | Web search |
| **State Management** | TanStack Query | 5.x | Frontend data fetching |
| **Styling** | Tailwind CSS | 3.4+ | UI components |
| **JSON Repair** | json-repair | 0.25+ | LLM output fixing |

## Key Design Decisions

### 1. Plain Async Python Orchestration

**Decision**: Use explicit async/await over LangGraph/DSPy/AutoGen

**Rationale**:
- Full transparency and control over execution flow
- Easier debugging and tracing
- No hidden abstractions or magic
- Direct state management without framework overhead

**Implementation**: `src/agent/orchestrator.py`

### 2. Step-by-Step Reflection

**Decision**: Reflector evaluates after EACH research step (not batched)

**Rationale**:
- Early termination when sufficient coverage achieved
- Adaptive replanning when plan isn't working
- Finer-grained control over research quality

**Decisions**: CONTINUE / ADJUST / COMPLETE

### 3. Tiered Model Routing

**Decision**: Three model tiers with automatic failover

| Tier | Use Cases | Example |
|------|-----------|---------|
| Simple | Classification, quick decisions | Gemini Flash |
| Analytical | Planning, research, verification | Claude Sonnet |
| Complex | Synthesis, extended reasoning | Claude with ER |

**Rationale**:
- Cost optimization (simple tasks use cheaper models)
- Quality optimization (complex tasks use better models)
- Resilience (failover to alternate endpoints)

### 4. Evidence-First Generation

**Decision**: Pre-select evidence BEFORE synthesis

**Rationale**:
- Constrained generation produces verifiable claims
- Every claim has evidence available for verification
- Prevents hallucination of unsupported facts

**Implementation**: Stage 1 of citation pipeline

### 5. Isolated Verification

**Decision**: Verify claims WITHOUT generation context

**Rationale**:
- Prevents "I generated this so it's true" bias
- Independent verification produces honest verdicts
- Based on CoVe (Chain of Verification) research

**Implementation**: Stage 4 of citation pipeline

## Project Structure

```
├── src/                            # Python backend
│   ├── agent/
│   │   ├── orchestrator.py         # Main async pipeline
│   │   ├── state.py                # ResearchState model
│   │   ├── nodes/                  # 5 agent implementations
│   │   │   ├── coordinator.py      # Query classification
│   │   │   ├── background.py       # Quick web search
│   │   │   ├── planner.py          # Research plan
│   │   │   ├── researcher.py       # Step execution
│   │   │   ├── reflector.py        # Step decisions
│   │   │   └── synthesizer.py      # Report generation
│   │   └── prompts/                # Agent prompt templates
│   ├── api/v1/
│   │   ├── chats.py                # Chat CRUD endpoints
│   │   ├── messages.py             # Message endpoints
│   │   ├── research.py             # SSE streaming research
│   │   ├── citations.py            # Citation verification
│   │   └── utils/                  # Shared API utilities
│   │       ├── authorization.py    # Centralized auth checks
│   │       └── transformers.py     # Response builders
│   ├── services/
│   │   ├── base.py                 # BaseRepository[T] pattern
│   │   ├── loading.py              # Eager-loading options
│   │   ├── llm/                    # LLM client, routing
│   │   ├── citation/               # 7-stage pipeline
│   │   └── search/                 # Brave Search
│   ├── models/                     # SQLAlchemy models
│   ├── core/                       # Config, auth, tracing
│   └── db/                         # Database connection
│
├── frontend/                       # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── chat/               # Chat UI components
│   │   │   └── research/           # Research display
│   │   ├── hooks/                  # useStreamingQuery, etc.
│   │   └── pages/                  # ChatPage
│   └── tests/                      # Vitest tests
│
├── e2e/                            # Playwright E2E tests
├── tests/                          # Python backend tests
│   ├── unit/                       # Mocked tests
│   ├── integration/                # Real LLM tests
│   └── complex/                    # Long-running tests
│
├── config/
│   ├── app.yaml                    # Central configuration
│   └── app.test.yaml               # Test configuration
│
├── specs/                          # Feature specifications
│   ├── 001-deep-research-agent/
│   ├── 003-claim-level-citations/
│   └── 004-tiered-query-modes/
│
└── static/                         # Built frontend (gitignored)
```

## Data Flow

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ API Layer (FastAPI)                                              │
│ - Authentication via Databricks workspace identity              │
│ - SSE streaming for real-time updates                           │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Orchestrator                                                     │
│ - Routes to appropriate query mode (Simple/Web Search/Deep)     │
│ - Manages agent execution flow                                  │
│ - Handles persistence                                           │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5-Agent Pipeline                                                 │
│ Coordinator → Background → Planner → Researcher ↔ Reflector    │
│                                           ↓                     │
│                                      Synthesizer                │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7-Stage Citation Pipeline (optional)                            │
│ Evidence → Generation → Confidence → Verification → Correction │
│                              ↓                                  │
│                    Numeric QA → ARE Revision                    │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Persistence Layer                                                │
│ - Chat, Messages, Sources                                       │
│ - Claims, Citations, Evidence                                   │
│ - Research Sessions, Events                                     │
└─────────────────────────────────────────────────────────────────┘
    ↓
Response (SSE stream with events)
```

## Authentication Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client    │────▶│  FastAPI + Auth  │────▶│   Databricks    │
│  (Browser)  │     │    Middleware    │     │  WorkspaceClient│
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │                        │
                            ▼                        ▼
                    ┌──────────────┐         ┌─────────────────┐
                    │  User ID     │         │  OAuth Token    │
                    │  Extraction  │         │  Generation     │
                    └──────────────┘         └─────────────────┘
                                                     │
                                                     ▼
                                             ┌─────────────────┐
                                             │  LLM Endpoints  │
                                             │  Lakebase DB    │
                                             └─────────────────┘
```

## Deployment Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    Databricks Apps                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ App Container (Single Process)                              │ │
│  │                                                             │ │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │ │
│  │  │  FastAPI        │  │  Static Files   │                  │ │
│  │  │  Backend        │  │  (Frontend)     │                  │ │
│  │  │  Port 8000      │  │                 │                  │ │
│  │  └─────────────────┘  └─────────────────┘                  │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────┼────────────────────────────────────┐ │
│  │                       ▼                                    │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │            Lakebase (PostgreSQL)                    │   │ │
│  │  │  - OAuth token refresh (1-hour lifetime)            │   │ │
│  │  │  - Tables owned by developer, granted to app SP     │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                                                            │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │         Foundation Model Endpoints                  │   │ │
│  │  │  - Claude, Gemini, Llama via Databricks serving     │   │ │
│  │  │  - Automatic failover between endpoints             │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Service Layer Patterns

The service layer follows consistent patterns for maintainability and testability.

### BaseRepository Pattern

All CRUD services extend `BaseRepository[T]` for consistent database operations:

```python
# src/services/base.py
class BaseRepository(Generic[T]):
    """Generic repository pattern for SQLAlchemy models."""
    model: type[T]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entity: T) -> T:
        """Add entity with flush + refresh."""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def get(self, id: UUID) -> T | None:
        """Get by primary key."""
        return await self.session.get(self.model, id)

    async def get_or_raise(self, id: UUID) -> T:
        """Get by primary key or raise NotFoundError."""
        entity = await self.get(id)
        if not entity:
            raise NotFoundError(self.model.__name__, str(id))
        return entity
```

Services extend this base:

```python
class ChatService(BaseRepository[Chat]):
    model = Chat

    async def create(self, user_id: str, title: str | None = None) -> Chat:
        chat = Chat(user_id=user_id, title=title)
        return await self.add(chat)  # Uses BaseRepository.add()
```

### Shared Authorization Utilities

Centralized authorization functions in `src/api/v1/utils/authorization.py`:

| Function | Returns | Use Case |
|----------|---------|----------|
| `verify_chat_ownership(chat_id, user_id, db)` | `Chat` | Ensure user owns chat |
| `verify_chat_access(chat_id, user_id, db)` | `(is_draft, Chat \| None)` | Draft chat flow |
| `verify_message_ownership(message_id, user_id, db)` | `Message` | Ensure user owns message's chat |

```python
# Usage in API endpoints
from src.api.v1.utils import verify_chat_ownership

@router.get("/chats/{chat_id}/messages")
async def list_messages(chat_id: UUID, user: CurrentUser, db: AsyncSession):
    await verify_chat_ownership(chat_id, user.user_id, db)  # Raises NotFoundError
    # ... proceed with authorized request
```

### Response Transformers

Reusable response builders in `src/api/v1/utils/transformers.py`:

| Function | Purpose |
|----------|---------|
| `claim_to_response(claim)` | Transform Claim model to ClaimResponse |
| `build_verification_summary(model)` | Build summary from DB model |
| `build_citation_response(citation)` | Build citation with evidence |
| `build_evidence_span_response(span)` | Build evidence span with source |

### Eager-Loading Options

Centralized selectinload chains in `src/services/loading.py`:

```python
# Full claim graph for claim endpoints
CLAIM_WITH_CITATIONS_OPTIONS = [
    selectinload(Claim.citations).selectinload(Citation.evidence_span).selectinload(EvidenceSpan.source),
    selectinload(Claim.corrections),
    selectinload(Claim.numeric_detail),
]

# Usage
result = await db.execute(
    select(Claim).options(*CLAIM_WITH_CITATIONS_OPTIONS).where(...)
)
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/agent/orchestrator.py` | Main orchestration (1752 lines) |
| `src/agent/state.py` | ResearchState dataclass (532 lines) |
| `src/services/citation/pipeline.py` | Citation pipeline (2637 lines) |
| `src/services/llm/client.py` | LLM client with routing |
| `src/services/base.py` | BaseRepository pattern |
| `src/api/v1/utils/authorization.py` | Centralized auth utilities |
| `src/api/v1/utils/transformers.py` | Response builders |
| `src/core/app_config.py` | Pydantic configuration |
| `config/app.yaml` | Central YAML configuration |

## See Also

- [Agent Orchestration](./agents.md) - Detailed agent design
- [Citation Pipeline](./citation-pipeline.md) - Verification stages
- [Configuration](./configuration.md) - YAML config system
- [Deployment](./deployment.md) - Databricks Apps guide
