# Research: Deep Research Agent

**Date**: 2025-12-21
**Feature**: 001-deep-research-agent

## Executive Summary

This document consolidates research findings for implementing a deep research agent on Databricks. Key decisions include using **Plain Async Python** for agent orchestration (after evaluating LangGraph, DSPy, AutoGen, and PydanticAI), Brave Search API for web search, Databricks Lakebase for persistence, and MLflow for observability.

---

## 1. Agent Architecture: ReAct + Reflexion Pattern

### Decision
Use **Plain Async Python** to implement a hybrid ReAct/Reflexion agent pattern with 5-agent architecture.

### Rationale
After evaluating LangGraph, DSPy, AutoGen, and PydanticAI (see Section 11 for detailed analysis):
- Our workflow is nearly linear with one conditional loop—not complex enough for graph frameworks
- Custom rate-limiting infrastructure conflicts with LangChain/framework abstractions
- MLflow provides observability (no need for LangSmith)
- Databricks endpoints are OpenAI-compatible, enabling direct API calls
- Full control over flow and error handling improves debugging

### Alternatives Considered
| Alternative | Rejected Because |
|-------------|-----------------|
| LangGraph | Overkill for linear flow; conflicts with custom rate-limiting; loses value without LangChain |
| AutoGen | Wrong paradigm—designed for conversational multi-agent, not pipelines |
| DSPy | Wrong tool—for prompt optimization, not agent orchestration |
| PydanticAI | Possible but unnecessary abstraction for our simple flow |

### Implementation Pattern
```
ReAct Loop:
1. THOUGHT: Agent reasons about current state and knowledge gaps
2. ACTION: Agent calls tool (search, fetch) or decides to synthesize
3. OBSERVATION: Agent receives tool output
4. REFLECT: Agent evaluates progress, updates memory with learnings
5. REPEAT until sufficient information gathered or max iterations
```

### Key Best Practices
- Start simple with limited tools, add complexity gradually
- Use short feedback cycles between action and observation
- Design tools as independent, modular components
- Store reflection insights in persistent memory for multi-turn learning
- Implement explicit state for tracking research progress

### Sources
- [Using the ReAct Pattern in AI Agents](https://metadesignsolutions.com/using-the-react-pattern-in-ai-agents-best-practices-pitfalls-implementation-tips/)
- [Agentic AI Patterns: Demystifying ReAct, Reflexion and Auto-GPT](https://mainak-saha.medium.com/agentic-ai-patterns-demystifying-react-reflexion-and-auto-gpt-93dcec305611)
- [Reflexion Agent Pattern Documentation](https://agent-patterns.readthedocs.io/en/latest/patterns/reflexion.html)
- [You Don't Need AI Agent Frameworks](https://newsletter.owainlewis.com/p/you-dont-need-ai-agent-frameworks)

---

## 2. Web Search Integration: Brave Search API

### Decision
Use **brave-search-python-client** library for Brave Search API integration.

### Rationale
- Official Python client with modern tooling and practices
- Supports async operations (critical for agent performance)
- CLI and programmatic interfaces
- Free tier includes 2,000 requests/month for development
- Privacy-preserving alternative to Google Search

### Alternatives Considered
| Alternative | Rejected Because |
|-------------|-----------------|
| Tavily | Less established; Brave has broader web coverage |
| Google Custom Search | Privacy concerns; more expensive at scale |
| Bing Search API | Microsoft ecosystem dependency |
| SerpAPI | Third-party wrapper; adds latency and cost |

### Implementation Pattern
```python
from brave_search import AsyncBrave
import os

brave = AsyncBrave(api_key=os.environ["BRAVE_API_KEY"])
results = await brave.search(q=query, count=10)
```

### Key Best Practices
- Store API key in Databricks secrets, never in code
- Use async client for non-blocking operations in agent loop
- Implement rate limit handling with exponential backoff
- Cache search results to reduce API calls for repeated queries
- Parse result snippets for quick relevance assessment before full fetch

### Sources
- [Brave Search API Official](https://brave.com/search/api/)
- [brave-search-python-client Documentation](https://brave-search-python-client.readthedocs.io/)
- [LangChain Brave Search Integration](https://python.langchain.com/docs/integrations/tools/brave_search/)

---

## 3. Backend Architecture: FastAPI + Plain Async Python

### Decision
Use **FastAPI** for REST API with **Plain Async Python** for agent orchestration.

### Rationale
- FastAPI is the de facto standard for Python async APIs
- Native async/await support matches agent streaming requirements
- Built-in OpenAPI documentation generation
- Pydantic integration ensures constitution compliance (Principle II)
- Excellent dependency injection for testability

### Key Architecture Decisions

#### Async vs Sync Routes
- Use async routes for all LLM/agent operations
- Never block event loop with synchronous LLM calls
- Use thread pool for CPU-bound operations (PDF generation)

#### Streaming Implementation
- Use Server-Sent Events (SSE) for real-time reasoning display
- FastAPI StreamingResponse with async generators
- SSE is HTTP-based, works everywhere (unlike WebSockets)

#### Dependency Injection Pattern
```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    # Validate Databricks workspace identity
    ...
```

### Alternatives Considered
| Alternative | Rejected Because |
|-------------|-----------------|
| Flask | No native async; would require ASGI adapter |
| Django | Overkill for API-only service; slower iteration |
| Starlette (raw) | FastAPI adds essential features (validation, docs) |

### Sources
- [FastAPI Best Practices (zhanymkanov)](https://github.com/zhanymkanov/fastapi-best-practices)
- [Full Stack FastAPI Template](https://github.com/fastapi/full-stack-fastapi-template)
- [Building a Chat App with FastAPI and React](https://toxigon.com/creating-a-chat-application-with-fastapi-and-react)

---

## 4. Database: Databricks Lakebase (PostgreSQL)

### Decision
Use **Databricks Lakebase** with **SQLAlchemy async** ORM.

### Rationale
- Native Databricks integration with seamless authentication
- Serverless PostgreSQL with separated compute/storage
- Low latency (<10ms) and high concurrency support
- Synced Tables feature for Unity Catalog integration
- Branching/checkpointing for development workflows
- Familiar PostgreSQL with existing Python libraries

### Connection Pattern (Implemented)

**Configuration** (`.env`):
```bash
# Profile-based authentication (required for Lakebase OAuth)
DATABRICKS_CONFIG_PROFILE=your-profile-name

# Lakebase Configuration
LAKEBASE_INSTANCE_NAME=your-instance-name  # NOT the full hostname
LAKEBASE_DATABASE=deep_research
# Host is derived: {instance_name}.database.cloud.databricks.com

# Fallback for local development (when Lakebase not configured)
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/deep_research
```

**OAuth Credential Provider** (`backend/src/db/lakebase_auth.py`):
```python
from databricks.sdk import WorkspaceClient

class LakebaseCredentialProvider:
    def _generate_credential(self) -> LakebaseCredential:
        client = WorkspaceClient(profile=settings.databricks_config_profile)

        cred_response = client.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[settings.lakebase_instance_name],
        )

        return LakebaseCredential(
            token=cred_response.token,
            username="token",  # Lakebase OAuth uses "token" as username
            expires_at=expires_at,
        )

    def build_connection_url(self) -> str:
        cred = self.get_credential()
        host = f"{settings.lakebase_instance_name}.database.cloud.databricks.com"
        return (
            f"postgresql+asyncpg://token:{cred.token}"
            f"@{host}:{port}/{database}?sslmode=require"
        )
```

### Key Best Practices (Updated)
- Use `DATABRICKS_CONFIG_PROFILE` for profile-based OAuth authentication
- Use `LAKEBASE_INSTANCE_NAME` (not full hostname) - host is derived automatically
- OAuth tokens have 1-hour lifetime with automatic refresh (5-minute buffer)
- Username for Lakebase OAuth is always `"token"` (actual token is the password)
- SSL is required for Lakebase connections (`sslmode=require`)
- Fallback to direct `DATABASE_URL` for local PostgreSQL development
- Use connection pooling for concurrent user support
- Implement database migrations with Alembic

### Alternatives Considered
| Alternative | Rejected Because |
|-------------|-----------------|
| Delta Tables | Not suited for OLTP workloads; no transaction support |
| External PostgreSQL | Loses Databricks integration; separate auth |
| MongoDB | Not relational; spec requires SQL patterns |

### Sources
- [Databricks Lakebase Product Page](https://www.databricks.com/product/lakebase)
- [How to use Lakebase as a transactional data layer](https://www.databricks.com/blog/how-use-lakebase-transactional-data-layer-databricks-apps)
- [Use a notebook to access a database instance](https://docs.databricks.com/aws/en/oltp/instances/query/notebook)

---

## 5. Observability: MLflow Tracing

### Decision
Use **MLflow Tracing** for distributed tracing and LLM observability.

### Rationale
- Native Databricks integration
- OpenTelemetry-compatible (no vendor lock-in)
- Auto-tracing for 20+ GenAI libraries (OpenAI, LangChain, etc.)
- 100% free and open source
- Lightweight production SDK available (mlflow-tracing)

### Implementation Pattern
```python
import mlflow

# Enable auto-tracing for OpenAI calls
mlflow.openai.autolog()

# Enable auto-tracing for LangGraph
mlflow.langchain.autolog()

# Async logging for production (non-blocking)
mlflow.config.enable_async_logging(True)
```

### Key Best Practices
- Enable async logging for production (non-blocking)
- Use lightweight mlflow-tracing package in production
- Capture structured metadata (user_id, chat_id, research_depth)
- Log user feedback to traces for quality analysis
- Set up trace retention policies per compliance requirements

### Sources
- [MLflow Tracing for LLM Observability](https://mlflow.org/docs/latest/genai/tracing/)
- [Practical AI Observability: Getting Started with MLflow Tracing](https://mlflow.org/blog/ai-observability-mlflow-tracing)
- [Deploy agents with tracing on Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing)

---

## 6. Databricks Agent API Pattern

### Decision
Implement **ResponsesAgent** interface for Databricks-compatible agent deployment.

### Rationale
- Recommended interface for production-grade Databricks agents
- Compatible with OpenAI Responses schema
- Integrates with Databricks AI features (logging, evaluation, monitoring)
- Supports wrapping existing LangGraph agents
- Python 3.10+ required (matches our target)

### Implementation Pattern
```python
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

class DeepResearchAgent(ResponsesAgent):
    def __init__(self):
        # Initialize LangGraph agent
        self.agent = create_research_graph()

    def predict(
        self,
        request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        # Process request through LangGraph
        result = self.agent.invoke(request.messages)
        return ResponsesAgentResponse(...)
```

### Key Best Practices
- Don't store state at ResponsesAgent level (distributed environment)
- Initialize state in predict() method, not __init__()
- Use ResponsesAgentStreamEvent for streaming responses
- Wrap existing LangGraph agents rather than rewriting

### Sources
- [Author AI agents in code - Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent)
- [Get started with AI agents - Databricks](https://docs.databricks.com/aws/en/generative-ai/tutorials/agent-quickstart)

---

## 7. Frontend Architecture: React + TanStack Query

### Decision
Use **React 18** with **TanStack Query** and **SSE** for streaming.

### Rationale
- Follows claude-databricks-app-template patterns
- TanStack Query provides excellent async state management
- Official `streamedQuery` feature for streaming chat (React example available)
- Vite for fast development with HMR
- Shadcn/UI for accessible, customizable components

### Streaming Implementation
```typescript
// Using TanStack Query streamedQuery
import { streamedQuery } from '@tanstack/react-query'

const useChat = (chatId: string) => {
  return useQuery({
    queryKey: ['chat', chatId],
    queryFn: ({ signal }) =>
      streamedQuery({
        url: `/api/chat/${chatId}/stream`,
        signal,
      }),
  })
}
```

### Key Best Practices
- Use SSE (Server-Sent Events) rather than WebSockets for simplicity
- Implement optimistic updates for message sending
- Cache conversation history for instant navigation
- Add typing indicators and loading states
- Handle reconnection gracefully

### Accessibility (WCAG 2.1 AA)
- Use Shadcn/UI components (built with accessibility in mind)
- Implement keyboard navigation for all actions
- Ensure sufficient color contrast
- Add ARIA labels for screen readers
- Test with axe-core in CI

### Deployment Model
The React frontend is built to static assets during the build process and served directly by FastAPI. No Node.js runtime is required in production.

**Architecture:**
- Vite builds to `backend/static/` directory
- FastAPI serves static files when `SERVE_STATIC=true`
- SPA fallback routing returns `index.html` for client-side routes
- API and frontend served on same origin (no CORS complexity)

**Development vs Production:**
| Mode | Frontend | Backend | Access URL |
|------|----------|---------|------------|
| Development | Vite dev server (HMR) | uvicorn --reload | http://localhost:5173 |
| Production | Static files in backend | FastAPI serves all | http://localhost:8000 |

Development uses Vite's proxy to forward `/api` requests to the backend, preserving hot module replacement for rapid iteration.

### Sources
- [TanStack Query streamedQuery](https://tanstack.com/query/latest/docs/reference/streamedQuery)
- [React Query and Server Side Events](https://fragmentedthought.com/blog/2025/react-query-caching-with-server-side-events)
- [FastAPI and React in 2025](https://www.joshfinnie.com/blog/fastapi-and-react-in-2025/)

---

## 8. Model Routing & Failover

### Decision
Implement custom **typed model routing service** with Pydantic models, supporting tiered model roles with role-level defaults and endpoint-level overrides.

### Rationale
- Constitution requires typed interfaces (Principle II)
- Spec requires named model roles with configurable strategies
- External configuration (env vars, config files) per spec clarification
- No off-the-shelf solution meets all requirements
- Different endpoints have different context windows and rate limits
- O-series models require reasoning_effort/budget parameters

### Authentication Pattern (Implemented)

**LLM Client** (`backend/src/services/llm/client.py`):
```python
from databricks.sdk import WorkspaceClient
from openai import AsyncOpenAI

class LLMClient:
    def __init__(self, config: ModelConfig | None = None):
        settings = get_settings()

        # Get token - either from env or from WorkspaceClient
        token = settings.databricks_token
        base_url = f"{settings.databricks_host}/serving-endpoints"

        # Profile-based auth (preferred for Lakebase OAuth compatibility)
        if not token and settings.databricks_config_profile:
            w = WorkspaceClient(profile=settings.databricks_config_profile)
            w.config.authenticate()
            token = w.config.oauth_token().access_token
            base_url = f"{w.config.host}/serving-endpoints"

        if not token:
            raise ValueError("No Databricks token available")

        # Initialize OpenAI client for Databricks serving endpoints
        self._client = AsyncOpenAI(
            api_key=token,
            base_url=base_url,
        )
```

**Key Authentication Points**:
- Supports both direct `DATABRICKS_TOKEN` and profile-based auth
- Profile-based auth uses `WorkspaceClient.config.authenticate()` then `oauth_token().access_token`
- Base URL derived from workspace host: `{host}/serving-endpoints`
- Uses standard OpenAI `AsyncOpenAI` client with custom `base_url` and `api_key`

### Configuration Architecture

**Two-level configuration hierarchy:**
1. **Role-level** - Defines defaults for all endpoints in a role
2. **Endpoint-level** - Overrides role defaults; contains endpoint-specific values

```yaml
# models.yaml - Tiered model configuration
model_roles:
  simple:  # Tier 1: Lightweight tasks (query classification, simple responses)
    endpoints: [databricks-gpt-oss-20b, databricks-llama-4-maverick]
    temperature: 0.5
    max_tokens: 8000              # Output token limit (default)
    reasoning_effort: low
    rotation_strategy: priority
    fallback_on_429: true

  analytical:  # Tier 2: Medium complexity (research synthesis)
    endpoints: [databricks-gpt-oss-120b, databricks-gpt-oss-20b]
    temperature: 0.7
    max_tokens: 12000
    reasoning_effort: medium
    rotation_strategy: priority
    fallback_on_429: true

  complex:  # Tier 3: Heavy tasks (deep reasoning, reflection)
    endpoints: [databricks-gpt-oss-120b, databricks-claude-3-7-sonnet]
    temperature: 0.7
    max_tokens: 25000
    reasoning_effort: high
    reasoning_budget: 8000        # Token budget for extended thinking
    rotation_strategy: priority
    fallback_on_429: true

endpoints:
  databricks-gpt-oss-120b:
    max_context_window: 128000    # Input limit (REQUIRED - endpoint-specific)
    tokens_per_minute: 200000     # Rate limit (REQUIRED - endpoint-specific)

  databricks-gpt-oss-20b:
    max_context_window: 32000
    tokens_per_minute: 150000

  databricks-llama-4-maverick:
    max_context_window: 128000
    tokens_per_minute: 100000

  databricks-claude-3-7-sonnet:
    max_context_window: 200000
    tokens_per_minute: 50000
    temperature: 0.5              # Override role default for this endpoint
```

### Design Pattern
```python
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class SelectionStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"

class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelEndpoint(BaseModel):
    """Endpoint configuration with required and optional fields."""
    id: str
    endpoint_identifier: str
    # Required endpoint-specific values (no inheritance)
    max_context_window: int       # Input context limit
    tokens_per_minute: int        # Rate limit budget
    # Optional overrides (inherit from role if None)
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    reasoning_budget: int | None = None

class ModelRole(BaseModel):
    """Role configuration with defaults for all endpoints."""
    name: str                     # "simple", "analytical", "complex"
    endpoints: list[str]          # Ordered endpoint IDs
    # Role-level defaults
    temperature: float = 0.7
    max_tokens: int = 8000        # Output token limit
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    reasoning_budget: int | None = None
    # Routing behavior
    rotation_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    fallback_on_429: bool = True

class EndpointHealth(BaseModel):
    """Runtime state - NOT persisted to database (in-memory only)."""
    endpoint_id: str
    is_healthy: bool = True
    last_success_at: datetime | None = None
    consecutive_errors: int = 0
    rate_limited_until: datetime | None = None
    tokens_used_this_minute: int = 0
    minute_started_at: datetime | None = None

class ModelRouter:
    """Routes requests to endpoints based on role strategy and health."""

    def __init__(self, config: ModelConfig):
        self._roles: dict[str, ModelRole] = {}
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._health: dict[str, EndpointHealth] = {}  # In-memory only

    async def get_endpoint(self, role: str) -> ModelEndpoint:
        """Select endpoint based on role's strategy and health."""
        ...

    async def failover(self, role: str, failed_id: str) -> ModelEndpoint:
        """Select next healthy endpoint after failure."""
        ...

    def get_effective_config(
        self, role: str, endpoint_id: str
    ) -> dict[str, Any]:
        """Merge role defaults with endpoint overrides."""
        ...

    async def record_usage(self, endpoint_id: str, tokens: int) -> None:
        """Track token usage for rate limiting."""
        ...
```

### Key Best Practices
- Track endpoint health in-memory only (not persisted to DB)
- Each instance tracks health independently (no shared state across replicas)
- Use per-endpoint rate limiting with `tokens_per_minute` budgets
- Implement jitter in retry delays to prevent thundering herd
- Truncate context for smaller fallback endpoints (preserve system prompt + recent messages)
- Log all routing decisions to MLflow for debugging
- Configure via YAML config files (hot-reloadable without restart)
- Merge endpoint overrides with role defaults (endpoint wins)

### Rate Limiting Strategy
- Track `tokens_used_this_minute` per endpoint in EndpointHealth
- When `tokens_per_minute` limit reached, either:
  - Wait until next minute window, OR
  - Failover to next endpoint if `fallback_on_429=true`
- Reset token counter when `minute_started_at` is older than 1 minute

---

## 9. Query Intelligence Implementation

### Decision
Use **LLM-based classification** for query complexity and follow-up detection.

### Rationale
- Spec assumes LLM can classify queries without external tools
- Simpler than training separate classifiers
- Can leverage existing model endpoints
- Classification reasoning can be shown to users (transparency)

### Implementation Approach
```python
class QueryClassification(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    follow_up_type: Literal["new_topic", "clarification", "complex_follow_up"]
    is_ambiguous: bool
    clarifying_questions: list[str] = []
    reasoning: str

async def classify_query(
    query: str,
    conversation_history: list[Message]
) -> QueryClassification:
    """Use fast model to classify query before research."""
    ...
```

### Key Best Practices
- Use "fast" model role for classification (low latency)
- Include last 3-5 messages as context for follow-up detection
- Return classification reasoning to display to user
- Default to "moderate" if classification uncertain
- Allow user override via research depth selector

---

## 10. Multi-Agent Architecture

### Decision
Implement **5-agent architecture** with step-by-step reflection, inspired by deer-flow reference implementation but with key differentiations.

### Reference Implementation
deer-flow: Open-source multi-agent deep research framework
- Repository analyzed for architecture patterns
- Key files: `src/graph/builder.py`, `src/graph/nodes.py`, `src/graph/types.py`

### Architecture Overview

```
User Query
    │
    ▼
┌─────────────────┐
│   COORDINATOR   │  ← simple tier
│  - Classify     │
│  - Clarify      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  BACKGROUND     │  ← simple tier
│  INVESTIGATOR   │
└────────┬────────┘
         ▼
┌─────────────────┐
│    PLANNER      │  ← analytical tier
└────────┬────────┘
         ▼
    ┌────────────────────────────────────────┐
    │         RESEARCH LOOP                   │
    │  ┌─────────────────┐                   │
    │  │   RESEARCHER    │  ← analytical     │
    │  └────────┬────────┘                   │
    │           ▼                            │
    │  ┌─────────────────┐                   │
    │  │   REFLECTOR     │  ← simple tier    │
    │  │  CONTINUE/ADJUST/COMPLETE           │
    │  └─────────────────┘                   │
    └────────────────────────────────────────┘
         ▼
┌─────────────────┐
│  SYNTHESIZER    │  ← complex tier
└─────────────────┘
```

### Agent Definitions

| Agent | Model Tier | Responsibilities |
|-------|------------|------------------|
| **Coordinator** | simple | Query classification, clarification rounds (up to 3), route simple queries to direct response |
| **Background Investigator** | simple | Quick web search (~5s) to provide context for planning |
| **Planner** | analytical | Create structured research plan with typed steps (RESEARCH, ANALYSIS) |
| **Researcher** | analytical | Execute ONE step at a time, web search + crawl, store observations |
| **Reflector** | simple | Evaluate after EACH step: CONTINUE, ADJUST, or COMPLETE |
| **Synthesizer** | complex | Analyze all observations, generate final report with citations |

### Key Differentiator: Step-by-Step Reflection

| Aspect | deer-flow | Our Design |
|--------|-----------|------------|
| **Reflection timing** | After ALL steps complete | After EACH step |
| **Replan trigger** | Planner checks "has_enough_context" | Reflector detects findings mismatch |
| **Adaptivity** | Batch-then-replan | Real-time adjustment |
| **Cost** | Fewer LLM calls | More calls (but cheaper Reflector uses simple tier) |
| **Quality** | May waste effort on irrelevant steps | Pivots immediately on new discoveries |

### Rationale for Step-by-Step Reflection
- Research is inherently exploratory—you don't know what you'll find
- Early findings can completely change the research direction
- Avoids wasting effort on steps that become irrelevant
- The Reflector is lightweight (simple tier) so cost is minimal
- Better research quality justifies the additional LLM calls

### Features Adopted from deer-flow
- **Clarification Rounds**: Up to 3 rounds for vague queries
- **Background Investigation**: Quick pre-planning search
- **Structured Plans**: Typed steps with RESEARCH/ANALYSIS types
- **JSON Repair**: Handle malformed LLM outputs
- **Plan Validation**: Auto-repair missing fields

### Features NOT Adopted
| Feature | Reason |
|---------|--------|
| Batch-then-replan | Replaced with step-by-step reflection |
| Human-in-the-loop plan approval | Adds friction; defer to v2 |
| Multiple report styles | Out of scope for MVP |
| Podcast/PPT generation | Out of scope |
| Coder agent | No Python execution requirement |
| MCP integration | Complex; defer to v2 |

---

## 11. Orchestration Framework: Plain Async Python

### Decision
**Plain Async Python** — No framework required for our architecture.

### Rationale
After comprehensive analysis of LangGraph, DSPy, AutoGen, and PydanticAI, we determined that:
1. Our workflow is nearly linear with one conditional loop—not complex enough for graph frameworks
2. Custom rate-limiting infrastructure conflicts with framework abstractions
3. MLflow provides observability (no need for LangSmith)
4. Databricks endpoints are OpenAI-compatible, enabling direct API calls

### Framework Evaluation

#### LangGraph
**Verdict: NOT WORTH IT**

Can work without LangChain, but loses most value:
- Must avoid prebuilt components (`ToolNode`, `create_react_agent`, `MessageGraph`)
- Would manually implement all LLM calls anyway
- Only remaining benefits: checkpointing + graph visualization (we don't need these)
- Our custom rate-limiting conflicts with LangChain model abstractions

From [GitHub discussion](https://github.com/langchain-ai/langgraph/discussions/1645):
> "We have a fair number of LangGraph users who don't use LangChain. They typically use the framework directly, without the prebuilt components."

#### DSPy (Stanford)
**Verdict: WRONG TOOL**

DSPy is for **prompt optimization**, NOT agent orchestration:
- Replaces manual prompt engineering with automated optimization
- Designed to reduce hallucinations through prompt tuning
- Stateless by default—manual state management required

Integration is possible via OpenAI-compatible endpoints:
```python
lm = dspy.LM(
    "openai/your-model-name",
    api_base="https://databricks-endpoint/serving-endpoints",
    api_key=token,
)
```

But DSPy won't orchestrate agents—it's **complementary**, not a replacement. Could use later to optimize prompts.

#### AutoGen (Microsoft)
**Verdict: WRONG PARADIGM**

AutoGen is for **conversational multi-agent systems**:
- Agents "talk to each other" via asynchronous message passing
- Designed for complex agent topologies and negotiations

Our pattern is a **pipeline**, not a conversation:
| AutoGen Pattern | Our Pattern |
|-----------------|-------------|
| Agents talk back and forth | Sequential pipeline with loops |
| Message-based collaboration | State passed through functions |
| Complex agent topologies | Linear flow with one branch |

#### PydanticAI
**Verdict: POSSIBLE BUT UNNECESSARY**

Good middle ground:
- Model-agnostic, supports OpenAI-compatible endpoints
- Faster than LangGraph ([benchmarks](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more))
- Pydantic-native, OpenTelemetry compatible

But for our simple flow, even PydanticAI adds unnecessary abstraction.

### Decision Matrix

| Framework | Custom LLM | Complexity Fit | Rate Limiting | Observability | Verdict |
|-----------|-----------|----------------|---------------|---------------|---------|
| LangGraph | Yes (w/o LangChain) | Overkill | Conflicts | LangSmith | ❌ |
| DSPy | Yes | Wrong tool | N/A | N/A | ❌ |
| AutoGen | Yes | Wrong paradigm | Unknown | Built-in | ❌ |
| PydanticAI | Yes | Possible | Compatible | OpenTelemetry | ⚠️ |
| **Plain Python** | **Yes** | **Perfect** | **Direct control** | **MLflow** | ✅ |

### Implementation Pattern

From [Anthropic engineering guidance](https://newsletter.owainlewis.com/p/you-dont-need-ai-agent-frameworks):
> "Frameworks often create extra layers of abstraction that can obscure underlying prompts and responses, making them harder to debug. Simplicity scales while complexity fails."

**Main Orchestration:**
```python
async def run_research(query: str) -> Report:
    state = ResearchState(query=query)

    # Coordinator phase
    state = await coordinator(state)
    if state.is_simple_query:
        return state.direct_response

    # Background investigation
    state = await background_investigator(state)

    # Plan + Research loop with step-by-step reflection
    while state.plan_iterations < MAX_PLAN_ITERATIONS:
        state = await planner(state)

        for step in state.current_plan.pending_steps():
            state = await researcher(state, step)
            decision = await reflector(state)

            if decision.decision == "complete":
                break
            elif decision.decision == "adjust":
                break
            # else "continue" → next step

        if decision.decision == "complete":
            break

    return await synthesizer(state)
```

**Core Patterns:**

1. **Routing** (Strategy Pattern):
```python
async def coordinator(state: ResearchState) -> ResearchState:
    classification = await classify_query(state.query)
    if classification.complexity == "simple":
        state.is_simple_query = True
        state.direct_response = await simple_response(state)
    return state
```

2. **Parallelization** (asyncio.gather):
```python
async def background_investigator(state: ResearchState) -> ResearchState:
    results = await asyncio.gather(
        search_web(state.query),
        find_related_topics(state.query),
    )
    state.background_results = results
    return state
```

3. **Streaming** (Async generators):
```python
async def stream_research(query: str) -> AsyncGenerator[StreamEvent, None]:
    state = ResearchState(query=query)
    yield AgentStartedEvent(agent="coordinator")
    state = await coordinator(state)
    yield AgentCompletedEvent(agent="coordinator")
    # ... continue streaming events
```

### Why This Works Best

| Concern | Plain Python Solution |
|---------|----------------------|
| Custom rate limiting | Direct integration, no framework conflicts |
| Databricks endpoints | Standard `openai` client with custom base_url |
| Observability | MLflow tracing (already planned) |
| Streaming | Async generators + SSE |
| State management | Pydantic `ResearchState` model |
| Debugging | Full visibility, no framework abstractions |
| Testing | Standard pytest, no framework mocking |
| Deployment | Smaller footprint, fewer dependencies |

### When to Reconsider

Add a framework only if:
- Checkpointing for sessions lasting hours/across server restarts becomes critical
- Workflow becomes significantly more complex (multiple parallel branches)
- Graph visualization needed for debugging complex state transitions

### Sources
- [You Don't Need AI Agent Frameworks](https://newsletter.owainlewis.com/p/you-dont-need-ai-agent-frameworks)
- [Best AI Agent Frameworks in 2025](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)
- [LangGraph without LangChain Discussion](https://github.com/langchain-ai/langgraph/discussions/1645)
- [DSPy Language Models](https://dspy.ai/learn/programming/language_models/)
- [Pydantic AI vs LangGraph](https://www.zenml.io/blog/pydantic-ai-vs-langgraph)

---

## 12. Technology Stack Summary

| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| Agent Orchestration | Plain async Python | - | No framework required (see Section 11) |
| HTTP Client | httpx | Latest | Async HTTP client for Databricks endpoints |
| LLM Client | openai | Latest | With custom `base_url` for Databricks endpoints |
| Backend Framework | FastAPI | 0.100+ | |
| Python | Python | 3.11+ | |
| ORM | SQLAlchemy (async) | 2.0+ | |
| Database | Databricks Lakebase | Preview | |
| Search API | Brave Search | v1 | |
| Observability | MLflow | 2.14+ | |
| Frontend Framework | React | 18.x | |
| Build Tool | Vite | 5.x | |
| TypeScript | TypeScript | 5.x | |
| UI Components | Shadcn/UI | Latest | |
| State Management | TanStack Query | 5.x | |
| Styling | Tailwind CSS | 3.x | |
| Testing (Backend) | pytest + pytest-asyncio | Latest | |
| Testing (Frontend) | Vitest + Playwright | Latest | |
| Type Checking | mypy (strict) | Latest | |
| Linting | ruff | Latest | |
| JSON Repair | json_repair | Latest | From deer-flow patterns |

---

## Appendix: Resolved NEEDS CLARIFICATION Items

All technical context items resolved:

| Item | Resolution |
|------|------------|
| Language/Version | Python 3.11+ (backend), TypeScript 5.x (frontend) |
| Primary Dependencies | FastAPI, httpx, openai, MLflow, SQLAlchemy, TanStack Query, Shadcn/UI |
| Agent Orchestration | Plain async Python (no framework—see Section 11) |
| Storage | Databricks Lakebase via SQLAlchemy async |
| Testing | pytest (backend), Vitest + Playwright (frontend) |
| Target Platform | Databricks Apps (serverless) |
| Project Type | Web application (frontend + backend) |
| Performance Goals | From spec: 2 min research, 3s load, 5s clarifications |
| Constraints | 99% availability, 10-100 users, WCAG 2.1 AA |
| Scale/Scope | Team-scale per spec assumptions |

---

## 13. End-to-End Testing with Playwright

**Added**: 2025-12-22
**User Story**: 9 - Automated End-to-End Testing

### Decision
Use **Playwright** with **TypeScript** for browser-based e2e testing with Page Object Model architecture.

### Rationale
- Playwright is the modern standard for e2e testing (faster and more reliable than Selenium/Cypress)
- TypeScript matches frontend codebase for consistent DX
- Page Object Model encapsulates page interactions for maintainability
- Built-in auto-waiting eliminates flaky tests from timing issues
- Excellent CI/CD integration with GitHub Actions
- Cross-browser support (Chromium, Firefox, WebKit) out of the box

### Alternatives Considered
| Alternative | Rejected Because |
|-------------|-----------------|
| Cypress | Slower execution, limited cross-browser support |
| Selenium | Legacy API, more brittle, slower |
| Puppeteer | Chrome-only, lower-level API |
| TestCafe | Less community support, fewer features |

### Implementation Patterns

#### Page Object Model
```typescript
// pages/chat.page.ts
export class ChatPage {
  constructor(private page: Page) {}

  async sendMessage(text: string) {
    await this.page.getByRole('textbox', { name: /message/i }).fill(text);
    await this.page.getByRole('button', { name: /send/i }).click();
  }

  async waitForAgentResponse(timeout = 120000) {
    // Wait for loading indicator to disappear
    await this.page.waitForSelector('[data-testid="loading-indicator"]', {
      state: 'hidden',
      timeout
    });
    // Verify response appeared
    await this.page.waitForSelector('[data-testid="agent-response"]', {
      state: 'visible'
    });
  }

  async getLastAgentResponse(): Promise<string> {
    const responses = this.page.locator('[data-testid="agent-response"]');
    return responses.last().textContent() ?? '';
  }
}
```

#### Handling SSE Streaming Responses
```typescript
// utils/wait-helpers.ts
export async function waitForStreamingComplete(page: Page): Promise<void> {
  // Wait for the streaming indicator to disappear
  await page.waitForFunction(() => {
    const indicator = document.querySelector('[data-testid="streaming-indicator"]');
    return !indicator || indicator.getAttribute('data-streaming') !== 'true';
  }, { timeout: 120000 });
}
```

#### Test Fixtures
```typescript
// fixtures/chat.fixture.ts
import { test as base } from '@playwright/test';
import { ChatPage } from '../pages/chat.page';

export const test = base.extend<{ chatPage: ChatPage }>({
  chatPage: async ({ page }, use) => {
    await page.goto('/');
    const chatPage = new ChatPage(page);
    await use(chatPage);
  },
});
```

### Test Data Strategy

| Test Type | Query Examples | Purpose |
|-----------|----------------|---------|
| Smoke | "Hello", "What is 2+2?" | Fast validation, <10s |
| Research Flow | "What is Python used for?" | Moderate research, <30s |
| Follow-up | "Can you explain more?" | Context verification |
| Stop/Cancel | Long queries | Test cancel within 2s |

### Configuration
```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e/tests',
  timeout: 120000,            // 2 min for research operations
  expect: { timeout: 10000 }, // 10s for assertions
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { open: 'never' }],
    ['list'],
  ],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:8000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'make prod',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
```

### CI/CD Integration
```yaml
# .github/workflows/e2e.yml
name: E2E Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - name: Install dependencies
        run: cd frontend && npm ci
      - name: Install Playwright browsers
        run: cd frontend && npx playwright install --with-deps
      - name: Start backend
        run: make dev-backend &
      - name: Run e2e tests
        run: cd frontend && npm run test:e2e
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: frontend/playwright-report/
```

### Key Best Practices
- Use `data-testid` attributes for stable selectors
- Configure extended timeouts for research operations (30-120s)
- Enable trace recording for debugging failed tests
- Run tests sequentially in CI for stability
- Use fixtures for reusable setup/teardown
- Test simple queries in smoke tests (fast feedback)
- Reserve complex queries for dedicated research flow tests

### Dependencies to Add
```json
{
  "devDependencies": {
    "@playwright/test": "^1.42.0"
  },
  "scripts": {
    "test:e2e": "playwright test",
    "test:e2e:ui": "playwright test --ui",
    "test:e2e:debug": "playwright test --debug"
  }
}
```

### Sources
- [Playwright Documentation](https://playwright.dev/docs/intro)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- [Page Object Model](https://playwright.dev/docs/pom)

---

## 14. Web Content Extraction: Trafilatura

**Added**: 2025-12-22

### Decision
Use **trafilatura** for HTML content extraction, replacing BeautifulSoup + manual heuristics.

### Rationale
- F-Score 0.909 in content extraction benchmarks (best among open-source)
- Purpose-built for main content extraction with automatic boilerplate removal
- Extracts metadata (title, author, date) automatically
- Used by HuggingFace, IBM, Microsoft Research, Stanford, Allen Institute
- Apache 2.0 license (since v1.8.0)

### Benchmark Comparison (750 documents)

| Library | F-Score | Precision | Recall | Notes |
|---------|---------|-----------|--------|-------|
| **Trafilatura 1.2.2** | **0.909** | 0.914 | 0.904 | **Best overall** |
| readabilipy 0.2.0 | 0.874 | - | - | Balanced |
| news-please 1.5.22 | 0.808 | - | - | - |
| readability-lxml 0.8.1 | 0.801 | 0.891 | - | Good precision |
| goose3 3.1.9 | 0.793 | 0.934 | Low | Highest precision, poor recall |

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| BeautifulSoup (previous) | Generic parser, manual heuristics, lower quality (~0.70-0.75 estimated) |
| readability-lxml | Lower F-Score (0.801), less metadata |
| goose3 | Poor recall despite high precision |
| newspaper3k | Unmaintained |

### Implementation Pattern

```python
from trafilatura import bare_extraction

result = bare_extraction(
    html,
    url=base_url,
    include_comments=False,
    include_tables=True,
    include_links=False,
    with_metadata=True,
)

if result is None:
    return "", None

title = result.get("title")
content = result.get("text") or ""
```

### Key Benefits for Research Agent

| Aspect | BeautifulSoup + Manual | Trafilatura |
|--------|------------------------|-------------|
| **Purpose** | Generic HTML parsing | Main content extraction |
| **Boilerplate removal** | Manual (incomplete) | Automatic (sophisticated) |
| **F-Score** | ~0.70-0.75 estimated | **0.909** |
| **Metadata** | Title only | Title, author, date, categories |
| **Maintenance** | Our code (35+ lines) | Battle-tested library |
| **LLM data quality** | Lower | Higher (less noise) |

**Key Insight**: Better content extraction = less noise for LLM = better research quality.

### Async Compatibility

Trafilatura's `extract()` is synchronous (CPU-bound parsing). Our httpx fetching remains async:
- Direct call after async fetch (recommended) - parsing is fast (<100ms typically)
- Thread pool (`asyncio.run_in_executor()`) for many documents concurrently

### Sources
- [Trafilatura Documentation](https://trafilatura.readthedocs.io/)
- [Trafilatura Evaluation Benchmarks](https://trafilatura.readthedocs.io/en/latest/evaluation.html)
- [GitHub - adbar/trafilatura](https://github.com/adbar/trafilatura)
- [ACL Paper](https://aclanthology.org/2021.acl-demo.15/)
- [BeautifulSoup Alternatives 2025](https://oxylabs.io/blog/beautifulsoup-alternatives)

---

## 15. Comprehensive Testing Infrastructure

**Added**: 2025-12-24
**User Stories**: 9 (E2E), 10 (Python Unit), 11 (Frontend Unit)

### Overview

This section consolidates research findings for implementing comprehensive testing infrastructure covering Python unit tests, frontend unit tests, and enhanced E2E tests.

### Decision: Use pytest with pytest-asyncio and Layered Fixtures (Python)

**Rationale**:
- pytest is the de-facto standard for Python testing with excellent async support
- pytest-asyncio 0.23+ provides auto mode for simpler async test handling
- Layered fixtures (root → type-specific → test-specific) reduce duplication

**Alternatives Considered**:
| Alternative | Rejected Because |
|-------------|-----------------|
| unittest | Less flexible fixture model, verbose async handling |
| nose2 | Less active development, smaller ecosystem |
| ward | Too experimental for production use |

### Mock Patterns for External Dependencies

#### Database (AsyncSession) Mocking

```python
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mocked AsyncSession."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    return session
```

#### LLM Client Mocking

```python
from unittest.mock import AsyncMock
from src.services.llm.client import LLMClient

@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mocked LLMClient."""
    client = AsyncMock(spec=LLMClient)
    client.generate = AsyncMock(return_value="mocked response")
    client.generate_structured = AsyncMock(return_value={"key": "value"})
    return client
```

#### HTTP Client (httpx) Mocking

```python
from unittest.mock import AsyncMock
import httpx

@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create a mocked httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    response = AsyncMock(spec=httpx.Response)
    response.json = MagicMock(return_value={"results": []})
    response.status_code = 200
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    return client
```

### Decision: Use FastAPI TestClient with Dependency Overrides (API Testing)

**Rationale**:
- TestClient provides synchronous interface to async endpoints
- Dependency overrides allow injecting mocks without modifying source
- Supports full request/response cycle testing

**API Testing Pattern**:

```python
from fastapi.testclient import TestClient
from src.main import app
from src.db.session import get_db
from src.middleware.auth import get_current_user

def override_get_db():
    """Return mock database session."""
    return mock_db_session

def override_get_current_user():
    """Return mock user."""
    return UserIdentity(user_id="test-user-123")

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

def test_list_chats():
    response = client.get("/api/v1/chats")
    assert response.status_code == 200
    assert "items" in response.json()
```

### Decision: Test Agents at Three Levels

1. **Prompt Construction**: Verify prompts are built correctly from inputs
2. **Response Parsing**: Verify LLM responses are parsed/validated correctly
3. **State Transitions**: Verify agent updates state correctly

**Agent Testing Pattern**:

```python
async def test_coordinator_classifies_simple_query(mock_llm_client):
    """Test Coordinator correctly identifies simple queries."""
    # Arrange
    mock_llm_client.generate_structured.return_value = {
        "classification": {
            "complexity": "simple",
            "is_followup": False,
            "needs_research": False
        }
    }

    coordinator = CoordinatorNode(mock_llm_client)
    state = ResearchState(query="What is 2+2?")

    # Act
    result = await coordinator.run(state)

    # Assert
    assert result.is_simple_query is True
    assert result.query_classification.complexity == "simple"
    mock_llm_client.generate_structured.assert_called_once()
```

### Decision: Use Vitest with React Testing Library (Frontend)

**Rationale**:
- Vitest is native to Vite ecosystem (already using Vite)
- React Testing Library promotes testing user behavior, not implementation
- jsdom provides browser-like environment for unit tests

**Hook Testing Pattern**:

```typescript
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useChats } from '@/hooks/useChats';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useChats', () => {
  it('fetches chats on mount', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: [], total: 0 }),
    });

    const { result } = renderHook(() => useChats(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.items).toEqual([]);
  });
});
```

**SSE (EventSource) Mocking**:

```typescript
import { vi } from 'vitest';

class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(public url: string) {}
  close() {}

  simulateMessage(data: string) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }
}

global.EventSource = MockEventSource as any;
```

### Coverage Configuration

**Python (pyproject.toml)**:
```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
fail_under = 60
```

**Frontend (vitest.config.ts)**:
```typescript
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json'],
      exclude: ['node_modules/', 'tests/', '**/*.d.ts'],
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 60,
      },
    },
  },
});
```

### Test Selector Strategy (E2E)

**Decision**: Use `data-testid` attributes exclusively

**Rationale**:
- CSS classes can change with styling updates
- Text content can change with i18n/copy changes
- data-testid is stable and explicitly for testing

```typescript
// Good: data-testid
await page.getByTestId('message-input').fill('Hello');

// Avoid: CSS class (fragile)
await page.locator('.input-field').fill('Hello');

// Avoid: Text content (i18n breaks)
await page.getByText('Send').click();
```

### Summary of Testing Decisions

| Area | Decision | Key Technology |
|------|----------|----------------|
| Python Testing | pytest + pytest-asyncio | Auto async mode |
| Mocking | AsyncMock + dependency overrides | unittest.mock |
| API Testing | FastAPI TestClient | Dependency injection |
| Frontend Testing | Vitest + React Testing Library | jsdom environment |
| E2E Testing | Playwright | Page Object Model |
| Selectors | data-testid attributes | Stable, explicit |
| Coverage | pytest-cov + v8 | Threshold enforcement |
| CI | GitHub Actions | Parallel jobs, caching |

### Sources
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)

---

## 13. Central YAML Configuration (2025-12-24)

### Decision

Implement a central YAML configuration file (`config/app.yaml`) that consolidates all model endpoints, model roles, agent settings, and search configuration. Use custom environment variable interpolation and Pydantic v2 for validation.

### Research Topics

#### 13.1 YAML Configuration Patterns in Python

**Findings**:

1. **PyYAML** is the standard library for YAML parsing in Python
   - Part of the established ecosystem
   - Uses `safe_load()` to prevent code execution attacks
   - Already in project dependencies

2. **Configuration Hierarchy Pattern**:
   - Environment-specific files: `config/app.yaml`, `config/app.local.yaml`
   - Environment variable overrides for secrets
   - Default values in code as fallback

3. **Validation at Boundaries**:
   - Load YAML → Parse to dict → Validate with Pydantic
   - Fail fast on startup if config is invalid
   - Clear error messages pointing to specific fields

#### 13.2 Environment Variable Interpolation

**Alternatives Evaluated**:

| Approach | Pros | Cons |
|----------|------|------|
| **Custom regex preprocessor** | Simple, no deps, full control | Must handle edge cases |
| **python-dotenv** | Well-tested, common | Doesn't integrate with YAML |
| **dynaconf** | Full-featured config library | Overkill, adds dependency |
| **envsubst (shell)** | Standard Unix tool | Requires shell preprocessing |

**Decision**: Custom regex preprocessor with pattern `\$\{([^}:]+)(?::-([^}]*))?\}`
- Matches `${VAR}` (required) and `${VAR:-default}` (optional)
- Simple to implement and test
- Recursive application for nested structures

#### 13.3 Pydantic YAML Integration

**Approach**:
```python
raw_config = yaml.safe_load(file)
config = AppConfig.model_validate(raw_config)
```

**Pydantic v2 Features Used**:
- `model_validator(mode="after")` for cross-field validation (endpoint references)
- `Field(gt=0, ge=0, le=2)` for numeric constraints
- `Field(pattern=r"...")` for string validation
- `frozen=True` for immutable config

#### 13.4 Hot Reload Patterns (Optional FR-090)

**Decision**: Defer to future enhancement

**Rationale**:
- Complexity tradeoff: Hot reload adds complexity for edge cases
- Most config changes are deployment events anyway
- Kubernetes/Docker pattern is to restart containers on config changes
- Implement cache clearing function for testing and manual reload

#### 13.5 Default Configuration Strategy

**Decision**: Provide complete in-code defaults

**Rationale**:
- Developers should be able to run without any config file
- Default endpoints work with standard Databricks setup
- Log info message when using defaults

### Summary of Decisions

| Decision Area | Choice | Rationale |
|--------------|--------|-----------|
| YAML Library | PyYAML | Standard, already in deps |
| Env Var Interpolation | Custom regex | Simple, no new deps |
| Validation | Pydantic v2 | Strong typing, good errors |
| Config Location | `config/app.yaml` | Standard location at repo root |
| Hot Reload | Deferred | Complexity vs value |
| Default Fallback | In-code defaults | Dev experience |

### Configuration Schema

```yaml
# config/app.yaml - Top-level structure
default_role: analytical

endpoints:
  databricks-gpt-oss-20b:
    endpoint_identifier: databricks-gpt-oss-20b
    max_context_window: 32000
    tokens_per_minute: 200000

models:
  micro:
    endpoints: [databricks-gpt-oss-20b]
    temperature: 0.5
    max_tokens: 4000

  simple:
    endpoints: [databricks-gpt-oss-20b]
    temperature: 0.5
    max_tokens: 8000
    reasoning_effort: low

  analytical:
    endpoints: [databricks-gpt-oss-120b]
    temperature: 0.7
    max_tokens: 12000
    reasoning_effort: medium

  complex:
    endpoints: [databricks-gpt-oss-120b]
    temperature: 0.7
    max_tokens: 25000
    reasoning_effort: high
    reasoning_budget: 8000

agents:
  researcher:
    max_search_queries: 2
    max_urls_to_crawl: 3
  planner:
    max_plan_iterations: 3
  coordinator:
    max_clarification_rounds: 3

search:
  brave:
    requests_per_second: 1.0
    freshness: "month"

truncation:
  log_preview: 200
  error_message: 500
```

### Implementation Notes

1. **Config Loading Order**:
   1. Check `config/app.yaml`
   2. Check fallback path (for package installations)
   3. Use in-code defaults

2. **Validation Sequence**:
   1. Load YAML file
   2. Interpolate environment variables
   3. Validate with Pydantic
   4. Cache result (single load per process)

3. **Error Handling**:
   - FileNotFoundError → Fall back to defaults
   - yaml.YAMLError → Fail fast with parse error
   - ValidationError → Fail fast with field-level errors
   - Missing env var → Fail fast with variable name

### Sources
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [12-Factor App Config](https://12factor.net/config)
