# Data Model: Deep Research Agent

**Date**: 2025-12-21
**Feature**: 001-deep-research-agent
**Source**: [spec.md](./spec.md) Key Entities section

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    USER                                          │
│  (Databricks workspace identity - not stored, resolved at runtime)              │
└─────────────────────────────────────────────────────────────────────────────────┘
         │ owns 1:N                    │ has 1:1
         ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐
│        Chat         │      │   UserPreferences   │
├─────────────────────┤      ├─────────────────────┤
│ id (PK)             │      │ user_id (PK)        │
│ user_id             │      │ system_instructions │
│ title               │      │ default_depth       │
│ created_at          │      │ ui_preferences      │
│ updated_at          │      │ created_at          │
│ deleted_at (soft)   │      │ updated_at          │
│ is_archived         │      └─────────────────────┘
└─────────────────────┘
         │ contains 1:N
         ▼
┌─────────────────────┐
│      Message        │
├─────────────────────┤
│ id (PK)             │
│ chat_id (FK)        │
│ role (user/agent)   │
│ content             │
│ created_at          │
│ updated_at          │
│ is_edited           │
└─────────────────────┘
         │ has 0:1 (agent messages only)
         ▼
┌─────────────────────┐      ┌─────────────────────┐
│  ResearchSession    │      │  MessageFeedback    │
├─────────────────────┤      ├─────────────────────┤
│ id (PK)             │      │ id (PK)             │
│ message_id (FK)     │◄─────│ message_id (FK)     │
│ query_classification│      │ rating (+1/-1)      │
│ research_depth      │      │ error_report        │
│ reasoning_steps     │      │ created_at          │
│ status              │      └─────────────────────┘
│ started_at          │
│ completed_at        │
└─────────────────────┘
         │ contains 1:N
         ▼
┌─────────────────────┐
│       Source        │
├─────────────────────┤
│ id (PK)             │
│ session_id (FK)     │
│ url                 │
│ title               │
│ snippet             │
│ relevance_score     │
│ fetched_at          │
│ fetch_status        │
└─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION ENTITIES                                   │
│                    (Loaded from external config, cached in memory)              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│     ModelRole       │      │   ModelEndpoint     │      │   EndpointHealth    │
├─────────────────────┤      ├─────────────────────┤      ├─────────────────────┤
│ name (PK)           │──1:N─│ id (PK)             │──1:1─│ endpoint_id (PK)    │
│ selection_strategy  │      │ role_name (FK)      │      │ is_healthy          │
│ description         │      │ endpoint_identifier │      │ last_success_at     │
└─────────────────────┘      │ temperature         │      │ error_count         │
                             │ max_context_window  │      │ rate_limited_until  │
                             └─────────────────────┘      └─────────────────────┘

┌─────────────────────┐
│     AuditLog        │
├─────────────────────┤
│ id (PK)             │
│ user_id             │
│ action_type         │
│ target_entity       │
│ target_id           │
│ metadata (JSONB)    │
│ created_at          │
└─────────────────────┘
```

---

## Entity Definitions

### User (Runtime Resolution)

**Note**: Users are not stored in the database. User identity is resolved at runtime via Databricks WorkspaceClient. User ID is the Databricks workspace user identifier.

```python
class User(Protocol):
    """User identity from Databricks workspace."""
    user_id: str          # Databricks workspace user ID
    email: str            # User email (from workspace)
    display_name: str     # User display name
```

---

### Chat

Represents a conversation thread between a user and the agent.

```python
class ChatStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class Chat(BaseModel):
    id: UUID
    user_id: str                    # Databricks workspace user ID
    title: str | None = None        # Custom or auto-generated title
    status: ChatStatus = ChatStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None  # Soft delete timestamp

    class Config:
        from_attributes = True
```

**Validation Rules**:
- `user_id`: Required, non-empty string
- `title`: Max 200 characters, auto-generated from first message if not provided
- `deleted_at`: If set, chat is in soft-delete state (recoverable for 30 days)

**State Transitions**:
```
ACTIVE ──archive()──► ARCHIVED
ACTIVE ──delete()───► DELETED (soft)
ARCHIVED ──unarchive()──► ACTIVE
ARCHIVED ──delete()──► DELETED (soft)
DELETED ──(30 days)──► PURGED (permanent)
DELETED ──restore()──► ACTIVE (if within 30 days)
```

---

### Message

A single exchange in a chat conversation.

```python
class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"  # For clarifying questions from agent

class Message(BaseModel):
    id: UUID
    chat_id: UUID                   # FK to Chat
    role: MessageRole
    content: str
    created_at: datetime
    updated_at: datetime | None = None
    is_edited: bool = False         # True if user edited this message

    class Config:
        from_attributes = True
```

**Validation Rules**:
- `content`: Required, non-empty for user messages; may be empty for agent streaming placeholder
- `role`: Must be valid MessageRole enum value
- `is_edited`: Only applicable to USER role messages

**Uniqueness**: `id` is globally unique (UUID v4)

---

### ResearchSession

The execution context for a research query.

```python
class ResearchDepth(str, Enum):
    AUTO = "auto"
    LIGHT = "light"       # 1-2 search iterations
    MEDIUM = "medium"     # 3-5 search iterations
    EXTENDED = "extended" # 6-10 search iterations

class ResearchStatus(str, Enum):
    PENDING = "pending"
    CLASSIFYING = "classifying"  # Analyzing query complexity
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class QueryClassification(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    follow_up_type: Literal["new_topic", "clarification", "complex_follow_up"]
    is_ambiguous: bool
    clarifying_questions: list[str] = []
    reasoning: str

class ReasoningStep(BaseModel):
    step_number: int
    action: str                     # "search", "fetch", "reflect", "synthesize"
    input_data: dict[str, Any]      # Action-specific input
    output_summary: str             # Brief summary of result
    timestamp: datetime

class ResearchSession(BaseModel):
    id: UUID
    message_id: UUID                # FK to Message (agent response)
    query_classification: QueryClassification | None
    research_depth: ResearchDepth
    reasoning_steps: list[ReasoningStep] = []
    status: ResearchStatus = ResearchStatus.PENDING
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True
```

**State Transitions**:
```
PENDING ──start()──► CLASSIFYING
CLASSIFYING ──classified()──► RESEARCHING (or skip if simple)
CLASSIFYING ──ambiguous()──► (ask clarifying questions, wait)
RESEARCHING ──complete()──► SYNTHESIZING
SYNTHESIZING ──finish()──► COMPLETED
ANY ──cancel()──► CANCELLED
ANY ──error()──► FAILED
```

---

### Source

A web resource referenced in research.

```python
class FetchStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # Rate limited or inaccessible

class Source(BaseModel):
    id: UUID
    session_id: UUID               # FK to ResearchSession
    url: str
    title: str | None = None
    snippet: str | None = None     # Search result snippet
    full_content: str | None = None  # Fetched page content (truncated)
    relevance_score: float | None = None  # 0.0 to 1.0
    fetch_status: FetchStatus = FetchStatus.PENDING
    fetched_at: datetime | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True
```

**Validation Rules**:
- `url`: Valid URL format, max 2000 characters
- `relevance_score`: 0.0 to 1.0, nullable if not computed
- `full_content`: Truncated to max 50,000 characters to fit in context

---

### UserPreferences

Persistent user settings.

```python
class UserPreferences(BaseModel):
    user_id: str                    # PK, Databricks workspace user ID
    system_instructions: str | None = None  # Custom instructions for all chats
    default_depth: ResearchDepth = ResearchDepth.AUTO
    ui_preferences: dict[str, Any] = {}  # Theme, layout, etc.
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
```

**Validation Rules**:
- `system_instructions`: Max 10,000 characters
- `ui_preferences`: Valid JSON object

---

### MessageFeedback

User-provided feedback on agent responses.

```python
class FeedbackRating(int, Enum):
    NEGATIVE = -1
    POSITIVE = 1

class MessageFeedback(BaseModel):
    id: UUID
    message_id: UUID               # FK to Message (agent response)
    user_id: str                   # Databricks workspace user ID
    rating: FeedbackRating
    error_report: str | None = None  # Factual error description
    created_at: datetime

    class Config:
        from_attributes = True
```

**Validation Rules**:
- `error_report`: Max 5,000 characters
- One feedback per message per user (upsert behavior)

**MLflow Integration**: Feedback is logged to MLflow traces for quality analysis.

---

### AuditLog

Record of user actions for security and compliance.

```python
class AuditActionType(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    ARCHIVE = "archive"
    RESTORE = "restore"

class AuditLog(BaseModel):
    id: UUID
    user_id: str                   # Databricks workspace user ID
    action_type: AuditActionType
    target_entity: str             # "chat", "message", "preferences"
    target_id: str                 # Entity ID
    metadata: dict[str, Any] = {}  # Additional context
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True
```

**Retention**: Minimum 1 year per spec requirements.

---

### AppConfig (Central Configuration Entity)

Central application configuration loaded from YAML file. Added in FR-081 to FR-090.

```python
class AppConfig(BaseModel):
    """Central application configuration loaded from config/app.yaml."""

    # Default model role
    default_role: str = "analytical"

    # Model endpoint definitions
    endpoints: dict[str, EndpointConfig] = {}

    # Model role definitions (micro, simple, analytical, complex)
    models: dict[str, ModelRoleConfig] = {}

    # Agent configurations
    agents: AgentsConfig

    # Search service configurations
    search: SearchConfig

    # Truncation limits for consistency
    truncation: TruncationConfig

    @model_validator(mode="after")
    def validate_endpoint_references(self) -> "AppConfig":
        """Ensure all role endpoints exist in endpoints dict."""
        for role_name, role_config in self.models.items():
            for endpoint_id in role_config.endpoints:
                if endpoint_id not in self.endpoints:
                    raise ValueError(
                        f"Role '{role_name}' references undefined endpoint: {endpoint_id}"
                    )
        return self

    class Config:
        frozen = True  # Immutable configuration
```

**Configuration Source**: `config/app.yaml` with environment variable interpolation (`${VAR:-default}`)

**Key Features**:
- Loaded once at startup, cached in memory
- Falls back to sensible defaults when config file absent
- Validates endpoint references across roles
- Supports environment variable interpolation for secrets

---

### ModelRole (Configuration Entity)

Named model capability tier with role-level defaults. Loaded from `config/app.yaml`.

```python
class SelectionStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"

class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelRoleConfig(BaseModel):
    """Configuration for a model role (tier)."""
    endpoints: list[str]               # Ordered list of endpoint IDs (priority order)
    temperature: float = 0.7
    max_tokens: int = 8000             # Output token limit
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    reasoning_budget: int | None = None  # For high reasoning effort (o-series)
    tokens_per_minute: int = 100000    # Rate limit for this role
    rotation_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    fallback_on_429: bool = True       # Enable failover on rate limit

    class Config:
        frozen = True  # Immutable configuration
```

**Configuration Example**:
```yaml
model_roles:
  simple:
    endpoints: [databricks-gpt-oss-20b, databricks-llama-4-maverick]
    temperature: 0.5
    max_tokens: 8000
    reasoning_effort: low
    rotation_strategy: priority
    fallback_on_429: true

  analytical:
    endpoints: [databricks-gpt-oss-120b, databricks-gpt-oss-20b]
    temperature: 0.7
    max_tokens: 12000
    reasoning_effort: medium

  complex:
    endpoints: [databricks-gpt-oss-120b, databricks-claude-3-7-sonnet]
    temperature: 0.7
    max_tokens: 25000
    reasoning_effort: high
    reasoning_budget: 8000
```

---

### ModelEndpoint (Configuration Entity)

A configured connection to a specific model with endpoint-specific settings. Loaded from `config/app.yaml`.

```python
class EndpointConfig(BaseModel):
    """Configuration for a single model endpoint."""
    endpoint_identifier: str           # Databricks serving endpoint name

    # Endpoint-specific (REQUIRED - no inheritance)
    max_context_window: int            # Input context limit (e.g., 128000, 32000)
    tokens_per_minute: int             # Rate limit - per endpoint

    # Optional overrides (inherit from role if not set)
    temperature: float | None = None
    max_tokens: int | None = None      # Output token limit override
    reasoning_effort: ReasoningEffort | None = None
    reasoning_budget: int | None = None
    supports_structured_output: bool = False  # JSON mode support

    class Config:
        frozen = True
```

**Override Semantics**: When computing effective configuration for a request:
1. Start with role-level defaults
2. Merge endpoint-specific overrides (endpoint values win)
3. max_context_window and tokens_per_minute are always endpoint-specific

---

### AgentsConfig (Configuration Entity)

Configuration for all agent components. Loaded from `config/app.yaml`.

```python
class ResearcherConfig(BaseModel):
    """Configuration for the Researcher agent."""
    max_search_queries: int = 2
    max_search_results: int = 10
    max_urls_to_crawl: int = 3
    content_preview_length: int = 3000
    content_storage_length: int = 10000
    max_previous_observations: int = 3
    page_contents_limit: int = 8000
    max_generated_queries: int = 3

class PlannerConfig(BaseModel):
    """Configuration for the Planner agent."""
    max_plan_iterations: int = 3

class CoordinatorConfig(BaseModel):
    """Configuration for the Coordinator agent."""
    max_clarification_rounds: int = 3
    enable_clarification: bool = True

class SynthesizerConfig(BaseModel):
    """Configuration for the Synthesizer agent."""
    max_report_length: int = 50000

class AgentsConfig(BaseModel):
    """Configuration for all agents."""
    researcher: ResearcherConfig
    planner: PlannerConfig
    coordinator: CoordinatorConfig
    synthesizer: SynthesizerConfig
```

---

### SearchConfig (Configuration Entity)

Configuration for search services. Loaded from `config/app.yaml`.

```python
class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""
    requests_per_second: float = 1.0
    default_result_count: int = 10
    freshness: str = "month"  # pd, pw, pm, py

class SearchConfig(BaseModel):
    """Configuration for search services."""
    brave: BraveSearchConfig
```

---

### TruncationConfig (Configuration Entity)

Centralized truncation limits for consistency across codebase. Loaded from `config/app.yaml`.

```python
class TruncationConfig(BaseModel):
    """Configuration for text truncation limits."""
    log_preview: int = 200      # Truncation limit for log messages
    error_message: int = 500    # Truncation limit for error messages
    query_display: int = 100    # Truncation limit for query display
    source_snippet: int = 300   # Truncation limit for source snippets
```

**Configuration Example**:
```yaml
endpoints:
  databricks-gpt-oss-120b:
    max_context_window: 128000
    tokens_per_minute: 200000

  databricks-gpt-oss-20b:
    max_context_window: 32000
    tokens_per_minute: 150000

  databricks-claude-3-7-sonnet:
    max_context_window: 200000
    tokens_per_minute: 50000
    temperature: 0.5  # Override role default
```

---

### EndpointHealth (Runtime State - In-Memory Only)

Tracks endpoint availability and rate limiting. **NOT persisted to database** - maintained in-memory per process instance.

```python
class EndpointHealth(BaseModel):
    """Runtime state - NOT persisted to database."""
    endpoint_id: str               # FK to ModelEndpoint.id
    is_healthy: bool = True
    last_success_at: datetime | None = None
    consecutive_errors: int = 0
    rate_limited_until: datetime | None = None

    # Per-endpoint rate limiting
    tokens_used_this_minute: int = 0
    minute_started_at: datetime | None = None

    def mark_success(self) -> None:
        self.is_healthy = True
        self.last_success_at = datetime.utcnow()
        self.consecutive_errors = 0
        self.rate_limited_until = None

    def mark_failure(self, rate_limited: bool = False) -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= 3:
            self.is_healthy = False
        if rate_limited:
            # Exponential backoff with jitter
            delay = min(60, 2 ** self.consecutive_errors) + random.uniform(0, 1)
            self.rate_limited_until = datetime.utcnow() + timedelta(seconds=delay)

    def record_tokens(self, tokens: int) -> None:
        """Track token usage for rate limiting."""
        now = datetime.utcnow()
        if self.minute_started_at is None or (now - self.minute_started_at).seconds >= 60:
            self.minute_started_at = now
            self.tokens_used_this_minute = 0
        self.tokens_used_this_minute += tokens

    def can_handle_request(self, estimated_tokens: int, tokens_per_minute: int) -> bool:
        """Check if endpoint can handle request within rate limit."""
        if self.rate_limited_until and datetime.utcnow() < self.rate_limited_until:
            return False
        return (self.tokens_used_this_minute + estimated_tokens) <= tokens_per_minute
```

**Scope**: Each application instance independently tracks endpoint health. Health state is not shared across replicas.

---

### QueryClassification (Embedded Value Object)

Result of analyzing a user query (embedded in ResearchSession).

```python
class QueryClassification(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    follow_up_type: Literal["new_topic", "clarification", "complex_follow_up"]
    is_ambiguous: bool
    clarifying_questions: list[str] = []  # 1-3 questions if ambiguous
    recommended_depth: ResearchDepth
    reasoning: str                 # Explanation shown to user

    class Config:
        frozen = True
```

---

## Multi-Agent Architecture Entities

The following entities support the 5-agent architecture (Coordinator, Planner, Researcher, Reflector, Synthesizer). These are runtime entities stored in ResearchState, not persisted to the database.

### StepType (Enumeration)

Type of research plan step, determines which agent executes it.

```python
class StepType(str, Enum):
    RESEARCH = "research"    # Web search/crawl - executed by Researcher agent
    ANALYSIS = "analysis"    # Pure reasoning - executed by Synthesizer agent
```

---

### StepStatus (Enumeration)

Execution status of a plan step.

```python
class StepStatus(str, Enum):
    PENDING = "pending"          # Not yet started
    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"      # Successfully finished
    SKIPPED = "skipped"          # Skipped by Reflector (COMPLETE decision)
```

---

### PlanStep (Value Object)

A single step in a research plan.

```python
class PlanStep(BaseModel):
    id: str                       # Unique step identifier within plan
    title: str                    # Brief description (e.g., "Search for recent papers")
    description: str              # Detailed instructions for the step
    step_type: StepType           # RESEARCH or ANALYSIS
    needs_search: bool            # Whether web search is required
    status: StepStatus = StepStatus.PENDING
    observation: str | None = None  # Result from Researcher after execution

    class Config:
        frozen = False  # Mutable - status and observation updated during execution
```

**Validation Rules**:
- `title`: Max 200 characters
- `description`: Max 2000 characters
- `observation`: Updated by Researcher agent after step execution

---

### Plan (Value Object)

A structured research plan created by the Planner agent.

```python
class Plan(BaseModel):
    id: str                       # Unique plan identifier
    title: str                    # Plan title summarizing research goal
    thought: str                  # Planner's reasoning for this plan structure
    steps: list[PlanStep]         # Ordered list of steps to execute
    has_enough_context: bool = False  # True if Planner determines no research needed
    iteration: int = 1            # Current iteration (incremented on ADJUST)

    class Config:
        frozen = False  # Mutable - updated during execution and replanning
```

**Behavior**:
- When Reflector decides ADJUST, Planner creates a new Plan with incremented iteration
- Steps are executed sequentially, not in parallel
- `has_enough_context=True` skips research and goes directly to Synthesizer

---

### ReflectionDecision (Enumeration)

The decision made by the Reflector agent after evaluating a completed step.

```python
class ReflectionDecision(str, Enum):
    CONTINUE = "continue"   # Proceed to next step in plan
    ADJUST = "adjust"       # Return to Planner for replanning
    COMPLETE = "complete"   # Skip remaining steps, go to Synthesizer
```

**Decision Criteria**:
```
CONTINUE → Plan still valid
  - Findings align with expectations
  - No new research directions discovered
  - More steps needed to answer query

ADJUST → Need to replan
  - Findings contradict initial assumptions
  - Found significantly more/less than expected
  - New research directions emerged
  - Some remaining steps now seem irrelevant

COMPLETE → Have enough information
  - Original question sufficiently answered
  - Remaining steps would be redundant
  - Quality threshold reached
```

---

### ReflectionResult (Value Object)

The output from the Reflector agent.

```python
class ReflectionResult(BaseModel):
    decision: ReflectionDecision
    reasoning: str                # Explanation for the decision
    suggested_changes: list[str] | None = None  # Hints for Planner (on ADJUST)

    class Config:
        frozen = True
```

**Usage**:
- Reflector evaluates after EACH step (not after all steps - key differentiator from deer-flow)
- `suggested_changes` provides guidance to Planner when decision is ADJUST
- `reasoning` is logged for observability and debugging

---

### ResearchState (Runtime State)

Extended state for the multi-agent workflow. Passed between agents during execution.

```python
class ResearchState(BaseModel):
    """Runtime state for multi-agent research workflow."""

    # Original query context
    query: str
    conversation_history: list[Message] = []

    # Clarification (Coordinator phase)
    enable_clarification: bool = True
    clarification_rounds: int = 0
    max_clarification_rounds: int = 3
    clarification_history: list[str] = []
    is_clarification_complete: bool = False

    # Query classification
    query_classification: QueryClassification | None = None
    is_simple_query: bool = False  # Coordinator can answer directly
    direct_response: str | None = None  # For simple queries

    # Background investigation (pre-planning)
    background_investigation_results: str = ""

    # Planning
    current_plan: Plan | None = None
    plan_iterations: int = 0
    max_plan_iterations: int = 3  # Prevent infinite replan loops

    # Step execution (Researcher phase)
    current_step_index: int = 0
    last_observation: str = ""  # Most recent step result
    all_observations: list[str] = []  # Aggregated findings from all steps

    # Reflection
    last_reflection: ReflectionResult | None = None
    reflection_history: list[ReflectionResult] = []  # For debugging/tracing

    # Final output (Synthesizer phase)
    final_report: str = ""
    sources: list[Source] = []

    class Config:
        arbitrary_types_allowed = True
```

**Agent Responsibilities**:

| Agent | Reads | Writes |
|-------|-------|--------|
| Coordinator | query, conversation_history | query_classification, is_simple_query, direct_response, clarification_* |
| Background Investigator | query | background_investigation_results |
| Planner | query, background_investigation_results, all_observations | current_plan, plan_iterations |
| Researcher | current_plan, current_step_index | last_observation, all_observations, current_step_index |
| Reflector | current_plan, last_observation, all_observations | last_reflection, reflection_history |
| Synthesizer | all_observations, sources | final_report |

---

## Database Schema (PostgreSQL/Lakebase)

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Chats table
CREATE TABLE chats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(200),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    CONSTRAINT chk_status CHECK (status IN ('active', 'archived', 'deleted'))
);

CREATE INDEX idx_chats_user_id ON chats(user_id);
CREATE INDEX idx_chats_user_status ON chats(user_id, status);
CREATE INDEX idx_chats_deleted_at ON chats(deleted_at) WHERE deleted_at IS NOT NULL;

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    is_edited BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT chk_role CHECK (role IN ('user', 'agent', 'system'))
);

CREATE INDEX idx_messages_chat_id ON messages(chat_id);
CREATE INDEX idx_messages_chat_created ON messages(chat_id, created_at);

-- Full-text search index for chat search (FR-037)
CREATE INDEX idx_messages_content_fts ON messages
    USING GIN (to_tsvector('english', content));

-- Research sessions table
CREATE TABLE research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    query_classification JSONB,
    research_depth VARCHAR(20) NOT NULL DEFAULT 'auto',
    reasoning_steps JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,

    CONSTRAINT chk_depth CHECK (research_depth IN ('auto', 'light', 'medium', 'extended')),
    CONSTRAINT chk_session_status CHECK (status IN (
        'pending', 'classifying', 'researching', 'synthesizing',
        'completed', 'cancelled', 'failed'
    ))
);

CREATE INDEX idx_sessions_message_id ON research_sessions(message_id);

-- Sources table
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    url VARCHAR(2000) NOT NULL,
    title VARCHAR(500),
    snippet TEXT,
    full_content TEXT,
    relevance_score REAL,
    fetch_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    fetched_at TIMESTAMPTZ,
    error_message TEXT,

    CONSTRAINT chk_fetch_status CHECK (fetch_status IN ('pending', 'success', 'failed', 'skipped')),
    CONSTRAINT chk_relevance CHECK (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1))
);

CREATE INDEX idx_sources_session_id ON sources(session_id);

-- User preferences table
CREATE TABLE user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    system_instructions TEXT,
    default_depth VARCHAR(20) NOT NULL DEFAULT 'auto',
    ui_preferences JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pref_depth CHECK (default_depth IN ('auto', 'light', 'medium', 'extended'))
);

-- Message feedback table
CREATE TABLE message_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    rating SMALLINT NOT NULL,
    error_report TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rating CHECK (rating IN (-1, 1)),
    CONSTRAINT uq_feedback_message_user UNIQUE (message_id, user_id)
);

CREATE INDEX idx_feedback_message_id ON message_feedback(message_id);

-- Audit log table (append-only)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(20) NOT NULL,
    target_entity VARCHAR(50) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_action_type CHECK (action_type IN (
        'create', 'read', 'update', 'delete', 'export', 'archive', 'restore'
    ))
);

CREATE INDEX idx_audit_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_target ON audit_logs(target_entity, target_id);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_chats_updated_at
    BEFORE UPDATE ON chats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

---

## Data Volume Estimates

| Entity | Estimated Volume | Retention |
|--------|-----------------|-----------|
| Chats | 100-1000 per user | Indefinite (soft delete 30d) |
| Messages | 10-100 per chat | Same as chat |
| ResearchSessions | 1 per agent message | Same as message |
| Sources | 5-20 per session | Same as session |
| UserPreferences | 1 per user | Indefinite |
| MessageFeedback | 0-1 per agent message | Indefinite |
| AuditLogs | ~50 per user per day | 1 year minimum |

**Scale Assumptions** (from spec):
- 10-100 concurrent users
- Team-scale usage, not enterprise-wide

---

## E2E Testing Types (Playwright)

**Added**: 2025-12-22
**User Story**: 9 - Automated End-to-End Testing

The following TypeScript types support the Playwright e2e testing infrastructure.

### Page Object Types

```typescript
// pages/chat.page.ts
import { Page, Locator } from '@playwright/test';

export interface MessageLocator {
  role: 'user' | 'agent';
  content: Locator;
  timestamp: Locator;
  actions?: {
    edit?: Locator;
    regenerate?: Locator;
    copy?: Locator;
  };
}

export class ChatPage {
  readonly page: Page;
  readonly messageInput: Locator;
  readonly sendButton: Locator;
  readonly stopButton: Locator;
  readonly loadingIndicator: Locator;
  readonly streamingIndicator: Locator;
  readonly messageList: Locator;

  constructor(page: Page) {
    this.page = page;
    this.messageInput = page.getByRole('textbox', { name: /message/i });
    this.sendButton = page.getByRole('button', { name: /send/i });
    this.stopButton = page.getByTestId('stop-button');
    this.loadingIndicator = page.getByTestId('loading-indicator');
    this.streamingIndicator = page.getByTestId('streaming-indicator');
    this.messageList = page.getByTestId('message-list');
  }

  async sendMessage(text: string): Promise<void>;
  async waitForAgentResponse(timeout?: number): Promise<void>;
  async getMessages(): Promise<MessageLocator[]>;
  async getLastAgentResponse(): Promise<string>;
  async editMessage(index: number, newText: string): Promise<void>;
  async regenerateResponse(): Promise<void>;
  async stopGeneration(): Promise<void>;
}
```

### Test Fixture Types

```typescript
// fixtures/types.ts
import { Page } from '@playwright/test';
import { ChatPage } from '../pages/chat.page';
import { SidebarPage } from '../pages/sidebar.page';

export interface TestFixtures {
  chatPage: ChatPage;
  sidebarPage: SidebarPage;
  authenticatedPage: Page;
}

export interface TestOptions {
  baseURL: string;
  timeout: number;
}
```

### Test Data Types

```typescript
// utils/test-data.ts
export interface TestQuery {
  text: string;
  type: 'smoke' | 'research' | 'follow-up' | 'long-running';
  expectedResponseTime: number;  // milliseconds
}

export interface TestConversation {
  queries: TestQuery[];
  expectedCitations?: number;
  expectedReasoningSteps?: number;
}

export const SMOKE_QUERIES: TestQuery[] = [
  { text: 'Hello', type: 'smoke', expectedResponseTime: 10000 },
  { text: 'What is 2+2?', type: 'smoke', expectedResponseTime: 10000 },
];

export const RESEARCH_QUERIES: TestQuery[] = [
  { text: 'What is Python used for?', type: 'research', expectedResponseTime: 60000 },
  { text: 'Explain machine learning basics', type: 'research', expectedResponseTime: 60000 },
];

export const FOLLOW_UP_QUERIES: TestQuery[] = [
  { text: 'Can you explain that more?', type: 'follow-up', expectedResponseTime: 30000 },
  { text: 'What about performance?', type: 'follow-up', expectedResponseTime: 30000 },
];
```

### Assertion Helpers

```typescript
// utils/assertions.ts
import { expect, Page } from '@playwright/test';

export interface ResponseValidation {
  hasContent: boolean;
  hasCitations?: boolean;
  citationCount?: number;
  hasReasoningSteps?: boolean;
  maxResponseTimeMs?: number;
}

export async function validateAgentResponse(
  page: Page,
  validation: ResponseValidation
): Promise<void> {
  const response = page.getByTestId('agent-response').last();

  if (validation.hasContent) {
    await expect(response).not.toBeEmpty();
  }

  if (validation.hasCitations) {
    const citations = response.locator('[data-testid="citation"]');
    await expect(citations).toHaveCount({ minimum: validation.citationCount ?? 1 });
  }

  if (validation.hasReasoningSteps) {
    const reasoning = page.getByTestId('reasoning-panel');
    await expect(reasoning).toBeVisible();
  }
}

export async function validateStopBehavior(
  page: Page,
  maxStopTimeMs: number = 2000
): Promise<void> {
  const startTime = Date.now();
  await page.getByTestId('stop-button').click();

  await expect(page.getByTestId('loading-indicator')).toBeHidden();
  const stopTime = Date.now() - startTime;

  expect(stopTime).toBeLessThan(maxStopTimeMs);
}
```

### Test Scenario Types

```typescript
// tests/types.ts
export interface E2ETestScenario {
  name: string;
  description: string;
  steps: E2ETestStep[];
  acceptance: AcceptanceScenario;
}

export interface E2ETestStep {
  action: 'navigate' | 'send' | 'wait' | 'click' | 'edit' | 'verify';
  target?: string;
  value?: string;
  timeout?: number;
}

export interface AcceptanceScenario {
  given: string;
  when: string;
  then: string;
}

// Mapping to User Story 9 acceptance scenarios
export const US9_SCENARIOS: E2ETestScenario[] = [
  {
    name: 'research-flow',
    description: 'User Story 9.1: Research query flow',
    steps: [
      { action: 'navigate', target: '/' },
      { action: 'send', value: 'What are the latest developments in AI?' },
      { action: 'wait', timeout: 120000 },
      { action: 'verify', target: 'agent-response' },
      { action: 'verify', target: 'citation' },
    ],
    acceptance: {
      given: 'the application is running',
      when: 'the e2e test suite is executed',
      then: 'tests simulate a user opening the chat interface, submitting a research query, and receiving a response with citations',
    },
  },
  // ... additional scenarios for 9.2-9.6
];
```
