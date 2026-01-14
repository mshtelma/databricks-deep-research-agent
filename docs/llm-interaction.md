# LLM Interaction Methods

## Overview

The Deep Research Agent uses tiered model routing for cost/quality optimization, structured output via Pydantic models, and the ReAct pattern for intelligent research decisions.

## Model Tier Routing

### Three Tiers

| Tier | Use Cases | Example Models | Temperature |
|------|-----------|----------------|-------------|
| **Simple** | Classification, quick decisions | Gemini Flash | 0.3 |
| **Analytical** | Planning, research, verification | Claude Sonnet | 0.7 |
| **Complex** | Synthesis, extended reasoning | Claude with ER | 0.7 |

### Tier Assignment by Agent

| Agent | Tier | Rationale |
|-------|------|-----------|
| Coordinator | Simple | Quick classification |
| Background | Simple | Fast search queries |
| Planner | Analytical | Complex planning |
| Researcher | Analytical | Research decisions |
| Reflector | Simple | Binary decisions |
| Synthesizer | Complex | Quality writing |

## LLM Client Architecture

**File**: `src/services/llm/client.py`

```python
class LLMClient:
    async def complete(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.ANALYTICAL,
        response_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Complete with automatic tier routing and failover.

        Features:
        - Endpoint selection by tier priority
        - Automatic retry with backoff on 429
        - Fallback to next endpoint on failure
        - MLflow span tracing
        - Structured output via Pydantic models
        """
```

### Key Features

1. **Endpoint Selection**: Picks first available endpoint for tier
2. **Automatic Retry**: Exponential backoff on rate limits
3. **Failover**: Falls back to alternate endpoints
4. **MLflow Tracing**: Spans for each LLM call
5. **Structured Output**: Pydantic model validation

## Structured Output Pattern

All agents use Pydantic models for structured LLM responses:

### Example: Coordinator

```python
from pydantic import BaseModel
from typing import Literal

class CoordinatorOutput(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    follow_up_type: Literal["new_topic", "clarification", "complex_follow_up"]
    is_ambiguous: bool
    clarifying_questions: list[str] = []
    recommended_depth: Literal["auto", "light", "medium", "extended"] = "auto"
    reasoning: str

# Usage
response = await llm.complete(
    messages=[
        {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ],
    tier=ModelTier.SIMPLE,
    response_format=CoordinatorOutput,  # ← Structured output
)

output = CoordinatorOutput.model_validate(response.parsed)
```

### Example: Planner

```python
class PlanStep(BaseModel):
    title: str
    description: str
    step_type: Literal["research", "analysis"]
    needs_search: bool

class PlanOutput(BaseModel):
    title: str
    thought: str
    steps: list[PlanStep]
    has_enough_context: bool = False
```

### Example: Reflector

```python
class ReflectorOutput(BaseModel):
    decision: Literal["continue", "adjust", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = None
```

## Rate Limiting & Resilience

### Configuration

```yaml
rate_limiting:
  max_retries: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 60.0
  backoff_strategy: exponential  # or "linear"
  jitter: true
```

### Retry Logic

```python
async def _complete_with_retry(self, ...):
    for attempt in range(max_retries):
        try:
            return await self._complete(...)
        except RateLimitError:
            delay = self._calculate_delay(attempt)
            await asyncio.sleep(delay)
    raise RateLimitExhausted()
```

### Backoff Strategies

| Strategy | Formula | Example (initial=1s) |
|----------|---------|---------------------|
| Exponential | `initial * 2^attempt` | 1s, 2s, 4s, 8s |
| Linear | `initial * (attempt + 1)` | 1s, 2s, 3s, 4s |

### Endpoint Health Tracking

```python
@dataclass
class EndpointHealth:
    endpoint_name: str
    consecutive_failures: int = 0
    last_failure_time: datetime | None = None
    backoff_until: datetime | None = None

    def is_healthy(self) -> bool:
        if self.backoff_until is None:
            return True
        return datetime.now(UTC) > self.backoff_until
```

## Context Truncation

**File**: `src/services/llm/truncation.py`

When context exceeds model limits:

```python
def truncate_messages(
    messages: list[dict],
    max_tokens: int,
    preserve_system: bool = True,
    preserve_recent: int = 3,
) -> list[dict]:
    """
    Truncate messages to fit context window.

    Priority:
    1. System prompt (always preserved)
    2. Recent N messages (preserved)
    3. Older messages (truncated/removed)
    """
```

### Truncation Strategy

```
┌─────────────────────────────────────────────────┐
│ Context Window                                  │
├─────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────┐ │
│ │ System Prompt (always preserved)            │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────┐ │
│ │ Older messages (truncated first)            │ │
│ │ ...                                         │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────┐ │
│ │ Recent 3 messages (preserved)               │ │
│ │ - User query                                │ │
│ │ - Last assistant response                   │ │
│ │ - Current user message                      │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## ReAct Researcher Pattern

**File**: `src/agent/nodes/react_researcher.py`

The ReAct (Reasoning + Acting) pattern gives the LLM control over research decisions.

### ReAct Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ReAct RESEARCHER LOOP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  System Prompt:                                                              │
│  "You are a research assistant. Use tools to find information.               │
│   Available tools: web_search(query), web_crawl(index)                       │
│   Stop when you have 3+ high-quality sources OR 10+ crawls."                 │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LOOP while tool_call_count < max_tool_calls:                           │ │
│  │                                                                        │ │
│  │  1. LLM decides action                                                 │ │
│  │     ↓                                                                  │ │
│  │  2. Parse tool calls from response                                     │ │
│  │     ↓                                                                  │ │
│  │  ┌─────────────────┐    ┌─────────────────┐                           │ │
│  │  │ web_search(q)   │ OR │ web_crawl(idx)  │                           │ │
│  │  │ → results [0-4] │    │ → page content  │                           │ │
│  │  │ → url_registry  │    │ → quality check │                           │ │
│  │  └─────────────────┘    └─────────────────┘                           │ │
│  │     ↓                                                                  │ │
│  │  3. Append result to message history                                   │ │
│  │     ↓                                                                  │ │
│  │  4. Yield event: tool_call, tool_result                                │ │
│  │     ↓                                                                  │ │
│  │  IF no tool calls in response → LLM satisfied → BREAK                  │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output: high_quality_sources, all observations                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Definitions

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_crawl",
            "description": "Fetch full content from a search result by index",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"}
                },
                "required": ["index"]
            }
        }
    }
]
```

### URL Registry (Hallucination Prevention)

```python
class UrlRegistry:
    """
    LLM only sees INDEX numbers, not raw URLs.
    Prevents URL hallucination.
    """
    def __init__(self):
        self._urls: dict[int, UrlEntry] = {}
        self._next_index: int = 0

    def add(self, url: str, title: str) -> int:
        """Add URL and return index for LLM reference."""
        index = self._next_index
        self._urls[index] = UrlEntry(url=url, title=title)
        self._next_index += 1
        return index

    def get(self, index: int) -> UrlEntry:
        """Get actual URL for crawling."""
        return self._urls[index]
```

**Why URL Registry?**
- LLMs can hallucinate URLs
- By providing indices, LLM can only reference URLs we've seen
- Prevents navigation to non-existent pages

### Classic vs ReAct Modes

| Aspect | Classic | ReAct |
|--------|---------|-------|
| Control | Fixed pipeline | LLM decides |
| Searches | Configured count | Quality-based |
| Crawls | Top N results | LLM selects |
| Stopping | After N operations | LLM satisfied |
| Speed | Faster | More thorough |
| Depth | Light | Medium/Extended |

### ReAct State

```python
@dataclass
class ReactResearchState:
    messages: list[dict]                # Conversation history with LLM
    high_quality_sources: list[str]     # URLs deemed high-quality
    low_quality_sources: list[str]      # URLs deemed low-quality
    tool_call_count: int                # Calls made this step
    crawled_content: dict[str, str]     # URL → page content
    url_registry: UrlRegistry           # Index → URL mapping
```

### Event Emission

```python
# Tool call event
yield ToolCallEvent(
    tool="web_search",
    args={"query": "AI regulations 2024"},
    call_number=3
)

# Tool result event
yield ToolResultEvent(
    tool="web_search",
    result_preview="Found 5 results for AI regulations...",
    sources_crawled=2
)

# Research complete event
yield ResearchCompleteEvent(
    reason="Found 3+ high-quality sources",
    tool_calls=8,
    high_quality_sources=4
)
```

## MLflow Tracing

All LLM calls are traced in MLflow:

```python
with mlflow.start_span(name="llm_analytical", span_type="LLM") as span:
    span.set_attributes({
        "llm.tier": tier.value,
        "llm.endpoint": endpoint_name,
        "llm.temperature": temperature,
        "llm.max_tokens": max_tokens,
    })

    response = await self._complete(...)

    span.set_attributes({
        "llm.input_tokens": response.usage.prompt_tokens,
        "llm.output_tokens": response.usage.completion_tokens,
    })
```

### Trace Hierarchy

```
research_orchestration (CHAIN)
├── coordinator (AGENT)
│   └── llm_simple (LLM)
├── background (AGENT)
│   └── llm_simple (LLM)
├── planner (AGENT)
│   └── llm_analytical (LLM)
├── researcher_step_1 (AGENT)
│   ├── llm_analytical (LLM) - search queries
│   └── llm_analytical (LLM) - observation
├── reflector (AGENT)
│   └── llm_simple (LLM)
└── synthesizer (AGENT)
    └── llm_complex (LLM)
```

## Configuration

```yaml
# Model endpoints
endpoints:
  databricks-gemini-flash:
    endpoint_identifier: google-gemini-flash
    max_context_window: 1000000
    tokens_per_minute: 500000

  databricks-claude-sonnet:
    endpoint_identifier: anthropic-claude-sonnet
    max_context_window: 200000
    tokens_per_minute: 100000

  databricks-claude-sonnet-er:
    endpoint_identifier: anthropic-claude-sonnet-er
    max_context_window: 200000
    tokens_per_minute: 50000

# Model roles (tiers)
models:
  simple:
    endpoints:
      - databricks-gemini-flash
    temperature: 0.3
    max_tokens: 4000

  analytical:
    endpoints:
      - databricks-claude-sonnet
      - databricks-gemini-flash  # fallback
    temperature: 0.7
    max_tokens: 8000
    fallback_on_429: true

  complex:
    endpoints:
      - databricks-claude-sonnet-er
      - databricks-claude-sonnet  # fallback
    temperature: 0.7
    max_tokens: 16000
    reasoning_effort: medium

# Rate limiting
rate_limiting:
  max_retries: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 60.0
  backoff_strategy: exponential
  jitter: true
```

## See Also

- [Agent Orchestration](./agents.md) - Agent responsibilities
- [Citation Pipeline](./citation-pipeline.md) - Verification stages
- [Configuration](./configuration.md) - YAML settings
