# LLM Client

> Model tier routing, health tracking, fallback, and token budget management.

## Overview
FrameworkLLMClient wraps AsyncOpenAI with model tier routing, per-endpoint health tracking, rate-limit awareness, and automatic fallback.

## ModelTier Enum
- simple: Fast, cheap models for classification/routing (e.g., Llama 70B, GPT-4o-mini)
- analytical: Balanced models for planning/reflection (e.g., Llama 70B, GPT-4o)
- complex: Reasoning-heavy models for synthesis (e.g., Llama 405B, GPT-4o)

## FrameworkLLMClient

The main LLM client. It is **not** a Protocol -- it depends on the `openai` package directly.

### Constructor

```python
FrameworkLLMClient(
    openai_client: AsyncOpenAI,
    model_mapping: dict[str, str | ModelTierConfig],
    *,
    embedding_model: str | None = None,
    client_provider: Callable[[], AsyncOpenAI] | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `openai_client` | An `AsyncOpenAI` instance used for all LLM calls. |
| `model_mapping` | Maps tier names (or `ModelTier` values) to either a plain model name (`str`) or a `ModelTierConfig` with multiple endpoints and fallback rules. |
| `embedding_model` | Optional model name for embedding calls. Required only if you use `embed()` / `embed_single()`. |
| `client_provider` | Optional zero-arg callable that returns a fresh `AsyncOpenAI`. Used for OAuth token refresh on long-running workflows -- the client is replaced transparently before each call. |

### Factory: `from_databricks()`

```python
@classmethod
def from_databricks(
    cls,
    *,
    model: str = "databricks-claude-haiku-4-5",
    model_mapping: dict[str, str] | None = None,
) -> FrameworkLLMClient
```

Creates a client authenticated against Databricks serving endpoints. Auth chain (tried in order):

1. **Direct token** -- `DATABRICKS_HOST` + `DATABRICKS_TOKEN` env vars.
2. **SDK auto-detect** -- `WorkspaceClient()` with no args (covers profiles, Azure MSI, and all other SDK-supported auth methods). Uses a `client_provider` callback so OAuth tokens refresh automatically.

If `model_mapping` is not provided, all three tiers (`simple`, `analytical`, `complex`) map to the single `model` argument.

### `complete()`

```python
async def complete(
    messages: list[dict[str, Any]],
    tier: str | ModelTier = ModelTier.analytical,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: Any | None = None,
    structured_output: type | None = None,
) -> LLMResponse
```

Sends messages to the LLM and returns an `LLMResponse`. When a `ModelTierConfig` is configured for the given tier, the client will:

1. Select the best healthy endpoint via `_select_endpoint()`.
2. Track token usage against TPM limits.
3. On 429 errors, automatically try fallback endpoints if `fallback_on_429=True`.
4. Use exponential backoff with jitter for transient 5xx failures (up to 3 retries).
5. Mark endpoints healthy/unhealthy based on success/failure.

When `structured_output` is provided (a Pydantic model class), the client uses the OpenAI `beta.chat.completions.parse()` API. If structured output validation fails, it falls back to `response_format: json_object`, then to plain text if the model does not support JSON mode.

### `stream()`

```python
async def stream(
    messages: list[dict[str, Any]],
    tier: str | ModelTier = ModelTier.analytical,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[str | ToolCall, None]
```

Streams response tokens and tool calls. Yields `str` chunks for content deltas and `ToolCall` objects for completed tool calls (emitted after the stream ends). Supports the same rate-limit fallback as `complete()`.

### `embed()` / `embed_single()`

```python
async def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]
async def embed_single(text: str) -> list[float]
```

Batch or single-text embedding via `openai.embeddings.create()`. Uses the configured `embedding_model` unless overridden by the `model` parameter. Raises `ValueError` if no embedding model is configured and none is provided.

### Tier Resolution

The `resolve_model(tier)` method maps a tier to a concrete model name:

- **`str` value** in `model_mapping` -- the string itself is the model name.
- **`ModelTierConfig` value** -- the best healthy endpoint is selected via `_select_endpoint()`, which respects the rotation strategy, health state, and TPM limits.

## ModelTierConfig

Rich model tier configuration with multiple endpoints and fallback. Replaces a simple `str` mapping when rate limiting or failover is needed.

```python
@dataclass(frozen=True)
class ModelTierConfig:
    endpoints: list[str]          # Priority-ordered endpoint names
    fallback_on_429: bool = True  # Auto-switch on rate limit
    rotation_strategy: Literal["PRIORITY", "ROUND_ROBIN"] = "PRIORITY"
    tokens_per_minute: int = 0    # 0 = unlimited
```

| Field | Description |
|-------|-------------|
| `endpoints` | List of model endpoint names, tried in order (PRIORITY) or round-robin. |
| `fallback_on_429` | When `True`, a 429 on the primary endpoint triggers automatic failover to the next healthy endpoint. |
| `rotation_strategy` | `"PRIORITY"` tries endpoints in declared order. `"ROUND_ROBIN"` distributes requests evenly across healthy endpoints. |
| `tokens_per_minute` | Per-endpoint TPM limit for proactive throttling. `0` means unlimited. |

## LLMResponse (frozen dataclass)

```python
@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = []
    usage: dict[str, int] = {}       # prompt_tokens, completion_tokens, total_tokens
    model: str = ""
    finish_reason: str = "stop"
    structured: Any | None = None    # Parsed Pydantic model when structured_output is used
```

## ToolCall (frozen dataclass)

```python
@dataclass(frozen=True)
class ToolCall:
    id: str
    function_name: str
    arguments: str   # JSON string
```

## Endpoint Health Tracking

Each endpoint has an `EndpointHealth` instance that tracks runtime state:

| Field | Description |
|-------|-------------|
| `is_healthy` | Set to `False` after 3 consecutive errors. Healthy endpoints are preferred during selection. |
| `consecutive_errors` | Incremented on failure, reset to 0 on success. |
| `rate_limited_until` | Monotonic timestamp. Endpoint is skipped until this time elapses (60-second cooldown on 429). |
| `tokens_used_this_minute` | Running count of tokens consumed. Resets after 60 seconds. Used for proactive TPM enforcement. |

Key methods:

- `mark_success()` -- resets `consecutive_errors` to 0 and sets `is_healthy = True`.
- `mark_failure(rate_limited=False)` -- increments error counter; marks unhealthy after 3 consecutive failures; sets 60-second rate-limit window when `rate_limited=True`.
- `can_handle_request(estimated_tokens, tpm_limit)` -- returns `False` if the endpoint is rate-limited, unhealthy, or would exceed its TPM budget. Returns `True` when TPM limit is 0 (unlimited).

### Retry and Fallback Flow

1. The client calls the selected endpoint with exponential backoff (up to 3 retries) for 5xx errors.
2. `RateLimitError` (429) is **not** retried with backoff -- instead the endpoint is marked rate-limited and a fallback is attempted immediately.
3. 403 errors trigger a single token refresh (via `client_provider`) before retrying.
4. 4xx errors other than 403 and 429 are raised immediately (not retried).

## Token Budget

`TokenBudget` provides workflow-level token budget tracking.

```python
@dataclass
class TokenBudget:
    max_total_tokens: int = 0   # 0 = unlimited
```

### Core Methods

| Method | Description |
|--------|-------------|
| `track_usage(node_id, prompt_tokens, completion_tokens)` | Records token usage. Raises `TokenBudgetExceededError` if the cumulative total exceeds `max_total_tokens`. |
| `check_budget(estimated_tokens=0)` | Returns `True` if the estimated tokens fit within the remaining budget. Always `True` when unlimited. |

### Properties

| Property | Description |
|----------|-------------|
| `total_used` | Total tokens consumed so far across all nodes. |
| `remaining` | Remaining tokens. Returns `-1` when the budget is unlimited. |

### Per-Node Usage

Token usage is tracked per node via `NodeTokenUsage`:

```python
@dataclass
class NodeTokenUsage:
    node_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
```

Access individual node usage with `get_node_usage(node_id)` or all nodes with `get_all_usage()`.

### Serialization

`to_dict()` returns a plain dictionary with `max_total_tokens`, `total_used`, `remaining`, and per-node breakdowns -- suitable for logging or persistence.

## Usage Examples
```python
from databricks_deep_research import FrameworkLLMClient, ModelTier

llm = FrameworkLLMClient(
    client=AsyncOpenAI(base_url="..."),
    model_mapping={
        "simple": "llama-70b",
        "analytical": "llama-70b",
        "complex": "llama-405b",
    },
)

# Simple completion
response = await llm.complete(
    messages=[{"role": "user", "content": "Classify this query: ..."}],
    tier=ModelTier.simple,
)

# Structured output
from pydantic import BaseModel
class PlanOutput(BaseModel):
    steps: list[str]

plan = await llm.complete(
    messages=[...],
    tier=ModelTier.analytical,
    structured_output=PlanOutput,
)
```

## Custom Tiers

The `model_mapping` parameter accepts any string key — you are not limited to the three built-in `ModelTier` values:

```python
model_mapping={
    "simple": "fast-model",
    "analytical": "balanced-model",
    "complex": "reasoning-model",
    "bulk_analysis": "high-throughput-model",  # custom tier
}
```

Agent nodes reference custom tiers via `model_tier: bulk_analysis` in their YAML config.

## YAML-Level Configuration

Workflow YAML files can define their own model tier mappings in a top-level `models:` section, making the workflow self-contained — no Python `model_mapping` needed. See [Model Configuration](../guides/model-configuration.md) for the full guide.

## See Also
- [Authentication](../getting-started/authentication.md) -- Configuring credentials
- [Architecture](architecture.md) -- Where LLM client fits
- [Agent System](agent-system.md) -- How agents use the LLM client
