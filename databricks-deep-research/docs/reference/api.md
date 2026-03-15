# Public API Reference

> All exports from `databricks_deep_research`.

## Overview

The framework exports its public API from the top-level package. Import everything from `databricks_deep_research`:

```python
from databricks_deep_research import WorkflowRunner, load_workflow, FrameworkLLMClient
```

The canonical list of public symbols is defined in `__all__` inside `databricks_deep_research/__init__.py`. Only those symbols are part of the stable API.

---

## Workflow Definition

### `WorkflowDefinition` -- class

Top-level, serialisable description of a complete workflow. Contains the root node tree, tool declarations, pool configs, required inputs, output keys, token budget, and timeout. Typically created via `load_workflow()` / `from_yaml()` or built programmatically.

**Key fields:** `id`, `name`, `description`, `version`, `root` (WorkflowNode), `tools` (list[ToolDeclaration]), `pools`, `sources`, `required_inputs`, `output_keys`, `token_budget`, `timeout_seconds`.

**Convenience methods:** `WorkflowDefinition.from_yaml(path)`, `definition.to_yaml(path)` (wired at import time by `workflow.loader`).

See [Architecture](../concepts/architecture.md).

### `WorkflowNode` -- class

A single node in the workflow tree. Recursive structure: composite nodes (`sequence`, `parallel`, `loop`, `conditional`, `plan_and_execute`) carry `children`, while leaf nodes (`agent`, `tool`, `subworkflow`) must leave `children` empty.

**Key fields:** `id`, `type` (NodeType), `label`, `config` (dict), `children` (list[WorkflowNode]), `error_handling` (ErrorConfig | None).

See [Architecture](../concepts/architecture.md).

### `NodeType` -- enum (StrEnum)

Discriminator for the eight supported workflow node kinds.

| Value | Kind |
|-------|------|
| `agent` | LLM agent node (leaf) |
| `tool` | Direct tool invocation (leaf) |
| `sequence` | Execute children in order (composite) |
| `parallel` | Execute children concurrently (composite) |
| `loop` | Repeat children until condition or max iterations (composite) |
| `conditional` | Select a branch based on state (composite) |
| `subworkflow` | Embed another workflow (leaf) |
| `plan_and_execute` | Dynamic plan with item iteration and replan cycles (composite) |

See [Architecture](../concepts/architecture.md).

---

## Workflow State

### `WorkflowState` -- class (dataclass)

Mutable state object shared across all nodes in a workflow run. Backed by an append-only log of `StateEntry` records with an internal index for O(1) latest-value lookup. Thread-safe via `asyncio.Lock`.

**Key fields:** `query`, `log`, `pools`, `model_overrides`, `enterprise_tools`, `user_token`, `domain_filter`, `is_cancelled`.

**Key methods:**

| Method | Description |
|--------|-------------|
| `append(node_id, key, value)` | Append an entry to the log |
| `get(key)` | Get latest value for key (O(1)) |
| `get_all(key)` | Get all values ever appended under key |
| `get_nested(dot_path)` | Resolve dot-separated path (e.g. `"coordination.complexity"`) |
| `extract_output(key)` | Extract readable text from output key |
| `to_dict()` / `from_dict(data)` | Serialise/deserialise for checkpointing |

See [Architecture](../concepts/architecture.md).

---

## Runtime State

### `RuntimeState` -- class (BaseModel)

Typed, structured view of workflow execution organized by capability domain. Populated during execution via `TypedRuntimeStateStore` and available on `WorkflowRunResult.runtime_state`.

**Key fields:** `request` (RequestState), `workflow` (WorkflowLifecycleState), `nodes` (dict[str, NodeExecutionState]), `artifacts` (dict[str, ArtifactEnvelope]), `diagnostics` (RuntimeDiagnostics), `metrics` (RuntimeMetrics), `capabilities` (CapabilityStates).

See [Runtime State Concept](../concepts/runtime-state.md).

### `WorkflowRunRequest` -- class (dataclass)

Structured input for workflow execution via `WorkflowRunner`.

**Fields:** `definition` (WorkflowDefinition), `query` (str), `inputs` (dict), `enterprise_tools` (list[ResearchTool] | None), `tool_registry` (ToolRegistry | None), `tool_factories` (list[ToolFactory] | None), `factory_context` (ToolFactoryContext | None), `strict_tool_resolution` (bool).

### `WorkflowRunResult` -- class (dataclass)

Structured output from workflow execution.

**Key fields:** `runtime_state` (RuntimeState), `events` (list[StreamEvent]).

**Properties:** `artifacts` (dict -- all artifacts), `output` (str -- synthesis report text), `sources` (list[dict] -- evidence sources).

---

## Execution Context

### `ExecutionContext` -- class (dataclass)

Immutable bag of cross-cutting concerns available to every node during execution. Carries the LLM client, checkpoint handler, model overrides, user token, enterprise tools, and tracing flag.

**Key fields:** `llm_client`, `checkpoint_handler`, `model_overrides`, `user_token`, `enterprise_tools`, `trace_enabled`, `tool_call_cache`.

See [Architecture](../concepts/architecture.md).

---

## Workflow Execution

### `WorkflowExecutor` -- class

Core executor that walks the workflow tree depth-first, yielding `StreamEvent` objects as an async generator. Handles all eight node types, tool resolution, pool initialization, token budget tracking, and error handling.

```python
executor = WorkflowExecutor(definition, llm_client, factory_context=ctx)
async for event in executor.execute(state):
    print(event.event_type)
```

See [Architecture](../concepts/architecture.md).

### `run_workflow()` -- async function

Convenience function that runs a workflow to completion and returns `(WorkflowState, list[StreamEvent])`. Collects all events; for streaming, use `WorkflowExecutor.execute()` directly.

```python
state, events = await run_workflow(definition, llm_client, initial_state={"query": "..."})
```

### `run_workflow_from_yaml()` -- async function

Loads a YAML workflow file and runs it to completion. Combines `load_workflow()` + `run_workflow()`.

```python
state, events = await run_workflow_from_yaml("workflow.yaml", llm_client)
```

---

## High-Level Runner

### `WorkflowRunner` -- class

High-level convenience API that wraps client creation, tool factory setup, workflow loading, and execution into a clean interface. Designed for scripts, notebooks, and examples.

```python
runner = WorkflowRunner.from_databricks()
result = await runner.run("research.yaml", query="What is AI?")
print(result.output)
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `from_databricks(model=, model_mapping=)` | Factory with Databricks auth |
| `run(workflow, query=, state=)` | Run to completion, return `WorkflowResult` |
| `stream(workflow, query=, state=)` | Async generator of `StreamEvent` |
| `last_result` | Result from most recent `run()` or `stream()` |
| `aclose()` | Close the underlying LLM client |

See [Quick Start](../getting-started/quickstart.md).

### `WorkflowResult` -- class (dataclass)

Result of a completed workflow run. Wraps the final `WorkflowState`, collected events, and the workflow definition.

**Key properties:**

| Property | Description |
|----------|-------------|
| `state` | The final `WorkflowState` |
| `events` | List of `StreamEvent` collected during execution |
| `definition` | The `WorkflowDefinition` that was run |
| `output` | Primary text output (first non-empty output key) |
| `sources` | Sources from the pool |

See [Quick Start](../getting-started/quickstart.md).

---

## Loaders

### `load_workflow(path)` -- function

Load a `WorkflowDefinition` from a YAML file. Parses nodes recursively, validates structure, and wires tool/source declarations.

**Raises:** `FileNotFoundError`, `WorkflowValidationError`.

### `load_workflow_from_string(yaml_content)` -- function

Parse a `WorkflowDefinition` from a YAML string. Same validation as `load_workflow()`.

**Raises:** `WorkflowValidationError`.

### `save_workflow(definition, path)` -- function

Serialise a `WorkflowDefinition` to a YAML file. Parent directories must already exist.

See [Architecture](../concepts/architecture.md).

---

## LLM Client

### `FrameworkLLMClient` -- class

Thin wrapper around `AsyncOpenAI` with tiered model routing, structured output support, rate-limit-aware endpoint selection, and automatic failover.

**Key methods:**

| Method | Description |
|--------|-------------|
| `from_databricks(model=, model_mapping=)` | Factory with Databricks auth (direct token or SDK auto-detect) |
| `complete(messages, tier=, ...)` | Send messages to LLM, return `LLMResponse` |
| `stream(messages, tier=, ...)` | Stream response tokens and tool calls |
| `resolve_model(tier)` | Resolve a `ModelTier` to a concrete endpoint name |
| `embed(texts, model=)` | Batch embed texts via OpenAI embeddings API |
| `embed_single(text)` | Embed a single text |
| `aclose()` / `close()` | Close the underlying client |

See [Architecture](../concepts/architecture.md).

### `LLMResponse` -- class (frozen dataclass)

Response from an LLM call. Contains the response content, any tool calls, token usage, model name, finish reason, and optional structured (parsed Pydantic) output.

**Key fields:** `content` (str), `tool_calls` (list[ToolCall]), `usage` (dict), `model` (str), `finish_reason` (str), `structured` (Any | None).

### `ModelTier` -- enum (StrEnum)

Model tier for routing to appropriate endpoints.

| Value | Description |
|-------|-------------|
| `simple` | Fast, lightweight tasks |
| `analytical` | Balanced reasoning (default) |
| `complex` | Deep reasoning tasks |

### `ToolCall` -- class (frozen dataclass)

A tool call requested by the LLM.

**Fields:** `id` (str), `function_name` (str), `arguments` (str -- JSON).

---

## Tools

### `ResearchTool` -- protocol

Protocol that all tools (builtin, enterprise, custom) must implement. Uses constructor dependency injection for tool dependencies.

**Required members:**

| Member | Description |
|--------|-------------|
| `definition` (property) | Returns `ToolDefinition` with name, description, and JSON Schema |
| `validate_arguments(arguments)` | Validate and transform raw LLM arguments |
| `execute(arguments, context)` | Async execution returning `ToolResult` |

See [Architecture](../concepts/architecture.md).

### `ToolContext` -- class (frozen dataclass)

Per-call context passed to tools at execution time. Only per-call values that change between invocations belong here; tool dependencies are constructor-injected.

**Key fields:** `query`, `url_registry`, `current_step`, `background_summary`, `recent_observations`, `discovered_sources`.

### `ToolDefinition` -- class (frozen dataclass)

Tool definition combining identity and JSON Schema for LLM function calling.

**Fields:** `name` (str), `description` (str), `parameters` (dict -- JSON Schema), `source_type` (str), `source_kind` (str), `metadata` (dict).

### `ToolResult` -- class (frozen dataclass)

Result returned by a tool execution.

**Fields:** `content` (str), `success` (bool), `sources` (list[SourceInfo]), `data` (dict), `error` (str | None).

### `ToolFactoryContext` -- class (dataclass)

Dependencies available to tool factories at creation time. Provides auto-detected defaults for workspace client, search client, and user token.

**Key fields:** `workspace_client`, `user_token`, `search_client`, `crawler`, `file_index`, `extras`.

**Factory:** `ToolFactoryContext.from_defaults(workspace_client=, user_token=, brave_api_key=, extras=)`.

---

## Events

### `StreamEvent` -- class (Pydantic BaseModel)

Base class for all workflow execution events. Every event carries an `event_type` literal discriminator, `node_id`, and ISO 8601 `timestamp`.

### `FrameworkEvent` -- type alias

Annotated discriminated union of all concrete event types. Use for type-safe deserialization and pattern matching.

**Concrete event types included in the union:**

| Category | Events |
|----------|--------|
| Node lifecycle | `NodeStartedEvent`, `NodeCompletedEvent`, `NodeErrorEvent`, `NodeSkippedEvent`, `NodeBudgetExceededEvent` |
| Loop | `LoopIterationEvent`, `LoopExitEvent` |
| Conditional | `BranchSelectedEvent` |
| Agent | `AgentOutputEvent`, `AgentStreamChunkEvent` |
| Domain-specific | `PlanCreatedEvent`, `ReflectionDecisionEvent`, `CoordinatorClassifiedEvent`, `BackgroundCompletedEvent`, `SynthesisStartedEvent` |
| Plan-and-execute | `ItemStartedEvent`, `ItemCompletedEvent`, `ItemsExtractedEvent`, `EvaluationDecisionEvent`, `ReplanTriggeredEvent`, `PlanAndExecuteExitEvent` |
| Tool | `ToolCallEvent`, `ToolResultEvent`, `ToolCacheHitEvent` |
| Checkpoint | `CheckpointSavedEvent`, `CheckpointResumedEvent` |
| Token budget | `TokenUsageEvent`, `TokenBudgetExceededEvent` |
| Conversation | `ConversationCompactedEvent` |
| Workflow-level | `WorkflowStartedEvent`, `WorkflowCompletedEvent` |
| Verification | `ClaimGeneratedEvent`, `ClaimVerifiedEvent`, `CitationCorrectedEvent`, `NumericClaimDetectedEvent`, `VerificationSummaryEvent` |

See [Architecture](../concepts/architecture.md).

---

## Tracing

### `trace_span(name, span_type=, attributes=)` -- async context manager

Create an MLflow span if `mlflow` is installed, otherwise no-op. Async-safe: handles `asyncio.gather()` context issues gracefully.

```python
async with trace_span("my_step", span_type="CHAIN", attributes={"key": "val"}):
    ...
```

---

## Errors

### `WorkflowError` -- class (Exception)

Base exception for all framework errors.

### `WorkflowValidationError` -- class (WorkflowError)

Raised when a workflow definition fails load-time validation. Carries an `errors: list[str]` field with specific validation messages.

### `WorkflowCancelledError` -- class (WorkflowError)

Raised when a workflow is cancelled mid-execution (via `state.is_cancelled`).

### `TokenBudgetExceededError` -- class (WorkflowError)

Raised when the token budget is exhausted. Carries `used: int` and `limit: int` fields.

### `NodeBudgetExceededError` -- class (WorkflowError)

Raised when a node exceeds its configured wall-clock budget (`budget_seconds`). Carries `node_id: str`, `budget_seconds: float`, and `elapsed_ms: float` fields.

---

## Configuration Helpers

### `parse_model_config(raw)` -- function

Parse a raw dict (from YAML or Python) into `dict[str, str | ModelTierConfig]` suitable for `FrameworkLLMClient(model_mapping=...)`. Accepts both simple strings (`{"simple": "model-name"}`) and rich endpoint dicts with `endpoints`, `fallback_on_429`, and `rotation_strategy` (PRIORITY or ROUND_ROBIN). Raises `ValueError` for invalid configs.

---

## Import Examples

```python
# Common imports
from databricks_deep_research import (
    WorkflowRunner,
    load_workflow,
    FrameworkLLMClient,
)

# Full workflow setup
from databricks_deep_research import (
    WorkflowDefinition, WorkflowNode, NodeType,
    WorkflowExecutor, WorkflowState, ExecutionContext,
    FrameworkLLMClient, ModelTier,
    load_workflow, save_workflow,
)

# Runtime state and structured results
from databricks_deep_research import (
    RuntimeState, WorkflowRunRequest, WorkflowRunResult,
)

# Streaming with events
from databricks_deep_research import (
    WorkflowRunner, WorkflowResult,
    StreamEvent, FrameworkEvent,
)

# Tool development
from databricks_deep_research import (
    ResearchTool, ToolContext, ToolDefinition, ToolResult,
    ToolFactoryContext,
)

# Error handling
from databricks_deep_research import (
    WorkflowError,
    WorkflowValidationError,
    WorkflowCancelledError,
    TokenBudgetExceededError,
    NodeBudgetExceededError,
)

# Configuration helpers
from databricks_deep_research import parse_model_config
```

---

## See Also

- [Quick Start](../getting-started/quickstart.md)
- [Architecture](../concepts/architecture.md)
- [Runtime State](../concepts/runtime-state.md)
