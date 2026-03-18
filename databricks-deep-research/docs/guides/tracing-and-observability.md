# Tracing and Observability

> Integrate with MLflow for distributed tracing of workflow execution.

## Overview
The framework provides `trace_span` integration with MLflow for observability. Each node, LLM call, and tool execution can be traced. When MLflow is not installed, tracing is a no-op with zero overhead.

## Setup
```bash
pip install "databricks-deep-research[tracing]"
# or
pip install mlflow>=2.10
```

```python
import mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/my/experiment")
```

## trace_span

`trace_span` is an **async context manager** that wraps `mlflow.start_span`. It creates an MLflow span when MLflow is installed and silently does nothing when it is not.

### Signature

```python
from databricks_deep_research import trace_span

async with trace_span(
    name: str,                                  # Span name shown in MLflow UI
    span_type: str = "CHAIN",                   # Maps to mlflow.entities.SpanType
    attributes: dict[str, Any] | None = None,   # Arbitrary key/value metadata
) -> AsyncGenerator[Any, None]:
    ...
```

### How it creates MLflow spans

1. **Guard check** -- if MLflow is not installed (`_HAS_MLFLOW is False`), the context manager yields `None` immediately and returns. No tracing code executes, so there is zero runtime cost.
2. **Span type resolution** -- the string `span_type` (e.g., `"CHAIN"`, `"LLM"`, `"TOOL"`) is resolved to the matching `mlflow.entities.SpanType` enum member via `getattr(SpanType, span_type, SpanType.CHAIN)`. Unrecognised types fall back to `CHAIN`.
3. **Span creation** -- `mlflow.start_span(name=name, span_type=mlflow_type)` returns a context manager. The implementation calls `__enter__()` manually (rather than using `async with`) so it can handle the `asyncio.gather()` edge case described below.
4. **Attributes** -- if an `attributes` dict is provided and the span was created successfully, `span.set_attributes(attributes)` attaches the metadata to the span.
5. **Cleanup** -- on exit, `__exit__(None, None, None)` is called manually. A `ValueError` ("Token was created in a different Context") is caught and suppressed because it is benign when spans are used inside `asyncio.gather()`.

### Asyncio safety

Concurrent tasks launched with `asyncio.gather()` may execute across different contextvars contexts. MLflow span tokens are context-local, so closing a span from a different context raises `ValueError`. `trace_span` catches this error during cleanup, making it safe to use in parallel agent/tool execution without extra handling.

### Span attributes

Common attributes attached by the framework:

| Attribute | Example | Set by |
|-----------|---------|--------|
| `node_id` | `"planner"` | Workflow executor |
| `tier` | `"analytical"` | LLM client |
| `tool_name` | `"web_search"` | Tool runner |
| `model` | `"databricks-llama-70b"` | LLM client |
| `prompt_tokens` | `1200` | LLM client |
| `completion_tokens` | `450` | LLM client |
| `total_tokens` | `1650` | LLM client |
| `step_index` | `3` | Researcher node |
| `success` | `true` | Tool runner |

### Nested spans for hierarchical tracing

MLflow automatically nests spans opened within an active parent span. The framework leverages this to produce a trace tree:

```
workflow_execution            (root, CHAIN)
  +-- planner                 (CHAIN)
  |     +-- llm_call          (LLM)
  +-- researcher_step_1       (CHAIN)
  |     +-- web_search        (TOOL)
  |     +-- web_crawl         (TOOL)
  |     +-- llm_call          (LLM)
  +-- reflector               (CHAIN)
  |     +-- llm_call          (LLM)
  +-- synthesizer             (CHAIN)
        +-- llm_call          (LLM)
```

## Automatic Tracing

The framework automatically traces:

- **Each workflow execution** -- a root span wrapping the entire research pipeline.
- **Each node execution** -- child spans for planner, researcher, reflector, synthesizer, and other nodes.
- **Each LLM call** -- spans with `span_type="LLM"` recording the model endpoint, tier, and token counts.
- **Each tool call** -- spans with `span_type="TOOL"` recording the tool name, duration, and success/failure.

No configuration is needed beyond having MLflow installed and an experiment set. Remove MLflow from the environment to disable tracing entirely.

## Custom Tracing

You can create your own spans anywhere in application code:

```python
from databricks_deep_research.tracing import trace_span

async def my_custom_step(query: str) -> str:
    async with trace_span("my_operation", attributes={"query": query}):
        result = await do_something(query)
        return result
```

Use `span_type` to categorise custom spans:

```python
async with trace_span("embedding_lookup", span_type="RETRIEVER", attributes={"k": 10}):
    results = await vector_store.search(embedding, k=10)
```

The yielded `span` object is the raw MLflow span (or `None` when MLflow is absent). You can set additional attributes on it mid-execution:

```python
async with trace_span("research", span_type="CHAIN") as span:
    results = await search(query)
    if span is not None:
        span.set_attributes({"result_count": len(results)})
```

## Viewing Traces

- **MLflow UI**: navigate to your experiment, then open the **Traces** tab to see the span tree for each run.
- **Databricks**: MLflow is integrated into the workspace. Open the experiment from the sidebar and select a trace to inspect span durations, attributes, and parent-child relationships.

## Token Usage Analysis

Token counts are recorded as span attributes on LLM spans (`prompt_tokens`, `completion_tokens`, `total_tokens`). To analyse token usage across a research session:

1. Open the trace in the MLflow UI.
2. Filter spans by `span_type = LLM`.
3. Sum `total_tokens` across all LLM spans to get the session total.

For programmatic access:

```python
import mlflow

client = mlflow.tracking.MlflowClient()
# Retrieve traces for your experiment and inspect span attributes
```

## See Also

- [Architecture](../concepts/architecture.md)
- [LLM Client](../concepts/llm-client.md)
