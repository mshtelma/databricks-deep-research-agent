# Error Handling

> Configure per-node error recovery with retry, skip, and fail strategies.

## Overview
Every workflow node can have an error_handling config that controls what happens when the node fails.

## ErrorConfig
```yaml
error_handling:
  on_error: retry    # fail | skip | retry
  max_retries: 2
  retry_delay_seconds: 1.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on_error` | `str` | `"fail"` | Strategy when the node raises an exception: `fail`, `skip`, or `retry`. |
| `max_retries` | `int` | `2` | Maximum retry attempts (only used when `on_error` is `retry`). |
| `retry_delay_seconds` | `float` | `1.0` | Base delay between retries in seconds. Actual delay uses exponential back-off: `retry_delay_seconds * 2^attempt`. |

## Strategies

### fail (default)
Node failure stops the workflow. A `NodeErrorEvent` is emitted and the exception propagates up as a `WorkflowError`.

### skip
Node failure is logged, a `NodeSkippedEvent` is emitted, and execution continues to the next node. Useful for optional enrichment steps.

### retry
Node is retried up to `max_retries` times with exponential back-off (`retry_delay_seconds * 2^attempt`) between attempts. A `NodeErrorEvent` with `will_retry=True` is emitted before each retry. If all retries are exhausted, a final `NodeErrorEvent` is emitted and the exception propagates (same as `fail`).

## Per-Node Configuration
```yaml
root:
  id: pipeline
  type: sequence
  children:
    - id: background
      type: agent
      config:
        subtype: background
      error_handling:
        on_error: skip         # Background is optional
    - id: research
      type: agent
      config:
        subtype: researcher
      error_handling:
        on_error: retry
        max_retries: 2
        retry_delay_seconds: 1.0
    - id: synthesize
      type: agent
      config:
        subtype: synthesizer   # No error_handling = fail on error
```

## Exception Hierarchy
Based on actual errors.py:
- `WorkflowError` (base)
  - `WorkflowValidationError` -- invalid workflow definition (carries a list of error strings)
  - `WorkflowCancelledError` -- user/system cancellation
  - `TokenBudgetExceededError` -- token budget exceeded (carries `used` and `limit`)

## Token Budget Exceeded
- Set via `WorkflowDefinition.token_budget` (0 means unlimited)
- `TokenBudgetExceededError` raised when cumulative usage exceeds budget
- `TokenBudgetExceededEvent` emitted before the error
- Useful for cost control

## Cancellation
- Set `state.is_cancelled = True` from an external signal
- Executor checks between nodes (at the top of `_exec_node`)
- `WorkflowCancelledError` raised, caught at the top-level `execute()` method
- Cancellation always takes priority -- it is re-raised before any error handling

## Events
- `NodeErrorEvent` -- emitted when a node encounters an error (includes `error_message`, `will_retry`, and `retry_attempt` fields)
- `NodeSkippedEvent` -- emitted when a node is skipped (`on_error: skip`), carries a `reason` string

## Best Practices
- Use `skip` for optional enrichment (background, additional sources)
- Use `retry` for transient failures (API rate limits, network errors)
- Use `fail` for critical steps (synthesis, final output)
- Set `token_budget` to prevent runaway costs
- Handle cancellation in long-running tools

## See Also
- [Workflow Engine](../concepts/workflow-engine.md)
- [Error Reference](../reference/error-reference.md)
- [Events](../concepts/events.md)
