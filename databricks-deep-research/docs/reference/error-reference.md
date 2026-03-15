# Error Reference

> Complete exception hierarchy and error handling reference.

## Exception Hierarchy
```
WorkflowError (base)
├── PlanningContractError
├── WorkflowValidationError
├── WorkflowCancelledError
├── TokenBudgetExceededError
├── NodeBudgetExceededError
└── WorkflowExecutionError
```

## WorkflowError
- Base exception for all framework errors
- Fields: message (str)
- When raised: Any unhandled error during workflow execution

## WorkflowValidationError
- Invalid workflow definition
- Fields: message (str), errors (list[str])
- When raised: load_workflow() or validate_workflow() finds issues
- Common causes: missing node IDs, undeclared tools, invalid config

## WorkflowCancelledError
- Workflow was cancelled by user or system
- Fields: message (str)
- When raised: state.is_cancelled is True
- How to cancel: set state.is_cancelled = True from external signal

## TokenBudgetExceededError
- Token budget limit exceeded
- Fields: message (str), used (int), limit (int)
- When raised: cumulative token usage exceeds WorkflowDefinition.token_budget
- Prevention: set appropriate token_budget, use cheaper model tiers

## NodeBudgetExceededError
- Node exceeded its configured wall-clock time budget
- Fields: message (str), node_id (str), budget_seconds (float), elapsed_ms (float)
- When raised: A node's execution time exceeds `budget_seconds` configured on the `WorkflowNode`
- Prevention: increase `budget_seconds` or simplify node logic

## PlanningContractError
- Planning loop cannot satisfy its execution contract
- Fields: message (str), reason (str)
- When raised: Plan-and-execute runtime cannot extract valid plan items or normalize planner output
- Note: Internal -- not part of public API exports

## WorkflowExecutionError
- Workflow failed after emitting partial progress
- Fields: message (str), state (WorkflowState), events (list[StreamEvent]), cause (Exception)
- When raised: Unhandled exception during workflow execution, wrapping partial state for recovery
- Note: Internal -- not part of public API exports

## Handling Errors
```python
from databricks_deep_research import (
    WorkflowError, WorkflowValidationError,
    WorkflowCancelledError, TokenBudgetExceededError,
    NodeBudgetExceededError,
)

try:
    result = await runner.run(query="...")
except WorkflowValidationError as e:
    print(f"Invalid workflow: {e.errors}")
except TokenBudgetExceededError as e:
    print(f"Budget exceeded: {e.used}/{e.limit} tokens")
except NodeBudgetExceededError as e:
    print(f"Node {e.node_id} timed out: {e.elapsed_ms:.0f}ms of {e.budget_seconds}s budget")
except WorkflowCancelledError:
    print("Workflow was cancelled")
except WorkflowError as e:
    print(f"Workflow error: {e}")
```

## Per-Node Error Handling
Link to error-handling.md guide for ErrorConfig (fail/skip/retry).

## See Also
- [Error Handling Guide](../guides/error-handling.md)
- [Events](../concepts/events.md) -- NodeErrorEvent, TokenBudgetExceededEvent, NodeBudgetExceededEvent
