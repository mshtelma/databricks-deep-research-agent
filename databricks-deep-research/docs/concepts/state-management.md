# State Management

> The append-only log that powers workflow execution and auditability.

## Overview
WorkflowState is the central mutable object during workflow execution. It uses an append-only log for auditability with O(1) latest-value lookup for performance.

## Design Principles
- Append-only: state.append() adds entries, never overwrites
- Every write is timestamped (ISO 8601)
- Full history preserved for debugging and replay
- O(1) reads via _latest_index hash map
- Thread-safe via asyncio.Lock

## StateEntry (frozen dataclass)
Fields: node_id (str), key (str), value (Any), timestamp (str - ISO 8601)

## WorkflowState
### Constructor Fields
- query: str — the user's research query
- log: list[StateEntry] — append-only log
- pools: dict[str, Any] — named pool instances
- model_overrides: dict[str, str] — per-run model overrides
- enterprise_tools: list[Any] — loaded enterprise tool instances
- user_token: str | None — OBO token for Databricks
- domain_filter: str | None — domain filter for searches
- is_cancelled: bool — cancellation flag
- runtime_store: TypedRuntimeStateStore | None — typed runtime state projection (see [Runtime State](runtime-state.md))

### Key Methods

| Method | Description |
|--------|-------------|
| `append(node_id: str, key: str, value: Any) -> None` | Add entry to log and update the latest-value index |
| `get(key: str) -> Any \| None` | O(1) latest value via `_latest_index` lookup |
| `get_all(key: str) -> list[Any]` | All historical values for a key (oldest first) |
| `get_nested(dot_path: str) -> Any \| None` | Resolve dot-separated paths like `"coordination.complexity"` |
| `extract_output(key: str) -> str \| None` | Smart text extraction from Pydantic models, dicts, or plain strings |
| `to_dict() -> dict[str, Any]` | Serialize for checkpointing (excludes internal bookkeeping) |
| `from_dict(data: dict) -> WorkflowState` | Deserialize and rebuild `_latest_index` by replaying the log |

### Internal Mechanisms
- `_latest_index: dict[str, int]` — maps each key to the index of its most recent entry in the log, enabling O(1) lookup. Updated on every `append()` call.
- `_lock: asyncio.Lock` — protects concurrent writes from parallel nodes. Since multiple agent nodes can execute simultaneously (e.g., in a `parallel` workflow node), the lock ensures the log and index stay consistent.

## Usage Patterns

### Writing State
```python
state.append("planner", "plan", {"steps": [...]})
state.append("researcher", "step_1_findings", "The research found...")
state.append("reflector", "reflection", {"decision": "CONTINUE"})
```

### Reading State
```python
plan = state.get("plan")
all_findings = state.get_all("step_1_findings")  # historical values
complexity = state.get_nested("coordination.complexity")
```

### Nested Key Resolution
`get_nested("coordination.complexity")` resolves dot-separated paths in two stages:

1. The first segment (`"coordination"`) is used as the log key — it calls `self.get("coordination")` to retrieve the latest value.
2. Each subsequent segment (`"complexity"`) is resolved against the current object via attribute access (`getattr`). If attribute access fails, it falls back to item access (`obj["complexity"]`). If both fail, it returns `None`.

This means the method works with Pydantic models, dataclasses, and plain dicts interchangeably.

### Smart Output Extraction
`extract_output(key)` retrieves the latest value for a key and tries to return readable text:

1. If the value is already a `str`, return it directly.
2. Try common text field names in priority order: `report`, `direct_response`, `summary`, `findings`, `observation` — via attribute access (Pydantic/dataclass) and then dict key access.
3. Fall back to `str(value)`.

### Serialization
```python
checkpoint = state.to_dict()
# ... save checkpoint ...
restored = WorkflowState.from_dict(checkpoint)
```

`to_dict()` excludes runtime-only fields (`_latest_index`, `_lock`, `enterprise_tools`, `pools`) since they are not JSON-safe or are ephemeral. `from_dict()` rebuilds `_latest_index` by replaying log entries in order, so the restored state has identical O(1) lookup behavior.

## Pool Integration
Pools live on `state.pools` as a `dict[str, Any]` mapping pool names to pool instances. Agent nodes write research findings to pools via `PoolWriteConfig`, which specifies the target pool name and deduplication rules. Other nodes can then read from pools to access aggregated research across the workflow.

## Cancellation
```python
state.is_cancelled = True  # from external signal
# Executor checks state.is_cancelled between nodes
```

The executor inspects `state.is_cancelled` before entering each node. When set to `True`, the workflow aborts gracefully without executing further nodes.

## RuntimeState (Typed State)

`WorkflowState` also carries an optional `runtime_store` field -- a `TypedRuntimeStateStore` instance
that maintains a typed `RuntimeState` projection alongside the append-only log. While `WorkflowState`
is the primary read/write interface during execution (used by the executor and agents), `RuntimeState`
provides structured access to execution results organized by capability domain (coordination, planning,
evidence, synthesis, verification).

See [Runtime State](runtime-state.md) for the full concept reference.

## See Also
- [Architecture](architecture.md)
- [Pool System](pool-system.md)
- [Workflow Engine](workflow-engine.md)
- [Runtime State](runtime-state.md)
