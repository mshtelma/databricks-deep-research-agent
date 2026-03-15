# Runtime State

> Typed workflow execution state for structured observability and result extraction.

## Overview

`RuntimeState` provides a structured, typed view of workflow execution. It complements `WorkflowState` (the append-only log used by the executor and agents) with a Pydantic model organized by capability domain -- coordination, planning, evidence, synthesis, verification, and more.

`RuntimeState` is a public export:

```python
from databricks_deep_research import RuntimeState, WorkflowRunResult
```

## Relationship to WorkflowState

| Aspect | WorkflowState | RuntimeState |
|--------|--------------|--------------|
| **Structure** | Append-only log of `StateEntry` records | Typed Pydantic model organized by capability |
| **Access pattern** | O(1) latest-value lookup by key | Structured field access (e.g. `state.capabilities.planning`) |
| **Primary users** | Executor, agent harness, conditions | Results extraction, observability, downstream consumers |
| **Mutation model** | `append()` — never overwrite | Store methods — `set_coordination()`, `ingest_evidence()`, etc. |

**Bridge:** `WorkflowState.runtime_store` holds the `TypedRuntimeStateStore` instance that maintains the `RuntimeState` projection. Both are populated during execution; `RuntimeState` is the preferred way to access structured results after a workflow completes.

## RuntimeState Model

Top-level fields of the `RuntimeState` Pydantic model:

| Field | Type | Description |
|-------|------|-------------|
| `request` | `RequestState` | Query, inputs, and request ID |
| `workflow` | `WorkflowLifecycleState` | Status, timing, token/source/step totals, errors |
| `nodes` | `dict[str, NodeExecutionState]` | Per-node execution tracking (status, duration, artifacts) |
| `artifacts` | `dict[str, ArtifactEnvelope]` | Typed artifacts with quality and provenance metadata |
| `diagnostics` | `RuntimeDiagnostics` | Parse failures, blocked reasons, fallback activations, policy decisions |
| `metrics` | `RuntimeMetrics` | Source, observation, and cache counters |
| `capabilities` | `CapabilityStates` | Domain-specific state projections (see below) |

### RequestState

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | `""` | The user's research query |
| `inputs` | `dict[str, Any]` | `{}` | Additional workflow inputs |
| `request_id` | `str` | `""` | Unique request identifier |

### WorkflowLifecycleState

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `workflow_id` | `str` | `""` | Unique workflow run identifier |
| `workflow_name` | `str` | `""` | Human-readable workflow name |
| `terminal_status` | `Literal["running", "completed", "failed", "cancelled"]` | `"running"` | Current execution status |
| `start_time` | `str` | *(now)* | ISO 8601 start timestamp |
| `duration_ms` | `float` | `0.0` | Wall-clock execution time |
| `error_type` | `str \| None` | `None` | Error class name on failure |
| `error_message` | `str \| None` | `None` | Error message on failure |
| `total_tokens` | `int` | `0` | Aggregate token usage |
| `total_sources` | `int` | `0` | Total sources consumed |
| `total_steps_executed` | `int` | `0` | Plan steps executed |
| `blocked_steps` | `int` | `0` | Plan steps blocked |
| `missing_declared_tools` | `int` | `0` | Tools declared but not resolved |
| `plan_exit_reasons` | `list[str]` | `[]` | Reasons plan-and-execute nodes exited |

### NodeExecutionState

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_id` | `str` | *required* | Node identifier |
| `node_type` | `str` | `""` | Node type (agent, tool, etc.) |
| `label` | `str` | `""` | Human-readable label |
| `status` | `Literal["pending", "running", "completed", "failed", "skipped", "blocked"]` | `"pending"` | Execution status |
| `duration_ms` | `float` | `0.0` | Execution time |
| `output_key` | `str \| None` | `None` | State key where output was written |
| `output_preview` | `str` | `""` | Truncated output preview |
| `input_artifact_refs` | `list[str]` | `[]` | Input artifact IDs |
| `output_artifact_refs` | `list[str]` | `[]` | Output artifact IDs |
| `diagnostic_refs` | `list[str]` | `[]` | Diagnostic record IDs |
| `metrics` | `NodeMetrics` | *(default)* | Per-node metrics (artifacts_published, diagnostics_recorded) |

## Capability States

The `capabilities` field holds domain-specific projections, each initialized lazily (default `None`).

### CoordinationState

Populated by the coordinator agent after query classification.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `complexity` | `str` | `""` | Classified complexity level |
| `is_simple` | `bool` | `False` | Whether query is simple enough for direct response |
| `recommended_depth` | `str` | `"standard"` | Recommended research depth |
| `direct_response` | `str \| None` | `None` | Direct answer for simple queries |
| `follow_up_type` | `str \| None` | `None` | Follow-up type for conversation queries |

### BackgroundState

Populated by the background investigator after initial research.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `summary` | `str` | `""` | Background investigation summary |
| `data_landscape` | `dict[str, Any]` | `{}` | Structured data landscape information |
| `query_decomposition` | `list[str]` | `[]` | Sub-queries derived from original query |
| `discovered_sources` | `list[Any]` | `[]` | Sources found during background research |

### PlanningState

Tracks planning cycles, step completion, and blocking.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `current_cycle` | `int` | `0` | Current plan cycle number |
| `current_plan_title` | `str` | `""` | Title of current plan |
| `current_plan_thought` | `str` | `""` | Planner's reasoning |
| `has_enough_context` | `bool` | `False` | Whether enough context exists |
| `cycles` | `list[PlanCycleRecord]` | `[]` | History of all plan cycles |
| `completed_step_ids` | `list[str]` | `[]` | IDs of completed steps |
| `blocked_step_ids` | `list[str]` | `[]` | IDs of blocked steps |

### RetrievalState

Tracks tool retrieval requests and results.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `requests` | `list[RetrievalRequestRecord]` | `[]` | Retrieval requests issued |
| `results` | `list[RetrievalResultRecord]` | `[]` | Retrieval results received |
| `cache_keys_seen` | `list[str]` | `[]` | Cache keys encountered |
| `tool_usage` | `dict[str, int]` | `{}` | Per-tool call counts |

### EvidenceState

Tracks sources and observations with deduplication.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sources` | `list[SourceRecord]` | `[]` | All source records |
| `observations` | `list[ObservationRecord]` | `[]` | All observation records |
| `source_urls_seen` | `list[str]` | `[]` | URLs seen for dedup |
| `observation_hashes_seen` | `list[str]` | `[]` | Observation hashes for dedup |
| `last_delta` | `EvidenceDelta` | *(default)* | Counts from last ingestion (new/duplicate sources and observations) |

### SynthesisState

Tracks synthesis mode and output artifacts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `Literal["full", "partial", "insufficient", "transform"]` | `"full"` | Synthesis mode |
| `input_pack` | `SynthesisInputPack` | *(default)* | Summary of inputs (observation/source counts, previews) |
| `report_artifact_id` | `str \| None` | `None` | ID of the published report artifact |
| `verification_artifact_ids` | `list[str]` | `[]` | IDs of verification artifacts |

### VerificationState

Tracks claim verification results.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `claims` | `list[dict[str, Any]]` | `[]` | Raw claim data |
| `verification_details` | `dict[str, Any]` | `{}` | Detailed verification results |
| `summary` | `VerificationSummaryRecord` | *(default)* | Aggregate verification statistics |
| `verification_artifact_ids` | `list[str]` | `[]` | Artifact IDs from verification |

## Artifacts

Artifacts are typed, versioned outputs with quality and provenance metadata.

### ArtifactEnvelope

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `artifact_id` | `str` | *required* | Unique artifact identifier |
| `artifact_type` | `str` | *required* | Type discriminator (e.g. `"report"`, `"verification"`) |
| `producer_node_id` | `str` | *required* | ID of the node that produced this artifact |
| `created_at` | `str` | *(now)* | ISO 8601 creation timestamp |
| `schema_version` | `str` | `"1"` | Schema version for forward compatibility |
| `payload` | `Any` | `None` | The artifact content |
| `quality` | `ArtifactQuality` | *(default)* | Quality assessment |
| `provenance` | `ArtifactProvenance` | *(default)* | Lineage tracking |
| `tags` | `dict[str, str]` | `{}` | Free-form metadata tags |

### ArtifactQuality

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | `Literal["success", "blocked", "degraded", "malformed", "failed", "informational"]` | `"informational"` | Quality status |
| `confidence` | `float \| None` | `None` | Confidence score |
| `substantive` | `bool` | `True` | Whether the artifact contains meaningful content |
| `quality_flags` | `list[str]` | `[]` | Additional quality indicators |

### ArtifactProvenance

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_node_ids` | `list[str]` | `[]` | Nodes that contributed to this artifact |
| `tool_refs` | `list[str]` | `[]` | Tools used to produce this artifact |
| `source_refs` | `list[str]` | `[]` | Source IDs referenced |
| `upstream_artifact_refs` | `list[str]` | `[]` | Artifact IDs this depends on |

## TypedRuntimeStateStore

The store manages the `RuntimeState` instance and provides methods for structured updates.

### Constructor

```python
store = TypedRuntimeStateStore(query="What is AI?", workflow_id="wf-123", workflow_name="research")
```

### Key Methods

| Method | Description |
|--------|-------------|
| `snapshot()` | Deep copy of current `RuntimeState` |
| `runtime()` | Direct reference to current `RuntimeState` (mutable) |
| `publish_artifact(...)` | Create an `ArtifactEnvelope` and store it |
| `set_artifact(key, value)` | Legacy accessor (wraps `publish_artifact`) |
| `get_artifact(key)` | Retrieve artifact payload by ID |
| `start_node(node_id, node_type, label)` | Record node execution start |
| `complete_node(node_id, duration_ms, ...)` | Record node completion |
| `fail_node(node_id, error_message, ...)` | Record node failure |
| `block_node(node_id, reason)` | Record node blocked |
| `set_workflow_completed(duration_ms)` | Mark workflow as completed |
| `set_workflow_failed(error_type, error_message)` | Mark workflow as failed |
| `set_workflow_cancelled()` | Mark workflow as cancelled |
| `set_coordination(output)` | Store coordinator classification result |
| `set_background(...)` | Store background investigation results |
| `record_diagnostic(category, severity, message, ...)` | Add a diagnostic record |
| `begin_plan_cycle(cycle)` | Start a new planning cycle |
| `finalize_plan_cycle(...)` | Complete a planning cycle with steps and feedback |
| `mark_step_completed(step_id)` | Mark a plan step as completed |
| `mark_step_blocked(step_id, reason)` | Mark a plan step as blocked |
| `ingest_evidence(sources, observations)` | Add sources and observations with dedup |
| `record_retrieval_request(...)` | Track a tool retrieval request |
| `record_retrieval_result(...)` | Track a tool retrieval result |
| `record_retrieval_cache_hit(cache_key)` | Record a retrieval cache hit |
| `build_synthesis_input_pack()` | Build summary of synthesis inputs |
| `set_synthesis_mode(mode)` | Set the synthesis mode |
| `publish_report_artifact(node_id, report)` | Publish the synthesis report |
| `publish_verification_payload(node_id, payload)` | Publish verification results |

## WorkflowRunRequest

Structured input for workflow execution via `WorkflowRunner`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `definition` | `WorkflowDefinition` | *required* | The workflow to execute |
| `query` | `str` | `""` | Research query |
| `inputs` | `dict[str, Any]` | `{}` | Additional workflow inputs |
| `enterprise_tools` | `list[ResearchTool] \| None` | `None` | Pre-built enterprise tool instances |
| `tool_registry` | `ToolRegistry \| None` | `None` | Custom tool registry |
| `tool_factories` | `list[ToolFactory] \| None` | `None` | Additional tool factories |
| `factory_context` | `ToolFactoryContext \| None` | `None` | Context for tool factory creation |
| `strict_tool_resolution` | `bool` | `False` | Fail on unresolvable tool declarations |

## WorkflowRunResult

Structured output from workflow execution.

| Field | Type | Description |
|-------|------|-------------|
| `runtime_state` | `RuntimeState` | The final typed runtime state |
| `events` | `list[StreamEvent]` | All events emitted during execution |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `artifacts` | `dict[str, Any]` | All artifacts from `runtime_state.artifacts` |
| `output` | `str` | Synthesis report text (extracted from the report artifact via `synthesis.report_artifact_id`) |
| `sources` | `list[dict[str, Any]]` | All evidence sources as dicts |

## See Also

- [State Management](state-management.md)
- [API Reference](../reference/api.md)
- [Architecture](architecture.md)
