# Event Types Reference

> Complete reference for all framework event types with fields.

## Overview

All events are Pydantic `BaseModel` instances with an `event_type` literal discriminator. The union type `FrameworkEvent` covers every event the executor can yield. Consuming applications pattern-match on `event_type` to route events to their own transport (SSE, WebSocket, logging, etc.).

Source: `databricks_deep_research/events/types.py`

## Common Fields

Every event inherits from `StreamEvent` and carries three base fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `str` (narrowed to a `Literal` per subclass) | Discriminator tag identifying the event kind. |
| `node_id` | `str` | ID of the workflow node that emitted the event. |
| `timestamp` | `str` | ISO 8601 timestamp of when the event was created. |

---

## Event Catalog

### Workflow Lifecycle

#### `WorkflowStartedEvent`

**event_type:** `"workflow_started"`

Emitted when workflow execution begins.

| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | `str` | Unique identifier of the workflow run. |
| `workflow_name` | `str` | Human-readable workflow name. |

#### `WorkflowCompletedEvent`

**event_type:** `"workflow_completed"`

Emitted when workflow execution finishes.

| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | `str` | Unique identifier of the workflow run. |
| `duration_ms` | `float` | Wall-clock execution time in milliseconds. |
| `total_tokens` | `int` | Aggregate token usage across all LLM calls. |
| `final_report` | `str` | Final synthesized report text. Defaults to `""`. |
| `structured_output` | `Any \| None` | Optional structured output payload. Defaults to `None`. |
| `total_sources` | `int` | Number of sources consumed. Defaults to `0`. |
| `total_steps_executed` | `int` | Number of plan steps that ran. Defaults to `0`. |

---

### Node Lifecycle

#### `NodeStartedEvent`

**event_type:** `"node_started"`

Emitted when any node begins executing.

| Field | Type | Description |
|-------|------|-------------|
| `node_type` | `str` | The type of node (e.g. `"agent"`, `"tool"`, `"loop"`). |
| `label` | `str` | Human-readable label for the node. |

#### `NodeCompletedEvent`

**event_type:** `"node_completed"`

Emitted when any node finishes successfully.

| Field | Type | Description |
|-------|------|-------------|
| `duration_ms` | `float` | Wall-clock execution time of the node in milliseconds. |

#### `NodeErrorEvent`

**event_type:** `"node_error"`

Emitted when a node encounters an error.

| Field | Type | Description |
|-------|------|-------------|
| `error_message` | `str` | Description of the error. |
| `will_retry` | `bool` | Whether the executor will retry this node. Defaults to `False`. |
| `retry_attempt` | `int` | Current retry attempt number (0-based). Defaults to `0`. |

#### `NodeSkippedEvent`

**event_type:** `"node_skipped"`

Emitted when a node is skipped due to `error_handling=skip`.

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `str` | Why the node was skipped. |

#### `NodeBudgetExceededEvent`

**event_type:** `"node_budget_exceeded"`

Emitted when a node exceeds its configured wall-clock budget (`budget_seconds`).

| Field | Type | Description |
|-------|------|-------------|
| `budget_seconds` | `float` | The configured time budget in seconds. |
| `elapsed_ms` | `float` | Actual elapsed time in milliseconds. |
| `reason` | `str` | Reason for the budget exceedance. Defaults to `"budget_exceeded"`. |

---

### Agent Output

#### `AgentOutputEvent`

**event_type:** `"agent_output"`

Emitted when an agent produces its final output.

| Field | Type | Description |
|-------|------|-------------|
| `output_key` | `str` | State key where the output is stored. |
| `output_preview` | `str` | Truncated preview of the output value. |

#### `AgentStreamChunkEvent`

**event_type:** `"agent_stream_chunk"`

Emitted for streaming synthesis -- token-by-token output chunks.

| Field | Type | Description |
|-------|------|-------------|
| `chunk` | `str` | The text chunk (typically one or a few tokens). |
| `subtype` | `str` | Optional subtype qualifier (e.g. `"synthesis"`). Defaults to `""`. |

---

### Domain-Specific

#### `CoordinatorClassifiedEvent`

**event_type:** `"coordinator_classified"`

Emitted when the coordinator classifies a query by complexity and routes it to the appropriate research depth.

| Field | Type | Description |
|-------|------|-------------|
| `complexity` | `str` | Classified complexity level of the query. |
| `recommended_depth` | `str` | Recommended research depth (e.g. `"light"`, `"standard"`, `"extended"`). |
| `is_simple` | `bool` | Whether the query is simple enough for a direct response. Defaults to `False`. |
| `direct_response` | `str \| None` | If simple, the direct answer. Defaults to `None`. |
| `follow_up_type` | `str \| None` | Type of follow-up if the query continues a conversation. Defaults to `None`. |
| `reasoning` | `str` | Explanation of the classification decision. Defaults to `""`. |

#### `PlanCreatedEvent`

**event_type:** `"plan_created"`

Emitted when a planner creates or updates a research plan.

| Field | Type | Description |
|-------|------|-------------|
| `plan_id` | `str` | Unique identifier for this plan. |
| `title` | `str` | Title of the research plan. |
| `thought` | `str` | The planner's reasoning about what to investigate. |
| `steps` | `list[dict[str, Any]]` | Ordered list of plan steps with their configuration. |
| `iteration` | `int` | Plan iteration number (increases on replan). Defaults to `1`. |
| `has_enough_context` | `bool` | Whether the planner believes enough context exists already. Defaults to `False`. |

#### `ReflectionDecisionEvent`

**event_type:** `"reflection_decision"`

Emitted when a reflector makes a CONTINUE/ADJUST/COMPLETE decision after a research step.

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | One of `"continue"`, `"adjust"`, `"replan"`, or `"complete"`. |
| `reasoning` | `str` | Explanation for the decision. |
| `suggested_changes` | `list[str] \| None` | Suggested adjustments when decision is `"adjust"`. Normalized to `[]` if `None`. |

#### `BackgroundCompletedEvent`

**event_type:** `"background_completed"`

Emitted when background investigation completes.

| Field | Type | Description |
|-------|------|-------------|
| `sources_discovered` | `int` | Number of sources found during background research. Defaults to `0`. |
| `data_landscape_summary` | `str` | Summary of the data landscape. Defaults to `""`. |
| `data_landscape` | `dict[str, Any]` | Structured data landscape information. Defaults to `{}`. |
| `query_decomposition` | `list[str]` | Sub-queries derived from the original query. Defaults to `[]`. |

#### `SynthesisStartedEvent`

**event_type:** `"synthesis_started"`

Emitted when synthesis begins.

| Field | Type | Description |
|-------|------|-------------|
| `total_observations` | `int` | Number of research observations available for synthesis. Defaults to `0`. |
| `total_sources` | `int` | Number of sources available for synthesis. Defaults to `0`. |

---

### Plan-and-Execute

Events emitted by `plan_and_execute` workflow nodes, which extract items from a plan and execute them iteratively with optional re-planning.

#### `ItemStartedEvent`

**event_type:** `"item_started"`

Emitted when a plan-and-execute item begins execution.

| Field | Type | Description |
|-------|------|-------------|
| `item_index` | `int` | Zero-based index of the current item. |
| `item_summary` | `str` | Human-readable summary of what the item will do. |
| `total_items` | `int` | Total number of items in the current plan. |

#### `ItemCompletedEvent`

**event_type:** `"item_completed"`

Emitted when a plan-and-execute item finishes.

| Field | Type | Description |
|-------|------|-------------|
| `item_index` | `int` | Zero-based index of the completed item. |
| `items_processed` | `int` | Cumulative count of items processed so far. |

#### `ItemsExtractedEvent`

**event_type:** `"items_extracted"`

Emitted when items are extracted from the plan in a plan-and-execute node.

| Field | Type | Description |
|-------|------|-------------|
| `total_items` | `int` | Number of items extracted. |
| `items_path` | `str` | State path where items are stored. |
| `cycle` | `int` | Current plan-execute-evaluate cycle number. |

#### `EvaluationDecisionEvent`

**event_type:** `"evaluation_decision"`

Emitted when the plan-and-execute evaluator makes a decision about whether to continue, replan, or complete.

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | One of `"continue"`, `"adjust"`, `"replan"`, or `"complete"`. |
| `reasoning` | `str` | Explanation for the evaluation decision. |
| `items_processed` | `int` | Number of items processed at decision time. |

#### `ReplanTriggeredEvent`

**event_type:** `"replan_triggered"`

Emitted when a plan-and-execute node triggers a replan cycle.

| Field | Type | Description |
|-------|------|-------------|
| `cycle` | `int` | Current replan cycle number. |
| `reason` | `str` | Why replanning was triggered. |
| `items_remaining` | `int` | Number of items still unprocessed from previous plan. |

#### `PlanAndExecuteExitEvent`

**event_type:** `"plan_and_execute_exit"`

Emitted when a plan-and-execute node exits.

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `str` | Exit reason (e.g. `"complete"`, `"max_cycles"`). |
| `total_items_processed` | `int` | Total items processed across all cycles. |
| `replan_cycles` | `int` | Number of replan cycles executed. |
| `total_planned` | `int` | Total items that were planned. Defaults to `0`. |

---

### Tool Calls

#### `ToolCallEvent`

**event_type:** `"tool_call"`

Emitted when an agent or tool node calls a tool.

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool being called. |
| `arguments` | `dict[str, Any]` | Arguments passed to the tool. Defaults to `{}`. |

#### `ToolResultEvent`

**event_type:** `"tool_result"`

Emitted when a tool returns its result.

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool that returned. |
| `result_summary` | `str` | Truncated summary of the result. |
| `source_count` | `int` | Number of sources in the result. Defaults to `0`. |
| `raw_source_count` | `int` | Raw source count before filtering. Defaults to `0`. |
| `accepted_source_count` | `int` | Sources that passed quality filters. Defaults to `0`. |
| `rejected_source_count` | `int` | Sources rejected by quality filters. Defaults to `0`. |
| `tool_success` | `bool` | Whether the tool call succeeded. Defaults to `True`. |
| `tool_error` | `str` | Error message if the tool failed. Defaults to `""`. |

#### `ToolCacheHitEvent`

**event_type:** `"tool_cache_hit"`

Emitted when a tool call is skipped due to a dedup cache hit.

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool whose call was deduplicated. |
| `cache_key` | `str` | The cache key that matched. |

---

### Loop Control

#### `LoopIterationEvent`

**event_type:** `"loop_iteration"`

Emitted at the start of each loop iteration.

| Field | Type | Description |
|-------|------|-------------|
| `iteration` | `int` | Current iteration number (1-based). |
| `max_iterations` | `int` | Maximum iterations configured for this loop. |

#### `LoopExitEvent`

**event_type:** `"loop_exit"`

Emitted when a loop terminates.

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `str` | Exit reason: `"condition_met"`, `"max_iterations"`, or `"parse_failure"`. |
| `total_iterations` | `int` | Number of iterations completed before exit. |

---

### Conditional

#### `BranchSelectedEvent`

**event_type:** `"branch_selected"`

Emitted when a conditional node selects a branch.

| Field | Type | Description |
|-------|------|-------------|
| `branch_index` | `int` | Zero-based index of the selected branch. |
| `condition_summary` | `str` | Human-readable description of the condition that matched. |

---

### Token Budget

#### `TokenUsageEvent`

**event_type:** `"token_usage"`

Periodic token usage report emitted after LLM calls.

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Number of prompt tokens used. |
| `completion_tokens` | `int` | Number of completion tokens used. |
| `total_tokens` | `int` | Sum of prompt and completion tokens. |
| `budget_remaining` | `int` | Tokens remaining in budget. `-1` if unlimited. |

#### `TokenBudgetExceededEvent`

**event_type:** `"token_budget_exceeded"`

Emitted when the token budget is exhausted.

| Field | Type | Description |
|-------|------|-------------|
| `used` | `int` | Total tokens consumed. |
| `limit` | `int` | Configured token budget limit. |

---

### Conversation

#### `ConversationCompactedEvent`

**event_type:** `"conversation_compacted"`

Emitted when an agent conversation is compacted to save context window space.

| Field | Type | Description |
|-------|------|-------------|
| `tokens_saved` | `int` | Number of tokens reclaimed by compaction. |

---

### Checkpoint

#### `CheckpointSavedEvent`

**event_type:** `"checkpoint_saved"`

Emitted after workflow state is checkpointed for resumability.

| Field | Type | Description |
|-------|------|-------------|
| `checkpoint_size` | `int` | Size of the serialized checkpoint in bytes. |

#### `CheckpointResumedEvent`

**event_type:** `"checkpoint_resumed"`

Emitted when execution resumes from a previously saved checkpoint.

| Field | Type | Description |
|-------|------|-------------|
| `resumed_from` | `str` | ISO 8601 timestamp of the checkpoint being resumed. |

---

### Citation / Verification

Events emitted by the 7-stage citation verification pipeline.

#### `ClaimGeneratedEvent`

**event_type:** `"claim_generated"`

Emitted when a claim is generated during interleaved synthesis.

| Field | Type | Description |
|-------|------|-------------|
| `claim_text` | `str` | The text of the generated claim. |
| `claim_index` | `int` | Zero-based index of the claim in the output. |
| `citation_keys` | `list[str]` | Citation keys attached to this claim. Defaults to `[]`. |
| `claim_role` | `str` | Role of the claim (e.g. `"fact"`, `"opinion"`). Defaults to `"fact"`. |

#### `ClaimVerifiedEvent`

**event_type:** `"claim_verified"`

Emitted when a claim passes NLI (Natural Language Inference) verification.

| Field | Type | Description |
|-------|------|-------------|
| `claim_index` | `int` | Index of the verified claim. |
| `verdict` | `str` | Verification verdict: `"supported"`, `"not_supported"`, or `"not_enough_info"`. |
| `confidence` | `float` | Overall confidence score for the verdict. |
| `verification_confidence` | `float` | Confidence from the verification model. Defaults to `0.0`. |
| `routing_confidence_level` | `str` | Confidence level used for routing (e.g. `"high"`, `"low"`). Defaults to `""`. |
| `routing_confidence_score` | `float` | Numeric routing confidence score. Defaults to `0.0`. |
| `evidence_match_score` | `float` | How well evidence matches the claim. Defaults to `0.0`. |
| `used_quick_verification` | `bool` | Whether quick (non-NLI) verification was used. Defaults to `False`. |
| `verification_latency_ms` | `float` | Time spent on verification in milliseconds. Defaults to `0.0`. |
| `claim_role` | `str` | Role of the claim. Defaults to `"fact"`. |
| `verification_method` | `str` | Method used for verification. Defaults to `""`. |
| `evidence_snippet` | `str` | Snippet of the supporting evidence. Defaults to `""`. |
| `claim_text` | `str` | Text of the claim that was verified. Defaults to `""`. |

#### `CitationCorrectedEvent`

**event_type:** `"citation_corrected"`

Emitted when a citation is corrected post-verification.

| Field | Type | Description |
|-------|------|-------------|
| `claim_index` | `int` | Index of the claim whose citation was corrected. |
| `action` | `str` | Correction action: `"keep"`, `"replace"`, `"remove"`, or `"add_alternate"`. |
| `original_key` | `str` | The original citation key. Defaults to `""`. |
| `corrected_key` | `str` | The corrected citation key. Defaults to `""`. |

#### `NumericClaimDetectedEvent`

**event_type:** `"numeric_claim_detected"`

Emitted when a numeric claim is detected and queued for QA verification.

| Field | Type | Description |
|-------|------|-------------|
| `claim_index` | `int` | Index of the claim containing the numeric value. |
| `numeric_value` | `str` | The numeric value extracted from the claim. |
| `verification_status` | `str` | Current verification status (e.g. `"pending"`). Defaults to `"pending"`. |

#### `VerificationSummaryEvent`

**event_type:** `"verification_summary"`

Emitted at the end of the verification pipeline with aggregate statistics.

| Field | Type | Description |
|-------|------|-------------|
| `total_claims` | `int` | Total number of claims processed. |
| `verified_claims` | `int` | Number of claims that were verified. |
| `corrected_citations` | `int` | Number of citations that were corrected. |
| `removed_claims` | `int` | Number of claims removed as unsupported. |
| `softened_claims` | `int` | Number of claims softened with hedging language. |
| `overall_confidence` | `float` | Aggregate confidence score across all claims. |
| `analysis_summary` | `dict[str, Any]` | Detailed analysis breakdown. Defaults to `{}`. |
| `routing_summary` | `dict[str, Any]` | Summary of confidence routing decisions. Defaults to `{}`. |

---

## FrameworkEvent Union

The `FrameworkEvent` type is an `Annotated` discriminated union over all concrete event types. It uses `Field(discriminator="event_type")` so Pydantic can deserialize any event from a dict by inspecting the `event_type` field.

```python
from typing import Annotated
from pydantic import Field

FrameworkEvent = Annotated[
    NodeStartedEvent
    | NodeCompletedEvent
    | NodeErrorEvent
    | NodeSkippedEvent
    | NodeBudgetExceededEvent
    | LoopIterationEvent
    | LoopExitEvent
    | BranchSelectedEvent
    | AgentOutputEvent
    | AgentStreamChunkEvent
    | PlanCreatedEvent
    | ReflectionDecisionEvent
    | CoordinatorClassifiedEvent
    | BackgroundCompletedEvent
    | SynthesisStartedEvent
    | ItemStartedEvent
    | ItemCompletedEvent
    | ItemsExtractedEvent
    | EvaluationDecisionEvent
    | ReplanTriggeredEvent
    | PlanAndExecuteExitEvent
    | ToolCallEvent
    | ToolResultEvent
    | ToolCacheHitEvent
    | CheckpointSavedEvent
    | CheckpointResumedEvent
    | TokenUsageEvent
    | TokenBudgetExceededEvent
    | ConversationCompactedEvent
    | WorkflowStartedEvent
    | WorkflowCompletedEvent
    | ClaimGeneratedEvent
    | ClaimVerifiedEvent
    | CitationCorrectedEvent
    | NumericClaimDetectedEvent
    | VerificationSummaryEvent,
    Field(discriminator="event_type"),
]
```

---

## Output Models

These Pydantic models define the typed contracts for domain-specific agent outputs. They are not events themselves but are the structured data produced by agent subtypes and referenced by domain events.

### `PlanOutput`

Typed output for planner agents.

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Title of the research plan. |
| `thought` | `str` | Planner's reasoning about what to investigate. |
| `steps` | `list[dict[str, Any]]` | Ordered list of plan steps. |
| `has_enough_context` | `bool` | Whether enough context exists to skip research. Defaults to `False`. |
| `iteration` | `int` | Plan iteration number. Defaults to `1`. |

### `ReflectionOutput`

Typed output for reflector agents (domain-level reflection subtype).

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `Literal["continue", "adjust", "replan", "complete"]` | The reflector's decision. |
| `reasoning` | `str` | Explanation for the decision. |
| `suggested_changes` | `list[str] \| None` | Suggested adjustments. Normalized to `[]` if `None`. |

### `EvaluationOutput`

Typed output for evaluator agents in `plan_and_execute` nodes.

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `Literal["continue", "replan", "complete"]` | The evaluator's decision. |
| `reasoning` | `str` | Explanation for the decision. |
| `suggested_changes` | `list[str] \| None` | Suggested changes for replanning. Normalized to `[]` if `None`. |

### `CoordinatorOutput`

Typed output for coordinator agents.

| Field | Type | Description |
|-------|------|-------------|
| `complexity` | `str` | Classified complexity level. |
| `is_simple` | `bool` | Whether the query is simple. Defaults to `False`. |
| `recommended_depth` | `str` | Recommended research depth. Defaults to `"standard"`. |
| `direct_response` | `str \| None` | Direct answer for simple queries. Defaults to `None`. |
| `follow_up_type` | `str \| None` | Follow-up type for conversation queries. Defaults to `None`. |

### `ResearcherOutput`

Typed output for researcher agents.

| Field | Type | Description |
|-------|------|-------------|
| `search_queries` | `list[str]` | Search queries executed. Defaults to `[]`. |
| `observation` | `str` | Researcher's observation from the findings. Defaults to `""`. |
| `key_points` | `list[str]` | Key points extracted. Defaults to `[]`. |
| `sources_used` | `list[str]` | Source identifiers used. Defaults to `[]`. |
| `research_status` | `Literal["ok", "blocked", "insufficient_data"]` | Status of research. Defaults to `"ok"`. |
| `blocking_reason` | `str \| None` | Why research is blocked, if applicable. Defaults to `None`. |
| `findings` | `str` | Raw findings text. Defaults to `""`. |
| `sources_found` | `int` | Number of sources found. Defaults to `0`. |

### `SynthesizerOutput`

Typed output for synthesizer agents.

| Field | Type | Description |
|-------|------|-------------|
| `report` | `str` | The synthesized report text. |
| `structured_output` | `Any \| None` | Optional structured output payload. Defaults to `None`. |

### `BackgroundOutput`

Typed output for background investigator agents.

| Field | Type | Description |
|-------|------|-------------|
| `data_landscape` | `dict[str, Any]` | Structured data landscape information. Defaults to `{}`. |
| `summary` | `str` | Summary of background investigation. Defaults to `""`. |
| `query_decomposition` | `list[str]` | Sub-queries derived from the original query. Defaults to `[]`. |

---

## Deferred Events

The following events are designed but not yet implemented (deferred beyond P0):

- **`GateWaitingEvent`** -- HITL (human-in-the-loop) gate is waiting for input.
- **`GateResumedEvent`** -- HITL gate received input and resumed.
- **`GateTimeoutEvent`** -- HITL gate timed out waiting for input.

These will be added when HITL gate support is implemented in the workflow engine.

---

## See Also

- [Events Concept](../concepts/events.md)
- [Streaming and Events Guide](../guides/streaming-and-events.md)
