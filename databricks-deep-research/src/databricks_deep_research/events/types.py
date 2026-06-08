"""
Streaming Event Types.

The executor yields these events as an async generator.
The consuming application maps them to its own transport (e.g., SSE).

Changes from original:
- Pydantic BaseModel (not dataclass(frozen=True))
- `event_type: Literal[...]` discriminator on every event
- Verification events added for P0d reclaim mode
- Gate events deferred beyond P0
- Domain-neutral step events replaced with plan_and_execute item events
- EvaluationOutput added for framework plan_and_execute evaluator
- FrameworkEvent uses Annotated discriminated union
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

# --- Base event ---


class StreamEvent(BaseModel):
    """Base class for all workflow execution events.

    Every event carries an `event_type` literal discriminator for safe
    deserialization and pattern matching.
    """

    event_type: str
    node_id: str
    timestamp: str  # ISO 8601


# --- Node lifecycle events ---


class NodeStartedEvent(StreamEvent):
    """Emitted when any node begins executing."""

    event_type: Literal["node_started"] = "node_started"
    node_type: str
    label: str


class NodeCompletedEvent(StreamEvent):
    """Emitted when any node finishes successfully."""

    event_type: Literal["node_completed"] = "node_completed"
    duration_ms: float


class NodeErrorEvent(StreamEvent):
    """Emitted when a node encounters an error."""

    event_type: Literal["node_error"] = "node_error"
    error_message: str
    will_retry: bool = False
    retry_attempt: int = 0


class NodeSkippedEvent(StreamEvent):
    """Emitted when a node is skipped due to error_handling=skip."""

    event_type: Literal["node_skipped"] = "node_skipped"
    reason: str


class NodeBudgetExceededEvent(StreamEvent):
    """Emitted when a node exceeds its configured wall-clock budget."""

    event_type: Literal["node_budget_exceeded"] = "node_budget_exceeded"
    budget_seconds: float
    elapsed_ms: float
    reason: str = "budget_exceeded"


# --- Loop events ---


class LoopIterationEvent(StreamEvent):
    """Emitted at the start of each loop iteration."""

    event_type: Literal["loop_iteration"] = "loop_iteration"
    iteration: int
    max_iterations: int


class LoopExitEvent(StreamEvent):
    """Emitted when a loop terminates."""

    event_type: Literal["loop_exit"] = "loop_exit"
    reason: str  # "condition_met", "max_iterations", "parse_failure"
    total_iterations: int


# --- Conditional events ---


class BranchSelectedEvent(StreamEvent):
    """Emitted when a conditional node selects a branch."""

    event_type: Literal["branch_selected"] = "branch_selected"
    branch_index: int
    condition_summary: str


# --- Agent events ---


class AgentOutputEvent(StreamEvent):
    """Emitted when an agent produces its final output."""

    event_type: Literal["agent_output"] = "agent_output"
    output_key: str
    output_preview: str  # Truncated preview of the output


class AgentStreamChunkEvent(StreamEvent):
    """Emitted for streaming synthesis -- token-by-token output."""

    event_type: Literal["agent_stream_chunk"] = "agent_stream_chunk"
    chunk: str
    subtype: str = ""


class ModelCallEvent(StreamEvent):
    """Emitted by FrameworkLLMClient after model resolution, before the HTTP
    call. Lets observers see which concrete model handled each tier
    request — used by the scaffold-and-run test to verify the designer's
    architect used Opus (and critic used GPT-5)."""

    event_type: Literal["model_call"] = "model_call"
    node_id: str = ""
    tier: str = ""           # logical tier name ("complex", "critic", etc.)
    model: str = ""          # concrete endpoint identifier (e.g. "databricks-claude-opus-4-6")
    request_id: str = ""     # OpenAI-API-style request id, optional


# --- Typed output models (per-subtype Pydantic contracts) ---


class PlanOutput(BaseModel):
    """Typed output for planner agents."""

    title: str
    thought: str
    steps: list[dict[str, Any]]
    has_enough_context: bool = False
    iteration: int = 1


class ReflectionOutput(BaseModel):
    """Typed output for reflector agents (domain-level reflection subtype)."""

    decision: Literal["continue", "adjust", "replan", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = Field(default=None)

    @field_validator("suggested_changes", mode="before")
    @classmethod
    def _normalize_suggested_changes(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class EvaluationOutput(BaseModel):
    """Typed output for evaluator agents in plan_and_execute nodes."""

    decision: Literal["continue", "replan", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = Field(default=None)

    @field_validator("suggested_changes", mode="before")
    @classmethod
    def _normalize_suggested_changes(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class CoordinatorOutput(BaseModel):
    """Typed output for coordinator agents."""

    complexity: str
    is_simple: bool = False
    recommended_depth: str = "standard"
    direct_response: str | None = None
    follow_up_type: str | None = None


class ResearcherOutput(BaseModel):
    """Typed output for researcher agents."""

    search_queries: list[str] = Field(default_factory=list)
    observation: str = ""
    key_points: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    research_status: Literal["ok", "blocked", "insufficient_data"] = "ok"
    blocking_reason: str | None = None
    findings: str = ""
    sources_found: int = 0


class SynthesizerOutput(BaseModel):
    """Typed output for synthesizer agents."""

    report: str
    structured_output: Any | None = None


class BackgroundOutput(BaseModel):
    """Typed output for background investigator agents."""

    data_landscape: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    query_decomposition: list[str] = Field(default_factory=list)


# --- Domain-specific events (emitted by builtin subtypes) ---


class PlanCreatedEvent(StreamEvent):
    """Emitted when a planner creates or updates a research plan."""

    event_type: Literal["plan_created"] = "plan_created"
    plan_id: str
    title: str
    thought: str
    steps: list[dict[str, Any]]
    iteration: int = 1
    has_enough_context: bool = False


class ReflectionDecisionEvent(StreamEvent):
    """Emitted when a reflector makes a CONTINUE/ADJUST/COMPLETE decision."""

    event_type: Literal["reflection_decision"] = "reflection_decision"
    decision: str  # "continue", "adjust", "complete"
    reasoning: str
    suggested_changes: list[str] | None = Field(default=None)
    evidence_sufficiency: str | None = None
    failure_mode: str | None = None

    @field_validator("suggested_changes", mode="before")
    @classmethod
    def _normalize_suggested_changes(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class CoordinatorClassifiedEvent(StreamEvent):
    """Emitted when the coordinator classifies a query."""

    event_type: Literal["coordinator_classified"] = "coordinator_classified"
    complexity: str
    recommended_depth: str
    is_simple: bool = False
    direct_response: str | None = None
    follow_up_type: str | None = None
    reasoning: str = ""


class BackgroundCompletedEvent(StreamEvent):
    """Emitted when background investigation completes."""

    event_type: Literal["background_completed"] = "background_completed"
    sources_discovered: int = 0
    data_landscape_summary: str = ""
    data_landscape: dict[str, Any] = Field(default_factory=dict)
    query_decomposition: list[str] = Field(default_factory=list)


class SynthesisStartedEvent(StreamEvent):
    """Emitted when synthesis begins."""

    event_type: Literal["synthesis_started"] = "synthesis_started"
    total_observations: int = 0
    total_sources: int = 0


# --- Plan-and-execute events ---


class ItemStartedEvent(StreamEvent):
    """Emitted when a plan_and_execute item begins execution."""

    event_type: Literal["item_started"] = "item_started"
    item_index: int
    item_summary: str
    total_items: int


class ItemCompletedEvent(StreamEvent):
    """Emitted when a plan_and_execute item finishes."""

    event_type: Literal["item_completed"] = "item_completed"
    item_index: int
    items_processed: int


class ItemsExtractedEvent(StreamEvent):
    """Emitted when items are extracted from the plan in plan_and_execute."""

    event_type: Literal["items_extracted"] = "items_extracted"
    total_items: int
    items_path: str
    cycle: int


class EvaluationDecisionEvent(StreamEvent):
    """Emitted when the plan_and_execute evaluator makes a decision."""

    event_type: Literal["evaluation_decision"] = "evaluation_decision"
    decision: str  # "continue", "adjust", "replan", "complete"
    reasoning: str
    items_processed: int
    evidence_sufficiency: str | None = None
    failure_mode: str | None = None


class ReplanTriggeredEvent(StreamEvent):
    """Emitted when a plan_and_execute node triggers a replan cycle."""

    event_type: Literal["replan_triggered"] = "replan_triggered"
    cycle: int
    reason: str
    items_remaining: int


class PlanAndExecuteExitEvent(StreamEvent):
    """Emitted when a plan_and_execute node exits."""

    event_type: Literal["plan_and_execute_exit"] = "plan_and_execute_exit"
    reason: str
    total_items_processed: int
    replan_cycles: int
    total_planned: int = 0
    completion_mode: str = "normal"
    evidence_sufficiency: str | None = None
    failure_mode: str | None = None


# --- Tool events ---


class ToolCallEvent(StreamEvent):
    """Emitted when an agent or tool node calls a tool."""

    event_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(StreamEvent):
    """Emitted when a tool returns its result."""

    event_type: Literal["tool_result"] = "tool_result"
    tool_name: str
    result_summary: str
    source_count: int = 0
    raw_source_count: int = 0
    accepted_source_count: int = 0
    rejected_source_count: int = 0
    tool_success: bool = True
    tool_error: str = ""


class ToolCacheHitEvent(StreamEvent):
    """Emitted when a tool call is skipped due to dedup cache hit."""

    event_type: Literal["tool_cache_hit"] = "tool_cache_hit"
    tool_name: str
    cache_key: str


# --- Checkpoint events ---


class CheckpointSavedEvent(StreamEvent):
    """Emitted after state is checkpointed."""

    event_type: Literal["checkpoint_saved"] = "checkpoint_saved"
    checkpoint_size: int  # bytes


class CheckpointResumedEvent(StreamEvent):
    """Emitted when execution resumes from a checkpoint."""

    event_type: Literal["checkpoint_resumed"] = "checkpoint_resumed"
    resumed_from: str  # ISO timestamp of checkpoint


# --- Token budget events ---


class TokenUsageEvent(StreamEvent):
    """Periodic token usage report."""

    event_type: Literal["token_usage"] = "token_usage"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    budget_remaining: int  # -1 if unlimited


class TokenBudgetExceededEvent(StreamEvent):
    """Emitted when token budget is exhausted."""

    event_type: Literal["token_budget_exceeded"] = "token_budget_exceeded"
    used: int
    limit: int


# --- Conversation events ---


class ConversationCompactedEvent(StreamEvent):
    """Emitted when agent conversation is compacted to save context."""

    event_type: Literal["conversation_compacted"] = "conversation_compacted"
    tokens_saved: int


# --- Workflow-level events ---


class WorkflowStartedEvent(StreamEvent):
    """Emitted when workflow execution begins."""

    event_type: Literal["workflow_started"] = "workflow_started"
    workflow_id: str
    workflow_name: str


class WorkflowCompletedEvent(StreamEvent):
    """Emitted when workflow execution finishes."""

    event_type: Literal["workflow_completed"] = "workflow_completed"
    workflow_id: str
    duration_ms: float
    total_tokens: int
    final_report: str = ""
    structured_output: Any | None = None
    total_sources: int = 0
    total_steps_executed: int = 0


class WorkflowFailedEvent(StreamEvent):
    """Emitted when workflow execution fails fatally."""

    event_type: Literal["workflow_failed"] = "workflow_failed"
    workflow_id: str
    duration_ms: float
    error_type: str
    error_message: str
    total_sources: int = 0
    total_steps_executed: int = 0


# --- Verification events (P0d: citation pipeline) ---


class ClaimGeneratedEvent(StreamEvent):
    """Emitted when a claim is generated during interleaved synthesis."""

    event_type: Literal["claim_generated"] = "claim_generated"
    claim_text: str
    claim_index: int
    citation_keys: list[str] = Field(default_factory=list)
    claim_role: str = "fact"


class ClaimVerifiedEvent(StreamEvent):
    """Emitted when a claim passes NLI verification."""

    event_type: Literal["claim_verified"] = "claim_verified"
    claim_index: int
    verdict: str  # "supported", "not_supported", "not_enough_info"
    confidence: float
    verification_confidence: float = 0.0
    routing_confidence_level: str = ""
    routing_confidence_score: float = 0.0
    evidence_match_score: float = 0.0
    used_quick_verification: bool = False
    verification_latency_ms: float = 0.0
    claim_role: str = "fact"
    verification_method: str = ""
    evidence_snippet: str = ""
    claim_text: str = ""
    # Numeric citation keys (e.g. "1", "2") matching the markers in the rendered
    # report, so the live UI can color this claim's markers before persistence.
    # Mirrors ClaimGeneratedEvent.citation_keys. See _normalize_verification_records.
    citation_key: str | None = None
    citation_keys: list[str] = Field(default_factory=list)


class CitationCorrectedEvent(StreamEvent):
    """Emitted when a citation is corrected post-verification."""

    event_type: Literal["citation_corrected"] = "citation_corrected"
    claim_index: int
    action: str  # "keep", "replace", "remove", "add_alternate"
    original_key: str = ""
    corrected_key: str = ""


class NumericClaimDetectedEvent(StreamEvent):
    """Emitted when a numeric claim is detected and queued for QA verification."""

    event_type: Literal["numeric_claim_detected"] = "numeric_claim_detected"
    claim_index: int
    numeric_value: str
    verification_status: str = "pending"


class VerificationSummaryEvent(StreamEvent):
    """Emitted at the end of the verification pipeline with aggregate stats."""

    event_type: Literal["verification_summary"] = "verification_summary"
    total_claims: int
    verified_claims: int
    corrected_citations: int
    removed_claims: int
    softened_claims: int
    overall_confidence: float
    analysis_summary: dict[str, Any] = Field(default_factory=dict)
    routing_summary: dict[str, Any] = Field(default_factory=dict)


# --- Gate (HITL) events (Phase 2) ---
from databricks_deep_research.events.hitl import (  # noqa: E402
    GateDeniedEvent,
    GateResumedEvent,
    GateTimeoutEvent,
    GateWaitingEvent,
)


# --- Discriminated union type ---

FrameworkEvent = Annotated[
    # Node lifecycle
    NodeStartedEvent
    | NodeCompletedEvent
    | NodeErrorEvent
    | NodeSkippedEvent
    | NodeBudgetExceededEvent
    # Loop
    | LoopIterationEvent
    | LoopExitEvent
    # Conditional
    | BranchSelectedEvent
    # Agent
    | AgentOutputEvent
    | AgentStreamChunkEvent
    | ModelCallEvent
    # Domain-specific (builtin subtypes)
    | PlanCreatedEvent
    | ReflectionDecisionEvent
    | CoordinatorClassifiedEvent
    | BackgroundCompletedEvent
    | SynthesisStartedEvent
    # Plan-and-execute
    | ItemStartedEvent
    | ItemCompletedEvent
    | ItemsExtractedEvent
    | EvaluationDecisionEvent
    | ReplanTriggeredEvent
    | PlanAndExecuteExitEvent
    # Tool
    | ToolCallEvent
    | ToolResultEvent
    | ToolCacheHitEvent
    # Checkpoint
    | CheckpointSavedEvent
    | CheckpointResumedEvent
    # Token budget
    | TokenUsageEvent
    | TokenBudgetExceededEvent
    # Conversation
    | ConversationCompactedEvent
    # Workflow-level
    | WorkflowStartedEvent
    | WorkflowCompletedEvent
    | WorkflowFailedEvent
    # Verification (P0d)
    | ClaimGeneratedEvent
    | ClaimVerifiedEvent
    | CitationCorrectedEvent
    | NumericClaimDetectedEvent
    | VerificationSummaryEvent
    # HITL gate (Phase 2)
    | GateWaitingEvent
    | GateResumedEvent
    | GateDeniedEvent
    | GateTimeoutEvent,
    Field(discriminator="event_type"),
]
