from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RequestState(BaseModel):
    query: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class ArtifactQuality(BaseModel):
    status: Literal["success", "blocked", "degraded", "malformed", "failed", "informational"] = "informational"
    confidence: float | None = None
    substantive: bool = True
    quality_flags: list[str] = Field(default_factory=list)


class ArtifactProvenance(BaseModel):
    source_node_ids: list[str] = Field(default_factory=list)
    tool_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    upstream_artifact_refs: list[str] = Field(default_factory=list)


class ArtifactEnvelope(BaseModel):
    artifact_id: str
    artifact_type: str
    producer_node_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    schema_version: str = "1"
    payload: Any = None
    quality: ArtifactQuality = Field(default_factory=ArtifactQuality)
    provenance: ArtifactProvenance = Field(default_factory=ArtifactProvenance)
    tags: dict[str, str] = Field(default_factory=dict)


class WorkflowLifecycleState(BaseModel):
    workflow_id: str = ""
    workflow_name: str = ""
    terminal_status: Literal["running", "completed", "failed", "cancelled"] = "running"
    start_time: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    duration_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    total_tokens: int = 0
    total_sources: int = 0
    total_steps_executed: int = 0
    blocked_steps: int = 0
    missing_declared_tools: int = 0
    plan_exit_reasons: list[str] = Field(default_factory=list)


class NodeMetrics(BaseModel):
    artifacts_published: int = 0
    diagnostics_recorded: int = 0


class NodeExecutionState(BaseModel):
    node_id: str
    node_type: str = ""
    label: str = ""
    status: Literal["pending", "running", "completed", "failed", "skipped", "blocked"] = "pending"
    duration_ms: float = 0.0
    output_key: str | None = None
    output_preview: str = ""
    input_artifact_refs: list[str] = Field(default_factory=list)
    output_artifact_refs: list[str] = Field(default_factory=list)
    diagnostic_refs: list[str] = Field(default_factory=list)
    metrics: NodeMetrics = Field(default_factory=NodeMetrics)


class DiagnosticRecord(BaseModel):
    diagnostic_id: str
    category: str
    severity: Literal["info", "warning", "error"] = "info"
    message: str
    node_id: str | None = None


class RuntimeDiagnostics(BaseModel):
    records: list[DiagnosticRecord] = Field(default_factory=list)
    parse_failures: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    fallback_activations: list[str] = Field(default_factory=list)
    policy_decisions: list[str] = Field(default_factory=list)


class RuntimeMetrics(BaseModel):
    raw_sources: int = 0
    accepted_sources: int = 0
    rejected_sources: int = 0
    observation_writes: int = 0
    source_writes: int = 0
    source_dedup_hits: int = 0
    observation_dedup_hits: int = 0
    retrieval_cache_hits: int = 0
    retrieval_cache_misses: int = 0




class BackgroundState(BaseModel):
    summary: str = ""
    data_landscape: dict[str, Any] = Field(default_factory=dict)
    query_decomposition: list[str] = Field(default_factory=list)
    discovered_sources: list[Any] = Field(default_factory=list)


class CoordinationState(BaseModel):
    complexity: str = ""
    is_simple: bool = False
    recommended_depth: str = "standard"
    direct_response: str | None = None
    follow_up_type: str | None = None


class ResearchStepState(BaseModel):
    step_id: str
    title: str = ""
    description: str = ""
    step_type: str = ""
    status: Literal["pending", "running", "completed", "blocked", "failed", "skipped"] = "pending"
    blocking_reason: str | None = None


class PlanCycleRecord(BaseModel):
    cycle: int = 0
    has_enough_context: bool = False
    plan_title: str = ""
    plan_thought: str = ""
    steps: list[ResearchStepState] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    exit_reason: str | None = None




class SourceRecord(BaseModel):
    source_id: str = ""
    url: str = ""
    title: str = ""
    snippet: str = ""
    source_type: str = ""
    tool_name: str = ""
    accepted: bool = True
    evidence_quality: str = "empty"
    admission_status: str = "accepted"
    admission_reason_code: str = ""


class ObservationRecord(BaseModel):
    observation_id: str = ""
    text: str
    step_id: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    substantive: bool = True


class EvidenceDelta(BaseModel):
    new_sources: int = 0
    new_observations: int = 0
    duplicate_sources: int = 0
    duplicate_observations: int = 0




class RetrievalRequestRecord(BaseModel):
    request_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    canonical_key: str = ""
    scope: str = ""


class RetrievalResultRecord(BaseModel):
    request_id: str
    tool_name: str
    source_count: int = 0
    raw_source_count: int = 0
    accepted_source_count: int = 0
    accepted_substantive_count: int = 0
    accepted_low_value_count: int = 0
    rejected_source_count: int = 0
    tool_success: bool = True
    tool_error: str = ""
    cache_hit: bool = False
    evidence_quality: str = "empty"
    failure_mode: str = "none"
    needs_adaptation: bool = False




class SynthesisInputPack(BaseModel):
    observation_count: int = 0
    source_count: int = 0
    blocked_reason_count: int = 0
    observations_preview: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)




class VerificationSummaryRecord(BaseModel):
    raw: dict[str, Any] = Field(default_factory=dict)
    total_claims: int = 0
    verified_claims: int = 0
    corrected_citations: int = 0
    removed_claims: int = 0
    softened_claims: int = 0
    overall_confidence: float = 0.0
    analysis_summary: dict[str, Any] = Field(default_factory=dict)


class VerificationState(BaseModel):
    claims: list[dict[str, Any]] = Field(default_factory=list)
    verification_details: dict[str, Any] = Field(default_factory=dict)
    summary: VerificationSummaryRecord = Field(default_factory=VerificationSummaryRecord)
    verification_artifact_ids: list[str] = Field(default_factory=list)


class SynthesisState(BaseModel):
    mode: Literal["full", "partial", "insufficient", "transform"] = "full"
    input_pack: SynthesisInputPack = Field(default_factory=SynthesisInputPack)
    report_artifact_id: str | None = None
    verification_artifact_ids: list[str] = Field(default_factory=list)


class RetrievalState(BaseModel):
    requests: list[RetrievalRequestRecord] = Field(default_factory=list)
    results: list[RetrievalResultRecord] = Field(default_factory=list)
    cache_keys_seen: list[str] = Field(default_factory=list)
    tool_usage: dict[str, int] = Field(default_factory=dict)


class EvidenceState(BaseModel):
    sources: list[SourceRecord] = Field(default_factory=list)
    observations: list[ObservationRecord] = Field(default_factory=list)
    source_urls_seen: list[str] = Field(default_factory=list)
    observation_hashes_seen: list[str] = Field(default_factory=list)
    last_delta: EvidenceDelta = Field(default_factory=EvidenceDelta)


class PlanningState(BaseModel):
    current_cycle: int = 0
    current_plan_title: str = ""
    current_plan_thought: str = ""
    has_enough_context: bool = False
    cycles: list[PlanCycleRecord] = Field(default_factory=list)
    completed_step_ids: list[str] = Field(default_factory=list)
    blocked_step_ids: list[str] = Field(default_factory=list)


class CapabilityStates(BaseModel):
    coordination: CoordinationState | None = None
    background: BackgroundState | None = None
    planning: PlanningState | None = None
    retrieval: RetrievalState | None = None
    evidence: EvidenceState | None = None
    synthesis: SynthesisState | None = None
    verification: VerificationState | None = None


class RuntimeState(BaseModel):
    request: RequestState = Field(default_factory=RequestState)
    workflow: WorkflowLifecycleState = Field(default_factory=WorkflowLifecycleState)
    nodes: dict[str, NodeExecutionState] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactEnvelope] = Field(default_factory=dict)
    diagnostics: RuntimeDiagnostics = Field(default_factory=RuntimeDiagnostics)
    metrics: RuntimeMetrics = Field(default_factory=RuntimeMetrics)
    capabilities: CapabilityStates = Field(default_factory=CapabilityStates)
