from __future__ import annotations

from typing import Any

from databricks_deep_research.agents.execution.output_normalizer import (
    source_is_substantive,
)
from databricks_deep_research.events.types import CoordinatorOutput
from databricks_deep_research.workflow.runtime_core.models import (
    ArtifactEnvelope,
    ArtifactProvenance,
    ArtifactQuality,
    BackgroundState,
    CoordinationState,
    DiagnosticRecord,
    EvidenceDelta,
    EvidenceState,
    NodeExecutionState,
    ObservationRecord,
    PlanCycleRecord,
    PlanningState,
    ResearchStepState,
    RetrievalRequestRecord,
    RetrievalResultRecord,
    RetrievalState,
    RuntimeState,
    SourceRecord,
    SynthesisInputPack,
    SynthesisState,
    VerificationState,
    VerificationSummaryRecord,
)


class TypedRuntimeStateStore:
    def __init__(self, *, query: str = "", workflow_id: str = "", workflow_name: str = "") -> None:
        self._state = RuntimeState()
        self._state.request.query = query
        self._state.workflow.workflow_id = workflow_id
        self._state.workflow.workflow_name = workflow_name

    def snapshot(self) -> RuntimeState:
        return self._state.model_copy(deep=True)

    def runtime(self) -> RuntimeState:
        return self._state

    def publish_artifact(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        producer_node_id: str,
        payload: Any,
        status: str = "informational",
        substantive: bool = True,
        tags: dict[str, str] | None = None,
    ) -> ArtifactEnvelope:
        envelope = ArtifactEnvelope(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            producer_node_id=producer_node_id,
            payload=payload,
            quality=ArtifactQuality(status=status, substantive=substantive),
            provenance=ArtifactProvenance(source_node_ids=[producer_node_id]),
            tags=tags or {},
        )
        self._state.artifacts[artifact_id] = envelope
        node = self._state.nodes.get(producer_node_id)
        if node is not None:
            node.output_artifact_refs.append(artifact_id)
            node.metrics.artifacts_published += 1
        return envelope

    def set_artifact(self, key: str, value: Any) -> None:
        self.publish_artifact(
            artifact_id=key,
            artifact_type="legacy_output",
            producer_node_id="legacy",
            payload=value,
        )

    def get_artifact(self, key: str) -> Any | None:
        artifact = self._state.artifacts.get(key)
        return artifact.payload if artifact is not None else None

    def start_node(self, *, node_id: str, node_type: str, label: str) -> None:
        self._state.nodes[node_id] = NodeExecutionState(
            node_id=node_id,
            node_type=node_type,
            label=label,
            status="running",
        )

    def complete_node(self, *, node_id: str, duration_ms: float, output_key: str | None = None, output_preview: str = "") -> None:
        node = self._state.nodes.get(node_id)
        if node is None:
            node = NodeExecutionState(node_id=node_id)
            self._state.nodes[node_id] = node
        node.status = "completed"
        node.duration_ms = duration_ms
        node.output_key = output_key
        node.output_preview = output_preview[:400]

    def fail_node(self, *, node_id: str, duration_ms: float = 0.0) -> None:
        node = self._state.nodes.get(node_id)
        if node is None:
            node = NodeExecutionState(node_id=node_id)
            self._state.nodes[node_id] = node
        node.status = "failed"
        node.duration_ms = duration_ms

    def set_workflow_completed(self, *, duration_ms: float, total_tokens: int, total_sources: int, total_steps_executed: int, blocked_steps: int, missing_declared_tools: int) -> None:
        wf = self._state.workflow
        wf.terminal_status = "completed"
        wf.duration_ms = duration_ms
        wf.total_tokens = total_tokens
        wf.total_sources = total_sources
        wf.total_steps_executed = total_steps_executed
        wf.blocked_steps = blocked_steps
        wf.missing_declared_tools = missing_declared_tools

    def set_workflow_failed(self, *, duration_ms: float, error_type: str, error_message: str) -> None:
        wf = self._state.workflow
        wf.terminal_status = "failed"
        wf.duration_ms = duration_ms
        wf.error_type = error_type
        wf.error_message = error_message

    def set_workflow_cancelled(self, *, duration_ms: float) -> None:
        wf = self._state.workflow
        wf.terminal_status = "cancelled"
        wf.duration_ms = duration_ms

    def set_coordination(self, output: CoordinatorOutput) -> None:
        self._state.capabilities.coordination = CoordinationState(
            complexity=output.complexity,
            is_simple=output.is_simple,
            recommended_depth=output.recommended_depth,
            direct_response=output.direct_response,
            follow_up_type=output.follow_up_type,
        )


    def block_node(self, *, node_id: str, reason: str, duration_ms: float = 0.0) -> None:
        node = self._state.nodes.get(node_id)
        if node is None:
            node = NodeExecutionState(node_id=node_id)
            self._state.nodes[node_id] = node
        node.status = "blocked"
        node.duration_ms = duration_ms
        self.record_diagnostic(category="blocked", severity="warning", message=reason, node_id=node_id)

    def record_diagnostic(self, *, category: str, severity: str, message: str, node_id: str | None = None) -> str:
        diagnostic_id = f"diag_{len(self._state.diagnostics.records)}"
        record = DiagnosticRecord(
            diagnostic_id=diagnostic_id,
            category=category,
            severity=severity,
            message=message,
            node_id=node_id,
        )
        self._state.diagnostics.records.append(record)
        if node_id and node_id in self._state.nodes:
            self._state.nodes[node_id].diagnostic_refs.append(diagnostic_id)
            self._state.nodes[node_id].metrics.diagnostics_recorded += 1
        return diagnostic_id

    def ensure_planning(self) -> PlanningState:
        if self._state.capabilities.planning is None:
            self._state.capabilities.planning = PlanningState()
        return self._state.capabilities.planning

    def begin_plan_cycle(self, *, cycle: int, title: str = "", thought: str = "", has_enough_context: bool = False, steps: list[dict[str, Any]] | None = None) -> None:
        planning = self.ensure_planning()
        planning.current_cycle = cycle
        planning.current_plan_title = title
        planning.current_plan_thought = thought
        planning.has_enough_context = has_enough_context
        record = PlanCycleRecord(
            cycle=cycle,
            plan_title=title,
            plan_thought=thought,
            has_enough_context=has_enough_context,
            steps=[
                ResearchStepState(
                    step_id=str(step.get("id", f"step-{idx}")),
                    title=str(step.get("title", "")),
                    description=str(step.get("description", "")),
                    step_type=str(step.get("step_type", "")),
                )
                for idx, step in enumerate(steps or []) if isinstance(step, dict)
            ],
        )
        planning.cycles = [r for r in planning.cycles if r.cycle != cycle] + [record]

    def finalize_plan_cycle(self, *, cycle: int, exit_reason: str) -> None:
        planning = self.ensure_planning()
        for record in planning.cycles:
            if record.cycle == cycle:
                record.exit_reason = exit_reason
                break
        if exit_reason not in self._state.workflow.plan_exit_reasons:
            self._state.workflow.plan_exit_reasons.append(exit_reason)

    def mark_step_completed(self, *, step_id: str) -> None:
        planning = self.ensure_planning()
        if step_id not in planning.completed_step_ids:
            planning.completed_step_ids.append(step_id)
        for record in planning.cycles:
            for step in record.steps:
                if step.step_id == step_id:
                    step.status = "completed"

    def mark_step_blocked(self, *, step_id: str, reason: str) -> None:
        planning = self.ensure_planning()
        if step_id not in planning.blocked_step_ids:
            planning.blocked_step_ids.append(step_id)
        self._state.workflow.blocked_steps += 1
        for record in planning.cycles:
            for step in record.steps:
                if step.step_id == step_id:
                    step.status = "blocked"
                    step.blocking_reason = reason


    def ensure_evidence(self) -> EvidenceState:
        if self._state.capabilities.evidence is None:
            self._state.capabilities.evidence = EvidenceState()
        return self._state.capabilities.evidence

    def ingest_evidence(
        self,
        *,
        producer_node_id: str,
        sources: list[SourceRecord],
        observations: list[ObservationRecord],
    ) -> EvidenceDelta:
        evidence = self.ensure_evidence()
        delta = EvidenceDelta()
        seen_urls = set(evidence.source_urls_seen)
        seen_obs = set(evidence.observation_hashes_seen)

        for source in sources:
            if not source_is_substantive(source):
                self.record_diagnostic(
                    category="evidence",
                    severity="info",
                    message=(
                        "Skipped non-substantive source record during evidence ingestion"
                    ),
                    node_id=producer_node_id,
                )
                continue
            key = source.url or source.source_id or source.title
            if key and key in seen_urls:
                delta.duplicate_sources += 1
                continue
            if key:
                seen_urls.add(key)
                evidence.source_urls_seen.append(key)
            evidence.sources.append(source)
            delta.new_sources += 1

        for observation in observations:
            key = observation.text.strip()
            if key in seen_obs:
                delta.duplicate_observations += 1
                continue
            seen_obs.add(key)
            evidence.observation_hashes_seen.append(key)
            evidence.observations.append(observation)
            delta.new_observations += 1

        evidence.last_delta = delta
        self._state.metrics.source_writes += delta.new_sources
        self._state.metrics.observation_writes += delta.new_observations
        self._state.metrics.source_dedup_hits += delta.duplicate_sources
        self._state.metrics.observation_dedup_hits += delta.duplicate_observations

        self.publish_artifact(
            artifact_id=f"evidence_delta_{len(self._state.artifacts)}",
            artifact_type="evidence_delta",
            producer_node_id=producer_node_id,
            payload=delta.model_dump(mode="json"),
            status="success",
            substantive=bool(delta.new_sources or delta.new_observations),
        )
        return delta


    def ensure_retrieval(self) -> RetrievalState:
        if self._state.capabilities.retrieval is None:
            self._state.capabilities.retrieval = RetrievalState()
        return self._state.capabilities.retrieval

    def record_retrieval_request(
        self,
        *,
        request_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        canonical_key: str,
        scope: str = "",
    ) -> None:
        retrieval = self.ensure_retrieval()
        retrieval.requests.append(RetrievalRequestRecord(
            request_id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            canonical_key=canonical_key,
            scope=scope,
        ))
        retrieval.tool_usage[tool_name] = retrieval.tool_usage.get(tool_name, 0) + 1
        if canonical_key not in retrieval.cache_keys_seen:
            retrieval.cache_keys_seen.append(canonical_key)
            self._state.metrics.retrieval_cache_misses += 1

    def record_retrieval_cache_hit(
        self,
        *,
        request_id: str,
        tool_name: str,
        canonical_key: str = "",  # noqa: ARG002
    ) -> None:
        retrieval = self.ensure_retrieval()
        retrieval.results.append(RetrievalResultRecord(
            request_id=request_id,
            tool_name=tool_name,
            cache_hit=True,
        ))
        self._state.metrics.retrieval_cache_hits += 1

    def record_retrieval_result(
        self,
        *,
        request_id: str,
        tool_name: str,
        source_count: int,
        raw_source_count: int,
        accepted_source_count: int,
        rejected_source_count: int,
        tool_success: bool,
        tool_error: str,
    ) -> None:
        retrieval = self.ensure_retrieval()
        retrieval.results.append(RetrievalResultRecord(
            request_id=request_id,
            tool_name=tool_name,
            source_count=source_count,
            raw_source_count=raw_source_count,
            accepted_source_count=accepted_source_count,
            rejected_source_count=rejected_source_count,
            tool_success=tool_success,
            tool_error=tool_error,
            cache_hit=False,
        ))
        self._state.metrics.raw_sources += raw_source_count
        self._state.metrics.accepted_sources += accepted_source_count
        self._state.metrics.rejected_sources += rejected_source_count


    def ensure_synthesis(self) -> SynthesisState:
        if self._state.capabilities.synthesis is None:
            self._state.capabilities.synthesis = SynthesisState()
        return self._state.capabilities.synthesis

    def build_synthesis_input_pack(self) -> SynthesisInputPack:
        evidence = self._state.capabilities.evidence
        diagnostics = self._state.diagnostics
        if evidence is None:
            return SynthesisInputPack(
                blocked_reason_count=len(diagnostics.blocked_reasons),
            )
        substantive_sources = [
            source for source in evidence.sources if source_is_substantive(source)
        ]
        return SynthesisInputPack(
            observation_count=len(evidence.observations),
            source_count=len(substantive_sources),
            blocked_reason_count=len(diagnostics.blocked_reasons),
            observations_preview=[obs.text[:200] for obs in evidence.observations[:5]],
            source_urls=[src.url for src in substantive_sources[:10] if src.url],
        )

    def set_synthesis_mode(self, mode: str) -> None:
        synthesis = self.ensure_synthesis()
        synthesis.mode = mode  # type: ignore[assignment]
        synthesis.input_pack = self.build_synthesis_input_pack()

    def publish_report_artifact(self, *, producer_node_id: str, report: Any, mode: str) -> str:
        synthesis = self.ensure_synthesis()
        synthesis.mode = mode  # type: ignore[assignment]
        synthesis.input_pack = self.build_synthesis_input_pack()
        artifact_id = f"report_{producer_node_id}_{len(self._state.artifacts)}"
        self.publish_artifact(
            artifact_id=artifact_id,
            artifact_type="report",
            producer_node_id=producer_node_id,
            payload=report,
            status="success",
            substantive=bool(str(report).strip()),
            tags={"mode": mode},
        )
        synthesis.report_artifact_id = artifact_id
        return artifact_id


    def ensure_verification(self) -> VerificationState:
        if self._state.capabilities.verification is None:
            self._state.capabilities.verification = VerificationState()
        return self._state.capabilities.verification

    def publish_verification_payload(
        self,
        *,
        producer_node_id: str,
        payload: dict[str, Any],
    ) -> str:
        verification = self.ensure_verification()
        claims = list(payload.get("claims", []) or [])
        summary_data = dict(payload.get("verification_summary", {}) or {})
        verification.claims = claims
        verification.verification_details = payload
        verification.summary = VerificationSummaryRecord(
            raw=summary_data,
            total_claims=int(summary_data.get("total_claims", len(claims)) or 0),
            verified_claims=int(summary_data.get("verified_claims", summary_data.get("supported_count", 0)) or 0),
            corrected_citations=int(summary_data.get("corrected_citations", 0) or 0),
            removed_claims=int(summary_data.get("removed_claims", 0) or 0),
            softened_claims=int(summary_data.get("softened_claims", 0) or 0),
            overall_confidence=float(summary_data.get("overall_confidence", 0.0) or 0.0),
            analysis_summary=dict(summary_data.get("analysis_summary", {}) or {}),
        )
        artifact_id = f"verification_{producer_node_id}_{len(self._state.artifacts)}"
        self.publish_artifact(
            artifact_id=artifact_id,
            artifact_type="verification",
            producer_node_id=producer_node_id,
            payload=payload,
            status="success",
            substantive=bool(claims or summary_data),
        )
        verification.verification_artifact_ids.append(artifact_id)
        synthesis = self._state.capabilities.synthesis
        if synthesis is not None:
            synthesis.verification_artifact_ids.append(artifact_id)
        return artifact_id


    def set_background(
        self,
        *,
        summary: str,
        data_landscape: dict[str, Any],
        query_decomposition: list[str],
        discovered_sources: list[Any],
    ) -> None:
        self._state.capabilities.background = BackgroundState(
            summary=summary,
            data_landscape=data_landscape,
            query_decomposition=query_decomposition,
            discovered_sources=discovered_sources,
        )
