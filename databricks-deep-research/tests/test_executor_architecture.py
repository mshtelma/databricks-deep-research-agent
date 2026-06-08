from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "databricks_deep_research"

from databricks_deep_research.workflow import executor as executor_module  # noqa: E402
from databricks_deep_research.workflow.executor import (  # noqa: E402
    PlanCycleContext,
    _build_available_source_catalog,
    _build_evaluator_runtime_context,
    _build_planner_runtime_context,
    _extract_raw_plan_contract,
    _finalize_plan_contract,
    _format_all_observations,
    _format_source_quality,
    _normalize_executable_plan_contract,
    _populate_synthesis_state,
)


def test_runtime_modules_do_not_import_executor() -> None:
    runtime_dir = SRC_ROOT / "workflow" / "runtime"
    forbidden = (
        "from databricks_deep_research.workflow.executor import",
        "import databricks_deep_research.workflow.executor",
    )
    for path in runtime_dir.glob("*.py"):
        content = path.read_text()
        assert all(marker not in content for marker in forbidden), path


def test_executor_compatibility_surface_exports_expected_symbols() -> None:
    assert PlanCycleContext is not None
    assert callable(_extract_raw_plan_contract)
    assert callable(_finalize_plan_contract)
    assert callable(_normalize_executable_plan_contract)
    assert callable(_build_planner_runtime_context)
    assert callable(_build_evaluator_runtime_context)
    assert callable(_build_available_source_catalog)
    assert callable(_format_all_observations)
    assert callable(_format_source_quality)
    assert callable(_populate_synthesis_state)


def test_exec_plan_and_execute_delegates_to_runner() -> None:
    source = Path(executor_module.__file__).read_text()
    assert "async for event in run_plan_execute(runtime, deps):" in source


def test_workflow_state_exposes_typed_runtime_surface() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.state import WorkflowState

    state = WorkflowState(query="hello")
    state.runtime_store = TypedRuntimeStateStore(query="hello", workflow_id="wf", workflow_name="WF")
    state.append("coord", "coordination", {"complexity": "simple"})

    runtime = state.runtime_state()
    assert runtime is not None
    assert runtime.request.query == "hello"
    assert runtime.artifacts["coordination"].artifact_type == "legacy_output"
    assert runtime.artifacts["coordination"].payload == {"complexity": "simple"}


def test_typed_runtime_store_tracks_artifact_envelopes_and_planning_state() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore

    store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    store.start_node(node_id="planner", node_type="agent", label="Planner")
    store.publish_artifact(
        artifact_id="plan_1",
        artifact_type="plan",
        producer_node_id="planner",
        payload={"title": "Plan"},
        status="success",
    )
    store.begin_plan_cycle(
        cycle=0,
        title="Plan",
        thought="Think",
        has_enough_context=False,
        steps=[{"id": "step-1", "title": "One", "description": "Desc", "step_type": "research"}],
    )
    store.mark_step_completed(step_id="step-1")
    store.finalize_plan_cycle(cycle=0, exit_reason="items_exhausted")

    runtime = store.snapshot()
    assert runtime.artifacts["plan_1"].artifact_type == "plan"
    assert runtime.nodes["planner"].output_artifact_refs == ["plan_1"]
    assert runtime.capabilities.planning is not None
    assert runtime.capabilities.planning.cycles[0].steps[0].status == "completed"
    assert runtime.workflow.plan_exit_reasons == ["items_exhausted"]


def test_typed_runtime_store_ingests_evidence_with_dedup_metrics() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.runtime_core.models import (
        ObservationRecord,
        SourceRecord,
    )

    store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    delta1 = store.ingest_evidence(
        producer_node_id="researcher",
        sources=[SourceRecord(url="https://a", title="A")],
        observations=[ObservationRecord(text="Obs")],
    )
    delta2 = store.ingest_evidence(
        producer_node_id="researcher",
        sources=[SourceRecord(url="https://a", title="A")],
        observations=[ObservationRecord(text="Obs")],
    )

    runtime = store.snapshot()
    assert delta1.new_sources == 1
    assert delta1.new_observations == 1
    assert delta2.duplicate_sources == 1
    assert delta2.duplicate_observations == 1
    assert runtime.metrics.source_writes == 1
    assert runtime.metrics.observation_writes == 1
    assert runtime.metrics.source_dedup_hits == 1
    assert runtime.metrics.observation_dedup_hits == 1
    assert runtime.capabilities.evidence is not None
    assert len(runtime.capabilities.evidence.sources) == 1
    assert len(runtime.capabilities.evidence.observations) == 1


def test_typed_runtime_store_tracks_retrieval_requests_and_results() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore

    store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    store.record_retrieval_request(
        request_id="r1",
        tool_name="web_search",
        arguments={"query": "ai"},
        canonical_key="k1",
        scope="s",
    )
    store.record_retrieval_result(
        request_id="r1",
        tool_name="web_search",
        source_count=3,
        raw_source_count=4,
        accepted_source_count=3,
        rejected_source_count=1,
        tool_success=True,
        tool_error="",
    )
    store.record_retrieval_cache_hit(
        request_id="r2",
        tool_name="web_search",
        canonical_key="k1",
    )

    runtime = store.snapshot()
    assert runtime.capabilities.retrieval is not None
    assert len(runtime.capabilities.retrieval.requests) == 1
    assert len(runtime.capabilities.retrieval.results) == 2
    assert runtime.metrics.raw_sources == 4
    assert runtime.metrics.accepted_sources == 3
    assert runtime.metrics.rejected_sources == 1
    assert runtime.metrics.retrieval_cache_hits == 1


def test_typed_runtime_store_builds_synthesis_input_and_report_artifact() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.runtime_core.models import (
        ObservationRecord,
        SourceRecord,
    )

    store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    store.ingest_evidence(
        producer_node_id="researcher",
        sources=[SourceRecord(url="https://a", title="A")],
        observations=[ObservationRecord(text="Observation")],
    )
    pack = store.build_synthesis_input_pack()
    assert pack.observation_count == 1
    assert pack.source_count == 1

    artifact_id = store.publish_report_artifact(
        producer_node_id="synth",
        report="Final report",
        mode="full",
    )
    runtime = store.snapshot()
    assert runtime.capabilities.synthesis is not None
    assert runtime.capabilities.synthesis.report_artifact_id == artifact_id
    assert runtime.artifacts[artifact_id].artifact_type == "report"


def test_typed_runtime_store_publishes_verification_payload() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore

    store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    store.ensure_synthesis()
    artifact_id = store.publish_verification_payload(
        producer_node_id="synth",
        payload={
            "claims": [{"claim_text": "A"}],
            "verification_summary": {
                "total_claims": 1,
                "verified_claims": 1,
                "supported_count": 1,
                "partial_count": 0,
                "unsupported_count": 0,
                "contradicted_count": 0,
                "abstained_count": 0,
                "supported_rate": 1.0,
                "unsupported_rate": 0.0,
                "warning": False,
                "overall_confidence": 0.9,
                "analysis_summary": {"total_claims": 1},
            },
        },
    )
    runtime = store.snapshot()
    assert runtime.capabilities.verification is not None
    assert runtime.capabilities.verification.summary.total_claims == 1
    assert runtime.capabilities.verification.summary.raw["supported_count"] == 1
    assert runtime.capabilities.verification.claims[0]["claim_text"] == "A"
    assert runtime.artifacts[artifact_id].artifact_type == "verification"
    assert artifact_id in runtime.capabilities.synthesis.verification_artifact_ids


def test_runtime_selector_preserves_full_verification_summary() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.runtime_core.selectors import (
        select_verification_summary,
    )
    from databricks_deep_research.workflow.state import WorkflowState

    summary = {
        "total_claims": 3,
        "verified_claims": 2,
        "supported_count": 2,
        "partial_count": 1,
        "unsupported_count": 0,
        "contradicted_count": 0,
        "abstained_count": 0,
        "supported_rate": 2 / 3,
        "unsupported_rate": 0.0,
        "warning": False,
        "analysis_summary": {"total_claims": 1, "supported_count": 1},
    }

    state = WorkflowState(query="q")
    state.runtime_store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    state.runtime_store.publish_verification_payload(
        producer_node_id="synth",
        payload={"claims": [{"claim_text": "A"}], "verification_summary": summary},
    )

    assert select_verification_summary(state) == summary


def test_runtime_selectors_prefer_typed_state_over_legacy_state() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.runtime_core.models import (
        ObservationRecord,
        SourceRecord,
    )
    from databricks_deep_research.workflow.runtime_core.selectors import (
        select_all_observations_text,
        select_latest_observation_text,
        select_sources_count,
    )
    from databricks_deep_research.workflow.state import WorkflowState

    state = WorkflowState(query="q")
    state.append("legacy", "findings", "legacy finding")
    state.append("legacy", "all_observations", "legacy obs")
    state.runtime_store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    state.runtime_store.ingest_evidence(
        producer_node_id="researcher",
        sources=[SourceRecord(url="https://a", title="A")],
        observations=[ObservationRecord(text="typed finding")],
    )

    assert select_latest_observation_text(state) == "typed finding"
    assert "typed finding" in select_all_observations_text(state)
    assert select_sources_count(state, {}) == 1


def test_migrated_paths_do_not_duplicate_plan_cycle_runtime_updates() -> None:
    content = (SRC_ROOT / "workflow" / "runtime" / "plan_execute_runner.py").read_text()
    assert content.count("state.runtime_store.begin_plan_cycle(") == 1
    assert content.count('state.runtime_store.finalize_plan_cycle(cycle=cycle, exit_reason="empty_plan")') == 1


def test_migrated_research_projection_keeps_structured_payload_and_minimal_compatibility_keys() -> None:
    content = (SRC_ROOT / "agents" / "execution" / "state_projection.py").read_text()
    assert 'state.append(node_id, f"{config.output_key}_structured", structured_findings)' in content
    assert 'state.append(node_id, "research_status"' in content
    assert 'state.append(node_id, "blocking_reason"' in content


def test_migrated_verification_bridge_prefers_runtime_store_for_summary_state() -> None:
    content = (SRC_ROOT / "agents" / "builtins" / "synthesizer.py").read_text()
    assert 'if state.runtime_store is None:' in content


def test_background_selector_prefers_typed_state() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.runtime_core.selectors import select_background_summary
    from databricks_deep_research.workflow.state import WorkflowState

    state = WorkflowState(query="q")
    state.append("legacy", "background_summary", "legacy summary")
    state.runtime_store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    state.runtime_store.set_background(
        summary="typed summary",
        data_landscape={},
        query_decomposition=[],
        discovered_sources=[],
    )
    assert select_background_summary(state) == "typed summary"


def test_synthesis_context_compat_write_is_disabled_when_runtime_store_exists() -> None:
    content = (SRC_ROOT / "agents" / "harness.py").read_text()
    assert "if state.runtime_store is not None:\n        return" in content


def test_synthesizer_verification_extraction_uses_typed_selectors() -> None:
    content = (SRC_ROOT / "agents" / "builtins" / "synthesizer.py").read_text()
    assert 'select_claims(state)' in content
    assert 'select_verification_summary(state)' in content
    assert 'select_analysis_summary(state)' in content


def test_discovered_sources_recovery_uses_typed_selector() -> None:
    content = (SRC_ROOT / "workflow" / "runtime" / "plan_execute_recovery.py").read_text()
    assert 'select_discovered_sources(state)' in content
    assert 'state.get("discovered_sources")' not in content


def test_populate_synthesis_state_keeps_compatibility_writes() -> None:
    content = (SRC_ROOT / "workflow" / "runtime" / "plan_execute_execution.py").read_text()
    assert 'state.append(sid, "steps_executed", str(total_items_processed))' in content
    assert 'state.append(sid, "plan_iterations", str(replan_cycles + 1))' in content


def test_harness_build_input_can_resolve_migrated_keys_via_selectors() -> None:
    content = (SRC_ROOT / "agents" / "harness.py").read_text()
    assert 'resolve_input_key(state, key)' in content


def test_runtime_selectors_resolve_plan_and_findings_keys() -> None:
    from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
    from databricks_deep_research.workflow.runtime_core.models import ObservationRecord
    from databricks_deep_research.workflow.runtime_core.selectors import resolve_input_key
    from databricks_deep_research.workflow.state import WorkflowState

    state = WorkflowState(query="q")
    state.runtime_store = TypedRuntimeStateStore(query="q", workflow_id="wf", workflow_name="WF")
    state.runtime_store.begin_plan_cycle(cycle=0, title="Plan", thought="Think", has_enough_context=False, steps=[])
    state.runtime_store.ingest_evidence(
        producer_node_id="researcher",
        sources=[],
        observations=[ObservationRecord(text="Typed finding")],
    )

    assert resolve_input_key(state, "plan")["title"] == "Plan"
    assert resolve_input_key(state, "findings") == "Typed finding"
    assert resolve_input_key(state, "observation") == "Typed finding"


def test_migrated_runtime_modules_avoid_legacy_reads_for_typed_domains() -> None:
    targets = [
        SRC_ROOT / "workflow" / "runtime" / "planner_context.py",
        SRC_ROOT / "workflow" / "runtime" / "plan_execute_context.py",
        SRC_ROOT / "workflow" / "runtime" / "plan_execute_recovery.py",
    ]
    forbidden = [
        'state.get("findings")',
        'state.get("plan")',
        'state.get("discovered_sources")',
        'state.get("background_summary")',
    ]
    for path in targets:
        content = path.read_text()
        for marker in forbidden:
            assert marker not in content, f"{path}: {marker}"


def test_migrated_harness_build_input_prefers_selector_resolution() -> None:
    content = (SRC_ROOT / "agents" / "harness.py").read_text()
    assert 'resolved = resolve_input_key(state, key)' in content


def test_migrated_synthesizer_keeps_legacy_verification_state_only_without_runtime_store() -> None:
    content = (SRC_ROOT / "agents" / "builtins" / "synthesizer.py").read_text()
    assert 'if state.runtime_store is None:' in content
    assert content.count('state.append(node_id, "verification_details", payload)') == 1


def test_runner_result_prefers_typed_runtime_output_and_sources() -> None:
    content = (SRC_ROOT / "runner.py").read_text()
    assert 'runtime.capabilities.synthesis.report_artifact_id' in content
    assert 'runtime.capabilities.evidence.sources' in content


def test_migrated_modules_keep_legacy_writes_only_as_compatibility_fallbacks() -> None:
    harness = (SRC_ROOT / "agents" / "harness.py").read_text()
    synth = (SRC_ROOT / "agents" / "builtins" / "synthesizer.py").read_text()
    assert "if state.runtime_store is not None:\n        return" in harness
    assert 'if state.runtime_store is None:' in synth


def test_public_api_no_longer_exports_workflow_state() -> None:
    content = (SRC_ROOT / "__init__.py").read_text()
    assert '"WorkflowState"' not in content


def test_public_api_exports_typed_runtime_run_models() -> None:
    content = (SRC_ROOT / "__init__.py").read_text()
    assert '"RuntimeState"' in content
    assert '"WorkflowRunRequest"' in content
    assert '"WorkflowRunResult"' in content


def test_executor_exposes_typed_run_helper() -> None:
    content = (SRC_ROOT / "workflow" / "executor.py").read_text()
    assert 'async def run_workflow_typed(' in content


def test_executor_exposes_both_compat_and_typed_run_helpers() -> None:
    content = (SRC_ROOT / "workflow" / "executor.py").read_text()
    assert 'async def run_workflow(' in content
    assert 'async def run_workflow_typed(' in content


def test_runner_uses_workflow_executor_compat_path() -> None:
    content = (SRC_ROOT / "runner.py").read_text()
    assert "executor = WorkflowExecutor(" in content
    assert "definition," in content
    assert "effective_client," in content
    assert "factory_context=self._factory," in content
