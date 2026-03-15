from __future__ import annotations

from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
from databricks_deep_research.workflow.state import WorkflowState


def test_get_prefers_log_value_over_runtime_selector() -> None:
    state = WorkflowState(runtime_store=TypedRuntimeStateStore())
    state.runtime_store.publish_verification_payload(
        producer_node_id="synth",
        payload={
            "claims": [{"claim_text": "runtime claim"}],
            "verification_summary": {"total_claims": 1, "verified_claims": 1},
            "analysis_summary": {"analysis_claim_count": 0},
        },
    )

    log_summary = {"total_claims": 99, "verified_claims": 98}
    state.append("node", "verification_summary", log_summary)

    assert state.get("verification_summary") == log_summary


def test_get_falls_back_to_runtime_backed_verification_keys() -> None:
    state = WorkflowState(runtime_store=TypedRuntimeStateStore())
    payload = {
        "claims": [{"claim_text": "claim one", "claim_role": "fact"}],
        "verifications": [{"claim_text": "claim one", "verdict": "supported"}],
        "corrections": [],
        "numeric_claims": [],
        "verification_summary": {
            "total_claims": 1,
            "verified_claims": 1,
            "supported_count": 1,
            "analysis_summary": {"analysis_claim_count": 0},
        },
        "analysis_summary": {"analysis_claim_count": 0},
    }
    state.runtime_store.publish_verification_payload(
        producer_node_id="synth",
        payload=payload,
    )

    assert state.get("claims") == payload["claims"]
    assert state.get("verification_details") == payload
    assert state.get("verification_summary") == payload["verification_summary"]
    assert state.get("analysis_summary") == payload["analysis_summary"]
