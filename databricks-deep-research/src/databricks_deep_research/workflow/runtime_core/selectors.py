from __future__ import annotations

from typing import Any

from databricks_deep_research.workflow.runtime_core.models import RuntimeState


def get_runtime(state: Any) -> RuntimeState | None:
    runtime_state = getattr(state, "runtime_state", None)
    if callable(runtime_state):
        return runtime_state()
    return None


def select_data_landscape(state: Any) -> dict[str, Any]:
    runtime = get_runtime(state)
    background = getattr(runtime.capabilities, "background", None) if runtime else None
    if isinstance(background, dict):
        value = background.get("data_landscape")
        if isinstance(value, dict):
            return value
    fallback = state.get("data_landscape") if hasattr(state, "get") else None
    return fallback if isinstance(fallback, dict) else {}


def select_discovered_sources(state: Any) -> list[Any]:
    runtime = get_runtime(state)
    background = getattr(runtime.capabilities, "background", None) if runtime else None
    if isinstance(background, dict):
        value = background.get("discovered_sources")
        if isinstance(value, list):
            return value
    fallback = state.get("discovered_sources") if hasattr(state, "get") else None
    return fallback if isinstance(fallback, list) else []


def select_current_plan_title(state: Any) -> str:
    runtime = get_runtime(state)
    planning = getattr(runtime.capabilities, "planning", None) if runtime else None
    if planning is not None and getattr(planning, "current_plan_title", ""):
        return str(planning.current_plan_title)
    plan = state.get("plan") if hasattr(state, "get") else None
    if isinstance(plan, dict):
        return str(plan.get("title", "") or "")
    return str(getattr(plan, "title", "") or "")


def select_latest_observation_text(state: Any) -> str:
    runtime = get_runtime(state)
    evidence = getattr(runtime.capabilities, "evidence", None) if runtime else None
    if evidence is not None and getattr(evidence, "observations", None):
        observations = evidence.observations
        if observations:
            return str(observations[-1].text)
    value = state.get("findings") if hasattr(state, "get") else None
    return str(value or "")


def select_all_observations_text(state: Any) -> str:
    runtime = get_runtime(state)
    evidence = getattr(runtime.capabilities, "evidence", None) if runtime else None
    if evidence is not None:
        return "\n".join(f"- {obs.text[:300]}" for obs in evidence.observations[:30])
    return ""


def select_sources_count(state: Any, pools: dict[str, Any]) -> int:
    runtime = get_runtime(state)
    evidence = getattr(runtime.capabilities, "evidence", None) if runtime else None
    if evidence is not None:
        return len(evidence.sources)
    sources_pool = pools.get("sources")
    return sources_pool.count() if sources_pool else 0


def select_verification_payload(state: Any) -> dict[str, Any]:
    runtime = get_runtime(state)
    verification = getattr(runtime.capabilities, "verification", None) if runtime else None
    if verification is not None and verification.verification_details:
        return verification.verification_details
    return {}



def select_background_summary(state: Any) -> str:
    runtime = get_runtime(state)
    background = getattr(runtime.capabilities, "background", None) if runtime else None
    if background is not None and getattr(background, "summary", ""):
        return str(background.summary)
    fallback = state.get("background_summary") if hasattr(state, "get") else None
    if fallback:
        return str(fallback)
    background_value = state.get("background") if hasattr(state, "get") else None
    if isinstance(background_value, dict):
        return str(background_value.get("summary", ""))
    return str(getattr(background_value, "summary", "") or "")



def select_claims(state: Any) -> list[dict[str, Any]]:
    runtime = get_runtime(state)
    verification = getattr(runtime.capabilities, "verification", None) if runtime else None
    if verification is not None and verification.claims:
        return verification.claims
    return []


def select_verification_summary(state: Any) -> dict[str, Any]:
    runtime = get_runtime(state)
    verification = getattr(runtime.capabilities, "verification", None) if runtime else None
    if verification is not None:
        summary = verification.summary
        if summary.raw:
            return dict(summary.raw)
        return {
            "total_claims": summary.total_claims,
            "verified_claims": summary.verified_claims,
            "corrected_citations": summary.corrected_citations,
            "removed_claims": summary.removed_claims,
            "softened_claims": summary.softened_claims,
            "overall_confidence": summary.overall_confidence,
            "analysis_summary": summary.analysis_summary,
        }
    return {}


def select_analysis_summary(state: Any) -> dict[str, Any]:
    runtime = get_runtime(state)
    verification = getattr(runtime.capabilities, "verification", None) if runtime else None
    if verification is not None and verification.summary.analysis_summary:
        return verification.summary.analysis_summary
    return {}


def select_final_report(state: Any) -> str:
    runtime = get_runtime(state)
    synthesis = getattr(runtime.capabilities, "synthesis", None) if runtime else None
    if synthesis is not None and synthesis.report_artifact_id:
        artifact = runtime.artifacts.get(synthesis.report_artifact_id)
        if artifact is not None and artifact.payload is not None:
            return str(artifact.payload)
    return ""


def select_steps_executed(state: Any) -> int:
    runtime = get_runtime(state)
    planning = getattr(runtime.capabilities, "planning", None) if runtime else None
    if planning is not None:
        return len(getattr(planning, "completed_step_ids", []) or [])
    return 0


def select_plan_iterations(state: Any) -> int:
    runtime = get_runtime(state)
    planning = getattr(runtime.capabilities, "planning", None) if runtime else None
    if planning is not None:
        cycles = getattr(planning, "cycles", None) or []
        return len(cycles) if cycles else max(0, int(getattr(planning, "current_cycle", 0)) + 1)
    return 0


def select_sources_list(state: Any) -> str:
    runtime = get_runtime(state)
    evidence = getattr(runtime.capabilities, "evidence", None) if runtime else None
    if evidence is None or not getattr(evidence, "sources", None):
        return ""
    lines = []
    for item in evidence.sources[:50]:
        title = getattr(item, "title", "Source") or "Source"
        url = getattr(item, "url", "") or ""
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)



def select_current_step(state: Any) -> Any | None:
    fallback = state.get("current_step") if hasattr(state, "get") else None
    return fallback


def select_step_title(state: Any) -> str:
    current_step = select_current_step(state)
    if isinstance(current_step, dict):
        return str(current_step.get("title", "") or current_step.get("description", ""))
    return str(getattr(current_step, "title", "") or getattr(current_step, "description", "") or "")


def select_findings_text(state: Any) -> str:
    latest = select_latest_observation_text(state)
    if latest:
        return latest
    fallback = state.get("findings") if hasattr(state, "get") else None
    return str(fallback or "")


def select_plan_object(state: Any) -> Any | None:
    runtime = get_runtime(state)
    planning = getattr(runtime.capabilities, "planning", None) if runtime else None
    if planning is not None and planning.current_plan_title:
        return {
            "title": planning.current_plan_title,
            "thought": planning.current_plan_thought,
            "has_enough_context": planning.has_enough_context,
        }
    return state.get("plan") if hasattr(state, "get") else None


def resolve_input_key(state: Any, key: str) -> Any | None:
    mapped = {
        "background_summary": select_background_summary,
        "data_landscape": select_data_landscape,
        "discovered_sources": select_discovered_sources,
        "plan": select_plan_object,
        "findings": select_findings_text,
        "observation": select_latest_observation_text,
        "all_observations": select_all_observations_text,
        "sources_count": lambda s: select_sources_count(s, {}),
        "current_step": select_current_step,
        "step_title": select_step_title,
        "claims": select_claims,
        "verification_summary": select_verification_summary,
        "analysis_summary": select_analysis_summary,
        "verification_details": select_verification_payload,
    }
    resolver = mapped.get(key)
    if resolver is not None:
        return resolver(state)
    return None
