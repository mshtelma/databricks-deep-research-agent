from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.config import AgentNodeConfig


@dataclass(frozen=True)
class NormalizedResearchOutput:
    state_text: str
    observation_text: str
    findings_text: str
    sources: list[Any]
    search_queries: list[str]
    key_points: list[str]
    research_status: str
    blocking_reason: str | None
    repair_mode: str | None
    skip_observation_writes: bool
    skip_source_writes: bool
    substantive_source_count: int = 0
    low_value_source_count: int = 0
    evidence_quality_summary: str = "empty"


def has_substantive_text(value: str, *, min_length: int = 30) -> bool:
    text = value.strip()
    if len(text) < min_length:
        return False
    if text in {"{}", "[]", '""'}:
        return False
    return any(char.isalpha() for char in text)


def is_semantically_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return all(is_semantically_empty(item) for item in value)
    if isinstance(value, dict):
        return all(is_semantically_empty(item) for item in value.values())
    return False


def build_observation_from_sources(sources: list[Any], *, max_items: int = 3) -> str:
    lines: list[str] = []
    for source in sources[:max_items]:
        if not isinstance(source, dict):
            lines.append(str(source).strip())
            continue
        title = str(source.get("title", "")).strip()
        snippet = str(source.get("snippet", "")).strip()
        url = str(source.get("url", "")).strip()
        if not any((title, snippet, url)):
            continue
        parts: list[str] = []
        if title:
            parts.append(title)
        if snippet:
            parts.append(snippet)
        if url:
            parts.append(url)
        lines.append(" - ".join(parts))
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return "Relevant sources identified:\n" + "\n".join(f"- {line}" for line in lines)


def merge_and_dedup_sources(existing_sources: list[Any], fallback_sources: list[Any]) -> list[Any]:
    merged = [item for item in existing_sources if not is_semantically_empty(item)]
    merged.extend(item for item in fallback_sources if not is_semantically_empty(item))
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in merged:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def normalize_research_output(
    parsed: Any,
    config: AgentNodeConfig,
    tool_sources: list[Any] | None = None,
) -> NormalizedResearchOutput | None:
    if config.subtype != "researcher":
        return None
    normalized_tool_sources = [
        item for item in (tool_sources or []) if not is_semantically_empty(item)
    ]
    if isinstance(parsed, str):
        text_value = parsed.strip()
        merged_sources = merge_and_dedup_sources([], normalized_tool_sources)
        if not text_value and merged_sources:
            text_value = build_observation_from_sources(merged_sources)
        return NormalizedResearchOutput(
            state_text=text_value,
            observation_text=text_value,
            findings_text=text_value,
            sources=merged_sources,
            search_queries=[],
            key_points=[],
            research_status="ok" if text_value else "insufficient_data",
            blocking_reason=None,
            repair_mode="source_backed_observation" if merged_sources and not parsed.strip() else None,
            skip_observation_writes=not bool(text_value),
            skip_source_writes=not bool(merged_sources),
        )
    if not isinstance(parsed, dict):
        return None
    normalized_sources = parsed.get("sources", [])
    if not isinstance(normalized_sources, list):
        normalized_sources = []
    merged_sources = merge_and_dedup_sources(normalized_sources, normalized_tool_sources)
    substantive_source_count = sum(1 for source in merged_sources if isinstance(source, dict) and source.get("admission_status") == "accepted")
    low_value_source_count = sum(1 for source in merged_sources if isinstance(source, dict) and source.get("admission_status") == "accepted_low_value")
    evidence_quality_summary = "full_text" if substantive_source_count else "metadata_only" if low_value_source_count else "empty"
    fallback_text = ""
    for key in ("summary", "analysis", "response", "answer", "message"):
        value = parsed.get(key)
        if isinstance(value, str) and has_substantive_text(value, min_length=20):
            fallback_text = value.strip()
            break
    observation_text = str(
        parsed.get("observation") or parsed.get("findings") or parsed.get(config.output_key) or fallback_text or ""
    ).strip()
    findings_text = str(parsed.get("findings") or observation_text).strip()
    state_text = observation_text or findings_text
    raw_observation = str(
        parsed.get("observation") or parsed.get("findings") or parsed.get(config.output_key) or fallback_text or ""
    ).strip()
    if not state_text and merged_sources:
        state_text = build_observation_from_sources(merged_sources)
        observation_text = state_text
        findings_text = state_text
    derived_status = "ok" if substantive_source_count else "insufficient_data" if low_value_source_count or merged_sources else ("ok" if state_text else "insufficient_data")
    return NormalizedResearchOutput(
        state_text=state_text,
        observation_text=observation_text,
        findings_text=findings_text,
        sources=merged_sources,
        search_queries=list(parsed.get("search_queries", []) or []),
        key_points=list(parsed.get("key_points", []) or []),
        research_status=str(parsed.get("research_status", derived_status)),
        blocking_reason=parsed.get("blocking_reason"),
        repair_mode=("source_backed_observation" if merged_sources and not raw_observation else "fallback_text_field" if fallback_text and not raw_observation else None),
        skip_observation_writes=not bool(observation_text.strip()),
        skip_source_writes=not bool(merged_sources),
        substantive_source_count=substantive_source_count,
        low_value_source_count=low_value_source_count,
        evidence_quality_summary=evidence_quality_summary,
    )


from databricks_deep_research.workflow.runtime_core.models import ObservationRecord, SourceRecord


def build_source_records(sources: list[Any], *, tool_name: str = "") -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for idx, source in enumerate(sources):
        if isinstance(source, dict):
            records.append(SourceRecord(
                source_id=str(source.get("id", "") or f"src-{idx}"),
                url=str(source.get("url", "") or ""),
                title=str(source.get("title", "") or ""),
                snippet=str(source.get("snippet", "") or ""),
                source_type=str(source.get("source_type", "") or ""),
                tool_name=tool_name,
                accepted=str(source.get("admission_status", "accepted")) != "rejected",
                evidence_quality=str(source.get("evidence_quality", "empty") or "empty"),
                admission_status=str(source.get("admission_status", "accepted") or "accepted"),
                admission_reason_code=str(source.get("admission_reason_code", "") or ""),
            ))
        else:
            records.append(SourceRecord(source_id=f"src-{idx}", title=str(source), tool_name=tool_name, accepted=True))
    return records


def build_observation_records(
    normalized: NormalizedResearchOutput,
    *,
    step_id: str | None = None,
) -> list[ObservationRecord]:
    text = normalized.observation_text.strip()
    if not text or normalized.skip_observation_writes:
        return []
    return [ObservationRecord(
        observation_id=step_id or "observation-0",
        text=text,
        step_id=step_id,
        substantive=True,
    )]
