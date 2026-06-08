from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.react_loop import _looks_like_planning

logger = logging.getLogger(__name__)

_SUBSTANTIVE_EVIDENCE_QUALITIES = frozenset(
    {
        "full_text",
        "snippet_only",
        "cached",
        "builtin",
        "extracted",
        "structured",
    }
)
_LOW_VALUE_EVIDENCE_QUALITIES = frozenset({"metadata_only", "title_only"})
_EMPTY_EVIDENCE_QUALITIES = frozenset({"", "empty", "unknown", "none", "null"})
_STRIP_FROM_STATE_TEXT = frozenset(
    {
        "candidate_queries",
        "expected_sources",
        "next_steps",
        "planned_queries",
        "planned_searches",
        "query_plan",
        "research_plan",
        "search_plan",
        "search_queries",
        "sources",
        "sources_found",
        "sources_used",
        "tool_plan",
    }
)


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


def _source_value(source: Any, field: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(field, default)
    return getattr(source, field, default)


def source_has_usable_text(source: Any) -> bool:
    """Return True when a source carries text that can support a citation."""
    if source is None:
        return False
    for field in ("content", "snippet", "quote", "evidence", "text", "summary"):
        value = _source_value(source, field)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
    metrics = _source_value(source, "metrics") or _source_value(
        source, "structured_metrics"
    )
    return isinstance(metrics, (dict, list)) and bool(metrics)


def source_is_substantive(source: Any) -> bool:
    """Return True when a source is citeable evidence, not discovery metadata."""
    if source is None:
        return False
    status = str(
        _source_value(source, "admission_status", "accepted") or "accepted"
    ).lower()
    if status in {"rejected", "blocked", "failed"}:
        return False

    explicit_quality = (
        "evidence_quality" in source
        if isinstance(source, dict)
        else hasattr(source, "evidence_quality")
    )
    quality = str(_source_value(source, "evidence_quality", "") or "").lower()
    if quality in _SUBSTANTIVE_EVIDENCE_QUALITIES:
        return source_has_usable_text(source) or quality in {
            "cached",
            "builtin",
            "structured",
        }
    if quality in _LOW_VALUE_EVIDENCE_QUALITIES or status == "accepted_low_value":
        return False
    if explicit_quality and quality in _EMPTY_EVIDENCE_QUALITIES:
        return False
    return source_has_usable_text(source)


def source_is_low_value(source: Any) -> bool:
    """Return True for accepted discovery records that are not citeable."""
    if source is None:
        return False
    status = str(
        _source_value(source, "admission_status", "accepted") or "accepted"
    ).lower()
    if status in {"rejected", "blocked", "failed"}:
        return False
    quality = str(_source_value(source, "evidence_quality", "") or "").lower()
    return status == "accepted_low_value" or quality in _LOW_VALUE_EVIDENCE_QUALITIES


def filter_substantive_sources(sources: list[Any]) -> list[Any]:
    return [source for source in sources if source_is_substantive(source)]


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
    # DR_LEAK_TRACE output_normalize_in: capture what the normalizer received.
    try:
        _parsed_head = (
            parsed[:300] if isinstance(parsed, str)
            else str(parsed)[:300]
        ).replace("\n", "\\n")
        logger.info(
            "DR_LEAK_TRACE phase=output_normalize_in "
            "output_key=%s parsed_type=%s parsed_head=%r tool_sources=%d",
            config.output_key,
            type(parsed).__name__,
            _parsed_head,
            len(tool_sources or []),
        )
    except Exception as _exc:  # pragma: no cover — diagnostic only
        logger.debug("DR_LEAK_TRACE output_normalize_in skipped: %s", _exc)
    normalized_tool_sources = [
        item for item in (tool_sources or []) if not is_semantically_empty(item)
    ]
    if isinstance(parsed, str):
        text_value = parsed.strip()
        merged_sources = merge_and_dedup_sources([], normalized_tool_sources)
        citeable_sources = filter_substantive_sources(merged_sources)
        # Defense-in-depth against budget-exhausted researchers that serialized
        # a planning sentence ("Let me crawl...") as their final output. The
        # ReAct loop's exit guard catches most cases; this catches anything
        # that slips through (e.g., non-budget exits, normalization bypass).
        if text_value and _looks_like_planning(text_value):
            logger.warning(
                "NORMALIZE_PLANNING_LEAK output_key=%s preview=%r",
                config.output_key, text_value[:160],
            )
            if citeable_sources:
                text_value = build_observation_from_sources(citeable_sources)
                status = "ok"
                repair = "source_backed_observation"
                blocking: str | None = None
            else:
                text_value = ""
                status = "incomplete"
                repair = "planning_leak_dropped"
                blocking = "tool_budget_exhausted"
            return NormalizedResearchOutput(
                state_text=text_value,
                observation_text=text_value,
                findings_text=text_value,
                sources=citeable_sources,
                search_queries=[],
                key_points=[],
                research_status=status,
                blocking_reason=blocking,
                repair_mode=repair,
                skip_observation_writes=not bool(text_value),
                skip_source_writes=not bool(citeable_sources),
                substantive_source_count=len(citeable_sources),
                low_value_source_count=sum(1 for source in merged_sources if source_is_low_value(source)),
                evidence_quality_summary="full_text" if citeable_sources else "empty",
            )
        if not text_value and citeable_sources:
            text_value = build_observation_from_sources(citeable_sources)
        return NormalizedResearchOutput(
            state_text=text_value,
            observation_text=text_value,
            findings_text=text_value,
            sources=citeable_sources,
            search_queries=[],
            key_points=[],
            research_status="ok" if text_value else "insufficient_data",
            blocking_reason=None,
            repair_mode="source_backed_observation" if citeable_sources and not parsed.strip() else None,
            skip_observation_writes=not bool(text_value),
            skip_source_writes=not bool(citeable_sources),
            substantive_source_count=len(citeable_sources),
            low_value_source_count=sum(1 for source in merged_sources if source_is_low_value(source)),
            evidence_quality_summary="full_text" if citeable_sources else "empty",
        )
    if not isinstance(parsed, dict):
        return None
    normalized_sources = parsed.get("sources", [])
    if not isinstance(normalized_sources, list):
        normalized_sources = []
    merged_sources = merge_and_dedup_sources(normalized_sources, normalized_tool_sources)
    citeable_sources = filter_substantive_sources(merged_sources)
    substantive_source_count = len(citeable_sources)
    low_value_source_count = sum(1 for source in merged_sources if source_is_low_value(source))
    evidence_quality_summary = "full_text" if substantive_source_count else "metadata_only" if low_value_source_count else "empty"
    # Serialize the dict as state_text, excluding sources (already extracted
    # above for pool writes). Sources can be 100K-500K chars of raw search
    # results — including them in state_text bloats downstream agent prompts.
    import json as _json

    state_dict = {k: v for k, v in parsed.items() if k not in _STRIP_FROM_STATE_TEXT}
    state_text = _json.dumps(state_dict, default=str, ensure_ascii=False)
    observation_text = state_text
    findings_text = state_text
    logger.info(
        "NORMALIZE_OUTPUT output_key=%s state_text_source=dict_serialized state_text_len=%d preview=%r",
        config.output_key, len(state_text), state_text[:150],
    )
    derived_status = "ok" if substantive_source_count else "insufficient_data" if low_value_source_count or merged_sources else ("ok" if state_text else "insufficient_data")
    return NormalizedResearchOutput(
        state_text=state_text,
        observation_text=observation_text,
        findings_text=findings_text,
        sources=citeable_sources,
        search_queries=list(parsed.get("search_queries", []) or []),
        key_points=list(parsed.get("key_points", []) or []),
        research_status=str(parsed.get("research_status", derived_status)),
        blocking_reason=parsed.get("blocking_reason"),
        repair_mode=None,
        skip_observation_writes=not bool(observation_text.strip()),
        skip_source_writes=not bool(citeable_sources),
        substantive_source_count=substantive_source_count,
        low_value_source_count=low_value_source_count,
        evidence_quality_summary=evidence_quality_summary,
    )


from databricks_deep_research.workflow.runtime_core.models import (  # noqa: E402
    ObservationRecord,
    SourceRecord,
)


def build_source_records(sources: list[Any], *, tool_name: str = "") -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for idx, source in enumerate(sources):
        if not source_is_substantive(source):
            continue
        if isinstance(source, dict):
            records.append(SourceRecord(
                source_id=str(source.get("id", "") or f"src-{idx}"),
                url=str(source.get("url", "") or ""),
                title=str(source.get("title", "") or ""),
                snippet=str(source.get("snippet", "") or ""),
                source_type=str(source.get("source_type", "") or ""),
                tool_name=tool_name,
                accepted=True,
                evidence_quality=str(source.get("evidence_quality", "empty") or "empty"),
                admission_status=str(source.get("admission_status", "accepted") or "accepted"),
                admission_reason_code=str(source.get("admission_reason_code", "") or ""),
            ))
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
