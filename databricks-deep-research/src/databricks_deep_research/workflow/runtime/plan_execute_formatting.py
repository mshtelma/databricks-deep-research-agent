from __future__ import annotations

from typing import Any


def obs_to_text(obs: Any) -> str:
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        return str(obs.get("content") or obs.get("findings") or obs)
    return str(obs)


def extract_step_title(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("title", "description", "name", "query", "step"):
            val = item.get(key)
            if val:
                return str(val)[:200]
        return str(item)[:200]
    if isinstance(item, str):
        return item[:200]
    return str(item)[:200]


def format_plan_for_reflector(_plan_data: Any, items: list[Any], current_idx: int) -> str:
    lines: list[str] = []
    for idx, step in enumerate(items):
        title = extract_step_title(step)
        status = "COMPLETED" if idx < current_idx else "JUST COMPLETED" if idx == current_idx else "PENDING"
        lines.append(f"  Step {idx + 1}/{len(items)} [{status}]: {title}")
    return "\n".join(lines) or "(no plan available)"


def format_source_topics(pools: dict[str, Any]) -> str:
    sources_pool = pools.get("sources")
    if not sources_pool or sources_pool.count() == 0:
        return "(no sources collected yet)"
    topics = sources_pool.topics()
    if topics:
        return "\n".join(f"- {topic}" for topic in topics)
    recent = sources_pool.get_recent(10)
    lines: list[str] = []
    for source in recent:
        title = source.get("title") or source.get("url") or str(source)[:100] if isinstance(source, dict) else str(source)[:100]
        lines.append(f"- {title}")
    return "\n".join(lines) or "(no sources collected yet)"


_REFLECTOR_MAX_OBSERVATIONS = 5
_REFLECTOR_MAX_CONTEXT_CHARS = 2000


def format_all_observations(pools: dict[str, Any]) -> str:
    obs_pool = pools.get("observations")
    if not obs_pool or obs_pool.count() == 0:
        return "(no observations yet)"
    total = obs_pool.count()
    recent = obs_pool.get_recent(_REFLECTOR_MAX_OBSERVATIONS)
    lines: list[str] = []
    topics = obs_pool.topics()
    if topics:
        lines.append(f"Topics covered ({len(topics)}): {', '.join(topics[:15])}")
    if total > _REFLECTOR_MAX_OBSERVATIONS:
        lines.append(f"({total - _REFLECTOR_MAX_OBSERVATIONS} earlier observations omitted)")
    start_idx = max(total - _REFLECTOR_MAX_OBSERVATIONS + 1, 1)
    for idx, obs in enumerate(recent, start_idx):
        lines.append(f"Observation {idx}: {obs_to_text(obs)[:300]}")
    result = "\n".join(lines)
    if len(result) > _REFLECTOR_MAX_CONTEXT_CHARS:
        result = result[:_REFLECTOR_MAX_CONTEXT_CHARS] + "\n...(truncated)"
    return result


def format_source_quality(pools: dict[str, Any]) -> str:
    sources_pool = pools.get("sources")
    if not sources_pool or sources_pool.count() == 0:
        return "(no sources yet)"
    items = sources_pool.get_recent(sources_pool.count())
    total = len(items)
    domains: set[str] = set()
    with_snippets = 0
    substantive = 0
    low_value = 0
    qualities: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            url = item.get("url", "")
            if url:
                try:
                    from urllib.parse import urlparse
                    domains.add(urlparse(url).netloc)
                except Exception:
                    pass
            if item.get("snippet"):
                with_snippets += 1
            status = str(item.get("admission_status", ""))
            if status == "accepted":
                substantive += 1
            elif status == "accepted_low_value":
                low_value += 1
            quality = str(item.get("evidence_quality", "unknown"))
            qualities[quality] = qualities.get(quality, 0) + 1
    quality_summary = ", ".join(f"{k}={v}" for k, v in sorted(qualities.items()))
    return f"{total} sources from {len(domains)} domains, {with_snippets}/{total} with evidence snippets, substantive={substantive}, low_value={low_value}, qualities: {quality_summary}"
