"""Semantic labels for Designer-generated workflow objects."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_MAX_LABEL_CHARS = 72

_ROLE_LABELS: dict[str, str] = {
    "coordinator": "Coordinator",
    "planner": "Planner",
    "researcher": "Researcher",
    "reflector": "Reflector",
    "synthesizer": "Synthesizer",
    "background": "Background Researcher",
}

_NODE_FALLBACKS: dict[str, str] = {
    "agent": "Workflow Agent",
    "tool": "Tool Step",
    "sequence": "Workflow Sequence",
    "parallel": "Parallel Workstreams",
    "loop": "Iteration Loop",
    "conditional": "Decision Branch",
    "subworkflow": "Subworkflow",
    "plan_and_execute": "Plan and Execute",
}

_ROLE_FALLBACKS: dict[str, str] = {
    "coordinator": "Research Coordinator",
    "planner": "Research Planner",
    "researcher": "Evidence Researcher",
    "reflector": "Coverage Reflector",
    "synthesizer": "Report Synthesizer",
    "background": "Background Researcher",
}

_GENERIC_LABEL_RE = re.compile(
    r"^(?:"
    r"agent|block|node|step|lane|researcher|planner|coordinator|reflector|"
    r"synthesizer|background|tool|sequence|parallel|loop|conditional|"
    r"subworkflow|plan(?:\s+and\s+execute)?"
    r")(?:\s*(?:#?\d+|[ivx]+|[a-z]))?$",
    flags=re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&/+-]*")
_FOCUS_PATTERNS = (
    re.compile(
        r"^\s*(?:lane focus|workstream|focus|topic|purpose|objective)\s*:\s*(.+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:investigate|research|analyze|analyse|audit|review|synthesize|"
        r"synthesise|produce|generate|classify|route)\s+(.+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    ),
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "then",
    "to",
    "with",
}


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip()


def _compact(value: str, *, max_chars: int = _MAX_LABEL_CHARS) -> str:
    cleaned = _clean_spaces(value).strip(" .,:;")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip(" .,:;") + "..."


def _is_generic_label(value: str) -> bool:
    normalized = _clean_spaces(value).casefold()
    return bool(_GENERIC_LABEL_RE.fullmatch(normalized))


def _title_from_text(value: str) -> str:
    words = _WORD_RE.findall(value)
    if not words:
        return ""
    selected: list[str] = []
    for word in words:
        if len(selected) >= 7:
            break
        lowered = word.casefold()
        if lowered in _STOPWORDS:
            continue
        selected.append(word)
    if not selected:
        return ""
    titled = " ".join(
        word if word.isupper() or any(ch.isdigit() for ch in word) else word.capitalize()
        for word in selected
    )
    return _compact(titled)


def _extract_focus(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern in _FOCUS_PATTERNS:
        match = pattern.search(text)
        if match:
            focus = match.group(1).strip()
            if "{" in focus:
                continue
            return focus
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line and "{" not in first_line and len(first_line.split()) <= 14:
        return first_line
    return ""


def _config_text_candidates(config: Mapping[str, Any]) -> list[str]:
    keys = (
        "description",
        "purpose",
        "objective",
        "output_key",
        "ref",
        "tool_name",
        "user_prompt_template",
        "system_prompt",
    )
    candidates: list[str] = []
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    return candidates


def semantic_lane_label(description: str, index: int) -> str:
    """Return a description-first label for a generated researcher lane."""
    title = _title_from_text(description)
    if title and not _is_generic_label(title):
        suffix = "" if "research" in title.casefold() else " Researcher"
        return _compact(f"{title}{suffix}")
    return f"Evidence Workstream {index} Researcher"


def semantic_node_label(
    *,
    node_type: str,
    config: Mapping[str, Any] | None,
    requested_label: str,
) -> str:
    """Keep meaningful labels and replace generic role/ordinal placeholders."""
    label = _compact(str(requested_label or ""))
    config_map = config or {}
    subtype = str(config_map.get("subtype") or "").strip().casefold()

    if label and not _is_generic_label(label):
        return label

    role = _ROLE_LABELS.get(subtype)
    for candidate in _config_text_candidates(config_map):
        focus = _extract_focus(candidate)
        title = _title_from_text(focus)
        if title and not _is_generic_label(title):
            if role and role.casefold() not in title.casefold():
                return _compact(f"{title} {role}")
            if node_type == "tool" and "tool" not in title.casefold():
                return _compact(f"{title} Tool")
            return title

    if subtype:
        return _ROLE_FALLBACKS.get(subtype, f"{subtype.replace('_', ' ').title()} Agent")
    return _NODE_FALLBACKS.get(node_type, "Workflow Object")


__all__ = ["semantic_lane_label", "semantic_node_label"]
