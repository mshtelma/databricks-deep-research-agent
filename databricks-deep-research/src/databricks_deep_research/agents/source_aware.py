"""Source-aware planning helpers for framework research loops."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceKind,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.agents.query_policy import (
    EvidenceContract,
    QueryPolicyRegistry,
    RetrievalIntent,
    RetrievalNeed,
    RetrievalOutcome,
)

logger = logging.getLogger(__name__)

_POLICY_REGISTRY = QueryPolicyRegistry()

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "into",
    "is", "it", "its", "latest", "most", "of", "on", "or", "our", "that", "the", "their",
    "this", "to", "what", "when", "which", "with",
}
_DOMAIN_TERMS = {
    "earnings", "revenue", "income", "profit", "margin", "guidance", "quarter", "quarterly",
    "transcript", "automation", "digital", "fulfillment", "sales", "eps", "financial",
    "architecture", "technical", "documentation", "policy", "runbook", "metrics", "growth",
}
_METRIC_TERMS = (
    "revenue", "net income", "operating margin", "gross margin", "eps", "sales", "growth",
)
_QUESTION_LEADINS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
)
_IMPERATIVE_LEADINS = (
    "compare",
    "analyze",
    "summarize",
    "show",
    "find",
    "explain",
    "list",
    "identify",
)
_EMPTY_RESULT_PATTERNS = (
    "no results found",
    "no results returned",
    "unable to retrieve",
    "timed out",
    "query failed",
    "error executing",
    "failed:",
)


@dataclass(frozen=True)
class NormalizedSourceHint:
    source_name: str
    source_type: str
    priority: int
    query_hint: str
    query_strategy: str
    reasoning: str


@dataclass(frozen=True)
class StepToolSelection:
    active_tools: list[ResearchTool]
    fallback_tools: list[ResearchTool]
    matched_hint_by_tool: dict[str, NormalizedSourceHint]
    reasons: dict[str, str]


@dataclass(frozen=True)
class PlannedToolArguments:
    arguments: dict[str, Any]
    original_query: str
    rewritten_query: str
    alternate_queries: list[str]
    strategy: str
    source_hint: NormalizedSourceHint | None


@dataclass(frozen=True)
class AdmittedToolResult:
    content: str
    accepted_sources: list[dict[str, Any]]
    rejected_sources: list[dict[str, Any]]
    raw_sources: list[dict[str, Any]]
    accepted_count: int
    rejected_count: int
    accepted_substantive_count: int = 0
    accepted_low_value_count: int = 0
    evidence_quality: str = "empty"
    failure_mode: str = "none"
    needs_adaptation: bool = False
    adaptation_hint: str = ""


def _is_web_search_tool(definition: ToolDefinition) -> bool:
    """True for web SEARCH tools (active lane), False for web CRAWL (fallback)."""
    kind = tool_source_kind(definition)
    if kind not in ("web", "web_search"):
        return False
    return "crawl" not in definition.name.lower()


def tool_source_kind(definition: ToolDefinition) -> str:
    """Return the tool's source kind, preferring explicit declaration over inference."""
    # Prefer explicit source_kind if set (not the default "builtin" for enterprise tools)
    source_kind = getattr(definition, "source_kind", None)
    if source_kind and source_kind != "builtin":
        return source_kind

    # Fallback: heuristic inference for tools without source_kind
    source_type = (definition.source_type or "").lower()
    metadata = definition.metadata or {}
    metadata_source_type = str(metadata.get("source_type", "")).lower()
    name = definition.name.lower()
    description = definition.description.lower()
    combined = " ".join([
        source_type,
        metadata_source_type,
        name,
        description,
        json.dumps(metadata, sort_keys=True).lower(),
    ])

    if "web_crawl" in combined:
        return "web_crawl"
    if "web_search" in combined:
        return "web_search"
    if "knowledge_assistant" in combined or "assistant" in name or "ask_" in name:
        return "knowledge_assistant"
    if "genie" in combined or "query_" in name:
        return "genie"
    if (
        "vector_search" in combined
        or "search_" in name
        or "vs_index" in combined
        or "semantic" in combined
    ):
        return "vector_search"
    if "file_search" in combined or "uploaded_file" in combined:
        return "uploaded_file"
    return source_type or metadata_source_type or "builtin"


def summarize_recent_observations(observations: list[Any], limit: int = 3) -> list[str]:
    """Normalize recent observations to plain text snippets."""
    recent = observations[-limit:]
    normalized: list[str] = []
    for item in recent:
        if isinstance(item, str):
            normalized.append(item[:500])
        elif isinstance(item, dict):
            normalized.append(str(item.get("content") or item.get("findings") or item)[:500])
        else:
            normalized.append(str(item)[:500])
    return normalized


def select_step_tools(
    tools: list[ResearchTool],
    current_step: Any | None,
) -> StepToolSelection:
    """Split tools into initially active and fallback sets for a step."""
    if current_step is None:
        return StepToolSelection(
            active_tools=tools,
            fallback_tools=[],
            matched_hint_by_tool={},
            reasons={tool.definition.name: "no current step" for tool in tools},
        )

    hints = _normalize_source_hints(current_step)
    excludes = _normalize_excluded_sources(current_step)
    step_text = _step_text(current_step)

    active_tools: list[ResearchTool] = []
    fallback_tools: list[ResearchTool] = []
    matched_hint_by_tool: dict[str, NormalizedSourceHint] = {}
    reasons: dict[str, str] = {}
    heuristic_scores: dict[str, int] = {}

    for tool in tools:
        definition = tool.definition
        signature = _tool_signature(definition)
        if excludes and any(excluded in signature for excluded in excludes):
            reasons[definition.name] = "excluded by step"
            continue

        matched_hint = _match_hint(definition, hints)
        if matched_hint is not None:
            matched_hint_by_tool[definition.name] = matched_hint
            reasons[definition.name] = (
                f"matched hint {matched_hint.source_name} "
                f"(priority={matched_hint.priority})"
            )
            if matched_hint.priority <= 2:
                active_tools.append(tool)
            else:
                fallback_tools.append(tool)
            continue

        heuristic_score = _score_tool_for_step(definition, step_text)
        heuristic_scores[definition.name] = heuristic_score
        if hints:
            if _is_web_search_tool(definition):
                active_tools.append(tool)
                reasons[definition.name] = "web search kept active alongside enterprise hints"
            elif heuristic_score > 0:
                reasons[definition.name] = f"unhinted heuristic score={heuristic_score}"
                fallback_tools.append(tool)
            else:
                reasons[definition.name] = "hidden until fallback"
            continue

        if heuristic_score > 0:
            active_tools.append(tool)
            reasons[definition.name] = f"heuristic score={heuristic_score}"
        else:
            fallback_tools.append(tool)
            reasons[definition.name] = "low heuristic score"

    def _tool_sort_key(tool: ResearchTool) -> tuple[int, int, int, str]:
        definition = tool.definition
        hint = matched_hint_by_tool.get(definition.name)
        hint_bucket = 0 if hint is not None else 1
        hint_priority = hint.priority if hint is not None else 99
        heuristic_score = heuristic_scores.get(definition.name, _score_tool_for_step(definition, step_text))
        return (hint_bucket, hint_priority, -heuristic_score, definition.name)

    active_tools = sorted(active_tools, key=_tool_sort_key)
    fallback_tools = sorted(fallback_tools, key=_tool_sort_key)

    if not active_tools and fallback_tools:
        active_tools = fallback_tools
        fallback_tools = []
        for tool in active_tools:
            reasons[tool.definition.name] = reasons.get(tool.definition.name, "defaulted to active")

    if not active_tools and not fallback_tools:
        active_tools = tools
        reasons = {tool.definition.name: "step filtering kept no tools; using all tools" for tool in tools}

    return StepToolSelection(
        active_tools=active_tools,
        fallback_tools=fallback_tools,
        matched_hint_by_tool=matched_hint_by_tool,
        reasons=reasons,
    )




def infer_retrieval_need(
    current_step: Any | None,
    root_query: str,
    background_summary: str = "",
    recent_observations: list[str] | None = None,
) -> RetrievalNeed:
    step_text = _step_text(current_step, fallback=root_query)
    step_title = str(_step_value(current_step, "title", "") or "")
    recent = list(recent_observations or [])
    entities = [match for match in re.findall(r"\b[A-Z][A-Za-z0-9&.-]+\b", f"{root_query} {step_text}") if match.lower() not in _QUESTION_LEADINS]
    focus_terms = _focus_terms(root_query, step_text, background_summary, *recent)[:12]
    phrases = []
    for text in [root_query, step_text, background_summary]:
        phrases.extend(_extract_phrases(text))

    lowered = f"{root_query} {step_text}".lower()
    intent = RetrievalIntent.fact_lookup
    contract = EvidenceContract.ranked_sources
    if any(term in lowered for term in ("revenue", "growth", "sales", "eps", "margin", "kpi", "metric")):
        intent = RetrievalIntent.metric_slice
        contract = EvidenceContract.numeric_table
    elif any(term in lowered for term in ("transcript", "quote", "document", "manual", "runbook", "policy")):
        intent = RetrievalIntent.quote_extraction if "quote" in lowered or "transcript" in lowered else RetrievalIntent.document_retrieval
        contract = EvidenceContract.quoted_document_content
    elif any(term in lowered for term in ("compare", "comparison", "benchmark", "industry", "public", "trend")):
        intent = RetrievalIntent.comparison
        contract = EvidenceContract.ranked_sources
    elif any(term in lowered for term in ("why", "how", "explain")):
        intent = RetrievalIntent.explanatory_qa
        contract = EvidenceContract.narrative_answer

    metadata_only_acceptable = contract == EvidenceContract.metadata_only_ok
    requires_public = any(term in lowered for term in ("public", "industry", "benchmark", "external", "compare"))
    requires_enterprise = any(term in lowered for term in ("internal", "our", "warehouse", "runbook", "transcript", "genie", "vector")) or not requires_public

    return RetrievalNeed(
        root_query=root_query,
        step_text=step_text,
        step_title=step_title,
        entities=entities[:5],
        focus_terms=focus_terms,
        phrases=phrases[:8],
        intent=intent,
        evidence_contract=contract,
        time_scope="recent" if any(term in lowered for term in ("recent", "current", "latest", "2024", "2025")) else "",
        comparison_target="public industry trends" if requires_public else "",
        requires_public_sources=requires_public,
        requires_enterprise_sources=requires_enterprise,
        metadata_only_acceptable=metadata_only_acceptable,
        source_hints=[hint.__dict__ for hint in _normalize_source_hints(current_step)],
        recent_observation_summary=recent[-3:],
    )

def plan_tool_arguments(
    definition: ToolDefinition,
    arguments: dict[str, Any],
    *,
    current_step: Any | None,
    root_query: str,
    background_summary: str,
    recent_observations: list[str],
) -> PlannedToolArguments:
    """Rewrite tool arguments using source-aware heuristics."""
    query_key = _query_argument_key(definition, arguments)
    original_query = str(arguments.get(query_key, "")).strip()
    if not original_query:
        return PlannedToolArguments(
            arguments=dict(arguments),
            original_query="",
            rewritten_query="",
            alternate_queries=[],
            strategy="noop",
            source_hint=None,
        )

    source_hint = _match_hint(definition, _normalize_source_hints(current_step))
    need = infer_retrieval_need(
        current_step,
        root_query,
        background_summary=background_summary,
        recent_observations=recent_observations,
    )
    kind = tool_source_kind(definition)
    plan = _POLICY_REGISTRY.plan(kind, definition, need, dict(arguments))
    updated = dict(plan.arguments)
    alternates: list[str] = []
    for alternate in plan.alternate_argument_sets:
        if query_key in alternate and alternate[query_key]:
            alternates.append(str(alternate[query_key]))
    if alternates:
        updated["_alternate_queries"] = alternates
    return PlannedToolArguments(
        arguments=updated,
        original_query=original_query,
        rewritten_query=plan.rendered_query_text or str(updated.get(query_key, original_query)),
        alternate_queries=alternates,
        strategy=plan.query_strategy,
        source_hint=source_hint,
    )


def admit_tool_result(
    definition: ToolDefinition,
    result: ToolResult,
    *,
    current_step: Any | None,
    root_query: str,
) -> AdmittedToolResult:
    """Keep only step-relevant sources and sanitize tool content for the LLM."""
    raw_sources = [_normalize_source(source, definition) for source in result.sources]
    raw_sources = [source for source in raw_sources if source is not None]

    if not raw_sources and result.success and result.content.strip():
        synthetic = _build_synthetic_source(definition, result.content)
        if synthetic is not None and not _tool_result_is_empty_or_error(result):
            raw_sources = [synthetic]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    profile = _build_query_profile(current_step, root_query)
    logger.info(
        "ADMISSION_PROFILE tool=%s source_kind=%s step_title=%r root_query=%r "
        "profile_terms=%s profile_phrases=%s raw_source_count=%d",
        definition.name,
        tool_source_kind(definition),
        _step_text(current_step)[:200],
        root_query[:200],
        profile["terms"][:8],
        profile["phrases"][:4],
        len(raw_sources),
    )
    if not profile["terms"] and not profile["phrases"]:
        is_valid = result.success and not _tool_result_is_empty_or_error(result)
        early_accepted = raw_sources if is_valid else []
        # Classify early-accepted sources by content quality
        early_substantive = []
        early_low = []
        for src in early_accepted:
            ct = str(src.get("content") or src.get("snippet") or "")
            if len(ct.strip()) > 80:
                early_substantive.append(src)
            else:
                early_low.append(src)
        if early_substantive:
            early_eq = "full_text"
        elif early_low:
            early_eq = "snippet_only"
        elif early_accepted:
            early_eq = "metadata_only"
        else:
            early_eq = "empty"
        return AdmittedToolResult(
            content=result.content,
            accepted_sources=early_accepted,
            rejected_sources=[] if is_valid else raw_sources,
            raw_sources=raw_sources,
            accepted_count=len(early_accepted),
            rejected_count=0 if is_valid else len(raw_sources),
            accepted_substantive_count=len(early_substantive),
            accepted_low_value_count=len(early_low),
            evidence_quality=early_eq,
        )

    for source in raw_sources:
        score, reason = _score_source_relevance(source, profile)
        source["admission_score"] = score
        source["admission_reason"] = reason
        if _should_accept_source(definition, result, source, score):
            accepted.append(source)
        else:
            rejected.append(source)

    if accepted:
        content = _format_admitted_sources(definition.name, accepted)
    elif rejected:
        rejected_titles = ", ".join(
            source.get("title") or source.get("url", "")
            for source in rejected[:5]
        )
        content = (
            f"No relevant results accepted from {definition.name} for this step. "
            f"Rejected titles: {rejected_titles}"
        )
    else:
        content = result.content

    # Classify accepted sources by content quality
    substantive = []
    low_value_list = []
    for src in accepted:
        content_text = str(src.get("content") or src.get("snippet") or "")
        if len(content_text.strip()) > 80:
            substantive.append(src)
        else:
            low_value_list.append(src)

    if substantive:
        eq = "full_text"
    elif low_value_list:
        eq = "snippet_only"
    elif accepted:
        eq = "metadata_only"
    else:
        eq = "empty"

    return AdmittedToolResult(
        content=content,
        accepted_sources=accepted,
        rejected_sources=rejected,
        raw_sources=raw_sources,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_substantive_count=len(substantive),
        accepted_low_value_count=len(low_value_list),
        evidence_quality=eq,
    )


def sources_match_query(query: str, sources: list[Any]) -> bool:
    """Return whether at least one source overlaps meaningfully with the query."""
    profile = _build_query_profile(query, query)
    for source in sources:
        normalized = source if isinstance(source, dict) else _normalize_source(source, ToolDefinition("", "", {}))
        if not normalized:
            continue
        score, _reason = _score_source_relevance(normalized, profile)
        if score >= 2:
            return True
    return False


def _normalize_source_hints(current_step: Any | None) -> list[NormalizedSourceHint]:
    raw_hints = _step_value(current_step, "source_hints", [])
    normalized: list[NormalizedSourceHint] = []
    if not isinstance(raw_hints, list):
        return normalized
    for item in raw_hints:
        if item is None:
            continue
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source_name", "")).strip()
        source_type = str(item.get("source_type", "")).strip()
        if not source_name and not source_type:
            continue
        normalized.append(NormalizedSourceHint(
            source_name=source_name,
            source_type=source_type,
            priority=int(item.get("priority", 2) or 2),
            query_hint=str(item.get("query_hint", "") or ""),
            query_strategy=str(item.get("query_strategy", "") or ""),
            reasoning=str(item.get("reasoning", "") or ""),
        ))
    return normalized


def _normalize_excluded_sources(current_step: Any | None) -> set[str]:
    raw_excludes = _step_value(current_step, "exclude_sources", [])
    if not isinstance(raw_excludes, list):
        return set()
    return {_normalize_text(str(item)) for item in raw_excludes if str(item).strip()}


def _step_value(current_step: Any | None, key: str, default: Any = None) -> Any:
    if current_step is None:
        return default
    if isinstance(current_step, dict):
        return current_step.get(key, default)
    return getattr(current_step, key, default)


def _step_text(current_step: Any | None, fallback: str = "") -> str:
    if current_step is None:
        return fallback
    title = str(_step_value(current_step, "title", "") or "")
    description = str(_step_value(current_step, "description", "") or "")
    if title and description:
        return f"{title}. {description}"
    return title or description or fallback


def _tool_signature(definition: ToolDefinition) -> str:
    metadata = definition.metadata or {}
    values = [
        definition.name,
        definition.description,
        definition.source_type,
        str(metadata.get("source_name", "")),
        str(metadata.get("source_description", "")),
        str(metadata.get("index_name", "")),
    ]
    return _normalize_text(" ".join(values))


def _match_hint(
    definition: ToolDefinition,
    hints: list[NormalizedSourceHint],
) -> NormalizedSourceHint | None:
    if not hints:
        return None
    signature = _tool_signature(definition)
    kind = tool_source_kind(definition)
    for hint in hints:
        if hint.source_name and _normalize_text(hint.source_name) in signature:
            return hint
    for hint in hints:
        if not hint.source_name and hint.source_type and _normalize_text(hint.source_type) == _normalize_text(kind):
            return hint
    return None


def _score_tool_for_step(definition: ToolDefinition, step_text: str) -> int:
    signature = _tool_signature(definition)
    tokens = _extract_tokens(step_text)
    score = 0
    for token in tokens:
        if token in signature:
            score += 1

    kind = tool_source_kind(definition)
    lowered_step = step_text.lower()
    if kind == "genie" and any(
        term in lowered_step
        for term in (
            "metric", "kpi", "revenue", "sales", "growth", "cloud",
            "product line", "product lines", "analytics", "financial",
        )
    ):
        score += 2
    if kind == "vector_search" and any(
        term in lowered_step
        for term in (
            "document", "transcript", "policy", "earnings", "release",
            "architecture", "deployment", "pipeline", "runbook", "infrastructure",
        )
    ):
        score += 2
    if kind == "knowledge_assistant" and any(term in lowered_step for term in ("why", "how", "explain", "reason")):
        score += 1
    if _is_web_search_tool(definition) and any(
        term in lowered_step
        for term in (
            "compare", "comparison", "benchmark", "industry",
            "public", "trend", "market", "external",
        )
    ):
        score += 2
    return score


def _query_argument_key(definition: ToolDefinition, arguments: dict[str, Any]) -> str:
    if "query" in arguments:
        return "query"
    if "question" in arguments:
        return "question"
    properties = definition.parameters.get("properties", {})
    if "query" in properties:
        return "query"
    if "question" in properties:
        return "question"
    return next(iter(arguments.keys()), "query")


def _compact_web_query(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9$.\-]+", text)
    compact = " ".join(words[:14])
    return compact or text[:120]


def _build_vector_queries(
    *,
    base_query: str,
    root_query: str,
    step_text: str,
    definition: ToolDefinition,
    background_summary: str,
) -> tuple[str, list[str]]:
    subject = _preferred_subject(base_query, root_query, step_text)
    doc_target = _vector_doc_target(definition)
    focus_terms = _focus_terms(base_query, step_text, root_query)
    focus_text = " ".join(focus_terms[:6]) if focus_terms else base_query
    primary = _compact_vector_query(f"{subject} {doc_target} {focus_text}")

    alternates = [
        _compact_vector_query(f"{subject} {focus_text}"),
        _compact_vector_query(f"{subject} {doc_target}"),
    ]
    if "transcript" in _tool_signature(definition):
        alternates.append(_compact_vector_query(
            f"{subject} earnings call transcript management commentary {focus_text}"
        ))
    if background_summary:
        summary_terms = " ".join(_focus_terms(background_summary, base_query)[:4])
        if summary_terms:
            alternates.append(_compact_vector_query(f"{subject} {summary_terms}"))

    seen: set[str] = set()
    deduped: list[str] = []
    for query in alternates:
        normalized = _normalize_text(query)
        if normalized and normalized not in seen and normalized != _normalize_text(primary):
            seen.add(normalized)
            deduped.append(query)
    return primary, deduped[:4]


def _build_genie_question(text: str, root_query: str) -> str:
    cleaned = _normalize_question(text or root_query)
    return cleaned or _normalize_question(root_query)


def _build_assistant_question(text: str, recent_observations: list[str]) -> str:
    prompt = text.strip().rstrip("?")
    if recent_observations:
        prior = recent_observations[-1][:180].rstrip(".")
        return f"Based on the prior finding '{prior}', what explains {prompt}?"
    return f"What explains {prompt}?"


def _vector_doc_target(definition: ToolDefinition) -> str:
    signature = _tool_signature(definition)
    if "earnings" in signature:
        return "quarterly earnings release investor results"
    if "transcript" in signature:
        return "earnings call transcript management commentary"
    if "policy" in signature:
        return "internal policy process document"
    if "knowledge" in signature or "manual" in signature:
        return "internal knowledge base manual"
    return "internal document"


def _preferred_subject(*texts: str) -> str:
    for text in texts:
        cleaned = _strip_query_leadins(text)
        matches = [
            match
            for match in re.findall(r"\b[A-Z][A-Za-z0-9&.-]+\b", cleaned)
            if match.lower() not in _QUESTION_LEADINS
            and match.lower() not in _IMPERATIVE_LEADINS
        ]
        if matches:
            return " ".join(matches[:3])
    tokens = _extract_tokens(" ".join(_strip_query_leadins(text) for text in texts))
    return " ".join(tokens[:4]) or "the target topic"


def _focus_terms(*texts: str) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for token in _extract_tokens(text):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _extract_tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9$]{2,}", lowered)
    return [
        token for token in tokens
        if token not in _STOPWORDS and (token in _DOMAIN_TERMS or len(token) >= 4 or token.isdigit())
    ]


def _normalize_source(source: Any, definition: ToolDefinition) -> dict[str, Any] | None:
    if source is None:
        return None

    if hasattr(source, "model_dump"):
        source = source.model_dump()
    elif hasattr(source, "__dict__") and not isinstance(source, dict):
        source = {
            "url": getattr(source, "url", ""),
            "title": getattr(source, "title", ""),
            "snippet": getattr(source, "snippet", ""),
            "content": getattr(source, "content", None),
            "source_type": getattr(source, "source_type", None),
            "source_kind": getattr(source, "source_kind", None),
            "relevance_score": getattr(source, "relevance_score", None),
        }

    if not isinstance(source, dict):
        return None

    url = str(source.get("url", "") or "").strip()
    if not url:
        return None

    metadata = definition.metadata or {}
    content = source.get("content")
    if content is not None and not content:
        logger.debug(
            "NORMALIZE_SOURCE_EMPTY_CONTENT url=%s title=%s",
            url[:200],
            str(source.get("title", ""))[:120],
        )
    explicit_source_kind = source.get("source_kind")
    inferred_source_kind = tool_source_kind(definition)
    source_type = source.get("source_type") or source.get("type") or definition.source_type
    if source_type == "builtin":
        source_type = definition.source_type
    source_kind = explicit_source_kind or inferred_source_kind
    if source_kind == "enterprise":
        source_kind = inferred_source_kind

    normalized = {
        "url": url,
        "title": str(source.get("title", "") or source.get("filename", "") or url),
        "snippet": str(source.get("snippet", "") or source.get("highlight", "") or "")[:800],
        "content": str(content)[:20000] if content else None,
        "source_type": source_type,
        "source_kind": source_kind,
        "source_name": source.get("source_name") or metadata.get("source_name") or definition.name,
        "source_description": metadata.get("source_description") or definition.description,
        "backend": metadata.get("backend") or "framework",
        "index_name": metadata.get("index_name"),
        "relevance_score": source.get("relevance_score"),
    }
    return normalized


def _build_synthetic_source(definition: ToolDefinition, content: str) -> dict[str, Any] | None:
    if not content.strip():
        return None
    metadata = definition.metadata or {}
    return {
        "url": metadata.get("source_url") or f"enterprise://{definition.name}",
        "title": metadata.get("source_name") or definition.name,
        "snippet": content[:800],
        "content": content[:20000],
        "source_type": definition.source_type,
        "source_kind": tool_source_kind(definition),
        "source_name": metadata.get("source_name") or definition.name,
        "source_description": metadata.get("source_description") or definition.description,
        "backend": metadata.get("backend") or "framework",
        "index_name": metadata.get("index_name"),
    }


def _build_query_profile(current_step: Any | None, root_query: str) -> dict[str, Any]:
    step_text = _step_text(current_step)
    hint_queries = [
        hint.query_hint for hint in _normalize_source_hints(current_step)
        if hint.query_hint
    ]
    terms = _focus_terms(root_query, step_text, *hint_queries)
    phrases: list[str] = []
    for text in [root_query, step_text, *hint_queries]:
        phrases.extend(_extract_phrases(text))
    return {
        "terms": terms[:12],
        "phrases": phrases[:8],
    }


_ENTERPRISE_SOURCE_KINDS = frozenset({
    SourceKind.vector_index,
    SourceKind.sql_analytics,
    SourceKind.qa_assistant,
    # Keep backward compat with old string values
    "vector_search",
    "genie",
    "knowledge_assistant",
})


def _score_source_relevance(source: dict[str, Any], profile: dict[str, Any]) -> tuple[int, str]:
    text = " ".join(
        str(source.get(key, "") or "")
        for key in ("title", "snippet", "content")
    ).lower()
    terms = profile["terms"]
    phrases = profile["phrases"]
    matched_terms = [term for term in terms if term in text]
    matched_phrases = [phrase for phrase in phrases if phrase and phrase in text]
    score = min(3, len(matched_terms)) + (2 if matched_phrases else 0)
    text_terms = set(_extract_tokens(text))
    if matched_terms and text_terms.intersection(_DOMAIN_TERMS):
        score += 1

    # Enterprise sources carry a relevance_score from upstream semantic ranking
    # (vector similarity, Genie SQL match, etc.). Trust this signal instead of
    # relying solely on keyword overlap, which fails when semantically relevant
    # chunks don't literally contain the query terms.
    source_kind = source.get("source_kind", source.get("source_type", ""))
    relevance_score = source.get("relevance_score")
    enterprise_boost = 0
    if source_kind in _ENTERPRISE_SOURCE_KINDS and relevance_score is not None:
        try:
            if float(relevance_score) > 0:
                enterprise_boost = 2
        except (TypeError, ValueError):
            pass
    score += enterprise_boost

    reason_parts: list[str] = []
    if matched_terms:
        reason_parts.append(f"matched terms={matched_terms[:4]}")
    if matched_phrases:
        reason_parts.append(f"phrases={matched_phrases[:2]}")
    if enterprise_boost:
        reason_parts.append(f"enterprise_boost=+{enterprise_boost} (relevance_score={relevance_score})")
    reason = ", ".join(reason_parts) if reason_parts else "no meaningful overlap with step profile"

    logger.debug(
        "ADMISSION_SCORE_BREAKDOWN source_title=%r source_kind=%s "
        "relevance_score=%s text_len=%d matched_terms=%s matched_phrases=%s "
        "enterprise_boost=%d final_score=%d threshold=2",
        source.get("title", "")[:120],
        source_kind,
        relevance_score,
        len(text),
        matched_terms[:6],
        matched_phrases[:4],
        enterprise_boost,
        score,
    )

    return score, reason


def _should_accept_source(
    definition: ToolDefinition,
    result: ToolResult,
    source: dict[str, Any],
    score: int,
) -> bool:
    if not result.success or _tool_result_is_empty_or_error(result):
        return False

    source_kind = str(source.get("source_kind") or tool_source_kind(definition))
    relevance_score = source.get("relevance_score")

    if source_kind in {SourceKind.sql_analytics, SourceKind.qa_assistant, "genie", "knowledge_assistant"}:
        return True

    if source_kind in {SourceKind.vector_index, "vector_search"}:
        if score >= 2:
            return True
        try:
            return float(relevance_score or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    return score >= 2


def _tool_result_is_empty_or_error(result: ToolResult) -> bool:
    if not result.success:
        return True
    content = result.content.strip().lower()
    if not content:
        return True
    return any(pattern in content for pattern in _EMPTY_RESULT_PATTERNS)


def _extract_phrases(text: str) -> list[str]:
    lowered = text.lower()
    phrases = re.findall(r"(q[1-4]\s+\d{4}|[a-z0-9$]+\s+(?:earnings|transcript|revenue|guidance|automation|margin|eps))", lowered)
    unique: list[str] = []
    for phrase in phrases:
        if phrase not in unique:
            unique.append(phrase)
    return unique


def _format_admitted_sources(tool_name: str, sources: list[dict[str, Any]]) -> str:
    lines = [f"Accepted relevant results from {tool_name}:"]
    for source in sources[:5]:
        title = source.get("title") or source.get("url", "")
        snippet = source.get("snippet") or source.get("content") or ""
        lines.append(f"- {title}: {str(snippet)[:300]}")
    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _compact_vector_query(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9$.&-]+", text)
    return " ".join(words[:18]).strip()


def _strip_query_leadins(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(
        r"^(?:(?:what|how|why|when|where|which|who)\b\s*)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:(?:compare|analyze|summarize|show|find|explain|list|identify)\b\s+)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ,.:;") or text.strip()


def _normalize_question(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    if cleaned.endswith("?"):
        return cleaned
    return f"{cleaned.rstrip('.')}?"
