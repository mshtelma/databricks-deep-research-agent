from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from databricks_deep_research.tools.protocol import ToolDefinition, ToolResult


class RetrievalIntent(StrEnum):
    fact_lookup = "fact_lookup"
    metric_slice = "metric_slice"
    document_retrieval = "document_retrieval"
    quote_extraction = "quote_extraction"
    benchmark_search = "benchmark_search"
    comparison = "comparison"
    explanatory_qa = "explanatory_qa"


class EvidenceContract(StrEnum):
    numeric_table = "numeric_table"
    quoted_document_content = "quoted_document_content"
    ranked_sources = "ranked_sources"
    narrative_answer = "narrative_answer"
    metadata_only_ok = "metadata_only_ok"
    metadata_only_not_ok = "metadata_only_not_ok"


class EvidenceQuality(StrEnum):
    full_text = "full_text"
    snippet_only = "snippet_only"
    metadata_only = "metadata_only"
    availability_only = "availability_only"
    empty = "empty"


class FailureMode(StrEnum):
    none = "none"
    off_topic = "off_topic"
    low_relevance = "low_relevance"
    metadata_only = "metadata_only"
    availability_only = "availability_only"
    schema_only = "schema_only"
    duplicate_low_yield = "duplicate_low_yield"
    tool_mismatch = "tool_mismatch"
    empty_result = "empty_result"


@dataclass(frozen=True)
class RetrievalNeed:
    root_query: str
    step_text: str
    step_title: str
    entities: list[str] = field(default_factory=list)
    focus_terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    intent: RetrievalIntent = RetrievalIntent.fact_lookup
    evidence_contract: EvidenceContract = EvidenceContract.ranked_sources
    time_scope: str = ""
    comparison_target: str = ""
    requires_public_sources: bool = False
    requires_enterprise_sources: bool = False
    metadata_only_acceptable: bool = False
    source_hints: list[dict[str, Any]] = field(default_factory=list)
    recent_observation_summary: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryPlan:
    arguments: dict[str, Any]
    alternate_argument_sets: list[dict[str, Any]] = field(default_factory=list)
    query_strategy: str = "unchanged"
    rendered_query_text: str = ""
    debug_features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalOutcome:
    content: str
    raw_sources: list[dict[str, Any]] = field(default_factory=list)
    accepted_sources: list[dict[str, Any]] = field(default_factory=list)
    accepted_substantive_sources: list[dict[str, Any]] = field(default_factory=list)
    accepted_low_value_sources: list[dict[str, Any]] = field(default_factory=list)
    rejected_sources: list[dict[str, Any]] = field(default_factory=list)
    evidence_quality: EvidenceQuality = EvidenceQuality.empty
    failure_mode: FailureMode = FailureMode.none
    sufficient_for_step: bool = False
    needs_adaptation: bool = False
    adaptation_hint: str = ""
    admission_summary: dict[str, Any] = field(default_factory=dict)


class QueryPolicy(Protocol):
    def build_query_plan(
        self,
        definition: ToolDefinition,
        need: RetrievalNeed,
        raw_arguments: dict[str, Any],
    ) -> QueryPlan: ...

    def assess_result(
        self,
        definition: ToolDefinition,
        result: ToolResult,
        need: RetrievalNeed,
        raw_sources: list[dict[str, Any]],
    ) -> RetrievalOutcome: ...


class DefaultQueryPolicy:
    def build_query_plan(self, _definition: ToolDefinition, _need: RetrievalNeed, raw_arguments: dict[str, Any]) -> QueryPlan:
        query_text = str(raw_arguments.get("query") or raw_arguments.get("question") or "")
        return QueryPlan(arguments=dict(raw_arguments), rendered_query_text=query_text, query_strategy="unchanged")

    def assess_result(self, _definition: ToolDefinition, result: ToolResult, _need: RetrievalNeed, raw_sources: list[dict[str, Any]]) -> RetrievalOutcome:
        accepted = list(raw_sources) if result.success else []
        quality = EvidenceQuality.empty if not accepted else EvidenceQuality.full_text
        return RetrievalOutcome(
            content=result.content,
            raw_sources=raw_sources,
            accepted_sources=accepted,
            accepted_substantive_sources=accepted,
            rejected_sources=[] if result.success else raw_sources,
            evidence_quality=quality,
            failure_mode=FailureMode.none if accepted else FailureMode.empty_result,
            sufficient_for_step=bool(accepted),
            needs_adaptation=not bool(accepted),
            adaptation_hint="broaden retrieval" if not accepted else "",
            admission_summary={
                "accepted_substantive_count": len(accepted),
                "accepted_low_value_count": 0,
            },
        )


class WebSearchQueryPolicy(DefaultQueryPolicy):
    """Pass-through policy that trusts the react-loop LLM's search queries.

    The react-loop LLM has full research context and generates targeted queries.
    This policy passes them through with only length normalization.
    Regex-based entity extraction is intentionally removed — the LLM IS
    the query optimizer.
    """

    def build_query_plan(self, _definition: ToolDefinition, need: RetrievalNeed, raw_arguments: dict[str, Any]) -> QueryPlan:
        key = "query" if "query" in raw_arguments else "question"
        raw_query = str(raw_arguments.get(key, "")).strip()

        # Use the LLM's query directly — it has full research context
        query = raw_query or need.step_text or need.root_query

        # Length-cap only: search engines handle up to ~32 words well
        words = query.split()
        if len(words) > 32:
            query = " ".join(words[:32])

        updated = dict(raw_arguments)
        updated[key] = query

        return QueryPlan(
            arguments=updated,
            query_strategy="web_passthrough",
            rendered_query_text=query,
        )

    def assess_result(self, _definition: ToolDefinition, result: ToolResult, _need: RetrievalNeed, raw_sources: list[dict[str, Any]]) -> RetrievalOutcome:
        """Classify web search results by content quality, not just URL validity."""
        substantive, low_value, rejected = [], [], []

        for src in raw_sources:
            url = str(src.get("url", ""))
            snippet = str(src.get("snippet") or src.get("content") or "")

            if not url.startswith(("http://", "https://")):
                rejected.append(src)
            elif len(snippet.strip()) > 50:
                substantive.append(src)
            else:
                low_value.append(src)

        if substantive:
            quality = EvidenceQuality.snippet_only
        elif low_value:
            quality = EvidenceQuality.metadata_only
        else:
            quality = EvidenceQuality.empty

        return RetrievalOutcome(
            content=result.content,
            raw_sources=raw_sources,
            accepted_sources=substantive + low_value,
            accepted_substantive_sources=substantive,
            accepted_low_value_sources=low_value,
            rejected_sources=rejected,
            evidence_quality=quality,
            failure_mode=(
                FailureMode.none if substantive
                else FailureMode.low_relevance if low_value
                else FailureMode.empty_result
            ),
            sufficient_for_step=bool(substantive),
            needs_adaptation=not bool(substantive),
            adaptation_hint=(
                "try different query phrasing or broaden search terms"
                if not substantive else ""
            ),
            admission_summary={
                "accepted_substantive_count": len(substantive),
                "accepted_low_value_count": len(low_value),
            },
        )


class VectorIndexQueryPolicy(DefaultQueryPolicy):
    """Pass-through policy for VS tools: trust the react-loop LLM's queries.

    The react-loop LLM has full research context (step title, description,
    root query, previous observations). Its queries are informed and varied.
    This policy passes them through with word-count capping for embedding
    model compatibility.
    """

    _MAX_QUERY_WORDS = 40  # Embedding models handle ~512 tokens; cap conservatively

    def build_query_plan(self, _definition: ToolDefinition, need: RetrievalNeed, raw_arguments: dict[str, Any]) -> QueryPlan:
        key = "query" if "query" in raw_arguments else "question"
        raw_query = str(raw_arguments.get(key, "")).strip()

        # Trust the LLM's query — fall back to step context only if empty
        query = raw_query or need.step_text or need.root_query

        # Word-count cap for embedding model compatibility
        words = query.split()
        if len(words) > self._MAX_QUERY_WORDS:
            query = " ".join(words[: self._MAX_QUERY_WORDS])

        updated = dict(raw_arguments)
        updated[key] = query

        # Build alternates from step context for optional diversity
        alternates: list[dict[str, Any]] = []
        if need.entities and need.phrases:
            alt = " ".join(need.entities[:2] + need.phrases[:2]).strip()
            if alt and alt.lower() != query.lower():
                alternates.append({**updated, key: alt})

        return QueryPlan(
            arguments=updated,
            alternate_argument_sets=alternates[:2],
            query_strategy="vector_passthrough",
            rendered_query_text=query,
        )

    def assess_result(self, _definition: ToolDefinition, result: ToolResult, need: RetrievalNeed, raw_sources: list[dict[str, Any]]) -> RetrievalOutcome:
        accepted_substantive, accepted_low, rejected = [], [], []
        for src in raw_sources:
            content = str(src.get("content") or "").strip()
            snippet = str(src.get("snippet") or "").strip()
            if content and len(content) > 80:
                src["evidence_quality"] = EvidenceQuality.full_text.value
                src["admission_status"] = "accepted"
                src["admission_reason_code"] = "relevant_full_text"
                accepted_substantive.append(src)
            elif snippet:
                src["evidence_quality"] = EvidenceQuality.metadata_only.value
                src["admission_status"] = "accepted_low_value"
                src["admission_reason_code"] = "metadata_only_relevant"
                accepted_low.append(src)
            else:
                src["evidence_quality"] = EvidenceQuality.empty.value
                src["admission_status"] = "rejected"
                src["admission_reason_code"] = "empty_result"
                rejected.append(src)
        quality = EvidenceQuality.full_text if accepted_substantive else EvidenceQuality.metadata_only if accepted_low else EvidenceQuality.empty
        low_only = not accepted_substantive and bool(accepted_low)
        return RetrievalOutcome(
            content=result.content,
            raw_sources=raw_sources,
            accepted_sources=accepted_substantive + accepted_low,
            accepted_substantive_sources=accepted_substantive,
            accepted_low_value_sources=accepted_low,
            rejected_sources=rejected,
            evidence_quality=quality,
            failure_mode=FailureMode.metadata_only if low_only else FailureMode.none if accepted_substantive else FailureMode.empty_result,
            sufficient_for_step=bool(accepted_substantive) or (bool(accepted_low) and need.metadata_only_acceptable),
            needs_adaptation=not bool(accepted_substantive) and not need.metadata_only_acceptable,
            adaptation_hint="switch to exact artifact/entity query or quote-seeking retrieval" if low_only else "broaden vector retrieval" if not accepted_substantive else "",
            admission_summary={
                "accepted_substantive_count": len(accepted_substantive),
                "accepted_low_value_count": len(accepted_low),
            },
        )


class SqlAnalyticsQueryPolicy(DefaultQueryPolicy):
    def build_query_plan(self, _definition: ToolDefinition, need: RetrievalNeed, raw_arguments: dict[str, Any]) -> QueryPlan:
        key = "question" if "question" in raw_arguments else "query"
        measures = [term for term in need.focus_terms if term in {"revenue", "growth", "sales", "eps", "margin", "profit"}]
        subject = " ".join(need.entities[:3]) or need.root_query
        query = f"Show {', '.join(measures) if measures else 'the key metrics'} for {subject}".strip()
        if need.time_scope:
            query += f" for {need.time_scope}"
        updated = dict(raw_arguments)
        updated[key] = query
        alternates = [{**updated, key: f"Break down {', '.join(measures) if measures else 'metrics'} by period for {subject}".strip()}]
        return QueryPlan(arguments=updated, alternate_argument_sets=alternates, query_strategy="genie_metric_slice", rendered_query_text=query)

    def assess_result(self, _definition: ToolDefinition, result: ToolResult, _need: RetrievalNeed, raw_sources: list[dict[str, Any]]) -> RetrievalOutcome:
        content = result.content.lower()
        availability_only = "data exists" in content or "accepted" in content or "available" in content
        has_numbers = any(ch.isdigit() for ch in result.content)
        accepted_substantive = list(raw_sources) if has_numbers and raw_sources else []
        accepted_low = list(raw_sources) if availability_only and raw_sources and not accepted_substantive else []
        quality = EvidenceQuality.full_text if accepted_substantive else EvidenceQuality.availability_only if accepted_low else EvidenceQuality.empty
        return RetrievalOutcome(
            content=result.content,
            raw_sources=raw_sources,
            accepted_sources=accepted_substantive + accepted_low,
            accepted_substantive_sources=accepted_substantive,
            accepted_low_value_sources=accepted_low,
            rejected_sources=[] if (accepted_substantive or accepted_low) else raw_sources,
            evidence_quality=quality,
            failure_mode=FailureMode.none if accepted_substantive else FailureMode.availability_only if accepted_low else FailureMode.empty_result,
            sufficient_for_step=bool(accepted_substantive),
            needs_adaptation=not bool(accepted_substantive),
            adaptation_hint="add concrete metrics, grouping, or time filters" if not accepted_substantive else "",
            admission_summary={
                "accepted_substantive_count": len(accepted_substantive),
                "accepted_low_value_count": len(accepted_low),
            },
        )


class QaAssistantQueryPolicy(DefaultQueryPolicy):
    def build_query_plan(self, _definition: ToolDefinition, need: RetrievalNeed, raw_arguments: dict[str, Any]) -> QueryPlan:
        key = "question" if "question" in raw_arguments else "query"
        query = str(raw_arguments.get(key) or need.step_text or need.root_query).strip().rstrip("?")
        if need.intent == RetrievalIntent.explanatory_qa:
            query = f"What explains {query}?"
        updated = dict(raw_arguments)
        updated[key] = query
        return QueryPlan(arguments=updated, query_strategy="assistant_explanatory_followup", rendered_query_text=query)

    def assess_result(self, _definition: ToolDefinition, result: ToolResult, _need: RetrievalNeed, raw_sources: list[dict[str, Any]]) -> RetrievalOutcome:
        accepted = list(raw_sources) if raw_sources else []
        quality = EvidenceQuality.full_text if accepted else EvidenceQuality.empty
        return RetrievalOutcome(
            content=result.content,
            raw_sources=raw_sources,
            accepted_sources=accepted,
            accepted_substantive_sources=accepted,
            accepted_low_value_sources=[],
            rejected_sources=[] if accepted else raw_sources,
            evidence_quality=quality,
            failure_mode=FailureMode.none if accepted else FailureMode.empty_result,
            sufficient_for_step=bool(accepted),
            needs_adaptation=not bool(accepted),
            adaptation_hint="ask a narrower explanatory question" if not accepted else "",
            admission_summary={
                "accepted_substantive_count": len(accepted),
                "accepted_low_value_count": 0,
            },
        )


class QueryPolicyRegistry:
    def __init__(self) -> None:
        self._default = DefaultQueryPolicy()
        self._policies = {
            "web": WebSearchQueryPolicy(),
            "web_search": WebSearchQueryPolicy(),
            "vector_index": VectorIndexQueryPolicy(),
            "vector_search": VectorIndexQueryPolicy(),
            "sql_analytics": SqlAnalyticsQueryPolicy(),
            "genie": SqlAnalyticsQueryPolicy(),
            "qa_assistant": QaAssistantQueryPolicy(),
            "knowledge_assistant": QaAssistantQueryPolicy(),
        }

    def resolve(self, source_kind: str) -> QueryPolicy:
        return self._policies.get(source_kind, self._default)

    def plan(self, source_kind: str, definition: ToolDefinition, need: RetrievalNeed, raw_arguments: dict[str, Any]) -> QueryPlan:
        return self.resolve(source_kind).build_query_plan(definition, need, raw_arguments)

    def assess(self, source_kind: str, definition: ToolDefinition, result: ToolResult, need: RetrievalNeed, raw_sources: list[dict[str, Any]]) -> RetrievalOutcome:
        return self.resolve(source_kind).assess_result(definition, result, need, raw_sources)
