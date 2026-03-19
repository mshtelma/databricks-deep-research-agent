"""Synthesizer builtin -- final report generation with optional reclaim mode."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.grounding import resolve_grounding_mode
from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.agents.output_models import SynthesizerOutput
from databricks_deep_research.citation.analysis_grounding import (
    AnalysisGroundingVerifier,
)
from databricks_deep_research.citation.citation_corrector import (
    CitationCorrector,
    CitationCorrectorConfig,
)
from databricks_deep_research.citation.citation_keys import build_citation_key_map
from databricks_deep_research.citation.claim_generator import (
    GenerationMode as ClaimGenerationMode,
)
from databricks_deep_research.citation.claim_generator import (
    InterleavedGenerationConfig as ClaimGenerationConfig,
)
from databricks_deep_research.citation.claim_generator import (
    InterleavedGenerator,
)
from databricks_deep_research.citation.confidence_classifier import (
    ConfidenceClassifier,
    ConfidenceClassifierConfig,
)
from databricks_deep_research.citation.config import (
    CitationConfig,
    IsolatedVerificationConfig,
    VerificationRetrievalConfig,
)
from databricks_deep_research.citation.config import (
    GenerationMode as CitationGenerationMode,
)
from databricks_deep_research.citation.evidence_selector import (
    EvidenceSelectionConfig,
    EvidenceSelector,
)
from databricks_deep_research.citation.isolated_verifier import IsolatedVerifier
from databricks_deep_research.citation.numeric_verifier import (
    NumericVerifier,
    NumericVerifierConfig,
)
from databricks_deep_research.citation.pipeline import (
    CitationVerificationPipeline,
    VerificationEvent,
)
from databricks_deep_research.citation.types import ClaimInfo, VerificationSummaryInfo
from databricks_deep_research.citation.verification_retriever import (
    VerificationRetriever,
)
from databricks_deep_research.events.types import (
    CitationCorrectedEvent,
    ClaimGeneratedEvent,
    ClaimVerifiedEvent,
    NumericClaimDetectedEvent,
    StreamEvent,
    SynthesisStartedEvent,
    VerificationSummaryEvent,
)
from databricks_deep_research.workflow.state import WorkflowState
from databricks_deep_research.workflow.runtime_core.selectors import (
    select_analysis_summary,
    select_claims,
    select_verification_payload,
    select_verification_summary,
)

import re as _re

_PLACEHOLDER_TITLES = frozenset({
    "untitled", "unknown", "source", "n/a", "na", "none", "null", "",
})
_PLACEHOLDER_TITLE_PATTERNS = _re.compile(
    r"^(vector search result \d+|doc_\d+|row_\d+)$", _re.I,
)


def _is_placeholder_title(title: str) -> bool:
    """True if title carries no information value."""
    stripped = title.strip()
    return (
        not stripped
        or stripped.lower() in _PLACEHOLDER_TITLES
        or bool(_PLACEHOLDER_TITLE_PATTERNS.match(stripped))
        or len(stripped) < 3
    )


logger = logging.getLogger(__name__)

# Default max tool calls for synthesizer (pool search for evidence)
DEFAULT_MAX_TOOL_CALLS = 10

# Reclaim-mode specific constants
_RECLAIM_MAX_TOOL_CALLS = 5
_RECLAIM_TARGET_WORD_COUNT = 600
_RECLAIM_MAX_TOKENS = 2000


class _EvidenceSelectorAdapter:
    """Bridge the concrete EvidenceSelector to the pipeline protocol."""

    def __init__(self, selector: EvidenceSelector) -> None:
        self._selector = selector

    async def select_evidence_spans(
        self,
        query: str,
        sources: list[dict[str, Any]],
        max_spans_per_source: int,
    ) -> list[Any]:
        result = await self._selector.select_evidence(
            query,
            sources,
            max_spans_per_source=max_spans_per_source,
            filter_quality=False,
        )
        return result.evidence


def _is_reclaim_mode(config: AgentNodeConfig) -> bool:
    """Determine whether the synthesizer should run in reclaim mode."""
    return resolve_grounding_mode(config) == "reclaim"


def _get_reclaim_config(config: AgentNodeConfig) -> dict[str, Any]:
    """Extract reclaim-specific settings from ``output_schema``."""
    schema = config.output_schema or {}
    return {
        "target_word_count": schema.get(
            "target_word_count", _RECLAIM_TARGET_WORD_COUNT
        ),
        "max_tokens": schema.get("max_tokens", _RECLAIM_MAX_TOKENS),
        "generation_mode": schema.get("generation_mode", "strict"),
        "enable_are_retrieval": schema.get("enable_are_retrieval", False),
    }


def _build_citation_config(config: AgentNodeConfig) -> CitationConfig:
    """Convert synthesizer output_schema settings into a CitationConfig."""
    reclaim_cfg = _get_reclaim_config(config)
    generation_mode_raw = str(reclaim_cfg["generation_mode"]).lower()
    try:
        generation_mode = CitationGenerationMode(generation_mode_raw)
    except ValueError:
        logger.warning(
            "SYNTHESIZER_INVALID_GENERATION_MODE mode=%s fallback=strict",
            generation_mode_raw,
        )
        generation_mode = CitationGenerationMode.STRICT

    return CitationConfig(
        generation_mode=generation_mode,
        enable_verification_retrieval=bool(reclaim_cfg["enable_are_retrieval"]),
        isolated_verification=IsolatedVerificationConfig(
            verification_model_tier="bulk_analysis",
            quick_verification_tier="fast",
        ),
        verification_retrieval=VerificationRetrievalConfig(
            decomposition_tier="simple",
            entailment_tier="bulk_analysis",
            reconstruction_tier="analytical",
            softening_tier="fast",
        ),
    )


def _build_reclaim_system_prompt() -> str:
    """Return a specialised system prompt for reclaim mode."""
    return (
        "You are the Synthesizer agent for a deep research system operating "
        "in verified citation mode.\n\n"
        "## Core Principles\n\n"
        "1. ACCURACY: Every factual claim MUST reference evidence using "
        "``[N]`` markers where N is the evidence index.\n"
        "2. BREVITY: Prefer fewer, denser sentences over verbose explanations.\n"
        "3. GROUNDING: Do NOT state facts that lack evidence in the pool.\n"
        "4. HEDGING: When evidence is weak, hedge with \"reportedly\" or "
        "\"according to\".\n\n"
        "## Citation Format\n\n"
        "Use numbered markers: ``[1]``, ``[2]``, ``[1][3]``.\n"
        "Place markers immediately after the supported claim.\n"
        "Multiple markers for a single sentence are allowed and encouraged.\n\n"
        "## Report Structure\n\n"
        "- ## for main sections (2-3 max)\n"
        "- Bullet lists for key facts\n"
        "- Use markdown tables for comparisons\n\n"
        "## Writing Rules\n\n"
        "- Lead with the answer, not background.\n"
        "- One fact per sentence maximum.\n"
        "- No meta-commentary, filler, or follow-up offers.\n"
        "- End with substantive content.\n"
    )


def _build_reclaim_user_prompt() -> str:
    """Return a specialised user prompt template for reclaim mode."""
    return (
        "Create a verified research report based on the gathered observations.\n\n"
        "## Original Query\n{query}\n\n"
        "## Research Summary\n"
        "- Research depth: {research_depth}\n"
        "- Plan iterations: {plan_iterations}\n"
        "- Steps executed: {steps_executed}\n"
        "- Sources found: {sources_count}\n\n"
        "## All Research Observations\n{all_observations}\n\n"
        "## Available Sources\n{sources_list}\n\n"
        "## Background Discovery Sources (fallback only)\n{fallback_discovery_sources}\n\n"
        "## STRICT Length Requirement\n"
        "- Target: {min_words}-{max_words} words\n"
        "- Aim for the upper bound if content warrants it\n"
        "- DO NOT exceed {max_words} words\n\n"
        "## Instructions\n"
        "Create a well-structured markdown report that:\n"
        "1. Directly answers the user's query\n"
        "2. Synthesizes all relevant findings\n"
        "3. Uses NUMBERED citations [N] matching evidence indices\n"
        "4. Every factual claim must be backed by at least one [N] marker\n"
        "5. Omits obvious caveats unless critical\n\n"
        "Respond with the markdown report directly (no JSON wrapper)."
    )


def _normalize_source(source: Any) -> dict[str, Any] | None:
    """Normalize a pool source into a citation-pipeline-friendly dict."""
    if isinstance(source, dict):
        url = source.get("url")
        if not url:
            return None
        content = source.get("content")
        title = source.get("title") or source.get("filename") or ""
        snippet = source.get("snippet") or (
            content[:500] if isinstance(content, str) else ""
        )
        if not snippet and title and not _is_placeholder_title(str(title)):
            snippet = str(title)
        normalized = dict(source)
        normalized["url"] = str(url)
        normalized["canonical_url"] = source.get("canonical_url") or normalized["url"]
        normalized["snippet"] = snippet or ""
        normalized["content"] = content if isinstance(content, str) else ""
        normalized["title"] = title
        normalized["source_type"] = (
            source.get("source_type") or source.get("type") or "web"
        )
        return normalized

    url = getattr(source, "url", None)
    if not url:
        return None

    content = getattr(source, "content", None)
    title = getattr(source, "title", "") or ""
    snippet = getattr(source, "snippet", None) or (
        content[:500] if isinstance(content, str) else ""
    )
    if not snippet and title and not _is_placeholder_title(title):
        snippet = title
    return {
        "url": str(url),
        "canonical_url": getattr(source, "canonical_url", None) or str(url),
        "title": title,
        "snippet": snippet or "",
        "content": content if isinstance(content, str) else "",
        "source_type": getattr(source, "source_type", None)
        or getattr(source, "type", None)
        or "web",
        "source_kind": getattr(source, "source_kind", None),
        "relevance_score": getattr(source, "relevance_score", None),
    }


def _collect_sources(pools: dict[str, Any]) -> list[dict[str, Any]]:
    """Read normalized sources from the shared sources pool."""
    sources_pool = pools.get("sources")
    if sources_pool is None or sources_pool.count() == 0:
        return []

    normalized: list[dict[str, Any]] = []
    for item in sources_pool.get_recent(sources_pool.count()):
        source = _normalize_source(item)
        if source is not None:
            normalized.append(source)
    return normalized


def _stringify_observation(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("observation") or value.get("findings") or value.get("summary")
        if isinstance(text, str):
            return text.strip()
    return str(value).strip()


def _collect_observations(state: WorkflowState, pools: dict[str, Any]) -> list[str]:
    """Read findings from state first, then fall back to the observations pool."""
    observations: list[str] = []
    seen: set[str] = set()

    for item in state.get_all("findings"):
        text = _stringify_observation(item)
        if text and text not in seen:
            observations.append(text)
            seen.add(text)

    observations_pool = pools.get("observations")
    if observations_pool is not None and observations_pool.count() > 0:
        for item in observations_pool.get_recent(observations_pool.count()):
            text = _stringify_observation(item)
            if text and text not in seen:
                observations.append(text)
                seen.add(text)

    return observations


def _hydrate_sparse_sources(
    sources: list[dict[str, Any]],
    observations: list[str],
) -> list[dict[str, Any]]:
    """Fill in snippet/content for sources that arrived text-free."""
    if not sources or not observations:
        return sources

    hydrated: list[dict[str, Any]] = []
    obs_idx = 0
    for source in sources:
        has_text = bool(source.get("snippet") or source.get("content"))
        if has_text:
            hydrated.append(source)
        else:
            hydrated_source = dict(source)
            observation = observations[min(obs_idx, len(observations) - 1)]
            hydrated_source["snippet"] = observation[:500]
            if not hydrated_source.get("title"):
                hydrated_source["title"] = observation[:80]
            hydrated.append(hydrated_source)
            obs_idx += 1

    if obs_idx > 0:
        logger.warning(
            "SYNTHESIZER_RECLAIM_HYDRATED_SPARSE_SOURCES hydrated=%d total=%d",
            obs_idx,
            len(sources),
        )

    return hydrated


def _build_reclaim_pipeline(
    llm_client: Any,
    citation_config: CitationConfig,
) -> CitationVerificationPipeline:
    """Construct the framework-native citation verification pipeline."""
    evidence_cfg = citation_config.evidence_preselection
    generation_cfg = citation_config.interleaved_generation
    classifier_cfg = citation_config.confidence_classification
    correction_cfg = citation_config.citation_correction
    numeric_cfg = citation_config.numeric_qa_verification

    claim_generation_mode = ClaimGenerationMode.NATURAL
    if citation_config.generation_mode == CitationGenerationMode.STRICT:
        claim_generation_mode = ClaimGenerationMode.STRICT

    return CitationVerificationPipeline(
        llm_client,
        evidence_selector=_EvidenceSelectorAdapter(
            EvidenceSelector(
                llm_client,
                EvidenceSelectionConfig(
                    max_spans_per_source=evidence_cfg.max_spans_per_source,
                    min_span_length=evidence_cfg.min_span_length,
                    max_span_length=evidence_cfg.max_span_length,
                    relevance_threshold=evidence_cfg.relevance_threshold,
                    numeric_content_boost=evidence_cfg.numeric_content_boost,
                    chunk_size=evidence_cfg.chunk_size,
                    chunk_overlap=evidence_cfg.chunk_overlap,
                    max_chunks_per_source=evidence_cfg.max_chunks_per_source,
                ),
            )
        ),
        claim_generator=InterleavedGenerator(
            llm_client,
            ClaimGenerationConfig(
                min_evidence_similarity=generation_cfg.min_evidence_similarity,
                generation_mode=claim_generation_mode,
            ),
        ),
        confidence_classifier=ConfidenceClassifier(
            ConfidenceClassifierConfig(
                high_threshold=classifier_cfg.high_threshold,
                low_threshold=classifier_cfg.low_threshold,
                quote_match_bonus=classifier_cfg.quote_match_bonus,
                hedging_word_penalty=classifier_cfg.hedging_word_penalty,
            )
        ),
        isolated_verifier=IsolatedVerifier(
            llm_client,
            citation_config.isolated_verification,
        ),
        citation_corrector=CitationCorrector(
            llm_client,
            CitationCorrectorConfig(
                lambda_weight=correction_cfg.lambda_weight,
                correction_threshold=correction_cfg.correction_threshold,
                allow_alternate_citations=correction_cfg.allow_alternate_citations,
            ),
        ),
        numeric_verifier=NumericVerifier(
            llm_client,
            NumericVerifierConfig(
                rounding_tolerance=numeric_cfg.rounding_tolerance,
                answer_comparison_method=numeric_cfg.answer_comparison_method.value,
                require_unit_match=numeric_cfg.require_unit_match,
                require_entity_match=numeric_cfg.require_entity_match,
            ),
        ),
        analysis_grounding_verifier=AnalysisGroundingVerifier(
            llm_client,
            citation_config.grounding_validation,
        ),
        verification_retriever=VerificationRetriever(
            llm_client,
            trigger_on_verdicts=list(
                citation_config.verification_retrieval.trigger_on_verdicts
            ),
            max_atomic_facts_per_claim=(
                citation_config.verification_retrieval.max_atomic_facts_per_claim
            ),
            decomposition_timeout_seconds=(
                citation_config.verification_retrieval.decomposition_timeout_seconds
            ),
            max_searches_per_fact=(
                citation_config.verification_retrieval.max_searches_per_fact
            ),
            max_external_urls_per_search=(
                citation_config.verification_retrieval.max_external_urls_per_search
            ),
            entailment_threshold=(
                citation_config.verification_retrieval.entailment_threshold
            ),
            internal_search_threshold=(
                citation_config.verification_retrieval.internal_search_threshold
            ),
            softening_strategy=(
                citation_config.verification_retrieval.softening_strategy
            ),
            search_timeout_seconds=(
                citation_config.verification_retrieval.search_timeout_seconds
            ),
            crawl_timeout_seconds=(
                citation_config.verification_retrieval.crawl_timeout_seconds
            ),
            decomposition_tier=(
                citation_config.verification_retrieval.decomposition_tier
            ),
            entailment_tier=(
                citation_config.verification_retrieval.entailment_tier
            ),
            reconstruction_tier=(
                citation_config.verification_retrieval.reconstruction_tier
            ),
            softening_tier=citation_config.verification_retrieval.softening_tier,
        ),
        config=citation_config,
    )


def _build_url_to_index_map(sources: list[dict[str, Any]]) -> dict[str, str]:
    """Map source URLs to the numeric citation indices expected by the UI."""
    mapping: dict[str, str] = {}
    for index, source in enumerate(sources):
        numeric_index = str(index)
        for url in (source.get("url"), source.get("canonical_url")):
            if url:
                mapping[str(url)] = numeric_index
    return mapping


def _build_key_to_numeric_index_map(
    evidence_pool: list[Any],
    url_to_index: dict[str, str],
) -> dict[str, str]:
    """Map human-readable evidence keys back to numeric source indices."""
    key_to_numeric: dict[str, str] = {}
    if not evidence_pool:
        return key_to_numeric

    key_map = build_citation_key_map(evidence_pool)
    for evidence_index, key in key_map.items():
        if evidence_index >= len(evidence_pool):
            continue
        evidence = evidence_pool[evidence_index]
        source_pool_index = getattr(evidence, "source_pool_index", None)
        if isinstance(source_pool_index, int):
            numeric_index = str(source_pool_index)
        else:
            canonical_source_url = getattr(evidence, "canonical_source_url", None)
            source_url = (
                canonical_source_url
                if isinstance(canonical_source_url, str) and canonical_source_url
                else getattr(evidence, "source_url", "")
            )
            numeric_index = url_to_index.get(source_url)
        if numeric_index is not None:
            key_to_numeric[key] = numeric_index

    return key_to_numeric


def _replace_human_citations_with_numeric(
    report: str,
    key_to_numeric: dict[str, str],
) -> str:
    """Convert ``[Arxiv]``-style markers back to numeric ``[N]`` markers."""
    numeric_report = report
    for key in sorted(key_to_numeric, key=len, reverse=True):
        numeric_report = numeric_report.replace(
            f"[{key}]",
            f"[{key_to_numeric[key]}]",
        )
    return numeric_report


def _claim_to_state_dict(
    claim: ClaimInfo,
    url_to_index: dict[str, str],
    key_to_numeric: dict[str, str],
) -> dict[str, Any]:
    """Convert a framework ClaimInfo into a JSON-friendly state dict."""
    citation_keys: list[str] = []
    evidences = claim.evidences or ([claim.evidence] if claim.evidence else [])
    for evidence in evidences:
        if evidence is None:
            continue
        if isinstance(evidence.source_pool_index, int):
            numeric_index = str(evidence.source_pool_index)
        else:
            numeric_index = url_to_index.get(
                evidence.canonical_source_url or evidence.source_url
            )
        if numeric_index is not None and numeric_index not in citation_keys:
            citation_keys.append(numeric_index)

    if not citation_keys:
        citation_keys = [
            key_to_numeric.get(key, key)
            for key in (claim.citation_keys or ([claim.citation_key] if claim.citation_key else []))
        ]

    if not citation_keys and claim.evidence and claim.evidence.source_url:
        numeric_index = url_to_index.get(
            claim.evidence.canonical_source_url or claim.evidence.source_url
        )
        if numeric_index is not None:
            citation_keys = [numeric_index]

    return {
        "claim_text": claim.claim_text,
        "claim_type": claim.claim_type,
        "position_start": claim.position_start,
        "position_end": claim.position_end,
        "evidence": claim.evidence.to_dict() if claim.evidence else None,
        "evidences": [evidence.to_dict() for evidence in claim.evidences],
        "confidence_level": claim.confidence_level,
        "routing_confidence_score": claim.routing_confidence_score,
        "verification_verdict": claim.verification_verdict,
        "verification_confidence": claim.verification_confidence,
        "verification_reasoning": claim.verification_reasoning,
        "verification_method": claim.verification_method,
        "evidence_match_score": claim.evidence_match_score,
        "used_quick_verification": claim.used_quick_verification,
        "verification_latency_ms": claim.verification_latency_ms,
        "abstained": claim.abstained,
        "citation_key": citation_keys[0] if citation_keys else None,
        "citation_keys": citation_keys,
        "claim_role": claim.claim_role,
        "verification_text": claim.verification_text,
        "analysis_parent_claim_indices": claim.analysis_parent_claim_indices,
        "from_free_block": claim.from_free_block,
        "has_fallback_evidence": claim.has_fallback_evidence,
    }


def _recalculate_claim_positions(
    report: str,
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Best-effort claim position recalculation after citation normalization."""
    import re as _re

    clean_report: str | None = None  # Lazy-computed once on first miss
    cursor = 0
    for claim in claims:
        claim_text = claim.get("claim_text", "")
        if not claim_text:
            continue
        position = report.find(claim_text, cursor)
        if position < 0:
            position = report.find(claim_text)
        if position < 0:
            # Fuzzy: strip citation markers and retry
            if clean_report is None:
                clean_report = _re.sub(r"\s*\[[^\]]+\]", "", report)
            position = clean_report.find(claim_text)
        if position >= 0:
            claim["position_start"] = position
            claim["position_end"] = position + len(claim_text)
            cursor = position + len(claim_text)
    return claims


def _build_framework_summary(summary: VerificationSummaryInfo | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize summary data into the framework/app bridge contract."""
    if summary is None:
        return {}

    if isinstance(summary, VerificationSummaryInfo):
        total_claims = summary.total_claims
        verified_claims = summary.supported_count
        corrected_citations = summary.citation_corrections
        removed_claims = summary.contradicted_count
        softened_claims = summary.unsupported_count
        # Weighted confidence: supported=1.0, partial=0.5, exclude abstained
        non_abstained = total_claims - summary.abstained_count
        if non_abstained > 0:
            overall_confidence = (
                verified_claims + summary.partial_count * 0.5
            ) / non_abstained
        else:
            overall_confidence = 0.0
        return {
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "corrected_citations": corrected_citations,
            "removed_claims": removed_claims,
            "softened_claims": softened_claims,
            "overall_confidence": overall_confidence,
            "supported_count": summary.supported_count,
            "partial_count": summary.partial_count,
            "unsupported_count": summary.unsupported_count,
            "contradicted_count": summary.contradicted_count,
            "abstained_count": summary.abstained_count,
            "supported_rate": summary.supported_rate,
            "warning": summary.warning,
            "claim_revisions": summary.claim_revisions,
            "atomic_facts_total": summary.atomic_facts_total,
            "atomic_facts_verified": summary.atomic_facts_verified,
            "atomic_facts_softened": summary.atomic_facts_softened,
            "claims_fully_verified": summary.claims_fully_verified,
            "claims_partially_softened": summary.claims_partially_softened,
            "claims_fully_softened": summary.claims_fully_softened,
            "external_searches": summary.external_searches,
            "new_sources_added": summary.new_sources_added,
            "analysis_summary": summary.analysis_summary.to_dict(),
            "routing_summary": summary.routing_summary,
        }

    total_claims = int(summary.get("total_claims", 0))
    verified_claims = int(
        summary.get(
            "verified_claims",
            summary.get("supported", summary.get("supported_count", 0)),
        )
    )
    corrected_citations = int(
        summary.get(
            "corrected_citations",
            summary.get("citation_corrections", 0),
        )
    )
    removed_claims = int(
        summary.get(
            "removed_claims",
            summary.get("contradicted", summary.get("contradicted_count", 0)),
        )
    )
    softened_claims = int(
        summary.get(
            "softened_claims",
            summary.get("unsupported", summary.get("unsupported_count", 0)),
        )
    )
    overall_confidence_raw = summary.get("overall_confidence")
    if overall_confidence_raw is not None:
        overall_confidence = float(overall_confidence_raw)
    else:
        partial_count = int(summary.get("partial", summary.get("partial_count", 0)))
        abstained_count = int(summary.get("abstained_count", 0))
        non_abstained = total_claims - abstained_count
        if non_abstained > 0:
            overall_confidence = (verified_claims + partial_count * 0.5) / non_abstained
        else:
            overall_confidence = 0.0

    abstained_count = int(summary.get("abstained_count", 0))
    non_abstained = total_claims - abstained_count
    supported_rate = summary.get("supported_rate")
    if supported_rate is None:
        supported_rate = verified_claims / non_abstained if non_abstained > 0 else 0.0

    return {
        "total_claims": total_claims,
        "verified_claims": verified_claims,
        "corrected_citations": corrected_citations,
        "removed_claims": removed_claims,
        "softened_claims": softened_claims,
        "overall_confidence": float(overall_confidence),
        "supported_count": int(summary.get("supported", summary.get("supported_count", verified_claims))),
        "partial_count": int(summary.get("partial", summary.get("partial_count", 0))),
        "unsupported_count": int(summary.get("unsupported", summary.get("unsupported_count", softened_claims))),
        "contradicted_count": int(summary.get("contradicted", summary.get("contradicted_count", removed_claims))),
        "abstained_count": abstained_count,
        "supported_rate": float(supported_rate),
        "warning": bool(summary.get("warning", False)),
        "verification_skipped": bool(summary.get("verification_skipped", False)),
        "reason": summary.get("reason", ""),
        "claim_revisions": int(summary.get("claim_revisions", 0)),
        "atomic_facts_total": int(summary.get("atomic_facts_total", 0)),
        "atomic_facts_verified": int(summary.get("atomic_facts_verified", 0)),
        "atomic_facts_softened": int(summary.get("atomic_facts_softened", 0)),
        "claims_fully_verified": int(summary.get("claims_fully_verified", 0)),
        "claims_partially_softened": int(summary.get("claims_partially_softened", 0)),
        "claims_fully_softened": int(summary.get("claims_fully_softened", 0)),
        "external_searches": int(summary.get("external_searches", 0)),
        "new_sources_added": int(summary.get("new_sources_added", 0)),
        "analysis_summary": (
            summary.get("analysis_summary", {})
            if isinstance(summary.get("analysis_summary"), dict)
            else {}
        ),
        "routing_summary": (
            summary.get("routing_summary", {})
            if isinstance(summary.get("routing_summary"), dict)
            else {}
        ),
    }


def _normalize_verification_records(
    verifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for verification in verifications:
        records.append(
            {
                "claim_index": verification.get("claim_index", 0),
                "verdict": verification.get("verdict", "unsupported"),
                "confidence": float(verification.get("confidence", 0.0)),
                "verification_confidence": float(
                    verification.get(
                        "verification_confidence",
                        verification.get("confidence", 0.0),
                    )
                ),
                "routing_confidence_level": verification.get(
                    "routing_confidence_level",
                    verification.get("confidence_level", ""),
                ),
                "routing_confidence_score": float(
                    verification.get("routing_confidence_score", 0.0)
                ),
                "evidence_match_score": float(
                    verification.get("evidence_match_score", 0.0)
                ),
                "used_quick_verification": bool(
                    verification.get("used_quick_verification", False)
                ),
                "verification_latency_ms": float(
                    verification.get("verification_latency_ms", 0.0)
                ),
                "claim_role": verification.get("claim_role", "fact"),
                "verification_method": verification.get("verification_method", ""),
                "evidence_snippet": verification.get("evidence_preview", ""),
                "claim_text": verification.get("claim_text", ""),
            }
        )
    return records


def _normalize_corrections(
    corrections: list[dict[str, Any]],
    key_to_numeric: dict[str, str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for correction in corrections:
        original_key = str(correction.get("original_key", "") or "")
        corrected_key = str(correction.get("corrected_key", "") or "")
        original_source_pool_index = correction.get("original_source_pool_index")
        corrected_source_pool_index = correction.get("corrected_source_pool_index")
        normalized.append(
            {
                "claim_index": correction.get("claim_index", 0),
                "action": correction.get(
                    "action",
                    correction.get("correction_type", "keep"),
                ),
                "original_key": (
                    str(original_source_pool_index)
                    if original_source_pool_index is not None
                    else key_to_numeric.get(original_key, original_key)
                ),
                "corrected_key": (
                    str(corrected_source_pool_index)
                    if corrected_source_pool_index is not None
                    else key_to_numeric.get(corrected_key, corrected_key)
                ),
            }
        )
    return normalized


def _normalize_numeric_claims(
    numeric_claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for numeric_claim in numeric_claims:
        normalized.append(
            {
                "claim_index": numeric_claim.get("claim_index", 0),
                "numeric_value": numeric_claim.get("raw_value", ""),
                "verification_status": (
                    "verified"
                    if numeric_claim.get("qa_verified")
                    else "pending"
                ),
            }
        )
    return normalized


def _extract_claim_generated_events(
    node_id: str,
    timestamp: str,
    claims: list[dict[str, Any]],
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for idx, claim in enumerate(claims):
        events.append(
            ClaimGeneratedEvent(
                node_id=node_id,
                timestamp=timestamp,
                claim_text=claim.get("claim_text", ""),
                claim_index=idx,
                citation_keys=claim.get("citation_keys", []),
                claim_role=claim.get("claim_role", "fact"),
            )
        )
    return events


def _extract_claim_verified_events(
    node_id: str,
    timestamp: str,
    verifications: list[dict[str, Any]],
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for verification in verifications:
        events.append(
            ClaimVerifiedEvent(
                node_id=node_id,
                timestamp=timestamp,
                claim_index=verification.get("claim_index", 0),
                verdict=verification.get("verdict", "unsupported"),
                confidence=verification.get("confidence", 0.0),
                verification_confidence=verification.get(
                    "verification_confidence",
                    verification.get("confidence", 0.0),
                ),
                routing_confidence_level=verification.get(
                    "routing_confidence_level",
                    "",
                ),
                routing_confidence_score=verification.get(
                    "routing_confidence_score",
                    0.0,
                ),
                evidence_match_score=verification.get(
                    "evidence_match_score",
                    0.0,
                ),
                used_quick_verification=verification.get(
                    "used_quick_verification",
                    False,
                ),
                verification_latency_ms=verification.get(
                    "verification_latency_ms",
                    0.0,
                ),
                claim_role=verification.get("claim_role", "fact"),
                verification_method=verification.get("verification_method", ""),
                evidence_snippet=verification.get("evidence_snippet", ""),
                claim_text=verification.get("claim_text", ""),
            )
        )
    return events


def _extract_citation_corrected_events(
    node_id: str,
    timestamp: str,
    corrections: list[dict[str, Any]],
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for correction in corrections:
        events.append(
            CitationCorrectedEvent(
                node_id=node_id,
                timestamp=timestamp,
                claim_index=correction.get("claim_index", 0),
                action=correction.get("action", "keep"),
                original_key=correction.get("original_key", ""),
                corrected_key=correction.get("corrected_key", ""),
            )
        )
    return events


def _extract_numeric_claim_events(
    node_id: str,
    timestamp: str,
    numeric_claims: list[dict[str, Any]],
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for numeric_claim in numeric_claims:
        events.append(
            NumericClaimDetectedEvent(
                node_id=node_id,
                timestamp=timestamp,
                claim_index=numeric_claim.get("claim_index", 0),
                numeric_value=numeric_claim.get("numeric_value", ""),
                verification_status=numeric_claim.get(
                    "verification_status", "pending"
                ),
            )
        )
    return events


def _extract_verification_summary_event(
    node_id: str,
    timestamp: str,
    summary: dict[str, Any],
) -> VerificationSummaryEvent:
    return VerificationSummaryEvent(
        node_id=node_id,
        timestamp=timestamp,
        total_claims=summary.get("total_claims", 0),
        verified_claims=summary.get("verified_claims", 0),
        corrected_citations=summary.get("corrected_citations", 0),
        removed_claims=summary.get("removed_claims", 0),
        softened_claims=summary.get("softened_claims", 0),
        overall_confidence=summary.get("overall_confidence", 0.0),
        analysis_summary=summary.get("analysis_summary", {}),
        routing_summary=summary.get("routing_summary", {}),
    )


def _extract_verification_payload(
    state: WorkflowState,
    output: Any,
) -> dict[str, Any]:
    """Prefer framework state, fall back to legacy structured-output payloads."""
    details = state.get("verification_details")
    if isinstance(details, dict) and details:
        return details

    claims = select_claims(state)
    summary = select_verification_summary(state)
    analysis_summary = select_analysis_summary(state)
    if claims or summary:
        return {
            "claims": claims,
            "verifications": [],
            "corrections": [],
            "numeric_claims": [],
            "verification_summary": summary,
            "analysis_summary": analysis_summary,
        }

    if isinstance(output, SynthesizerOutput) and isinstance(output.structured_output, dict):
        return output.structured_output
    if isinstance(output, dict):
        if any(
            key in output
            for key in (
                "claims",
                "verifications",
                "corrections",
                "numeric_claims",
                "verification_summary",
                "analysis_summary",
            )
        ):
            return output
        structured = output.get("structured_output")
        if isinstance(structured, dict):
            return structured

    return {}


def _reclaim_post_process(
    node_id: str,
    output: Any,
    state: WorkflowState,
    timestamp: str,
) -> list[StreamEvent]:
    """Emit verification events from framework state or legacy structured output."""
    events: list[StreamEvent] = []
    verification_data = select_verification_payload(state) or _extract_verification_payload(state, output)
    if not verification_data:
        logger.debug("RECLAIM_POST_PROCESS node_id=%s no_verification_data", node_id)
        return events

    claims = verification_data.get("claims", [])
    if claims:
        events.extend(_extract_claim_generated_events(node_id, timestamp, claims))

    verifications = verification_data.get("verifications", [])
    if verifications:
        events.extend(
            _extract_claim_verified_events(node_id, timestamp, verifications)
        )

    corrections = verification_data.get("corrections", [])
    if corrections:
        events.extend(
            _extract_citation_corrected_events(node_id, timestamp, corrections)
        )

    numeric_claims = verification_data.get("numeric_claims", [])
    if numeric_claims:
        events.extend(_extract_numeric_claim_events(node_id, timestamp, numeric_claims))

    summary = verification_data.get("verification_summary", {})
    if summary:
        events.append(
            _extract_verification_summary_event(node_id, timestamp, summary)
        )
    elif claims:
        total = len(claims)
        verified = sum(
            1 for verification in verifications if verification.get("verdict") == "supported"
        )
        corrected = len(corrections)
        removed = sum(
            1 for verification in verifications if verification.get("verdict") == "contradicted"
        )
        softened = sum(
            1 for verification in verifications if verification.get("verdict") == "unsupported"
        )
        events.append(
            VerificationSummaryEvent(
                node_id=node_id,
                timestamp=timestamp,
                total_claims=total,
                verified_claims=verified,
                corrected_citations=corrected,
                removed_claims=removed,
                softened_claims=softened,
                overall_confidence=(verified / total if total > 0 else 0.0),
            )
        )

    return events


async def _run_fallback_synthesis(
    llm_client: Any,
    config: AgentNodeConfig,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[Any, dict[str, int]]:
    """Fallback to a normal synthesizer completion when verification skips."""
    response = await llm_client.complete(
        messages,
        config.model_tier,
        max_tokens=max_tokens,
        structured_output=config.output_model,
    )
    content: Any = response.content
    if response.structured is not None:
        content = response.structured
    return content, response.usage


def _extract_report_text(content: Any) -> str:
    """Extract plain report text from synthesizer output payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("report") or content.get("content") or "")
    report = getattr(content, "report", None)
    if isinstance(report, str):
        return report
    return str(content or "")


def _persist_grounding_state(
    node_id: str,
    state: WorkflowState,
    report_content: str,
    pipeline: CitationVerificationPipeline,
    sources: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    numeric_claims: list[dict[str, Any]],
    summary_data: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Persist pipeline outputs in framework/app bridge format."""
    url_to_index = _build_url_to_index_map(sources)
    key_to_numeric = _build_key_to_numeric_index_map(
        pipeline.last_evidence_pool,
        url_to_index,
    )

    if isinstance(pipeline.last_final_content, str) and pipeline.last_final_content:
        report_content = pipeline.last_final_content

    if report_content:
        report_content = _replace_human_citations_with_numeric(
            report_content,
            key_to_numeric,
        )

    claims = [
        _claim_to_state_dict(claim, url_to_index, key_to_numeric)
        for claim in pipeline.last_generated_claims
    ]
    claims = _recalculate_claim_positions(report_content, claims)

    if not summary_data:
        summary_data = _build_framework_summary(pipeline.last_verification_summary)

    payload = {
        "claims": claims,
        "verifications": _normalize_verification_records(verifications),
        "corrections": _normalize_corrections(corrections, key_to_numeric),
        "numeric_claims": _normalize_numeric_claims(numeric_claims),
        "verification_summary": summary_data,
        "analysis_summary": summary_data.get("analysis_summary", {}),
    }

    if claims or summary_data:
        if state.runtime_store is None:
            state.append(node_id, "claims", claims)
            state.append(node_id, "verification_summary", summary_data)
            state.append(
                node_id,
                "analysis_summary",
                summary_data.get("analysis_summary", {}),
            )
            state.append(node_id, "verification_details", payload)
        if state.runtime_store is not None:
            state.runtime_store.publish_verification_payload(
                producer_node_id=node_id,
                payload=payload,
            )
        logger.debug(
            "SYNTHESIZER_GROUNDING_STATE_WRITTEN node_id=%s claims=%d summary_keys=%s "
            "analysis_summary_keys=%s",
            node_id,
            len(claims),
            sorted(summary_data.keys()),
            sorted((summary_data.get("analysis_summary") or {}).keys()),
        )

    return report_content, payload


async def _execute(
    node_id: str,
    config: AgentNodeConfig,
    state: WorkflowState,
    llm_client: Any,
    _tools: list[Any],
    pools: dict[str, Any],
    _agent_input: Any,
    messages: list[dict[str, Any]],
    _tool_context: Any,
) -> AgentOutput | None:
    """Run grounded synthesis modes through the framework citation pipeline."""
    grounding_mode = resolve_grounding_mode(config)
    if grounding_mode == "none":
        return None

    reclaim_cfg = _get_reclaim_config(config)
    citation_config = _build_citation_config(config)
    sources = _collect_sources(pools)
    observations = _collect_observations(state, pools)
    sources = _hydrate_sparse_sources(sources, observations)
    # Filter sources that still have no usable text after hydration
    pre_filter = len(sources)
    sources = [
        s for s in sources
        if (s.get("content") or "").strip() or (s.get("snippet") or "").strip()
    ]
    if pre_filter > len(sources):
        logger.warning(
            "SYNTHESIZER_EMPTY_SOURCES_FILTERED before=%d after=%d",
            pre_filter,
            len(sources),
        )
    pipeline = _build_reclaim_pipeline(llm_client, citation_config)
    mode_label = grounding_mode.upper()
    logger.info(
        "SYNTHESIZER_%s_START node_id=%s query=%s sources=%d observations=%d "
        "target_words=%s max_tokens=%s",
        mode_label,
        node_id,
        state.query[:120],
        len(sources),
        len(observations),
        reclaim_cfg["target_word_count"],
        reclaim_cfg["max_tokens"],
    )

    report_content = ""
    verifications: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    numeric_claims: list[dict[str, Any]] = []
    summary_data: dict[str, Any] = {}
    token_usage: dict[str, int] = {}

    draft_content = ""
    if grounding_mode == "classical_lite":
        max_tokens = int(config.conversation_budget or reclaim_cfg["max_tokens"])
        draft_output, token_usage = await _run_fallback_synthesis(
            llm_client,
            config,
            messages,
            max_tokens,
        )
        draft_content = _extract_report_text(draft_output)
        report_content = draft_content
        logger.info(
            "SYNTHESIZER_CLASSICAL_LITE_DRAFT node_id=%s draft_chars=%d token_usage=%s",
            node_id,
            len(draft_content),
            token_usage,
        )

    async for item in pipeline.run_full_pipeline(
        sources=sources,
        observations=observations,
        query=state.query,
        target_word_count=int(reclaim_cfg["target_word_count"]),
        max_tokens=int(reclaim_cfg["max_tokens"]),
        draft_content=draft_content or None,
    ):
        if isinstance(item, str):
            report_content = item
            continue

        if not isinstance(item, VerificationEvent):
            continue

        data = dict(item.data)
        if item.event_type == "content_revised":
            report_content = str(data.get("content", report_content))
        elif item.event_type == "claim_verified":
            verifications.append(data)
        elif item.event_type == "citation_corrected":
            corrections.append(data)
        elif item.event_type == "numeric_claim_detected":
            numeric_claims.append(data)
        elif item.event_type == "verification_summary":
            summary_data = _build_framework_summary(data)

    logger.info(
        "SYNTHESIZER_%s_PIPELINE_RESULT node_id=%s content_chars=%d claims=%d "
        "verifications=%d corrections=%d numeric=%d",
        mode_label,
        node_id,
        len(report_content),
        len(pipeline.last_generated_claims),
        len(verifications),
        len(corrections),
        len(numeric_claims),
    )
    report_content, _payload = _persist_grounding_state(
        node_id,
        state,
        report_content,
        pipeline,
        sources,
        verifications,
        corrections,
        numeric_claims,
        summary_data,
    )

    if not report_content:
        logger.info(
            "SYNTHESIZER_%s_FALLBACK node_id=%s reason=no_pipeline_content",
            mode_label,
            node_id,
        )
        fallback_output, fallback_usage = await _run_fallback_synthesis(
            llm_client,
            config,
            messages,
            int(reclaim_cfg["max_tokens"]),
        )
        report_content = _extract_report_text(fallback_output)
        if not token_usage:
            token_usage = fallback_usage

    logger.info(
        "SYNTHESIZER_%s_COMPLETE node_id=%s final_chars=%d token_usage=%s",
        mode_label,
        node_id,
        len(str(report_content)),
        token_usage,
    )

    return AgentOutput(
        content=report_content,
        output_key=config.output_key,
        token_usage=token_usage,
    )


def _post_process(
    node_id: str,
    output: Any,
    config: AgentNodeConfig,
    state: WorkflowState,
) -> list[StreamEvent]:
    """Emit synthesis and optional verification events."""
    observations = state.get_all("findings")
    sources_list = state.get_all("sources")

    state_obs = len(observations) if observations else 0
    state_src = 0
    for entry in sources_list:
        if isinstance(entry, list):
            state_src += len(entry)
        elif entry is not None:
            state_src += 1

    obs_pool = state.pools.get("observations") if state.pools else None
    src_pool = state.pools.get("sources") if state.pools else None
    pool_obs = obs_pool.count() if obs_pool else 0
    pool_src = src_pool.count() if src_pool else 0

    total_obs = max(state_obs, pool_obs)
    total_src = max(state_src, pool_src)

    logger.info(
        "SYNTHESIZER_CONTEXT_COUNTS node_id=%s state_observations=%d pool_observations=%d "
        "state_sources=%d pool_sources=%d total_observations=%d total_sources=%d",
        node_id,
        state_obs,
        pool_obs,
        state_src,
        pool_src,
        total_obs,
        total_src,
    )

    timestamp = datetime.now(tz=UTC).isoformat()
    events: list[StreamEvent] = [
        SynthesisStartedEvent(
            node_id=node_id,
            timestamp=timestamp,
            total_observations=total_obs,
            total_sources=total_src,
        )
    ]

    if resolve_grounding_mode(config) != "none":
        events.extend(_reclaim_post_process(node_id, output, state, timestamp))

    return events


def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    _runtime_context: dict[str, Any] | None = None,
) -> AgentNodeConfig:
    """Fill in synthesizer defaults; switch prompts for reclaim mode."""
    updates: dict[str, Any] = {}

    if _is_reclaim_mode(config):
        logger.info("SYNTHESIZER_ENRICH_RECLAIM node_subtype=%s", config.subtype)
        if not config.system_prompt:
            updates["system_prompt"] = _build_reclaim_system_prompt()
        if not config.user_prompt_template:
            updates["user_prompt_template"] = _build_reclaim_user_prompt()
        if config.max_tool_calls is None:
            updates["max_tool_calls"] = _RECLAIM_MAX_TOOL_CALLS
    else:
        if not config.system_prompt:
            from databricks_deep_research.agents.prompts.synthesizer import (
                SYNTHESIZER_SYSTEM_PROMPT,
            )

            updates["system_prompt"] = SYNTHESIZER_SYSTEM_PROMPT

        if not config.user_prompt_template:
            from databricks_deep_research.agents.prompts.synthesizer import (
                SYNTHESIZER_USER_PROMPT,
            )

            updates["user_prompt_template"] = SYNTHESIZER_USER_PROMPT

        if config.max_tool_calls is None:
            updates["max_tool_calls"] = DEFAULT_MAX_TOOL_CALLS

    if updates:
        return config.model_copy(update=updates)
    return config


register_builtin(
    "synthesizer",
    post_process=_post_process,
    enrich_config=_enrich_config,
    execute=_execute,
    output_model=SynthesizerOutput,
)

__all__: list[str] = []
