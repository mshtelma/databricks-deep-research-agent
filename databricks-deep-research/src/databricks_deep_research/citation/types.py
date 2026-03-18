"""Citation data types for the deep-research framework.

Ported from the app's citation verification pipeline. These are clean,
self-contained Pydantic/dataclass models with NO app dependencies.

Types cover the 7-stage citation verification pipeline:
1. Evidence Pre-Selection   -> RankedEvidence, EvidenceInfo
2. Interleaved Generation   -> InterleavedClaim
3. Confidence Classification -> ConfidenceLevel, ConfidenceResult
4. Isolated Verification    -> VerificationVerdict, VerificationResult
5. Citation Correction      -> CorrectionAction, CorrectionResult, CorrectionMetrics
6. Numeric QA Verification  -> NumericValue, QAVerificationResult, NumericVerificationResult
7. ARE Verification         -> (uses above types)

Plus summary types: ClaimInfo, EvidenceInfo, VerificationSummaryInfo, ContentQuality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerificationVerdict(StrEnum):
    """Four-tier verification verdict for a claim against evidence."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class CorrectionAction(StrEnum):
    """Types of citation corrections that can be applied."""

    KEEP = "keep"
    REPLACE = "replace"
    REMOVE = "remove"
    ADD_ALTERNATE = "add_alternate"


class ConfidenceLevel(StrEnum):
    """HaluGate-style confidence levels for verification routing."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimRole(StrEnum):
    """Role of generated content in reclaim mode."""

    FACT = "fact"
    ANALYSIS = "analysis"
    FREE = "free"


class VerificationMethod(StrEnum):
    """Verification strategy used for a claim."""

    ENTAILMENT = "entailment"
    NUMERIC_QA = "numeric_qa"
    GROUNDING = "grounding"
    STRUCTURAL = "structural"


# ---------------------------------------------------------------------------
# Evidence types  (Stage 1)
# ---------------------------------------------------------------------------


@dataclass
class EvidenceInfo:
    """Pre-selected evidence span for citation verification.

    Created during Stage 1 (Evidence Pre-Selection) of the citation pipeline.
    """

    source_url: str
    quote_text: str
    canonical_source_url: str | None = None
    source_title: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    relevance_score: float | None = None
    has_numeric_content: bool = False
    source_pool_index: int | None = None
    evidence_pool_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_url": self.source_url,
            "canonical_source_url": self.canonical_source_url,
            "source_title": self.source_title,
            "quote_text": self.quote_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "section_heading": self.section_heading,
            "relevance_score": self.relevance_score,
            "has_numeric_content": self.has_numeric_content,
            "source_pool_index": self.source_pool_index,
            "evidence_pool_index": self.evidence_pool_index,
        }


@dataclass
class RankedEvidence:
    """An evidence span with relevance ranking.

    Produced by Stage 1 (Evidence Pre-Selection) -- each span is ranked by
    relevance to the research query and carries metadata about its source.
    """

    source_id: UUID | None
    source_url: str
    quote_text: str
    start_offset: int | None
    end_offset: int | None
    section_heading: str | None
    relevance_score: float
    has_numeric_content: bool
    canonical_source_url: str | None = None
    source_title: str | None = None
    source_pool_index: int | None = None
    evidence_pool_index: int | None = None
    is_snippet_based: bool = False
    """True if evidence was derived from a search snippet rather than
    full crawled content.  Snippet-based evidence has lower confidence."""


# ---------------------------------------------------------------------------
# Claim types  (Stage 2 -- Interleaved Generation)
# ---------------------------------------------------------------------------


@dataclass
class InterleavedClaim:
    """A claim generated with evidence constraint (ReClaim pattern).

    Produced during Stage 2 (Interleaved Generation) where the LLM generates
    claims constrained by pre-selected evidence.
    """

    claim_text: str
    claim_type: str  # "general" or "numeric"
    position_start: int
    position_end: int
    evidence: RankedEvidence | None
    evidence_index: int | None
    evidences: list[RankedEvidence] = field(default_factory=list)
    evidence_indices: list[int] = field(default_factory=list)
    confidence_score: float | None = None
    claim_role: str = ClaimRole.FACT.value
    citation_key: str | None = None
    """Primary human-readable key like ``"Arxiv"``, ``"Zhipu"``."""
    citation_keys: list[str] | None = None
    """All keys for multi-marker sentences (e.g. ``["Arxiv", "Zhipu"]``)."""
    verification_text: str | None = None
    analysis_parent_claim_indices: list[int] = field(default_factory=list)
    from_free_block: bool = False


@dataclass
class ClaimInfo:
    """Atomic claim extracted from generated content.

    Created during Stage 2 (Interleaved Generation) of the citation pipeline
    and enriched by subsequent stages (confidence, verification, correction).
    """

    claim_text: str
    claim_type: str  # "general" or "numeric"
    position_start: int
    position_end: int
    evidence: EvidenceInfo | None = None
    evidences: list[EvidenceInfo] = field(default_factory=list)
    confidence_level: str | None = None  # "high", "medium", "low"
    routing_confidence_score: float | None = None
    verification_verdict: str | None = None
    """One of ``"supported"``, ``"partial"``, ``"unsupported"``, ``"contradicted"``."""
    verification_confidence: float | None = None
    verification_reasoning: str | None = None
    verification_method: str | None = None
    evidence_match_score: float | None = None
    used_quick_verification: bool = False
    verification_latency_ms: float | None = None
    abstained: bool = False
    citation_key: str | None = None
    """Primary key like ``"Arxiv"``, ``"Zhipu"``."""
    citation_keys: list[str] | None = None
    """All keys for multi-marker sentences."""
    claim_role: str = ClaimRole.FACT.value
    verification_text: str | None = None
    analysis_parent_claim_indices: list[int] = field(default_factory=list)
    from_free_block: bool = False
    """True if extracted from a ``<free>`` block (needs verification)."""
    has_fallback_evidence: bool = False
    """True if evidence was assigned via fallback keyword matching (not LLM citation)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "evidences": [evidence.to_dict() for evidence in self.evidences],
            "confidence_level": self.confidence_level,
            "routing_confidence_score": self.routing_confidence_score,
            "verification_verdict": self.verification_verdict,
            "verification_confidence": self.verification_confidence,
            "verification_reasoning": self.verification_reasoning,
            "verification_method": self.verification_method,
            "evidence_match_score": self.evidence_match_score,
            "used_quick_verification": self.used_quick_verification,
            "verification_latency_ms": self.verification_latency_ms,
            "abstained": self.abstained,
            "citation_key": self.citation_key,
            "citation_keys": self.citation_keys,
            "claim_role": self.claim_role,
            "verification_text": self.verification_text,
            "analysis_parent_claim_indices": self.analysis_parent_claim_indices,
            "from_free_block": self.from_free_block,
            "has_fallback_evidence": self.has_fallback_evidence,
        }


# ---------------------------------------------------------------------------
# Confidence types  (Stage 3)
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceResult:
    """Result of confidence classification for a claim."""

    level: ConfidenceLevel
    score: float  # 0.0 -- 1.0
    indicators: list[str]  # Matched linguistic indicators
    reasoning: str


# ---------------------------------------------------------------------------
# Verification types  (Stage 4 -- Isolated Verification)
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of isolated claim verification (CoVe pattern)."""

    verdict: VerificationVerdict
    reasoning: str
    key_match: str | None = None
    issues: list[str] | None = None
    confidence: float = 0.0
    abstained: bool = False


# ---------------------------------------------------------------------------
# Correction types  (Stage 5 -- Citation Correction)
# ---------------------------------------------------------------------------


@dataclass
class CorrectionResult:
    """Result of citation correction for a single claim."""

    claim_text: str
    correction_type: CorrectionAction
    original_evidence: RankedEvidence | None
    corrected_evidence: RankedEvidence | None
    original_evidence_index: int | None = None
    corrected_evidence_index: int | None = None
    alternate_evidence: list[RankedEvidence] = field(default_factory=list)
    alternate_evidence_indices: list[int] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    evidence_match_score: float = 0.0


@dataclass
class CorrectionMetrics:
    """Aggregate metrics for citation corrections across a report."""

    total_claims: int = 0
    kept: int = 0
    replaced: int = 0
    removed: int = 0
    added_alternate: int = 0

    @property
    def correction_rate(self) -> float:
        """Percentage of claims that needed correction."""
        if self.total_claims == 0:
            return 0.0
        corrected = self.replaced + self.removed + self.added_alternate
        return corrected / self.total_claims


# ---------------------------------------------------------------------------
# Numeric verification types  (Stage 6)
# ---------------------------------------------------------------------------


@dataclass
class NumericValue:
    """Parsed numeric value with unit and context."""

    raw_text: str
    normalized_value: Decimal | None
    unit: str | None
    entity: str | None
    multiplier: int = 1


@dataclass
class QAVerificationResult:
    """Result of a single QA-based numeric verification."""

    question: str
    claim_answer: str
    evidence_answer: str
    match: bool
    normalized_comparison: dict[str, Any] | None


@dataclass
class NumericVerificationResult:
    """Complete numeric verification result (QAFactEval pattern)."""

    claim_text: str
    parsed_value: NumericValue
    qa_results: list[QAVerificationResult]
    overall_match: bool
    derivation_type: str  # "direct" or "computed"
    confidence: float


@dataclass
class AnalysisSummaryInfo:
    """Summary of analysis-grounding results for reclaim output."""

    total_claims: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    grounded_rate: float = 0.0
    unsupported_rate: float = 0.0
    warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "grounded_rate": self.grounded_rate,
            "unsupported_rate": self.unsupported_rate,
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Summary types
# ---------------------------------------------------------------------------


@dataclass
class VerificationSummaryInfo:
    """Summary of verification results for a message.

    Created after Stage 4 (Isolated Verification) completes.
    Updated with Stage 7 metrics after ARE-style verification.
    """

    total_claims: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    abstained_count: int = 0
    fact_rate_denominator: int = 0
    supported_rate: float = 0.0
    unsupported_rate: float = 0.0
    contradicted_rate: float = 0.0
    warning: bool = False
    citation_corrections: int = 0

    # Stage 7: ARE-style Verification Retrieval metrics
    claim_revisions: int = 0
    atomic_facts_total: int = 0
    atomic_facts_verified: int = 0
    atomic_facts_softened: int = 0
    claims_fully_verified: int = 0
    claims_partially_softened: int = 0
    claims_fully_softened: int = 0
    external_searches: int = 0
    new_sources_added: int = 0
    analysis_summary: AnalysisSummaryInfo = field(default_factory=AnalysisSummaryInfo)
    routing_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "abstained_count": self.abstained_count,
            "fact_rate_denominator": self.fact_rate_denominator,
            "supported_rate": self.supported_rate,
            "unsupported_rate": self.unsupported_rate,
            "contradicted_rate": self.contradicted_rate,
            "warning": self.warning,
            "citation_corrections": self.citation_corrections,
            # Stage 7 metrics
            "claim_revisions": self.claim_revisions,
            "atomic_facts_total": self.atomic_facts_total,
            "atomic_facts_verified": self.atomic_facts_verified,
            "atomic_facts_softened": self.atomic_facts_softened,
            "claims_fully_verified": self.claims_fully_verified,
            "claims_partially_softened": self.claims_partially_softened,
            "claims_fully_softened": self.claims_fully_softened,
            "external_searches": self.external_searches,
            "new_sources_added": self.new_sources_added,
            "analysis_summary": self.analysis_summary.to_dict(),
            "routing_summary": self.routing_summary,
        }


# ---------------------------------------------------------------------------
# Content quality types  (Pre-Stage 1 filtering)
# ---------------------------------------------------------------------------


@dataclass
class ContentQuality:
    """Result of content quality evaluation for a source.

    Used to filter out low-quality sources (paywalls, abstract-only pages,
    navigation-heavy content) before citation pipeline processing.
    """

    score: float  # 0.0 -- 1.0, higher is better
    has_specific_facts: bool
    has_numeric_data: bool
    is_abstract_only: bool
    is_paywall: bool
    is_navigation_heavy: bool
    word_count: int
    reason: str  # Human-readable explanation


# ---------------------------------------------------------------------------
# Pydantic output models (used as structured LLM output schemas)
# ---------------------------------------------------------------------------


class EvidenceSpanOutput(BaseModel):
    """A single evidence span extracted by LLM."""

    quote_text: str = Field(description="Exact quote from source (50-500 chars)")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance 0.0-1.0")
    has_numeric: bool = Field(description="True if contains numbers/statistics")
    section: str | None = Field(default=None, description="Section heading if identifiable")


class VerificationOutput(BaseModel):
    """Structured output from isolated verification LLM call."""

    verdict: str = Field(description="SUPPORTED, PARTIAL, UNSUPPORTED, or CONTRADICTED")
    reasoning: str = Field(default="", description="Explanation of why this verdict was chosen")
    key_match: str | None = Field(
        default=None,
        description="Specific part of evidence that supports/contradicts",
    )
    issues: list[str] | None = Field(default=None, description="Specific issues found")
    verification_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the verification verdict",
    )

    @field_validator("issues", mode="before")
    @classmethod
    def _coerce_issues(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            compact = normalized.casefold().rstrip(".!")
            none_like_prefixes = (
                "none",
                "none identified",
                "no issue",
                "no issues",
                "no specific issues",
                "no material issues",
                "nothing identified",
            )
            if any(compact.startswith(prefix) for prefix in none_like_prefixes):
                return None
            return [normalized]
        return value


class BatchVerificationItem(BaseModel):
    """Single claim result inside a batched verification response."""

    claim_index: int = Field(description="0-based index of claim in input batch")
    verdict: str = Field(description="SUPPORTED, PARTIAL, UNSUPPORTED, or CONTRADICTED")
    reasoning: str = Field(default="", max_length=500)
    key_match: str | None = Field(default=None, max_length=200)
    verification_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class BatchVerificationOutput(BaseModel):
    """Structured output for batched verification LLM call."""

    results: list[BatchVerificationItem] = Field(
        description="Verification results in same order as input claims"
    )


class CorrectionDecisionOutput(BaseModel):
    """Structured output from citation correction LLM call."""

    action: str = Field(description="keep, replace, or remove")
    evidence_index: int | None = Field(
        default=None,
        description="1-indexed evidence option if replacing, null otherwise",
    )
    reasoning: str = Field(default="", description="Brief explanation of the decision")
