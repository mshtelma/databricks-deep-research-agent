"""Citation pipeline configuration for the deep-research framework.

Ported from the app's ``core.app_config`` citation config hierarchy.
Self-contained -- only depends on pydantic and the standard library.

The top-level ``CitationConfig`` mirrors the app's ``CitationVerificationConfig``
with all 7 stage sub-configs, plus enums for strategy/method selection.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Strategy / method enums
# ---------------------------------------------------------------------------


class RelevanceMethod(StrEnum):
    """Method for computing evidence relevance scores."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class AnswerComparisonMethod(StrEnum):
    """Method for comparing answers in numeric QA verification."""

    EXACT_MATCH = "exact_match"
    F1 = "f1"
    LERC = "lerc"


class ConfidenceEstimationMethod(StrEnum):
    """Method for estimating claim confidence levels."""

    LINGUISTIC = "linguistic"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    HYBRID = "hybrid"


class CorrectionMethod(StrEnum):
    """Method for citation correction."""

    KEYWORD_SEMANTIC_HYBRID = "keyword_semantic_hybrid"
    KEYWORD_ONLY = "keyword_only"
    SEMANTIC_ONLY = "semantic_only"


class ClaimDisposition(StrEnum):
    """Action to take for a claim based on its verification verdict."""

    KEEP = "keep"
    REMOVE = "remove"
    SOFTEN = "soften"


class SofteningStrategy(StrEnum):
    """Strategy for softening unverified claims in Stage 7.

    - HEDGE: Add hedging words ("reportedly", "allegedly")
    - QUALIFY: Add qualifying phrases ("it is believed that")
    - PARENTHETICAL: Add parenthetical markers ("(unverified)")
    """

    HEDGE = "hedge"
    QUALIFY = "qualify"
    PARENTHETICAL = "parenthetical"


class GenerationMode(StrEnum):
    """Generation mode for research reports.

    - CLASSICAL: Free-form prose with ``[Title](url)`` links. Best text quality.
                 Skips verification stages 3--6.
    - NATURAL: Light-touch ``[N]`` citations, runs full verification.
    - STRICT: Heavy ``[N]`` constraints. Maximum citations.
    """

    CLASSICAL = "classical"
    NATURAL = "natural"
    STRICT = "strict"


class SynthesisMode(StrEnum):
    """Synthesis approach for report generation.

    - INTERLEAVED: Evidence dumped into context, LLM generates with ``[N]`` markers.
    - REACT: LLM uses tools to retrieve evidence before each factual claim.
    """

    INTERLEAVED = "interleaved"
    REACT = "react"


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------


class EvidencePreselectionConfig(BaseModel):
    """Stage 1: Evidence Pre-Selection configuration."""

    max_spans_per_source: int = Field(default=10, ge=1, le=50)
    min_span_length: int = Field(default=50, ge=10)
    # DEPRECATED: use CitationConfig.max_evidence_chars (the pipeline-wide
    # cap applied to all 5 truncation sites). This stage-specific knob is
    # retained for one release cycle for backward compat — if set in YAML
    # it routes to max_evidence_chars with a DeprecationWarning. Default
    # bumped from 500 to 3000 to match the new pipeline-wide default.
    max_span_length: int = Field(default=3000, ge=50)
    relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    numeric_content_boost: float = Field(default=0.2, ge=0.0, le=1.0)
    relevance_computation_method: RelevanceMethod = RelevanceMethod.HYBRID

    # Chunking config for long sources
    chunk_size: int = Field(default=8000, ge=1000, le=20000)
    chunk_overlap: int = Field(default=1000, ge=0, le=5000)
    max_chunks_per_source: int = Field(default=5, ge=1, le=10)
    max_sources: int = Field(default=60, ge=10, le=300)

    model_config = {"frozen": True}


class InterleavedGenerationConfig(BaseModel):
    """Stage 2: Interleaved Generation configuration."""

    max_claims_per_section: int = Field(default=10, ge=1, le=50)
    min_evidence_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    retry_on_entailment_failure: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)
    max_evidence_spans: int = Field(
        default=120, ge=20, le=500,
        description="Max evidence spans to include in the generation prompt",
    )

    model_config = {"frozen": True}


class ConfidenceClassificationConfig(BaseModel):
    """Stage 3: Confidence Classification configuration."""

    high_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    quote_match_bonus: float = Field(default=0.4, ge=0.0, le=1.0)
    hedging_word_penalty: float = Field(default=0.2, ge=0.0, le=1.0)
    estimation_method: ConfidenceEstimationMethod = ConfidenceEstimationMethod.LINGUISTIC

    model_config = {"frozen": True}


class IsolatedVerificationConfig(BaseModel):
    """Stage 4: Isolated Verification configuration.

    Defaults use only framework-canonical tiers (``simple|analytical|complex``)
    so the bare ``FrameworkLLMClient.from_databricks(...)`` used by shell-app
    deployments resolves them without app-level extensions
    (``bulk_analysis``, ``fast``). Cost-aware: ``analytical`` for the full path
    and ``simple`` for the quick path keeps shell-app verification cheap by
    default while still producing real entailment judgments. Override per-agent
    via ``output_schema.isolated_verification`` in agent YAML when a richer
    tier is justified.
    """

    enable_nei_verdict: bool = True
    verification_model_tier: str = Field(default="analytical")
    quick_verification_tier: str = Field(default="simple")
    max_concurrent_verifications: int = Field(default=10, ge=1, le=50)

    model_config = {"frozen": True}


class CitationCorrectionConfig(BaseModel):
    """Stage 5: Citation Correction configuration."""

    correction_method: CorrectionMethod = CorrectionMethod.KEYWORD_SEMANTIC_HYBRID
    lambda_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    correction_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    allow_alternate_citations: bool = True

    model_config = {"frozen": True}


class NumericQAVerificationConfig(BaseModel):
    """Stage 6: Numeric QA Verification configuration."""

    rounding_tolerance: float = Field(default=0.05, ge=0.0, le=0.5)
    answer_comparison_method: AnswerComparisonMethod = AnswerComparisonMethod.F1
    require_unit_match: bool = True
    require_entity_match: bool = True

    model_config = {"frozen": True}


class VerificationRetrievalConfig(BaseModel):
    """Stage 7: ARE-style Verification Retrieval configuration.

    Implements the ARE (Atomic fact decomposition-based Retrieval and Editing)
    pattern for verifying and revising unsupported/partial claims.

    Scientific basis:
    - ARE: https://arxiv.org/abs/2410.16708
    - FActScore: https://arxiv.org/abs/2305.14251
    - SAFE: https://arxiv.org/abs/2403.18802
    """

    # Trigger conditions
    trigger_on_verdicts: list[str] = Field(
        default_factory=lambda: ["unsupported", "partial"],
        description="Verdicts that trigger verification retrieval",
    )

    # Atomic decomposition settings
    max_atomic_facts_per_claim: int = Field(
        default=5, ge=1, le=10,
        description="Maximum atomic facts to extract from a single claim",
    )

    # Search budget (per atomic fact)
    max_searches_per_fact: int = Field(
        default=2, ge=1, le=5,
        description="Max search attempts per atomic fact",
    )
    max_external_urls_per_search: int = Field(
        default=3, ge=1, le=10,
        description="Max URLs to crawl per external search",
    )

    # Entailment thresholds
    entailment_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum entailment score to accept evidence",
    )
    internal_search_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Similarity threshold for internal pool match",
    )

    # Reconstruction behavior
    softening_strategy: SofteningStrategy = Field(
        default=SofteningStrategy.HEDGE,
        description="Strategy for softening unverified facts",
    )

    # Timeouts
    decomposition_timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0,
    )
    search_timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0,
    )
    crawl_timeout_seconds: float = Field(
        default=15.0, ge=1.0, le=60.0,
    )

    # Model tiers for LLM calls.
    #
    # Defaults restricted to framework-canonical tiers
    # (``simple|analytical|complex``) so shell-app deployments using
    # ``FrameworkLLMClient.from_databricks(...)`` resolve them without
    # app-level extensions (``bulk_analysis``, ``fast``). Cost-aware
    # selection: structured fact decomposition / reconstruction / softening
    # are all simple-tier appropriate; entailment scoring needs analytical.
    decomposition_tier: str = Field(default="simple")
    entailment_tier: str = Field(default="analytical")
    reconstruction_tier: str = Field(default="simple")
    softening_tier: str = Field(default="simple")

    model_config = {"frozen": True}


class GroundingValidationConfig(BaseModel):
    """Grounding validation for ``<analysis>`` and ``<free>`` blocks.

    Validates that analysis blocks are logically derived from preceding
    citations and free blocks contain only structural content.
    """

    enabled: bool = Field(
        default=True,
        description="Enable grounding validation for analysis blocks",
    )
    max_blocks_to_validate: int = Field(default=20, ge=1, le=50)
    min_analysis_length: int = Field(default=30, ge=10, le=100)
    allow_topic_sentences: bool = Field(default=True)
    max_preceding_citations: int = Field(default=10, ge=1, le=20)
    hedging_prefix: str = Field(default="Based on the evidence presented, ")
    abstained_unsupported_remove_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "When a claim has verdict in {unsupported, contradicted} AND "
            "abstained=True AND verification_confidence is below this "
            "threshold, treat as REMOVE instead of KEEP. Default 0.5 — "
            "raise toward 1.0 for stricter removal, lower toward 0.0 to "
            "always keep abstained claims (legacy behavior)."
        ),
    )

    model_config = {"frozen": True}


class PostVerificationConfig(BaseModel):
    """Post-generation verification of structured output.

    Runs stages 4--6 on claims extracted from structured output
    (e.g. MeetingPrepOutput) without requiring interleaved generation.
    """

    enabled: bool = Field(default=True)
    max_claims_to_verify: int = Field(default=50, ge=1, le=200)
    include_stage4_isolation: bool = Field(default=True)
    include_stage5_correction: bool = Field(default=True)
    include_stage6_numeric: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    skip_low_priority_claims: bool = Field(default=True)

    model_config = {"frozen": True}


class ReactSynthesisConfig(BaseModel):
    """Configuration for ReAct-based synthesis mode."""

    # Tool budget
    max_tool_calls: int = Field(default=40, ge=5, le=250)
    tool_budget_per_section: int = Field(default=10, ge=3, le=50)

    # Grounding settings
    retrieval_window_size: int = Field(default=3, ge=1, le=10)
    grounding_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # Hybrid grounding check thresholds
    embedding_high_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    embedding_low_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    use_llm_judge_for_borderline: bool = Field(default=True)

    # Post-processing
    enable_post_processing: bool = Field(default=False)

    # Section-based synthesis
    use_sectioned_synthesis: bool = Field(default=False)

    model_config = {"frozen": True}


class ClaimDispositionConfig(BaseModel):
    """Stage 8: Post-verification claim disposition.

    Maps each verification verdict to an action (keep, remove, soften).

    Defaults favour SOFTEN over REMOVE for every non-contradicted verdict,
    and SOFTEN over KEEP for any verdict the verifier marked uncertain.
    Rationale (recorded for future contributors):

      * REMOVE leaves visible holes in the report — truncated tables,
        mid-sentence breaks, dangling section headers — that downstream
        reviewers (coverage reflectors) flag as defects. SOFTEN hedges
        the claim in place, preserving structure while signalling
        uncertainty.
      * KEEP for ``partial`` or ``abstained`` presents an unverified or
        only-partly-verified claim as flat fact, which is overconfident.
      * CONTRADICTED stays REMOVE — a contradicted claim is wrong, not
        merely uncited; hedging would mislead readers.

    Callers that need the old strict-removal behaviour (e.g., compliance
    pipelines) can construct ``ClaimDispositionConfig(
    unsupported=ClaimDisposition.REMOVE, ...)`` explicitly.
    """

    supported: ClaimDisposition = Field(default=ClaimDisposition.KEEP)
    partial: ClaimDisposition = Field(default=ClaimDisposition.SOFTEN)
    unsupported: ClaimDisposition = Field(default=ClaimDisposition.SOFTEN)
    contradicted: ClaimDisposition = Field(default=ClaimDisposition.REMOVE)
    abstained: ClaimDisposition = Field(default=ClaimDisposition.SOFTEN)
    analysis_partial: ClaimDisposition = Field(default=ClaimDisposition.SOFTEN)
    analysis_unsupported: ClaimDisposition = Field(default=ClaimDisposition.SOFTEN)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Top-level citation config
# ---------------------------------------------------------------------------


class CitationConfig(BaseModel):
    """Configuration for the 7-stage citation verification pipeline.

    Stages:
    1. Evidence Pre-Selection   -- extract relevant quotes from sources
    2. Interleaved Generation   -- generate claims with ``[N]`` citations
    3. Confidence Classification -- route claims by confidence level
    4. Isolated Verification    -- produce verdicts
    5. Citation Correction      -- swap citations from existing pool
    6. Numeric QA Verification  -- deep verification of numeric claims
    7. ARE Verification Retrieval -- atomic fact decomposition + search + revision
    """

    # Master toggle
    enabled: bool = True

    # Pipeline-wide cap on evidence quote length (chars). Applied at all five
    # truncation sites: evidence selection (Stage 1), claim generation prompt
    # (Stage 2), single-claim NLI verification (Stage 4 full path), batch
    # verification (Stage 4 batch path), and retry verification. This is the
    # single source of truth — supersedes the per-stage ad-hoc caps that
    # previously diverged (500/1000/1500). Override per-agent via the agent's
    # output_schema citation_pipeline.max_evidence_chars.
    max_evidence_chars: int = Field(
        default=3000, ge=200, le=10000,
        description=(
            "Pipeline-wide cap on evidence quote length applied across all "
            "5 truncation sites. Default 3000 covers typical multi-row "
            "markdown tables; raise for richer tabular corpora, lower for "
            "budget-constrained prompts."
        ),
    )

    # Synthesis mode
    synthesis_mode: SynthesisMode = SynthesisMode.INTERLEAVED
    generation_mode: GenerationMode = GenerationMode.STRICT
    react_synthesis: ReactSynthesisConfig = Field(default_factory=ReactSynthesisConfig)

    # Stage toggles
    enable_evidence_preselection: bool = True
    enable_interleaved_generation: bool = True
    enable_confidence_classification: bool = True
    enable_citation_correction: bool = True
    enable_numeric_qa_verification: bool = True
    enable_verification_retrieval: bool = False

    # Stage 4 (isolated per-claim NLI verification) master gate. Distinct from
    # the top-level ``enabled`` (which also gates citation *generation*): set
    # this False to run the cheap "grounding-only" lane — Stages 1-3 still
    # generate, link, and render ``[N]`` citations, but the expensive per-claim
    # NLI overlay (Stages 4a/4b) AND the verdict-based Stage 8 disposition are
    # skipped. Claims persist as resolvable-but-unverified (a normal clickable
    # citation, no verdict). Default True preserves full verification behavior.
    enable_isolated_verification: bool = True

    # Defect E (prevention): when numeric QA corroborates a claim's value
    # against the cited evidence (verbatim match -> 0.95, or all QA pairs
    # agreeing -> 1.0), promote a PARTIAL/UNSUPPORTED NLI verdict to SUPPORTED
    # so the grounded number is KEPT rather than hedged ("Reportedly, $X").
    # Never promotes CONTRADICTED. Set False to restore NLI-only verdicts.
    numeric_match_promotes_verdict: bool = True

    # Per-stage configuration
    evidence_preselection: EvidencePreselectionConfig = Field(
        default_factory=EvidencePreselectionConfig,
    )
    interleaved_generation: InterleavedGenerationConfig = Field(
        default_factory=InterleavedGenerationConfig,
    )
    confidence_classification: ConfidenceClassificationConfig = Field(
        default_factory=ConfidenceClassificationConfig,
    )
    isolated_verification: IsolatedVerificationConfig = Field(
        default_factory=IsolatedVerificationConfig,
    )
    citation_correction: CitationCorrectionConfig = Field(
        default_factory=CitationCorrectionConfig,
    )
    numeric_qa_verification: NumericQAVerificationConfig = Field(
        default_factory=NumericQAVerificationConfig,
    )
    verification_retrieval: VerificationRetrievalConfig = Field(
        default_factory=VerificationRetrievalConfig,
    )
    grounding_validation: GroundingValidationConfig = Field(
        default_factory=GroundingValidationConfig,
    )
    post_verification: PostVerificationConfig = Field(
        default_factory=PostVerificationConfig,
    )
    claim_disposition: ClaimDispositionConfig = Field(
        default_factory=ClaimDispositionConfig,
    )

    model_config = {"frozen": True}
