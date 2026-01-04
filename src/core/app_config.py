"""Central application configuration loaded from YAML."""

import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.core.yaml_loader import load_yaml_config

logger = logging.getLogger(__name__)

# Default config paths
_this_file = Path(__file__).resolve()
_src_root = _this_file.parent.parent  # app_config.py -> core -> src
_project_root = _src_root.parent  # src -> project root
DEFAULT_CONFIG_PATH = _project_root / "config" / "app.yaml"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for LLM calls."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionStrategy(str, Enum):
    """Endpoint selection strategy."""

    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class BackoffStrategy(str, Enum):
    """Backoff strategy for rate limit retries."""

    EXPONENTIAL = "exponential"  # delay = base * (2 ** attempt)
    LINEAR = "linear"  # delay = base * (attempt + 1)


class DomainFilterMode(str, Enum):
    """Domain filter operation mode."""

    INCLUDE = "include"  # Whitelist only - only listed domains allowed
    EXCLUDE = "exclude"  # Blacklist only - listed domains blocked
    BOTH = "both"  # Whitelist then blacklist - must be in include AND not in exclude


class ResearcherMode(str, Enum):
    """Researcher implementation mode for research type profiles."""

    REACT = "react"  # ReAct loop with LLM-controlled tool calls
    CLASSIC = "classic"  # Single-pass with fixed searches/crawls per step


class EndpointConfig(BaseModel):
    """Configuration for a single model endpoint."""

    endpoint_identifier: str
    max_context_window: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)

    # Optional overrides (inherit from role if not set)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_budget: int | None = Field(default=None, gt=0)
    supports_structured_output: bool = False
    # Some models (e.g., GPT-5) don't support temperature parameter
    supports_temperature: bool = True

    model_config = {"frozen": True}


class ModelRoleConfig(BaseModel):
    """Configuration for a model role (tier)."""

    endpoints: list[str] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=8000, gt=0)
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    reasoning_budget: int | None = Field(default=None, gt=0)
    tokens_per_minute: int = Field(default=100000, gt=0)
    rotation_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    fallback_on_429: bool = True

    model_config = {"frozen": True}


class ResearcherConfig(BaseModel):
    """Configuration for the Researcher agent."""

    max_search_queries: int = Field(default=2, ge=1, le=10)
    max_search_results: int = Field(default=10, ge=1, le=50)
    max_urls_to_crawl: int = Field(default=3, ge=1, le=20)
    content_preview_length: int = Field(default=3000, ge=100)
    content_storage_length: int = Field(default=10000, ge=1000)
    max_previous_observations: int = Field(default=3, ge=1, le=10)
    page_contents_limit: int = Field(default=8000, ge=1000)
    max_generated_queries: int = Field(default=3, ge=1, le=10)

    model_config = {"frozen": True}


class PlannerConfig(BaseModel):
    """Configuration for the Planner agent."""

    max_plan_iterations: int = Field(default=3, ge=1, le=10)

    model_config = {"frozen": True}


class CoordinatorConfig(BaseModel):
    """Configuration for the Coordinator agent."""

    max_clarification_rounds: int = Field(default=3, ge=0, le=5)
    enable_clarification: bool = True

    model_config = {"frozen": True}


class ReportLimitConfig(BaseModel):
    """Word/token limits for a single research depth level."""

    min_words: int = Field(ge=50)
    max_words: int = Field(ge=100)
    max_tokens: int = Field(ge=500)

    model_config = {"frozen": True}


class SynthesizerConfig(BaseModel):
    """Configuration for the Synthesizer agent."""

    max_report_length: int = Field(default=50000, ge=1000)
    report_limits: dict[str, ReportLimitConfig] = Field(
        default_factory=lambda: {
            "light": ReportLimitConfig(min_words=200, max_words=400, max_tokens=1000),
            "medium": ReportLimitConfig(min_words=400, max_words=800, max_tokens=2000),
            "extended": ReportLimitConfig(min_words=800, max_words=1500, max_tokens=4000),
        }
    )

    model_config = {"frozen": True}


class BackgroundConfig(BaseModel):
    """Configuration for the Background Investigator agent."""

    max_search_queries: int = Field(default=2, ge=1, le=5)
    max_results_per_query: int = Field(default=3, ge=1, le=10)
    max_total_results: int = Field(default=5, ge=1, le=20)

    model_config = {"frozen": True}


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    researcher: ResearcherConfig = Field(default_factory=ResearcherConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)

    model_config = {"frozen": True}


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    requests_per_second: float = Field(default=1.0, gt=0, le=10)
    default_result_count: int = Field(default=10, ge=1, le=50)
    freshness: str = Field(default="pm", pattern=r"^(pd|pw|pm|py)$")

    model_config = {"frozen": True}


class DomainFilterConfig(BaseModel):
    """Configuration for domain whitelist/blacklist filtering.

    Supports wildcard patterns:
    - "*.gov" - matches any .gov domain (cdc.gov, www.nasa.gov)
    - "*.edu" - matches any .edu domain
    - "news.*" - matches news.com, news.org, etc.
    - "exact.com" - exact match only

    Filter modes:
    - include: Only domains matching include_domains are allowed
    - exclude: Domains matching exclude_domains are blocked
    - both: Must match include_domains AND not match exclude_domains
    """

    mode: DomainFilterMode = DomainFilterMode.EXCLUDE
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    log_filtered: bool = False

    model_config = {"frozen": True}


class SearchConfig(BaseModel):
    """Configuration for search services."""

    brave: BraveSearchConfig = Field(default_factory=BraveSearchConfig)
    domain_filter: DomainFilterConfig = Field(default_factory=DomainFilterConfig)

    model_config = {"frozen": True}


class TruncationConfig(BaseModel):
    """Configuration for text truncation limits."""

    log_preview: int = Field(default=200, ge=10)
    error_message: int = Field(default=500, ge=50)
    query_display: int = Field(default=100, ge=10)
    source_snippet: int = Field(default=300, ge=50)

    model_config = {"frozen": True}


class RelevanceMethod(str, Enum):
    """Method for computing relevance scores."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class AnswerComparisonMethod(str, Enum):
    """Method for comparing answers in numeric QA verification."""

    EXACT_MATCH = "exact_match"
    F1 = "f1"
    LERC = "lerc"


class ConfidenceEstimationMethod(str, Enum):
    """Method for estimating confidence levels."""

    LINGUISTIC = "linguistic"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    HYBRID = "hybrid"


class CorrectionMethod(str, Enum):
    """Method for citation correction."""

    KEYWORD_SEMANTIC_HYBRID = "keyword_semantic_hybrid"
    KEYWORD_ONLY = "keyword_only"
    SEMANTIC_ONLY = "semantic_only"


class SofteningStrategy(str, Enum):
    """Strategy for softening unverified claims in Stage 7.

    - HEDGE: Add hedging words ("reportedly", "allegedly", "according to some sources")
    - QUALIFY: Add qualifying phrases ("it is believed that", "some evidence suggests")
    - PARENTHETICAL: Add parenthetical markers ("(unverified)", "(needs citation)")
    """

    HEDGE = "hedge"
    QUALIFY = "qualify"
    PARENTHETICAL = "parenthetical"


class GenerationMode(str, Enum):
    """Generation mode for research reports.

    - CLASSICAL: Free-form prose with inline [Title](url) links. Best text quality.
                 Uses existing stream_synthesis(). Skips verification stages 3-6.
    - NATURAL: Light-touch [N] citations with balanced quality + verification.
               Uses NATURAL_GENERATION_PROMPT. Runs verification stages 3-6.
    - STRICT: Heavy [N] constraints. Current behavior with maximum citations.
              Uses INTERLEAVED_GENERATION_PROMPT. Runs verification stages 3-6.
    """

    CLASSICAL = "classical"
    NATURAL = "natural"
    STRICT = "strict"


class SynthesisMode(str, Enum):
    """Synthesis approach for report generation.

    - INTERLEAVED: Current approach - dump evidence into context, LLM generates
                   with [N] markers, claims parsed post-hoc.
    - REACT: ReAct-based synthesis - LLM uses tools to retrieve evidence
             before each factual claim. Enforces grounded generation.
    """

    INTERLEAVED = "interleaved"
    REACT = "react"


class ReactSynthesisConfig(BaseModel):
    """Configuration for ReAct-based synthesis mode."""

    # Tool budget
    max_tool_calls: int = Field(default=40, ge=5, le=250)
    tool_budget_per_section: int = Field(default=10, ge=3, le=50)

    # Grounding settings
    retrieval_window_size: int = Field(
        default=3, ge=1, le=10,
        description="Size of sliding window for grounding inference"
    )
    grounding_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum similarity for claim to be considered grounded"
    )

    # Hybrid grounding check thresholds
    embedding_high_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Similarity above this is automatically grounded"
    )
    embedding_low_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Similarity below this is automatically ungrounded"
    )
    use_llm_judge_for_borderline: bool = Field(
        default=True,
        description="Use LLM judge for borderline similarity cases"
    )

    # Post-processing
    enable_post_processing: bool = Field(
        default=False,
        description="Run coherence polish pass after synthesis (disabled by default)"
    )

    # Section-based synthesis
    use_sectioned_synthesis: bool = Field(
        default=False,
        description="Process research steps as separate sections"
    )

    model_config = {"frozen": True}


class EvidencePreselectionConfig(BaseModel):
    """Configuration for Stage 1: Evidence Pre-Selection."""

    max_spans_per_source: int = Field(default=10, ge=1, le=50)
    min_span_length: int = Field(default=50, ge=10)
    max_span_length: int = Field(default=500, ge=50)
    relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    numeric_content_boost: float = Field(default=0.2, ge=0.0, le=1.0)
    relevance_computation_method: RelevanceMethod = RelevanceMethod.HYBRID

    # Chunking config for long sources (backward compatible defaults)
    chunk_size: int = Field(default=8000, ge=1000, le=20000)
    chunk_overlap: int = Field(default=1000, ge=0, le=5000)
    max_chunks_per_source: int = Field(default=5, ge=1, le=10)

    model_config = {"frozen": True}


class InterleavedGenerationConfig(BaseModel):
    """Configuration for Stage 2: Interleaved Generation."""

    max_claims_per_section: int = Field(default=10, ge=1, le=50)
    min_evidence_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    retry_on_entailment_failure: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)

    model_config = {"frozen": True}


class ConfidenceClassificationConfig(BaseModel):
    """Configuration for Stage 3: Confidence Classification."""

    high_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    quote_match_bonus: float = Field(default=0.3, ge=0.0, le=1.0)
    hedging_word_penalty: float = Field(default=0.2, ge=0.0, le=1.0)
    estimation_method: ConfidenceEstimationMethod = ConfidenceEstimationMethod.LINGUISTIC

    model_config = {"frozen": True}


class IsolatedVerificationConfig(BaseModel):
    """Configuration for Stage 4: Isolated Verification."""

    enable_nei_verdict: bool = True
    verification_model_tier: str = Field(default="analytical")
    quick_verification_tier: str = Field(default="simple")

    model_config = {"frozen": True}


class CitationCorrectionConfig(BaseModel):
    """Configuration for Stage 5: Citation Correction."""

    correction_method: CorrectionMethod = CorrectionMethod.KEYWORD_SEMANTIC_HYBRID
    lambda_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    correction_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    allow_alternate_citations: bool = True

    model_config = {"frozen": True}


class NumericQAVerificationConfig(BaseModel):
    """Configuration for Stage 6: Numeric QA Verification."""

    rounding_tolerance: float = Field(default=0.05, ge=0.0, le=0.5)
    answer_comparison_method: AnswerComparisonMethod = AnswerComparisonMethod.F1
    require_unit_match: bool = True
    require_entity_match: bool = True

    model_config = {"frozen": True}


class VerificationRetrievalConfig(BaseModel):
    """Configuration for Stage 7: ARE-style Verification Retrieval.

    Implements the ARE (Atomic fact decomposition-based Retrieval and Editing) pattern
    for verifying and revising unsupported/partial claims. Based on research from:
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

    # Search budget (per atomic fact, not per claim)
    max_searches_per_fact: int = Field(
        default=2, ge=1, le=5,
        description="Max search attempts per atomic fact (includes reformulations)",
    )
    max_external_urls_per_search: int = Field(
        default=3, ge=1, le=10,
        description="Max URLs to crawl per external search",
    )

    # Entailment thresholds
    entailment_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum entailment score to accept evidence as supporting",
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
        description="Timeout for atomic decomposition LLM call",
    )
    search_timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0,
        description="Timeout for each external search",
    )
    crawl_timeout_seconds: float = Field(
        default=15.0, ge=1.0, le=60.0,
        description="Timeout for web crawling",
    )

    # Model tiers for LLM calls
    decomposition_tier: str = Field(
        default="simple",
        description="Model tier for atomic fact decomposition",
    )
    entailment_tier: str = Field(
        default="analytical",
        description="Model tier for entailment checking",
    )
    reconstruction_tier: str = Field(
        default="analytical",
        description="Model tier for claim reconstruction",
    )

    model_config = {"frozen": True}


class GroundingValidationConfig(BaseModel):
    """Configuration for grounding validation of <analysis> and <free> blocks.

    Validates that:
    - <analysis> blocks are logically derived from preceding <cite> claims
    - <free> blocks contain only structural content (no hidden factual claims)

    Based on SOTA research (FACTS Grounding, FActScore, SAFE).
    """

    enabled: bool = Field(
        default=True,
        description="Enable grounding validation for analysis blocks",
    )
    max_blocks_to_validate: int = Field(
        default=20, ge=1, le=50,
        description="Maximum analysis/free blocks to validate per report",
    )
    min_analysis_length: int = Field(
        default=30, ge=10, le=100,
        description="Minimum character length for analysis block to require validation",
    )
    allow_topic_sentences: bool = Field(
        default=True,
        description="Allow short (<=50 chars) analysis blocks after headers without citations",
    )
    max_preceding_citations: int = Field(
        default=10, ge=1, le=20,
        description="Maximum preceding citations to include in grounding context",
    )
    hedging_prefix: str = Field(
        default="Based on the evidence presented, ",
        description="Hedging prefix for ungrounded analysis",
    )

    model_config = {"frozen": True}


class CitationVerificationConfig(BaseModel):
    """Configuration for the 7-stage citation verification pipeline.

    Stages:
    1. Evidence Pre-Selection - Extract relevant quotes from sources
    2. Interleaved Generation - Generate claims with [N] citations
    3. Confidence Classification - Route claims by confidence level
    4. Isolated Verification - Produce verdicts (supported/partial/unsupported/contradicted)
    5. Citation Correction - Swap citations from existing pool
    6. Numeric QA Verification - Deep verification of numeric claims
    7. ARE Verification Retrieval - Atomic fact decomposition + external search + revision
    """

    # Master toggle
    enabled: bool = True

    # Synthesis mode: controls the overall synthesis approach
    # - "interleaved": Current approach - evidence in context, [N] markers
    # - "react": ReAct-based - LLM uses tools to retrieve evidence before claims
    synthesis_mode: SynthesisMode = SynthesisMode.INTERLEAVED

    # Generation mode: controls synthesis approach and verification stages
    # - "classical": Free-form prose with [Title](url) links, skips verification
    # - "natural": Light-touch [N] citations, runs full verification
    # - "strict": Heavy [N] constraints (current behavior), runs full verification
    # NOTE: Only applies when synthesis_mode=INTERLEAVED
    generation_mode: GenerationMode = GenerationMode.STRICT

    # ReAct synthesis configuration (only applies when synthesis_mode=REACT)
    react_synthesis: ReactSynthesisConfig = Field(
        default_factory=ReactSynthesisConfig
    )

    # Stage toggles (only apply to "natural" and "strict" modes)
    enable_evidence_preselection: bool = True
    enable_interleaved_generation: bool = True
    enable_confidence_classification: bool = True
    enable_citation_correction: bool = True
    enable_numeric_qa_verification: bool = True
    enable_verification_retrieval: bool = False

    # Stage configurations
    evidence_preselection: EvidencePreselectionConfig = Field(
        default_factory=EvidencePreselectionConfig
    )
    interleaved_generation: InterleavedGenerationConfig = Field(
        default_factory=InterleavedGenerationConfig
    )
    confidence_classification: ConfidenceClassificationConfig = Field(
        default_factory=ConfidenceClassificationConfig
    )
    isolated_verification: IsolatedVerificationConfig = Field(
        default_factory=IsolatedVerificationConfig
    )
    citation_correction: CitationCorrectionConfig = Field(
        default_factory=CitationCorrectionConfig
    )
    numeric_qa_verification: NumericQAVerificationConfig = Field(
        default_factory=NumericQAVerificationConfig
    )
    verification_retrieval: VerificationRetrievalConfig = Field(
        default_factory=VerificationRetrievalConfig
    )
    grounding_validation: GroundingValidationConfig = Field(
        default_factory=GroundingValidationConfig
    )

    # Warning thresholds
    unsupported_claim_warning_threshold: float = Field(default=0.20, ge=0.0, le=1.0)

    # Post-verification claim processing (Stage 8)
    enable_free_block_extraction: bool = Field(
        default=True,
        description="Extract claims from <free> blocks that contain factual content",
    )
    enable_claim_removal: bool = Field(
        default=True,
        description="Remove contradicted claims from final report",
    )
    enable_claim_softening: bool = Field(
        default=True,
        description="Soften unsupported claims with hedging language",
    )
    max_free_block_claims: int = Field(
        default=20, ge=0, le=100,
        description="Maximum claims to extract from <free> blocks (0 = unlimited)",
    )
    free_block_min_length: int = Field(
        default=30, ge=10, le=200,
        description="Minimum character length for <free> block to be considered for claim extraction",
    )

    model_config = {"frozen": True}


class RateLimitingConfig(BaseModel):
    """Configuration for rate limit retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=100)
    base_delay_seconds: float = Field(default=2.0, gt=0, le=30)
    max_delay_seconds: float = Field(default=60.0, gt=0, le=300)
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True

    model_config = {"frozen": True}

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed).

        Args:
            attempt: Current attempt number (0 = first retry)

        Returns:
            Delay in seconds (capped at max_delay_seconds)
        """
        if self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay: float = self.base_delay_seconds * (2**attempt)
        else:  # LINEAR
            delay = self.base_delay_seconds * (attempt + 1)

        return min(delay, self.max_delay_seconds)


# =============================================================================
# Research Type Profiles (FR-100)
# =============================================================================


class StepLimits(BaseModel):
    """Step limits for a research type profile."""

    min: int = Field(ge=1, le=20, description="Minimum steps before early completion")
    max: int = Field(ge=1, le=30, description="Maximum steps to execute")
    prompt_guidance: str | None = Field(
        default=None,
        description="Optional guidance text for planner prompt to shape step generation",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_min_max(self) -> "StepLimits":
        """Ensure min does not exceed max."""
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot exceed max ({self.max})")
        return self


class ResearcherTypeConfig(BaseModel):
    """Researcher configuration for a specific research type profile.

    Supports two modes:
    - classic: Single-pass researcher with fixed searches/crawls per step
    - react: ReAct loop where LLM controls tool calls within a budget
    """

    mode: ResearcherMode = Field(
        default=ResearcherMode.CLASSIC,
        description="Researcher implementation: 'react' or 'classic'",
    )
    # Classic mode settings
    max_search_queries: int = Field(
        default=3, ge=1, le=10, description="Max search queries per step (classic mode)"
    )
    max_urls_to_crawl: int = Field(
        default=5, ge=1, le=20, description="Max URLs to crawl per step (classic mode)"
    )
    # ReAct mode settings
    max_tool_calls: int = Field(
        default=15, ge=1, le=50, description="Max tool calls in ReAct loop (react mode)"
    )

    model_config = {"frozen": True}


class ResearchTypeConfig(BaseModel):
    """Complete configuration for a single research type (light/medium/extended).

    This consolidates all research-type-specific settings in one place:
    - Step limits and planner guidance
    - Report word/token limits
    - Researcher mode and limits
    - Citation verification overrides
    """

    steps: StepLimits
    report_limits: ReportLimitConfig
    researcher: ResearcherTypeConfig = Field(default_factory=ResearcherTypeConfig)
    citation_verification: CitationVerificationConfig | None = Field(
        default=None,
        description="Optional per-type overrides for citation verification",
    )

    model_config = {"frozen": True}


class ResearchTypesConfig(BaseModel):
    """Container for all research type profiles (light/medium/extended)."""

    light: ResearchTypeConfig
    medium: ResearchTypeConfig
    extended: ResearchTypeConfig

    model_config = {"frozen": True}

    def get(self, depth: str) -> ResearchTypeConfig:
        """Get configuration for a research depth.

        Args:
            depth: One of 'light', 'medium', 'extended'

        Returns:
            ResearchTypeConfig for the specified depth

        Raises:
            ValueError: If depth is not a valid research type
        """
        if depth == "light":
            return self.light
        elif depth == "medium":
            return self.medium
        elif depth == "extended":
            return self.extended
        else:
            raise ValueError(
                f"Invalid research depth: '{depth}'. Must be 'light', 'medium', or 'extended'"
            )


class AppConfig(BaseModel):
    """Central application configuration loaded from YAML."""

    default_role: str = "analytical"
    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    models: dict[str, ModelRoleConfig] = Field(default_factory=dict)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    citation_verification: CitationVerificationConfig = Field(
        default_factory=CitationVerificationConfig
    )
    # Research type profiles (FR-100) - optional, falls back to legacy if not set
    research_types: ResearchTypesConfig | None = Field(
        default=None,
        description="Research type profiles for light/medium/extended. If not set, uses legacy scattered configs.",
    )

    @model_validator(mode="after")
    def validate_endpoint_references(self) -> "AppConfig":
        """Ensure all role endpoints exist in endpoints dict."""
        errors: list[str] = []

        for role_name, role_config in self.models.items():
            for endpoint_id in role_config.endpoints:
                if endpoint_id not in self.endpoints:
                    errors.append(
                        f"Role '{role_name}' references undefined endpoint: '{endpoint_id}'"
                    )

        if self.default_role and self.models and self.default_role not in self.models:
            errors.append(f"default_role '{self.default_role}' not found in models")

        if errors:
            raise ValueError("\n".join(errors))

        return self

    model_config = {"frozen": True}


def get_default_config() -> AppConfig:
    """Create AppConfig with sensible defaults (no YAML file needed).

    Returns:
        AppConfig with default endpoints and roles for development.
    """
    return AppConfig(
        default_role="analytical",
        endpoints={
            "databricks-llama-70b": EndpointConfig(
                endpoint_identifier="databricks-meta-llama-3-1-70b-instruct",
                max_context_window=128000,
                tokens_per_minute=200000,
            ),
            "databricks-llama-8b": EndpointConfig(
                endpoint_identifier="databricks-meta-llama-3-1-8b-instruct",
                max_context_window=128000,
                tokens_per_minute=300000,
            ),
        },
        models={
            "simple": ModelRoleConfig(
                endpoints=["databricks-llama-8b", "databricks-llama-70b"],
                temperature=0.3,
                max_tokens=4000,
                reasoning_effort=ReasoningEffort.LOW,
            ),
            "analytical": ModelRoleConfig(
                endpoints=["databricks-llama-70b"],
                temperature=0.7,
                max_tokens=8000,
                reasoning_effort=ReasoningEffort.MEDIUM,
            ),
            "complex": ModelRoleConfig(
                endpoints=["databricks-llama-70b"],
                temperature=0.7,
                max_tokens=16000,
                reasoning_effort=ReasoningEffort.HIGH,
                reasoning_budget=8000,
            ),
        },
    )


@lru_cache(maxsize=1)
def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load application configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, searches default locations.

    Returns:
        Validated AppConfig instance

    Note:
        Falls back to default configuration if no config file is found.
        This allows running without explicit configuration in development.
    """
    # Determine config file path
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.info(f"Config file not found at {config_path}, using default configuration")
        return get_default_config()

    try:
        raw_config = load_yaml_config(config_path)
        config = AppConfig.model_validate(raw_config)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def get_app_config() -> AppConfig:
    """Get the cached application configuration.

    This is the primary entry point for accessing configuration.
    Supports APP_CONFIG_PATH environment variable to override default config path.
    """
    config_path_str = os.getenv("APP_CONFIG_PATH")
    if config_path_str:
        return load_app_config(Path(config_path_str))
    return load_app_config()


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing and hot reload)."""
    load_app_config.cache_clear()
