"""Central application configuration loaded from YAML."""

import logging
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from deep_research.core.yaml_loader import load_yaml_config

logger = logging.getLogger(__name__)

# Default config paths
_this_file = Path(__file__).resolve()
_src_root = _this_file.parent.parent.parent  # app_config.py -> core -> deep_research -> src
_project_root = _src_root.parent  # src -> project root
DEFAULT_CONFIG_PATH = _project_root / "config" / "app.yaml"


class ReasoningEffort(StrEnum):
    """Reasoning effort levels for LLM calls."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class SelectionStrategy(StrEnum):
    """Endpoint selection strategy."""

    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class BackoffStrategy(StrEnum):
    """Backoff strategy for rate limit retries."""

    EXPONENTIAL = "exponential"  # delay = base * (2 ** attempt)
    LINEAR = "linear"  # delay = base * (attempt + 1)


class DomainFilterMode(StrEnum):
    """Domain filter operation mode."""

    INCLUDE = "include"  # Whitelist only - only listed domains allowed
    EXCLUDE = "exclude"  # Blacklist only - listed domains blocked
    BOTH = "both"  # Whitelist then blacklist - must be in include AND not in exclude


class ResearcherMode(StrEnum):
    """Researcher implementation mode for research type profiles."""

    REACT = "react"  # ReAct loop with LLM-controlled tool calls
    CLASSIC = "classic"  # Single-pass with fixed searches/crawls per step


class QueryRewriteStrategyConfig(BaseModel):
    """Per-source-type query rewriting strategy configuration.

    Controls how queries are transformed before being sent to each source type.
    Part of enterprise query optimization feature.
    """

    strategy: str = Field(
        default="direct",
        description="Rewrite strategy: direct | multi_query | query2doc | schema_aware | step_back",
    )
    max_alternate_queries: int = Field(
        default=3, ge=1, le=5,
        description="Max alternate queries for multi-query strategy",
    )
    enable_query2doc: bool = Field(
        default=False,
        description="Generate pseudo-doc expansion (works with ALL index types)",
    )
    model_tier: str = Field(
        default="fast",
        description="Model tier for rewriting LLM calls",
    )
    timeout_seconds: float = Field(
        default=10.0, gt=0,
        description="Timeout for each rewrite call",
    )
    fallback_on_failure: bool = Field(
        default=True,
        description="If rewrite fails, use original query (never block research)",
    )

    model_config = {"frozen": True}


class QueryRewritingConfig(BaseModel):
    """Global query rewriting configuration.

    Controls whether and how enterprise tool queries are rewritten
    before execution. Each source type can have its own strategy.
    """

    enabled: bool = Field(
        default=True,
        description="Master toggle for query rewriting",
    )
    strategies: dict[str, QueryRewriteStrategyConfig] = Field(
        default_factory=dict,
        description="Per-source-type strategy configuration",
    )

    model_config = {"frozen": True}


class ParallelToolExecutionConfig(BaseModel):
    """Configuration for parallel tool execution (007-enterprise-data-sources).

    Enables parallel execution of tools from different sources (web, vector_search,
    genie) to reduce latency. Same-source tools may still be serialized by rate
    limiters. Dependencies like (web_crawl -> web_search) are respected.

    Expected performance improvement:
    - Cross-source queries (Web + VS + Genie): 20-40% latency reduction
    - Same-source queries: 10-20% (rate limiters serialize)
    - Single tool: No improvement (no parallelism)
    """

    enabled: bool = Field(
        default=True,
        description="Enable parallel tool execution for cross-source queries",
    )
    max_parallel_per_batch: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent tools per execution batch",
    )
    tool_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Per-tool timeout in seconds",
    )
    batch_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Per-batch timeout in seconds (covers all tools in batch)",
    )

    model_config = {"frozen": True}


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
    # Claude models support prompt caching via cache_control parameter
    supports_prompt_caching: bool = False
    supports_reasoning: bool = True

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

    # Parallel tool execution (007-enterprise-data-sources)
    parallel_tool_execution: ParallelToolExecutionConfig = Field(
        default_factory=ParallelToolExecutionConfig,
        description="Configuration for parallel tool execution",
    )

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
    # DEPRECATED: report_limits moved to research_types.*.report_limits
    # Kept for backward compatibility with configs that don't have research_types
    report_limits: dict[str, ReportLimitConfig] = Field(default_factory=dict)

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
    # Process-wide concurrency cap for the framework's BraveSearchAdapter.
    # Default lowered from 4 → 2 after the NVDA-trace 429 cascade: 7 lanes
    # firing ~5 queries each = ~35 in-flight competing for permits, then
    # bursting when permits free up and tripping Brave's API-side rate
    # limiter. With 2, bursts are throttled at the source.
    max_concurrency: int = Field(default=2, ge=1, le=32)
    # Number of attempts on 429 responses (with exponential backoff + jitter).
    max_retries: int = Field(default=3, ge=1, le=10)
    # Additional intra-permit sleep (uniform random in
    # [0, inter_call_jitter_seconds)) before each request fires. Smooths
    # bursts so consecutive calls inside the semaphore don't hit Brave
    # back-to-back. Set to 0 to disable.
    inter_call_jitter_seconds: float = Field(default=0.15, ge=0.0, le=2.0)

    model_config = {"frozen": True}


class DatabricksSearchConfig(BaseModel):
    """Configuration for Databricks built-in web search (model-serving grounding).

    Built-in search is a *billed model generation* per query (latency/cost far
    exceeds a search REST API), available only on **pay-per-token** endpoints and
    unavailable on provisioned-throughput / HIPAA-BAA / cross-region-disabled
    workspaces. Used only when ``SearchConfig.provider == "databricks"``.
    """

    # Serving endpoint that performs the search. Gemini (single fast call) is the
    # default; gpt-5 (OpenAI Responses) returns direct URLs but is slower/agentic.
    endpoint: str = Field(default="databricks-gemini-3-1-flash-lite")
    # "openai" | "gemini"; auto-detected from the endpoint name when omitted.
    model_family: str | None = Field(default=None)
    max_results: int = Field(default=10, ge=1, le=20)
    # Process-wide cap on concurrent built-in-search generations (heavy calls).
    max_concurrency: int = Field(default=4, ge=1, le=32)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    # Resolve Gemini grounding-redirect URLs to canonical publisher URLs (no-op
    # for the OpenAI path, which already returns direct URLs).
    resolve_redirects: bool = Field(default=True)
    # Push a per-agent INCLUDE-mode allowlist into the OpenAI Responses web_search
    # ``filters.allowed_domains`` (bare domains; OpenAI endpoints only; subdomains
    # auto-included). When False, allowlists rely on the instruction hint + post-hoc
    # URL filter. No effect on Gemini (no API knob) or on exclude mode.
    push_allowed_domains: bool = Field(default=True)
    # Selectable built-in-search endpoints per model family — the single source
    # for (a) the family→default endpoint (first entry = cheapest/default), (b)
    # the designer's per-family endpoint dropdown, and (c) endpoint→family lookup
    # in validation. Only list endpoints that support built-in web search; adding
    # a custom one here makes it selectable in the designer and family-mapped.
    endpoints_by_family: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "gemini": ["databricks-gemini-3-1-flash-lite"],
            "openai": ["databricks-gpt-5-mini", "databricks-gpt-5"],
        }
    )

    model_config = {"frozen": True}

    def default_endpoint_for_family(self, family: str | None) -> str:
        """Cheapest/default endpoint for *family*; the global default otherwise.

        Keeps the endpoint consistent with a declared ``model_family`` so a tool
        that pins the family but omits the endpoint never inherits the global
        (possibly different-family) endpoint — the exact mismatch that drives the
        OpenAI Responses API onto a Gemini endpoint (a hard 400).
        """
        if isinstance(family, str):
            endpoints = self.endpoints_by_family.get(family)
            if endpoints:
                return endpoints[0]
        return self.endpoint

    def family_for_endpoint(self, endpoint: str) -> str | None:
        """Infer the model family of a serving endpoint, or ``None`` if unknown.

        Prefers the explicit :attr:`endpoints_by_family` mapping, then falls back
        to the endpoint-name heuristic (mirrors the framework adapter's private
        ``_detect_family`` in ``tools/builtins/databricks_web_search.py`` but
        returns ``None`` instead of raising on an undetectable name, so callers
        can choose to trust an explicit family for custom endpoints).
        """
        for fam, endpoints in self.endpoints_by_family.items():
            if endpoint in endpoints:
                return fam
        name = endpoint.lower()
        if "gemini" in name:
            return "gemini"
        if any(token in name for token in ("gpt", "openai", "o1", "o3")):
            return "openai"
        return None


class DomainFilterConfig(BaseModel):
    """Configuration for per-agent domain handling.

    Combines two orthogonal mechanisms:

    1. **Binary filter** — ``mode`` + ``include_domains`` / ``exclude_domains``
       run at web-search result time. A URL matching ``exclude_domains`` is
       dropped before admission; in INCLUDE/BOTH mode, URLs must additionally
       match ``include_domains``.

    2. **Reputation ranking** — ``preferred_domains`` / ``deprecated_domains``
       feed the framework's source-admission scorer with signed deltas
       (boost / penalty). They do NOT filter; they only re-rank survivors of
       step (1). A source matching both nets to the sum of deltas.

    All four lists accept the same wildcard syntax (``*.gov``, ``news.*``,
    ``exact.com``). Precedence: exclude → include → reputation.

    Filter modes (binary, unchanged from prior versions):
      * ``include``: Only domains matching include_domains are allowed.
      * ``exclude``: Domains matching exclude_domains are blocked.
      * ``both``: Must match include_domains AND not match exclude_domains.

    All lists default to empty so legacy callers that pre-date the
    reputation fields construct successfully without behaviour change.
    """

    mode: DomainFilterMode = DomainFilterMode.EXCLUDE
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    # Soft reputation ranking signals — never hard-filter; only adjust
    # source admission scores. See databricks_deep_research.agents
    # .source_reputation.SourceReputationScorer for the application logic.
    preferred_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Wildcard domain patterns whose sources receive a positive "
            "admission-score delta. Used for ranking only — does not "
            "filter. Editable per-agent via the designer UI / chat."
        ),
    )
    deprecated_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Wildcard domain patterns whose sources receive a negative "
            "admission-score delta. Used for ranking only — does not "
            "filter. Editable per-agent via the designer UI / chat."
        ),
    )
    log_filtered: bool = False

    model_config = {"frozen": True}

    @property
    def has_reputation_signal(self) -> bool:
        """True if either reputation list contains at least one pattern.

        Callers use this to decide whether to construct a
        ``SourceReputationScorer`` at all — saves an allocation on the
        per-source hot path when no agent has populated reputation.
        """
        return bool(self.preferred_domains) or bool(self.deprecated_domains)


# Out-of-the-box web-search provider. Databricks model-serving built-in web
# search is the default so research works on a Databricks workspace with NO
# external search subscription; "brave"/"jina" are opt-in external APIs that
# require their own API key. Single source of truth for the default — reused by
# the Field default below and every defensive fallback so they cannot drift.
DEFAULT_SEARCH_PROVIDER: Final = "databricks"


class SearchConfig(BaseModel):
    """Configuration for search services."""

    # Active web-search provider for the builtin web_search tool. "databricks"
    # (default) uses model-serving built-in web search; "brave"/"jina" are opt-in
    # external search APIs (each needs a key). Per-workflow YAML / per-agent
    # designer config can still override via the web tool's config.provider.
    provider: Literal["databricks", "brave", "jina"] = Field(
        default=DEFAULT_SEARCH_PROVIDER
    )
    brave: BraveSearchConfig = Field(default_factory=BraveSearchConfig)
    databricks: DatabricksSearchConfig = Field(default_factory=DatabricksSearchConfig)
    domain_filter: DomainFilterConfig = Field(default_factory=DomainFilterConfig)

    model_config = {"frozen": True}


# Supported web-search providers, derived from the SearchConfig.provider Literal
# so the agent-designer registry enum and any other consumer cannot drift from
# the canonical set. Order follows the Literal declaration (databricks first =
# the default provider).
SEARCH_PROVIDERS: tuple[str, ...] = get_args(
    SearchConfig.model_fields["provider"].annotation
)


def resolve_effective_provider(
    tool_provider: object, global_provider: str | None = None
) -> str:
    """Resolve a web tool's effective search provider.

    Precedence (high → low): a non-empty per-tool ``config.provider`` wins; else
    the workspace ``search.provider`` (``global_provider``); else the built-in
    :data:`DEFAULT_SEARCH_PROVIDER`. Centralizes the precedence rule so the
    orchestrator runtime fill and the designer normalizer cannot disagree.
    """
    if isinstance(tool_provider, str) and tool_provider:
        return tool_provider
    if isinstance(global_provider, str) and global_provider:
        return global_provider
    return DEFAULT_SEARCH_PROVIDER


def fill_databricks_search_defaults(
    config: dict[str, Any],
    db: DatabricksSearchConfig,
    *,
    min_results: int = 0,
) -> bool:
    """Fill ABSENT Databricks built-in web-search keys from the app defaults.

    Used by both the designer normalizer and the app orchestrator so a web tool
    that selects ``provider: databricks`` without spelling out the endpoint /
    tuning inherits the workspace ``search.databricks`` block. Only fills keys
    that are absent — it never overwrites an explicit per-tool value (including a
    deliberate ``resolve_redirects: false`` or a smaller ``timeout_seconds``).

    ``min_results`` raises the ``max_results`` floor: ``web_research`` passes its
    ``total_results`` as the search ``count``, and the adapter caps the returned
    count at ``max_results`` — so without this floor a ``total_results: 20`` tool
    would be silently truncated to the default ``max_results`` (10).

    Returns ``True`` if it mutated ``config``.
    """
    before = dict(config)
    # Resolve the endpoint CONSISTENTLY with any declared family: a tool that
    # pins ``model_family`` but omits ``model`` gets THAT family's default
    # endpoint, not the global (possibly different-family) default. Without this,
    # ``model_family: openai`` + the global Gemini endpoint => OpenAI Responses
    # API on a Gemini endpoint => hard 400 and zero search results.
    if "model" not in config:
        config["model"] = db.default_endpoint_for_family(config.get("model_family"))
    if db.model_family is not None:
        config.setdefault("model_family", db.model_family)
    config.setdefault("timeout_seconds", db.timeout_seconds)
    config.setdefault("resolve_redirects", db.resolve_redirects)
    config.setdefault("push_allowed_domains", db.push_allowed_domains)
    if "max_results" not in config:
        config["max_results"] = max(db.max_results, min_results)
    return config != before


class TruncationConfig(BaseModel):
    """Configuration for text truncation limits."""

    log_preview: int = Field(default=200, ge=10)
    error_message: int = Field(default=500, ge=50)
    query_display: int = Field(default=100, ge=10)
    source_snippet: int = Field(default=300, ge=50)

    model_config = {"frozen": True}


class RelevanceMethod(StrEnum):
    """Method for computing relevance scores."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class AnswerComparisonMethod(StrEnum):
    """Method for comparing answers in numeric QA verification."""

    EXACT_MATCH = "exact_match"
    F1 = "f1"
    LERC = "lerc"


class ConfidenceEstimationMethod(StrEnum):
    """Method for estimating confidence levels."""

    LINGUISTIC = "linguistic"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    HYBRID = "hybrid"


class CorrectionMethod(StrEnum):
    """Method for citation correction."""

    KEYWORD_SEMANTIC_HYBRID = "keyword_semantic_hybrid"
    KEYWORD_ONLY = "keyword_only"
    SEMANTIC_ONLY = "semantic_only"


class SofteningStrategy(StrEnum):
    """Strategy for softening unverified claims in Stage 7.

    - HEDGE: Add hedging words ("reportedly", "allegedly", "according to some sources")
    - QUALIFY: Add qualifying phrases ("it is believed that", "some evidence suggests")
    - PARENTHETICAL: Add parenthetical markers ("(unverified)", "(needs citation)")
    """

    HEDGE = "hedge"
    QUALIFY = "qualify"
    PARENTHETICAL = "parenthetical"


class GenerationMode(StrEnum):
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


class SynthesisMode(StrEnum):
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
    # DEPRECATED: use CitationVerificationConfig.max_evidence_chars (the
    # pipeline-wide cap applied to all 5 truncation sites). This stage-
    # specific knob is retained for one release cycle for backward compat
    # — if set in YAML it routes to max_evidence_chars with a
    # DeprecationWarning. Default bumped from 500 to 3000 to match the new
    # pipeline-wide default.
    max_span_length: int = Field(default=3000, ge=50)
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
    verification_model_tier: str = Field(default="bulk_analysis")
    quick_verification_tier: str = Field(default="bulk_analysis")

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
        default="bulk_analysis",
        description="Model tier for atomic fact decomposition (Gemini for analysis)",
    )
    entailment_tier: str = Field(
        default="bulk_analysis",
        description="Model tier for entailment checking (Gemini for NLI)",
    )
    reconstruction_tier: str = Field(
        default="analytical",
        description="Model tier for claim reconstruction (Claude for synthesis quality)",
    )
    softening_tier: str = Field(
        default="fast",
        description="Model tier for softening unverified claims (GPT 5.2 for simple rewrites)",
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


class PostVerificationConfig(BaseModel):
    """Configuration for post-generation verification of structured output.

    Post-verification runs stages 4-6 on claims extracted from structured output
    (e.g., MeetingPrepOutput). This enables citation verification for JSON schemas
    without requiring ReClaim-style interleaved generation.

    Stages used:
    - Stage 4: IsolatedVerifier (CoVe pattern - verify without generation context)
    - Stage 5: CitationCorrector (CiteFix pattern - find better citations)
    - Stage 6: NumericVerifier (QAFactEval pattern - verify numeric claims)
    """

    enabled: bool = Field(
        default=True,
        description="Enable post-verification for structured output",
    )
    max_claims_to_verify: int = Field(
        default=50, ge=1, le=200,
        description="Maximum number of claims to verify (high priority claims first)",
    )
    include_stage4_isolation: bool = Field(
        default=True,
        description="Run Stage 4 (IsolatedVerifier) for CoVe-style verification",
    )
    include_stage5_correction: bool = Field(
        default=True,
        description="Run Stage 5 (CitationCorrector) to find better citations",
    )
    include_stage6_numeric: bool = Field(
        default=True,
        description="Run Stage 6 (NumericVerifier) for numeric claim verification",
    )
    confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum confidence threshold for verification acceptance",
    )
    skip_low_priority_claims: bool = Field(
        default=True,
        description="Skip low-priority claims under 300 chars to reduce API calls",
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
    post_verification: PostVerificationConfig = Field(
        default_factory=PostVerificationConfig,
        description="Configuration for post-generation verification of structured output",
    )

    # Concurrency for NLI verification (Stage 4)
    max_concurrent_verifications: int = Field(
        default=10, ge=1, le=50,
        description="Max concurrent NLI verification calls (Stage 4). Higher = faster but more API load.",
    )

    # Warning thresholds
    unsupported_claim_warning_threshold: float = Field(default=0.20, ge=0.0, le=1.0)

    # Post-verification claim processing (Stage 8)
    enable_free_block_extraction: bool = Field(
        default=True,
        description="Extract claims from <free> blocks that contain factual content",
    )
    claim_disposition: dict[str, str] = Field(
        default_factory=lambda: {
            "supported": "keep",
            "partial": "keep",
            "unsupported": "remove",
            "contradicted": "remove",
            "abstained": "keep",
            "analysis_partial": "keep",
            "analysis_unsupported": "remove",
        },
        description="Stage 8: Maps each verdict to an action (keep, remove, soften).",
    )

    @field_validator("claim_disposition")
    @classmethod
    def _validate_claim_disposition(cls, v: dict[str, str]) -> dict[str, str]:
        valid_actions = {"keep", "remove", "soften"}
        for key, action in v.items():
            if action not in valid_actions:
                raise ValueError(
                    f"claim_disposition[{key!r}] = {action!r}, must be one of {valid_actions}"
                )
        return v
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


class PromptCachingConfig(BaseModel):
    """Configuration for prompt caching (Claude models on Databricks).

    Prompt caching reduces costs by up to 90% on cached content. Works by storing
    the KV cache for common prefixes (like system prompts) and reusing it across
    requests. Claude models on Databricks support this via the cache_control parameter.

    Implementation is transparent - higher layers (agents) don't know about caching.
    The LLM client transforms messages internally when caching is enabled.
    """

    # Master toggle - enables/disables all prompt caching
    enabled: bool = Field(
        default=True,
        description="Enable prompt caching for supported endpoints",
    )
    # Minimum tokens required to cache (Claude requires 1024)
    min_tokens_threshold: int = Field(
        default=1024,
        ge=128,
        le=10000,
        description="Minimum estimated tokens before caching is applied",
    )
    # Cache type for Claude (ephemeral = 5 min TTL, refreshes on use)
    cache_type: str = Field(
        default="ephemeral",
        description="Cache type to use (ephemeral for Claude)",
    )
    # Which message types to cache
    cache_system_prompt: bool = Field(
        default=True,
        description="Apply cache_control to system messages",
    )
    # Observability
    log_cache_usage: bool = Field(
        default=True,
        description="Log when cache_control is applied",
    )

    model_config = {"frozen": True}


# =============================================================================
# Query Mode Configuration (Tiered Query Modes)
# =============================================================================


class QueryModeResearcherConfig(BaseModel):
    """Researcher configuration for a specific query mode.

    Used to override default researcher settings when running in web_search mode.
    """

    mode: ResearcherMode = Field(
        default=ResearcherMode.CLASSIC,
        description="Researcher implementation: 'react' or 'classic'",
    )
    max_search_queries: int = Field(
        default=2, ge=1, le=10, description="Max search queries"
    )
    max_urls_to_crawl: int = Field(
        default=3, ge=1, le=20, description="Max URLs to crawl"
    )

    model_config = {"frozen": True}


class QueryModeCitationConfig(BaseModel):
    """Citation verification configuration for a specific query mode.

    Used to override default citation settings for lightweight modes.
    """

    enabled: bool = True
    generation_mode: GenerationMode = GenerationMode.NATURAL
    enable_numeric_qa_verification: bool = False
    enable_verification_retrieval: bool = False

    model_config = {"frozen": True}


class QueryModeConfig(BaseModel):
    """Configuration for a single query mode (simple, web_search, deep_research).

    Query modes determine the processing pipeline:
    - simple: Direct LLM response, no web search, no research session
    - web_search: Quick search with 2-5 sources, lightweight session
    - deep_research: Full research pipeline with plan, steps, verification
    """

    model_role: str = Field(
        default="analytical",
        description="Model tier to use for this mode (simple, analytical, complex)",
    )
    emit_events: bool = Field(
        default=True,
        description="Whether to emit streaming events for progress tracking",
    )
    create_session: bool = Field(
        default=True,
        description="Whether to create a research session in the database",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Total timeout for the request (per attempt for web_search)",
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum retry attempts on timeout (for web_search mode)",
    )

    # Note: Web search mode routing is handled programmatically in orchestrator
    # (creates synthetic 1-step plan, skips coordinator/reflector in code)
    # No skip_* flags needed - they were a superseded design approach.

    # For deep_research mode: inherit from research_types
    use_research_types: bool = Field(
        default=False,
        description="Inherit configuration from research_types section",
    )

    # Mode-specific overrides
    researcher: QueryModeResearcherConfig | None = Field(
        default=None,
        description="Researcher configuration overrides for this mode",
    )
    citation_verification: QueryModeCitationConfig | None = Field(
        default=None,
        description="Citation verification overrides for this mode",
    )

    model_config = {"frozen": True}


class QueryModesConfig(BaseModel):
    """Container for all query mode configurations."""

    simple: QueryModeConfig = Field(
        default_factory=lambda: QueryModeConfig(
            model_role="simple",
            emit_events=False,
            create_session=False,
            timeout_seconds=30,
        )
    )
    web_search: QueryModeConfig = Field(
        default_factory=lambda: QueryModeConfig(
            model_role="analytical",
            emit_events=True,
            create_session=True,
            timeout_seconds=15,
            researcher=QueryModeResearcherConfig(
                mode=ResearcherMode.CLASSIC,
                max_search_queries=2,
                max_urls_to_crawl=3,
            ),
            citation_verification=QueryModeCitationConfig(
                enabled=True,
                generation_mode=GenerationMode.NATURAL,
                enable_numeric_qa_verification=False,
                enable_verification_retrieval=False,
            ),
        )
    )
    deep_research: QueryModeConfig = Field(
        default_factory=lambda: QueryModeConfig(
            model_role="complex",
            emit_events=True,
            create_session=True,
            use_research_types=True,
        )
    )

    model_config = {"frozen": True}

    def get(self, mode: str) -> QueryModeConfig:
        """Get configuration for a query mode.

        Args:
            mode: One of 'simple', 'web_search', 'deep_research'

        Returns:
            QueryModeConfig for the specified mode

        Raises:
            ValueError: If mode is not a valid query mode
        """
        if mode == "simple":
            return self.simple
        elif mode == "web_search":
            return self.web_search
        elif mode == "deep_research":
            return self.deep_research
        else:
            raise ValueError(
                f"Invalid query mode: '{mode}'. Must be 'simple', 'web_search', or 'deep_research'"
            )


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


# =============================================================================
# Vector Search Configuration (US1)
# =============================================================================


class VectorSearchEndpointConfig(BaseModel):
    """Configuration for a single Vector Search endpoint.

    Example YAML:
        product_docs:
          endpoint_name: vs-endpoint-prod
          index_name: catalog.schema.product_docs_index
          columns: ["title", "content", "url"]
          description: Search product documentation
          num_results: 5
    """

    endpoint_name: str = Field(
        description="Databricks Vector Search endpoint name",
    )
    index_name: str = Field(
        description="Fully qualified index name (catalog.schema.index)",
    )
    columns: list[str] = Field(
        default_factory=lambda: ["title", "content", "url"],
        description="Columns to return from search results",
    )
    tool_name: str | None = Field(
        default=None,
        description="Custom tool name. Defaults to 'search_{endpoint_name}'",
    )
    description: str | None = Field(
        default=None,
        description="Custom description for LLM",
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Default number of results to return",
    )
    filters: dict[str, object] | None = Field(
        default=None,
        description="Optional filters to apply to all searches",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this endpoint is enabled",
    )

    model_config = {"frozen": True}


class VectorSearchConfig(BaseModel):
    """Configuration for Vector Search integration.

    Example YAML:
        vector_search:
          enabled: true
          endpoints:
            product_docs:
              endpoint_name: vs-endpoint-prod
              index_name: catalog.schema.product_docs_index
            api_reference:
              endpoint_name: vs-endpoint-api
              index_name: catalog.schema.api_docs_index
    """

    enabled: bool = Field(
        default=False,
        description="Whether Vector Search is enabled globally",
    )
    endpoints: dict[str, VectorSearchEndpointConfig] = Field(
        default_factory=dict,
        description="Vector Search endpoints by name",
    )

    model_config = {"frozen": True}


# =============================================================================
# Knowledge Assistant Configuration (US2)
# =============================================================================


class KnowledgeAssistantEndpointConfig(BaseModel):
    """Configuration for a single Knowledge Assistant endpoint.

    Example YAML:
        product_assistant:
          endpoint_name: product-knowledge-assistant
          description: Ask questions about our products
    """

    endpoint_name: str = Field(
        description="Databricks serving endpoint name",
    )
    tool_name: str | None = Field(
        default=None,
        description="Custom tool name. Defaults to 'ask_{endpoint_name}'",
    )
    description: str | None = Field(
        default=None,
        description="Custom description for LLM",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this endpoint is enabled",
    )

    model_config = {"frozen": True}


class KnowledgeAssistantsConfig(BaseModel):
    """Configuration for Knowledge Assistant integration.

    Example YAML:
        knowledge_assistants:
          enabled: true
          endpoints:
            product_assistant:
              endpoint_name: product-knowledge-assistant
    """

    enabled: bool = Field(
        default=False,
        description="Whether Knowledge Assistants are enabled globally",
    )
    endpoints: dict[str, KnowledgeAssistantEndpointConfig] = Field(
        default_factory=dict,
        description="Knowledge Assistant endpoints by name",
    )

    model_config = {"frozen": True}


# =============================================================================
# Plugin Configuration
# =============================================================================


class PluginConfig(BaseModel):
    """Configuration for a single plugin.

    Plugin-specific configuration is stored as a dict to allow
    plugins to define their own configuration schema.
    """

    enabled: bool = Field(
        default=True,
        description="Whether this plugin is enabled",
    )
    settings: dict[str, object] = Field(
        default_factory=dict,
        description="Plugin-specific settings (schema defined by plugin)",
    )

    model_config = {"frozen": True}


class LifecycleHooksConfig(BaseModel):
    """Configuration for plugin lifecycle hooks feature.

    Controls gradual rollout and performance tuning of lifecycle callbacks.

    Gradual rollout strategy (via YAML config):
    - Phase 1: enabled: false (default, no hooks)
    - Phase 2: enabled: true, allowlist: [on_job_submitted]
    - Phase 3: allowlist: [on_job_submitted, on_job_completed, on_job_failed]
    - Phase 4: allowlist: [] (empty = all enabled)
    - Phase 5: Remove feature flag entirely (always enabled)
    """

    enabled: bool = Field(
        default=False,
        description="Enable lifecycle hooks (experimental - gradual rollout)",
    )
    allowlist: set[str] = Field(
        default_factory=set,
        description="Only these hooks are enabled (empty = all allowed)",
    )
    denylist: set[str] = Field(
        default_factory=set,
        description="These hooks are explicitly disabled",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for plugin hooks (prevents infinite hangs)",
    )
    max_sync_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Max thread pool workers for sync hooks",
    )

    model_config = {"frozen": True}


class PluginsConfig(BaseModel):
    """Container for all plugin configurations.

    Example YAML:
        plugins:
          lifecycle_hooks:
            enabled: true
            allowlist: ["on_job_submitted", "on_job_completed"]
            timeout_seconds: 30
          vector_search:
            enabled: true
            settings:
              endpoint_name: my-vs-endpoint
              index_name: my-index
          knowledge_assistant:
            enabled: true
            settings:
              endpoint_name: my-ka-endpoint
    """

    # Dict of plugin_name -> PluginConfig
    # This allows arbitrary plugin names to be configured
    configs: dict[str, PluginConfig] = Field(
        default_factory=dict,
        description="Per-plugin configuration by name",
    )

    # Lifecycle hooks configuration
    lifecycle_hooks: LifecycleHooksConfig = Field(
        default_factory=LifecycleHooksConfig,
        description="Plugin lifecycle callback configuration",
    )

    model_config = {"frozen": True}

    def get(self, plugin_name: str) -> PluginConfig | None:
        """Get configuration for a specific plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            PluginConfig if found, None otherwise
        """
        return self.configs.get(plugin_name)

    def is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if plugin exists and is enabled, False otherwise
        """
        config = self.configs.get(plugin_name)
        return config.enabled if config else True  # Default to enabled if not configured


class JobConfig(BaseModel):
    """Configuration for background research job management.

    Controls concurrency limits, heartbeat intervals, and zombie detection.
    These settings can be overridden in config/app.yaml or config/app.e2e.yaml.
    """

    max_concurrent_per_user: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Maximum concurrent research jobs per user",
    )
    heartbeat_interval_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Interval between heartbeat updates for active jobs",
    )
    zombie_threshold_seconds: int = Field(
        default=30,
        ge=5,
        le=600,
        description="Seconds without heartbeat before a job is considered a zombie",
    )

    model_config = {"frozen": True}


class AgentDesignerToolCatalogConfig(BaseModel):
    """Designer tool-catalog rendering configuration."""

    max_chars: int = Field(
        default=4000,
        ge=1,
        le=20000,
        description="Hard character ceiling for rendered tool catalogs",
    )
    summary_only_above_n_tools: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Render summary-only catalog entries above this tool count",
    )
    include_probes: bool = Field(
        default=True,
        description="Include sanitized SafeProbe samples in rendered catalogs",
    )

    model_config = {"frozen": True}


class AgentDesignerProbeConfig(BaseModel):
    """Designer SafeProbe sampling configuration."""

    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        description="Per-tool SafeProbe timeout",
    )
    max_concurrent_probes: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Maximum SafeProbe calls to run concurrently",
    )
    max_output_chars: int = Field(
        default=800,
        ge=0,
        le=20000,
        description="Maximum sanitized probe output characters per tool",
    )
    persist: bool = Field(
        default=False,
        description="Default probe persistence policy; false avoids storing samples",
    )

    model_config = {"frozen": True}


class AgentDesignerConfig(BaseModel):
    """Agent Designer configuration."""

    tool_catalog: AgentDesignerToolCatalogConfig = Field(
        default_factory=AgentDesignerToolCatalogConfig
    )
    probe: AgentDesignerProbeConfig = Field(default_factory=AgentDesignerProbeConfig)

    model_config = {"frozen": True}


class AppConfig(BaseModel):
    """Central application configuration loaded from YAML."""

    default_role: str = "analytical"
    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    models: dict[str, ModelRoleConfig] = Field(default_factory=dict)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    prompt_caching: PromptCachingConfig = Field(default_factory=PromptCachingConfig)
    citation_verification: CitationVerificationConfig = Field(
        default_factory=CitationVerificationConfig
    )
    # Research type profiles (FR-100) - optional, falls back to legacy if not set
    research_types: ResearchTypesConfig | None = Field(
        default=None,
        description="Research type profiles for light/medium/extended. If not set, uses legacy scattered configs.",
    )
    # Query mode configuration (Tiered Query Modes feature)
    query_modes: QueryModesConfig = Field(
        default_factory=QueryModesConfig,
        description="Query mode configurations for simple/web_search/deep_research.",
    )
    # Plugin configuration
    plugins: PluginsConfig = Field(
        default_factory=PluginsConfig,
        description="Plugin configurations for extending Deep Research Agent.",
    )
    # Vector Search configuration (US1)
    vector_search: VectorSearchConfig = Field(
        default_factory=VectorSearchConfig,
        description="Vector Search integration configuration.",
    )
    # Knowledge Assistants configuration (US2)
    knowledge_assistants: KnowledgeAssistantsConfig = Field(
        default_factory=KnowledgeAssistantsConfig,
        description="Knowledge Assistants integration configuration.",
    )
    # Job management configuration
    jobs: JobConfig = Field(
        default_factory=JobConfig,
        description="Background job management configuration (concurrency limits, heartbeats).",
    )
    # Query rewriting configuration (enterprise query optimization)
    query_rewriting: QueryRewritingConfig = Field(
        default_factory=QueryRewritingConfig,
        description="Source-specific query rewriting configuration for enterprise tools.",
    )
    # Agent Designer catalog/probe controls
    agent_designer: AgentDesignerConfig = Field(
        default_factory=AgentDesignerConfig,
        description="Agent Designer tool catalog and SafeProbe settings.",
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

    Note:
        Uses modern Databricks-hosted model endpoints (Claude, GPT-5, Gemini).
        These are the current production endpoints as of 2025.
    """
    return AppConfig(
        default_role="analytical",
        endpoints={
            "haiku": EndpointConfig(
                endpoint_identifier="databricks-claude-haiku-4-5",
                max_context_window=128000,
                tokens_per_minute=50000,
                supports_structured_output=True,
                supports_prompt_caching=True,
            ),
            "sonnet": EndpointConfig(
                endpoint_identifier="databricks-claude-sonnet-4-6",
                max_context_window=128000,
                tokens_per_minute=50000,
                supports_structured_output=True,
                supports_prompt_caching=True,
            ),
            "opus": EndpointConfig(
                endpoint_identifier="databricks-claude-opus-4-7",
                max_context_window=128000,
                tokens_per_minute=200000,
                supports_structured_output=True,
                supports_prompt_caching=True,
            ),
            "gpt5": EndpointConfig(
                endpoint_identifier="databricks-gpt-5-5",
                max_context_window=128000,
                tokens_per_minute=200000,
                supports_structured_output=True,
                supports_temperature=False,
            ),
            "gpt5mini": EndpointConfig(
                endpoint_identifier="databricks-gpt-5-mini",
                max_context_window=128000,
                tokens_per_minute=50000,
                supports_structured_output=True,
                supports_temperature=False,
            ),
        },
        models={
            "simple": ModelRoleConfig(
                endpoints=["haiku", "gpt5mini"],
                temperature=0.7,
                max_tokens=8000,
                reasoning_effort=ReasoningEffort.LOW,
            ),
            "analytical": ModelRoleConfig(
                endpoints=["haiku", "gpt5mini"],
                temperature=0.7,
                max_tokens=8000,
                reasoning_effort=ReasoningEffort.MEDIUM,
            ),
            "complex": ModelRoleConfig(
                endpoints=["opus", "sonnet", "gpt5"],
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
