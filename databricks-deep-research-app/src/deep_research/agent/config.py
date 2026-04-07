"""Agent configuration accessors - loads from central AppConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deep_research.core.app_config import (
    BackgroundConfig,
    CitationVerificationConfig,
    CoordinatorConfig,
    ParallelToolExecutionConfig,
    PlannerConfig,
    QueryModeConfig,
    QueryModesConfig,
    ReportLimitConfig,
    ResearcherConfig,
    ResearcherMode,
    ResearcherTypeConfig,
    ResearchTypeConfig,
    StepLimits,
    SynthesizerConfig,
    get_app_config,
)


def get_researcher_config() -> ResearcherConfig:
    """Get Researcher agent configuration."""
    return get_app_config().agents.researcher


def get_planner_config() -> PlannerConfig:
    """Get Planner agent configuration."""
    return get_app_config().agents.planner


def get_coordinator_config() -> CoordinatorConfig:
    """Get Coordinator agent configuration."""
    return get_app_config().agents.coordinator


def get_synthesizer_config() -> SynthesizerConfig:
    """Get Synthesizer agent configuration."""
    return get_app_config().agents.synthesizer


def get_background_config() -> BackgroundConfig:
    """Get Background Investigator agent configuration."""
    return get_app_config().agents.background


def get_parallel_tool_execution_config() -> ParallelToolExecutionConfig:
    """Get parallel tool execution configuration.

    Returns:
        ParallelToolExecutionConfig with enabled, timeouts, and limits.
    """
    return get_app_config().agents.researcher.parallel_tool_execution


def get_truncation_limit(limit_name: str) -> int:
    """Get a truncation limit by name.

    Args:
        limit_name: One of 'log_preview', 'error_message', 'query_display', 'source_snippet'

    Returns:
        The configured truncation limit

    Raises:
        AttributeError: If limit_name is not a valid truncation limit
    """
    config = get_app_config().truncation
    value: int = getattr(config, limit_name)
    return value


# =============================================================================
# Research Type Profile Accessors (FR-100)
# =============================================================================

# Legacy fallback mapping (used when research_types is not configured)
_LEGACY_DEPTH_TO_STEPS: dict[str, tuple[int, int]] = {
    "light": (1, 3),
    "medium": (3, 6),
    "extended": (5, 10),
}


def _build_legacy_research_type_config(depth: str) -> ResearchTypeConfig:
    """Build ResearchTypeConfig from legacy scattered configs.

    This is used for backward compatibility when research_types is not defined
    in app.yaml.

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        ResearchTypeConfig constructed from legacy configs
    """
    config = get_app_config()

    # Get step limits from legacy mapping
    min_steps, max_steps = _LEGACY_DEPTH_TO_STEPS.get(depth, (3, 6))

    # Get report limits from legacy synthesizer config
    report_limits = config.agents.synthesizer.report_limits.get(
        depth,
        ReportLimitConfig(min_words=400, max_words=800, max_tokens=2000),
    )

    # Build researcher config from legacy global config
    researcher_config = ResearcherTypeConfig(
        mode=ResearcherMode.CLASSIC,
        max_search_queries=config.agents.researcher.max_search_queries,
        max_urls_to_crawl=config.agents.researcher.max_urls_to_crawl,
        max_tool_calls=15,  # Default for legacy
    )

    return ResearchTypeConfig(
        steps=StepLimits(min=min_steps, max=max_steps),
        report_limits=report_limits,
        researcher=researcher_config,
        citation_verification=None,  # Use global config
    )


def get_research_type_config(depth: str) -> ResearchTypeConfig:
    """Get complete configuration for a research depth.

    This is the PRIMARY accessor for research-type-specific settings.
    Falls back to legacy scattered configs if research_types is not defined.

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        ResearchTypeConfig with all settings for this depth
    """
    config = get_app_config()

    # New path: use research_types if defined
    if config.research_types is not None:
        return config.research_types.get(depth)

    # Legacy fallback: construct from scattered configs
    return _build_legacy_research_type_config(depth)


def get_step_limits(depth: str) -> StepLimits:
    """Get step limits for a research depth.

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        StepLimits with min, max, and optional prompt_guidance
    """
    return get_research_type_config(depth).steps


def get_report_limits(depth: str) -> ReportLimitConfig:
    """Get report word/token limits for a research depth.

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        ReportLimitConfig with min_words, max_words, max_tokens
    """
    return get_research_type_config(depth).report_limits


def get_researcher_config_for_depth(depth: str) -> ResearcherTypeConfig:
    """Get researcher configuration for a research depth.

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        ResearcherTypeConfig with mode, max_search_queries, max_urls_to_crawl, max_tool_calls
    """
    return get_research_type_config(depth).researcher


def get_citation_config_for_depth(depth: str) -> CitationVerificationConfig:
    """Get citation verification config for a research depth.

    Merges per-depth overrides with global config. Fields defined in
    per-depth override global; unspecified fields inherit from global.

    The merge logic:
    1. Start with global config as base
    2. For each field in per-type config:
       - If it differs from Pydantic default → it was explicitly set → use it
       - If it matches Pydantic default → inherit from global

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        Merged CitationVerificationConfig
    """
    config = get_app_config()
    global_config = config.citation_verification
    type_config = get_research_type_config(depth)

    # If no per-type overrides, return global config
    if type_config.citation_verification is None:
        return global_config

    per_type = type_config.citation_verification

    # Deep merge: global as base, per-type overrides on top.
    # Use model_fields_set to reliably detect explicitly set fields,
    # instead of comparing against defaults (which fails when a user
    # explicitly sets a value that happens to match the default).
    global_dict = global_config.model_dump()
    explicitly_set = per_type.model_fields_set

    merged = global_dict.copy()
    for field_name in CitationVerificationConfig.model_fields:
        if field_name in explicitly_set:
            merged[field_name] = getattr(per_type, field_name)

    return CitationVerificationConfig(**merged)


# =============================================================================
# Query Mode Accessors (Tiered Query Modes)
# =============================================================================


def get_query_modes_config() -> QueryModesConfig:
    """Get all query mode configurations.

    Returns:
        QueryModesConfig containing simple, web_search, and deep_research configs
    """
    return get_app_config().query_modes


def get_query_mode_config(mode: str) -> QueryModeConfig:
    """Get configuration for a specific query mode.

    Args:
        mode: One of 'simple', 'web_search', 'deep_research'

    Returns:
        QueryModeConfig for the specified mode

    Raises:
        ValueError: If mode is not a valid query mode
    """
    return get_app_config().query_modes.get(mode)


def get_query_rewrite_config(source_type: str) -> QueryRewriteConfig | None:
    """Get query rewriting config for a source type from app config.

    Converts the YAML-level QueryRewriteStrategyConfig into the tool-level
    QueryRewriteConfig that the rewriter module expects.

    Args:
        source_type: Source type identifier (vector_search, genie, knowledge_assistant).

    Returns:
        QueryRewriteConfig if rewriting is enabled for this source type, None otherwise.
    """
    from deep_research.agent.tools.query_rewriter import QueryRewriteConfig

    app_config = get_app_config()
    rewriting_config = app_config.query_rewriting

    if not rewriting_config.enabled:
        return None

    strategy_config = rewriting_config.strategies.get(source_type)
    if not strategy_config:
        return None

    return QueryRewriteConfig(
        enabled=True,
        strategy=strategy_config.strategy,
        max_alternate_queries=strategy_config.max_alternate_queries,
        enable_query2doc=strategy_config.enable_query2doc,
        model_tier=strategy_config.model_tier,
        timeout_seconds=strategy_config.timeout_seconds,
        fallback_on_failure=strategy_config.fallback_on_failure,
    )


if TYPE_CHECKING:
    from deep_research.agent.state import ResearchState
    from deep_research.agent.tools.query_rewriter import QueryRewriteConfig
    from deep_research.services.llm.types import ModelTier


def get_endpoint_override(state: ResearchState, tier: ModelTier) -> str | None:
    """Get endpoint override for a tier from state's model_overrides.

    Args:
        state: Current research state.
        tier: Model tier to check for override.

    Returns:
        Endpoint ID string if override is set, None otherwise.
    """
    if not state.model_overrides:
        return None
    return state.model_overrides.get(tier.value)
