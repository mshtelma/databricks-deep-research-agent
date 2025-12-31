"""Agent configuration accessors - loads from central AppConfig."""

from src.core.app_config import (
    BackgroundConfig,
    CitationVerificationConfig,
    CoordinatorConfig,
    PlannerConfig,
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

    If the research type has per-type citation_verification overrides,
    returns those. Otherwise returns the global citation_verification config.

    Args:
        depth: One of 'light', 'medium', 'extended'

    Returns:
        CitationVerificationConfig (per-type if defined, else global)
    """
    config = get_app_config()
    type_config = get_research_type_config(depth)

    # If per-type config exists, use it
    if type_config.citation_verification is not None:
        return type_config.citation_verification

    # Otherwise use global config
    return config.citation_verification
