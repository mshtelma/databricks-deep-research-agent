"""Agent configuration accessors - loads from central AppConfig."""

from src.core.app_config import (
    BackgroundConfig,
    CoordinatorConfig,
    PlannerConfig,
    ResearcherConfig,
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
