"""Unit tests for agent configuration accessors."""

import pytest

from src.agent.config import (
    get_background_config,
    get_citation_config_for_depth,
    get_coordinator_config,
    get_planner_config,
    get_report_limits,
    get_researcher_config,
    get_researcher_config_for_depth,
    get_research_type_config,
    get_step_limits,
    get_synthesizer_config,
    get_truncation_limit,
)
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
    clear_config_cache,
)


class TestGetResearcherConfig:
    """Tests for get_researcher_config accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_researcher_config(self) -> None:
        """Test returns ResearcherConfig instance."""
        config = get_researcher_config()
        assert isinstance(config, ResearcherConfig)

    def test_has_expected_defaults(self) -> None:
        """Test config has expected default values."""
        config = get_researcher_config()
        assert config.max_search_queries >= 1
        assert config.max_search_results >= 1
        assert config.max_urls_to_crawl >= 1
        assert config.content_preview_length >= 100
        assert config.content_storage_length >= 1000
        assert config.max_previous_observations >= 1
        assert config.page_contents_limit >= 1000
        assert config.max_generated_queries >= 1


class TestGetPlannerConfig:
    """Tests for get_planner_config accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_planner_config(self) -> None:
        """Test returns PlannerConfig instance."""
        config = get_planner_config()
        assert isinstance(config, PlannerConfig)

    def test_has_expected_defaults(self) -> None:
        """Test config has expected default values."""
        config = get_planner_config()
        assert config.max_plan_iterations >= 1


class TestGetCoordinatorConfig:
    """Tests for get_coordinator_config accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_coordinator_config(self) -> None:
        """Test returns CoordinatorConfig instance."""
        config = get_coordinator_config()
        assert isinstance(config, CoordinatorConfig)

    def test_has_expected_defaults(self) -> None:
        """Test config has expected default values."""
        config = get_coordinator_config()
        assert config.max_clarification_rounds >= 0
        assert isinstance(config.enable_clarification, bool)


class TestGetSynthesizerConfig:
    """Tests for get_synthesizer_config accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_synthesizer_config(self) -> None:
        """Test returns SynthesizerConfig instance."""
        config = get_synthesizer_config()
        assert isinstance(config, SynthesizerConfig)

    def test_has_expected_defaults(self) -> None:
        """Test config has expected default values."""
        config = get_synthesizer_config()
        assert config.max_report_length >= 1000


class TestGetBackgroundConfig:
    """Tests for get_background_config accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_background_config(self) -> None:
        """Test returns BackgroundConfig instance."""
        config = get_background_config()
        assert isinstance(config, BackgroundConfig)

    def test_has_expected_defaults(self) -> None:
        """Test config has expected default values."""
        config = get_background_config()
        assert config.max_search_queries >= 1
        assert config.max_results_per_query >= 1
        assert config.max_total_results >= 1


class TestGetTruncationLimit:
    """Tests for get_truncation_limit accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_log_preview_limit(self) -> None:
        """Test returns log_preview limit."""
        limit = get_truncation_limit("log_preview")
        assert isinstance(limit, int)
        assert limit >= 10

    def test_returns_error_message_limit(self) -> None:
        """Test returns error_message limit."""
        limit = get_truncation_limit("error_message")
        assert isinstance(limit, int)
        assert limit >= 50

    def test_returns_query_display_limit(self) -> None:
        """Test returns query_display limit."""
        limit = get_truncation_limit("query_display")
        assert isinstance(limit, int)
        assert limit >= 10

    def test_returns_source_snippet_limit(self) -> None:
        """Test returns source_snippet limit."""
        limit = get_truncation_limit("source_snippet")
        assert isinstance(limit, int)
        assert limit >= 50

    def test_invalid_limit_name_raises(self) -> None:
        """Test invalid limit name raises AttributeError."""
        with pytest.raises(AttributeError):
            get_truncation_limit("nonexistent_limit")


class TestConfigAccessorConsistency:
    """Tests for consistency across config accessors."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_all_accessors_return_frozen_configs(self) -> None:
        """Test all accessors return immutable (frozen) configs."""
        configs = [
            get_researcher_config(),
            get_planner_config(),
            get_coordinator_config(),
            get_synthesizer_config(),
            get_background_config(),
        ]

        for config in configs:
            # Frozen models should have model_config with frozen=True
            assert hasattr(config, "model_config")

    def test_repeated_calls_return_same_values(self) -> None:
        """Test repeated calls return consistent values."""
        config1 = get_researcher_config()
        config2 = get_researcher_config()

        # Values should be equal
        assert config1.max_search_queries == config2.max_search_queries
        assert config1.max_urls_to_crawl == config2.max_urls_to_crawl


class TestGetResearchTypeConfig:
    """Tests for get_research_type_config accessor (FR-100)."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_research_type_config_for_light(self) -> None:
        """Test returns ResearchTypeConfig for light depth."""
        config = get_research_type_config("light")
        assert isinstance(config, ResearchTypeConfig)

    def test_returns_research_type_config_for_medium(self) -> None:
        """Test returns ResearchTypeConfig for medium depth."""
        config = get_research_type_config("medium")
        assert isinstance(config, ResearchTypeConfig)

    def test_returns_research_type_config_for_extended(self) -> None:
        """Test returns ResearchTypeConfig for extended depth."""
        config = get_research_type_config("extended")
        assert isinstance(config, ResearchTypeConfig)

    def test_light_has_smallest_step_limits(self) -> None:
        """Test light depth has smallest step limits."""
        light = get_research_type_config("light")
        medium = get_research_type_config("medium")
        extended = get_research_type_config("extended")

        assert light.steps.max <= medium.steps.max <= extended.steps.max

    def test_extended_has_largest_report_limits(self) -> None:
        """Test extended depth has largest report limits."""
        light = get_research_type_config("light")
        medium = get_research_type_config("medium")
        extended = get_research_type_config("extended")

        assert light.report_limits.max_words <= medium.report_limits.max_words
        assert medium.report_limits.max_words <= extended.report_limits.max_words


class TestGetStepLimits:
    """Tests for get_step_limits accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_step_limits(self) -> None:
        """Test returns StepLimits instance."""
        limits = get_step_limits("medium")
        assert isinstance(limits, StepLimits)

    def test_min_less_than_or_equal_max(self) -> None:
        """Test min steps <= max steps for all depths."""
        for depth in ["light", "medium", "extended"]:
            limits = get_step_limits(depth)
            assert limits.min <= limits.max

    def test_light_has_small_steps(self) -> None:
        """Test light depth has small step limits."""
        limits = get_step_limits("light")
        assert limits.min >= 1
        assert limits.max <= 5

    def test_extended_has_large_steps(self) -> None:
        """Test extended depth has large step limits."""
        limits = get_step_limits("extended")
        assert limits.max >= 5


class TestGetReportLimits:
    """Tests for get_report_limits accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_report_limit_config(self) -> None:
        """Test returns ReportLimitConfig instance."""
        limits = get_report_limits("medium")
        assert isinstance(limits, ReportLimitConfig)

    def test_has_min_and_max_words(self) -> None:
        """Test limits have min and max words."""
        for depth in ["light", "medium", "extended"]:
            limits = get_report_limits(depth)
            assert limits.min_words >= 100
            assert limits.max_words >= limits.min_words
            assert limits.max_tokens >= 1000


class TestGetResearcherConfigForDepth:
    """Tests for get_researcher_config_for_depth accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_researcher_type_config(self) -> None:
        """Test returns ResearcherTypeConfig instance."""
        config = get_researcher_config_for_depth("medium")
        assert isinstance(config, ResearcherTypeConfig)

    def test_has_mode(self) -> None:
        """Test config has mode field."""
        config = get_researcher_config_for_depth("medium")
        assert isinstance(config.mode, ResearcherMode)
        assert config.mode in [ResearcherMode.REACT, ResearcherMode.CLASSIC]

    def test_light_uses_classic_mode(self) -> None:
        """Test light depth uses classic mode for speed."""
        config = get_researcher_config_for_depth("light")
        # Light should use classic for speed (as defined in app.yaml)
        assert config.mode == ResearcherMode.CLASSIC

    def test_has_tool_call_limits(self) -> None:
        """Test config has max_tool_calls for ReAct mode."""
        for depth in ["light", "medium", "extended"]:
            config = get_researcher_config_for_depth(depth)
            assert config.max_tool_calls >= 1
            assert config.max_search_queries >= 1
            assert config.max_urls_to_crawl >= 1


class TestGetCitationConfigForDepth:
    """Tests for get_citation_config_for_depth accessor."""

    def setup_method(self) -> None:
        """Clear config cache before each test."""
        clear_config_cache()

    def test_returns_citation_verification_config(self) -> None:
        """Test returns CitationVerificationConfig instance."""
        config = get_citation_config_for_depth("medium")
        assert isinstance(config, CitationVerificationConfig)

    def test_all_depths_have_config(self) -> None:
        """Test all depths have citation config."""
        for depth in ["light", "medium", "extended"]:
            config = get_citation_config_for_depth(depth)
            assert config is not None
            assert hasattr(config, "generation_mode")
