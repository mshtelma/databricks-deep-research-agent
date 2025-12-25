"""Unit tests for agent configuration accessors."""

import pytest

from src.agent.config import (
    get_background_config,
    get_coordinator_config,
    get_planner_config,
    get_researcher_config,
    get_synthesizer_config,
    get_truncation_limit,
)
from src.core.app_config import (
    BackgroundConfig,
    CoordinatorConfig,
    PlannerConfig,
    ResearcherConfig,
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
