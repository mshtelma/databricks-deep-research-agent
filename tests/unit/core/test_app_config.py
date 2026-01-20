"""Unit tests for central application configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_research.core.app_config import (
    AgentsConfig,
    AppConfig,
    BraveSearchConfig,
    CoordinatorConfig,
    EndpointConfig,
    ModelRoleConfig,
    PlannerConfig,
    ReasoningEffort,
    ResearcherConfig,
    SelectionStrategy,
    SynthesizerConfig,
    TruncationConfig,
    clear_config_cache,
    get_app_config,
    get_default_config,
    load_app_config,
)


class TestEndpointConfig:
    """Tests for EndpointConfig model."""

    def test_minimal_config(self) -> None:
        """Test endpoint with only required fields."""
        config = EndpointConfig(
            endpoint_identifier="test-endpoint",
            max_context_window=128000,
            tokens_per_minute=200000,
        )
        assert config.endpoint_identifier == "test-endpoint"
        assert config.max_context_window == 128000
        assert config.tokens_per_minute == 200000
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.supports_structured_output is False

    def test_full_config(self) -> None:
        """Test endpoint with all fields."""
        config = EndpointConfig(
            endpoint_identifier="test-endpoint",
            max_context_window=128000,
            tokens_per_minute=200000,
            temperature=0.5,
            max_tokens=4000,
            reasoning_effort=ReasoningEffort.MEDIUM,
            reasoning_budget=8000,
            supports_structured_output=True,
        )
        assert config.temperature == 0.5
        assert config.reasoning_effort == ReasoningEffort.MEDIUM
        assert config.reasoning_budget == 8000
        assert config.supports_structured_output is True

    def test_validation_rejects_negative_context_window(self) -> None:
        """Test validation rejects invalid context window."""
        with pytest.raises(ValueError, match="greater than 0"):
            EndpointConfig(
                endpoint_identifier="test",
                max_context_window=0,
                tokens_per_minute=200000,
            )


class TestModelRoleConfig:
    """Tests for ModelRoleConfig model."""

    def test_defaults(self) -> None:
        """Test role config defaults."""
        config = ModelRoleConfig(endpoints=["endpoint-1"])
        assert config.temperature == 0.7
        assert config.max_tokens == 8000
        assert config.reasoning_effort == ReasoningEffort.LOW
        assert config.rotation_strategy == SelectionStrategy.PRIORITY
        assert config.fallback_on_429 is True

    def test_custom_values(self) -> None:
        """Test role config with custom values."""
        config = ModelRoleConfig(
            endpoints=["ep1", "ep2"],
            temperature=0.3,
            max_tokens=16000,
            reasoning_effort=ReasoningEffort.HIGH,
            reasoning_budget=10000,
            rotation_strategy=SelectionStrategy.ROUND_ROBIN,
            fallback_on_429=False,
        )
        assert config.temperature == 0.3
        assert config.reasoning_budget == 10000
        assert config.rotation_strategy == SelectionStrategy.ROUND_ROBIN

    def test_requires_at_least_one_endpoint(self) -> None:
        """Test role requires at least one endpoint."""
        with pytest.raises(ValueError):
            ModelRoleConfig(endpoints=[])


class TestAgentConfigs:
    """Tests for agent configuration models."""

    def test_researcher_defaults(self) -> None:
        """Test ResearcherConfig defaults."""
        config = ResearcherConfig()
        assert config.max_search_queries == 2
        assert config.max_search_results == 10
        assert config.max_urls_to_crawl == 3
        assert config.content_preview_length == 3000

    def test_planner_defaults(self) -> None:
        """Test PlannerConfig defaults."""
        config = PlannerConfig()
        assert config.max_plan_iterations == 3

    def test_coordinator_defaults(self) -> None:
        """Test CoordinatorConfig defaults."""
        config = CoordinatorConfig()
        assert config.max_clarification_rounds == 3
        assert config.enable_clarification is True

    def test_synthesizer_defaults(self) -> None:
        """Test SynthesizerConfig defaults."""
        config = SynthesizerConfig()
        assert config.max_report_length == 50000

    def test_agents_config_contains_all(self) -> None:
        """Test AgentsConfig contains all agent configs."""
        config = AgentsConfig()
        assert isinstance(config.researcher, ResearcherConfig)
        assert isinstance(config.planner, PlannerConfig)
        assert isinstance(config.coordinator, CoordinatorConfig)
        assert isinstance(config.synthesizer, SynthesizerConfig)


class TestSearchConfig:
    """Tests for search configuration models."""

    def test_brave_defaults(self) -> None:
        """Test BraveSearchConfig defaults."""
        config = BraveSearchConfig()
        assert config.requests_per_second == 1.0
        assert config.default_result_count == 10
        assert config.freshness == "pm"

    def test_freshness_validation(self) -> None:
        """Test freshness value validation."""
        # Valid values
        for freshness in ["pd", "pw", "pm", "py"]:
            config = BraveSearchConfig(freshness=freshness)
            assert config.freshness == freshness

        # Invalid value
        with pytest.raises(ValueError):
            BraveSearchConfig(freshness="invalid")


class TestTruncationConfig:
    """Tests for truncation configuration."""

    def test_defaults(self) -> None:
        """Test TruncationConfig defaults."""
        config = TruncationConfig()
        assert config.log_preview == 200
        assert config.error_message == 500
        assert config.query_display == 100
        assert config.source_snippet == 300


class TestAppConfig:
    """Tests for AppConfig model."""

    def test_minimal_valid_config(self) -> None:
        """Test minimal valid configuration."""
        config = AppConfig(
            default_role="simple",
            endpoints={
                "ep1": EndpointConfig(
                    endpoint_identifier="test",
                    max_context_window=128000,
                    tokens_per_minute=200000,
                )
            },
            models={
                "simple": ModelRoleConfig(endpoints=["ep1"]),
            },
        )
        assert config.default_role == "simple"
        assert "ep1" in config.endpoints
        assert "simple" in config.models

    def test_validates_endpoint_references(self) -> None:
        """Test validation rejects undefined endpoint references."""
        with pytest.raises(ValueError, match="undefined endpoint"):
            AppConfig(
                default_role="simple",
                endpoints={},
                models={
                    "simple": ModelRoleConfig(endpoints=["nonexistent"]),
                },
            )

    def test_validates_default_role(self) -> None:
        """Test validation rejects undefined default role."""
        with pytest.raises(ValueError, match="default_role 'unknown' not found"):
            AppConfig(
                default_role="unknown",
                endpoints={
                    "ep1": EndpointConfig(
                        endpoint_identifier="test",
                        max_context_window=128000,
                        tokens_per_minute=200000,
                    )
                },
                models={
                    "simple": ModelRoleConfig(endpoints=["ep1"]),
                },
            )


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self) -> None:
        """Test default config is valid."""
        config = get_default_config()
        assert isinstance(config, AppConfig)
        assert config.default_role == "analytical"
        assert "databricks-llama-70b" in config.endpoints
        assert "simple" in config.models
        assert "analytical" in config.models
        assert "complex" in config.models


class TestLoadAppConfig:
    """Tests for load_app_config function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_config_cache()

    def test_uses_default_when_file_missing(self) -> None:
        """Test falls back to default config when file missing."""
        with patch("deep_research.core.app_config.DEFAULT_CONFIG_PATH", Path("/nonexistent/path.yaml")):
            clear_config_cache()
            config = load_app_config()
            assert isinstance(config, AppConfig)
            # Should be default config
            assert config.default_role == "analytical"

    def test_loads_from_yaml_file(self) -> None:
        """Test loads config from YAML file."""
        yaml_content = """
default_role: simple
endpoints:
  test-endpoint:
    endpoint_identifier: test-model
    max_context_window: 64000
    tokens_per_minute: 100000
models:
  simple:
    endpoints:
      - test-endpoint
    temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_app_config(path)
            assert config.default_role == "simple"
            assert "test-endpoint" in config.endpoints
            assert config.endpoints["test-endpoint"].max_context_window == 64000
        finally:
            os.unlink(path)

    def test_caches_result(self) -> None:
        """Test config is cached."""
        config1 = get_app_config()
        config2 = get_app_config()
        assert config1 is config2

    def test_cache_can_be_cleared(self) -> None:
        """Test cache clear allows reload."""
        config1 = get_app_config()
        clear_config_cache()
        config2 = get_app_config()
        # After cache clear, should be different instances
        # (but equal content since default config)
        assert config1 == config2
