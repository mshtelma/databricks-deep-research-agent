"""Unit tests for central application configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_research.core.app_config import (
    SEARCH_PROVIDERS,
    _deep_merge_dicts,
    AgentsConfig,
    AppConfig,
    BraveSearchConfig,
    CoordinatorConfig,
    DatabricksSearchConfig,
    EndpointConfig,
    ModelRoleConfig,
    OrchestrationSettingsConfig,
    PlannerConfig,
    ReasoningEffort,
    ResearcherConfig,
    SearchConfig,
    SelectionStrategy,
    SynthesizerConfig,
    TruncationConfig,
    clear_config_cache,
    fill_databricks_search_defaults,
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

    def test_default_provider_is_databricks(self) -> None:
        """Default provider is Databricks built-in search (Brave is opt-in)."""
        sc = SearchConfig()
        assert sc.provider == "databricks"
        assert isinstance(sc.databricks, DatabricksSearchConfig)
        assert sc.databricks.endpoint
        assert sc.databricks.resolve_redirects is True

    def test_resolve_effective_provider_precedence(self) -> None:
        """Per-tool provider wins; blank inherits the global; else the default."""
        from deep_research.core.app_config import (
            DEFAULT_SEARCH_PROVIDER,
            resolve_effective_provider,
        )

        assert DEFAULT_SEARCH_PROVIDER == "databricks"
        # Explicit per-tool provider wins over the global default.
        assert resolve_effective_provider("brave", "databricks") == "brave"
        assert resolve_effective_provider("jina", "brave") == "jina"
        # Blank / empty / None inherits the workspace global.
        assert resolve_effective_provider(None, "brave") == "brave"
        assert resolve_effective_provider("", "databricks") == "databricks"
        # No global supplied → built-in default.
        assert resolve_effective_provider(None, None) == "databricks"
        assert resolve_effective_provider(None) == "databricks"

    def test_provider_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError):
            SearchConfig(provider="bing")  # type: ignore[arg-type]

    def test_databricks_provider_parses(self) -> None:
        sc = SearchConfig(
            provider="databricks",
            databricks={"endpoint": "databricks-gpt-5", "max_results": 5,
                        "timeout_seconds": 45},
        )
        assert sc.provider == "databricks"
        assert sc.databricks.endpoint == "databricks-gpt-5"
        assert sc.databricks.max_results == 5
        assert sc.databricks.timeout_seconds == 45

    def test_search_providers_is_canonical_and_derived(self) -> None:
        """SEARCH_PROVIDERS mirrors the provider Literal (single source)."""
        from typing import get_args

        assert set(SEARCH_PROVIDERS) == {"brave", "jina", "databricks"}
        assert set(SEARCH_PROVIDERS) == set(
            get_args(SearchConfig.model_fields["provider"].annotation)
        )
        assert SEARCH_PROVIDERS[0] == "databricks"  # default first

    def test_fill_databricks_defaults_fills_absent(self) -> None:
        db = DatabricksSearchConfig(endpoint="databricks-gpt-5")
        cfg: dict[str, object] = {}
        changed = fill_databricks_search_defaults(cfg, db, min_results=15)
        assert changed is True
        assert cfg["model"] == "databricks-gpt-5"
        assert cfg["timeout_seconds"] == db.timeout_seconds
        assert cfg["resolve_redirects"] is True
        assert cfg["push_allowed_domains"] is True
        assert cfg["max_results"] == 15  # min_results floor applied

    def test_fill_databricks_defaults_preserves_present_values(self) -> None:
        """Present keys (incl. falsy resolve_redirects=False / small max_results)
        are never overwritten — proves the absent-only contract."""
        db = DatabricksSearchConfig()
        cfg = {
            "model": "x",
            "resolve_redirects": False,
            "timeout_seconds": 5.0,
            "max_results": 3,
            "push_allowed_domains": False,
        }
        changed = fill_databricks_search_defaults(cfg, db, min_results=99)
        assert changed is False
        assert cfg == {
            "model": "x",
            "resolve_redirects": False,
            "timeout_seconds": 5.0,
            "max_results": 3,
            "push_allowed_domains": False,
        }

    def test_fill_databricks_defaults_max_results_floor(self) -> None:
        db = DatabricksSearchConfig(max_results=10)
        cfg: dict[str, object] = {}
        fill_databricks_search_defaults(cfg, db, min_results=5)
        assert cfg["max_results"] == 10  # max(app default 10, floor 5)

    def test_push_allowed_domains_default_true(self) -> None:
        assert DatabricksSearchConfig().push_allowed_domains is True

    def test_fill_databricks_defaults_preserves_push_flag_false(self) -> None:
        db = DatabricksSearchConfig()
        cfg: dict[str, object] = {"push_allowed_domains": False}
        fill_databricks_search_defaults(cfg, db)
        assert cfg["push_allowed_domains"] is False

    def test_endpoints_by_family_default(self) -> None:
        """Per-family endpoint map: first entry is the cheapest/default."""
        db = DatabricksSearchConfig()
        assert (
            db.endpoints_by_family["gemini"][0] == "databricks-gemini-3-1-flash-lite"
        )
        assert db.endpoints_by_family["openai"][0] == "databricks-gpt-5-mini"

    def test_default_endpoint_for_family(self) -> None:
        db = DatabricksSearchConfig()
        assert db.default_endpoint_for_family("openai") == "databricks-gpt-5-mini"
        assert (
            db.default_endpoint_for_family("gemini")
            == "databricks-gemini-3-1-flash-lite"
        )
        # Unknown family / None falls back to the global default endpoint.
        assert db.default_endpoint_for_family(None) == db.endpoint
        assert db.default_endpoint_for_family("anthropic") == db.endpoint

    def test_family_for_endpoint(self) -> None:
        db = DatabricksSearchConfig()
        # Mapped endpoints resolve via the explicit map.
        assert db.family_for_endpoint("databricks-gemini-3-1-flash-lite") == "gemini"
        assert db.family_for_endpoint("databricks-gpt-5-mini") == "openai"
        # Unmapped endpoints resolve via the name heuristic.
        assert db.family_for_endpoint("some-custom-gpt-4o") == "openai"
        assert db.family_for_endpoint("vendor-gemini-x") == "gemini"
        # Undetectable names return None (caller trusts an explicit family).
        assert db.family_for_endpoint("acme-search-v2") is None

    def test_fill_family_only_uses_family_default_endpoint(self) -> None:
        """A tool that pins model_family but omits the endpoint gets THAT
        family's default endpoint — not the global (Gemini) default. This is
        the fix for the openai-family-on-a-Gemini-endpoint 400."""
        db = DatabricksSearchConfig()  # global endpoint = gemini-lite
        cfg: dict[str, object] = {"provider": "databricks", "model_family": "openai"}
        fill_databricks_search_defaults(cfg, db)
        assert cfg["model"] == "databricks-gpt-5-mini"
        assert cfg["model_family"] == "openai"

    def test_fill_no_family_uses_global_endpoint(self) -> None:
        db = DatabricksSearchConfig()
        cfg: dict[str, object] = {"provider": "databricks"}
        fill_databricks_search_defaults(cfg, db)
        assert cfg["model"] == db.endpoint  # gemini-lite global default

    def test_fill_explicit_model_not_overwritten(self) -> None:
        """An explicit (even contradictory) endpoint is preserved — design-time
        validation, not the fill, rejects a mismatch."""
        db = DatabricksSearchConfig()
        cfg: dict[str, object] = {
            "model": "databricks-gemini-3-1-flash-lite",
            "model_family": "openai",
        }
        fill_databricks_search_defaults(cfg, db)
        assert cfg["model"] == "databricks-gemini-3-1-flash-lite"


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


class TestOrchestrationSettingsConfig:
    """Tests for the research-run timeout watchdog config + wiring."""

    def test_default_timeout_is_1800(self) -> None:
        assert OrchestrationSettingsConfig().research_timeout_seconds == 1800

    def test_app_config_orchestration_default(self) -> None:
        assert AppConfig().orchestration.research_timeout_seconds == 1800

    def test_rejects_below_minimum(self) -> None:
        with pytest.raises(ValueError):
            OrchestrationSettingsConfig(research_timeout_seconds=30)

    def test_rejects_above_maximum(self) -> None:
        with pytest.raises(ValueError):
            OrchestrationSettingsConfig(research_timeout_seconds=20000)

    def test_orchestration_config_default_reads_app_yaml(self) -> None:
        """OrchestrationConfig pulls its default from app.yaml (1800);
        an explicit constructor value still wins."""
        from deep_research.agent.orchestration_config import OrchestrationConfig

        clear_config_cache()
        assert OrchestrationConfig().research_timeout_seconds == 1800
        assert (
            OrchestrationConfig(research_timeout_seconds=999).research_timeout_seconds
            == 999
        )


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self) -> None:
        """Test default config is valid."""
        config = get_default_config()
        assert isinstance(config, AppConfig)
        assert config.default_role == "analytical"
        # Check that at least one endpoint is configured
        assert len(config.endpoints) > 0
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


class TestDeepMergeDicts:
    """Tests for the _deep_merge_dicts overlay helper."""

    def test_nested_leaf_replaced_siblings_preserved(self) -> None:
        base = {
            "search": {
                "provider": "databricks",
                "databricks": {"endpoint": "gemini", "max_results": 10},
            }
        }
        overlay = {"search": {"databricks": {"endpoint": "gpt-5"}}}
        merged = _deep_merge_dicts(base, overlay)
        assert merged["search"]["databricks"]["endpoint"] == "gpt-5"
        # Siblings at every nesting level are preserved.
        assert merged["search"]["databricks"]["max_results"] == 10
        assert merged["search"]["provider"] == "databricks"

    def test_list_value_is_replaced_not_merged(self) -> None:
        base = {"models": {"simple": {"endpoints": ["a", "b"]}}}
        overlay = {"models": {"simple": {"endpoints": ["c"]}}}
        merged = _deep_merge_dicts(base, overlay)
        assert merged["models"]["simple"]["endpoints"] == ["c"]

    def test_new_key_added(self) -> None:
        assert _deep_merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_inputs_not_mutated(self) -> None:
        base = {"search": {"databricks": {"endpoint": "gemini"}}}
        _deep_merge_dicts(base, {"search": {"databricks": {"endpoint": "gpt-5"}}})
        assert base["search"]["databricks"]["endpoint"] == "gemini"


class TestConfigOverlay:
    """Tests for base + partial-overlay config layering (APP_CONFIG_OVERLAY)."""

    def setup_method(self) -> None:
        clear_config_cache()

    def _write(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(content)
        f.flush()
        f.close()
        return Path(f.name)

    def test_overlay_overrides_only_target_leaf(self) -> None:
        base = self._write(
            "search:\n"
            "  provider: databricks\n"
            "  databricks:\n"
            "    endpoint: databricks-gemini-3-1-flash-lite\n"
            "    max_results: 10\n"
            "    timeout_seconds: 30\n"
        )
        overlay = self._write(
            "search:\n  databricks:\n    endpoint: databricks-gpt-5-4-mini\n"
        )
        try:
            config = load_app_config(base, overlay)
            # Overridden leaf:
            assert config.search.databricks.endpoint == "databricks-gpt-5-4-mini"
            # Untouched siblings inherit from the base config:
            assert config.search.databricks.max_results == 10
            assert config.search.databricks.timeout_seconds == 30
            assert config.search.provider == "databricks"
        finally:
            os.unlink(base)
            os.unlink(overlay)

    def test_missing_overlay_falls_back_to_base(self) -> None:
        base = self._write(
            "search:\n  databricks:\n    endpoint: databricks-gemini-3-1-flash-lite\n"
        )
        missing = Path(tempfile.gettempdir()) / "no-such-app-overlay-xyz.yaml"
        try:
            config = load_app_config(base, missing)
            assert (
                config.search.databricks.endpoint
                == "databricks-gemini-3-1-flash-lite"
            )
        finally:
            os.unlink(base)

    def test_no_overlay_leaves_base_unchanged(self) -> None:
        base = self._write(
            "search:\n  databricks:\n    endpoint: databricks-gemini-3-1-flash-lite\n"
        )
        try:
            config = load_app_config(base, None)
            assert (
                config.search.databricks.endpoint
                == "databricks-gemini-3-1-flash-lite"
            )
        finally:
            os.unlink(base)

    def test_overlay_raises_research_timeout(self) -> None:
        """The fevm-style overlay raises orchestration.research_timeout_seconds
        without disturbing other sections (e.g. search)."""
        base = self._write(
            "search:\n"
            "  databricks:\n"
            "    endpoint: databricks-gpt-5-4-mini\n"
            "orchestration:\n"
            "  research_timeout_seconds: 1800\n"
        )
        overlay = self._write(
            "orchestration:\n  research_timeout_seconds: 3600\n"
        )
        try:
            config = load_app_config(base, overlay)
            assert config.orchestration.research_timeout_seconds == 3600
            # Untouched section inherits from base:
            assert config.search.databricks.endpoint == "databricks-gpt-5-4-mini"
        finally:
            os.unlink(base)
            os.unlink(overlay)
