"""Unit tests for ModelConfig.get_or_create_endpoint().

Tests the dual-resolution strategy:
1. YAML alias lookup returns rich metadata
2. Unknown strings create ad-hoc ModelEndpoint with conservative defaults
"""

from unittest.mock import MagicMock, patch

from deep_research.services.llm.types import ModelEndpoint


class TestGetOrCreateEndpoint:
    """Tests for ModelConfig.get_or_create_endpoint()."""

    @patch("deep_research.services.llm.config.get_app_config")
    def test_yaml_alias_returns_configured_endpoint(
        self, mock_get_config: MagicMock
    ) -> None:
        """Known YAML alias returns the YAML-configured ModelEndpoint."""
        mock_config = MagicMock()
        mock_config.endpoints = {
            "haiku": MagicMock(
                endpoint_identifier="databricks-claude-haiku-4-5",
                max_context_window=200000,
                tokens_per_minute=100000,
                temperature=None,
                max_tokens=None,
                reasoning_effort=None,
                reasoning_budget=None,
                supports_structured_output=True,
                supports_temperature=True,
                supports_prompt_caching=True,
            ),
        }
        mock_config.models = {}
        mock_config.default_role = "analytical"
        mock_get_config.return_value = mock_config

        from deep_research.services.llm.config import ModelConfig

        config = ModelConfig()

        endpoint = config.get_or_create_endpoint("haiku")

        assert endpoint.id == "haiku"
        assert endpoint.endpoint_identifier == "databricks-claude-haiku-4-5"
        assert endpoint.max_context_window == 200000
        assert endpoint.supports_structured_output is True
        assert endpoint.supports_prompt_caching is True

    @patch("deep_research.services.llm.config.get_app_config")
    def test_direct_identifier_returns_adhoc_endpoint(
        self, mock_get_config: MagicMock
    ) -> None:
        """Unknown string returns ad-hoc ModelEndpoint with safe defaults."""
        mock_config = MagicMock()
        mock_config.endpoints = {}
        mock_config.models = {}
        mock_config.default_role = "analytical"
        mock_get_config.return_value = mock_config

        from deep_research.services.llm.config import ModelConfig

        config = ModelConfig()

        endpoint = config.get_or_create_endpoint("my-custom-fine-tuned-llama")

        assert isinstance(endpoint, ModelEndpoint)
        assert endpoint.id == "my-custom-fine-tuned-llama"
        assert endpoint.endpoint_identifier == "my-custom-fine-tuned-llama"

    @patch("deep_research.services.llm.config.get_app_config")
    def test_adhoc_has_correct_defaults(
        self, mock_get_config: MagicMock
    ) -> None:
        """Ad-hoc endpoint: structured_output=False, temperature=True, caching=False."""
        mock_config = MagicMock()
        mock_config.endpoints = {}
        mock_config.models = {}
        mock_config.default_role = "analytical"
        mock_get_config.return_value = mock_config

        from deep_research.services.llm.config import ModelConfig

        config = ModelConfig()

        endpoint = config.get_or_create_endpoint("some-arbitrary-endpoint")

        assert endpoint.max_context_window == 128_000
        assert endpoint.tokens_per_minute == 50_000
        assert endpoint.supports_structured_output is False
        assert endpoint.supports_temperature is True
        assert endpoint.supports_prompt_caching is False

    @patch("deep_research.services.llm.config.get_app_config")
    def test_adhoc_id_equals_identifier(
        self, mock_get_config: MagicMock
    ) -> None:
        """Ad-hoc: both id and endpoint_identifier are the input string."""
        mock_config = MagicMock()
        mock_config.endpoints = {}
        mock_config.models = {}
        mock_config.default_role = "analytical"
        mock_get_config.return_value = mock_config

        from deep_research.services.llm.config import ModelConfig

        config = ModelConfig()
        name = "databricks-meta-llama-3-1-70b-instruct"

        endpoint = config.get_or_create_endpoint(name)

        assert endpoint.id == name
        assert endpoint.endpoint_identifier == name

    @patch("deep_research.services.llm.config.get_app_config")
    def test_yaml_alias_preferred_over_adhoc(
        self, mock_get_config: MagicMock
    ) -> None:
        """When alias exists in YAML, return YAML config (not ad-hoc)."""
        mock_config = MagicMock()
        mock_config.endpoints = {
            "llama": MagicMock(
                endpoint_identifier="databricks-meta-llama-3-1-70b-instruct",
                max_context_window=128000,
                tokens_per_minute=200000,
                temperature=None,
                max_tokens=None,
                reasoning_effort=None,
                reasoning_budget=None,
                supports_structured_output=False,
                supports_temperature=True,
                supports_prompt_caching=False,
            ),
        }
        mock_config.models = {}
        mock_config.default_role = "analytical"
        mock_get_config.return_value = mock_config

        from deep_research.services.llm.config import ModelConfig

        config = ModelConfig()

        endpoint = config.get_or_create_endpoint("llama")

        # Should return the YAML-configured one with 200k TPM, not ad-hoc 50k
        assert endpoint.tokens_per_minute == 200000
        assert endpoint.endpoint_identifier == "databricks-meta-llama-3-1-70b-instruct"
