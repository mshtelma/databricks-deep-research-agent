"""Unit tests for the LLM adapter (llm_adapter.py).

Tests that `create_framework_llm_client()` correctly extracts the
AsyncOpenAI client and builds the model tier mapping from the app's
LLMClient and ModelConfig.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_research.services.llm.types import ModelEndpoint, ModelRole, ModelTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_endpoint(endpoint_id: str, identifier: str) -> ModelEndpoint:
    """Create a minimal ModelEndpoint for testing."""
    return ModelEndpoint(
        id=endpoint_id,
        endpoint_identifier=identifier,
        max_context_window=128_000,
        tokens_per_minute=200_000,
    )


def _make_role(name: str, endpoint_ids: list[str]) -> ModelRole:
    """Create a minimal ModelRole for testing."""
    return ModelRole(name=name, endpoints=endpoint_ids)


def _make_mock_config(
    roles: dict[str, ModelRole],
    endpoints: dict[str, ModelEndpoint],
) -> MagicMock:
    """Build a mock ModelConfig that responds to get_role / get_endpoint.

    The adapter iterates ALL ModelTier members and calls ``get_role(tier)``
    where ``tier`` is a ``ModelTier`` enum member (a ``str`` subclass).
    Missing tiers raise ``KeyError`` (caught by the adapter's except clause).
    """
    config = MagicMock()

    def get_role(name: str) -> ModelRole:
        # ModelTier is a str enum, so dict lookup with enum key works
        if name not in roles:
            raise KeyError(f"Unknown role: {name}")
        return roles[name]

    def get_endpoint(eid: str) -> ModelEndpoint:
        if eid not in endpoints:
            raise KeyError(f"Unknown endpoint: {eid}")
        return endpoints[eid]

    config.get_role = MagicMock(side_effect=get_role)
    config.get_endpoint = MagicMock(side_effect=get_endpoint)
    return config


def _make_mock_llm_client(
    config: MagicMock,
    openai_client: MagicMock | None = None,
) -> MagicMock:
    """Build a mock app LLMClient."""
    llm = MagicMock()
    llm._config = config
    llm._ensure_fresh_client = MagicMock(return_value=openai_client or MagicMock())
    return llm


# ---------------------------------------------------------------------------
# Tests — _build_model_mapping
# ---------------------------------------------------------------------------


class TestBuildModelMapping:
    """Tests for _build_model_mapping."""

    def test_maps_all_tiers(self) -> None:
        """Each ModelTier maps to the primary endpoint identifier."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep_simple = _make_endpoint("ep-simple", "databricks-llama-8b")
        ep_analytical = _make_endpoint("ep-analytical", "databricks-llama-70b")
        ep_complex = _make_endpoint("ep-complex", "databricks-o3-mini")

        roles = {
            "simple": _make_role("simple", ["ep-simple"]),
            "analytical": _make_role("analytical", ["ep-analytical"]),
            "complex": _make_role("complex", ["ep-complex"]),
        }
        endpoints = {
            "ep-simple": ep_simple,
            "ep-analytical": ep_analytical,
            "ep-complex": ep_complex,
        }
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        mapping = _build_model_mapping(llm)

        assert mapping["simple"] == "databricks-llama-8b"
        assert mapping["analytical"] == "databricks-llama-70b"
        assert mapping["complex"] == "databricks-o3-mini"

    def test_uses_primary_endpoint_only(self) -> None:
        """When a role has multiple endpoints, the first is used."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep1 = _make_endpoint("ep-primary", "model-primary")
        ep2 = _make_endpoint("ep-fallback", "model-fallback")
        roles = {"analytical": _make_role("analytical", ["ep-primary", "ep-fallback"])}
        endpoints = {"ep-primary": ep1, "ep-fallback": ep2}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        mapping = _build_model_mapping(llm)

        assert mapping["analytical"] == "model-primary"

    def test_overrides_applied(self) -> None:
        """Model overrides replace the config-derived mapping."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep = _make_endpoint("ep-analytical", "databricks-llama-70b")
        roles = {"analytical": _make_role("analytical", ["ep-analytical"])}
        endpoints = {"ep-analytical": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        overrides = {"analytical": "my-custom-model", "simple": "my-simple-model"}
        mapping = _build_model_mapping(llm, overrides=overrides)

        assert mapping["analytical"] == "my-custom-model"
        assert mapping["simple"] == "my-simple-model"

    def test_fallback_simple_from_analytical(self) -> None:
        """When 'simple' is missing, it falls back to 'analytical'."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep = _make_endpoint("ep-analytical", "databricks-llama-70b")
        roles = {"analytical": _make_role("analytical", ["ep-analytical"])}
        endpoints = {"ep-analytical": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        mapping = _build_model_mapping(llm)

        assert mapping["simple"] == "databricks-llama-70b"
        assert mapping["analytical"] == "databricks-llama-70b"

    def test_fallback_complex_from_analytical(self) -> None:
        """When 'complex' is missing, it falls back to 'analytical'."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep = _make_endpoint("ep-analytical", "databricks-llama-70b")
        roles = {"analytical": _make_role("analytical", ["ep-analytical"])}
        endpoints = {"ep-analytical": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        mapping = _build_model_mapping(llm)

        assert mapping["complex"] == "databricks-llama-70b"

    def test_fallback_when_analytical_missing_uses_any(self) -> None:
        """When 'analytical' is missing, all three core tiers fall back to any available."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep = _make_endpoint("ep-bulk", "databricks-gemini")
        roles = {"bulk_analysis": _make_role("bulk_analysis", ["ep-bulk"])}
        endpoints = {"ep-bulk": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        mapping = _build_model_mapping(llm)

        assert mapping["simple"] == "databricks-gemini"
        assert mapping["analytical"] == "databricks-gemini"
        assert mapping["complex"] == "databricks-gemini"

    def test_empty_endpoints_raises(self) -> None:
        """A role with empty endpoints list is skipped; if no tiers map, ValueError is raised."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        roles = {"simple": _make_role("simple", [])}
        config = _make_mock_config(roles, {})
        # get_role succeeds but endpoints is empty → IndexError caught
        llm = _make_mock_llm_client(config)

        # With no valid endpoints at all, _build_model_mapping now raises
        with pytest.raises(ValueError, match="LLM_ADAPTER_NO_TIERS"):
            _build_model_mapping(llm)

    def test_missing_role_is_skipped(self) -> None:
        """When get_role raises ValueError for a tier, that tier is skipped."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        # Only provide analytical, the rest will raise ValueError
        ep = _make_endpoint("ep-a", "model-a")
        roles = {"analytical": _make_role("analytical", ["ep-a"])}
        endpoints = {"ep-a": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        mapping = _build_model_mapping(llm)

        assert mapping["analytical"] == "model-a"
        # simple and complex filled via fallback
        assert mapping["simple"] == "model-a"
        assert mapping["complex"] == "model-a"

    def test_overrides_take_precedence_over_fallback(self) -> None:
        """Overrides win even when the tier would have a fallback value."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        ep = _make_endpoint("ep-a", "model-a")
        roles = {"analytical": _make_role("analytical", ["ep-a"])}
        endpoints = {"ep-a": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        overrides = {"simple": "override-simple"}
        mapping = _build_model_mapping(llm, overrides=overrides)

        # Override wins for simple
        assert mapping["simple"] == "override-simple"
        # Analytical comes from config
        assert mapping["analytical"] == "model-a"


# ---------------------------------------------------------------------------
# Tests — create_framework_llm_client
# ---------------------------------------------------------------------------


class TestCreateFrameworkLLMClient:
    """Tests for create_framework_llm_client."""

    @patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
    def test_creates_client_with_correct_args(self, mock_fw_cls: MagicMock) -> None:
        """The factory extracts the OpenAI client and builds model mapping."""
        from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

        fake_openai = MagicMock(name="AsyncOpenAI")
        ep = _make_endpoint("ep-a", "model-a")
        roles = {"analytical": _make_role("analytical", ["ep-a"])}
        endpoints = {"ep-a": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config, openai_client=fake_openai)

        create_framework_llm_client(llm)

        llm._ensure_fresh_client.assert_called_once()
        mock_fw_cls.assert_called_once()
        call_kwargs = mock_fw_cls.call_args
        assert call_kwargs.kwargs["openai_client"] is fake_openai
        assert isinstance(call_kwargs.kwargs["model_mapping"], dict)

    @patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
    def test_passes_embedding_model(self, mock_fw_cls: MagicMock) -> None:
        """Embedding model is forwarded to FrameworkLLMClient."""
        from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

        ep = _make_endpoint("ep-a", "model-a")
        roles = {"analytical": _make_role("analytical", ["ep-a"])}
        endpoints = {"ep-a": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        create_framework_llm_client(llm, embedding_model="bge-large-en")

        call_kwargs = mock_fw_cls.call_args
        assert call_kwargs.kwargs["embedding_model"] == "bge-large-en"

    @patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
    def test_passes_model_overrides(self, mock_fw_cls: MagicMock) -> None:
        """Model overrides are applied to the mapping."""
        from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

        ep = _make_endpoint("ep-a", "model-a")
        roles = {"analytical": _make_role("analytical", ["ep-a"])}
        endpoints = {"ep-a": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        create_framework_llm_client(llm, model_overrides={"complex": "my-model"})

        call_kwargs = mock_fw_cls.call_args
        mapping = call_kwargs.kwargs["model_mapping"]
        assert mapping["complex"] == "my-model"

    @patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
    def test_returns_framework_client(self, mock_fw_cls: MagicMock) -> None:
        """The return value is what FrameworkLLMClient() returns."""
        from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

        sentinel = MagicMock(name="framework-client")
        mock_fw_cls.return_value = sentinel

        ep = _make_endpoint("ep-a", "model-a")
        roles = {"analytical": _make_role("analytical", ["ep-a"])}
        endpoints = {"ep-a": ep}
        config = _make_mock_config(roles, endpoints)
        llm = _make_mock_llm_client(config)

        result = create_framework_llm_client(llm)

        assert result is sentinel
