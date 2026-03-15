"""Tests for parse_model_config() and FrameworkLLMClient.derive()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.llm.client import (
    FrameworkLLMClient,
    ModelTierConfig,
    parse_model_config,
)


# ===================================================================
# parse_model_config
# ===================================================================


class TestParseModelConfig:
    """Tests for parse_model_config()."""

    def test_simple_string(self) -> None:
        """String values pass through unchanged."""
        result = parse_model_config({"simple": "my-model"})
        assert result == {"simple": "my-model"}

    def test_rich_dict(self) -> None:
        """Dict with endpoints -> ModelTierConfig."""
        result = parse_model_config({
            "analytical": {
                "endpoints": ["model-a", "model-b"],
                "fallback_on_429": True,
                "rotation_strategy": "PRIORITY",
                "tokens_per_minute": 100000,
            }
        })
        cfg = result["analytical"]
        assert isinstance(cfg, ModelTierConfig)
        assert cfg.endpoints == ["model-a", "model-b"]
        assert cfg.fallback_on_429 is True
        assert cfg.rotation_strategy == "PRIORITY"
        assert cfg.tokens_per_minute == 100000

    def test_mixed(self) -> None:
        """Some strings, some dicts in same mapping."""
        result = parse_model_config({
            "simple": "fast-model",
            "complex": {
                "endpoints": ["big-model"],
            },
        })
        assert result["simple"] == "fast-model"
        assert isinstance(result["complex"], ModelTierConfig)
        assert result["complex"].endpoints == ["big-model"]

    def test_custom_tier_names(self) -> None:
        """Custom names like 'fast', 'bulk_analysis' work."""
        result = parse_model_config({
            "bulk_analysis": "high-throughput-model",
            "fast": {"endpoints": ["quick-model"]},
        })
        assert "bulk_analysis" in result
        assert "fast" in result

    def test_rotation_strategy_uppercased(self) -> None:
        """'priority' -> 'PRIORITY'."""
        result = parse_model_config({
            "t": {"endpoints": ["m"], "rotation_strategy": "round_robin"}
        })
        cfg = result["t"]
        assert isinstance(cfg, ModelTierConfig)
        assert cfg.rotation_strategy == "ROUND_ROBIN"

    def test_fallback_defaults_true(self) -> None:
        """fallback_on_429 defaults to True when omitted."""
        result = parse_model_config({"t": {"endpoints": ["m"]}})
        cfg = result["t"]
        assert isinstance(cfg, ModelTierConfig)
        assert cfg.fallback_on_429 is True

    def test_tpm_defaults_zero(self) -> None:
        """tokens_per_minute defaults to 0 when omitted."""
        result = parse_model_config({"t": {"endpoints": ["m"]}})
        cfg = result["t"]
        assert isinstance(cfg, ModelTierConfig)
        assert cfg.tokens_per_minute == 0

    def test_invalid_type_raises(self) -> None:
        """Non-str/dict value -> ValueError."""
        with pytest.raises(ValueError, match="expected str or dict, got int"):
            parse_model_config({"t": 42})

    def test_invalid_type_list_raises(self) -> None:
        """List value -> ValueError."""
        with pytest.raises(ValueError, match="expected str or dict, got list"):
            parse_model_config({"t": ["a", "b"]})

    def test_missing_endpoints_key_raises(self) -> None:
        """Dict without 'endpoints' -> KeyError."""
        with pytest.raises(KeyError, match="endpoints"):
            parse_model_config({"t": {"fallback_on_429": True}})

    def test_invalid_rotation_strategy_raises(self) -> None:
        """Unknown strategy -> ValueError with valid options."""
        with pytest.raises(ValueError, match="Invalid rotation_strategy 'RANDOM'"):
            parse_model_config({
                "t": {"endpoints": ["m"], "rotation_strategy": "random"}
            })

    def test_empty_dict(self) -> None:
        """Empty input produces empty output."""
        assert parse_model_config({}) == {}


# ===================================================================
# FrameworkLLMClient.derive
# ===================================================================


def _make_client(
    model_mapping: dict[str, str | ModelTierConfig] | None = None,
    client_provider: object | None = None,
    embedding_model: str | None = None,
) -> FrameworkLLMClient:
    """Create a FrameworkLLMClient with a mock AsyncOpenAI."""
    mock_openai = MagicMock()
    return FrameworkLLMClient(
        openai_client=mock_openai,
        model_mapping=model_mapping or {"simple": "model-a", "analytical": "model-b"},
        embedding_model=embedding_model,
        client_provider=client_provider,  # type: ignore[arg-type]
    )


class TestDerive:
    """Tests for FrameworkLLMClient.derive()."""

    def test_returns_new_instance(self) -> None:
        """Derived client is a different object."""
        client = _make_client()
        derived = client.derive({"simple": "new-model"})
        assert derived is not client

    def test_overrides_existing_tier(self) -> None:
        """New mapping replaces same-named tier."""
        client = _make_client({"simple": "old-model", "analytical": "model-b"})
        derived = client.derive({"simple": "new-model"})
        assert derived._models["simple"] == "new-model"

    def test_preserves_unmentioned_tiers(self) -> None:
        """Tiers not in override mapping are kept from original."""
        client = _make_client({"simple": "model-a", "analytical": "model-b"})
        derived = client.derive({"simple": "new-model"})
        assert derived._models["analytical"] == "model-b"

    def test_adds_custom_tier(self) -> None:
        """New tier name not in original gets added."""
        client = _make_client({"simple": "model-a"})
        derived = client.derive({"bulk": "bulk-model"})
        assert "bulk" in derived._models
        assert derived._models["bulk"] == "bulk-model"
        assert derived._models["simple"] == "model-a"

    def test_shares_client_provider(self) -> None:
        """Derived client inherits client_provider for token refresh."""
        provider = MagicMock()
        client = _make_client(client_provider=provider)
        derived = client.derive({"simple": "new"})
        assert derived._client_provider is provider

    def test_fresh_health_state(self) -> None:
        """Derived client starts with empty endpoint health."""
        client = _make_client()
        # Simulate some health state on the parent
        client._endpoint_health["model-a"] = MagicMock()
        derived = client.derive({"simple": "new"})
        assert len(derived._endpoint_health) == 0

    def test_shares_embedding_model(self) -> None:
        """Derived client preserves embedding_model config."""
        client = _make_client(embedding_model="embed-v1")
        derived = client.derive({"simple": "new"})
        assert derived._embedding_model == "embed-v1"

    def test_shares_underlying_openai_client(self) -> None:
        """Derived client shares the same AsyncOpenAI instance."""
        client = _make_client()
        derived = client.derive({"simple": "new"})
        assert derived._client is client._client

    def test_derive_with_model_tier_config(self) -> None:
        """Derived client can accept ModelTierConfig values."""
        client = _make_client()
        cfg = ModelTierConfig(endpoints=["ep1", "ep2"], tokens_per_minute=50000)
        derived = client.derive({"complex": cfg})
        assert derived._models["complex"] is cfg
