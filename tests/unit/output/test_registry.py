"""Unit tests for OutputTypeRegistry."""

from typing import Any

import pytest

from deep_research.output.base import SynthesizerConfig
from deep_research.output.protocol import DefaultOutputTypeProvider, OutputTypeProvider
from deep_research.output.registry import (
    OutputTypeRegistry,
    get_output_registry,
    reset_output_registry,
)


class MockOutputProvider:
    """Mock output type provider for testing."""

    def __init__(
        self,
        name: str = "mock_output",
        description: str = "Mock output type",
        schema: dict[str, Any] | None = None,
        config: SynthesizerConfig | None = None,
        prompt: str | None = None,
    ):
        self._name = name
        self._description = description
        self._schema = schema or {"type": "object"}
        self._config = config or SynthesizerConfig(output_type=name)
        self._prompt = prompt

    @property
    def output_type_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def get_output_schema(self) -> dict[str, Any]:
        return self._schema

    def get_synthesizer_config(self) -> SynthesizerConfig:
        return self._config

    def get_synthesizer_prompt(self) -> str | None:
        return self._prompt


class TestOutputTypeRegistryCreation:
    """Tests for OutputTypeRegistry initialization."""

    def test_create_empty_has_default(self) -> None:
        """New registry should have default output type."""
        registry = OutputTypeRegistry()
        assert "synthesis_report" in registry
        assert len(registry) == 1

    def test_default_provider_is_correct(self) -> None:
        """Default provider should be DefaultOutputTypeProvider."""
        registry = OutputTypeRegistry()
        provider = registry.get("synthesis_report")
        assert isinstance(provider, DefaultOutputTypeProvider)


class TestOutputTypeRegistryRegistration:
    """Tests for registration methods."""

    def test_register_new_provider(self) -> None:
        """Should register new provider."""
        registry = OutputTypeRegistry()
        provider = MockOutputProvider("custom")
        registry.register("custom", provider)
        assert "custom" in registry
        assert registry.get("custom") is provider

    def test_register_duplicate_raises(self) -> None:
        """Should raise on duplicate registration."""
        registry = OutputTypeRegistry()
        provider = MockOutputProvider("test")
        registry.register("test", provider)
        with pytest.raises(ValueError) as exc_info:
            registry.register("test", MockOutputProvider("test"))
        assert "already registered" in str(exc_info.value)

    def test_register_duplicate_with_replace(self) -> None:
        """Should replace when replace=True."""
        registry = OutputTypeRegistry()
        provider1 = MockOutputProvider("test", description="First")
        provider2 = MockOutputProvider("test", description="Second")
        registry.register("test", provider1)
        registry.register("test", provider2, replace=True)
        assert registry.get("test") is provider2

    def test_unregister_existing(self) -> None:
        """Should unregister existing provider."""
        registry = OutputTypeRegistry()
        registry.register("test", MockOutputProvider("test"))
        assert registry.unregister("test") is True
        assert "test" not in registry

    def test_unregister_missing(self) -> None:
        """Should return False for missing provider."""
        registry = OutputTypeRegistry()
        assert registry.unregister("nonexistent") is False


class TestOutputTypeRegistryLookup:
    """Tests for lookup methods."""

    def test_get_existing(self) -> None:
        """Should return provider for existing type."""
        registry = OutputTypeRegistry()
        provider = MockOutputProvider("test")
        registry.register("test", provider)
        assert registry.get("test") is provider

    def test_get_missing(self) -> None:
        """Should return None for missing type."""
        registry = OutputTypeRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_default_existing(self) -> None:
        """Should return provider when found."""
        registry = OutputTypeRegistry()
        provider = MockOutputProvider("custom")
        registry.register("custom", provider)
        assert registry.get_or_default("custom") is provider

    def test_get_or_default_missing(self) -> None:
        """Should return default when not found."""
        registry = OutputTypeRegistry()
        result = registry.get_or_default("nonexistent")
        assert result.output_type_name == "synthesis_report"

    def test_get_or_default_none(self) -> None:
        """Should return default when None passed."""
        registry = OutputTypeRegistry()
        result = registry.get_or_default(None)
        assert result.output_type_name == "synthesis_report"

    def test_has_existing(self) -> None:
        """Should return True for existing type."""
        registry = OutputTypeRegistry()
        assert registry.has("synthesis_report") is True

    def test_has_missing(self) -> None:
        """Should return False for missing type."""
        registry = OutputTypeRegistry()
        assert registry.has("nonexistent") is False

    def test_contains_operator(self) -> None:
        """Should support 'in' operator."""
        registry = OutputTypeRegistry()
        assert "synthesis_report" in registry
        assert "nonexistent" not in registry


class TestOutputTypeRegistrySchemaAndConfig:
    """Tests for schema and config retrieval."""

    def test_get_schema_existing(self) -> None:
        """Should return schema for existing type."""
        registry = OutputTypeRegistry()
        custom_schema = {"type": "object", "properties": {"custom": {"type": "string"}}}
        registry.register("custom", MockOutputProvider("custom", schema=custom_schema))
        assert registry.get_schema("custom") == custom_schema

    def test_get_schema_missing(self) -> None:
        """Should return None for missing type."""
        registry = OutputTypeRegistry()
        assert registry.get_schema("nonexistent") is None

    def test_get_synthesizer_config_existing(self) -> None:
        """Should return config for existing type."""
        registry = OutputTypeRegistry()
        custom_config = SynthesizerConfig(
            output_type="custom",
            model_tier="analytical",
            temperature=0.3,
        )
        registry.register("custom", MockOutputProvider("custom", config=custom_config))
        config = registry.get_synthesizer_config("custom")
        assert config is not None
        assert config.output_type == "custom"
        assert config.temperature == 0.3

    def test_get_synthesizer_config_missing(self) -> None:
        """Should return None for missing type."""
        registry = OutputTypeRegistry()
        assert registry.get_synthesizer_config("nonexistent") is None

    def test_get_synthesizer_config_with_base(self) -> None:
        """Should merge with base config."""
        registry = OutputTypeRegistry()
        provider_config = SynthesizerConfig(
            output_type="custom",
            temperature=0.3,
        )
        registry.register("custom", MockOutputProvider("custom", config=provider_config))

        base_config = SynthesizerConfig(
            model_tier="simple",
            max_tokens=2000,
        )
        merged = registry.get_synthesizer_config("custom", base_config)
        assert merged is not None
        assert merged.output_type == "custom"
        assert merged.temperature == 0.3

    def test_get_synthesizer_prompt_existing(self) -> None:
        """Should return prompt for existing type."""
        registry = OutputTypeRegistry()
        registry.register(
            "custom",
            MockOutputProvider("custom", prompt="Custom prompt template"),
        )
        assert registry.get_synthesizer_prompt("custom") == "Custom prompt template"

    def test_get_synthesizer_prompt_none(self) -> None:
        """Should return None when provider has no prompt."""
        registry = OutputTypeRegistry()
        registry.register("custom", MockOutputProvider("custom", prompt=None))
        assert registry.get_synthesizer_prompt("custom") is None

    def test_get_synthesizer_prompt_missing(self) -> None:
        """Should return None for missing type."""
        registry = OutputTypeRegistry()
        assert registry.get_synthesizer_prompt("nonexistent") is None


class TestOutputTypeRegistryEnumeration:
    """Tests for listing methods."""

    def test_list_output_types(self) -> None:
        """Should list all registered types."""
        registry = OutputTypeRegistry()
        registry.register("type1", MockOutputProvider("type1"))
        registry.register("type2", MockOutputProvider("type2"))
        types = registry.list_output_types()
        assert "synthesis_report" in types
        assert "type1" in types
        assert "type2" in types
        assert len(types) == 3

    def test_list_output_types_with_description(self) -> None:
        """Should list types with descriptions."""
        registry = OutputTypeRegistry()
        registry.register(
            "custom",
            MockOutputProvider("custom", description="Custom description"),
        )
        items = registry.list_output_types_with_description()
        custom_item = next(i for i in items if i["name"] == "custom")
        assert custom_item["description"] == "Custom description"

    def test_len(self) -> None:
        """Should return correct length."""
        registry = OutputTypeRegistry()
        assert len(registry) == 1  # Default
        registry.register("type1", MockOutputProvider("type1"))
        assert len(registry) == 2
        registry.register("type2", MockOutputProvider("type2"))
        assert len(registry) == 3


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_output_registry_singleton(self) -> None:
        """Should return same instance."""
        reset_output_registry()
        registry1 = get_output_registry()
        registry2 = get_output_registry()
        assert registry1 is registry2

    def test_reset_output_registry(self) -> None:
        """Should reset to new instance."""
        reset_output_registry()
        registry1 = get_output_registry()
        registry1.register("test", MockOutputProvider("test"))
        reset_output_registry()
        registry2 = get_output_registry()
        assert "test" not in registry2

    def test_global_registry_has_default(self) -> None:
        """Global registry should have default type."""
        reset_output_registry()
        registry = get_output_registry()
        assert "synthesis_report" in registry
