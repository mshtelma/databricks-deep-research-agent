"""
Output Type Registry
====================

Central registry for managing output types.

This module provides the OutputTypeRegistry class that manages
registration and lookup of output type providers.
"""

from typing import Any

from deep_research.output.base import SynthesizerConfig
from deep_research.output.protocol import DefaultOutputTypeProvider, OutputTypeProvider


class OutputTypeRegistry:
    """Central registry for output type providers.

    Manages registration, lookup, and enumeration of output types.
    Plugins register their output types here, and the synthesizer
    uses the registry to get configuration for the requested type.

    Thread-safe: Uses simple dict operations which are atomic in Python.

    Example:
        >>> registry = OutputTypeRegistry()
        >>> registry.register("meeting_prep", MeetingPrepProvider())
        >>> config = registry.get_synthesizer_config("meeting_prep")
        >>> schema = registry.get_schema("meeting_prep")
    """

    def __init__(self) -> None:
        """Initialize the registry with default output type."""
        self._providers: dict[str, OutputTypeProvider] = {}
        # Register default output type
        self.register("synthesis_report", DefaultOutputTypeProvider())

    def register(
        self,
        name: str,
        provider: OutputTypeProvider,
        replace: bool = False,
    ) -> None:
        """Register an output type provider.

        Args:
            name: Output type name
            provider: Provider instance
            replace: If True, replace existing provider

        Raises:
            ValueError: If name already registered and replace=False
        """
        if name in self._providers and not replace:
            raise ValueError(
                f"Output type '{name}' already registered. "
                f"Use replace=True to override."
            )
        self._providers[name] = provider

    def unregister(self, name: str) -> bool:
        """Unregister an output type.

        Args:
            name: Output type name to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            return True
        return False

    def get(self, name: str) -> OutputTypeProvider | None:
        """Get an output type provider by name.

        Args:
            name: Output type name

        Returns:
            Provider instance, or None if not found
        """
        return self._providers.get(name)

    def get_or_default(self, name: str | None) -> OutputTypeProvider:
        """Get an output type provider, falling back to default.

        Args:
            name: Output type name, or None for default

        Returns:
            Provider instance
        """
        if name is None:
            name = "synthesis_report"
        provider = self._providers.get(name)
        if provider is None:
            return self._providers["synthesis_report"]
        return provider

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """Get output schema for an output type.

        Args:
            name: Output type name

        Returns:
            JSON Schema dict, or None if not found
        """
        provider = self.get(name)
        if provider is None:
            return None
        return provider.get_output_schema()

    def get_synthesizer_config(
        self,
        name: str,
        base_config: SynthesizerConfig | None = None,
    ) -> SynthesizerConfig | None:
        """Get synthesizer configuration for an output type.

        Args:
            name: Output type name
            base_config: Optional base config to merge with

        Returns:
            SynthesizerConfig, or None if not found
        """
        provider = self.get(name)
        if provider is None:
            return None

        config = provider.get_synthesizer_config()
        if base_config:
            config = base_config.merge_with(config)
        return config

    def get_synthesizer_prompt(self, name: str) -> str | None:
        """Get custom synthesizer prompt for an output type.

        Args:
            name: Output type name

        Returns:
            Custom prompt string, or None
        """
        provider = self.get(name)
        if provider is None:
            return None
        return provider.get_synthesizer_prompt()

    def list_output_types(self) -> list[str]:
        """Get list of registered output type names.

        Returns:
            List of output type names
        """
        return list(self._providers.keys())

    def list_output_types_with_description(self) -> list[dict[str, str]]:
        """Get list of output types with descriptions.

        Returns:
            List of dicts with 'name' and 'description' keys
        """
        return [
            {"name": name, "description": provider.description}
            for name, provider in self._providers.items()
        ]

    def has(self, name: str) -> bool:
        """Check if output type is registered.

        Args:
            name: Output type name

        Returns:
            True if registered
        """
        return name in self._providers

    def __len__(self) -> int:
        """Get number of registered output types."""
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        """Check if output type is registered."""
        return name in self._providers


# Global registry instance
_global_registry: OutputTypeRegistry | None = None


def get_output_registry() -> OutputTypeRegistry:
    """Get the global output type registry.

    Returns:
        Global OutputTypeRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = OutputTypeRegistry()
    return _global_registry


def reset_output_registry() -> None:
    """Reset the global output type registry.

    Primarily for testing purposes.
    """
    global _global_registry
    _global_registry = None
