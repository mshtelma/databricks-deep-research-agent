"""
Output Type Provider Protocol
=============================

Protocol for plugins that provide custom output types.

This module defines the OutputTypeProvider protocol that plugins
implement to contribute custom output types to the research system.
"""

from typing import Any, Protocol, runtime_checkable

from deep_research.output.base import SynthesizerConfig


@runtime_checkable
class OutputTypeProvider(Protocol):
    """Protocol for plugins that provide custom output types.

    Implement this protocol to contribute custom output schemas
    and synthesizer configurations from a plugin.

    The provider defines:
    - Output type name and description
    - JSON Schema for structured output
    - Synthesizer configuration
    - Custom prompt template (optional)

    Example:
        >>> class MeetingPrepProvider:
        ...     @property
        ...     def output_type_name(self) -> str:
        ...         return "meeting_prep"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Meeting preparation document with agenda and talking points"
        ...
        ...     def get_output_schema(self) -> dict[str, Any]:
        ...         return {
        ...             "type": "object",
        ...             "properties": {
        ...                 "company_overview": {"type": "string"},
        ...                 "key_contacts": {"type": "array", "items": {...}},
        ...                 "talking_points": {"type": "array", "items": {...}},
        ...                 "agenda": {"type": "array", "items": {...}},
        ...             },
        ...             "required": ["company_overview", "talking_points"],
        ...         }
        ...
        ...     def get_synthesizer_config(self) -> SynthesizerConfig:
        ...         return SynthesizerConfig(
        ...             output_type="meeting_prep",
        ...             model_tier="complex",
        ...             temperature=0.3,
        ...         )
        ...
        ...     def get_synthesizer_prompt(self) -> str | None:
        ...         return "Generate a meeting preparation document..."
    """

    @property
    def output_type_name(self) -> str:
        """Unique name for this output type.

        This name is used to reference the output type in configuration
        and API requests.

        Returns:
            Unique output type identifier (e.g., "meeting_prep")
        """
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the output type.

        Describes what kind of output this type produces.

        Returns:
            Description string
        """
        ...

    def get_output_schema(self) -> dict[str, Any]:
        """Get JSON Schema for the output type.

        The schema defines the structure of the synthesizer output
        and is used for:
        - Structured output generation with the LLM
        - Validation of generated output
        - Documentation of the output format

        Returns:
            JSON Schema dictionary
        """
        ...

    def get_synthesizer_config(self) -> SynthesizerConfig:
        """Get configuration for the synthesizer agent.

        Returns configuration that controls how the synthesizer
        generates this output type.

        Returns:
            SynthesizerConfig instance
        """
        ...

    def get_synthesizer_prompt(self) -> str | None:
        """Get custom prompt template for the synthesizer.

        Optional custom prompt that replaces the default synthesizer
        prompt for this output type.

        The prompt can include placeholders:
        - {query}: Original research query
        - {findings}: Collected research findings
        - {sources}: List of sources
        - {schema}: Output schema (for structured generation)

        Returns:
            Custom prompt template, or None to use default
        """
        ...


class DefaultOutputTypeProvider:
    """Default implementation of OutputTypeProvider.

    Provides the standard SynthesisReport output type.
    """

    @property
    def output_type_name(self) -> str:
        """Return default output type name."""
        return "synthesis_report"

    @property
    def description(self) -> str:
        """Return description of default output type."""
        return "Standard research synthesis report with findings and sources"

    def get_output_schema(self) -> dict[str, Any]:
        """Return default SynthesisReport schema."""
        from deep_research.output.base import SynthesisReport

        return SynthesisReport.get_schema()

    def get_synthesizer_config(self) -> SynthesizerConfig:
        """Return default synthesizer configuration."""
        return SynthesizerConfig()

    def get_synthesizer_prompt(self) -> str | None:
        """Return None to use default prompt."""
        return None
