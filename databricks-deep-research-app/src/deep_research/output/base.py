"""
Output Base Types
=================

Base types for research output configuration.

This module provides:
- SynthesisReport: Default output type for research
- SynthesizerConfig: Configuration for synthesizer agent
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthesisReport:
    """Default output type for research synthesis.

    The standard output format containing the research report
    and associated metadata.

    Attributes:
        title: Report title
        content: Main report content (markdown)
        summary: Executive summary
        key_findings: List of key findings
        sources: List of sources used
        metadata: Additional metadata (timestamps, query info, etc.)

    Example:
        >>> report = SynthesisReport(
        ...     title="AI Market Analysis",
        ...     content="# Market Overview\\n\\nThe AI market...",
        ...     summary="AI market expected to grow 25% annually...",
        ...     key_findings=["Finding 1", "Finding 2"],
        ...     sources=[{"url": "...", "title": "..."}],
        ... )
    """

    title: str = ""
    content: str = ""
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """Get JSON Schema for SynthesisReport.

        Returns:
            JSON Schema dict for structured output
        """
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Report title",
                },
                "content": {
                    "type": "string",
                    "description": "Main report content in markdown format",
                },
                "summary": {
                    "type": "string",
                    "description": "Executive summary of key points",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key findings",
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                            "type": {"type": "string"},
                        },
                    },
                    "description": "List of sources used",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata",
                },
            },
            "required": ["title", "content"],
        }


@dataclass
class SynthesizerConfig:
    """Configuration for the synthesizer agent.

    Controls how the synthesizer generates output based on
    the output type requirements.

    Attributes:
        output_type: Name of the output type (e.g., "synthesis_report")
        model_tier: LLM tier to use for synthesis
        temperature: Temperature for generation
        max_tokens: Maximum tokens in output
        include_sources: Whether to include source list
        include_citations: Whether to include inline citations
        custom_schema: Optional custom JSON schema for output
        custom_prompt: Optional custom prompt template
        extra_config: Additional output-type-specific config

    Example:
        >>> config = SynthesizerConfig(
        ...     output_type="meeting_prep",
        ...     model_tier="complex",
        ...     temperature=0.3,
        ...     custom_prompt="Generate a meeting preparation document...",
        ... )
    """

    output_type: str = "synthesis_report"
    model_tier: str = "complex"
    temperature: float = 0.7
    max_tokens: int = 8000
    include_sources: bool = True
    include_citations: bool = True
    custom_schema: dict[str, Any] | None = None
    custom_prompt: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    def get_effective_schema(self) -> dict[str, Any]:
        """Get the effective output schema.

        Returns custom_schema if set, otherwise the default
        SynthesisReport schema.

        Returns:
            JSON Schema for output validation
        """
        if self.custom_schema:
            return self.custom_schema
        return SynthesisReport.get_schema()

    def merge_with(self, other: "SynthesizerConfig") -> "SynthesizerConfig":
        """Merge with another config, with other taking precedence for non-defaults.

        Creates a new config where:
        - other's non-default values override this config's values
        - For fields without clear defaults (output_type, model_tier, custom_*),
          other's value takes precedence unless it's None/empty

        Args:
            other: Config to merge with (takes precedence for non-defaults)

        Returns:
            New merged SynthesizerConfig
        """
        return SynthesizerConfig(
            output_type=other.output_type or self.output_type,
            model_tier=other.model_tier or self.model_tier,
            # For numeric fields, use other's value only if it differs from default
            temperature=self.temperature if other.temperature == 0.7 else other.temperature,
            max_tokens=self.max_tokens if other.max_tokens == 8000 else other.max_tokens,
            include_sources=other.include_sources,
            include_citations=other.include_citations,
            custom_schema=other.custom_schema or self.custom_schema,
            custom_prompt=other.custom_prompt or self.custom_prompt,
            extra_config={**self.extra_config, **other.extra_config},
        )
