"""Unit tests for OutputTypeProvider protocol."""

from typing import Any

import pytest

from deep_research.output.base import SynthesizerConfig
from deep_research.output.protocol import DefaultOutputTypeProvider, OutputTypeProvider


class TestOutputTypeProviderProtocol:
    """Tests for OutputTypeProvider protocol."""

    def test_default_provider_implements_protocol(self) -> None:
        """DefaultOutputTypeProvider should implement protocol."""
        provider = DefaultOutputTypeProvider()
        assert isinstance(provider, OutputTypeProvider)

    def test_custom_provider_implements_protocol(self) -> None:
        """Custom implementation should be recognized."""

        class CustomProvider:
            @property
            def output_type_name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "Custom output type"

            def get_output_schema(self) -> dict[str, Any]:
                return {"type": "object"}

            def get_synthesizer_config(self) -> SynthesizerConfig:
                return SynthesizerConfig(output_type="custom")

            def get_synthesizer_prompt(self) -> str | None:
                return "Custom prompt"

        provider = CustomProvider()
        assert isinstance(provider, OutputTypeProvider)

    def test_incomplete_provider_not_recognized(self) -> None:
        """Incomplete implementation should not match protocol."""

        class IncompleteProvider:
            @property
            def output_type_name(self) -> str:
                return "incomplete"

            # Missing other required methods

        provider = IncompleteProvider()
        assert not isinstance(provider, OutputTypeProvider)


class TestDefaultOutputTypeProvider:
    """Tests for DefaultOutputTypeProvider."""

    def test_output_type_name(self) -> None:
        """Should return synthesis_report."""
        provider = DefaultOutputTypeProvider()
        assert provider.output_type_name == "synthesis_report"

    def test_description(self) -> None:
        """Should have meaningful description."""
        provider = DefaultOutputTypeProvider()
        assert "research" in provider.description.lower()
        assert "synthesis" in provider.description.lower()

    def test_get_output_schema(self) -> None:
        """Should return SynthesisReport schema."""
        provider = DefaultOutputTypeProvider()
        schema = provider.get_output_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "content" in schema["properties"]

    def test_get_synthesizer_config(self) -> None:
        """Should return default SynthesizerConfig."""
        provider = DefaultOutputTypeProvider()
        config = provider.get_synthesizer_config()
        assert isinstance(config, SynthesizerConfig)
        assert config.output_type == "synthesis_report"

    def test_get_synthesizer_prompt(self) -> None:
        """Should return None for default prompt."""
        provider = DefaultOutputTypeProvider()
        prompt = provider.get_synthesizer_prompt()
        assert prompt is None


class TestCustomOutputTypeProvider:
    """Tests for custom output type provider implementations."""

    def test_meeting_prep_provider(self) -> None:
        """Test a complete custom provider implementation."""

        class MeetingPrepProvider:
            @property
            def output_type_name(self) -> str:
                return "meeting_prep"

            @property
            def description(self) -> str:
                return "Meeting preparation document with agenda and talking points"

            def get_output_schema(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "company_overview": {
                            "type": "string",
                            "description": "Overview of the company",
                        },
                        "key_contacts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "role": {"type": "string"},
                                    "notes": {"type": "string"},
                                },
                            },
                        },
                        "talking_points": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "agenda": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "topic": {"type": "string"},
                                    "duration": {"type": "integer"},
                                    "notes": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["company_overview", "talking_points"],
                }

            def get_synthesizer_config(self) -> SynthesizerConfig:
                return SynthesizerConfig(
                    output_type="meeting_prep",
                    model_tier="complex",
                    temperature=0.3,
                    max_tokens=4000,
                    include_citations=False,
                    extra_config={"format": "structured"},
                )

            def get_synthesizer_prompt(self) -> str | None:
                return """Generate a meeting preparation document based on the research.

Include:
1. Company overview with recent news and financials
2. Key contacts with their roles and relevant background
3. 5-7 talking points for the meeting
4. Suggested agenda with time allocations

Query: {query}
Findings: {findings}
"""

        provider = MeetingPrepProvider()
        assert isinstance(provider, OutputTypeProvider)
        assert provider.output_type_name == "meeting_prep"

        schema = provider.get_output_schema()
        assert "company_overview" in schema["properties"]
        assert "talking_points" in schema["properties"]

        config = provider.get_synthesizer_config()
        assert config.temperature == 0.3
        assert config.include_citations is False

        prompt = provider.get_synthesizer_prompt()
        assert prompt is not None
        assert "{query}" in prompt
        assert "{findings}" in prompt
