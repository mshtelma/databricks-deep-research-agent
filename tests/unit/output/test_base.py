"""Unit tests for output base types."""

import pytest

from deep_research.output.base import SynthesisReport, SynthesizerConfig


class TestSynthesisReport:
    """Tests for SynthesisReport dataclass."""

    def test_create_minimal(self) -> None:
        """Should create with defaults."""
        report = SynthesisReport()
        assert report.title == ""
        assert report.content == ""
        assert report.summary == ""
        assert report.key_findings == []
        assert report.sources == []
        assert report.metadata == {}

    def test_create_with_values(self) -> None:
        """Should create with provided values."""
        report = SynthesisReport(
            title="Test Report",
            content="# Report\n\nContent here.",
            summary="Summary of findings.",
            key_findings=["Finding 1", "Finding 2"],
            sources=[{"url": "https://example.com", "title": "Example"}],
            metadata={"query": "test query"},
        )
        assert report.title == "Test Report"
        assert report.content == "# Report\n\nContent here."
        assert report.summary == "Summary of findings."
        assert len(report.key_findings) == 2
        assert len(report.sources) == 1
        assert report.metadata["query"] == "test query"

    def test_get_schema(self) -> None:
        """Should return valid JSON schema."""
        schema = SynthesisReport.get_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "content" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "key_findings" in schema["properties"]
        assert "sources" in schema["properties"]
        assert "metadata" in schema["properties"]

    def test_schema_required_fields(self) -> None:
        """Schema should have required fields."""
        schema = SynthesisReport.get_schema()
        assert "required" in schema
        assert "title" in schema["required"]
        assert "content" in schema["required"]

    def test_schema_key_findings_is_array(self) -> None:
        """Schema key_findings should be array of strings."""
        schema = SynthesisReport.get_schema()
        key_findings = schema["properties"]["key_findings"]
        assert key_findings["type"] == "array"
        assert key_findings["items"]["type"] == "string"

    def test_schema_sources_is_array_of_objects(self) -> None:
        """Schema sources should be array of objects."""
        schema = SynthesisReport.get_schema()
        sources = schema["properties"]["sources"]
        assert sources["type"] == "array"
        assert sources["items"]["type"] == "object"


class TestSynthesizerConfig:
    """Tests for SynthesizerConfig dataclass."""

    def test_create_with_defaults(self) -> None:
        """Should create with sensible defaults."""
        config = SynthesizerConfig()
        assert config.output_type == "synthesis_report"
        assert config.model_tier == "complex"
        assert config.temperature == 0.7
        assert config.max_tokens == 8000
        assert config.include_sources is True
        assert config.include_citations is True
        assert config.custom_schema is None
        assert config.custom_prompt is None
        assert config.extra_config == {}

    def test_create_with_custom_values(self) -> None:
        """Should accept custom values."""
        config = SynthesizerConfig(
            output_type="meeting_prep",
            model_tier="analytical",
            temperature=0.3,
            max_tokens=4000,
            include_sources=False,
            include_citations=True,
            custom_schema={"type": "object"},
            custom_prompt="Custom prompt here",
            extra_config={"key": "value"},
        )
        assert config.output_type == "meeting_prep"
        assert config.model_tier == "analytical"
        assert config.temperature == 0.3
        assert config.max_tokens == 4000
        assert config.include_sources is False
        assert config.custom_schema == {"type": "object"}
        assert config.custom_prompt == "Custom prompt here"
        assert config.extra_config["key"] == "value"

    def test_get_effective_schema_default(self) -> None:
        """Should return SynthesisReport schema by default."""
        config = SynthesizerConfig()
        schema = config.get_effective_schema()
        assert schema == SynthesisReport.get_schema()

    def test_get_effective_schema_custom(self) -> None:
        """Should return custom schema when set."""
        custom = {"type": "object", "properties": {"custom": {"type": "string"}}}
        config = SynthesizerConfig(custom_schema=custom)
        schema = config.get_effective_schema()
        assert schema == custom

    def test_merge_with_basic(self) -> None:
        """Should merge configs with other taking precedence."""
        base = SynthesizerConfig(
            output_type="base",
            model_tier="simple",
            temperature=0.5,
        )
        other = SynthesizerConfig(
            output_type="other",
            model_tier="complex",
        )
        merged = base.merge_with(other)

        assert merged.output_type == "other"
        assert merged.model_tier == "complex"
        # base.temperature (0.5) is preserved since other has default 0.7
        assert merged.temperature == 0.5

    def test_merge_with_other_overrides_when_non_default(self) -> None:
        """Should use other's value when other has non-default."""
        base = SynthesizerConfig(
            temperature=0.3,
            max_tokens=4000,
            custom_prompt="Base prompt",
        )
        other = SynthesizerConfig(
            temperature=0.5,  # Non-default
            max_tokens=2000,  # Non-default
        )
        merged = base.merge_with(other)

        # other has non-default values, so they override base
        assert merged.temperature == 0.5
        assert merged.max_tokens == 2000
        # custom_prompt: other is None, base is preserved via "or"
        assert merged.custom_prompt == "Base prompt"

    def test_merge_with_extra_config(self) -> None:
        """Should merge extra_config dicts."""
        base = SynthesizerConfig(
            extra_config={"key1": "value1", "key2": "base"},
        )
        other = SynthesizerConfig(
            extra_config={"key2": "other", "key3": "value3"},
        )
        merged = base.merge_with(other)

        assert merged.extra_config["key1"] == "value1"  # From base
        assert merged.extra_config["key2"] == "other"  # Overridden by other
        assert merged.extra_config["key3"] == "value3"  # From other
