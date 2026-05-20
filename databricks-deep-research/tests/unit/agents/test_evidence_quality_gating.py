from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.execution.output_normalizer import (
    build_source_records,
    normalize_research_output,
    source_is_substantive,
)


def _config() -> AgentNodeConfig:
    return AgentNodeConfig(subtype="researcher", output_key="findings")


def test_normalizer_drops_accepted_empty_sources_from_citeable_outputs() -> None:
    result = normalize_research_output(
        {
            "findings": "The search returned only metadata.",
            "sources": [
                {
                    "url": "https://example.com/metadata",
                    "title": "Metadata only",
                    "evidence_quality": "empty",
                    "admission_status": "accepted",
                }
            ],
        },
        _config(),
        tool_sources=[],
    )

    assert result is not None
    assert result.sources == []
    assert result.substantive_source_count == 0
    assert result.low_value_source_count == 0
    assert result.skip_source_writes is True
    assert result.research_status == "insufficient_data"


def test_normalizer_keeps_sources_with_usable_text() -> None:
    result = normalize_research_output(
        {
            "findings": "The crawled source contained usable text.",
            "sources": [
                {
                    "url": "https://example.com/article",
                    "title": "Article",
                    "snippet": "The article contains a concrete evidence sentence.",
                    "evidence_quality": "snippet_only",
                    "admission_status": "accepted",
                }
            ],
        },
        _config(),
        tool_sources=[],
    )

    assert result is not None
    assert len(result.sources) == 1
    assert result.substantive_source_count == 1
    assert result.skip_source_writes is False


def test_build_source_records_skips_metadata_only_sources() -> None:
    records = build_source_records(
        [
            {
                "url": "https://example.com/meta",
                "title": "Metadata",
                "evidence_quality": "metadata_only",
                "admission_status": "accepted_low_value",
            },
            {
                "url": "https://example.com/text",
                "title": "Text",
                "snippet": "Usable evidence.",
                "evidence_quality": "snippet_only",
                "admission_status": "accepted",
            },
        ]
    )

    assert [record.url for record in records] == ["https://example.com/text"]


def test_source_is_substantive_supports_typed_source_records() -> None:
    record = build_source_records(
        [
            {
                "url": "https://example.com/text",
                "title": "Text",
                "snippet": "Usable evidence.",
                "evidence_quality": "snippet_only",
                "admission_status": "accepted",
            }
        ]
    )[0]

    assert source_is_substantive(record)
