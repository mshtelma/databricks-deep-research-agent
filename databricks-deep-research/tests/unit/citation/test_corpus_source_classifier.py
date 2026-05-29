"""Defect E / Part 2: shared corpus-source classifier (generic over kinds,
no hardcoded domain) and source_kind plumbing onto evidence types."""
from __future__ import annotations

from databricks_deep_research.citation.types import (
    EvidenceInfo,
    RankedEvidence,
    is_corpus_source_value,
)


def test_corpus_kinds_and_types_are_corpus() -> None:
    for v in ("vector_index", "sql_analytics", "qa_assistant", "file",
              "vector_search", "genie", "knowledge_assistant"):
        assert is_corpus_source_value(v) is True


def test_web_and_unknown_are_not_corpus() -> None:
    for v in ("web", "builtin", "", None, "something_else"):
        assert is_corpus_source_value(v) is False


def test_evidence_types_default_source_kind_none() -> None:
    assert RankedEvidence(source_url="u", quote_text="q", relevance_score=1.0).source_kind is None
    assert EvidenceInfo(source_url="u", quote_text="q").source_kind is None
    assert EvidenceInfo(source_url="u", quote_text="q", source_kind="file").to_dict()["source_kind"] == "file"
