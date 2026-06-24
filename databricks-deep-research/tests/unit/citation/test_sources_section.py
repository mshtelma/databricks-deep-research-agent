"""Tests for the deterministic ``## Sources`` section (feature 4.2).

Covers ``render_sources_section`` in the synthesizer:
  * cited-vs-consulted partition driven by numeric markers in the report;
  * failed-crawl annotation via the UrlRegistry;
  * a report with no markers still renders a sane consulted-only list;
  * the ``append_sources_section`` gate helper default.
"""

from __future__ import annotations

from databricks_deep_research.agents.builtins.synthesizer import (
    _append_sources_section_enabled,
    render_sources_section,
)
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.citation.types import RankedEvidence
from databricks_deep_research.tools.protocol import UrlRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source(url: str, title: str) -> dict:
    return {"url": url, "canonical_url": url, "title": title, "snippet": "", "content": ""}


def _evidence(url: str, *, pool_index: int) -> RankedEvidence:
    return RankedEvidence(
        source_url=url,
        quote_text="some quote",
        relevance_score=1.0,
        source_pool_index=pool_index,
    )


# ---------------------------------------------------------------------------
# Partition: cited vs consulted
# ---------------------------------------------------------------------------


def test_partitions_cited_and_consulted() -> None:
    """Sources whose numeric marker appears in the report are listed under
    'Cited'; the rest under 'Consulted'."""
    sources = [
        _source("https://a.com", "Source A"),  # index 0 — cited
        _source("https://b.com", "Source B"),  # index 1 — not cited
        _source("https://c.com", "Source C"),  # index 2 — cited
    ]
    evidence_pool = [
        _evidence("https://a.com", pool_index=0),
        _evidence("https://b.com", pool_index=1),
        _evidence("https://c.com", pool_index=2),
    ]
    report = "Finding one [0] and finding two [2]."

    section = render_sources_section(report, sources, evidence_pool)

    assert section.startswith("\n## Sources")
    assert "### Cited" in section
    assert "### Consulted" in section
    # Cited entries
    assert "[0] [Source A](https://a.com)" in section
    assert "[2] [Source C](https://c.com)" in section
    # Consulted entry
    assert "[1] [Source B](https://b.com)" in section
    # B must appear after the Consulted heading, A before it (cited-first).
    assert section.index("Source A") < section.index("### Consulted")
    assert section.index("Source B") > section.index("### Consulted")


def test_evidence_without_marker_in_report_is_consulted() -> None:
    """An evidence-backed source whose marker is absent from the report is
    treated as consulted, not cited."""
    sources = [_source("https://a.com", "Source A")]
    evidence_pool = [_evidence("https://a.com", pool_index=0)]
    report = "No markers here at all."

    section = render_sources_section(report, sources, evidence_pool)

    assert "### Cited" not in section
    assert "### Consulted" in section
    assert "[0] [Source A](https://a.com)" in section


# ---------------------------------------------------------------------------
# No markers / empty inputs
# ---------------------------------------------------------------------------


def test_report_with_no_markers_renders_consulted_only() -> None:
    """A marker-free report still yields a sane consulted list (no Cited block)."""
    sources = [
        _source("https://a.com", "Source A"),
        _source("https://b.com", "Source B"),
    ]
    section = render_sources_section("Plain prose, no citations.", sources, evidence_pool=None)

    assert "## Sources" in section
    assert "### Cited" not in section
    assert "### Consulted" in section
    assert "[0] [Source A](https://a.com)" in section
    assert "[1] [Source B](https://b.com)" in section


def test_no_sources_returns_empty_string() -> None:
    """No sources -> empty string (caller appends nothing)."""
    assert render_sources_section("Anything [0].", [], evidence_pool=None) == ""


# ---------------------------------------------------------------------------
# Failed-crawl annotation
# ---------------------------------------------------------------------------


def test_failed_crawl_is_annotated() -> None:
    """A source whose URL has a recorded crawl failure is annotated."""
    sources = [_source("https://broken.com", "Broken Source")]
    registry = UrlRegistry()
    registry.register("https://broken.com")
    registry.record_non_retryable_failure("https://broken.com", "http_403")

    section = render_sources_section(
        "Prose with no markers.",
        sources,
        evidence_pool=None,
        url_registry=registry,
    )

    assert "[0] [Broken Source](https://broken.com)" in section
    assert "crawl failed: http_403" in section


def test_no_annotation_without_registry() -> None:
    """Without a registry, no failure annotations are emitted."""
    sources = [_source("https://broken.com", "Broken Source")]
    section = render_sources_section("Prose.", sources, evidence_pool=None)
    assert "crawl failed" not in section


# ---------------------------------------------------------------------------
# Gate helper
# ---------------------------------------------------------------------------


def test_append_sources_section_default_on() -> None:
    """The gate defaults ON when output_schema is absent or omits the key."""
    cfg = AgentNodeConfig(subtype="synthesizer")
    assert _append_sources_section_enabled(cfg) is True


def test_append_sources_section_can_be_disabled() -> None:
    """``output_schema.append_sources_section = False`` suppresses the block."""
    cfg = AgentNodeConfig(
        subtype="synthesizer",
        output_schema={"append_sources_section": False},
    )
    assert _append_sources_section_enabled(cfg) is False
