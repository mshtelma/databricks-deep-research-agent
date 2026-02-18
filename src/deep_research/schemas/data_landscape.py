"""Data Landscape schemas for enterprise data source discovery.

These models represent the results of background data source exploration
that happens before planning. The DataLandscape provides the planner
with awareness of available data sources and their relevance.

Part of 007-enterprise-data-sources feature (T027).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SourceDiscoveryResult:
    """Result from discovering/exploring a single data source.

    Created during background investigation when we query each
    enabled data source with exploratory queries to assess relevance.
    """

    source_name: str
    """Name of the data source."""

    source_type: str
    """Type of source (vector_search, genie, web_search, etc.)."""

    relevance_score: float
    """0.0-1.0 relevance score based on exploration results."""

    sample_results: list[dict[str, Any]] = field(default_factory=list)
    """Sample results from exploration (limited to 3-5)."""

    available_filters: list[str] = field(default_factory=list)
    """Filter columns available for this source (VS sources)."""

    suggested_queries: list[str] = field(default_factory=list)
    """Suggested queries that might work well with this source."""

    error_message: str | None = None
    """Error message if discovery failed for this source."""

    query_used: str | None = None
    """The exploratory query that was used."""

    response_time_ms: float = 0.0
    """Time taken to query this source in milliseconds."""

    has_results: bool = False
    """Whether any results were returned."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "relevance_score": self.relevance_score,
            "sample_results": self.sample_results,
            "available_filters": self.available_filters,
            "suggested_queries": self.suggested_queries,
            "error_message": self.error_message,
            "query_used": self.query_used,
            "response_time_ms": self.response_time_ms,
            "has_results": self.has_results,
        }


@dataclass
class DataLandscape:
    """Aggregated discovery results across all data sources.

    Provides the planner with a complete picture of available
    data sources and their relevance to the query.
    """

    query: str
    """The original user query."""

    discovery_results: list[SourceDiscoveryResult] = field(default_factory=list)
    """Results from exploring each enabled source."""

    top_sources: list[str] = field(default_factory=list)
    """Source names ranked by relevance (top 5)."""

    capabilities_map: dict[str, list[str]] = field(default_factory=dict)
    """Map from capability -> source names that have it."""

    total_discovery_time_ms: float = 0.0
    """Total time taken for all discovery queries."""

    sources_queried: int = 0
    """Number of sources that were queried."""

    sources_with_results: int = 0
    """Number of sources that returned relevant results."""

    discovered_at: datetime | None = None
    """When discovery was performed."""

    def get_source_by_name(self, name: str) -> SourceDiscoveryResult | None:
        """Get discovery result for a specific source."""
        for result in self.discovery_results:
            if result.source_name == name:
                return result
        return None

    def get_sources_by_type(self, source_type: str) -> list[SourceDiscoveryResult]:
        """Get all sources of a specific type."""
        return [r for r in self.discovery_results if r.source_type == source_type]

    def get_relevant_sources(self, min_score: float = 0.3) -> list[SourceDiscoveryResult]:
        """Get sources with relevance above threshold."""
        return [
            r for r in self.discovery_results
            if r.relevance_score >= min_score and r.has_results
        ]

    def to_planner_summary(self) -> str:
        """Convert to summary string for planner consumption.

        Returns a formatted string that describes available data sources
        and their relevance, suitable for including in the planner prompt.
        """
        if not self.discovery_results:
            return "No enterprise data sources available. Use web search only."

        lines: list[str] = [
            "## Available Data Sources",
            "",
            f"Discovery queried {self.sources_queried} sources in {self.total_discovery_time_ms:.0f}ms.",
            f"{self.sources_with_results} sources returned relevant results.",
            "",
        ]

        # Group by relevance tier
        high_relevance = [r for r in self.discovery_results if r.relevance_score >= 0.7]
        medium_relevance = [r for r in self.discovery_results if 0.3 <= r.relevance_score < 0.7]
        low_relevance = [r for r in self.discovery_results if r.relevance_score < 0.3 and r.has_results]

        if high_relevance:
            lines.append("### High Relevance Sources (use first)")
            for r in sorted(high_relevance, key=lambda x: -x.relevance_score):
                lines.append(f"- **{r.source_name}** ({r.source_type}): score={r.relevance_score:.2f}")
                if r.suggested_queries:
                    lines.append(f"  Suggested: {', '.join(r.suggested_queries[:2])}")
            lines.append("")

        if medium_relevance:
            lines.append("### Medium Relevance Sources (use if needed)")
            for r in sorted(medium_relevance, key=lambda x: -x.relevance_score):
                lines.append(f"- **{r.source_name}** ({r.source_type}): score={r.relevance_score:.2f}")
            lines.append("")

        if low_relevance:
            lines.append("### Low Relevance Sources (probably not useful)")
            for r in low_relevance:
                lines.append(f"- {r.source_name} ({r.source_type})")
            lines.append("")

        # Add capabilities
        if self.capabilities_map:
            lines.append("### Capabilities by Source Type")
            for capability, sources in self.capabilities_map.items():
                lines.append(f"- {capability}: {', '.join(sources)}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "discovery_results": [r.to_dict() for r in self.discovery_results],
            "top_sources": self.top_sources,
            "capabilities_map": self.capabilities_map,
            "total_discovery_time_ms": self.total_discovery_time_ms,
            "sources_queried": self.sources_queried,
            "sources_with_results": self.sources_with_results,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
        }
