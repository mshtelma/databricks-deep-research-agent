"""
Report generation utilities.

Provides reusable utilities for all report generation strategies:
- Citation processing and formatting
- Metadata generation
- Content sanitization
- Quality metrics extraction

All utilities are stateless and type-safe for maximum testability.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ...core.report_styles import ReportFormatter, ReportStyle
from ...core import get_logger

logger = get_logger(__name__)


# === Citation Processing ===

class CitationProcessor:
    """
    Handles all citation-related operations.

    Provides utilities for:
    - Field extraction from various citation formats
    - Deduplication by URL
    - References section formatting
    """

    @staticmethod
    def extract_field(citation: Any, field: str, default: str = "") -> str:
        """
        Extract field from citation regardless of format.

        Handles three formats:
        1. Citation object with attributes
        2. Dict with keys
        3. String representation (regex parsing)

        Args:
            citation: Citation in any format
            field: Field name ('source', 'title', 'url', etc.)
            default: Default if field not found

        Returns:
            Field value as string
        """
        # Try attribute access (Citation object)
        # But skip if it's a callable (like str.title method)
        if hasattr(citation, field):
            value = getattr(citation, field, None)
            if value is not None and not callable(value):
                return str(value)
            elif value is None:
                return default

        # Try dict access
        if isinstance(citation, dict):
            value = citation.get(field)
            return str(value) if value is not None else default

        # Fallback for strings - try regex parsing
        if isinstance(citation, str) and field in citation:
            import re
            pattern = rf"{field}='([^']*)'"
            match = re.search(pattern, citation)
            if match:
                return match.group(1)

        return default

    @staticmethod
    def deduplicate(citations: List[Any]) -> List[Any]:
        """
        Remove duplicate citations by URL.

        Uses URL as unique key. Preserves first occurrence.

        Args:
            citations: List of citations in any format

        Returns:
            Deduplicated list
        """
        if not citations:
            return []

        seen_urls = set()
        unique = []

        for citation in citations:
            url = CitationProcessor.extract_field(citation, 'source')
            if not url:
                url = CitationProcessor.extract_field(citation, 'url')

            if url and url not in seen_urls:
                unique.append(citation)
                seen_urls.add(url)

        logger.debug(f"Deduplicated citations: {len(citations)} → {len(unique)}")
        return unique

    @staticmethod
    def format_references_section(
        citations: List[Any],
        style: str,
        formatter: ReportFormatter
    ) -> str:
        """
        Format citations into a references section.

        Args:
            citations: List of citations (already deduplicated)
            style: Report style ('default', 'academic', etc.)
            formatter: ReportFormatter instance

        Returns:
            Formatted references section
        """
        if not citations:
            return ""

        # Convert style string to enum if needed
        if isinstance(style, str):
            try:
                style_enum = ReportStyle(style)
            except ValueError:
                style_enum = ReportStyle.DEFAULT
        else:
            style_enum = style

        # Format each citation
        formatted_citations = []
        for i, citation in enumerate(citations, 1):
            citation_dict = {
                "number": i,
                "title": CitationProcessor.extract_field(citation, 'title'),
                "url": CitationProcessor.extract_field(citation, 'source') or
                       CitationProcessor.extract_field(citation, 'url'),
                "author": "",  # Leave empty - avoid "Unknown"
                "date": ""     # Leave empty - avoid default dates
            }
            formatted = formatter.format_citation(citation_dict, style_enum)
            formatted_citations.append(formatted)

        # Build references section based on style
        references_header = formatter.format_section_header("References", style_enum)

        if style_enum == ReportStyle.ACADEMIC:
            # APA style bibliography
            return references_header + "\n".join(formatted_citations)

        elif style_enum == ReportStyle.TECHNICAL:
            # Numbered references
            lines = [references_header]
            for i, citation in enumerate(formatted_citations, 1):
                lines.append(f"[{i}] {citation}")
            return "\n".join(lines)

        elif style_enum == ReportStyle.SOCIAL_MEDIA:
            # Simple links (limit to 3 for brevity)
            lines = ["\n📚 Sources:"]
            for citation in citations[:3]:
                title = CitationProcessor.extract_field(citation, 'title')
                url = CitationProcessor.extract_field(citation, 'source')
                lines.append(f"• {title[:50]}... [{url}]")
            return "\n".join(lines)

        else:
            # Default format - bullet list
            lines = [references_header]
            for citation in formatted_citations:
                lines.append(f"• {citation}")
            return "\n".join(lines)


# === Quality Metrics ===

class QualityMetrics(BaseModel):
    """Quality metrics extracted from state."""
    factuality_score: float = Field(default=0.9, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    coverage_score: float = Field(default=0.7, ge=0.0, le=1.0)
    research_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)

    class Config:
        extra = "allow"


class MetadataBuilder:
    """
    Builds report metadata and footers.

    Provides utilities for:
    - Quality metrics extraction
    - Footer generation
    - Generation info
    """

    @staticmethod
    def extract_quality_metrics(state: Dict[str, Any]) -> QualityMetrics:
        """
        Extract quality metrics from state.

        Provides sensible defaults if metrics missing.

        Args:
            state: Research state dict

        Returns:
            QualityMetrics model
        """
        return QualityMetrics(
            factuality_score=state.get("factuality_score", 0.9),
            confidence_score=state.get("confidence_score", 0.8),
            coverage_score=state.get("coverage_score", 0.7),
            research_quality_score=state.get("research_quality_score", 0.8)
        )

    @staticmethod
    def build_footer(
        state: Dict[str, Any],
        style: str,
        quality_metrics: Optional[QualityMetrics] = None
    ) -> str:
        """
        Build report footer with metadata.

        Args:
            state: Research state dict
            style: Report style
            quality_metrics: Optional pre-extracted metrics

        Returns:
            Formatted footer string
        """
        # Convert style string to enum
        if isinstance(style, str):
            try:
                style_enum = ReportStyle(style)
            except ValueError:
                style_enum = ReportStyle.DEFAULT
        else:
            style_enum = style

        # Social media gets no footer
        if style_enum == ReportStyle.SOCIAL_MEDIA:
            return ""

        # Extract metrics if not provided
        if quality_metrics is None:
            quality_metrics = MetadataBuilder.extract_quality_metrics(state)

        # Build footer parts
        parts = [
            "\n" + "=" * 50 + "\n",
            "Generated by Deep Research Agent\n",
            f"Report Style: {style_enum.value}\n",
            f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n",
            f"Factuality Score: {quality_metrics.factuality_score:.2f}\n",
            f"Confidence Score: {quality_metrics.confidence_score:.2f}\n",
            f"Coverage Score: {quality_metrics.coverage_score:.2f}\n",
            f"Research Quality: {quality_metrics.research_quality_score:.2f}\n",
        ]

        # Add research progress if available
        plan = state.get("current_plan")
        completed_steps = state.get("completed_steps", [])

        if plan:
            total_steps, num_completed = MetadataBuilder._extract_step_counts(
                plan, completed_steps
            )
            if total_steps > 0:
                parts.append(
                    f"Research Steps Completed: {num_completed}/{total_steps}\n"
                )

        parts.append("=" * 50)

        return "".join(parts)

    @staticmethod
    def _extract_step_counts(plan: Any, completed_steps: List) -> tuple:
        """Extract total and completed step counts from plan."""
        # Handle dict
        if isinstance(plan, dict):
            total = len(plan.get('steps', []))
            completed = plan.get('completed_steps', len(completed_steps))
            return total, completed

        # Handle Plan object
        if hasattr(plan, 'steps'):
            total = len(plan.steps)
            completed = len(completed_steps)
            return total, completed

        # Fallback
        return 0, len(completed_steps)


# === Content Sanitization ===

class ReportSanitizer:
    """
    Sanitizes and normalizes report content.

    Provides utilities for:
    - Type normalization (list/None → string)
    - Safe concatenation
    - Content validation
    """

    @staticmethod
    def ensure_string(content: Any) -> str:
        """
        Convert content to string, handling edge cases.

        Handles:
        - None → empty string
        - list → joined with newlines
        - other types → str()

        Args:
            content: Content in any format

        Returns:
            String representation
        """
        if content is None:
            logger.debug("ensure_string: received None, returning empty string")
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            logger.debug(f"ensure_string: converting list of {len(content)} items to string")
            return "\n".join(str(item) for item in content if item is not None)

        logger.debug(f"ensure_string: converting {type(content).__name__} to string")
        return str(content)

    @staticmethod
    def append_section(
        report: str,
        section: str,
        separator: str = "\n\n"
    ) -> str:
        """
        Safely append section to report.

        Ensures both parts are strings before concatenation.

        Args:
            report: Main report content
            section: Section to append
            separator: Separator between sections

        Returns:
            Combined string
        """
        report_str = ReportSanitizer.ensure_string(report)
        section_str = ReportSanitizer.ensure_string(section)

        if not section_str:
            return report_str

        if not report_str:
            return section_str

        return report_str + separator + section_str


# === Main Facade ===

class ReporterUtilities:
    """
    Main facade for all reporter utilities.

    Provides simple interface to:
    - Citation processing
    - Metadata generation
    - Content sanitization

    Example:
        utils = ReporterUtilities()

        # Add citations
        report = utils.add_citations(report, citations, "academic")

        # Add metadata
        report = utils.add_metadata(report, state, "professional")
    """

    def __init__(self, formatter: Optional[ReportFormatter] = None):
        """
        Initialize utilities.

        Args:
            formatter: Optional ReportFormatter (creates default if None)
        """
        self.formatter = formatter or ReportFormatter()
        self.citations = CitationProcessor()
        self.metadata = MetadataBuilder()
        self.sanitizer = ReportSanitizer()

    def add_citations(
        self,
        report: str,
        citations: List[Any],
        style: str = "default"
    ) -> str:
        """
        Add citations section to report.

        Handles:
        - Deduplication
        - Style-specific formatting
        - Safe concatenation

        Args:
            report: Main report content
            citations: List of citations
            style: Report style

        Returns:
            Report with citations appended
        """
        if not citations:
            return report

        # Sanitize report
        report = self.sanitizer.ensure_string(report)

        # Deduplicate citations
        unique_citations = self.citations.deduplicate(citations)

        # Format references section
        references = self.citations.format_references_section(
            unique_citations,
            style,
            self.formatter
        )

        # Append to report
        return self.sanitizer.append_section(report, references)

    def add_metadata(
        self,
        report: str,
        state: Dict[str, Any],
        style: str = "default"
    ) -> str:
        """
        Add metadata footer to report.

        Args:
            report: Main report content
            state: Research state dict
            style: Report style

        Returns:
            Report with metadata appended
        """
        # Sanitize report
        report = self.sanitizer.ensure_string(report)

        # Build footer
        footer = self.metadata.build_footer(state, style)

        # Append to report
        return self.sanitizer.append_section(report, footer, separator="\n")

    def add_both(
        self,
        report: str,
        citations: List[Any],
        state: Dict[str, Any],
        style: str = "default"
    ) -> str:
        """
        Add both citations and metadata (convenience method).

        Args:
            report: Main report content
            citations: List of citations
            state: Research state dict
            style: Report style

        Returns:
            Complete report with citations and metadata
        """
        report = self.add_citations(report, citations, style)
        report = self.add_metadata(report, state, style)
        return report
