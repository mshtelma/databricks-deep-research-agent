"""
Phase 4: Main Pipeline Integration for Structured Report Generation

This module provides the main entry point that integrates all phases:
- Phase 0: Instrumentation
- Phase 1: Model extensions
- Phase 2: Structured generation
- Phase 3: Programmatic table building
- Phase 4: Assembly and rendering
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.report_generation.models import (
    CalculationContext,
    ComparisonEntry,
    SectionDraft
)
from ..core.report_models import (
    ContentBlock,
    ReportSection
)
from ..core.report_models_structured import StructuredReport
from ..core.multi_agent_state import EnhancedResearchState

from .reporter_instrumentation import ReporterInstrumentation
from .reporter_structured import StructuredReportGenerator

logger = logging.getLogger(__name__)


def _get_calc_attr(calc_context: Any, attr: str, default: Any = None) -> Any:
    """
    Safely extract attribute from calc_context (handles both dict and object).

    Args:
        calc_context: CalculationContext object or dict
        attr: Attribute name
        default: Default value if not found

    Returns:
        Attribute value or default
    """
    if isinstance(calc_context, dict):
        return calc_context.get(attr, default)
    else:
        return getattr(calc_context, attr, default)


class StructuredReportPipeline:
    """
    Main pipeline for structured report generation.

    Integrates all phases to produce reports with perfect markdown tables.
    """

    def __init__(self, llm, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.

        Args:
            llm: Language model for narrative generation
            config: Optional configuration
        """
        self.llm = llm
        self.config = config or {}
        self.instrumentation = ReporterInstrumentation()
        self.generator = StructuredReportGenerator(llm, config)

    async def generate_report(
        self,
        state: EnhancedResearchState,
        findings: Dict[str, Any],
        calc_context: CalculationContext
    ) -> str:
        """
        End-to-end pipeline using structured sections and deterministic tables.

        Args:
            state: Research state with plan
            findings: Compiled research findings
            calc_context: Calculation context with data

        Returns:
            Final markdown report with perfect tables
        """
        try:
            # Phase 0: Instrument and validate
            logger.info("Phase 0: Instrumenting state and context")
            metrics = self.instrumentation.instrument_state(state, calc_context)

            # Log critical issues
            if metrics["validation_results"]["critical_issues"]:
                for issue in metrics["validation_results"]["critical_issues"]:
                    logger.warning(f"Critical validation issue: {issue}")

            # Phase 2: Generate structured sections (no inline tables!)
            logger.info("Phase 2: Generating structured sections")
            dynamic_sections = self.generator.plan_sections_from_state(state)

            draft = await self.generator.generate_holistic_report_structured(
                findings,
                calc_context,
                dynamic_sections
            )

            # Phase 3: Build tables programmatically
            logger.info("Phase 3: Building tables from data")
            tables_by_id = {}

            # FIXED: Handle both dict and object formats for calc_context
            table_specs = _get_calc_attr(calc_context, "table_specifications", [])
            key_comparisons = _get_calc_attr(calc_context, "key_comparisons", [])
            calculations = _get_calc_attr(calc_context, "calculations", [])

            # FIX #5: Log table spec availability for debugging
            if not table_specs:
                logger.error(
                    "[STRUCTURED PIPELINE] CRITICAL: calc_context.table_specifications is empty! "
                    "No tables will be built. This is likely due to unified planning not populating table_specs."
                )
            else:
                logger.info(
                    f"[STRUCTURED PIPELINE] Found {len(table_specs)} table specs "
                    f"and {len(key_comparisons)} comparisons"
                )

            for spec in table_specs:
                table, metadata = self.generator.build_table_from_comparisons(
                    spec,
                    key_comparisons,
                    calculations
                )
                tables_by_id[spec.table_id] = (table, metadata)
                logger.info(
                    f"Built table '{spec.table_id}' with "
                    f"{len(table.rows) - 1} data rows"  # -1 for header
                )

            # Phase 4: Assemble final report
            logger.info("Phase 4: Assembling final report")
            rendered_sections = self._assemble_sections(
                draft.sections,
                tables_by_id
            )

            # Create structured report
            structured_report = StructuredReport(
                title=findings.get("report_title",
                                   findings.get("research_topic", "Research Report")),
                key_points=self._generate_key_points(calc_context),
                overview=self._generate_overview(findings, calc_context),
                sections=rendered_sections,
                references=self._format_citations(findings.get("citations", [])),
                appendix=self._generate_appendix(calc_context)
            )

            # Render to markdown
            final_report = self._render_structured_report(structured_report)

            # Validate final output
            self._validate_final_report(final_report)

            logger.info(
                f"Generated report with {len(rendered_sections)} sections, "
                f"{len(tables_by_id)} tables"
            )

            return final_report

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            # Return minimal error report
            return self._generate_error_report(findings, str(e))

    def _assemble_sections(
        self,
        draft_sections: List[SectionDraft],
        tables_by_id: Dict[str, tuple]
    ) -> List[ReportSection]:
        """
        Assemble sections with narrative and tables.

        Args:
            draft_sections: Sections with narrative and table references
            tables_by_id: Dictionary of table_id -> (Table, metadata)

        Returns:
            List of ReportSection objects
        """
        rendered_sections = []

        for section in draft_sections:
            content_blocks = []

            # Add narrative paragraphs
            for paragraph in section.paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    content_blocks.append(
                        ContentBlock(
                            block_type="paragraph",
                            text=paragraph
                        )
                    )

            # Insert referenced tables
            for table_id in section.table_refs:
                if table_id not in tables_by_id:
                    logger.warning(
                        f"Table '{table_id}' referenced but not found, skipping"
                    )
                    continue

                table, metadata = tables_by_id[table_id]

                # Convert to markdown first for ContentBlock
                markdown = self.generator.render_table_to_markdown(table)

                content_blocks.append(
                    ContentBlock(
                        block_type="table",
                        table=table,  # Pass the Table object (required for validation)
                        text=markdown  # Also store rendered markdown
                    )
                )

            # Only add section if it has content
            if content_blocks:
                rendered_sections.append(
                    ReportSection(
                        title=section.title,
                        content_blocks=content_blocks
                    )
                )

        return rendered_sections

    def _generate_key_points(
        self,
        calc_context: CalculationContext
    ) -> List[str]:
        """Generate key points from calculation context."""
        key_points = []

        # Use summary insights if available
        summary_insights = _get_calc_attr(calc_context, "summary_insights", [])
        if summary_insights:
            key_points = summary_insights[:5]

        # Add data quality note if issues exist
        data_quality_notes = _get_calc_attr(calc_context, "data_quality_notes", [])
        if data_quality_notes:
            key_points.append(
                f"Note: {data_quality_notes[0]}"
            )

        # Ensure we have at least 3 points
        calculations = _get_calc_attr(calc_context, "calculations", [])
        extracted_data = _get_calc_attr(calc_context, "extracted_data", [])

        while len(key_points) < 3:
            if calculations:
                key_points.append(
                    f"Analysis includes {len(calculations)} calculations"
                )
            elif extracted_data:
                key_points.append(
                    f"Analyzed {len(extracted_data)} data points"
                )
            else:
                key_points.append("Further analysis recommended")

        return key_points[:5]  # Max 5 points

    def _generate_overview(
        self,
        findings: Dict[str, Any],
        calc_context: CalculationContext
    ) -> str:
        """Generate overview paragraph."""
        topic = findings.get("research_topic", "the requested analysis")

        overview = f"This report presents a comprehensive analysis of {topic}."

        extracted_data = _get_calc_attr(calc_context, "extracted_data", [])
        if extracted_data:
            overview += (
                f" The analysis is based on {len(extracted_data)} "
                f"data points extracted from research observations."
            )

        key_comparisons = _get_calc_attr(calc_context, "key_comparisons", [])
        if key_comparisons:
            entities = set(
                c.primary_key if hasattr(c, 'primary_key') else c.get('primary_key')
                for c in key_comparisons
            )
            if entities:
                overview += (
                    f" Key comparisons were conducted across "
                    f"{len(entities)} entities."
                )

        return overview

    def _format_citations(self, citations: List[Any]) -> List[str]:
        """Format citations as markdown links."""
        formatted = []

        for citation in citations:
            if isinstance(citation, dict):
                title = citation.get("title", "Source")
                url = citation.get("url", "")
                if url:
                    formatted.append(f"[{title}]({url})")
                else:
                    formatted.append(title)
            elif isinstance(citation, str):
                formatted.append(citation)

        return formatted

    def _generate_appendix(
        self,
        calc_context: CalculationContext
    ) -> str:
        """Generate appendix with formula transparency and data quality notes."""
        appendix_parts = []

        # Section 1: Calculation Formulas (NEW - Phase 4: Formula Transparency)
        formula_section = self._generate_formula_appendix(calc_context)
        if formula_section:
            appendix_parts.append(formula_section)

        # Section 2: Data Quality Notes (existing)
        data_quality_notes = _get_calc_attr(calc_context, "data_quality_notes", [])
        if data_quality_notes:
            quality_lines = ["### Data Quality Notes\n"]
            for note in data_quality_notes:
                quality_lines.append(f"- {note}")
            appendix_parts.append("\n".join(quality_lines))

        return "\n\n".join(appendix_parts) if appendix_parts else ""
    
    def _generate_formula_appendix(
        self,
        calc_context: CalculationContext
    ) -> str:
        """Generate formula transparency appendix section.

        Shows all calculation formulas with their sources for full transparency.

        Args:
            calc_context: Calculation context with calculations

        Returns:
            Formatted markdown section or empty string
        """
        calculations = _get_calc_attr(calc_context, "calculations", [])
        if not calculations:
            return ""

        # Filter calculations that have formulas
        calculations_with_formulas = [
            calc for calc in calculations
            if hasattr(calc, 'formula') and calc.formula
        ]
        
        if not calculations_with_formulas:
            return ""
        
        lines = ["### Calculation Formulas\n"]
        lines.append(
            "All derived metrics were calculated using the following formulas. "
            "Each formula shows the exact calculation method and data sources used.\n"
        )
        
        # Group by metric type if possible, otherwise show in order
        for calc in calculations_with_formulas:
            # Format: **Metric Name**: `formula`
            metric_name = calc.description or "Unknown metric"
            formula = calc.formula
            
            # Clean up formula for display
            formula_display = self._format_formula_for_display(formula)
            
            lines.append(f"- **{metric_name}**: `{formula_display}`")
            
            # Add source information if available
            if hasattr(calc, 'provenance') and calc.provenance:
                provenance = calc.provenance
                if hasattr(provenance, 'observation_ids') and provenance.observation_ids:
                    obs_ids = ', '.join(map(str, provenance.observation_ids))
                    lines.append(f"  - *Source: Observations {obs_ids}*")
                
                if hasattr(provenance, 'notes') and provenance.notes:
                    lines.append(f"  - *Note: {provenance.notes}*")
        
        return "\n".join(lines)
    
    def _format_formula_for_display(self, formula: str) -> str:
        """Format formula string for readable display.
        
        Args:
            formula: Raw formula string
        
        Returns:
            Cleaned formula string
        """
        # Remove excessive whitespace
        formula = ' '.join(formula.split())
        
        # Limit length for display
        max_length = 150
        if len(formula) > max_length:
            formula = formula[:max_length] + "..."
        
        return formula

    def _render_structured_report(
        self,
        structured_report: StructuredReport
    ) -> str:
        """
        Render StructuredReport to markdown.

        Generates clean markdown with perfect table formatting.
        """
        parts = []

        # Title
        parts.append(f"# {structured_report.title}\n")

        # Timestamp
        parts.append(
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        )

        # Key Points
        if structured_report.key_points:
            parts.append("## Key Points\n")
            for point in structured_report.key_points:
                parts.append(f"- {point}")
            parts.append("")

        # Overview
        if structured_report.overview:
            parts.append("## Overview\n")
            parts.append(structured_report.overview)
            parts.append("")

        # Main sections
        for section in structured_report.sections:
            parts.append(f"## {section.title}\n")

            for block in section.content_blocks:
                if block.block_type == "paragraph":
                    parts.append(block.text)
                    parts.append("")
                elif block.block_type == "table":
                    # Table is already rendered as markdown in text field
                    parts.append(block.text)
                    parts.append("")

        # References
        if structured_report.references:
            parts.append("## References\n")
            for ref in structured_report.references:
                if not ref.startswith("-"):
                    ref = f"- {ref}"
                parts.append(ref)
            parts.append("")

        # Appendix
        if structured_report.appendix:
            parts.append("## Appendix\n")
            parts.append(structured_report.appendix)
            parts.append("")

        return "\n".join(parts)

    def _validate_final_report(self, report: str):
        """Validate the final report for common issues."""
        lines = report.split("\n")

        # Check for broken tables
        table_started = False
        for i, line in enumerate(lines):
            if "|" in line:
                if not table_started:
                    table_started = True
                    # Next line should be separator
                    if i + 1 < len(lines) and "---" not in lines[i + 1]:
                        logger.warning(
                            f"Potential missing separator at line {i + 1}"
                        )

                # Check for split cells (common issue)
                if "Singl | le" in line or "Marri | ied" in line:
                    logger.error(
                        f"CRITICAL: Found split table cell at line {i + 1}"
                    )
            else:
                table_started = False

        # Check for inline table markers (should not exist)
        if "[TABLE:" in report:
            logger.error("CRITICAL: Found inline table markers in final report")

        # Log report statistics
        logger.info(
            f"Report validation: {len(lines)} lines, "
            f"{report.count('|')} pipe characters, "
            f"{report.count('## ')} sections"
        )

    def _generate_error_report(
        self,
        findings: Dict[str, Any],
        error: str
    ) -> str:
        """Generate minimal error report."""
        return f"""# Research Report

## Error During Generation

An error occurred while generating the full report: {error}

## Research Topic

{findings.get('research_topic', 'Unknown')}

## Summary

Unable to generate complete analysis due to processing error.
Please review the error message and try again.

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""