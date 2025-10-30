"""
Template-based report generator (Tier 3).

Simplest generation strategy using one-shot LLM calls with templates.
Used as last-resort fallback when advanced strategies unavailable.

Features:
- Two-stage generation (analysis → formatting)
- Template support (explicit or auto-generated)
- Minimal complexity for maximum reliability
- Quick fallback when observations minimal
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from ...core import get_logger
from ...core.observation_models import StructuredObservation
from ...core.advanced_utilities import ReporterLLMInvocationManager

from .base import BaseReportGenerator
from .types import (
    ReportGenerationRequest,
    ReportGenerationResult,
    ReporterConfig,
    GenerationMode,
    ReportQuality,
    TableGenerationMode,
    GenerationError,
    ReportSection,
)
from .utilities import ReporterUtilities

logger = get_logger(__name__)


class TemplateGenerator(BaseReportGenerator):
    """
    Tier 3: Template-based report generation.

    Generates reports using explicit templates or basic structure.
    Simplest strategy - two LLM calls (analysis + formatting).

    Used when:
    - No dynamic sections available
    - Explicit template provided
    - Last resort fallback

    Quality: LOW (basic structure, generic content)
    Speed: FAST (fewest LLM calls)
    Reliability: HIGH (always works if observations exist)
    """

    # Strategy metadata
    name = "template"
    generation_mode = GenerationMode.TEMPLATE
    quality_level = ReportQuality.LOW

    def __init__(
        self,
        llm: Any,
        config: ReporterConfig,
        utilities: ReporterUtilities,
        **kwargs
    ):
        """
        Initialize template generator.

        Args:
            llm: Language model instance
            config: Typed reporter configuration
            utilities: Shared reporter utilities
            **kwargs: Additional args for BaseReportGenerator
        """
        super().__init__(llm, config, **kwargs)
        self.utilities = utilities
        self.llm_manager = ReporterLLMInvocationManager(llm)

    def can_handle(
        self,
        request: ReportGenerationRequest
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if template strategy can handle request.

        Template strategy can always handle if observations exist
        (it's the last resort fallback).

        Args:
            request: Generation request

        Returns:
            Tuple of (can_handle, reason_if_not)
        """
        if not request.observations:
            return False, "No observations available"

        # Template can always work as fallback
        return True, None

    async def generate(
        self,
        request: ReportGenerationRequest,
        config: ReporterConfig
    ) -> ReportGenerationResult:
        """
        Generate report using template strategy.

        Two-stage process:
        1. Analysis: LLM analyzes observations and extracts insights
        2. Formatting: LLM formats analysis into template structure

        Args:
            request: Typed generation request
            config: Typed reporter configuration

        Returns:
            Typed generation result

        Raises:
            GenerationError: If generation fails
        """
        start_time = time.time()

        async with self._generation_context(request) as trace_id:
            try:
                # Stage 1: Build context
                self._emit_progress("Building report context", 0.1)
                context = self._build_report_context(request)

                # Stage 2: Analysis phase
                self._emit_progress("Analyzing research data", 0.3)
                analysis = await self._analyze_research_data(
                    request, context, trace_id
                )
                self._count_llm_call()

                # Stage 3: Formatting phase
                self._emit_progress("Formatting report", 0.6)
                report_text = await self._format_into_template(
                    request, analysis, context, trace_id
                )
                self._count_llm_call()

                # Stage 4: Add citations and metadata
                self._emit_progress("Adding citations and metadata", 0.9)
                final_report = self.utilities.add_both(
                    report_text,
                    request.citations,
                    self._request_to_state_dict(request),
                    config.default_style
                )

                # Parse sections from report
                sections = self._parse_sections(final_report)

                # Build result
                generation_time = max(1, int((time.time() - start_time) * 1000))

                result = ReportGenerationResult(
                    final_report=final_report,
                    sections=sections,
                    generation_mode=GenerationMode.TEMPLATE,
                    table_mode=self._detect_table_mode(report_text),
                    quality=ReportQuality.LOW,
                    total_sections=len(sections),
                    total_tables=self._count_tables(report_text),
                    total_citations=len(request.citations),
                    observations_used=len(request.observations),
                    generation_time_ms=generation_time,
                    llm_calls=self._llm_call_count,
                    request_id=request.request_id
                )

                self._emit_progress("Template generation complete", 1.0)
                return result

            except Exception as e:
                logger.error(f"Template generation failed: {e}", exc_info=True)
                raise GenerationError(
                    f"Template generation failed: {str(e)}",
                    strategy=self.name,
                    recoverable=True,
                    fallback_suggestion=None,  # No fallback - this IS the fallback
                    original_error=e
                )

    # === Context Building (Static Methods) ===

    @staticmethod
    def build_observation_context(
        observations: List[Any],
        limit: int = 100
    ) -> Tuple[str, int]:
        """
        Convert observations into compact prompt context.

        Args:
            observations: List of StructuredObservation or dicts
            limit: Maximum observations to include

        Returns:
            Tuple of (context_string, num_truncated)
        """
        if not observations:
            return "No observations available.", 0

        lines = []
        truncated = 0

        for idx, obs in enumerate(observations):
            if idx >= limit:
                truncated = len(observations) - limit
                break

            # Handle StructuredObservation
            if isinstance(obs, StructuredObservation):
                content = obs.content
                source = obs.source_id or obs.step_id or obs.section_title
                metrics = obs.metric_values or {}
            # Handle dict
            elif isinstance(obs, dict):
                content = obs.get('content', str(obs))
                source = obs.get('source_id') or obs.get('step_id')
                metrics = obs.get('metric_values', {})
            else:
                # Fallback for unknown types
                content = str(obs)
                source = None
                metrics = {}

            # Build bullet point
            bullet = content.strip()
            if source:
                bullet += f" (Source: {source})"
            if metrics:
                # Show first 5 metrics
                metrics_str = ", ".join(
                    f"{k}: {v}" for k, v in list(metrics.items())[:5]
                )
                bullet += f" | Data: {metrics_str}"

            lines.append(f"- {bullet}")

        if truncated > 0:
            lines.append(f"- … {truncated} additional observations omitted")

        return "\n".join(lines), truncated

    @staticmethod
    def build_citation_context(citations: List[Any]) -> str:
        """
        Format citations for prompt context.

        Args:
            citations: List of citations in any format

        Returns:
            Formatted citation string
        """
        if not citations:
            return "No citations available."

        from .utilities import CitationProcessor

        lines = []
        for citation in citations:
            title = CitationProcessor.extract_field(citation, 'title', 'Untitled')
            source = CitationProcessor.extract_field(citation, 'source')
            url = CitationProcessor.extract_field(citation, 'url')

            entry = title
            if source and source not in title:
                entry = f"{entry} — {source}"
            if url:
                entry = f"{entry} ({url})"

            lines.append(f"- {entry}")

        return "\n".join(lines)

    @staticmethod
    def build_structured_table_context(
        observations: List[Any],
        max_columns: int = 6,
        max_rows: int = 12
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build markdown table from structured observation data.

        Extracts metric_values from observations and creates table.

        Args:
            observations: List of observations with metric data
            max_columns: Maximum table columns
            max_rows: Maximum table rows

        Returns:
            Tuple of (table_markdown, metadata_dict)
        """
        if not observations:
            return "", {}

        rows = []
        columns = []

        for obs in observations:
            # Extract metrics
            if isinstance(obs, StructuredObservation):
                metrics = obs.metric_values or {}
                label = obs.source_id or obs.step_id or obs.content[:80]
            elif isinstance(obs, dict):
                metrics = obs.get('metric_values', {})
                label = obs.get('source_id') or obs.get('content', '')[:80]
            else:
                continue

            if not metrics:
                continue

            # Build row
            row = {'Observation': str(label)[:80]}
            for key, value in metrics.items():
                if key not in columns and len(columns) < max_columns:
                    columns.append(key)
                if key in columns:
                    row[key] = str(value)

            rows.append(row)
            if len(rows) >= max_rows:
                break

        if not rows:
            return "", {}

        # Build markdown table
        ordered_cols = ['Observation'] + columns
        header = "| " + " | ".join(ordered_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(ordered_cols)) + " |"

        body_lines = []
        for row in rows:
            body_lines.append(
                "| " + " | ".join(row.get(col, "") for col in ordered_cols) + " |"
            )

        table_text = "\n".join([header, separator, *body_lines])
        metadata = {
            'table_columns': ordered_cols,
            'table_rows': len(rows)
        }

        return table_text, metadata

    # === Private Methods ===

    def _build_report_context(
        self,
        request: ReportGenerationRequest
    ) -> Dict[str, Any]:
        """Build all context needed for template generation."""
        obs_context, truncated = self.build_observation_context(
            request.observations,
            limit=100
        )

        citation_context = self.build_citation_context(request.citations)

        table_context, table_meta = self.build_structured_table_context(
            request.observations
        )

        # Extract entity list if available
        entities = []
        if request.query_constraints and hasattr(request.query_constraints, 'entities'):
            entities = request.query_constraints.entities

        return {
            'observations': obs_context,
            'observations_truncated': truncated,
            'citations': citation_context,
            'structured_table': table_context,
            'table_metadata': table_meta,
            'entities': entities,
            'topic': request.research_topic,
            'style': request.report_style
        }

    async def _analyze_research_data(
        self,
        request: ReportGenerationRequest,
        context: Dict[str, Any],
        trace_id: str
    ) -> str:
        """
        Stage 1: Analyze research data with LLM.

        Args:
            request: Generation request
            context: Built context dict
            trace_id: Trace ID for logging

        Returns:
            Analysis text from LLM
        """
        # Build guidelines
        guidelines = [
            f"Focus on the research topic: {context['topic']}",
            "Analyze all observations thoroughly",
            "Extract key insights and findings",
            "Perform calculations where needed",
            "Create comparative analysis",
            "Clearly state when data is missing",
        ]

        if context['entities']:
            guidelines.append(
                f"Cover these entities: {', '.join(context['entities'])}"
            )

        guideline_text = "\n".join(f"- {g}" for g in guidelines)

        # Build analysis prompt
        analysis_prompt = f"""# Task: Research Analysis

Perform detailed analysis for this research topic.

## Research Topic
{context['topic']}

## Guidelines
{guideline_text}

## Research Observations
{context['observations']}

## Citations
{context['citations']}

## Structured Data
{context['structured_table'] or 'No structured data available.'}

## Instructions
1. Analyze all research data thoroughly
2. Extract key insights and findings
3. Perform necessary calculations
4. Create comparative analysis
5. Note data quality and gaps

**Provide comprehensive analysis with insights, calculations, and findings.**
"""

        messages = [
            SystemMessage(
                content=(
                    "You are a research analyst. Analyze research data comprehensively. "
                    "Extract insights, perform calculations, identify patterns. "
                    "Your analysis will be used to create a structured report."
                )
            ),
            HumanMessage(content=analysis_prompt)
        ]

        # Invoke LLM
        logger.debug(f"[{trace_id}] Invoking LLM for analysis")
        analysis = self.llm_manager.invoke_with_smart_retry(
            messages,
            operation_name="template_analysis",
            state=self._request_to_state_dict(request)
        )

        logger.debug(f"[{trace_id}] Analysis complete: {len(analysis)} chars")
        return analysis

    async def _format_into_template(
        self,
        request: ReportGenerationRequest,
        analysis: str,
        context: Dict[str, Any],
        trace_id: str
    ) -> str:
        """
        Stage 2: Format analysis into template structure.

        Args:
            request: Generation request
            analysis: Analysis from stage 1
            context: Built context dict
            trace_id: Trace ID for logging

        Returns:
            Formatted report text
        """
        # Use explicit template if provided, else generate basic structure
        if request.has_template():
            template = request.report_template
            logger.debug(f"[{trace_id}] Using explicit template")
        else:
            template = self._generate_default_template(context)
            logger.debug(f"[{trace_id}] Using auto-generated template")

        # Build formatting prompt
        formatting_prompt = f"""Format the research analysis into the provided template structure.

## Your Analysis
{analysis}

## Template to Complete
{template}

## Critical Instructions
1. COMPLETE THE ENTIRE TEMPLATE - every section from start to finish
2. Replace ALL [bracketed instructions] with real content from your analysis
3. Use proper markdown formatting (headings, bullets, tables)
4. Include all sections - do not omit any
5. Do not stop early or truncate

**Output the complete template with every section filled.**
"""

        messages = [
            SystemMessage(
                content=(
                    "You are a report formatter. Complete the ENTIRE template provided. "
                    "Fill every section from start to finish. Replace all [bracketed instructions] "
                    "with real content. Do not stop early or omit sections."
                )
            ),
            HumanMessage(content=formatting_prompt)
        ]

        # Invoke LLM
        logger.debug(f"[{trace_id}] Invoking LLM for formatting")
        report_text = self.llm_manager.invoke_with_smart_retry(
            messages,
            operation_name="template_formatting",
            state=self._request_to_state_dict(request)
        )

        logger.debug(f"[{trace_id}] Formatting complete: {len(report_text)} chars")
        return report_text

    def _generate_default_template(self, context: Dict[str, Any]) -> str:
        """Generate basic template when none provided."""
        entities_section = ""
        if context['entities']:
            entities_section = f"""
## Coverage by Entity
[Provide analysis for each: {', '.join(context['entities'])}]
"""

        return f"""# {context['topic']}

## Executive Summary
[Provide 2-3 paragraph summary of key findings]

## Key Findings
[List 3-5 most important discoveries]

## Detailed Analysis
[Provide comprehensive analysis of the research data]
{entities_section}
## Data Quality
[Assess data completeness and reliability]

## Conclusion
[Summarize implications and recommendations]
"""

    def _detect_table_mode(self, text: str) -> TableGenerationMode:
        """Detect how tables were generated."""
        if '|' in text and '---' in text:
            return TableGenerationMode.LLM_MARKDOWN
        return TableGenerationMode.NONE

    def _request_to_state_dict(self, request: ReportGenerationRequest) -> Dict[str, Any]:
        """
        Convert request to state dict for backward compatibility.

        Only used for LLM invocation manager.
        """
        return {
            'research_topic': request.research_topic,
            'observations': [
                obs.to_dict() if hasattr(obs, 'to_dict') else obs
                for obs in request.observations
            ],
            'citations': request.citations,
            'factuality_score': request.factuality_score,
            'confidence_scores': request.confidence_scores
        }
