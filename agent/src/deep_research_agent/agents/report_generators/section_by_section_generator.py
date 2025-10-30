"""
Section-by-section report generator (Tier 2).

Production workhorse strategy for most research queries.
Generates sections individually using dynamic_sections from planner.

Features:
- Section-specific observation filtering
- Iterative section generation
- High-quality narrative synthesis
- LLM-generated tables (variable quality)
- Continuation awareness (sections reference previous content)

This is the DEFAULT strategy for qualitative research.
"""

import time
from typing import List, Dict, Any, Tuple, Optional, Sequence
from langchain_core.messages import SystemMessage, HumanMessage

from ...core import get_logger
from ...core.observation_models import StructuredObservation
from ...core.advanced_utilities import ReporterLLMInvocationManager
from ...core.observation_filter import ObservationFilter
from ...core.plan_models import DynamicSection
from ...core.table_processor import TableProcessor

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


class SectionBySectionGenerator(BaseReportGenerator):
    """
    Tier 2: Section-by-section report generation.

    Generates reports by processing each section individually based on
    dynamic_sections from the planner. Most reliable strategy for
    qualitative research.

    Process:
    1. Extract dynamic_sections from request
    2. For each section:
       - Filter observations for section (using step_id mapping)
       - Generate section content with LLM
       - Apply table validation
    3. Combine sections into final report
    4. Add citations and metadata

    Used when:
    - Dynamic sections available from planner
    - Qualitative research (non-metric focused)
    - Calculation results unavailable (hybrid fallback)

    Quality: MEDIUM (high narrative quality, variable table quality)
    Speed: MEDIUM (N LLM calls for N sections)
    Reliability: HIGH (fallback-friendly, handles missing data well)
    """

    # Strategy metadata
    name = "section_by_section"
    generation_mode = GenerationMode.SECTION_BY_SECTION
    quality_level = ReportQuality.MEDIUM

    def __init__(
        self,
        llm: Any,
        config: ReporterConfig,
        utilities: ReporterUtilities,
        **kwargs
    ):
        """
        Initialize section-by-section generator.

        Args:
            llm: Language model instance
            config: Typed reporter configuration
            utilities: Shared reporter utilities
            **kwargs: Additional args for BaseReportGenerator
        """
        super().__init__(llm, config, **kwargs)
        self.utilities = utilities
        self.llm_manager = ReporterLLMInvocationManager(llm)
        self.table_processor = TableProcessor()

        # Initialize observation filter with config
        self.observation_filter = ObservationFilter(
            fallback_limit=config.max_observations_per_section
        )

    def can_handle(
        self,
        request: ReportGenerationRequest
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if section-by-section strategy can handle request.

        Section-by-section requires dynamic_sections from planner.
        Falls back to template if sections unavailable.

        Args:
            request: Generation request

        Returns:
            Tuple of (can_handle, reason_if_not)
        """
        if not request.observations:
            return False, "No observations available"

        # Check for dynamic sections (required for section-by-section)
        if not request.has_dynamic_sections():
            return False, "No dynamic sections available (planner required)"

        # Section-by-section can handle if we have sections
        return True, None

    async def generate(
        self,
        request: ReportGenerationRequest,
        config: ReporterConfig
    ) -> ReportGenerationResult:
        """
        Generate report using section-by-section strategy.

        Four-phase process:
        1. Validate dynamic sections
        2. Generate each section individually
        3. Combine sections into final report
        4. Post-process with citations and metadata

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
                # Phase 1: Validate dynamic sections
                self._emit_progress("Validating section structure", 0.1)
                dynamic_sections = request.dynamic_sections

                if not dynamic_sections:
                    raise GenerationError(
                        "Dynamic sections unavailable",
                        strategy=self.name,
                        recoverable=True,
                        fallback_suggestion=GenerationMode.TEMPLATE
                    )

                logger.info(
                    f"[{trace_id}] Section-by-section: {len(dynamic_sections)} sections to generate"
                )

                # Phase 2: Generate sections individually
                self._emit_progress("Generating report sections", 0.2)
                section_dict, section_objects = await self._generate_all_sections(
                    request,
                    dynamic_sections,
                    trace_id
                )

                if not section_dict:
                    raise GenerationError(
                        "No sections generated (all skipped due to missing observations)",
                        strategy=self.name,
                        recoverable=True,
                        fallback_suggestion=GenerationMode.TEMPLATE
                    )

                # Phase 3: Combine sections into report
                self._emit_progress("Combining sections", 0.8)
                report_text = self._combine_sections(section_dict, request.research_topic)

                # Phase 4: Fix markdown tables (safety net)
                self._emit_progress("Validating tables", 0.85)
                report_text = self.table_processor.fix_markdown_tables(report_text)

                # Phase 5: Add citations and metadata
                self._emit_progress("Adding citations and metadata", 0.9)
                final_report = self.utilities.add_both(
                    report_text,
                    request.citations,
                    self._request_to_state_dict(request),
                    config.default_style
                )

                # Build result
                generation_time = max(1, int((time.time() - start_time) * 1000))

                result = ReportGenerationResult(
                    final_report=final_report,
                    sections=section_objects,
                    generation_mode=GenerationMode.SECTION_BY_SECTION,
                    table_mode=self._detect_table_mode(report_text),
                    quality=ReportQuality.MEDIUM,
                    total_sections=len(section_objects),
                    total_tables=self._count_tables(report_text),
                    total_citations=len(request.citations),
                    observations_used=len(request.observations),
                    generation_time_ms=generation_time,
                    llm_calls=self._llm_call_count,
                    request_id=request.request_id
                )

                # Emit section-specific metrics
                self._emit_section_metrics(result, len(dynamic_sections))

                self._emit_progress("Section-by-section generation complete", 1.0)
                return result

            except GenerationError:
                # Re-raise GenerationError as-is
                raise

            except Exception as e:
                logger.error(f"Section-by-section generation failed: {e}", exc_info=True)
                raise GenerationError(
                    f"Section-by-section generation failed: {str(e)}",
                    strategy=self.name,
                    recoverable=True,
                    fallback_suggestion=GenerationMode.TEMPLATE,
                    original_error=e
                )

    # === Phase-Specific Methods ===

    async def _generate_all_sections(
        self,
        request: ReportGenerationRequest,
        dynamic_sections: List[DynamicSection],
        trace_id: str
    ) -> Tuple[Dict[str, str], List[ReportSection]]:
        """
        Generate all sections iteratively.

        Each section:
        1. Filter observations using step_id mapping
        2. Generate content with LLM (with continuation awareness)
        3. Validate and clean tables
        4. Track for context in next section

        Args:
            request: Generation request
            dynamic_sections: List of dynamic sections from planner
            trace_id: Trace ID for logging

        Returns:
            Tuple of (section_dict, section_objects)
            - section_dict: {section_name: content_markdown}
            - section_objects: [ReportSection, ...]
        """
        section_dict = {}
        section_objects = []
        previous_section_titles = []

        # Sort by priority (as in original implementation)
        sorted_sections = sorted(dynamic_sections, key=lambda s: s.priority)
        total_sections = len(sorted_sections)

        for section_index, dynamic_section in enumerate(sorted_sections):
            section_name = dynamic_section.title

            # Update progress
            progress = 0.2 + (0.6 * (section_index / total_sections))
            self._emit_progress(
                f"Generating section: {section_name}",
                progress
            )

            logger.info(
                f"[{trace_id}] Generating section {section_index + 1}/{total_sections}: "
                f"{section_name}"
            )

            # Generate section content
            section_content = await self._generate_single_section(
                request,
                dynamic_section,
                previous_section_titles,
                section_index,
                total_sections,
                trace_id
            )

            # Check if section was skipped (no observations)
            if section_content is None:
                logger.warning(
                    f"[{trace_id}] Skipping section '{section_name}' - "
                    "no observations with content"
                )
                continue

            # Count LLM call
            self._count_llm_call()

            # Create ReportSection object
            section_obj = ReportSection(
                title=section_name,
                content=section_content,
                level=2,  # All sections are ##
                tables_count=self._count_tables(section_content),
                observations_used=0,  # Updated by filter (TODO: track this)
                confidence=0.8  # Medium quality
            )

            section_dict[section_name] = section_content
            section_objects.append(section_obj)
            previous_section_titles.append(section_name)

        logger.info(
            f"[{trace_id}] Generated {len(section_dict)}/{total_sections} sections "
            f"({total_sections - len(section_dict)} skipped)"
        )

        return section_dict, section_objects

    async def _generate_single_section(
        self,
        request: ReportGenerationRequest,
        dynamic_section: DynamicSection,
        previous_section_titles: List[str],
        section_index: int,
        total_sections: int,
        trace_id: str
    ) -> Optional[str]:
        """
        Generate content for a single section.

        Process:
        1. Filter observations for this section
        2. Build context (observations, citations, previous sections)
        3. Generate content with LLM
        4. Validate and clean

        Args:
            request: Generation request
            dynamic_section: Section definition from planner
            previous_section_titles: List of already-generated section names
            section_index: Current section index (0-based)
            total_sections: Total number of sections
            trace_id: Trace ID for logging

        Returns:
            Section content markdown, or None if section should be skipped
        """
        section_name = dynamic_section.title

        # Stage 1: Filter observations for this section
        state_dict = self._request_to_state_dict(request)
        filtered_observations = self.observation_filter.filter_for_section(
            section_name=section_name,
            all_observations=state_dict['observations'],
            state=state_dict
        )

        # None means no observations with content - skip section
        if filtered_observations is None:
            return None

        # Empty list means fallback observations - continue
        if not filtered_observations:
            logger.warning(
                f"[{trace_id}] Section '{section_name}' has no filtered observations, "
                "using fallback"
            )

        logger.info(
            f"[{trace_id}] Section '{section_name}' using {len(filtered_observations)} "
            "observations"
        )

        # Stage 2: Build context for LLM
        context = self._build_section_context(
            request,
            dynamic_section,
            filtered_observations,
            previous_section_titles
        )

        # Stage 3: Generate content with LLM
        section_content = await self._generate_section_with_llm(
            section_name,
            dynamic_section,
            context,
            section_index,
            total_sections,
            trace_id
        )

        # Stage 4: Clean and validate
        section_content = self._clean_section_content(section_content, section_name)

        return section_content

    def _build_section_context(
        self,
        request: ReportGenerationRequest,
        dynamic_section: DynamicSection,
        filtered_observations: List[Any],
        previous_section_titles: List[str]
    ) -> Dict[str, Any]:
        """
        Build all context needed for section generation.

        Args:
            request: Generation request
            dynamic_section: Section definition
            filtered_observations: Observations for this section
            previous_section_titles: Previously generated section names

        Returns:
            Context dictionary
        """
        # Build observation context
        from .template_generator import TemplateGenerator
        obs_context, truncated = TemplateGenerator.build_observation_context(
            filtered_observations,
            limit=self.config.max_observations_per_section
        )

        # Build citation context
        citation_context = TemplateGenerator.build_citation_context(request.citations)

        # Build previous sections context (for continuation awareness)
        previous_context = self._build_previous_sections_context(previous_section_titles)

        # Extract entities from constraints
        entities = []
        if request.query_constraints and hasattr(request.query_constraints, 'entities'):
            entities = request.query_constraints.entities

        return {
            'section_name': dynamic_section.title,
            'section_description': dynamic_section.purpose or "",
            'section_focus': getattr(dynamic_section, 'focus', ''),
            'observations': obs_context,
            'observations_count': len(filtered_observations),
            'observations_truncated': truncated,
            'citations': citation_context,
            'entities': entities,
            'topic': request.research_topic,
            'style': request.report_style,
            'previous_sections': previous_context,
            'previous_section_count': len(previous_section_titles)
        }

    def _build_previous_sections_context(
        self,
        previous_section_titles: List[str]
    ) -> str:
        """
        Build context string for continuation awareness.

        Args:
            previous_section_titles: List of previously generated section names

        Returns:
            Context string describing previous sections
        """
        if not previous_section_titles:
            return "This is the first section."

        if len(previous_section_titles) == 1:
            return f"Previously covered: {previous_section_titles[0]}"

        return (
            f"Previously covered sections: {', '.join(previous_section_titles)}. "
            "Build on this context without repeating content."
        )

    async def _generate_section_with_llm(
        self,
        section_name: str,
        dynamic_section: DynamicSection,
        context: Dict[str, Any],
        section_index: int,
        total_sections: int,
        trace_id: str
    ) -> str:
        """
        Generate section content using LLM.

        Args:
            section_name: Section title
            dynamic_section: Section definition
            context: Built context dict
            section_index: Current section index
            total_sections: Total sections
            trace_id: Trace ID for logging

        Returns:
            Generated section content (markdown)
        """
        # Build section-specific instructions
        instructions = [
            f"Generate comprehensive content for the '{section_name}' section",
            "Use all available observations and data",
            "Create tables for comparisons or quantitative data",
            "Ensure all tables have proper markdown syntax (| --- | separator)",
            "Cite sources where appropriate",
            "Be specific and data-driven",
        ]

        if context['section_description']:
            instructions.append(f"Section focus: {context['section_description']}")

        if context['entities']:
            instructions.append(
                f"Cover these entities: {', '.join(context['entities'])}"
            )

        if context['previous_sections']:
            instructions.append(
                f"Continuation: {context['previous_sections']}. "
                "Do not repeat content from previous sections."
            )

        instruction_text = "\n".join(f"- {inst}" for inst in instructions)

        # Build prompt
        prompt = f"""# Task: Generate Report Section

## Section: {section_name}
**Section {section_index + 1} of {total_sections}**

{context['section_description'] if context['section_description'] else ''}

## Context
**Research Topic**: {context['topic']}
**Report Style**: {context['style']}

{context['previous_sections']}

## Instructions
{instruction_text}

## Research Observations
{context['observations']}

## Citations Available
{context['citations']}

---

**Generate the complete content for the '{section_name}' section. Use proper markdown formatting, create tables where appropriate, and cite sources.**
"""

        messages = [
            SystemMessage(
                content=(
                    "You are a research report writer. Generate comprehensive, well-structured "
                    "section content with proper markdown formatting. Use tables to present "
                    "comparative data. Cite sources. Be specific and data-driven. "
                    "Ensure all tables have proper markdown syntax with separator rows (| --- |)."
                )
            ),
            HumanMessage(content=prompt)
        ]

        # Invoke LLM
        logger.debug(f"[{trace_id}] Invoking LLM for section: {section_name}")

        content = self.llm_manager.invoke_with_smart_retry(
            messages,
            section_name=f"section_{section_index}",
            state=self._request_to_state_dict_with_context(context)
        )

        logger.debug(
            f"[{trace_id}] Section '{section_name}' generated: {len(content)} chars"
        )

        return content

    def _clean_section_content(self, content: str, section_name: str) -> str:
        """
        Clean and validate section content.

        Args:
            content: Raw section content from LLM
            section_name: Section name for logging

        Returns:
            Cleaned content
        """
        # Remove redundant section heading if LLM added it
        # (we'll add proper heading in _combine_sections)
        lines = content.split('\n')
        if lines and lines[0].strip().startswith('#'):
            # Check if first line is heading for this section
            first_line = lines[0].strip().lstrip('#').strip()
            if first_line == section_name:
                # Remove redundant heading
                content = '\n'.join(lines[1:]).strip()

        return content

    def _combine_sections(
        self,
        section_dict: Dict[str, str],
        research_topic: str
    ) -> str:
        """
        Combine individual sections into final report.

        Args:
            section_dict: Dictionary of {section_name: content}
            research_topic: Research topic for title

        Returns:
            Complete report markdown
        """
        parts = [f"# {research_topic}", ""]

        for section_name, content in section_dict.items():
            # Add section heading
            parts.append(f"## {section_name}")
            parts.append("")

            # Add section content
            parts.append(content.strip())
            parts.append("")

        return "\n".join(parts)

    # === Helper Methods ===

    def _emit_section_metrics(
        self,
        result: ReportGenerationResult,
        sections_planned: int
    ) -> None:
        """
        Emit section-specific metrics.

        Args:
            result: Generation result
            sections_planned: Number of sections planned (before skipping)
        """
        metrics = {
            'generation_mode': 'section_by_section',
            'sections_planned': sections_planned,
            'sections_generated': result.total_sections,
            'sections_skipped': sections_planned - result.total_sections,
            'observations_used': result.observations_used,
            'tables_generated': result.total_tables,
            'generation_time_ms': result.generation_time_ms,
            'llm_calls': result.llm_calls,
        }

        self._emit_event('section_by_section_metrics', metrics)

    def _detect_table_mode(self, text: str) -> TableGenerationMode:
        """Detect how tables were generated."""
        if '|' in text and '---' in text:
            return TableGenerationMode.LLM_MARKDOWN
        return TableGenerationMode.NONE

    def _request_to_state_dict(self, request: ReportGenerationRequest) -> Dict[str, Any]:
        """
        Convert request to state dict for observation filter.

        Args:
            request: Generation request

        Returns:
            State dict compatible with ObservationFilter
        """
        # Build minimal state dict for ObservationFilter
        state = {
            'research_topic': request.research_topic,
            'observations': [
                obs.to_dict() if hasattr(obs, 'to_dict') else obs
                for obs in request.observations
            ],
            'citations': request.citations,
        }

        # Add plan with dynamic_sections
        if request.dynamic_sections:
            # Create a minimal plan object with dynamic_sections
            class MinimalPlan:
                def __init__(self, sections):
                    self.dynamic_sections = sections

            state['current_plan'] = MinimalPlan(request.dynamic_sections)

        return state

    def _request_to_state_dict_with_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert context to state dict for LLM invocation.

        Args:
            context: Section context dict

        Returns:
            State dict for LLM manager
        """
        return {
            'research_topic': context['topic'],
            'observations': [],  # Not needed for LLM invocation
            'citations': [],
            'section_name': context['section_name']
        }
