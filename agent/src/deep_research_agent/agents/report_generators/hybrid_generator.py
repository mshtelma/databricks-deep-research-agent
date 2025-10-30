"""
Hybrid multi-pass report generator (Tier 1).

Most sophisticated strategy using calculation results for deterministic tables.
Highest quality output when calculation data available.

Features:
- Calculation-based table generation
- Structured pipeline for programmatic tables
- Deterministic, auditable outputs
- Formula transparency
- Fallback to section-by-section when calc unavailable

This is the RECOMMENDED strategy for metric-heavy research queries.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from ...core import get_logger
from ...core.observation_models import StructuredObservation
from ...core.advanced_utilities import ReporterLLMInvocationManager
from ...core.adaptive_structure_validator import AdaptiveStructureValidator
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


class HybridGenerator(BaseReportGenerator):
    """
    Tier 1: Hybrid multi-pass report generation.

    Generates reports using calculation results for deterministic tables.
    Three-phase process:
    1. Extract calculation context (from CalculationAgent)
    2. Generate structured report with table specifications
    3. Programmatically generate tables from calculations

    Used when:
    - Calculation results available (from CalculationAgent)
    - Metric-heavy queries (tax, financial, etc.)
    - Deterministic, auditable outputs required

    Quality: HIGH (calculation-based, deterministic tables)
    Speed: MEDIUM (3-phase process with calculations)
    Reliability: MEDIUM (requires calculation results)
    """

    # Strategy metadata
    name = "hybrid"
    generation_mode = GenerationMode.HYBRID
    quality_level = ReportQuality.HIGH

    def __init__(
        self,
        llm: Any,
        config: ReporterConfig,
        utilities: ReporterUtilities,
        **kwargs
    ):
        """
        Initialize hybrid generator.

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

    def can_handle(
        self,
        request: ReportGenerationRequest
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if hybrid strategy can handle request.

        Hybrid requires calculation results from CalculationAgent.

        Args:
            request: Generation request

        Returns:
            Tuple of (can_handle, reason_if_not)
        """
        if not request.observations:
            return False, "No observations available"

        # Check for calculation context (required for hybrid)
        if not request.has_calculation_data():
            return False, "No calculation results available (CalculationAgent required)"

        # Hybrid can handle if we have calc data
        return True, None

    async def generate(
        self,
        request: ReportGenerationRequest,
        config: ReporterConfig
    ) -> ReportGenerationResult:
        """
        Generate report using hybrid strategy.

        Three-phase process:
        1. Validate calculation context
        2. Generate structured report using pipeline
        3. Post-process with citations and metadata

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
                # Phase 1: Validate hybrid configuration
                self._emit_progress("Validating hybrid configuration", 0.1)
                self._validate_hybrid_config(config)

                # Phase 2: Get calculation context
                self._emit_progress("Loading calculation results", 0.2)
                calc_context = request.calculation_context

                if calc_context is None:
                    raise GenerationError(
                        "Calculation context unavailable",
                        strategy=self.name,
                        recoverable=True,
                        fallback_suggestion="section_by_section"
                    )

                # Log calculation context stats
                logger.info(
                    f"[{trace_id}] Calculation context loaded: "
                    f"data_points={len(getattr(calc_context, 'extracted_data', []))}, "
                    f"calculations={len(getattr(calc_context, 'calculations', []))}, "
                    f"comparisons={len(getattr(calc_context, 'key_comparisons', []))}"
                )

                # Phase 3: Use structured pipeline for report generation
                self._emit_progress("Generating structured report", 0.4)
                report_text = await self._generate_with_structured_pipeline(
                    request,
                    calc_context,
                    trace_id
                )
                self._count_llm_call()

                # Phase 4: Fix markdown tables (safety net)
                self._emit_progress("Validating tables", 0.7)
                report_text = self.table_processor.fix_markdown_tables(report_text)

                # Phase 5: Add citations and metadata
                self._emit_progress("Adding citations and metadata", 0.9)
                final_report = self.utilities.add_both(
                    report_text,
                    request.citations,
                    self._request_to_state_dict(request, calc_context),
                    config.default_style
                )

                # Parse sections from report
                sections = self._parse_sections(final_report)

                # Build result
                generation_time = max(1, int((time.time() - start_time) * 1000))

                result = ReportGenerationResult(
                    final_report=final_report,
                    sections=sections,
                    generation_mode=GenerationMode.HYBRID,
                    table_mode=self._detect_table_mode(report_text),
                    quality=ReportQuality.HIGH,
                    total_sections=len(sections),
                    total_tables=self._count_tables(report_text),
                    total_citations=len(request.citations),
                    observations_used=len(request.observations),
                    calculations_used=len(getattr(calc_context, 'calculations', [])),
                    generation_time_ms=generation_time,
                    llm_calls=self._llm_call_count,
                    request_id=request.request_id
                )

                # Emit hybrid-specific metrics
                self._emit_hybrid_metrics(result, calc_context)

                self._emit_progress("Hybrid generation complete", 1.0)
                return result

            except GenerationError:
                # Re-raise GenerationError as-is
                raise

            except Exception as e:
                logger.error(f"Hybrid generation failed: {e}", exc_info=True)
                raise GenerationError(
                    f"Hybrid generation failed: {str(e)}",
                    strategy=self.name,
                    recoverable=True,
                    fallback_suggestion="section_by_section",
                    original_error=e
                )

    # === Phase-Specific Methods ===

    async def _generate_with_structured_pipeline(
        self,
        request: ReportGenerationRequest,
        calc_context: Any,
        trace_id: str
    ) -> str:
        """
        Use structured pipeline for programmatic table generation.

        This is the RECOMMENDED path (vs legacy anchor-based generation).

        Args:
            request: Generation request
            calc_context: Calculation context from CalculationAgent
            trace_id: Trace ID for logging

        Returns:
            Report text with programmatically generated tables
        """
        try:
            # Import pipeline (lazy import to avoid circular dependencies)
            from ...agents.reporter_pipeline import StructuredReportPipeline

            logger.info(f"[{trace_id}] Using structured pipeline for table generation")

            # Initialize pipeline
            pipeline = StructuredReportPipeline(self.llm, self.config.model_dump())

            # Build request dict for pipeline (it expects dict, not Pydantic)
            pipeline_request = {
                'research_topic': request.research_topic,
                'observations': request.observations,
                'dynamic_sections': request.dynamic_sections or [],
                'calculation_context': calc_context,
                'citations': request.citations,
            }

            # Generate report using pipeline
            report = await pipeline.generate(pipeline_request)

            logger.info(f"[{trace_id}] Structured pipeline complete: {len(report)} chars")
            return report

        except ImportError as e:
            logger.warning(f"StructuredReportPipeline not available: {e}")
            # Fall back to simple generation
            return await self._generate_simple_hybrid_report(request, calc_context, trace_id)

        except Exception as e:
            logger.error(f"Structured pipeline failed: {e}", exc_info=True)
            # Fall back to simple generation
            return await self._generate_simple_hybrid_report(request, calc_context, trace_id)

    async def _generate_simple_hybrid_report(
        self,
        request: ReportGenerationRequest,
        calc_context: Any,
        trace_id: str
    ) -> str:
        """
        Fallback: Simple hybrid report without structured pipeline.

        Generates narrative report mentioning calculations but without
        programmatic table generation.

        Args:
            request: Generation request
            calc_context: Calculation context
            trace_id: Trace ID for logging

        Returns:
            Simple hybrid report
        """
        logger.info(f"[{trace_id}] Using simple hybrid generation (fallback)")

        # Build context from calculations
        calc_summary = self._build_calculation_summary(calc_context)
        obs_context, _ = self._build_observation_context(request.observations, limit=50)

        # Build prompt
        prompt = f"""# Task: Generate Comprehensive Research Report

## Research Topic
{request.research_topic}

## Research Observations
{obs_context}

## Calculation Results
{calc_summary}

## Instructions
1. Create a comprehensive report using the observations and calculations
2. Include calculation results as tables where appropriate
3. Use proper markdown formatting (headings, bullets, tables)
4. Ensure all tables have proper separator rows (| --- |)
5. Cite data sources in your analysis

**Generate the complete report with proper structure and tables.**
"""

        messages = [
            SystemMessage(
                content=(
                    "You are a research report writer. Generate comprehensive reports "
                    "with proper markdown formatting. Use tables to present calculation results. "
                    "Ensure all tables have proper markdown syntax with separator rows."
                )
            ),
            HumanMessage(content=prompt)
        ]

        # Invoke LLM
        report = self.llm_manager.invoke_with_smart_retry(
            messages,
            section_name="hybrid_simple_generation",
            state=self._request_to_state_dict(request, calc_context)
        )

        logger.info(f"[{trace_id}] Simple hybrid generation complete: {len(report)} chars")
        return report

    # === Helper Methods ===

    @staticmethod
    def _build_observation_context(
        observations: List[Any],
        limit: int = 50
    ) -> Tuple[str, int]:
        """Build observation context (reuse from TemplateGenerator)."""
        from .template_generator import TemplateGenerator
        return TemplateGenerator.build_observation_context(observations, limit)

    def _build_calculation_summary(self, calc_context: Any) -> str:
        """
        Build summary of calculation results for prompt.

        Args:
            calc_context: Calculation context with data/calculations

        Returns:
            Formatted calculation summary
        """
        if not calc_context:
            return "No calculation results available."

        lines = []

        # Extracted data points
        extracted_data = getattr(calc_context, 'extracted_data', [])
        if extracted_data:
            lines.append(f"### Data Points Extracted: {len(extracted_data)}")
            for dp in extracted_data[:10]:  # Show first 10
                entity = getattr(dp, 'entity', '?')
                metric = getattr(dp, 'metric', '?')
                value = getattr(dp, 'value', '?')
                lines.append(f"- {entity}: {metric} = {value}")
            if len(extracted_data) > 10:
                lines.append(f"- ... {len(extracted_data) - 10} more data points")

        # Calculations performed
        calculations = getattr(calc_context, 'calculations', [])
        if calculations:
            lines.append(f"\\n### Calculations Performed: {len(calculations)}")
            for calc in calculations[:5]:  # Show first 5
                if hasattr(calc, 'to_dict'):
                    calc_dict = calc.to_dict()
                    lines.append(f"- {calc_dict.get('description', 'Calculation')}")
                else:
                    lines.append(f"- {str(calc)[:80]}")
            if len(calculations) > 5:
                lines.append(f"- ... {len(calculations) - 5} more calculations")

        # Key comparisons
        comparisons = getattr(calc_context, 'key_comparisons', [])
        if comparisons:
            lines.append(f"\\n### Key Comparisons: {len(comparisons)}")
            for comp in comparisons[:5]:  # Show first 5
                if hasattr(comp, 'to_dict'):
                    comp_dict = comp.to_dict()
                    lines.append(f"- {comp_dict.get('description', 'Comparison')}")
                else:
                    lines.append(f"- {str(comp)[:80]}")

        return "\\n".join(lines) if lines else "No calculation details available."

    def _validate_hybrid_config(self, config: ReporterConfig) -> None:
        """
        Validate hybrid-specific configuration.

        Args:
            config: Reporter configuration

        Raises:
            ValueError: If config invalid
        """
        # Check if use_structured_pipeline is enabled
        if not config.use_structured_pipeline:
            logger.warning(
                "use_structured_pipeline=False: Hybrid mode works best with structured pipeline"
            )

        # Check calc_selector_top_k
        if config.calc_selector_top_k < 20:
            logger.warning(
                f"calc_selector_top_k={config.calc_selector_top_k}: "
                "Low value may miss important observations"
            )

    def _emit_hybrid_metrics(
        self,
        result: ReportGenerationResult,
        calc_context: Any
    ) -> None:
        """
        Emit hybrid-specific metrics.

        Args:
            result: Generation result
            calc_context: Calculation context
        """
        metrics = {
            'generation_mode': 'hybrid_multipass',
            'observations_used': result.observations_used,
            'calculations_performed': len(getattr(calc_context, 'calculations', [])),
            'data_points_extracted': len(getattr(calc_context, 'extracted_data', [])),
            'key_comparisons': len(getattr(calc_context, 'key_comparisons', [])),
            'tables_generated': result.total_tables,
            'generation_time_ms': result.generation_time_ms,
            'llm_calls': result.llm_calls,
        }

        self._emit_event('hybrid_metrics', metrics)

    def _detect_table_mode(self, text: str) -> TableGenerationMode:
        """Detect how tables were generated."""
        if '|' in text and '---' in text:
            # Check if tables have formulas or calculations
            if 'Formula:' in text or 'Calculation:' in text:
                return TableGenerationMode.CALCULATION_BASED
            return TableGenerationMode.LLM_MARKDOWN
        return TableGenerationMode.NONE

    def _request_to_state_dict(
        self,
        request: ReportGenerationRequest,
        calc_context: Any = None
    ) -> Dict[str, Any]:
        """
        Convert request to state dict for backward compatibility.

        Args:
            request: Generation request
            calc_context: Optional calculation context

        Returns:
            State dict for LLM invocation
        """
        state = {
            'research_topic': request.research_topic,
            'observations': [
                obs.to_dict() if hasattr(obs, 'to_dict') else obs
                for obs in request.observations
            ],
            'citations': request.citations,
            'factuality_score': request.factuality_score,
            'confidence_scores': request.confidence_scores
        }

        if calc_context:
            state['calculation_results'] = calc_context
            state['metric_capability_enabled'] = True

        return state
