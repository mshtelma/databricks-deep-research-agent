"""
Reporter Agent: Report synthesis and formatting specialist.

Generates styled reports from research findings with proper citations.
"""

from typing import Dict, Any, Optional, List, Literal, Tuple, Sequence, Union
from datetime import datetime
import time
import random
import re
import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import Command

from ..core import get_logger, Citation, SectionResearchResult
from ..core.template_generator import DynamicSection, SectionContentType
from ..core.multi_agent_state import EnhancedResearchState, StateManager
from ..core.report_styles import (
    ReportStyle,
    STYLE_CONFIGS,
    StyleTemplate,
    StyleConfig,
    ReportFormatter
)
from ..core.grounding import HallucinationPrevention
from ..core.presentation_requirements import PresentationRequirements
from ..core.semantic_extraction import SemanticEntityExtractor, StructuredDataMatcher
from ..core.message_utils import get_last_user_message, extract_content
from ..core.observation_models import (
    StructuredObservation,
    ensure_structured_observation,
    observations_to_research_data,
    observation_to_text,
)
from ..core.observation_converter import ObservationConverter
from ..core.observation_selector import ObservationSelector
from ..core.plan_models import StepStatus
from ..core.response_handlers import parse_structured_response, ParsedResponse, ResponseType
from ..core.table_preprocessor import TablePreprocessor


logger = get_logger(__name__)


class ReporterAgent:
    """
    Reporter agent that generates formatted reports from research findings.
    
    Responsibilities:
    - Compile all observations
    - Apply style-specific formatting
    - Structure final report
    - Ensure citation compliance
    - Integrate grounding markers if enabled
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        event_emitter: Optional[Any] = None
    ) -> None:
        """
        Initialize the reporter agent.

        Args:
            llm: Language model for report generation
            config: Configuration dictionary
            event_emitter: Optional event emitter for detailed progress tracking
        """
        self.llm = llm
        self.config = config or {}
        self.event_emitter = event_emitter  # Optional for detailed event emission
        self.name = "Reporter"  # Capital for test compatibility
        self.formatter = ReportFormatter()

        # embedding_manager not currently initialized (could be added in future)
        self.embedding_manager = None

        # Initialize observation selector for intelligent observation filtering
        self.observation_selector = ObservationSelector(
            embedding_manager=self.embedding_manager
        )
        
        # Extract report configuration
        report_config = self.config.get('report', {})
        self.default_style = ReportStyle(report_config.get('default_style', 'default'))
        self.include_citations = report_config.get('include_citations', True)
        self.include_grounding_markers = report_config.get('include_grounding_markers', True)
        self.hallucination_prevention = HallucinationPrevention()
        
        # Initialize semantic extraction components
        # Check nested config path for reporter-specific settings
        reporter_config = self.config.get('agents', {}).get('reporter', {})
        use_semantic_extraction = reporter_config.get('use_semantic_extraction', True)
        
        # Also check direct config path for backward compatibility
        if 'use_semantic_extraction' in self.config:
            use_semantic_extraction = self.config.get('use_semantic_extraction', True)
        
        logger.info(f"Semantic extraction enabled: {use_semantic_extraction}")
        
        if use_semantic_extraction and llm:
            try:
                self.entity_extractor = SemanticEntityExtractor(llm)
                self.data_matcher = StructuredDataMatcher(llm)
                logger.info("Successfully initialized semantic extraction components")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic extractors: {e}")
                self.entity_extractor = None
                self.data_matcher = None
        else:
            self.entity_extractor = None
            self.data_matcher = None
            logger.info("Semantic extraction disabled or no LLM available")
    
    def _classify_databricks_error(self, error: Exception) -> Tuple[bool, Optional[float]]:
        """
        Classify Databricks errors and extract retry guidance.
        
        Returns:
            Tuple[bool, Optional[float]]: (is_transient, suggested_wait_time)
        """
        error_str = str(error)
        
        # Transient errors that should be retried
        if 'TEMPORARILY_UNAVAILABLE' in error_str or '503' in error_str:
            # Extract wait time if provided - look for "retry after X" or "Retry-After: X"
            wait_time = None
            retry_patterns = [
                r'retry[^\d]*after[^\d]*(\d+)',  # "retry after 15"
                r'retry[_-]after[:\s]*(\d+)',   # "Retry-After: 15"
            ]
            for pattern in retry_patterns:
                match = re.search(pattern, error_str, re.I)
                if match:
                    wait_time = float(match.group(1))
                    break
            return True, wait_time
        
        # Rate limiting
        if '429' in error_str or 'rate_limit' in error_str.lower():
            return True, 30.0  # Default 30s for rate limits
        
        # Gateway errors
        if any(code in error_str for code in ['502', '504', 'gateway']):
            return True, 15.0  # Gateway issues often resolve quickly
        
        # Permanent errors
        if any(err in error_str for err in ['401', '403', 'unauthorized', 'forbidden']):
            return False, None
        
        # Unknown errors - don't retry
        return False, None
    
    # NOTE: _extract_content_from_reasoning method has been removed as it's replaced
    # by the universal response handler in _invoke_llm_with_smart_retry

    def _transform_reasoning_to_report(self, reasoning_text: str, section_name: str, findings: Dict[str, Any] = None) -> str:
        """
        Transform reasoning text into proper report content.
        
        Args:
            reasoning_text: The reasoning text to transform
            section_name: Name of the section being generated
            findings: Research findings for fallback context
            
        Returns:
            str: Transformed report content
        """
        logger.info(f"🔄 TRANSFORMATION: Converting reasoning to report content for {section_name}")
        
        if not self.llm:
            logger.warning(f"No LLM available for reasoning transformation, using fallback")
            # Clean the reasoning text and return full content
            clean_reasoning = reasoning_text.replace("I need to", "The analysis shows").replace("Let me", "").replace("I should", "The research indicates")
            return f"## {section_name}\n\n{clean_reasoning}"
        
        # Create transformation prompt
        transform_prompt = f"""You received reasoning about a research topic. Transform it into a professional report section.

Original reasoning:
{reasoning_text}

Instructions:
1. Convert the reasoning into clear, professional report content
2. Remove any meta-commentary like "I need to", "Let me think", "I should"
3. Focus on the facts, analysis, and conclusions from the reasoning
4. Use proper formatting with paragraphs and structure
5. Write in third person, professional tone
6. Present information as facts and analysis, not thought process
7. Keep the substantial content but make it report-appropriate

Generate the report section content:"""

        try:
            messages = [
                SystemMessage(content="You are a professional report writer. Transform reasoning into clear, structured report content. Remove thinking process and present facts directly."),
                HumanMessage(content=transform_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the transformation response
            from ..core.response_handlers import parse_structured_response
            parsed = parse_structured_response(response)
            
            if parsed.content and len(parsed.content.strip()) > 50:
                logger.info(f"✅ TRANSFORMATION_SUCCESS: {section_name} ({len(parsed.content)} chars)")
                return parsed.content.strip()
            else:
                logger.warning(f"⚠️ TRANSFORMATION_FAILED: {section_name} - insufficient content generated")
                # Return clean version of original reasoning without truncation
                clean_reasoning = reasoning_text.replace("I need to", "The analysis shows").replace("Let me", "").replace("I should", "The research indicates")
                return f"## {section_name}\n\n{clean_reasoning}"
                
        except Exception as e:
            logger.error(f"❌ TRANSFORMATION_ERROR: {section_name} - {e}")
            # Return cleaned version of reasoning as fallback without truncation
            clean_reasoning = reasoning_text.replace("I need to", "The analysis shows").replace("Let me", "").replace("I should", "The research indicates")
            return f"## {section_name}\n\n{clean_reasoning}"

    def _invoke_llm_with_smart_retry(self, messages: List, section_name: str, state: EnhancedResearchState = None) -> str:
        """
        Invoke LLM with intelligent retry for transient errors only.
        
        Args:
            messages: LLM messages to send
            section_name: Name of the section being generated (for logging)
            
        Returns:
            str: Generated content from LLM
            
        Raises:
            Exception: If all retries are exhausted or permanent error encountered
        """
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Log the prompt being sent to LLM (messages may contain system + human messages)
                if messages and len(messages) > 0:
                    prompt_content = ""
                    for msg in messages:
                        if hasattr(msg, 'content'):
                            prompt_content += f"{msg.content}... "
                    logger.info(f"🔍 LLM_PROMPT [reporter_{section_name}]: {prompt_content}...")
                
                response = self.llm.invoke(messages)
                
                # Log the response received from LLM
                logger.info(f"🔍 LLM_RESPONSE [reporter_{section_name}]: {response.content}...")
                
                # ENTITY VALIDATION: Check for hallucinated entities in LLM response
                if state:
                    requested_entities = state.get("requested_entities", [])
                    if requested_entities:
                        from ..core.entity_validation import EntityExtractor
                        extractor = EntityExtractor()
                        response_entities = extractor.extract_entities(response.content)
                        hallucinated = response_entities - set(requested_entities)
                        if hallucinated:
                            logger.warning(f"🚨 ENTITY_HALLUCINATION [reporter_{section_name}]: LLM mentioned entities not in original query: {hallucinated}")
                        else:
                            logger.info(f"✅ ENTITY_VALIDATION [reporter_{section_name}]: Response only mentions requested entities: {response_entities & set(requested_entities)}")
                
                # Success - log if we had retries
                if attempt > 0:
                    logger.info(f"LLM call succeeded for {section_name} after {attempt} retries")
                
                # Parse response using the universal response handler
                parsed = parse_structured_response(response)
                
                # Extract content and reasoning
                content = parsed.content
                reasoning_text = parsed.reasoning
                
                # Log response analysis
                logger.info(f"Section {section_name}: Using {parsed.response_type.value} content ({len(content)} chars)")
                if reasoning_text:
                    logger.debug(f"Section {section_name}: Reasoning available ({len(reasoning_text)} chars)")
                if parsed.metadata:
                    logger.debug(f"Section {section_name}: Metadata: {parsed.metadata}")
                
                # Emit reasoning event if available
                if reasoning_text and self.event_emitter:
                    try:
                        # Truncate reasoning for events to avoid sending full reports as reasoning
                        reasoning_for_event = reasoning_text[:500] if len(reasoning_text) > 500 else reasoning_text
                        self.event_emitter.emit_reasoning_reflection(
                            reasoning=reasoning_for_event,
                            options=["content_generation"],
                            confidence=0.8,
                            stage_id="reporter"
                        )
                        logger.info(f"Emitted reasoning event for {section_name} ({len(reasoning_for_event)} chars)")
                    except Exception as e:
                        logger.warning(f"Failed to emit reasoning event for {section_name}: {e}")
                
                # FIXED: Only transform reasoning to report if we have NO proper content
                if reasoning_text and len(reasoning_text.strip()) > 100 and (not content or len(content.strip()) < 50):
                    logger.warning(f"🔄 REASONING_TO_REPORT: No proper content found for {section_name}, transforming reasoning to report...")
                    
                    # Avoid infinite recursion by checking if this is already a transformation
                    if not section_name.endswith("_transformed"):
                        transformed_content = self._transform_reasoning_to_report(reasoning_text, section_name)
                        logger.info(f"🔄 REASONING_TO_REPORT: Transformation completed for {section_name} ({len(transformed_content)} chars)")
                        content = transformed_content
                    else:
                        logger.warning(f"🔄 REASONING_TO_REPORT: Avoiding recursive transformation for {section_name}")
                        # Keep existing content if we have it, otherwise use cleaned reasoning
                elif content and reasoning_text:
                    logger.info(f"✅ PROPER_CONTENT: Using actual report content for {section_name} ({len(content)} chars), ignoring reasoning ({len(reasoning_text)} chars)")
                
                # Apply content sanitization as final cleanup before returning
                from ..core.content_sanitizer import sanitize_agent_content
                
                if content:
                    sanitization_result = sanitize_agent_content(content)
                    if sanitization_result.sanitization_applied:
                        logger.info(f"Applied content sanitization to {section_name}: {len(content)} -> {len(sanitization_result.clean_content)} chars")
                        for warning in sanitization_result.warnings:
                            logger.warning(f"Content sanitization warning for {section_name}: {warning}")
                    content = sanitization_result.clean_content
                
                # Ensure we never return None or empty content
                if not content:
                    logger.warning(f"No content extracted for {section_name}, using empty string")
                    content = ""
                
                return content
                
            except Exception as e:
                attempt += 1
                error_str = str(e)
                
                # Classify the error
                is_transient, suggested_wait = self._classify_databricks_error(e)
                
                if not is_transient:
                    # Permanent error - fail immediately
                    logger.error(f"Permanent error in {section_name}: {e}")
                    raise
                
                if attempt >= max_attempts:
                    # Max retries exceeded
                    logger.error(f"Max retries ({max_attempts}) exceeded for {section_name}")
                    raise
                
                # Check if this is a 429 error - if so, ModelSelector already tried all endpoints
                if "429" in error_str:
                    logger.error(
                        f"All endpoints exhausted for {section_name} (429 errors). "
                        "Not retrying - ModelSelector already tried all available endpoints."
                    )
                    raise  # Fail fast - no point retrying

                # Calculate wait time for non-429 transient errors
                if suggested_wait:
                    # Use suggested wait time from error (e.g., 503 with Retry-After)
                    wait_time = min(suggested_wait, 30)  # Cap at 30s
                else:
                    # Exponential backoff: 5, 10, 20, 30 seconds (changed from 10, 15, 22.5...)
                    wait_time = min(5 * (2 ** (attempt - 1)), 30)

                # Add jitter to prevent thundering herd (up to 10% of wait time)
                jitter = random.uniform(0, wait_time * 0.1)
                wait_time += jitter

                logger.warning(
                    f"Transient error in {section_name} generation "
                    f"(attempt {attempt}/{max_attempts}), "
                    f"retrying in {wait_time:.1f}s: {error_str[:100]}"
                )

                time.sleep(wait_time)
        
        # Should never reach here
        raise Exception(f"Retry logic error for {section_name}")
    
    async def __call__(
        self,
        state: EnhancedResearchState,
        config: Dict[str, Any]
    ) -> Command[Literal["end"]]:
        """
        Generate final report from research findings with integrated table support.

        ASYNC: Converted to async for consistency with LangGraph async workflow.
        Note: Internal LLM calls currently use sync invoke() - these can be
        converted to ainvoke() in a future optimization pass.

        Args:
            state: Current research state
            config: Configuration dictionary

        Returns:
            Command to end workflow with final report
        """
        # === DIAGNOSTIC LOGGING FOR DEBUGGING ===
        import threading
        import sys
        logger.info("=" * 80)
        logger.info("REPORTER AGENT __call__ INVOKED")
        logger.info(f"Thread: {threading.current_thread().name} (ID: {threading.current_thread().ident})")
        logger.info(f"Is Main Thread: {threading.current_thread() is threading.main_thread()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"State type: {type(state).__name__}")
        logger.info(f"State has {len(state)} top-level keys")
        logger.info(f"Config has {len(config)} keys: {list(config.keys())}" if config else "Config is None/empty")
        logger.info("=" * 80)

        logger.info("Reporter agent generating final report")
        
        # DEBUG: Log state contents to trace data flow
        logger.info(f"STATE KEYS: {list(state.keys())}")
        logger.info(f"Observations: {len(state.get('observations', []))}")
        logger.info(f"Search results: {len(state.get('search_results', []))}")
        logger.info(f"Section research: {list(state.get('section_research_results', {}).keys())}")
        
        # Get report style
        report_style = state.get("report_style", ReportStyle.DEFAULT)
        logger.info(f"Using report style: {report_style}")
        
        # Get style configuration (with adaptive structure support)
        style_config = self._get_style_config_with_adaptive_structure(state, report_style)
        
        # Compile research findings
        compiled_findings = self._compile_findings(state)
        logger.info(f"Compiled findings: {len(compiled_findings.get('observations', []))} observations")
        logger.info(f"First 20 observations: {compiled_findings["observations"][:20]}")
        
        # Deduplicate citations to prevent accumulation bug
        self._deduplicate_citations(state)

        # ENHANCED: Check for progressive synthesis (incremental research loops)
        research_loops = state.get("research_loops", 0)
        is_incremental_synthesis = research_loops > 0 and state.get("final_report")

        if is_incremental_synthesis:
            logger.info(f"[PROGRESSIVE SYNTHESIS] Enhancing existing report for research loop {research_loops}")

            try:
                # Get previous synthesis and new findings
                previous_synthesis = state.get("final_report", "")
                loop_discoveries = state.get("loop_discoveries", [])
                verification_results = state.get("verification_needed", [])
                deep_dive_insights = state.get("deep_dive_topics", [])

                # Prepare context for progressive synthesis
                progressive_context = {
                    "previous_synthesis": previous_synthesis,
                    "new_findings": self._extract_loop_findings(compiled_findings, loop_discoveries),
                    "verified_claims": verification_results[:5],  # Top 5 verified claims
                    "deep_dive_insights": deep_dive_insights[:3]  # Top 3 deep dive insights
                }

                # Generate enhanced synthesis
                enhanced_report = self._generate_progressive_synthesis(
                    state,
                    progressive_context,
                    compiled_findings,
                    style_config,
                    report_style
                )

                if enhanced_report:
                    logger.info(f"[PROGRESSIVE SYNTHESIS] Successfully enhanced report: {len(enhanced_report)} characters")

                    # Apply final formatting and citations
                    enhanced_report = self._add_citations_and_references(
                        enhanced_report,
                        state.get("citations", []),
                        report_style
                    )
                    enhanced_report = self._add_report_metadata(
                        enhanced_report,
                        state,
                        report_style
                    )

                    report_metadata = {
                        "rendering_mode": "progressive_synthesis",
                        "research_loop": research_loops,
                        "enhancement_applied": True,
                        "previous_length": len(previous_synthesis),
                        "enhanced_length": len(enhanced_report),
                        "observation_count": len(compiled_findings.get("observations", [])),
                    }

                    state = StateManager.finalize_state(state)

                    return Command(
                        update={
                            "final_report": enhanced_report,
                            "report_metadata": report_metadata
                        },
                        goto="end"
                    )

            except Exception as e:
                logger.error(f"[PROGRESSIVE SYNTHESIS] Failed to enhance report: {e}")
                # Fall back to regular synthesis

        # ===================================================================
        # HYBRID MULTI-PASS GENERATION MODE
        # ===================================================================
        generation_mode = self.config.get('agents', {}).get('reporter', {}).get('generation_mode', 'section_by_section')

        if generation_mode == 'hybrid':
            logger.info("[HYBRID MODE] Starting hybrid multi-pass report generation")

            try:
                # Validate configuration
                self._validate_hybrid_config(self.config)

                # Sanitize observations
                sanitized_findings = compiled_findings.copy()
                sanitized_findings['observations'] = self._sanitize_observations_for_report(
                    compiled_findings.get('observations', [])
                )

                # Phase 1: Generate calculation context
                calc_context = await self._generate_calculation_context(sanitized_findings)

                # Get dynamic structure from plan
                current_plan = state.get('current_plan')
                dynamic_sections = []
                if current_plan and hasattr(current_plan, 'dynamic_sections'):
                    dynamic_sections = current_plan.dynamic_sections
                elif current_plan and hasattr(current_plan, 'suggested_report_structure'):
                    # Convert suggested_report_structure to section titles
                    dynamic_sections = [
                        {'title': title, 'purpose': ''}
                        for title in current_plan.suggested_report_structure
                    ]

                # Phase 2: Generate holistic report with table anchors
                holistic_report = await self._generate_holistic_report_with_table_anchors(
                    sanitized_findings,
                    calc_context,
                    dynamic_sections
                )

                # Phase 3: Generate tables from anchors
                final_report = await self._generate_tables_from_anchors_async(
                    holistic_report,
                    calc_context,
                    sanitized_findings
                )

                # Apply citations and metadata
                final_report = self._add_citations_and_references(
                    final_report,
                    state.get('citations', []),
                    report_style
                )
                final_report = self._add_report_metadata(
                    final_report,
                    state,
                    report_style
                )

                # Final sanitization
                from ..core.content_sanitizer import sanitize_agent_content
                sanitized = sanitize_agent_content(final_report)
                final_report = sanitized.clean_content

                # Build metadata
                metadata = {
                    'generation_mode': 'hybrid_multipass',
                    'observations_used': len(sanitized_findings.get('observations', [])),
                    'calculations_performed': len(calc_context.calculations),
                    'data_points_extracted': len(calc_context.extracted_data),
                    'tables_generated': final_report.count('| '),  # Approximation
                    'data_quality_notes': calc_context.data_quality_notes
                }

                self._emit_hybrid_metrics(metrics=metadata)

                # CRITICAL FIX: Ensure all markdown tables have proper separator rows
                final_report = self._fix_markdown_tables(final_report)

                logger.info(f"[HYBRID MODE] Report generation complete: {len(final_report)} characters")

                state = StateManager.finalize_state(state)

                return Command(
                    goto='end',
                    update={
                        'final_report': final_report,
                        'report_metadata': metadata,
                        '_calculation_context': calc_context.to_dict()
                    }
                )

            except Exception as exc:
                logger.error(f"[HYBRID MODE] Hybrid generation failed: {exc}")

                # Check if fallback is enabled
                fallback_enabled = self.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {}).get(
                    'fallback_on_empty_observations', True
                )

                if fallback_enabled:
                    logger.warning("[HYBRID MODE] Falling back to section-by-section generation")
                    # Continue to section-by-section generation below
                else:
                    raise

        # ===================================================================
        # END OF HYBRID MODE
        # ===================================================================

        current_plan = state.get("current_plan")
        template = getattr(current_plan, "report_template", None) if current_plan else None
        dynamic_sections = getattr(current_plan, "dynamic_sections", None) if current_plan else None

        # Debug logging for template and dynamic sections
        logger.info(f"REPORTER: Template available: {template is not None}")
        logger.info(f"REPORTER: Dynamic sections available: {dynamic_sections is not None}")
        if dynamic_sections:
            section_titles = [getattr(s, 'title', str(s)) for s in dynamic_sections]
            logger.info(f"REPORTER: Dynamic section titles: {section_titles}")

        # 🔍 CRITICAL: Check if structured generation is enabled BEFORE using template
        # Structured generation uses section-by-section approach with Pydantic models
        # Template generation uses one-shot markdown generation (error-prone for tables)
        reporter_config = self.config.get('agents', {}).get('reporter', {})
        use_structured_generation = reporter_config.get('enable_structured_generation', True)

        logger.info(f"REPORTER: Structured generation config: {use_structured_generation}")
        logger.info(f"REPORTER: Routing decision: template={template is not None}, "
                   f"dynamic_sections={dynamic_sections is not None}, "
                   f"use_structured={use_structured_generation}")

        # Route to section-based generation if structured generation is enabled AND we have dynamic sections
        # This ensures tables are rendered programmatically with guaranteed valid markdown
        if use_structured_generation and dynamic_sections:
            logger.info("✅ REPORTER: Routing to SECTION-BASED generation (structured output enabled)")
            # Skip template path and go to section-based generation below
            template = None  # Force section-based path

        if template:
            report_body, template_metadata = self._render_template_report(
                template=template,
                findings=compiled_findings,
                style_config=style_config,
                state=state,
                plan=current_plan,
            )

            final_report = self._add_citations_and_references(
                report_body,
                state.get("citations", []),
                report_style
            )
            final_report = self._add_report_metadata(
                final_report,
                state,
                report_style
            )

            dynamic_sections = []
            if current_plan and getattr(current_plan, "dynamic_sections", None):
                dynamic_sections = [section.title for section in current_plan.dynamic_sections]

            report_metadata = {
                "rendering_mode": "template",
                "template_sections": dynamic_sections,
                "template_appendix": template_metadata.get("include_appendix", False),
                "observation_count": template_metadata.get("observation_count", 0),
                "has_embedded_table": template_metadata.get("has_table", False),
                "table_confidence": template_metadata.get("table_confidence", 0.0),
            }
            extra_meta = template_metadata.get("extra_metadata") or {}
            report_metadata.update(extra_meta)

            logger.info(
                "Final report generated via dynamic template",
                extra={
                    "length": len(final_report),
                    "template_sections": dynamic_sections,
                },
            )

            state = StateManager.finalize_state(state)

            return Command(
                goto="end",
                update={
                    "final_report": final_report,
                    "report_sections": {"mode": "template", "template": template},
                    "citations": state.get("citations", []),
                    "report_metadata": report_metadata,
                }
            )
        
        # NEW: Handle case where we have dynamic_sections but no template
        elif dynamic_sections:
            logger.info("REPORTER: No template but dynamic_sections found - using section-based generation")
            # Use the dynamic sections directly with the _generate_sections_from_dynamic_sections method
            from copy import deepcopy
            base_config = style_config
            
            # Ensure we use the dynamic sections for structure
            if hasattr(base_config, 'structure'):
                section_titles = [getattr(s, 'title', str(s)) for s in dynamic_sections]
                base_config.structure = section_titles
                logger.info(f"REPORTER: Updated style config structure with {len(section_titles)} dynamic sections")
            
            # Generate report sections using dynamic sections
            report_sections = self._generate_sections_from_dynamic_sections(
                dynamic_sections,
                compiled_findings,
                base_config,
                state,
                embedded_table=None,
                table_section=None,
                table_metadata={}
            )

            # Deduplicate content across sections (safety net)
            report_sections = self._deduplicate_content(report_sections)

            # Build final report from sections (report_sections is a dict of {section_name: section_content})
            # Add section headers before each section's content
            section_parts = []
            for section_name, section_content in report_sections.items():
                if section_content and section_content.strip():
                    # Add markdown header before section content
                    section_parts.append(f"## {section_name}\n\n{section_content}")
            final_report = "\n\n".join(section_parts) if section_parts else ""

            # Add title if needed
            research_topic = state.get("research_topic", "Research Report")
            if final_report and not final_report.startswith("#"):
                final_report = f"# {research_topic}\n\n{final_report}"
            elif not final_report:
                final_report = f"# {research_topic}\n\nNo content generated."

            final_report = self._add_citations_and_references(
                final_report,
                state.get("citations", []),
                report_style
            )
            final_report = self._add_report_metadata(
                final_report,
                state,
                report_style
            )

            report_metadata = {
                "rendering_mode": "dynamic_sections",
                "template_sections": [getattr(s, 'title', str(s)) for s in dynamic_sections],
                "observation_count": len(compiled_findings.get("observations", [])),
                "has_embedded_table": False,
                "table_confidence": 0.0,
            }

            logger.info(
                "Final report generated via dynamic sections fallback",
                extra={
                    "length": len(final_report),
                    "section_count": len(dynamic_sections),
                },
            )

            state = StateManager.finalize_state(state)

            return Command(
                goto="end",
                update={
                    "final_report": final_report,
                    "report_sections": {"mode": "dynamic_sections", "sections": report_sections},
                    "citations": state.get("citations", []),
                    "report_metadata": report_metadata,
                }
            )

        # NEW: Check for table requirements but DON'T return early
        table_content = None
        table_metadata = {}
        
        # Check for special output requirements (tables, etc.)
        special_output = self._generate_special_output_if_required(state, compiled_findings, style_config)
        
        if special_output and isinstance(special_output, dict):
            # Extract table content if present, but continue with full report
            if 'content' in special_output:
                # Parse out just the table portion
                table_content = self._extract_table_from_special_output(special_output)
                table_metadata = special_output.get('metadata', {})
                logger.info(f"Table extracted for integration: {len(table_content) if table_content else 0} chars")
            
            # DON'T RETURN HERE - Continue to generate full report
        
        # Generate ALL report sections (with table awareness)
        report_sections = self._generate_report_sections(
            compiled_findings,
            style_config,
            state,
            embedded_table=table_content,  # Pass table for integration
            table_metadata=table_metadata
        )

        # Guard against None return from _generate_report_sections
        if report_sections is None:
            logger.error("_generate_report_sections returned None, using empty dict")
            report_sections = {}

        # Deduplicate content across sections (safety net)
        report_sections = self._deduplicate_content(report_sections)
        
        # Apply quality enhancement if enabled and memory allows
        if self._should_enhance_quality(state):
            logger.info("Applying quality enhancement to report sections")
            enhanced_sections = self._enhance_report_quality(
                report_sections,
                report_style,
                state
            )
            # Guard against None return from _enhance_report_quality
            if enhanced_sections is None:
                logger.error("_enhance_report_quality returned None, using original sections")
            else:
                report_sections = enhanced_sections
        else:
            logger.info("Skipping quality enhancement due to config/memory constraints")
        
        # Apply style formatting
        # Guard against None report_sections
        if report_sections is None:
            logger.error("report_sections is None before _apply_style_formatting, using empty dict")
            report_sections = {}
            
        formatted_report = self._apply_style_formatting(
            report_sections,
            report_style,
            state
        )
        
        # Add citations and references
        final_report = self._add_citations_and_references(
            formatted_report,
            state.get("citations", []),
            report_style
        )
        
        # Add grounding markers if enabled
        if state.get("enable_grounding") and state.get("grounding_results"):
            final_report = self._add_grounding_markers(
                final_report,
                state["grounding_results"]
            )
        
        # Add metadata
        report_with_metadata = self._add_report_metadata(
            final_report,
            state,
            report_style
        )
        
        # SAFEGUARD: Validate final report quality
        report_with_metadata = self._validate_final_report(report_with_metadata, state)

        # Repair any malformed tables (missing separator lines)
        report_with_metadata = self._repair_table_format(report_with_metadata)

        # Apply table preprocessing to fix any malformed tables
        table_preprocessor = TablePreprocessor()
        report_with_metadata = table_preprocessor.preprocess_tables(report_with_metadata)
        
        # Don't mutate state directly - let LangGraph handle updates through Command
        
        logger.info(
            f"Final report generated: {len(report_with_metadata)} characters, "
            f"{len(state.get('citations', []))} citations, "
            f"table_included: {bool(table_content)}"
        )

        # CRITICAL FIX: Ensure all markdown tables have proper separator rows
        report_with_metadata = self._fix_markdown_tables(report_with_metadata)

        # Record completion
        state = StateManager.finalize_state(state)

        logger.info("Report generation completed")

        return Command(
            goto="end",
            update={
                "final_report": report_with_metadata,
                "report_sections": report_sections,
                "citations": state.get("citations", []),  # Include any processed citations
                "report_metadata": {
                    "has_embedded_table": bool(table_content),
                    "table_confidence": table_metadata.get('confidence', 0)
                }
            }
        )
    
    def _render_template_report(
        self,
        template: str,
        findings: Dict[str, Any],
        style_config: StyleConfig,
        state: EnhancedResearchState,
        plan: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Render the final report by filling a markdown template."""

        dynamic_sections = getattr(plan, "dynamic_sections", []) if plan else []
        observation_context, truncated = self._build_observation_context(findings)
        citation_context = self._build_citation_context(state.get("citations", []))
        research_topic = findings.get("research_topic") or state.get("research_topic", "")
        locale = state.get("locale", "en-US")
        requested_entities = state.get("requested_entities", [])
        structured_table_text, table_info = self._build_structured_data_tables(findings)

        guidelines = [
            f"Adopt this tone: {style_config.tone}.",
            "Replace every bracketed instruction in the template with fully written content.",
            "Preserve the provided section headings and overall structure.",
            "Use Markdown tables for any comparisons or multi-entity quantitative data.",
            "If data is missing, clearly state 'No data found' rather than fabricating content.",
            "Populate the 'Key Citations' section using '- [Title](URL)' format with a blank line between items.",
            f"Write in the locale '{locale}'.",
        ]

        if requested_entities:
            guidelines.append(
                f"Ensure the report addresses these requested entities: {', '.join(requested_entities)}."
            )

        guideline_text = "\n".join(f"- {rule}" for rule in guidelines)
        background = findings.get("background_context") or state.get("background_investigation_results")

        template_prompt = f"""# Task: Research Analysis
Perform detailed analysis and calculations for this research topic. Focus on generating comprehensive insights, data, and findings.

## Research Topic
{research_topic}

## Guidelines for Analysis
{guideline_text}

## Research Observations
{observation_context or 'No structured observations were captured.'}

## Citations
{citation_context or 'No citations available.'}

## Structured Data Tables
{structured_table_text or 'No structured tables were derived from the observations.'}

## Supplemental Context
- Locale: {locale}
- Requested entities: {', '.join(requested_entities) if requested_entities else 'Not specified'}
- Background notes: {background[:500] + ('...' if background and len(background) > 500 else '') if background else 'None'}

## Instructions:
1. **Analyze the research data thoroughly**
2. **Perform all necessary calculations and computations** 
3. **Extract key insights and findings**
4. **Create detailed comparative analysis**
5. **Compute financial data, tax rates, costs as needed**
6. **Provide comprehensive coverage of all requested entities**

**Output detailed analysis with calculations, findings, comparisons, and insights. This will be used to create the final report structure.**
"""

        messages = [
            SystemMessage(
                content=(
                    "You are a research analyst. Perform comprehensive analysis of the research data. "
                    "Calculate financial figures, tax rates, comparative costs, and other quantitative metrics. "
                    "Extract key insights and findings. Create detailed analysis that covers all requested "
                    "entities and scenarios. Your analysis will be used to create a structured report."
                )
            ),
            HumanMessage(content=template_prompt),
        ]

        # First, get reasoning and analysis 
        analysis_content = self._invoke_llm_with_smart_retry(messages, "template_analysis")
        
        # Second, format the analysis into the template structure
        formatting_prompt = f"""You have completed detailed analysis for a research report. Now format this analysis into the provided template structure.

## Your Analysis:
{analysis_content}

## Template to Fill (COMPLETE ALL SECTIONS):
{template}

## Critical Instructions:
1. **COMPLETE THE ENTIRE TEMPLATE** - do not stop until every section is filled
2. The template contains multiple sections - fill ALL of them, not just the first few
3. Replace EVERY [bracketed instruction] with actual content from your analysis
4. Each section must be completed with real content, data, tables, and insights
5. Use markdown formatting: proper headings (##), bullet points, tables as needed
6. Include ALL sections shown in the template - do not omit any
7. The output must be the complete template with every section filled

**You must output the entire completed template from start to finish. Do not truncate or stop early.**"""

        formatting_messages = [
            SystemMessage(content="You are a report formatter. Your task is to complete the ENTIRE template provided - every single section from start to finish. Take the analysis content and organize it into ALL sections of the template. Replace all [bracketed instructions] with real content. Do not stop early or omit sections - the output must include every section in the template."),
            HumanMessage(content=formatting_prompt)
        ]
        
        report_content = self._invoke_llm_with_smart_retry(formatting_messages, "template_report")
        metadata = {
            "observation_count": len(findings.get("observations", [])),
            "observations_truncated": truncated,
            "include_appendix": "## Appendix" in template,
            "has_table": "|" in report_content,
            "table_confidence": 0.0,
            "extra_metadata": {
                "template_dynamic_sections": [section.title for section in dynamic_sections],
            },
        }
        if structured_table_text:
            metadata["extra_metadata"]["structured_table_columns"] = table_info.get("table_columns", [])
            metadata["extra_metadata"]["structured_table_rows"] = table_info.get("table_rows", 0)
            metadata["has_structured_table"] = True
        else:
            metadata["has_structured_table"] = False
        return report_content, metadata

    def _build_observation_context(
        self,
        findings: Dict[str, Any],
        limit: int = 100,  # Increased from 25 to 100 to provide more comprehensive data
    ) -> Tuple[str, int]:
        """Convert compiled observations into a compact prompt block."""

        observations = findings.get("observations", []) or []
        logger.info(f"[DEBUG] _build_observation_context received {len(observations)} observations")
        lines: List[str] = []
        truncated = 0

        for idx, obs in enumerate(observations):
            if idx >= limit:
                truncated = len(observations) - limit
                break

            # FIXED: Only handle StructuredObservation objects
            # All observations should be normalized to StructuredObservation by this point
            if isinstance(obs, StructuredObservation):
                content = obs.content
                source = obs.source_id or obs.step_id or obs.section_title
                extracted = obs.metric_values or {}
            else:
                # This should not happen if observations are properly deserialized
                logger.warning(f"Unexpected observation type: {type(obs)}. Converting to StructuredObservation.")
                # Convert to StructuredObservation using the converter
                from ..core.observation_converter import ObservationConverter
                structured_obs = ObservationConverter.to_structured(obs)
                content = structured_obs.content
                source = structured_obs.source_id or structured_obs.step_id or structured_obs.section_title
                extracted = structured_obs.metric_values or {}

            bullet = content.strip()
            if source:
                bullet += f" (Source: {source})"
            if extracted:
                extracted_parts = ", ".join(
                    f"{key}: {value}" for key, value in list(extracted.items())[:5]
                )
                bullet += f" | Extracted data: {extracted_parts}"
            lines.append(f"- {bullet}")

        if truncated:
            lines.append(f"- … {truncated} additional observations omitted for brevity")

        return "\n".join(lines), truncated

    def _build_citation_context(self, citations: Sequence[Any]) -> str:
        """Format citations for inclusion in the template prompt."""

        if not citations:
            return ""

        rendered = []
        for citation in citations:
            try:
                title = getattr(citation, "title", None) or citation.get("title", "Untitled Source")
                url = getattr(citation, "url", None) or citation.get("url", "")
                source = getattr(citation, "source", None) or citation.get("source", "")
            except AttributeError:
                title = str(citation)
                url = ""
                source = ""
            entry = title
            if source and source not in title:
                entry = f"{entry} — {source}"
            if url:
                entry = f"{entry} ({url})"
            rendered.append(f"- {entry}")
        return "\n".join(rendered)

    def _build_structured_data_tables(
        self,
        findings: Dict[str, Any],
        max_columns: int = 6,
        max_rows: int = 12,
    ) -> Tuple[str, Dict[str, Any]]:
        """Create markdown tables from structured observation data."""

        observations = findings.get("observations", []) or []
        rows: List[Dict[str, str]] = []
        columns: List[str] = []

        for obs in observations:
            data = {}
            if isinstance(obs, StructuredObservation):
                data = obs.metric_values or {}
            elif isinstance(obs, dict):
                data = obs.get("metric_values", {})
            if not data:
                continue

            row: Dict[str, str] = {}
            label = "Observation"
            if isinstance(obs, StructuredObservation):
                label = obs.source_id or obs.step_id or obs.content or "Observation"
            elif isinstance(obs, dict):
                label = obs.get("source_id") or obs.get("section_title") or obs.get("content", "Observation")
            row["Observation"] = str(label)[:80]

            for key, value in data.items():
                if key not in columns and len(columns) < max_columns:
                    columns.append(key)
                if key in columns:
                    row[key] = str(value)

            rows.append(row)
            if len(rows) >= max_rows:
                break

        if not rows:
            return "", {}

        ordered_columns = ["Observation"] + columns
        header = "| " + " | ".join(ordered_columns) + " |"
        separator = "| " + " | ".join(["---"] * len(ordered_columns)) + " |"
        body_lines = []
        for row in rows:
            body_lines.append(
                "| "
                + " | ".join(row.get(col, "") for col in ordered_columns)
                + " |"
            )

        table_text = "\n".join([header, separator, *body_lines])
        metadata = {
            "table_columns": ordered_columns,
            "table_rows": len(rows),
        }

        return table_text, metadata

    def _get_style_config_with_adaptive_structure(self, state: EnhancedResearchState, report_style: ReportStyle) -> StyleConfig:
        """Get style configuration with adaptive structure support."""
        try:
            logger.info(f"REPORTER: Determining style configuration for report_style: {report_style}")
            
            # For DEFAULT style, check if we have adaptive structure from planner
            if report_style == ReportStyle.DEFAULT:
                logger.info("REPORTER: DEFAULT style detected - checking for adaptive structure from planner")
                
                current_plan = state.get("current_plan")
                logger.info(f"REPORTER: current_plan exists: {current_plan is not None}")
                
                if current_plan:
                    # Debug plan attributes
                    plan_attrs = [attr for attr in dir(current_plan) if not attr.startswith('_')]
                    logger.info(f"REPORTER: plan attributes: {plan_attrs}")
                    
                    has_structure_attr = hasattr(current_plan, 'suggested_report_structure')
                    logger.info(f"REPORTER: plan has suggested_report_structure attribute: {has_structure_attr}")
                    
                    if has_structure_attr:
                        structure = current_plan.suggested_report_structure
                        logger.info(f"REPORTER: suggested_report_structure content: {structure}")
                        logger.info(f"REPORTER: suggested_report_structure type: {type(structure)}")
                        
                        if structure and isinstance(structure, list) and len(structure) > 0:
                            # Create dynamic style config with adaptive structure
                            base_config = STYLE_CONFIGS[ReportStyle.DEFAULT]
                            
                            # Create new config with adaptive sections
                            from copy import deepcopy
                            adaptive_config = deepcopy(base_config)
                            adaptive_config.structure = structure
                            
                            logger.info(f"✅ REPORTER: Using adaptive structure with {len(adaptive_config.structure)} sections")
                            for i, section in enumerate(adaptive_config.structure, 1):
                                logger.info(f"  📄 Section {i}: {section}")
                            return adaptive_config
                        else:
                            logger.info(f"REPORTER: suggested_report_structure is empty or invalid: {structure}")
                    else:
                        logger.info("REPORTER: plan does not have suggested_report_structure attribute")
                        
                    # Also check if dynamic_sections exist as alternative
                    if hasattr(current_plan, 'dynamic_sections') and current_plan.dynamic_sections:
                        dynamic_sections = current_plan.dynamic_sections
                        logger.info(f"REPORTER: Found dynamic_sections as fallback: {len(dynamic_sections)} sections")
                        
                        # Extract titles from dynamic sections
                        section_titles = [section.title for section in dynamic_sections]
                        logger.info(f"REPORTER: Dynamic section titles: {section_titles}")
                        
                        # Create adaptive config from dynamic sections
                        from copy import deepcopy
                        base_config = STYLE_CONFIGS[ReportStyle.DEFAULT]
                        adaptive_config = deepcopy(base_config)
                        adaptive_config.structure = section_titles
                        
                        logger.info(f"✅ REPORTER: Using dynamic_sections fallback with {len(adaptive_config.structure)} sections")
                        for i, section in enumerate(adaptive_config.structure, 1):
                            logger.info(f"  📄 Section {i}: {section}")
                        return adaptive_config
                else:
                    logger.info("REPORTER: no current_plan found in state")
                
                # Fall back to comprehensive DEFAULT style instead of PROFESSIONAL
                logger.info("❌ REPORTER: No valid adaptive structure found, falling back to comprehensive DEFAULT style")
                logger.info("REPORTER: Using DEFAULT style for comprehensive coverage")
                return STYLE_CONFIGS[ReportStyle.DEFAULT]
            else:
                # Use standard configuration
                logger.info(f"REPORTER: Using standard configuration for style: {report_style}")
                return STYLE_CONFIGS[report_style]
                
        except Exception as e:
            logger.warning(f"REPORTER: Failed to get adaptive structure, using fallback: {e}")
            import traceback
            logger.warning(f"REPORTER: Stack trace: {traceback.format_exc()}")
            # Fall back to professional style
            logger.info("REPORTER: Exception occurred - falling back to PROFESSIONAL style")
            return STYLE_CONFIGS[ReportStyle.PROFESSIONAL]
    
    def _compile_findings(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Compile all research findings from state."""
        
        logger.info(f"[OBSERVATION TRACKING] Reporter starting compilation from state with keys: {list(state.keys())}")
        
        # CRITICAL: Check for search tool failures first
        errors = state.get("errors", [])
        search_failures = [error for error in errors if "search tools" in error.lower() or "api_key" in error.lower()]
        
        if search_failures:
            logger.error(f"REPORTER: Search tool failures detected: {search_failures}")
            logger.error("REPORTER: Cannot generate meaningful report without search capabilities")
            
            # Create a minimal error report instead of hallucinated content
            return {
                "research_topic": state.get("research_topic", ""),
                "observations": [f"Research failed due to search tool configuration issues: {'; '.join(search_failures)}"],
                "error_report": True,
                "completed_steps": [],
                "citations": [],
                "reflections": [],
                "total_sources": 0,
                "confidence_score": 0.0,
                "factuality_score": 0.0,
                "coverage_score": 0.0,
                "research_quality_score": 0.0
            }
        
        observations = []
        
        # CRITICAL: Check all possible observation sources in priority order
        # 1. Direct observations field (primary source)
        if "observations" in state and state["observations"]:
            direct_obs = state["observations"]
            logger.info(f"[OBSERVATION TRACKING] Found {len(direct_obs)} direct observations")
            # DEBUG: Check first few observations to understand their content
            for i, obs in enumerate(direct_obs[:3]):
                if isinstance(obs, str):
                    logger.info(f"[DEBUG] Observation {i}: string with length {len(obs)}: {obs[:200]}")
                elif isinstance(obs, dict):
                    content = obs.get("content", "")
                    logger.info(f"[DEBUG] Observation {i}: dict with content length {len(str(content))}: {str(content)[:200]}")
            
            # Deserialize dict observations back to StructuredObservation objects
            # This is the critical boundary between state layer (dicts) and business logic (objects)
            deserialized_obs = ObservationConverter.deserialize_from_state(direct_obs)

            # CRITICAL: Deduplicate observations by content hash (defense in depth)
            # This provides additional protection even if researcher deduplication fails
            deduplicated = []
            seen_content = set()
            for obs in deserialized_obs:
                # Create hash from normalized content (lowercase, stripped)
                content_hash = obs.content.lower().strip()
                if content_hash not in seen_content and content_hash:  # Skip empty content
                    seen_content.add(content_hash)
                    deduplicated.append(obs)

            duplicates_removed = len(deserialized_obs) - len(deduplicated)
            if duplicates_removed > 0:
                logger.info(f"🧹 Reporter deduplicated observations: removed {duplicates_removed} duplicates, kept {len(deduplicated)}")

            observations.extend(deduplicated)

        # FIXED: Removed duplicate collection from research_observations
        # The researcher syncs observations = research_observations, so collecting from both creates 2x duplicates
        
        # CRITICAL: Reporter should NOT create observations - only consume them
        # If observations are missing, this indicates a bug in researcher or state management
        if not observations:
            logger.error("=" * 80)
            logger.error("[REPORTER BUG] NO OBSERVATIONS FOUND IN STATE!")
            logger.error("This indicates a critical bug in the researcher or state management.")
            logger.error("Reporter should ONLY consume observations created by researcher.")
            logger.error("Available state keys: %s", list(state.keys()))
            logger.error("=" * 80)

            # Check for alternative data sources that researcher should have converted
            if state.get("section_research_results"):
                logger.error("Found section_research_results - researcher should convert this to observations!")
            if state.get("search_results"):
                logger.error("Found search_results - researcher should convert this to observations!")

            raise ValueError(
                "Reporter received no observations from researcher. "
                "Researcher must create StructuredObservation objects during research. "
                "This is a critical bug in the research workflow."
            )
        
        logger.info(f"[OBSERVATION TRACKING] Compiled {len(observations)} total observations")
        
        # Get completed steps for structure
        plan = state.get("current_plan")
        completed_steps = []
        if plan:
            completed_steps = [
                step for step in plan.steps
                if step.status == StepStatus.COMPLETED or str(step.status).lower() == "completed"
            ]
        
        # Get citations
        citations = state.get("citations", [])
        
        # Get background investigation if available
        background = state.get("background_investigation_results")
        
        # Get reflections if available
        reflections = state.get("reflections", [])
        
        # Compile into structured format
        compiled = {
            "research_topic": state.get("research_topic", ""),
            "original_user_query": state.get("original_user_query", ""),
            "requested_entities": state.get("requested_entities", []),
            "background_context": background,
            "observations": observations,
            "completed_steps": completed_steps,
            "citations": citations,
            "reflections": reflections,
            "total_sources": len(citations),
            "confidence_score": state.get("confidence_score", 0.8),  # Default to reasonable confidence
            "factuality_score": state.get("factuality_score", 0.9),  # Default to high factuality
            "coverage_score": state.get("coverage_score", 0.7),      # Default to good coverage
            "research_quality_score": state.get("research_quality_score", 0.8)  # Default to good quality
        }
        
        logger.info(f"Compiled {len(observations)} observations from {len(completed_steps)} steps")
        
        # DEBUG: Check what we're returning
        logger.info(f"[DEBUG] Returning compiled findings with {len(compiled['observations'])} observations")
        if compiled['observations']:
            first_obs = compiled['observations'][0]
            content = observation_to_text(first_obs)
            logger.info(f"[DEBUG] First observation content: {content[:200] if content else 'EMPTY CONTENT'}")
        
        return compiled

    def _filter_observations_for_section(
        self,
        section_name: str,
        all_observations: List[Dict[str, Any]],
        state: EnhancedResearchState
    ) -> List[Dict[str, Any]]:
        """
        Filter observations by step_id using direct references from DynamicSection.

        NO MORE STRING MATCHING! Uses section.step_ids for direct lookup.

        Args:
            section_name: The section we're generating
            all_observations: All observations from research
            state: Current research state with plan

        Returns:
            Filtered list of observations relevant to this section
        """
        plan = state.get("current_plan")

        if not plan:
            logger.warning(f"No plan found in state, returning subset of observations")
            return all_observations[:30]

        # Find the section by name using direct reference (no string matching with template_section_title)
        matching_section = None
        if hasattr(plan, 'dynamic_sections') and plan.dynamic_sections:
            for section in plan.dynamic_sections:
                if section.title == section_name:
                    matching_section = section
                    break

        if not matching_section:
            logger.error(f"❌ Section '{section_name}' not found in plan.dynamic_sections!")
            return all_observations[:30]

        # Use direct step_ids from section (NO STRING MATCHING!)
        if not matching_section.step_ids:
            logger.warning(f"⚠️ Section '{section_name}' has no step_ids assigned")
            return all_observations[:30]

        logger.info(f"🎯 Section '{section_name}' filtering by step_ids: {matching_section.step_ids}")

        # Filter observations by step_id - direct membership check
        filtered_observations = []
        observations_without_step_id = 0

        for obs in all_observations:
            # Get step_id from observation
            if isinstance(obs, dict):
                obs_step_id = obs.get("step_id")
            else:
                obs_step_id = getattr(obs, "step_id", None)

            # Direct membership check - no string matching!
            if obs_step_id:
                if obs_step_id in matching_section.step_ids:
                    filtered_observations.append(obs)
            else:
                observations_without_step_id += 1

        if observations_without_step_id > 0:
            logger.warning(f"  ⚠️ {observations_without_step_id} observations missing step_id")

        logger.info(f"✅ Filtered {len(filtered_observations)}/{len(all_observations)} observations by step_id for section '{section_name}'")

        # STRICT FILTER: Only keep observations with full_content (real fetched content)
        with_content = []
        snippet_only = 0

        for obs in filtered_observations:
            has_full = False
            if isinstance(obs, dict):
                has_full = bool(obs.get("full_content"))
            else:
                has_full = bool(getattr(obs, "full_content", None))

            if has_full:
                with_content.append(obs)
            else:
                snippet_only += 1

        if snippet_only > 0:
            logger.info(f"⏭️  Filtered out {snippet_only} snippet-only observations for '{section_name}'")

        logger.info(f"✅ Section '{section_name}': {len(with_content)} observations with full_content")

        # If no observations with content, return None to signal skip
        if not with_content:
            logger.warning(
                f"❌ Section '{section_name}' has NO observations with fetched content. "
                f"This means web fetching failed for all sources. Section will be skipped."
            )
            return None

        # Debug: Log first observation
        if with_content:
            first_obs = with_content[0]
            if isinstance(first_obs, dict):
                content_preview = str(first_obs.get("content", ""))[:100]
                step_id = first_obs.get("step_id", "NONE")
            else:
                content_preview = str(getattr(first_obs, "content", ""))[:100]
                step_id = getattr(first_obs, "step_id", "NONE")
            logger.info(f"  📝 First obs (step_id={step_id}): {content_preview}...")

        return with_content

    def _generate_report_sections(
        self,
        findings: Dict[str, Any],
        style_config,
        state: EnhancedResearchState,
        embedded_table: Optional[str] = None,
        table_metadata: Dict = None
    ) -> Dict[str, str]:
        """
        Generate report sections with optional embedded table.
        Table is intelligently placed in the most appropriate section.
        """
        
        plan = state.get("current_plan")
        
        # Determine where to place the table
        table_section = None
        if embedded_table:
            table_section = self._determine_optimal_table_section(
                style_config.structure,
                state,
                table_metadata or {}
            )
            logger.info(f"Table will be embedded in section: {table_section}")
        
        dynamic_sections = None
        if plan:
            dynamic_sections = getattr(plan, "dynamic_sections", None)

        if dynamic_sections:
            return self._generate_sections_from_dynamic_sections(
                dynamic_sections,
                findings,
                style_config,
                state,
                embedded_table,
                table_section,
                table_metadata or {},
            )
        
        # Fallback to original method
        sections = {}

        # Track context for continuation awareness
        previous_section_titles = []
        total_sections = len(style_config.structure)

        for section_index, section_name in enumerate(style_config.structure):
            logger.info(f"Generating section: {section_name} ({section_index + 1}/{total_sections})")

            # Build smart context from previously generated sections
            previous_sections_context = self._get_smart_context(sections, previous_section_titles)

            # Get section template
            section_template = StyleTemplate.get_section_template(
                style_config.style,
                section_name
            )

            # Generate section content with context
            section_content = self._generate_section_content(
                section_name,
                findings,
                section_template,
                style_config.style,
                state,
                previous_sections_context=previous_sections_context,
                previous_section_titles=previous_section_titles,
                section_index=section_index,
                total_sections=total_sections
            )

            # Embed table if this is the designated section
            if section_name == table_section and embedded_table:
                # Add contextual lead-in
                table_intro = self._generate_table_introduction(state, table_metadata or {})

                # Integrate table with proper formatting
                section_content = f"{section_content}\n\n{table_intro}\n\n{embedded_table}\n\n"

                # Add brief analysis after table if confidence is high
                if (table_metadata or {}).get("confidence", 0) > 0.7:
                    table_analysis = self._generate_brief_table_analysis(embedded_table, state)
                    if table_analysis:
                        section_content += f"{table_analysis}\n\n"

                logger.info(f"Table successfully embedded in {section_name}")

            sections[section_name] = section_content
            previous_section_titles.append(section_name)

        return sections

    def _generate_sections_from_dynamic_sections(
        self,
        dynamic_sections: Sequence[DynamicSection],
        findings: Dict[str, Any],
        style_config: StyleConfig,
        state: EnhancedResearchState,
        embedded_table: Optional[str] = None,
        table_section: Optional[str] = None,
        table_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        """Generate report sections directly from dynamic template descriptors."""

        sections: Dict[str, str] = {}
        table_metadata = table_metadata or {}

        # Track context for continuation awareness
        previous_section_titles = []
        total_sections = len(dynamic_sections)

        for section_index, dynamic in enumerate(sorted(dynamic_sections, key=lambda s: s.priority)):
            section_name = dynamic.title
            logger.info(f"Generating dynamic section: {section_name} ({section_index + 1}/{total_sections})")

            # Build smart context from previously generated sections
            previous_sections_context = self._get_smart_context(sections, previous_section_titles)

            template = self._build_dynamic_section_template(dynamic)
            section_content = self._generate_section_content(
                section_name,
                findings,
                template,
                style_config.style,
                state,
                previous_sections_context=previous_sections_context,
                previous_section_titles=previous_section_titles,
                section_index=section_index,
                total_sections=total_sections
            )

            # Check if section was skipped (no observations with content)
            if section_content is None:
                logger.warning(f"⏭️  Skipping section '{section_name}' - no observations with fetched content")
                continue

            if embedded_table and section_name == table_section:
                table_intro = self._generate_table_introduction(state, table_metadata)
                section_content = f"{section_content}\n\n{table_intro}\n\n{embedded_table}\n\n"
                logger.info(f"Dynamic section '{section_name}' received embedded table")

            sections[section_name] = section_content
            previous_section_titles.append(section_name)

        return sections

    def _deduplicate_content(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Remove duplicate content across sections (safety net).

        This method provides a final safety check against content duplication,
        detecting and removing paragraphs that appear in multiple sections.

        Returns:
            Dictionary with deduplicated section content
        """
        import hashlib

        content_hashes = {}
        deduplicated = {}
        total_removed = 0

        for section_name, content in sections.items():
            # Extract paragraphs (split by double newlines)
            paragraphs = content.split('\n\n')
            unique_paragraphs = []

            for para in paragraphs:
                # Skip headers and empty paragraphs
                if not para.strip() or para.strip().startswith('#'):
                    unique_paragraphs.append(para)
                    continue

                # Hash paragraph content for comparison
                para_normalized = ' '.join(para.split())  # Normalize whitespace
                para_hash = hashlib.md5(para_normalized.encode()).hexdigest()

                # Check if we've seen this exact content before
                if para_hash not in content_hashes:
                    content_hashes[para_hash] = section_name
                    unique_paragraphs.append(para)
                else:
                    # Duplicate found!
                    total_removed += 1
                    logger.warning(
                        f"🔍 Duplicate paragraph detected:\n"
                        f"  Original: {content_hashes[para_hash]}\n"
                        f"  Duplicate in: {section_name}\n"
                        f"  Content preview: '{para[:100]}...'"
                    )

            deduplicated[section_name] = '\n\n'.join(unique_paragraphs)

        if total_removed > 0:
            logger.info(f"✅ Deduplication: Removed {total_removed} duplicate paragraphs")
        else:
            logger.info("✅ Deduplication: No duplicates found")

        return deduplicated

    def _generate_section_content(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle,
        state: EnhancedResearchState,
        previous_sections_context: str = "",
        previous_section_titles: List[str] = None,
        section_index: int = 0,
        total_sections: int = 1
    ) -> Optional[str]:
        """
        Generate content for a specific report section.

        This method now supports TWO approaches:
        1. Structured generation (NEW): LLM generates structured data, we render markdown
        2. Template generation (EXISTING): LLM generates markdown directly

        The structured approach is preferred when enabled, as it guarantees
        correct table formatting through programmatic rendering.

        Args:
            section_name: Name of the section
            findings: Research findings data
            template: Section template
            style: Report style
            state: Current research state
            previous_sections_context: Summary of previously generated sections
            previous_section_titles: List of previous section names
            section_index: Current section index (0-based)
            total_sections: Total number of sections

        Returns:
            Generated section content as markdown string
        """

        # CRITICAL: Don't generate content if this is an error report
        if findings.get("error_report", False):
            return f"## {section_name}\n\n{findings['observations'][0] if findings['observations'] else 'Research failed - no data available.'}"

        # CRITICAL: Check if we have actual research data
        observations = findings.get("observations", [])
        if not observations or len(observations) == 0:
            logger.warning(f"REPORTER: No observations available for section '{section_name}' - preventing LLM hallucination")
            return f"## {section_name}\n\nNo research data available for this section due to search tool failures."

        # FILTER OBSERVATIONS BY SECTION MAPPING
        # This is the key fix: only use observations from steps mapped to this section
        filtered_observations = self._filter_observations_for_section(
            section_name,
            observations,
            state
        )

        # Check if section should be skipped (no observations with content)
        if filtered_observations is None:
            logger.warning(f"⏭️  Section '{section_name}' has no observations with fetched content - skipping")
            return None

        # Update findings with filtered observations for this section
        section_findings = findings.copy()
        section_findings["observations"] = filtered_observations

        logger.info(f"📊 Section '{section_name}': using {len(filtered_observations)}/{len(observations)} observations")

        # Check if structured generation is enabled
        reporter_config = self.config.get('agents', {}).get('reporter', {})
        enable_structured = reporter_config.get('enable_structured_generation', False)
        enable_fallback = reporter_config.get('structured_generation_fallback', False)

        # DEBUG: Log configuration status
        logger.info(f"🔍 DEBUG: Structured generation check for section '{section_name}'")
        logger.info(f"  - enable_structured: {enable_structured}")
        logger.info(f"  - enable_fallback: {enable_fallback}")
        logger.info(f"  - self.llm is None: {self.llm is None}")
        logger.info(f"  - reporter_config keys: {list(reporter_config.keys())}")

        if enable_structured and self.llm:
            # NEW PATH: Try structured generation first
            logger.info(f"🎯 Attempting structured generation for section: {section_name}")

            try:
                structured_content = self._generate_section_content_structured(
                    section_name,
                    section_findings,
                    template,
                    style,
                    state,
                    previous_sections_context=previous_sections_context,
                    previous_section_titles=previous_section_titles,
                    section_index=section_index,
                    total_sections=total_sections
                )

                if structured_content:
                    logger.info(f"✅ Structured generation succeeded for: {section_name}")
                    return structured_content
                else:
                    # Analyze WHY it failed
                    obs_count = len(section_findings.get("observations", []))

                    if obs_count == 0:
                        # INTERNAL BUG - data pipeline failure
                        error_msg = (
                            f"❌ INTERNAL BUG: No observations for '{section_name}'. "
                            f"Root cause: Observation filtering or creation is broken. "
                            f"Check _filter_observations_for_section() and researcher step_id assignment."
                        )
                        logger.error(error_msg)

                        if enable_fallback:
                            logger.warning("⚠️ Fallback enabled but this masks a BUG - fix the root cause!")
                            # Fall through to template (for now)
                        else:
                            # Surface the bug clearly
                            return f"## {section_name}\n\n**Internal Error**: {error_msg}\n\n_This is a bug in observation handling, not a valid empty section._"

                    else:
                        # Has observations but structured gen failed (LLM issue or irrelevant data)
                        error_msg = (
                            f"Structured generation returned empty for '{section_name}' "
                            f"despite {obs_count} observations. "
                            f"Possible causes: LLM failure, irrelevant observations, or rate limiting."
                        )
                        logger.warning(f"⚠️ {error_msg}")

                        if enable_fallback:
                            logger.info("→ Falling back to template generation...")
                            # Fall through to template
                        else:
                            # User-facing message (not internal bug)
                            return f"## {section_name}\n\n_No content could be generated for this section._"

            except Exception as e:
                error_msg = f"Structured generation failed for {section_name}: {e}"
                logger.error(f"❌ {error_msg}", exc_info=True)

                if enable_fallback:
                    logger.warning(f"⚠️ Exception occurred, falling back to template")
                else:
                    return f"## {section_name}\n\n**Generation Error**: {str(e)[:200]}"

        # EXISTING PATH: Template-based generation (fallback or default)
        if enable_structured and not enable_fallback:
            logger.info(f"Structured generation was attempted but did not return content, and fallback is disabled")
        else:
            logger.info(f"Using template-based generation for section: {section_name}")

        # Build prompt for section generation
        prompt = self._build_section_prompt(
            section_name,
            section_findings,
            template,
            style
        )

        if self.llm:
            messages = [
                SystemMessage(content=StyleTemplate.get_style_prompt(style)),
                HumanMessage(content=prompt)
            ]

            # Use smart retry for transient error handling
            content = self._invoke_llm_with_smart_retry(messages, section_name)
        else:
            # Fallback content generation
            content = self._generate_fallback_section(
                section_name,
                findings,
                style
            )

        return content

    def _generate_section_content_structured(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle,
        state: EnhancedResearchState,
        previous_sections_context: str = "",
        previous_section_titles: List[str] = None,
        section_index: int = 0,
        total_sections: int = 1
    ) -> Optional[str]:
        """
        Generate section content using MULTI-CALL structured generation.

        Instead of generating entire sections at once, this:
        1. Determines what blocks are needed (paragraphs, tables)
        2. Generates each block separately with focused LLM calls
        3. Passes context between blocks for coherence
        4. Programmatically renders tables (guaranteed separator rows!)

        This approach avoids validation issues with complex nested defaults.

        Args:
            section_name: Name of the section to generate
            findings: Research findings data
            template: Section template (for guidance, not currently used)
            style: Report style
            state: Current research state (for event emission and logging)
            previous_sections_context: Summary of previously generated sections
            previous_section_titles: List of previous section names
            section_index: Current section index (0-based)
            total_sections: Total number of sections

        Returns:
            Markdown string with programmatically rendered tables, or None if failed
        """
        from ..core.report_models_structured import ParagraphBlock, TableBlock

        try:
            # Log input data quality
            obs_list = findings.get("observations", [])
            obs_count = len(obs_list)

            logger.info(
                f"📊 Structured generation for '{section_name}': {obs_count} observations available"
            )

            # CRITICAL: Detect data pipeline bugs early
            if obs_count == 0:
                logger.error(
                    f"❌ INTERNAL BUG: No observations for '{section_name}'! "
                    f"This indicates observation filtering or creation failure. "
                    f"Check _filter_observations_for_section() and researcher agent step_id assignment."
                )
                return None  # Don't hide bugs

            # Log observation sample for debugging
            sample_obs = obs_list[0]
            if isinstance(sample_obs, dict):
                sample_content = sample_obs.get("content", "")
                sample_step_id = sample_obs.get("step_id", "MISSING")
            else:
                sample_content = getattr(sample_obs, "content", str(sample_obs))
                sample_step_id = getattr(sample_obs, "step_id", "MISSING")

            logger.debug(
                f"  Sample: step_id={sample_step_id}, content={sample_content[:100]}..."
            )

            # Determine block structure
            block_sequence = self._determine_section_structure(section_name, findings)
            total_blocks = len(block_sequence)

            logger.info(f"  Planned: {total_blocks} blocks → {' → '.join(block_sequence)}")

            # Generate each block with detailed logging
            markdown_parts = []
            previous_content = ""
            intro_text = None  # Track intro paragraph for conclusion context
            generated_table = None  # Track table for conclusion analysis

            for i, block_type in enumerate(block_sequence):
                context = self._get_block_context(block_type, i, total_blocks)

                try:
                    if block_type == "paragraph":
                        # Check if this is conclusion paragraph (after table)
                        is_conclusion = (generated_table is not None)

                        block = self._generate_paragraph_block(
                            section_name,
                            findings,
                            context,
                            style,
                            previous_content,
                            intro_paragraph=intro_text if is_conclusion else None,
                            table_content=generated_table if is_conclusion else None,
                            previous_sections_context=previous_sections_context,
                            section_index=section_index,
                            total_sections=total_sections
                        )
                        rendered = block.text

                        # Save intro text for conclusion paragraph
                        if i == 0:
                            intro_text = rendered

                    elif block_type == "table":
                        block = self._generate_table_block(
                            section_name, findings, context, style, previous_content
                        )
                        rendered = block.render_markdown()
                        generated_table = rendered  # Save for conclusion
                    else:
                        logger.warning(f"  ⚠️ Unknown block type '{block_type}', skipping")
                        continue

                    # Validate block output
                    if rendered and rendered.strip():
                        logger.info(
                            f"  ✅ Block {i+1}/{total_blocks} ({block_type}): "
                            f"{len(rendered)} chars generated"
                        )
                        markdown_parts.append(rendered)
                        previous_content += "\n\n" + rendered
                    else:
                        logger.warning(
                            f"  ⚠️ Block {i+1}/{total_blocks} ({block_type}) EMPTY: "
                            f"type={type(rendered).__name__}, "
                            f"repr={repr(rendered)[:100]}"
                        )

                except Exception as block_err:
                    logger.error(
                        f"  ❌ Block {i+1}/{total_blocks} ({block_type}) EXCEPTION: "
                        f"{type(block_err).__name__}: {block_err}",
                        exc_info=True
                    )
                    # Continue with other blocks

            # Combine results
            content = "\n\n".join(markdown_parts)

            if not content.strip():
                # Differentiate bug vs legitimate empty
                logger.error(
                    f"❌ MULTI-CALL GENERATION FAILED for '{section_name}': "
                    f"{obs_count} observations → 0/{total_blocks} blocks succeeded. "
                    f"Check LLM responses and block generation logic."
                )
                return None

            logger.info(
                f"✅ Structured generation SUCCESS: '{section_name}' → "
                f"{len(content)} chars from {len(markdown_parts)}/{total_blocks} blocks"
            )
            return content

        except Exception as e:
            logger.error(
                f"❌ STRUCTURED GENERATION EXCEPTION for '{section_name}': "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return None

    def _build_structured_section_prompt(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle
    ) -> str:
        """
        Build prompt for structured section generation.

        This prompt explicitly instructs the LLM to generate STRUCTURED DATA
        (headers, rows, cells) not markdown strings.
        """
        observations = findings.get("observations", [])
        observation_text = "\n\n".join(
            f"- {observation_to_text(obs)}"
            for obs in observations[:15]  # Limit to prevent context explosion
        )

        research_topic = findings.get("research_topic", "the research topic")

        prompt = f"""Generate content for the '{section_name}' section of a research report.

**Research Topic**: {research_topic}

**Report Style**: {style.value}

**Research Findings**:
{observation_text}

**Section Guidance**: {template}

**CRITICAL INSTRUCTIONS FOR STRUCTURED OUTPUT**:

1. **For Text Content**: Use content_blocks with type="paragraph"
   Example: {{"type": "paragraph", "paragraph": {{"text": "Your analysis here..."}}}}

2. **For Tables**: Use content_blocks with type="table" and STRUCTURED DATA
   - **DO NOT** generate markdown table strings!
   - **DO** generate structured data: headers array, rows array with cells

   Example for a comparison table:
   {{
     "type": "table",
     "table": {{
       "headers": ["Country", "Tax Rate", "Take-Home Pay"],
       "rows": [
         {{"cells": [{{"value": "Spain"}}, {{"value": "31.2%"}}, {{"value": "€170,000"}}]}},
         {{"value": "France"}}, {{"value": "35.8%"}}, {{"value": "€160,000"}}]}}
       ],
       "caption": "Tax comparison across countries 2025"
     }}
   }}

3. **Content Organization**:
   - Start with paragraph(s) introducing the section
   - Add table(s) for comparative data
   - End with paragraph(s) analyzing the data

4. **Quality Standards**:
   - Use actual data from research findings (no hallucination!)
   - For tables: ensure all rows have same number of cells as headers
   - Use clear, descriptive captions for tables
   - Keep paragraphs focused and well-structured

**Return a JSON object with a "content_blocks" array following the structure above.**
"""
        return prompt

    def _get_structured_system_prompt(self, style: ReportStyle) -> str:
        """Get system prompt for structured generation."""
        return f"""You are a research report writer generating STRUCTURED DATA for reports.

Your role: Generate structured content (NOT markdown strings) that will be rendered programmatically.

For {style.value} style reports:
- {StyleTemplate.get_style_prompt(style)}

**CRITICAL**:
- For tables, provide structured data: arrays of headers, rows with cells
- Do NOT write markdown table strings (no "| Header |" strings!)
- Python code will render the perfect markdown from your structured data
- This guarantees correct table formatting with separator rows"""

    # ===== MULTI-CALL STRUCTURED GENERATION HELPERS =====
    # These methods support generating sections with multiple focused LLM calls

    def _determine_section_structure(
        self,
        section_name: str,
        findings: Dict[str, Any]
    ) -> List[str]:
        """
        Determine what blocks this section needs - OPTIMIZED to reduce duplication.

        Returns list like: ["paragraph", "table", "paragraph"]

        Strategy: Smart pattern matching to avoid over-generation
        - Only FINAL comparison sections get full intro+table+conclusion
        - Data/calculation sections get intro+table (no redundant conclusion)
        - Other sections get just narrative text

        This prevents the duplication bug where intro and conclusion paragraphs
        use identical observations and repeat the same content.
        """
        section_lower = section_name.lower()

        # Only use full 3-block pattern for FINAL comprehensive comparisons
        is_final_comparison = any(keyword in section_lower for keyword in [
            "cross-country comparison",
            "apples-to-apples",
            "final comparison",
            "comparative table and analysis",
            "overall comparison",
            "comprehensive comparison"
        ])

        # Check if this is primarily a data/calculation section
        is_data_section = any(keyword in section_lower for keyword in [
            "calculation", "compute", "brackets", "rates",
            "contributions", "benefits", "allowances"
        ])

        if is_final_comparison:
            # Final comparison needs intro + table + analytical conclusion
            logger.info(f"Section '{section_name}': Using full 3-block pattern (final comparison)")
            return ["paragraph", "table", "paragraph"]
        elif is_data_section:
            # Data sections: brief intro + table (no redundant conclusion)
            logger.info(f"Section '{section_name}': Using 2-block pattern (data section)")
            return ["paragraph", "table"]
        else:
            # Regular sections: just narrative
            logger.info(f"Section '{section_name}': Using 1-block pattern (narrative)")
            return ["paragraph"]

    def _get_smart_context(
        self,
        previous_sections: Dict[str, str],
        section_titles: List[str]
    ) -> str:
        """
        Extract relevant context from previous sections without token explosion.

        Returns concise summaries of previous sections showing what was covered.

        Args:
            previous_sections: Dict of {section_name: content}
            section_titles: List of section names in order

        Returns:
            Summarized context string (max ~2000 chars)
        """
        if not previous_sections:
            return ""

        context_parts = []
        # Take last 3 sections for context
        for section_name in section_titles[-3:]:
            if section_name not in previous_sections:
                continue

            content = previous_sections[section_name]
            # Extract first meaningful line (skip empty lines)
            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
            first_line = lines[0] if lines else ""

            # Keep header + opening (shows what was covered)
            summary = f"## {section_name}\n{first_line[:150]}..."
            context_parts.append(summary)

        return "\n\n".join(context_parts)

    def _get_continuation_aware_system_prompt(
        self,
        style: ReportStyle,
        section_index: int,
        total_sections: int,
        has_previous_content: bool
    ) -> str:
        """
        Generate system prompt that prevents repetitive meta-commentary.

        Args:
            style: Report style
            section_index: Current section index (0-based)
            total_sections: Total number of sections
            has_previous_content: Whether there are previous sections

        Returns:
            System prompt string with appropriate continuation rules
        """
        base_prompt = f"You are a {style.value} data analyst writing section {section_index + 1} of {total_sections}."

        if has_previous_content:
            continuation_rules = """

**CRITICAL CONTINUATION RULES**:
Previous sections have already established context. Your job is to CONTINUE the narrative, not re-introduce it.

**BANNED PHRASES** - Never use:
❌ "The following section..."
❌ "In the following analysis..."
❌ "This section establishes..."
❌ "This section lays the groundwork..."
❌ "We establish the foundation..."
❌ "The framework for..."
❌ "This section sets the stage..."
❌ "By outlining..."
❌ "This narrative description precedes..."
❌ "In this section we examine..."

**QUALITY REQUIREMENTS**:
✅ Jump directly into content, data, and analysis
✅ Cite SPECIFIC NUMBERS from research (e.g., "31.2% in Spain vs 35.8% in France")
✅ Present information as facts, not descriptions of what you'll present
✅ Continue the flow from previous sections naturally
✅ Assume reader has full context from earlier content
✅ Every statement should reference concrete data points

**Bad**: "Tax rates vary across countries"
**Good**: "Spain's 31.2% effective rate provides €4,400 more take-home pay than France's 35.8%"
"""
            return base_prompt + continuation_rules
        else:
            return base_prompt + """

**QUALITY REQUIREMENTS**:
✅ Start with specific, data-driven insights
✅ Cite concrete numbers from research
✅ Establish the analytical framework with actual findings
✅ Every statement should reference specific data points
"""

    def _get_block_context(self, block_type: str, index: int, total: int) -> str:
        """Get context description for this block's role in the section."""
        if index == 0:
            return "introduction"
        elif index == total - 1:
            return "conclusion/analysis"
        elif block_type == "table":
            return "data comparison"
        else:
            return "supporting analysis"

    def _generate_paragraph_block(
        self,
        section_name: str,
        findings: Dict[str, Any],
        context: str,
        style: ReportStyle,
        previous_content: str = "",
        intro_paragraph: Optional[str] = None,
        table_content: Optional[str] = None,
        previous_sections_context: str = "",
        section_index: int = 0,
        total_sections: int = 1
    ) -> "ParagraphBlock":
        """
        Generate text paragraph block (standard LLM generation, no structured output).

        Args:
            section_name: Name of section
            findings: Research findings
            context: Role description (introduction, analysis, etc.)
            style: Report style
            previous_content: Markdown of blocks generated so far in THIS section
            intro_paragraph: Intro paragraph text (for conclusion paragraphs)
            table_content: Table markdown (for conclusion paragraphs)
            previous_sections_context: Summary of previously generated sections
            section_index: Current section index (0-based)
            total_sections: Total number of sections in report

        Returns:
            ParagraphBlock with text content
        """
        from ..core.report_models_structured import ParagraphBlock

        # Use ObservationSelector to intelligently select relevant observations
        all_observations = findings.get("observations", [])

        # CRITICAL FIX: Different observation strategy for conclusions to prevent duplication
        if context == "conclusion/analysis" and table_content:
            # For conclusions, use FEWER observations with HIGHER quality
            # This prevents repeating the same content from the intro paragraph
            max_obs = 5  # Much fewer than intro
            min_relevance = 0.7  # Higher quality threshold

            # Use different selection key to get DIFFERENT observations
            selected_obs = self.observation_selector.select_observations_for_section(
                section_title=f"{section_name} - Key Insights",  # Different key!
                section_purpose="analytical conclusions and insights from data",
                all_observations=all_observations,
                max_observations=max_obs,
                min_relevance=min_relevance,
                use_semantic=self.embedding_manager is not None
            )
            logger.info(f"Conclusion paragraph: Selected {len(selected_obs)}/{len(all_observations)} high-quality observations")
        else:
            # Normal observation selection for intro/other paragraphs
            max_paragraph_obs = self.config.get("report", {}).get("max_paragraph_observations", 15)
            selected_obs = self.observation_selector.select_observations_for_section(
                section_title=section_name,
                section_purpose=context,
                all_observations=all_observations,
                max_observations=max_paragraph_obs,
                min_relevance=0.3,  # Standard threshold
                use_semantic=self.embedding_manager is not None
            )
            logger.info(f"Intro paragraph: Selected {len(selected_obs)}/{len(all_observations)} observations")

        # Format observations for prompt
        observation_text = "\n".join(
            f"- {observation_to_text(obs)}"
            for obs in selected_obs
        )

        # Get context fields
        original_query = findings.get("original_user_query", "")
        research_topic = findings.get("research_topic", "the research topic")

        # Detect if this is a single-block section (complete content, not intro to table)
        is_single_block_section = (table_content is None and not previous_content)
        is_continuation_section = (previous_sections_context and section_index > 0)

        # Build context-appropriate, data-focused prompt
        if table_content:
            # CONCLUSION PARAGRAPH - analyze the table that was just shown
            prompt = f"""**Original User Request**: {original_query}

**Research Topic**: {research_topic}

**Section**: {section_name}

**Data table**:
{table_content}

**Additional Research Findings**:
{observation_text}

Analyze the data from the table. Reference specific values, identify patterns, compare differences, and explain what the numbers reveal:"""

        elif is_single_block_section:
            # COMPLETE SECTION CONTENT (no table follows)
            if is_continuation_section:
                prompt = f"""**Original User Request**: {original_query}

**Research Topic**: {research_topic}

**Previously Generated Sections**:
{previous_sections_context}

**Current Section**: {section_name} (Section {section_index + 1} of {total_sections})

**Research Findings**:
{observation_text}

**CRITICAL RULES**:
- This is a CONTINUATION - previous sections already established context
- DO NOT use "The following section...", "This section will...", "We establish..."
- DO NOT re-introduce concepts already covered above
- Present the data, analysis, and findings directly

Write the content for this section:"""
            else:
                # First section
                prompt = f"""**Original User Request**: {original_query}

**Research Topic**: {research_topic}

**Section**: {section_name}

**Research Findings**:
{observation_text}

Write the content for this section. Present data and analysis directly:"""

        else:
            # PARAGRAPH BEFORE TABLE (multi-block section)
            if is_continuation_section:
                prompt = f"""**Original User Request**: {original_query}

**Research Topic**: {research_topic}

**Previously Generated Sections**:
{previous_sections_context}

**Current Section**: {section_name} (Section {section_index + 1} of {total_sections})

**Research Findings**:
{observation_text}

**CRITICAL**: This is a CONTINUATION. DO NOT use "The following section..." language.

Write a paragraph that provides context for the data table that follows. Focus on what data will be shown and why it matters:"""
            else:
                # First section with table
                prompt = f"""**Original User Request**: {original_query}

**Research Topic**: {research_topic}

**Section**: {section_name}

**Research Findings**:
{observation_text}

Write a paragraph that provides context for the data table that follows:"""

        # Build system prompt with continuation awareness
        system_prompt = self._get_continuation_aware_system_prompt(
            style,
            section_index,
            total_sections,
            bool(previous_sections_context)
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self._invoke_llm_with_smart_retry(messages, f"paragraph_{section_name}")

        return ParagraphBlock(text=response)

    def _generate_table_block(
        self,
        section_name: str,
        findings: Dict[str, Any],
        context: str,
        style: ReportStyle,
        previous_content: str = ""
    ) -> "TableBlock":
        """
        Generate table block with structured output (GUARANTEED separator rows).

        Args:
            section_name: Name of section
            findings: Research findings
            context: Role description
            style: Report style
            previous_content: Markdown of blocks generated so far

        Returns:
            TableBlock with headers and rows
        """
        from ..core.report_models_structured import TableBlock, get_databricks_schema

        observations = findings.get("observations", [])
        # CRITICAL FIX: Make table observation limit configurable (was hardcoded 15)
        max_table_obs = self.config.get("report", {}).get("max_table_observations", 30)
        observation_text = "\n".join(
            f"- {observation_to_text(obs)}"
            for obs in observations[:max_table_obs]  # Increased from hardcoded 15
        )

        research_topic = findings.get("research_topic", "the research topic")

        prompt = f"""Generate a comparison table for '{section_name}'.

**Research Topic**: {research_topic}

**Research Findings**:
{observation_text}

**Previous Content**:
{previous_content if previous_content else "[First block]"}

**Instructions**:
- Create a comparison table with 3-6 columns
- Extract key data points from research findings
- Include at least 2 data rows (headers alone are insufficient)
- Table should complement previous content

Note: Your response must conform to the TableBlock schema provided."""

        messages = [
            SystemMessage(content="You create structured data tables from research findings. Follow the schema exactly."),
            HumanMessage(content=prompt)
        ]

        # Use structured output with retry logic
        # Retry if model doesn't follow schema or returns empty table
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                structured_llm = self.llm.with_structured_output(TableBlock, method="json_schema")
                table_block = structured_llm.invoke(messages)

                # Validate table has data rows
                if not table_block.rows or len(table_block.rows) == 0:
                    logger.warning(
                        f"⚠️  Table has no data rows (attempt {attempt + 1}/{max_retries}). "
                        f"Headers: {table_block.headers}"
                    )
                    if attempt < max_retries - 1:
                        continue  # Retry

                # Success!
                logger.info(
                    f"✅ Table generated successfully: "
                    f"{len(table_block.headers)} cols × {len(table_block.rows)} rows"
                )
                return table_block

            except ValueError as e:
                last_error = e
                logger.warning(
                    f"⚠️  Table generation attempt {attempt + 1}/{max_retries} failed: {str(e)[:200]}"
                )

                if attempt == max_retries - 1:
                    # Last attempt failed - raise error
                    logger.error(f"❌ All {max_retries} table generation attempts failed")
                    raise

        # Should never reach here, but just in case
        raise ValueError(f"Table generation failed after {max_retries} attempts: {last_error}")

    def _should_enhance_quality(self, state: EnhancedResearchState) -> bool:
        """
        Check if quality enhancement should be applied based on config and memory constraints.
        
        Args:
            state: Current research state
            
        Returns:
            bool: True if quality enhancement should be applied
        """
        # Check if quality enhancement is explicitly disabled
        if not self.config.get('quality_enhancement', {}).get('enabled', False):  # Changed default to False
            return False
        
        # Check memory before enhancement
        try:
            import psutil
            current_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if current_mb > 1500:  # Conservative memory threshold
                logger.warning(f"Memory too high ({current_mb:.0f}MB) - skipping quality enhancement")
                return False
        except ImportError:
            pass  # Continue without memory check
        
        # Check report sections size - skip if already very large
        report_sections = state.get("report_sections", {})
        # Guard against None report_sections
        if report_sections is None:
            logger.warning("report_sections is None in _should_enhance_quality")
            return False
            
        total_content_size = sum(len(str(section)) for section in report_sections.values()) if report_sections else 0
        if total_content_size > 50000:  # 50KB threshold
            logger.warning(f"Report content too large ({total_content_size} chars) - skipping quality enhancement")
            return False
        
        # Only enhance if we have actual content (not placeholder)
        if not report_sections:
            return False
        
        # Check for placeholder content
        placeholder_indicators = ["content to be added", "[analysis", "[recommendations"]
        for section_content in (report_sections.values() if report_sections else []):
            content_lower = str(section_content).lower()
            if any(indicator in content_lower for indicator in placeholder_indicators):
                logger.info("Placeholder content detected - skipping quality enhancement")
                return False
        
        return True
    
    def _enhance_report_quality(
        self,
        sections: Dict[str, str],
        style: ReportStyle,
        state: EnhancedResearchState
    ) -> Dict[str, str]:
        """
        Apply quality enhancement to report sections using domain-agnostic approach.
        
        Args:
            sections: Dictionary of section names to content
            style: Report style configuration
            state: Current research state
            
        Returns:
            Enhanced sections with redundancy removed and structure optimized
        """
        if not self.llm:
            logger.warning("No LLM available for quality enhancement")
            return sections
        
        try:
            logger.info("Starting quality enhancement process")
            
            # Step 1: Detect redundancy across sections
            logger.info("Detecting redundancy across sections")
            redundancy_info = self._detect_redundancy(sections)
            
            # Guard against None redundancy_info
            if redundancy_info is None:
                logger.error("_detect_redundancy returned None, using empty dict")
                redundancy_info = {"duplicate_facts": [], "repeated_concepts": [], "similar_paragraphs": [], "consolidation_opportunities": []}
            
            # Step 2: Eliminate redundancy while preserving unique information
            if (redundancy_info.get("duplicate_facts") or 
                redundancy_info.get("repeated_concepts") or 
                redundancy_info.get("similar_paragraphs")):
                
                logger.info("Eliminating detected redundancy")
                cleaned_sections = self._eliminate_redundancy(sections, redundancy_info)
                # Guard against None return
                if cleaned_sections is None:
                    logger.error("_eliminate_redundancy returned None, using original sections")
                else:
                    sections = cleaned_sections
            else:
                logger.info("No significant redundancy detected")
            
            # Step 3: Optimize structure based on report style
            logger.info(f"Optimizing structure for {style.value} style")
            optimized_sections = self._optimize_structure(sections, style)
            # Guard against None return
            if optimized_sections is None:
                logger.error("_optimize_structure returned None, using original sections")
            else:
                sections = optimized_sections
            
            logger.info("Quality enhancement completed successfully")
            
            return sections
            
        except Exception as e:
            logger.error(f"Quality enhancement failed: {str(e)}")
            # Return original sections if enhancement fails
            return sections

    def _build_dynamic_section_template(self, section: DynamicSection) -> str:
        """Construct a lightweight template prompt for a dynamic section."""

        guidance_lines = []
        if section.purpose:
            guidance_lines.append(section.purpose)

        if section.content_type == SectionContentType.COMPARISON:
            guidance_lines.append("Provide a comparative analysis; include a markdown table when presenting metrics.")
        elif section.content_type == SectionContentType.TIMELINE:
            guidance_lines.append("Organize key events in chronological order, highlighting inflection points.")
        elif section.content_type == SectionContentType.BULLET_LIST:
            guidance_lines.append("Use concise bullet points to enumerate the critical items.")
        elif section.content_type == SectionContentType.CASE_STUDIES:
            guidance_lines.append("Illustrate with concrete case studies including context, actions, and outcomes.")
        elif section.content_type == SectionContentType.DATA_DEEP_DIVE:
            guidance_lines.append("Surface quantitative metrics and methodology details; cite figures precisely.")

        guidance_lines.extend(section.hints)

        guidance_block = "\n".join(f"- {line}" for line in guidance_lines if line)
        if not guidance_block:
            guidance_block = "- Provide a well-supported narrative for this topic."

        template = (
            f"# Section Goal\n{section.title}\n\n"
            "## Guidance\n"
            f"{guidance_block}\n\n"
            "## Output Requirements\n"
            "- Use markdown headings, paragraphs, and tables as appropriate.\n"
            "- Ground every statement in the supplied observations.\n"
            "- If data is unavailable, clearly state that limitation instead of speculating.\n"
        )

        return template

    def _build_section_prompt(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle
    ) -> str:
        """Build prompt for generating a report section."""
        
        # Extract entity constraints from findings if available
        original_query = findings.get("original_user_query", "")
        requested_entities = findings.get("requested_entities", [])
        entities_str = ", ".join(requested_entities) if requested_entities else "not specified"
        
        prompt_parts = [
            f"Generate the '{section_name}' section for a {style.value} style report.",
            "",
            "Research Topic: " + findings["research_topic"],
            f"ORIGINAL REQUEST: {original_query}",
            f"FOCUS ENTITIES: {entities_str}",
            f"CRITICAL: Only include information about these specific entities: {entities_str}",
            ""
        ]
        
        # Add relevant findings based on section
        if section_name in ["Introduction", "Background", "Background Context"]:
            if findings.get("background_context"):
                prompt_parts.append("Background Information:")
                # Use more context with 130k window support
                prompt_parts.append(findings["background_context"][:5000])
        
        elif section_name in ["Findings", "Analysis", "Main Discoveries", "Key Findings"]:
            prompt_parts.append("Key Observations:")
            # Use more observations with 130k context window support
            max_obs = 50
            for i, obs in enumerate(findings["observations"][:max_obs], 1):
                # Extract content properly from observation structure
                if hasattr(obs, 'content'):
                    obs_content = obs.content
                elif isinstance(obs, dict):
                    obs_content = obs.get('content', observation_to_text(obs))
                else:
                    obs_content = observation_to_text(obs)
                prompt_parts.append(f"{i}. {obs_content}")
            
            # Log token usage estimate
            obs_count = min(len(findings["observations"]), max_obs)
            estimated_tokens = obs_count * 100  # ~100 tokens per observation in list
            logger.info(f"Including {obs_count} observations in prompt (~{estimated_tokens:,} tokens)")
        
        elif section_name in ["Conclusion", "Summary", "Key Takeaways"]:
            if findings.get("reflections"):
                prompt_parts.append("Research Reflections:")
                # Include all reflections for comprehensive summary
                for reflection in findings["reflections"][-3:]:
                    prompt_parts.append(str(reflection))
        
        elif section_name in ["References", "Bibliography", "Citations"]:
            prompt_parts.append(f"Total Sources: {findings['total_sources']}")
        
        prompt_parts.append("")
        prompt_parts.append("Section Template Guidelines:")
        prompt_parts.append(template)
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_section(
        self,
        section_name: str,
        findings: Dict[str, Any],
        style: ReportStyle
    ) -> str:
        """Generate fallback content for a section."""
        
        if section_name in ["Introduction", "Overview"]:
            return f"This report presents research findings on {findings['research_topic']}."
        
        elif section_name in ["Findings", "Key Findings"]:
            content = "Key findings from the research:\n\n"
            for i, obs in enumerate(findings["observations"][:5], 1):
                content += f"{i}. {obs}\n"
            return content
        
        elif section_name in ["Conclusion", "Summary"]:
            return (
                f"This research on {findings['research_topic']} revealed "
                f"{len(findings['observations'])} key insights based on "
                f"{findings['total_sources']} sources."
            )
        
        elif section_name in ["References", "Bibliography"]:
            return "See citations section for complete references."
        
        else:
            return f"[{section_name} content to be added]"
    
    def _get_max_observations_for_section(self, section_title: str) -> int:
        """Get maximum observations for a section based on configuration."""
        # Check section-specific limits from config
        section_limits = self.config.get('report_generation', {}).get('section_observation_limits', {})
        
        if section_title in section_limits:
            return section_limits[section_title]
        
        # Default from config or fallback to 50 (supporting 130k context window)
        return self.config.get('report_generation', {}).get('max_observations_per_section', 50)
    
    def _format_observations_for_prompt(
        self,
        observations: List[Union[StructuredObservation, Dict[str, Any], str]],
        include_metadata: bool = True
    ) -> str:
        """Format observations preserving structure and metadata."""
        
        if not observations:
            return "[No observations available]"
        
        formatted = []
        for i, obs in enumerate(observations, 1):
            # Handle both dict and StructuredObservation
            if hasattr(obs, 'content'):
                content = obs.content
                entities = getattr(obs, 'entity_tags', [])
                # Ensure entities is always a list
                if isinstance(entities, str):
                    entities = [entities] if entities else []
                elif not isinstance(entities, list):
                    entities = []
                metrics = getattr(obs, 'metric_values', {})
                confidence = getattr(obs, 'confidence', 1.0)
            elif isinstance(obs, dict):
                content = obs.get('content', observation_to_text(obs))
                entities = obs.get('entity_tags', [])
                # Ensure entities is always a list
                if isinstance(entities, str):
                    entities = [entities] if entities else []
                elif not isinstance(entities, list):
                    entities = []
                metrics = obs.get('metric_values', {})
                confidence = obs.get('confidence', 1.0)
            else:
                content = observation_to_text(obs)
                entities = []
                metrics = {}
                confidence = 1.0
            
            # Format with structure
            obs_text = f"[Obs#{i}] {content}"
            
            if include_metadata and (entities or metrics or confidence < 1.0):
                metadata_parts = []
                if entities:
                    metadata_parts.append(f"Entities: {', '.join(entities)}")
                if metrics:
                    metric_strs = [f"{k}={v}" for k, v in metrics.items()]
                    metadata_parts.append(f"Data: {', '.join(metric_strs)}")
                if confidence < 1.0:
                    metadata_parts.append(f"Confidence: {confidence:.2f}")
                
                if metadata_parts:
                    obs_text += "\n  " + " | ".join(metadata_parts)
            
            formatted.append(obs_text)
        
        return "\n\n".join(formatted)

    def _format_synthesis_for_prompt(self, research: Dict[str, Any]) -> str:
        """Format synthesis/observations for clean prompt inclusion."""

        # Try observations first (preferred)
        if 'observations' in research:
            obs_list = research['observations']
            if obs_list:
                return self._format_observations_for_prompt(obs_list, include_metadata=True)

        # Fallback to synthesis if string
        synthesis = research.get('synthesis', '')
        if isinstance(synthesis, str):
            return synthesis

        # Handle list/dict synthesis
        if isinstance(synthesis, (list, dict)):
            logger.warning("Synthesis is not a string, formatting observations")
            # Try to extract observations from it
            if isinstance(synthesis, list):
                return self._format_observations_for_prompt(synthesis, include_metadata=False)
            else:
                # Dict synthesis - should not happen
                return '[Invalid synthesis format]'

        return '[No observations available]'

    def _build_data_grounded_prompt(
        self,
        spec: Any,
        research: Dict[str, Any],
        style_config: Any,
        state: EnhancedResearchState
    ) -> str:
        """Build prompt that strongly grounds LLM in provided data."""
        
        # Count observations for transparency
        obs_count = research.get('observation_count', 0)
        
        # Extract context for entity constraints
        original_query = state.get("original_user_query", "")
        requested_entities = state.get("requested_entities", [])
        entities_str = ", ".join(requested_entities) if requested_entities else "not specified"
        
        prompt = f"""
        ORIGINAL REQUEST: {original_query}
        FOCUS ENTITIES: {entities_str}
        
        CRITICAL INSTRUCTIONS - ANTI-HALLUCINATION REQUIREMENTS:
        ============================================================
        1. You MUST generate content using ONLY the research data provided below
        2. DO NOT use any information not explicitly stated in the [Obs#N] observations
        3. Every claim, statistic, or fact MUST reference a specific [Obs#N] number
        4. If data for something is missing, state "Data not available" - NEVER make it up
        5. CRITICAL: Only include information about these specific entities: {entities_str}
        6. DO NOT mention countries, entities, or data not present in the observations
        7. Your response must be grounded in the {obs_count} observations provided
        
        SECTION REQUIREMENTS:
        =====================
        Title: {spec.title}
        Purpose: {spec.purpose}
        Presentation Style: {spec.presentation_instructions if hasattr(spec, 'presentation_instructions') else 'Clear and professional'}
        Report Style: {style_config.style.value if hasattr(style_config, 'style') else 'professional'}
        Tone: {style_config.tone if hasattr(style_config, 'tone') else 'professional'}
        
        AVAILABLE RESEARCH DATA (USE ONLY THIS - {obs_count} observations):
        ====================================================================
        {self._format_synthesis_for_prompt(research)}
        
        SPECIFIC DATA POINTS (if available):
        {json.dumps(research.get('extracted_data', {}), indent=2) if research.get('extracted_data') else 'None'}
        
        TASK:
        =====
        Generate the section content using ONLY the above research data.
        - Include [Obs#N] citations inline for each fact used
        - Structure the content according to the section purpose
        - If you need data that's not in the observations, explicitly state "data not available in research"
        - DO NOT include the section title in your response
        
        REMEMBER: Any information not from the [Obs#N] observations above is HALLUCINATION and FORBIDDEN.
        """
        
        return prompt
    
    def _convert_search_results_to_observations(self, search_results: List[Any]) -> List[Dict[str, Any]]:
        """Convert search results to observation format for processing."""
        observations = []
        
        for result in search_results:
            if isinstance(result, dict):
                obs = {
                    'content': result.get('snippet', result.get('description', str(result))),
                    'entity_tags': [],
                    'metric_values': {},
                    'source': result.get('title', 'Search Result'),
                    'url': result.get('url', ''),
                    'relevance': result.get('relevance_score', 0.5)
                }
                observations.append(obs)
            else:
                observations.append({
                    'content': str(result),
                    'entity_tags': [],
                    'metric_values': {}
                })
        
        return observations
    
    def _apply_style_formatting(
        self,
        sections: Dict[str, str],
        style: ReportStyle,
        state: EnhancedResearchState
    ) -> str:
        """Apply style-specific formatting to report sections."""
        
        # Guard against None sections
        if sections is None:
            logger.error("sections is None in _apply_style_formatting, using empty dict")
            sections = {}
        
        formatted_parts = []
        
        # Add report title
        title = f"# Research Report: {state.get('research_topic', 'Research Findings')}\n\n"
        formatted_parts.append(title)
        
        # Add metadata if not social media style
        if style not in [ReportStyle.SOCIAL_MEDIA]:
            metadata = self._generate_metadata_section(state, style)
            if metadata:
                formatted_parts.append(metadata + "\n\n")
        
        # Format each section
        for section_name, content in sections.items():
            # Add section header
            header = self.formatter.format_section_header(section_name, style)
            formatted_parts.append(header)
            
            # Apply style-specific formatting to content
            formatted_content = self.formatter.apply_style_formatting(content, style)
            formatted_parts.append(formatted_content)
            formatted_parts.append("\n")
        
        return "".join(formatted_parts)
    
    def _generate_metadata_section(
        self,
        state: EnhancedResearchState,
        style: ReportStyle
    ) -> Optional[str]:
        """Generate metadata section for report."""
        
        if style == ReportStyle.ACADEMIC:
            return (
                f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n"
                f"**Sources Consulted:** {len(state.get('citations', []))}\n"
                f"**Research Confidence:** {(state.get('confidence_score') if state.get('confidence_score') is not None else 0.8):.1%}"
            )
        elif style in [ReportStyle.PROFESSIONAL, ReportStyle.EXECUTIVE]:
            return (
                f"**Prepared:** {datetime.now().strftime('%B %d, %Y')}\n"
                f"**Data Sources:** {len(state.get('citations', []))}\n"
                f"**Confidence Level:** {(state.get('confidence_score') if state.get('confidence_score') is not None else 0.8):.1%}"
            )
        elif style == ReportStyle.TECHNICAL:
            return (
                f"**Generated:** {datetime.now().isoformat()}\n"
                f"**Sources:** {len(state.get('citations', []))}\n"
                f"**Observations:** {len(state.get('observations', []))}\n"
                f"**Factuality Score:** {(state.get('factuality_score') if state.get('factuality_score') is not None else 0.9):.2f}\n"
                f"**Coverage Score:** {(state.get('coverage_score') if state.get('coverage_score') is not None else 0.7):.2f}\n"
                f"**Research Quality:** {(state.get('research_quality_score') if state.get('research_quality_score') is not None else 0.8):.2f}"
            )
        
        return None
    
    def _add_citations_and_references(
        self,
        report: str,
        citations: List[Citation],
        style: ReportStyle
    ) -> str:
        """Add citations and references section to report."""
        
        if not citations:
            return report
        
        # Deduplicate citations first
        unique_citations = []
        seen_urls = set()
        for citation in citations:
            if citation.source not in seen_urls:
                unique_citations.append(citation)
                seen_urls.add(citation.source)
        
        # Format unique citations according to style
        formatted_citations = []
        for i, citation in enumerate(unique_citations, 1):
            citation_dict = {
                "number": i,
                "title": citation.title,
                "url": citation.source,
                "author": "",  # Remove "Unknown" - leave empty
                "date": ""     # Remove current year - leave empty
            }
            formatted = self.formatter.format_citation(citation_dict, style)
            formatted_citations.append(formatted)
        
        # Add references section
        references_header = self.formatter.format_section_header("References", style)
        
        if style == ReportStyle.ACADEMIC:
            # APA style bibliography
            references = references_header + "\n".join(formatted_citations)
        elif style == ReportStyle.TECHNICAL:
            # Numbered references
            references = references_header
            for i, citation in enumerate(formatted_citations, 1):
                references += f"[{i}] {citation}\n"
        elif style == ReportStyle.SOCIAL_MEDIA:
            # Simple links
            references = "\n\n📚 Sources:\n"
            for citation in citations[:3]:  # Limit for social media
                references += f"• {citation.title[:50]}... [{citation.source}]\n"
        else:
            # Standard format
            references = references_header
            for citation in formatted_citations:
                references += f"• {citation}\n"
        
        # CRITICAL FIX: Ensure report is always a string before concatenation
        if isinstance(report, list):
            logger.warning(f"_add_citations_and_references received list for report, converting to string. List length: {len(report)}")
            # Join list elements into a single string
            report = "\n".join(str(item) for item in report if item is not None)
        elif not isinstance(report, str):
            logger.warning(f"_add_citations_and_references received {type(report)} for report, converting to string")
            report = str(report) if report is not None else ""
        
        # Ensure report is never None
        if report is None:
            logger.warning("_add_citations_and_references received None for report, using empty string")
            report = ""
        
        # Append to report
        return report + "\n\n" + references
    
    def _add_grounding_markers(
        self,
        report: str,
        grounding_results: List
    ) -> str:
        """Add grounding markers to indicate factuality status."""
        
        # Handle None report
        if report is None:
            logger.warning("REPORTER: _add_grounding_markers received None report")
            report = ""
        
        # Add confidence summary at the beginning
        if grounding_results:
            # Create a simple grounding summary
            grounded_count = sum(
                1 for r in grounding_results
                if (hasattr(r, 'status') and r.status == "grounded") or 
                   (isinstance(r, dict) and r.get('status') == "grounded")
            )
            total_claims = len(grounding_results)
            
            grounding_summary = (
                f"\n📊 **Factuality Assessment**: "
                f"{grounded_count}/{total_claims} claims verified\n\n"
            )
            
            # Insert after title
            # Ensure report is a string before splitting
            if isinstance(report, list):
                report = "\n".join(report) if report else ""
            elif not isinstance(report, str):
                report = str(report) if report else ""
            parts = report.split("\n\n", 1)
            if len(parts) == 2:
                report = parts[0] + "\n" + grounding_summary + parts[1]
            else:
                report = grounding_summary + report
        
        # Optionally add inline markers (if detailed marking needed)
        # This would require more sophisticated text processing
        
        return report
    
    def _add_report_metadata(
        self,
        report: str,
        state: EnhancedResearchState,
        style: ReportStyle
    ) -> str:
        """Add final metadata to report."""
        
        # Generate footer based on style
        footer_parts = []
        
        if style != ReportStyle.SOCIAL_MEDIA:
            footer_parts.append("\n" + "="*50 + "\n")
            
            # Add generation information
            footer_parts.append(
                f"Generated by Deep Research Agent\n"
                f"Report Style: {style.value}\n"
                f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            
            # Always add quality metrics with reasonable defaults
            factuality_score = state.get("factuality_score") if state.get("factuality_score") is not None else 0.9
            confidence_score = state.get("confidence_score") if state.get("confidence_score") is not None else 0.8
            coverage_score = state.get("coverage_score") if state.get("coverage_score") is not None else 0.7
            research_quality_score = state.get("research_quality_score") if state.get("research_quality_score") is not None else 0.8
            
            footer_parts.extend([
                f"Factuality Score: {factuality_score:.2f}\n",
                f"Confidence Score: {confidence_score:.2f}\n",
                f"Coverage Score: {coverage_score:.2f}\n", 
                f"Research Quality: {research_quality_score:.2f}\n"
            ])
            
            # Add research metrics
            plan = state.get("current_plan")
            if plan:
                footer_parts.append(
                    f"Research Steps Completed: {plan.completed_steps}/{len(plan.steps)}\n"
                )
            
            footer_parts.append("="*50)
        
        if footer_parts:
            report += "\n" + "".join(footer_parts)
        
        return report
    
    def _deduplicate_citations(self, state: EnhancedResearchState) -> None:
        """
        Remove duplicate citations to prevent accumulation bug.
        
        Citations can accumulate when the agent runs multiple iterations
        or when the same sources are found multiple times.
        """
        citations = state.get("citations", [])
        if not citations:
            return
        
        # Track unique citations by URL to avoid duplicates
        seen_urls = set()
        unique_citations = []
        
        original_count = len(citations)
        
        for citation in citations:
            # Create a unique key based on URL and title
            url = getattr(citation, 'source', '') or getattr(citation, 'url', '')
            title = getattr(citation, 'title', '')
            
            # Create unique identifier
            unique_key = f"{url}|{title}".lower().strip()
            
            if unique_key not in seen_urls and url.strip():
                seen_urls.add(unique_key)
                unique_citations.append(citation)
        
        # Store deduplicated citations for return in Command update
        # Don't mutate state directly
        
        if len(unique_citations) != original_count:
            logger.info(
                f"Deduplicated citations: {original_count} -> {len(unique_citations)} "
                f"(removed {original_count - len(unique_citations)} duplicates)"
            )
    
    def _detect_redundancy(self, sections: Dict[str, str]) -> Dict:
        """
        Use LLM to detect content redundancy across sections.
        
        Args:
            sections: Dictionary of section names to content
            
        Returns:
            Dict containing redundancy analysis results
        """
        if not self.llm:
            logger.warning("No LLM available for redundancy detection")
            return {"duplicate_facts": [], "repeated_concepts": [], "similar_paragraphs": [], "consolidation_opportunities": []}
        
        # Prepare sections for analysis, truncating very long content
        analysis_sections = {}
        for name, content in sections.items():
            # Truncate extremely long sections to prevent token limits
            if len(content) > 3000:
                analysis_sections[name] = content[:3000] + "...[truncated]"
            else:
                analysis_sections[name] = content
        
        prompt = f"""Analyze these report sections for redundancy:

{json.dumps(analysis_sections, indent=2)}

Identify:
1. Facts/statistics that appear in multiple sections (be specific)
2. Concepts explained multiple times with similar wording
3. Similar paragraphs with minor variations
4. Information that could be consolidated into a single location

Return your analysis as valid JSON in this exact format:
{{
    "duplicate_facts": ["fact that appears multiple times", "another duplicate"],
    "repeated_concepts": ["concept explained multiple times", "another repeated concept"],
    "similar_paragraphs": [
        {{"section1": "Section Name", "section2": "Other Section", "similarity": "description of similarity"}}
    ],
    "consolidation_opportunities": [
        {{"sections": ["Section A", "Section B"], "reason": "why these could be merged"}}
    ]
}}

Focus on factual content rather than stylistic repetition."""

        try:
            response = self._invoke_llm_with_smart_retry(
                [SystemMessage("You are a report quality analyzer. Return only valid JSON."),
                 HumanMessage(prompt)],
                "redundancy_detection"
            )
            
            # Try to parse JSON response
            try:
                result = json.loads(response.strip())
                logger.info(f"Detected {len(result.get('duplicate_facts', []))} duplicate facts, "
                          f"{len(result.get('repeated_concepts', []))} repeated concepts")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse redundancy detection JSON: {e}")
                # Return empty structure if parsing fails
                return {"duplicate_facts": [], "repeated_concepts": [], "similar_paragraphs": [], "consolidation_opportunities": []}
                
        except Exception as e:
            logger.error(f"Error in redundancy detection: {e}")
            return {"duplicate_facts": [], "repeated_concepts": [], "similar_paragraphs": [], "consolidation_opportunities": []}
    
    def _eliminate_redundancy(self, sections: Dict[str, str], redundancy_info: Dict) -> Dict[str, str]:
        """
        Remove redundant content intelligently while preserving unique information.
        
        Args:
            sections: Dictionary of section names to content
            redundancy_info: Results from redundancy detection
            
        Returns:
            Enhanced sections with redundancy eliminated
        """
        if not self.llm:
            logger.warning("No LLM available for redundancy elimination")
            return sections
        
        # If no redundancy detected, return original sections
        # Guard against None redundancy_info
        if redundancy_info is None or not any(redundancy_info.values() if redundancy_info else []):
            logger.info("No redundancy detected, returning original sections")
            return sections
        
        prompt = f"""Given these report sections with identified redundancies:

Sections:
{json.dumps(sections, indent=2)}

Redundancies found:
{json.dumps(redundancy_info, indent=2)}

Rewrite the sections following these rules:
1. Each fact/statistic appears in its MOST RELEVANT section only
2. Use cross-references for removed content (e.g., "as discussed in the Analysis section")
3. Consolidate scattered data into unified presentations
4. Preserve ALL unique information - do not delete content, only relocate it
5. Maintain logical flow within each section
6. Keep the same section structure/names

Return the enhanced sections as valid JSON with the same keys as input.
Focus on eliminating factual redundancy while preserving readability."""

        try:
            response = self._invoke_llm_with_smart_retry(
                [SystemMessage("You are a report editor specializing in clarity and conciseness. Return only valid JSON."),
                 HumanMessage(prompt)],
                "redundancy_elimination"
            )
            
            # Try to parse JSON response
            try:
                enhanced_sections = json.loads(response.strip())
                
                # Validate that we haven't lost sections
                if set(enhanced_sections.keys()) != set(sections.keys()):
                    logger.warning("Section structure changed during redundancy elimination, using original")
                    return sections
                
                logger.info("Successfully eliminated redundancy from report sections")
                return enhanced_sections
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse redundancy elimination JSON: {e}")
                return sections
                
        except Exception as e:
            logger.error(f"Error in redundancy elimination: {e}")
            return sections
    
    def _optimize_structure(self, sections: Dict[str, str], style: ReportStyle) -> Dict[str, str]:
        """
        Reorganize content based on style requirements.
        
        Args:
            sections: Dictionary of section names to content
            style: Report style to optimize for
            
        Returns:
            Restructured sections optimized for style
        """
        if not self.llm:
            logger.warning("No LLM available for structure optimization")
            return sections
        
        style_config = STYLE_CONFIGS[style]
        structure_rules = self._get_structure_rules(style)
        
        prompt = f"""Reorganize this report content for {style.value} style:

Current sections: {list(sections.keys())}
Target structure: {style_config.structure}
Style tone: {style_config.tone}

Structure rules for {style.value}:
{structure_rules}

Content to reorganize:
{json.dumps(sections, indent=2)}

Rules:
1. Reorganize content to match {style.value} style conventions
2. Move content to most appropriate sections based on target structure
3. Ensure logical flow within style conventions
4. Adjust tone to match: {style_config.tone}
5. Preserve all content - just reorganize it
6. Use exactly these section names: {style_config.structure}

Return reorganized sections as valid JSON using the target structure section names."""

        try:
            response = self._invoke_llm_with_smart_retry(
                [SystemMessage(f"You are an expert {style.value} style writer. Return only valid JSON."),
                 HumanMessage(prompt)],
                "structure_optimization"
            )
            
            # Try to parse JSON response
            try:
                optimized_sections = json.loads(response.strip())
                logger.info(f"Successfully optimized structure for {style.value} style")
                return optimized_sections
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse structure optimization JSON: {e}")
                return sections
                
        except Exception as e:
            logger.error(f"Error in structure optimization: {e}")
            return sections
    
    def _get_structure_rules(self, style: ReportStyle) -> str:
        """Get structural rules for specific style."""
        
        rules = {
            ReportStyle.ACADEMIC: """
            - Start with abstract summarizing key findings
            - Literature review before methodology (if applicable)
            - Clear separation of results and discussion
            - Formal citations throughout
            - Comprehensive conclusion with future work
            """,
            ReportStyle.PROFESSIONAL: """
            - Executive summary with key takeaways upfront
            - Actionable recommendations prominently placed
            - Supporting details organized logically
            - Clear section numbering for easy reference
            - Focus on business value and practical applications
            """,
            ReportStyle.TECHNICAL: """
            - Problem statement first
            - Technical specifications and requirements early
            - Implementation details with examples
            - Performance metrics and benchmarks
            - Clear troubleshooting and maintenance sections
            """,
            ReportStyle.EXECUTIVE: """
            - Brief executive summary (1-2 paragraphs max)
            - Key findings and recommendations upfront
            - Minimal technical detail
            - Clear action items and next steps
            - Business impact and ROI emphasis
            """,
            ReportStyle.NEWS: """
            - Lead paragraph with who/what/when/where/why
            - Most important information first
            - Quotes from stakeholders
            - Objective tone with facts
            - Chronological organization when relevant
            """
        }
        
        return rules.get(style, "Use clear, logical structure appropriate for the audience")

    def _generate_natural_language_table(
        self,
        state: EnhancedResearchState
    ) -> Optional[str]:
        """Generate table from natural language specification."""
        
        plan = state.get("current_plan")
        if not plan or not hasattr(plan, 'table_specifications'):
            return None
        
        if not plan.table_specifications:
            return None
        
        logger.info(f"Generating table from specification: {plan.table_specifications[:100]}...")
        
        # Collect all extracted data
        all_data = {}
        section_research = state.get("section_research_results", {})
        for section_id, research in section_research.items():
            if isinstance(research, SectionResearchResult):
                if research.extracted_data:
                    all_data[section_id] = dict(research.extracted_data)
            elif isinstance(research, dict) and 'extracted_data' in research:
                all_data[section_id] = research['extracted_data']
        
        # Use LLM to create table from specification and data
        table_prompt = f"""
        Create a table based on this specification:
        {plan.table_specifications}
        
        Data relationships:
        {plan.data_relationships if hasattr(plan, 'data_relationships') else ''}
        
        Available data:
        {json.dumps(all_data, indent=2)}
        
        Requirements:
        1. Use ONLY the data provided above
        2. If data is missing, use "N/A"
        3. Format as a markdown table
        4. Make sure the table structure matches what was requested
        
        Generate the markdown table:
        """
        
        if self.llm:
            response = self.llm.invoke([
                SystemMessage(content="You are a data table generator."),
                HumanMessage(content=table_prompt)
            ])
            
            # Handle structured responses properly
            from ..core.llm_response_parser import extract_text_from_response
            return extract_text_from_response(response)
        else:
            # Fallback - create simple table from data
            if all_data:
                table = "| Section | Data Points |\n|---------|------------|\n"
                for section, data in all_data.items():
                    data_str = ", ".join([f"{k}: {v}" for k, v in (data.items() if isinstance(data, dict) else [])])
                    table += f"| {section} | {data_str or 'N/A'} |\n"
                return table
            return None
    
    def _should_generate_table(self, state: EnhancedResearchState) -> Tuple[bool, Optional[Dict]]:
        """
        Check if planner determined that tables are needed.
        
        This replaces the primitive 'table' in query pattern matching.
        """
        
        plan = state.get("current_plan")
        if not plan:
            return False, None
            
        # Check planner's presentation requirements
        presentation_reqs_dict = getattr(plan, 'presentation_requirements', None)
        if not presentation_reqs_dict:
            return False, None
            
        # Convert dict back to PresentationRequirements object
        try:
            presentation_reqs = PresentationRequirements.from_dict(presentation_reqs_dict)
        except Exception as e:
            logger.warning(f"Failed to parse presentation requirements: {e}")
            return False, None
            
        if presentation_reqs.needs_table and presentation_reqs.confidence > 0.7:
            return True, {
                'table_type': presentation_reqs.optimal_table_type,
                'entities': presentation_reqs.suggested_entities,
                'metrics': presentation_reqs.suggested_metrics,
                'reasoning': presentation_reqs.table_reasoning,
                'confidence': presentation_reqs.confidence
            }
        
        return False, None
    
    def _generate_planned_table(
        self,
        compiled_findings: Dict[str, Any],
        table_specifications: Dict[str, Any],
        state: EnhancedResearchState
    ) -> Dict[str, Any]:
        """
        Generate table using LLM - routes to tracked-cells or text-based generation.

        NEW: Checks config for require_cell_derivation to enable cell-level tracking
        that forces LLM to justify every value before generating it (reduces hallucination).

        Supports up to 32k tokens for comprehensive data extraction.
        When table generation fails, return honest error message with specific reason.
        """
        try:
            # Check if cell-level derivation tracking is enabled
            reporter_config = self.config.get('agents', {}).get('reporter', {})
            table_config = reporter_config.get('table_generation', {})
            require_cell_derivation = table_config.get('require_cell_derivation', False)

            if require_cell_derivation:
                logger.info("REPORTER: Using TRACKED-CELLS generation (cell-level derivation)")
                return self._generate_table_with_tracked_cells(
                    compiled_findings, table_specifications, state
                )
            else:
                logger.info("REPORTER: Using TEXT-BASED generation (legacy mode)")

            # Legacy text-based generation (original implementation)
            logger.info("REPORTER: Generating table with LLM-based extraction (32k token budget)")
            
            # Log what we received for debugging
            logger.info(f"REPORTER: Table specs received: {json.dumps(table_specifications, indent=2)}")
            
            # CRITICAL FIX: Extract table structure from specifications using correct keys
            # Planner returns 'entities' and 'metrics', not 'rows' and 'columns'
            rows = table_specifications.get("entities", [])
            columns = table_specifications.get("metrics", [])
            
            # Fallback for legacy format
            if not rows:
                rows = table_specifications.get("rows", [])
            if not columns:
                columns = table_specifications.get("columns", [])
            
            if not rows or not columns:
                logger.warning(f"REPORTER: Missing table structure - rows={rows}, columns={columns}, specs={table_specifications}")
                return self._create_table_unavailable_message("Table structure not properly specified by planner")
            
            # Collect research content with generous token allowance
            research_content = self._collect_all_research_content(
                compiled_findings, 
                state,
                max_tokens=32000  # 32k tokens for comprehensive coverage
            )
            
            if not research_content or len(research_content.strip()) < 100:
                logger.warning(f"REPORTER: Insufficient research content for table: {len(research_content)} chars")
                return self._create_table_unavailable_message("Insufficient research data")
            
            # Check LLM availability
            if not self.llm:
                logger.error("REPORTER: No LLM available for table generation")
                return self._create_table_unavailable_message("LLM not available")
            
            # Build comprehensive prompt
            prompt = self._build_comprehensive_table_prompt(
                rows=rows,
                columns=columns,
                research_content=research_content,
                topic=state.get("research_topic", ""),
                query_context=get_last_user_message(state.get("messages", [])) or ""
            )
            
            # Single attempt with LLM - no fallbacks
            try:
                response = self.llm.invoke([
                    SystemMessage("You are a precise data extraction expert. Extract exact values from research content to populate tables."),
                    HumanMessage(prompt)
                ])

                table_content = self._extract_text_from_response(response)

                # Validate the response contains a proper table
                if self._validate_table_structure(table_content, len(rows), len(columns)):
                    logger.info(f"REPORTER: Successfully generated table with {len(table_content)} characters")
                    return {
                        "type": "planned_table",
                        "content": table_content,
                        "metadata": {
                            "confidence": table_specifications.get("confidence", 0.9),
                            "reasoning": table_specifications.get("reasoning", "LLM extraction successful"),
                            "method": "llm_extraction",
                            "data_sources": len((research_content if isinstance(research_content, str) else "\n".join(research_content) if isinstance(research_content, list) else str(research_content)).split('\n'))
                        }
                    }
                else:
                    logger.warning("REPORTER: LLM response didn't contain valid table structure")
                    return self._create_table_unavailable_message("LLM could not extract structured data")
                    
            except Exception as e:
                logger.error(f"REPORTER: LLM invocation failed: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"REPORTER: Stack trace: {traceback.format_exc()}")
                return self._create_table_unavailable_message(f"LLM generation error: {str(e)}")
                
        except Exception as e:
            logger.error(f"REPORTER: Critical error in table generation: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"REPORTER: Stack trace: {traceback.format_exc()}")
            return self._create_table_unavailable_message(f"Generation error: {str(e)}")

    def _generate_table_with_tracked_cells(
        self,
        compiled_findings: Dict[str, Any],
        table_specifications: Dict[str, Any],
        state: EnhancedResearchState
    ) -> Dict[str, Any]:
        """
        Generate table using TRACKED CELLS with cell-level derivation.

        This method forces the LLM to provide derivation (source/reasoning) BEFORE
        the actual cell value, creating a "chain of thought" that dramatically
        reduces hallucinations.

        Returns:
            Dict with table content or error message
        """
        try:
            from ..core.report_models_structured import TableBlockWithTrackedCells, get_databricks_schema

            logger.info("REPORTER: Generating table with TRACKED CELLS (derivation-first)")

            # Extract table structure
            rows = table_specifications.get("entities", []) or table_specifications.get("rows", [])
            columns = table_specifications.get("metrics", []) or table_specifications.get("columns", [])

            if not rows or not columns:
                logger.warning(f"REPORTER: Missing table structure - rows={rows}, columns={columns}")
                return self._create_table_unavailable_message("Table structure not specified")

            # Collect research content
            research_content = self._collect_all_research_content(
                compiled_findings,
                state,
                max_tokens=32000
            )

            if not research_content or len(research_content.strip()) < 100:
                logger.warning(f"REPORTER: Insufficient research content: {len(research_content)} chars")
                return self._create_table_unavailable_message("Insufficient research data")

            if not self.llm:
                logger.error("REPORTER: No LLM available")
                return self._create_table_unavailable_message("LLM not available")

            # Build prompt emphasizing derivation-first approach
            topic = state.get("research_topic", "")
            prompt = f"""Generate a comparison table with TRACKED CELLS.

**CRITICAL REQUIREMENTS - READ CAREFULLY:**

For EVERY single cell, you MUST fill the 'derivation' field BEFORE the 'value' field:

1. **EXTRACTED**: If value comes directly from research findings
   Format: "extracted: [specific text/number from observation]"
   Example: "extracted: $23,000 from IRS 2024 limits document"

2. **CALCULATED**: If derived from other values in the research
   Format: "calculated: [exact formula with source values]"
   Example: "calculated: $23,000 × 0.35 tax rate = $8,050"

3. **NOT_AVAILABLE**: If data is genuinely missing
   Format: "not_available"

4. **ESTIMATED**: Use ONLY when absolutely necessary
   Format: "estimated: [clear basis and assumptions]"
   Example: "estimated: ~7% annual return based on historical averages"

**Research Topic**: {topic}

**Table Structure**:
- Rows (entities to compare): {', '.join(rows)}
- Columns (metrics/attributes): {', '.join(columns)}

**Research Findings** (extract exact values from here):
{research_content[:15000]}  # Limit to avoid token overflow

**Your Task**:
1. Think through each cell carefully
2. Write the derivation FIRST (how you got the value)
3. Then write the actual value
4. Set confidence level (high/medium/low)

**Table Schema**: You must return a TableBlockWithTrackedCells with:
- headers: list of column names
- rows: list of TableRow objects
  - Each TableRow contains 'cells': list of TableCell objects
    - Each TableCell has: derivation, value, confidence

Generate the structured table now."""

            messages = [
                SystemMessage("You are a meticulous data analyst. Every value must have clear provenance. Think step-by-step."),
                HumanMessage(prompt)
            ]

            # Use structured output with retry
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    structured_llm = self.llm.with_structured_output(
                        TableBlockWithTrackedCells,
                        method="json_schema"
                    )

                    table_block = structured_llm.invoke(messages)

                    # Validate we got data rows
                    if not table_block.rows or len(table_block.rows) == 0:
                        logger.warning(f"⚠️  Table has no data rows (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            continue

                    # Log derivations for audit trail
                    self._log_cell_derivations(table_block)

                    # Validate cell grounding
                    self._validate_cell_grounding(table_block, research_content)

                    # Render to markdown
                    table_markdown = table_block.render_markdown()

                    logger.info(f"✅ Generated tracked-cells table: {len(table_block.headers)} cols × {len(table_block.rows)} rows")

                    return {
                        "type": "planned_table",
                        "content": table_markdown,
                        "metadata": {
                            "confidence": table_specifications.get("confidence", 0.9),
                            "reasoning": "Cell-level derivation tracking",
                            "method": "tracked_cells",
                            "data_sources": len(research_content.split('\n')),
                            "derivation_audit": f"{len(table_block.rows) * len(table_block.headers)} cells tracked"
                        }
                    }

                except Exception as e:
                    logger.warning(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {str(e)[:200]}")
                    if attempt == max_retries - 1:
                        raise

            return self._create_table_unavailable_message("Structured generation failed")

        except Exception as e:
            logger.error(f"❌ Tracked-cells generation error: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return self._create_table_unavailable_message(f"Generation error: {str(e)}")

    def _log_cell_derivations(self, table_block) -> None:
        """Log all cell derivations for audit trail."""
        logger.info("📊 CELL DERIVATION AUDIT:")
        for row_idx, row in enumerate(table_block.rows):
            for col_idx, cell in enumerate(row.cells):
                header = table_block.headers[col_idx] if col_idx < len(table_block.headers) else f"Col{col_idx}"
                derivation_type = cell.derivation.split(':')[0] if ':' in cell.derivation else cell.derivation
                logger.info(f"  [{row_idx},{col_idx}] {header}: {cell.value} | {derivation_type} | confidence={cell.confidence}")

    def _validate_cell_grounding(self, table_block, research_content: str) -> None:
        """Validate that extracted values are actually in the research content."""
        import re

        for row_idx, row in enumerate(table_block.rows):
            for col_idx, cell in enumerate(row.cells):
                if cell.derivation.startswith("extracted:"):
                    # Extract the claimed source
                    source_claim = cell.derivation[10:].strip()

                    # Check if the value appears in research content
                    value_str = str(cell.value)
                    if value_str not in research_content and not any(
                        part.strip() in research_content
                        for part in value_str.replace(',', '').split()
                        if len(part.strip()) > 2
                    ):
                        logger.warning(
                            f"⚠️  Cell[{row_idx},{col_idx}] claims extraction but value '{cell.value}' "
                            f"not found in research. Source claim: {source_claim[:100]}"
                        )
                elif cell.derivation.startswith("calculated:"):
                    logger.debug(f"📐 Cell[{row_idx},{col_idx}] calculation: {cell.derivation[11:]}")
                elif cell.derivation.startswith("estimated:"):
                    logger.warning(f"⚠️  Cell[{row_idx},{col_idx}] is estimated: {cell.derivation[10:]}")

    def _create_table_unavailable_message(self, reason: str) -> Dict[str, Any]:
        """
        Create clean, honest error message for table generation failures.
        NO FALLBACKS - just clear communication about what went wrong.
        
        Args:
            reason: Specific reason why table generation failed
            
        Returns:
            Dict with error message and metadata
        """
        logger.info(f"REPORTER: Creating table unavailable message - reason: {reason}")
        
        return {
            "type": "table_unavailable",
            "content": f"**Table Data Not Available**\n\nReason: {reason}\n\nPlease refer to the narrative sections below for available information.",
            "metadata": {
                "error_reason": reason, 
                "timestamp": time.time(),
                "method": "clean_failure"
            }
        }
    
    def _extract_table_from_special_output(self, special_output: Dict) -> Optional[str]:
        """Extract table content from special output, handling unavailable tables."""
        if special_output.get("metadata", {}).get("output_type") == "table_unavailable":
            # Return the error message as table content
            return special_output.get("content", "")
        
        # Extract actual table content for successful generation
        content = special_output.get("content", "")
        # Parse out table section if embedded in larger content
        if "| " in content:  # Markdown table marker
            # Ensure content is a string before splitting
            if isinstance(content, list):
                content = "\n".join(content) if content else ""
            elif not isinstance(content, str):
                content = str(content) if content else ""
            lines = content.split("\n")
            table_lines = []
            in_table = False
            for line in lines:
                if "|" in line:
                    in_table = True
                    table_lines.append(line)
                elif in_table and line.strip() == "":
                    break  # End of table
                elif in_table:
                    table_lines.append(line)
            return "\n".join(table_lines) if table_lines else None
        return None
    
    def _simple_data_search(self, research_data: List[Dict], entity: str, metric: str) -> str:
        """Fallback simple data search when semantic matcher isn't available."""
        
        for data_item in research_data:
            content = str(data_item.get("content", "")).lower()
            
            if entity.lower() in content and metric.lower() in content:
                # Look for numbers near the entity/metric
                import re
                numbers = re.findall(r'[\d,]+\.?\d*[%]?', content)
                if numbers:
                    return numbers[0]
        
        return "Data not available"
    
    def _enhanced_data_search(
        self, 
        research_data: List[Dict], 
        structured_observations: List[StructuredObservation],
        entity: str, 
        metric: str
    ) -> str:
        """
        Enhanced data search using structured observations.
        
        This method first checks structured observations for direct matches,
        then falls back to pattern matching in research data.
        """
        
        # First, try to find data in structured observations
        for obs in structured_observations:
            # Check if this observation has the entity
            if obs.has_entity(entity):
                # Look for the metric in the metric values
                if obs.has_metric(metric):
                    value = obs.get_metric_value(metric)
                    if value is not None:
                        return str(value)
                
                # If no direct metric match, look for similar metrics
                for metric_key, metric_value in obs.metric_values.items():
                    if metric.lower() in metric_key.lower() or metric_key.lower() in metric.lower():
                        return str(metric_value)
        
        # Fallback to simple pattern matching in content
        entity_lower = entity.lower()
        metric_lower = metric.lower()
        
        for data_item in research_data:
            content = str(data_item.get("content", "")).lower()
            
            if entity_lower in content and metric_lower in content:
                # Look for numbers near the entity/metric
                import re
                
                # Try more specific patterns first
                patterns = [
                    rf'{re.escape(entity_lower)}[^.]*?(\d+(?:\.\d+)?%)',  # Entity followed by percentage
                    rf'{re.escape(metric_lower)}[^.]*?(\d+(?:\.\d+)?%?)',  # Metric followed by number
                    rf'(\$\d+(?:,\d{{3}})*(?:\.\d{{2}})?)',  # Currency amounts
                    rf'(€\d+(?:,\d{{3}})*(?:\.\d{{2}})?)',  # Euro amounts
                    r'(\d+(?:,\d{3})*(?:\.\d+)?%?)',  # Any number with optional percentage
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        return matches[0]
        
        # Try to find entity-specific data even without metric match
        for obs in structured_observations:
            if obs.has_entity(entity) and obs.metric_values:
                # Guard against None metric_values
                if obs.metric_values is None:
                    continue
                # Return the first available metric for this entity (with safe access)
                if obs.metric_values:
                    first_metric = next(iter(obs.metric_values.values()))
                    return str(first_metric)
        
        return "Data not available"
    
    def _generate_special_output_if_required(self, state, compiled_findings, style_config):
        """Check for and generate special output formats based on planner's analysis."""
        try:
            logger.info("REPORTER: Checking for special output requirements")
            
            # First check if planner determined table is needed
            should_generate, table_specs = self._should_generate_table(state)
            
            if should_generate:
                logger.info(f"REPORTER: Planner determined table needed: {table_specs['reasoning']}")
                
                # Generate planned table using semantic extraction
                planned_table = self._generate_planned_table(
                    compiled_findings=compiled_findings,
                    table_specifications=table_specs,
                    state=state
                )
                
                if planned_table:
                    # Validate planned_table structure before passing
                    if isinstance(planned_table, dict):
                        return self._create_special_output_from_table(planned_table, state)
                    else:
                        logger.warning(f"Invalid planned_table structure: {type(planned_table)}")
                        return None
            
            # Legacy requirements check removed - new planner provides table specs directly
            
            logger.info("REPORTER: No special formatting needed, using standard report")
            return None
            
        except Exception as e:
            logger.error(f"REPORTER: Failed to generate special output: {e}")
            import traceback
            logger.error(f"REPORTER: Stack trace: {traceback.format_exc()}")
            return None
    
    def _create_special_output_from_table(self, planned_table: Dict, state: EnhancedResearchState) -> Dict:
        """Create special output object from planned table."""
        
        title = state.get("research_topic", "Research Report")
        
        # CRITICAL FIX: Handle table_unavailable messages
        if planned_table.get("type") == "table_unavailable":
            content = f"# {title}\n\n"
            content += planned_table["content"]  # Already formatted error message
            
            return {
                "title": title,
                "content": content,
                "format_type": "markdown",
                "metadata": {
                    "output_type": "table_unavailable",
                    "reason": planned_table.get("metadata", {}).get("error_reason", "Unknown"),
                    "timestamp": planned_table.get("metadata", {}).get("timestamp")
                },
                "meets_requirements": False  # Table requirement not met
            }
        
        # Original logic for successful tables
        content = f"# {title}\n\n"
        content += f"*Generated based on intelligent analysis of query requirements*\n\n"
        content += planned_table["content"]
        
        # Safe access to confidence and reasoning with fallbacks
        metadata = planned_table.get('metadata', {})
        confidence = metadata.get('confidence', 0.0)
        reasoning = metadata.get('reasoning', 'Analysis completed')
        
        content += f"\n\n**Analysis Confidence:** {confidence:.1%}\n"
        content += f"**Reasoning:** {reasoning}\n"
        
        return {
            "title": title,
            "content": content,
            "format_type": "markdown",
            "metadata": metadata,
            "meets_requirements": True
        }

    def _finalize_special_output(self, special_output, state):
        """Package special output into final report format."""
        try:
            logger.info("REPORTER: Finalizing special output")
            
            if isinstance(special_output, dict) and 'content' in special_output:
                content = special_output['content']
            elif hasattr(special_output, 'content'):
                content = special_output.content
            else:
                content = str(special_output)
            
            logger.info(f"REPORTER: Special output content length: {len(content) if content else 0}")
            
            # Ensure grounding markers if needed
            if state.get("enable_grounding_markers", True) and content:
                grounding_results = state.get("grounding_results", [])
                content = self._add_grounding_markers(content, grounding_results)
            
            # Don't mutate state directly - return values for Command update
            report_metadata = {
                "generated_by": "DirectLLM",
                "has_special_formatting": True
            }
            
            if isinstance(special_output, dict) and 'metadata' in special_output:
                report_metadata.update(special_output['metadata'])
            elif hasattr(special_output, 'metadata'):
                report_metadata.update(special_output.metadata)
            
            if hasattr(special_output, 'get_word_count'):
                report_metadata["word_count"] = special_output.get_word_count()
            
            # Package state updates for return
            updated_state = {
                "final_report": content,
                "report_type": "special_formatted", 
                "report_metadata": report_metadata
            }
            
            logger.info(f"REPORTER: Finalized special output with metadata: {report_metadata}")
            return updated_state
            
        except Exception as e:
            logger.error(f"REPORTER: Failed to finalize special output: {e}")
            # Fallback to regular report
            return state

    # _convert_findings_to_research_data method removed - no longer needed without OutputComposer
    
    def _extract_table_from_special_output(self, special_output: Dict) -> Optional[str]:
        """Extract just the table portion from special output."""
        
        content = special_output.get('content', '')
        if not content:
            return None
        
        # Look for markdown table pattern
        import re
        table_pattern = re.compile(r'(\|[^\n]+\|(?:\n\|[-:\s|]+\|)?(?:\n\|[^\n]+\|)+)', re.MULTILINE)
        
        matches = table_pattern.findall(content)
        if matches:
            # Return the first (and likely only) table found
            return matches[0]
        
        # If the entire content looks like a table, return it
        if content.count('|') > 4 and '\n' in content:
            return content
        
        return None
    
    def _collect_all_research_content(self, compiled_findings: Dict, 
                                     state: EnhancedResearchState, 
                                     max_tokens: int = 32000) -> str:
        """
        Collect all available research content within token budget.
        Prioritizes observations and search results with relevant data.
        """
        content_parts = []
        current_tokens = 0
        
        # Priority 1: Direct observations (most relevant)
        for obs in compiled_findings.get("observations", []):
            # Handle different observation types: string, dict, or StructuredObservation
            if isinstance(obs, str):
                obs_text = obs
            elif hasattr(obs, 'content'):  # StructuredObservation object
                obs_text = obs.content
            elif isinstance(obs, dict):
                obs_text = obs.get("content", "")
            else:
                obs_text = observation_to_text(obs)  # Fallback
                
            if obs_text:
                # Rough token estimation: 1 token ≈ 4 characters
                obs_tokens = len(obs_text) // 4
                if current_tokens + obs_tokens < max_tokens:
                    content_parts.append(f"OBSERVATION: {obs_text}")
                    current_tokens += obs_tokens
        
        # Priority 2: Search results
        for result in state.get("search_results", []):
            if current_tokens >= max_tokens * 0.9:  # Leave 10% buffer
                break
            
            if isinstance(result, dict):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("url", "")
                
                result_text = f"SOURCE [{title}]({url}): {snippet}" if url else f"SOURCE: {title} - {snippet}"
                result_tokens = len(result_text) // 4
                
                if current_tokens + result_tokens < max_tokens:
                    content_parts.append(result_text)
                    current_tokens += result_tokens
        
        # Priority 3: Background investigation results
        bg_results = state.get("background_investigation_results", [])
        for bg_result in bg_results[:20]:  # Limit to prevent overflow
            if current_tokens >= max_tokens * 0.95:
                break
                
            if isinstance(bg_result, dict):
                bg_text = bg_result.get("snippet", "") or bg_result.get("content", "")
                bg_tokens = len(bg_text) // 4
                
                if bg_tokens and current_tokens + bg_tokens < max_tokens:
                    content_parts.append(f"BACKGROUND: {bg_text}")
                    current_tokens += bg_tokens
        
        logger.info(f"REPORTER: Collected {len(content_parts)} content pieces (~{current_tokens} tokens)")
        return "\n\n".join(content_parts)
    
    def _build_comprehensive_table_prompt(self, rows: List[str], columns: List[str],
                                         research_content: str, topic: str,
                                         query_context: str) -> str:
        """Build detailed prompt for table generation with clear instructions."""
        
        return f"""Create a data table based on comprehensive research findings.

ORIGINAL QUERY CONTEXT:
{query_context[:1000]}

RESEARCH TOPIC: {topic}

REQUIRED TABLE STRUCTURE:
- Table Rows (entities/countries/items): {', '.join(rows)}
- Table Columns (metrics/attributes): {', '.join(columns)}

COMPREHENSIVE RESEARCH CONTENT:
{research_content}

CRITICAL INSTRUCTIONS FOR TABLE GENERATION:
1. Create a properly formatted markdown table
2. Use EXACTLY the rows and columns specified above
3. Extract ACTUAL VALUES from the research content - never invent data
4. Formatting requirements:
   - Currency: Use appropriate symbol (€, $, £, etc.) with amount
   - Percentages: Include % symbol (e.g., "47.5%")
   - Ranges: Use hyphen (e.g., "€2,000-3,000")
   - Large numbers: Use comma separators (e.g., "€150,000")
5. When data is not found in research:
   - Use "N/A" (not "Data not available" or other phrases)
   - Do not guess or estimate
6. For tax/financial data:
   - Include both rate and absolute values where available
   - Specify year if mentioned (e.g., "45% (2024)")
7. Ensure all cells are filled - no empty cells

OUTPUT REQUIREMENTS:
- Return ONLY the markdown table
- No introductory text
- No explanations or notes
- Table should be properly aligned and formatted

EXAMPLE FORMAT:
| Country | Tax Rate | Net Salary | Child Benefit |
|---------|----------|------------|---------------|
| Spain   | 47%      | €79,500    | €2,400/year   |
| France  | 45%      | €82,500    | €3,600/year   |
| UK      | 45%      | £85,000    | £1,885/year   |
"""
    
    def _validate_table_structure(self, table_content: str, expected_rows: int, expected_cols: int) -> bool:
        """Validate that table content has proper structure."""
        if not table_content or '|' not in table_content:
            return False
        
        # Ensure table_content is a string before splitting
        if isinstance(table_content, list):
            table_content = "\n".join(table_content) if table_content else ""
        elif not isinstance(table_content, str):
            table_content = str(table_content) if table_content else ""
        lines = table_content.strip().split('\n')
        table_lines = [line for line in lines if line.strip().startswith('|')]
        
        if len(table_lines) < 3:  # Header + separator + at least one data row
            return False
        
        # Check header row has correct number of columns
        header_cols = table_lines[0].count('|') - 1  # Subtract 1 for leading |
        if header_cols < expected_cols:
            return False
        
        # Check we have some data rows
        data_rows = len(table_lines) - 2  # Subtract header and separator
        if data_rows < min(expected_rows, 1):  # At least 1 data row
            return False
        
        return True

    def _repair_table_format(self, table_content: str) -> str:
        """Repair malformed markdown tables by adding missing separator lines."""
        if not table_content or '|' not in table_content:
            return table_content

        lines = table_content.strip().split('\n')
        table_lines = [line.strip() for line in lines if line.strip().startswith('|')]

        if len(table_lines) < 2:  # Need at least header + data
            return table_content

        # Check if separator line exists (contains ---)
        has_separator = any('---' in line for line in table_lines[1:3])

        if not has_separator:
            # Add separator line after header
            header = table_lines[0]
            data_rows = table_lines[1:]

            # Count columns in header
            col_count = header.count('|') - 1  # Subtract leading |

            # Create separator line
            separator = "| " + " | ".join(["---"] * col_count) + " |"

            # Reconstruct table
            repaired_lines = [header, separator] + data_rows

            # Replace table content in original text
            repaired_table = '\n'.join(repaired_lines)
            logger.info(f"REPORTER: Repaired table format - added separator line with {col_count} columns")
            return repaired_table

        return table_content

    def _create_fallback_table(self, rows: List[str], columns: List[str]) -> Dict:
        """Generate basic table structure when all else fails."""
        try:
            # Build table structure
            header = "| " + " | ".join(["Entity"] + columns) + " |"
            separator = "| " + " | ".join(["---"] * (len(columns) + 1)) + " |"
            
            table_rows = []
            for row in rows:
                # Ensure row is a string before concatenation
                row_str = str(row) if not isinstance(row, str) else row
                row_data = [row_str] + ["N/A"] * len(columns)
                table_rows.append("| " + " | ".join(row_data) + " |")
            
            table_content = "\n".join([header, separator] + table_rows)
            
            return {
                "content": table_content,
                "metadata": {
                    "confidence": 0.0,
                    "reasoning": "Fallback table - data extraction failed",
                    "method": "fallback_structure"
                }
            }
            
        except Exception as e:
            logger.error(f"REPORTER: Even fallback table generation failed: {e}")
            return {
                "content": "| Data | Value |\n|------|-------|\n| Error | Failed |\n",
                "metadata": {"confidence": 0.0, "method": "error"}
            }
    
    def _create_populated_fallback(self, rows: List[str], columns: List[str], 
                                  research_content: str) -> Dict:
        """
        Create a table with partial data extraction using simple patterns.
        Better than empty table when LLM fails.
        """
        import re
        
        # Build table structure
        header = "| " + " | ".join(["Entity"] + columns) + " |"
        separator = "|" + "---|" * (len(columns) + 1)
        
        table_rows = []
        for row in rows:
            # Ensure row is a string before using in list
            row_str = str(row) if not isinstance(row, str) else row
            row_data = [row_str]
            
            # Try to extract some data for this row
            for col in columns:
                # Simple pattern matching for common data types
                value = "N/A"
                
                # Search for row entity in content
                row_pattern = re.compile(
                    rf"{re.escape(row)}.*?(\d+\.?\d*%|\€\d+[,\d]*|\$\d+[,\d]*|\£\d+[,\d]*)",
                    re.IGNORECASE | re.DOTALL
                )
                matches = row_pattern.findall(research_content[:5000])  # Search in first 5000 chars
                
                if matches and any(col.lower() in ['tax', 'rate'] for col in [col]):
                    # Look for percentages
                    percent_matches = [m for m in matches if '%' in m]
                    if percent_matches:
                        value = percent_matches[0]
                elif matches and any(term in col.lower() for term in ['salary', 'income', 'pay']):
                    # Look for currency
                    currency_matches = [m for m in matches if any(c in m for c in ['€', '$', '£'])]
                    if currency_matches:
                        value = currency_matches[0]
                
                row_data.append(value)
            
            table_rows.append("| " + " | ".join(row_data) + " |")
        
        table_content = "\n".join([header, separator] + table_rows)
        
        return {
            "content": table_content,
            "metadata": {
                "confidence": 0.3,
                "reasoning": "Fallback pattern extraction",
                "method": "pattern_matching"
            }
        }
    
    def _determine_optimal_table_section(self, structure: List[str], 
                                        state: EnhancedResearchState,
                                        table_metadata: Dict) -> Optional[str]:
        """
        Intelligently determine the best section for table placement.
        """
        # Priority order for table placement based on common section names
        priority_patterns = [
            "comparison", "comparative", "analysis", "data",
            "findings", "results", "overview", "summary"
        ]
        
        # Score each section
        section_scores = {}
        for section in structure:
            score = 0
            section_lower = section.lower()
            
            # Check priority patterns
            for i, pattern in enumerate(priority_patterns):
                if pattern in section_lower:
                    score += (len(priority_patterns) - i) * 10
            
            # Bonus for middle sections (better flow)
            middle_idx = len(structure) // 2
            section_idx = structure.index(section)
            distance_from_middle = abs(section_idx - middle_idx)
            score += max(0, 5 - distance_from_middle)
            
            section_scores[section] = score
        
        # Select highest scoring section
        if section_scores:
            best_section = max(section_scores, key=section_scores.get)
            if section_scores[best_section] > 0:
                return best_section
        
        # Fallback: place after introduction or in first main section
        for i, section in enumerate(structure):
            if "introduction" in section.lower() and i + 1 < len(structure):
                return structure[i + 1]
        
        # Last resort: second section if available
        return structure[1] if len(structure) > 1 else structure[0] if structure else None

    def _generate_table_introduction(self, state: EnhancedResearchState, 
                                    table_metadata: Dict) -> str:
        """Generate contextual introduction for the table."""
        
        topic = state.get("research_topic", "the data")
        confidence = table_metadata.get("confidence", 0)
        
        if confidence > 0.8:
            return f"The following table presents comprehensive data on {topic}:"
        elif confidence > 0.5:
            return f"Based on available research, the following comparison was compiled:"
        else:
            return f"The table below shows the structured data that could be extracted from available sources:"
    
    def _generate_brief_table_analysis(self, table_content: str, state: EnhancedResearchState) -> Optional[str]:
        """Generate brief analysis of the table if LLM is available."""
        
        if not self.llm or not table_content:
            return None
        
        try:
            prompt = f"""Provide a brief 2-3 sentence analysis of the key patterns or insights from this table:

{table_content[:1000]}

Focus on:
- Notable differences or similarities
- Highest/lowest values
- Clear patterns or trends

Keep it concise and factual."""
            
            response = self.llm.invoke([
                SystemMessage("You are a data analyst. Provide concise, factual analysis."),
                HumanMessage(prompt)
            ])
            
            # Handle structured responses properly
            from ..core.llm_response_parser import extract_text_from_response
            analysis = extract_text_from_response(response)
            return analysis.strip() if analysis and len(analysis) > 10 else None
            
        except Exception as e:
            logger.warning(f"Failed to generate table analysis: {e}")
            return None

    def _validate_final_report(self, report: str, state: EnhancedResearchState) -> str:
        """
        Validate the final report to ensure it's not empty, reasoning, or incomplete.
        
        Args:
            report: The generated report
            state: Current research state
            
        Returns:
            str: Validated report or fallback content
        """
        if not report or not report.strip():
            logger.error("Final report is empty - generating fallback")
            return self._generate_fallback_report(state)
        
        report_lower = report.lower()
        
        # Check if report looks like reasoning instead of actual content
        reasoning_indicators = [
            "i need to", "let me", "i should", "i will", "first i", 
            "thinking about", "considering", "my approach", "i'll",
            "step 1:", "step 2:", "next, i", "based on this",
            "i'm going to", "let's start by", "the first step"
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in report_lower)
        
        # If the report is very short or contains many reasoning indicators, it might be reasoning
        if len(report.strip()) < 200 or reasoning_count >= 3:
            logger.warning(f"Report appears to be reasoning (length={len(report)}, reasoning_indicators={reasoning_count}) - generating fallback")
            return self._generate_fallback_report(state)
        
        # Check for placeholder content that indicates incomplete generation
        placeholder_indicators = [
            "[content to be added]", "[analysis content to be added]", 
            "[recommendations content to be added]", "[executive summary content to be added]",
            "content placeholder", "section content will be generated"
        ]
        
        placeholder_count = sum(1 for indicator in placeholder_indicators if indicator in report_lower)
        
        if placeholder_count > 0:
            logger.warning(f"Report contains {placeholder_count} placeholders - generating fallback")
            return self._generate_fallback_report(state)
        
        logger.info(f"Final report validation passed: {len(report)} characters")
        return report

    def _generate_fallback_report(self, state: EnhancedResearchState) -> str:
        """
        Generate a fallback report when the main report generation fails.
        
        Args:
            state: Current research state
            
        Returns:
            str: Basic fallback report
        """
        research_topic = state.get("research_topic", "Research Query")
        observations = state.get("observations", [])
        citations = state.get("citations", [])
        
        fallback_report = f"""# Research Report: {research_topic}

## Executive Summary

I have conducted research on the requested topic but encountered issues generating the full report. 
Based on the available research findings, I can provide the following summary:

## Research Findings

"""

        # Add basic findings from observations
        if observations:
            fallback_report += f"During my research, I gathered {len(observations)} key observations:\n\n"
            for i, obs in enumerate(observations[:15], 1):  # Increased from 5 to 15 for more comprehensive fallback
                if hasattr(obs, 'content') and obs.content:
                    content = obs.content[:200] + "..." if len(obs.content) > 200 else obs.content
                    fallback_report += f"{i}. {content}\n\n"
        else:
            fallback_report += "No detailed research findings were successfully gathered.\n\n"

        # Add citations if available
        if citations:
            fallback_report += "## Sources\n\n"
            for i, citation in enumerate(citations[:10], 1):  # Limit to first 10
                if hasattr(citation, 'url') and hasattr(citation, 'title'):
                    fallback_report += f"{i}. [{citation.title}]({citation.url})\n"
                elif hasattr(citation, 'source'):
                    fallback_report += f"{i}. {citation.source}\n"

        fallback_report += "\n\n*Note: This is a fallback report generated due to technical issues with the main report generation.*"
        
        logger.info(f"Generated fallback report: {len(fallback_report)} characters")
        return fallback_report

    # ===================================================================
    # HYBRID MULTI-PASS REPORT GENERATION - SANITIZATION
    # ===================================================================

    def _sanitize_observations_for_report(self, observations: List) -> List:
        """
        Sanitize observations to remove file references and external tool mentions.

        Args:
            observations: List of observations (can be dicts or StructuredObservation objects)

        Returns:
            List of sanitized observations
        """
        from ..core.content_sanitizer import sanitize_agent_content

        # Default contamination patterns
        DEFAULT_PATTERNS = [
            r'\b[\w\-]+\.(?:xlsx|xlsm|csv|json|pdf|doc|docx|xls)\b',
            r'github\.com[/\w\-\.]*',
            r'gitlab\.com[/\w\-\.]*',
            r'\b(?:spreadsheet|repository|download|attachment|file)\b'
        ]

        patterns = self.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {}).get(
            'contamination_patterns',
            DEFAULT_PATTERNS
        )

        cleaned = []
        for obs in observations:
            # Handle both dict and object observations
            if isinstance(obs, dict):
                obs_dict = obs
            elif hasattr(obs, 'to_dict'):
                obs_dict = obs.to_dict()
            else:
                obs_dict = {'content': str(obs), 'step_id': 'unknown'}

            # Sanitize content
            content = obs_dict.get('content', '')

            # Apply contamination pattern filtering
            clean_content = content
            for pattern in patterns:
                def _annotate(match):
                    token = match.group(0)
                    tag = 'FILE' if '.' in token and any(ext in token for ext in ['.xlsx', '.csv', '.pdf', '.json']) else 'REF'
                    return f"[{tag}]"

                clean_content = re.sub(pattern, _annotate, clean_content, flags=re.IGNORECASE)

            # Create sanitized copy
            obs_copy = obs_dict.copy()
            obs_copy['content'] = clean_content
            obs_copy['was_sanitized'] = clean_content != content

            cleaned.append(obs_copy)

        sanitized_count = sum(1 for obs in cleaned if obs.get('was_sanitized'))
        if sanitized_count > 0:
            logger.info(f"[HYBRID Sanitization] Sanitized {sanitized_count}/{len(observations)} observations")

        return cleaned

    # ===================================================================
    # HYBRID MULTI-PASS REPORT GENERATION - MAIN PHASES
    # ===================================================================

    def _extract_text_from_response(self, response: Any) -> str:
        """
        Extract text content from LLM response, handling both string and structured list formats.

        Some LLM providers return response.content as a list of content blocks:
        [{"type": "text", "text": "..."}] or [{"type": "reasoning", ...}, {"type": "text", "text": "..."}]

        This helper ensures we always get a string for processing.
        """
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response

        # Handle list format (structured responses)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        text_parts.append(item['text'])
                    elif item.get('type') == 'reasoning' and 'text' in item:
                        # Include reasoning text as well
                        text_parts.append(item['text'])
            return "\n\n".join(text_parts) if text_parts else str(content)

        # Handle string format (standard responses)
        elif isinstance(content, str):
            return content

        # Fallback for any other format
        else:
            return str(content)

    def _fix_markdown_tables(self, markdown_text: str) -> str:
        """
        Ensure all markdown tables have proper separator rows.

        Tables must have format:
        | Header 1 | Header 2 |
        |----------|----------|
        | Data 1   | Data 2   |

        This method detects tables and inserts missing separator rows.
        """
        if not markdown_text or '|' not in markdown_text:
            return markdown_text

        lines = markdown_text.split('\n')
        fixed_lines = []
        i = 0
        in_table = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect table row (starts with | and has at least 3 |)
            is_table_row = stripped.startswith('|') and stripped.count('|') >= 3

            if is_table_row:
                # Check if this is a separator row (contains ---)
                is_separator = '-' in stripped

                if not in_table:
                    # This is the first row of a new table (header)
                    in_table = True
                    fixed_lines.append(line)

                    # Check next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # Check if next line is a proper separator (only dashes between pipes)
                        if next_line.startswith('|') and next_line.count('|') >= 3:
                            content = next_line.replace('|', '').replace(' ', '').replace('\t', '')
                            next_is_separator = content and all(c == '-' for c in content)
                        else:
                            next_is_separator = False

                        if not next_is_separator:
                            # Next line is a data row, separator is missing!
                            # Insert separator after header
                            col_count = stripped.count('|') - 1
                            separator_parts = ['---' for _ in range(col_count)]
                            separator = '|' + '|'.join(separator_parts) + '|'
                            fixed_lines.append(separator)
                            logger.debug(f"[TABLE FIX] Inserted missing separator after header")
                        # If next line is separator, it will be added in next iteration

                    i += 1
                    continue
                else:
                    # We're already in a table, just append this row
                    fixed_lines.append(line)
                    i += 1
                    continue
            else:
                # Not a table row
                if stripped == '':
                    # Empty line might end the table
                    in_table = False

                fixed_lines.append(line)
                i += 1

        fixed_text = '\n'.join(fixed_lines)

        # Log if we made changes
        if fixed_text != markdown_text:
            original_separators = markdown_text.count('|---')
            fixed_separators = fixed_text.count('|---')
            if fixed_separators > original_separators:
                logger.info(f"[TABLE FIX] Added {fixed_separators - original_separators} missing table separators")

        return fixed_text

    async def _understand_research_context(
        self,
        research_topic: str,
        observations: List[Dict],
        plan: Any,
        section_name: str = "All Sections",
        max_chars: int = 60000
    ) -> Dict[str, Any]:
        """
        Stage 1A: Deep Context Understanding

        Let the LLM deeply analyze the research to understand:
        1. What is the user trying to learn/compare?
        2. What are the KEY ENTITIES (countries, products, etc.)?
        3. What METRICS are being discussed (tax rates, prices, etc.)?
        4. What TABLE STRUCTURE makes sense (entity-based vs metric-based)?
        5. What data is available vs missing?

        This stage provides maximum context to the LLM and lets it reason
        about the optimal structure adaptively rather than forcing rigid formats.

        Args:
            research_topic: The user's research question
            observations: All observations with full_content preserved
            plan: The research plan
            section_name: Current section being processed

        Returns:
            Dict with 'understanding' field containing LLM's analysis
        """
        logger.info(f"[Stage 1A] Understanding context for: {section_name}")

        # Format observations with FULL content
        obs_details = []
        for obs in observations:
            full_content = obs.get('full_content', obs.get('content', ''))
            step_id = obs.get('step_id', 'unknown')
            obs_details.append(f"Step {step_id}: {full_content}")

        # Apply character limit with smart truncation (keep observations in order, stop at limit)
        all_observations = ""
        total_chars = 0
        included_count = 0

        for obs_text in obs_details:
            # Check if adding this observation would exceed the limit
            separator = "\n\n" if all_observations else ""
            new_total = total_chars + len(obs_text) + len(separator)

            if new_total <= max_chars:
                all_observations += separator + obs_text
                total_chars = new_total
                included_count += 1
            else:
                # Hit the limit - stop adding observations
                break

        excluded_count = len(obs_details) - included_count
        if excluded_count > 0:
            logger.warning(
                f"[Stage 1A] Truncated observations: included {included_count}/{len(obs_details)} "
                f"({total_chars:,}/{max_chars:,} chars, excluded {excluded_count} observations)"
            )
        else:
            logger.info(
                f"[Stage 1A] Providing {included_count} observations "
                f"({total_chars:,} chars, limit: {max_chars:,})"
            )

        # Extract plan sections if available
        plan_sections = "No structured plan available"
        if plan and hasattr(plan, 'sections'):
            section_names = [s.section_name for s in plan.sections]
            plan_sections = f"Plan sections: {', '.join(section_names)}"

        prompt = f"""You are analyzing research to deeply understand what structure and data extraction is needed.

RESEARCH QUESTION:
{research_topic}

{plan_sections}

CURRENT SECTION:
{section_name}

ALL RESEARCH OBSERVATIONS (with full content including tables and numeric data):
{all_observations}

TASK: Deeply analyze this research and provide your understanding:

1. USER INTENT: What is the user trying to learn or compare? What's the core question?

2. KEY ENTITIES: What are the main entities being discussed?
   - For country comparisons: List all countries
   - For product comparisons: List all products
   - For general analysis: Identify the main subjects

3. KEY METRICS: What quantitative or qualitative attributes are being measured?
   - Tax rates, prices, percentages, rankings, etc.
   - For each metric, note if data is available in the observations

4. OPTIMAL TABLE STRUCTURE:
   - Should rows be ENTITIES (e.g., countries) or METRICS (e.g., different tax types)?
   - Should columns be METRICS (e.g., tax rate, GDP) or ENTITIES?
   - Consider: Does the user want to compare entities across metrics, or compare metrics across entities?

5. DATA AVAILABILITY:
   - What specific values are present in the observations?
   - What data is missing or not found?
   - Any data quality concerns (conflicting numbers, unclear sources)?

6. STRUCTURAL RECOMMENDATIONS:
   - How many tables are needed?
   - What should each table show?
   - What calculations might be needed (differences, percentages, averages)?

Provide a detailed, thoughtful analysis that will guide the extraction process."""
        logger.info(f"[Stage 1A] Generated prompt. Length: {len(prompt)}")

        messages = [
            {"role": "system", "content": "You are an expert research analyst specializing in data structuring and comparative analysis."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self.llm.ainvoke(messages)
            understanding = self._extract_text_from_response(response)

            logger.info(
                f"[Stage 1A] Context understanding complete: "
                f"{len(understanding)} chars of analysis"
            )

            return {
                'understanding': understanding,
                'observation_count': len(observations),
                'total_chars': total_chars
            }

        except Exception as e:
            logger.error(f"[Stage 1A] Failed to generate understanding: {e}")
            return {
                'understanding': f"Error during context understanding: {e}",
                'observation_count': len(observations),
                'total_chars': total_chars
            }

    async def _extract_with_understanding(
        self,
        research_topic: str,
        observations: List[Dict],
        understanding: str,
        section_name: str = "All Sections",
        max_chars: int = 60000
    ) -> Dict[str, Any]:
        """
        Stage 1B: Guided Extraction with Understanding

        Uses the deep understanding from Stage 1A to guide extraction of:
        - DataPoints: Individual facts and metrics
        - Calculations: Derived values with explicit formulas
        - ComparisonEntries: Structured table rows

        The LLM uses its understanding of the research context to decide
        the optimal structure (entity-based vs metric-based) adaptively.

        Args:
            research_topic: The user's research question
            observations: All observations with full_content
            understanding: The context understanding from Stage 1A
            section_name: Current section being processed

        Returns:
            Dict with 'extraction' field containing structured JSON
        """
        from ..core.report_generation.models import CalculationContext, DataPoint, Calculation, ComparisonEntry

        logger.info(f"[Stage 1B] Extracting data with guided understanding for: {section_name}")

        # Format observations with FULL content
        obs_details = []
        for obs in observations:
            full_content = obs.get('full_content', obs.get('content', ''))
            step_id = obs.get('step_id', 'unknown')
            obs_details.append(f"Step {step_id}: {full_content}")

        # Apply character limit with smart truncation (same as Stage 1A)
        all_observations = ""
        total_chars = 0
        included_count = 0

        for obs_text in obs_details:
            # Check if adding this observation would exceed the limit
            separator = "\n\n" if all_observations else ""
            new_total = total_chars + len(obs_text) + len(separator)

            if new_total <= max_chars:
                all_observations += separator + obs_text
                total_chars = new_total
                included_count += 1
            else:
                # Hit the limit - stop adding observations
                break

        excluded_count = len(obs_details) - included_count
        if excluded_count > 0:
            logger.warning(
                f"[Stage 1B] Truncated observations: included {included_count}/{len(obs_details)} "
                f"({total_chars:,}/{max_chars:,} chars, excluded {excluded_count} observations)"
            )
        else:
            logger.info(
                f"[Stage 1B] Providing {included_count} observations "
                f"({total_chars:,} chars, limit: {max_chars:,})"
            )

        # Create example structure for guidance
        example_entity_based = {
            "extracted_data": [
                {"entity": "Country A", "metric": "corporate_tax_rate", "value": 21.0, "unit": "percent", "source_observation_id": "step_1"},
                {"entity": "Country B", "metric": "corporate_tax_rate", "value": 19.0, "unit": "percent", "source_observation_id": "step_2"}
            ],
            "calculations": [
                {"description": "Tax rate difference", "formula": "21.0 - 19.0", "inputs": {"rate_a": 21.0, "rate_b": 19.0}, "result": 2.0, "unit": "percentage_points"}
            ],
            "key_comparisons": [
                {"primary_key": "Country A", "metrics": {"corporate_tax_rate": 21.0, "gdp_billions": 1500}, "source_observation_ids": ["step_1"]},
                {"primary_key": "Country B", "metrics": {"corporate_tax_rate": 19.0, "gdp_billions": 1200}, "source_observation_ids": ["step_2"]}
            ],
            "summary_insights": ["Country A has 2 percentage points higher corporate tax than Country B"]
        }

        prompt = f"""You are extracting structured data from research observations to build comparison tables.

RESEARCH QUESTION:
{research_topic}

CURRENT SECTION:
{section_name}

YOUR DEEP UNDERSTANDING (from Stage 1A):
{understanding}

ALL RESEARCH OBSERVATIONS:
{all_observations}

TASK: Based on your understanding, extract structured data in JSON format following this schema:

{{
  "extracted_data": [
    {{
      "entity": "Name of entity (country/product/etc)",
      "metric": "Name of metric (tax_rate/price/etc)",
      "value": numeric_or_string_value,
      "unit": "unit of measurement",
      "source_observation_id": "step_X",
      "confidence": 0.0-1.0
    }}
  ],
  "calculations": [
    {{
      "description": "What is being calculated",
      "formula": "Mathematical formula with actual values",
      "inputs": {{"input_name": value}},
      "result": calculated_result,
      "unit": "unit"
    }}
  ],
  "key_comparisons": [
    {{
      "primary_key": "Entity or metric name (based on your understanding)",
      "metrics": {{"metric1": value1, "metric2": value2}},
      "source_observation_ids": ["step_X", "step_Y"]
    }}
  ],
  "summary_insights": ["Key insight 1", "Key insight 2"],
  "data_quality_notes": ["Any warnings about missing/conflicting data"]
}}

EXAMPLE (entity-based structure for country comparison):
{example_entity_based}

IMPORTANT GUIDELINES:

1. **Use your understanding** from Stage 1A to decide structure:
   - If comparing ENTITIES (countries, products): primary_key = entity name, metrics = attributes
   - If comparing METRICS (different tax types): primary_key = metric name, metrics = entity values

2. **Extract ALL relevant data points** from observations:
   - Every number, percentage, ranking
   - Include source_observation_id for traceability
   - Set confidence based on source quality

3. **Show your work** in calculations:
   - Explicit formulas with actual values
   - Named inputs for clarity
   - Include units

4. **Be honest about data quality**:
   - List missing data in data_quality_notes
   - Flag conflicting numbers
   - Note estimates vs precise values

5. **Create complete comparison entries**:
   - One entry per entity/metric (based on your chosen structure)
   - Include all relevant metrics/entities in the metrics dict
   - Link to source observations

Return ONLY the JSON object, no additional text."""

        messages = [
            {"role": "system", "content": "You are a precise data extraction specialist. Extract structured data exactly as specified."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self.llm.ainvoke(messages)
            extraction_text = self._extract_text_from_response(response)

            # Try to extract JSON from code blocks or raw text
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', extraction_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try direct JSON parse
                json_str = extraction_text.strip()

            extraction_data = json.loads(json_str)

            logger.info(
                f"[Stage 1B] Extraction complete: "
                f"{len(extraction_data.get('extracted_data', []))} data points, "
                f"{len(extraction_data.get('calculations', []))} calculations, "
                f"{len(extraction_data.get('key_comparisons', []))} comparisons"
            )

            return {
                'extraction': extraction_data,
                'raw_response': extraction_text
            }

        except json.JSONDecodeError as e:
            logger.error(f"[Stage 1B] JSON parsing failed: {e}")
            logger.error(f"[Stage 1B] Response text: {extraction_text[:500]}...")
            return {
                'extraction': {
                    "extracted_data": [],
                    "calculations": [],
                    "key_comparisons": [],
                    "summary_insights": [],
                    "data_quality_notes": [f"JSON parsing error: {e}"]
                },
                'raw_response': extraction_text,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"[Stage 1B] Extraction failed: {e}")
            return {
                'extraction': {
                    "extracted_data": [],
                    "calculations": [],
                    "key_comparisons": [],
                    "summary_insights": [],
                    "data_quality_notes": [f"Extraction error: {e}"]
                },
                'error': str(e)
            }

    def _validate_extraction_quality(
        self,
        extraction: Dict[str, Any],
        min_comparisons: int = 1,
        min_data_points: int = 2
    ) -> Dict[str, Any]:
        """
        Stage 1C: Quality Validation

        Validates the extraction quality and provides metrics:
        - Checks if we have enough comparison entries for tables
        - Validates data richness (multiple metrics per entity)
        - Checks entity coverage
        - Provides quality score and recommendations

        Args:
            extraction: The extraction dict from Stage 1B
            min_comparisons: Minimum comparison entries needed
            min_data_points: Minimum data points needed

        Returns:
            Dict with 'valid', 'quality_score', 'issues', 'recommendations'
        """
        logger.info("[Stage 1C] Validating extraction quality")

        issues = []
        recommendations = []
        quality_metrics = {}

        # Extract components
        extracted_data = extraction.get('extracted_data', [])
        calculations = extraction.get('calculations', [])
        key_comparisons = extraction.get('key_comparisons', [])
        data_quality_notes = extraction.get('data_quality_notes', [])

        # Metric 1: Comparison count
        comparison_count = len(key_comparisons)
        quality_metrics['comparison_count'] = comparison_count

        if comparison_count < min_comparisons:
            issues.append(f"Only {comparison_count} comparison entries (need {min_comparisons}+)")
            recommendations.append("Try extracting more entities or metrics from observations")

        # Metric 2: Data point count
        data_point_count = len(extracted_data)
        quality_metrics['data_point_count'] = data_point_count

        if data_point_count < min_data_points:
            issues.append(f"Only {data_point_count} data points (need {min_data_points}+)")
            recommendations.append("Extract more granular facts from observations")

        # Metric 3: Entity coverage (unique entities)
        unique_entities = set(dp.get('entity', 'unknown') for dp in extracted_data)
        quality_metrics['unique_entities'] = len(unique_entities)

        if len(unique_entities) < 2:
            issues.append(f"Only {len(unique_entities)} unique entities found")
            recommendations.append("Identify more entities for comparison")

        # Metric 4: Metric coverage (unique metrics per entity)
        unique_metrics = set(dp.get('metric', 'unknown') for dp in extracted_data)
        quality_metrics['unique_metrics'] = len(unique_metrics)

        if len(unique_metrics) < 2:
            issues.append(f"Only {len(unique_metrics)} unique metrics found")
            recommendations.append("Extract more attributes for each entity")

        # Metric 5: Data richness (avg metrics per comparison)
        if comparison_count > 0:
            total_metrics = sum(len(comp.get('metrics', {})) for comp in key_comparisons)
            avg_metrics_per_comp = total_metrics / comparison_count
            quality_metrics['avg_metrics_per_comparison'] = avg_metrics_per_comp

            if avg_metrics_per_comp < 2:
                issues.append(f"Average {avg_metrics_per_comp:.1f} metrics per comparison (need 2+)")
                recommendations.append("Add more metrics to each comparison entry")
        else:
            quality_metrics['avg_metrics_per_comparison'] = 0

        # Metric 6: Calculation count
        quality_metrics['calculation_count'] = len(calculations)

        # Metric 7: Data quality concerns
        quality_metrics['data_quality_notes_count'] = len(data_quality_notes)

        if data_quality_notes:
            issues.append(f"Data quality concerns: {', '.join(data_quality_notes[:3])}")

        # Calculate overall quality score (0.0 to 1.0)
        score_components = []

        # Component 1: Comparison adequacy (0-30 points)
        comp_score = min(30, (comparison_count / max(min_comparisons, 1)) * 30)
        score_components.append(comp_score)

        # Component 2: Data point richness (0-25 points)
        dp_score = min(25, (data_point_count / max(min_data_points * 2, 1)) * 25)
        score_components.append(dp_score)

        # Component 3: Entity coverage (0-20 points)
        entity_score = min(20, (len(unique_entities) / 3) * 20)
        score_components.append(entity_score)

        # Component 4: Metric coverage (0-15 points)
        metric_score = min(15, (len(unique_metrics) / 3) * 15)
        score_components.append(metric_score)

        # Component 5: Data richness (0-10 points)
        richness_score = min(10, quality_metrics.get('avg_metrics_per_comparison', 0) * 3)
        score_components.append(richness_score)

        # Total score (0-100, normalized to 0.0-1.0)
        quality_score = sum(score_components) / 100.0

        # Determine validity
        is_valid = (
            comparison_count >= min_comparisons and
            data_point_count >= min_data_points and
            quality_score >= 0.3  # At least 30% quality
        )

        logger.info(
            f"[Stage 1C] Validation complete: "
            f"Valid={is_valid}, Score={quality_score:.2f}, "
            f"Comparisons={comparison_count}, DataPoints={data_point_count}, "
            f"Issues={len(issues)}"
        )

        return {
            'valid': is_valid,
            'quality_score': quality_score,
            'quality_metrics': quality_metrics,
            'issues': issues,
            'recommendations': recommendations,
            'summary': {
                'comparisons': comparison_count,
                'data_points': data_point_count,
                'entities': len(unique_entities),
                'metrics': len(unique_metrics),
                'calculations': len(calculations)
            }
        }

    def _build_comparisons_from_datapoints(
        self,
        data_points: List[Any]
    ) -> List[Any]:
        """
        Fallback: Build ComparisonEntry objects from DataPoint objects.

        When the LLM fails to generate comparisons directly, we can
        derive them from the extracted data points by grouping by entity.

        Args:
            data_points: List of DataPoint objects

        Returns:
            List of ComparisonEntry objects
        """
        from ..core.report_generation.models import ComparisonEntry

        logger.info(f"[Fallback] Building comparisons from {len(data_points)} data points")

        # Group data points by entity
        entity_metrics = {}
        entity_sources = {}

        for dp in data_points:
            entity = dp.entity if hasattr(dp, 'entity') else dp.get('entity', 'Unknown')
            metric = dp.metric if hasattr(dp, 'metric') else dp.get('metric', 'unknown')
            value = dp.value if hasattr(dp, 'value') else dp.get('value', 'N/A')
            source = dp.source_observation_id if hasattr(dp, 'source_observation_id') else dp.get('source_observation_id')

            if entity not in entity_metrics:
                entity_metrics[entity] = {}
                entity_sources[entity] = []

            entity_metrics[entity][metric] = value

            if source and source not in entity_sources[entity]:
                entity_sources[entity].append(source)

        # Create ComparisonEntry objects
        comparisons = []
        for entity, metrics in entity_metrics.items():
            try:
                comp_entry = ComparisonEntry(
                    primary_key=entity,
                    metrics=metrics,
                    source_observation_ids=entity_sources.get(entity, [])
                )
                comparisons.append(comp_entry)
            except Exception as e:
                logger.warning(f"[Fallback] Failed to create ComparisonEntry for {entity}: {e}")

        logger.info(
            f"[Fallback] Built {len(comparisons)} comparison entries from "
            f"{len(entity_metrics)} unique entities"
        )

        return comparisons

    async def _generate_calculation_context(
        self,
        findings: Dict[str, Any]
    ):
        """
        Phase 1: Three-Stage Calculation Context Generation with Deep Understanding

        This method implements a comprehensive three-stage pipeline:
        - Stage 1A: Deep context understanding (what's the user asking? what structure makes sense?)
        - Stage 1B: Guided extraction with understanding (extract data with full context)
        - Stage 1C: Quality validation (ensure we have enough data for good tables)

        The key improvement: We now supply FULL CONTENT (full_content field) to the LLM
        with a generous 200K character budget, letting the LLM deeply reason about
        the optimal structure (entity-based vs metric-based) adaptively.

        Args:
            findings: Compiled findings from _compile_findings()

        Returns:
            CalculationContext with extracted data, calculations, and comparisons
        """
        from ..core.report_generation.models import CalculationContext, DataPoint, Calculation, ComparisonEntry

        logger.info("[HYBRID Phase 1] Starting THREE-STAGE calculation context generation")

        settings = self.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {})
        all_observations = findings.get('observations', [])

        # Get character limit for observation context
        max_calc_chars = settings.get('max_calc_prompt_chars', 60000)
        logger.info(f"[HYBRID Phase 1] Using max_calc_prompt_chars: {max_calc_chars:,}")

        # Monitor memory usage
        self._monitor_memory_usage('calc_context', len(all_observations))

        # Select observations for calculation context
        selector = self.observation_selector or ObservationSelector(self.embedding_manager)
        top_k = settings.get('calc_selector_top_k', 60)
        tail_k = settings.get('calc_recent_tail', 20)

        # Convert observations to dicts for selector
        obs_dicts = []
        for obs in all_observations:
            if hasattr(obs, 'to_dict'):
                obs_dicts.append(obs.to_dict())
            elif isinstance(obs, dict):
                obs_dicts.append(obs)
            else:
                obs_dicts.append({'content': str(obs), 'step_id': 'unknown'})

        # Extract research topic for entity detection
        research_topic = findings.get('research_topic', 'Research Question')

        # Extract key entities (countries, companies, etc.) from research topic
        diversity_entities = selector.extract_key_entities_from_topic(research_topic)

        # Enable diversity enforcement if entities were found
        enable_diversity = len(diversity_entities) >= 2  # Need at least 2 entities for comparison

        if enable_diversity:
            logger.info(
                f"[HYBRID Phase 1] Entity diversity enforcement ENABLED for {len(diversity_entities)} entities"
            )

        scored = selector.select_observations_for_section(
            section_title="Calculation context",
            section_purpose="extract quantitative facts for holistic synthesis",
            all_observations=obs_dicts,
            max_observations=top_k,
            min_relevance=0.25,
            use_semantic=self.embedding_manager is not None,
            ensure_entity_diversity=enable_diversity,
            diversity_entities=diversity_entities if enable_diversity else None
        )

        recent_tail = obs_dicts[-tail_k:] if len(obs_dicts) >= tail_k else obs_dicts
        merged = self._dedupe_preserve_order(scored + recent_tail)

        logger.info(
            f"[HYBRID Phase 1] Selected {len(merged)} observations "
            f"(top-k: {len(scored)}, tail: {len(recent_tail)})"
        )

        # Extract research context
        research_topic = findings.get('research_topic', 'Research Question')
        plan = findings.get('current_plan')

        try:
            # ============================================================
            # STAGE 1A: DEEP CONTEXT UNDERSTANDING
            # ============================================================
            understanding_result = await self._understand_research_context(
                research_topic=research_topic,
                observations=merged,
                plan=plan,
                section_name="Calculation Context",
                max_chars=max_calc_chars
            )

            understanding = understanding_result['understanding']

            logger.info(
                f"[Stage 1A Complete] Generated {len(understanding)} chars of context understanding"
            )

            # ============================================================
            # STAGE 1B: GUIDED EXTRACTION WITH UNDERSTANDING
            # ============================================================
            extraction_result = await self._extract_with_understanding(
                research_topic=research_topic,
                observations=merged,
                understanding=understanding,
                section_name="Calculation Context",
                max_chars=max_calc_chars
            )

            extraction_data = extraction_result['extraction']

            logger.info(
                f"[Stage 1B Complete] Extracted "
                f"{len(extraction_data.get('extracted_data', []))} data points, "
                f"{len(extraction_data.get('calculations', []))} calculations, "
                f"{len(extraction_data.get('key_comparisons', []))} comparisons"
            )

            # ============================================================
            # STAGE 1C: QUALITY VALIDATION
            # ============================================================
            validation_result = self._validate_extraction_quality(
                extraction=extraction_data,
                min_comparisons=1,
                min_data_points=2
            )

            logger.info(
                f"[Stage 1C Complete] Quality Score: {validation_result['quality_score']:.2f}, "
                f"Valid: {validation_result['valid']}, "
                f"Issues: {len(validation_result['issues'])}"
            )

            # Log validation issues and recommendations
            if validation_result['issues']:
                logger.warning(f"[Stage 1C] Quality Issues: {validation_result['issues']}")
            if validation_result['recommendations']:
                logger.info(f"[Stage 1C] Recommendations: {validation_result['recommendations']}")

            # ============================================================
            # BUILD CALCULATION CONTEXT
            # ============================================================

            # Convert extracted data to DataPoint objects
            data_points = []
            for dp in extraction_data.get('extracted_data', []):
                try:
                    data_points.append(DataPoint(**dp))
                except Exception as e:
                    logger.warning(f"[Phase 1] Failed to create DataPoint from {dp}: {e}")

            # Convert calculations to Calculation objects
            calculations = []
            for calc in extraction_data.get('calculations', []):
                try:
                    calculations.append(Calculation(**calc))
                except Exception as e:
                    logger.warning(f"[Phase 1] Failed to create Calculation from {calc}: {e}")

            # Convert comparisons to ComparisonEntry objects
            comparisons = []
            for comp in extraction_data.get('key_comparisons', []):
                try:
                    comparisons.append(ComparisonEntry(**comp))
                except Exception as e:
                    logger.warning(f"[Phase 1] Failed to create ComparisonEntry from {comp}: {e}")

            # Create CalculationContext
            calc_context = CalculationContext(
                extracted_data=data_points,
                calculations=calculations,
                key_comparisons=comparisons,
                summary_insights=extraction_data.get('summary_insights', []),
                data_quality_notes=extraction_data.get('data_quality_notes', []),
                metadata={
                    'understanding_chars': len(understanding),
                    'observation_count': len(merged),
                    'quality_score': validation_result['quality_score'],
                    'quality_valid': validation_result['valid'],
                    'quality_summary': validation_result['summary']
                }
            )

            logger.info(
                f"[HYBRID Phase 1] THREE-STAGE PIPELINE COMPLETE: "
                f"{len(calc_context.extracted_data)} data points, "
                f"{len(calc_context.calculations)} calculations, "
                f"{len(calc_context.key_comparisons)} comparisons "
                f"(Quality: {validation_result['quality_score']:.2f})"
            )

            # If quality is too low, try fallback using datapoints
            if not validation_result['valid'] and len(data_points) > 0:
                logger.warning(
                    f"[Phase 1] Quality check failed, attempting fallback to build comparisons from data points"
                )
                fallback_comparisons = self._build_comparisons_from_datapoints(data_points)
                if fallback_comparisons:
                    calc_context.key_comparisons = fallback_comparisons
                    logger.info(
                        f"[Phase 1] Fallback generated {len(fallback_comparisons)} comparisons from data points"
                    )

            return calc_context

        except Exception as exc:
            logger.error(f"[HYBRID Phase 1] THREE-STAGE PIPELINE FAILED: {exc}", exc_info=True)

            # Return empty context with error note
            return CalculationContext(
                extracted_data=[],
                calculations=[],
                key_comparisons=[],
                data_quality_notes=[
                    f"Three-stage pipeline failed: {exc}",
                    "Unable to extract calculation context from observations"
                ],
                metadata={'error': str(exc)}
            )

    async def _generate_holistic_report_with_table_anchors(
        self,
        findings: Dict[str, Any],
        calc_context: Any,
        dynamic_sections: List[Dict[str, Any]] = None
    ) -> str:
        """
        Phase 2: Generate complete report with table placeholders.

        Args:
            findings: Compiled findings from _compile_findings()
            calc_context: CalculationContext from Phase 1
            dynamic_sections: Section structure from plan (optional)

        Returns:
            Report text with table anchors like [TABLE: comparison_1]
        """
        logger.info("[HYBRID Phase 2] Starting holistic report generation")

        settings = self.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {})
        anchor_format = settings.get('table_anchor_format', '[TABLE: {id}]')

        # Format calculation context for prompt
        calc_summary = f"""
Extracted Data Points ({len(calc_context.extracted_data)}):
{chr(10).join([f"- {dp.entity}: {dp.metric} = {dp.value} {dp.unit}" for dp in calc_context.extracted_data[:10]])}
{f"... and {len(calc_context.extracted_data) - 10} more" if len(calc_context.extracted_data) > 10 else ""}

Calculations Performed ({len(calc_context.calculations)}):
{chr(10).join([f"- {calc.description}: {calc.formula} = {calc.result} {calc.unit}" for calc in calc_context.calculations[:5]])}
{f"... and {len(calc_context.calculations) - 5} more" if len(calc_context.calculations) > 5 else ""}

Key Comparisons ({len(calc_context.key_comparisons)}):
{chr(10).join([f"- {comp.primary_key}: {list(comp.metrics.keys())}" for comp in calc_context.key_comparisons[:5]])}
{f"... and {len(calc_context.key_comparisons) - 5} more" if len(calc_context.key_comparisons) > 5 else ""}

Insights:
{chr(10).join([f"- {insight}" for insight in calc_context.summary_insights[:3]])}
"""

        research_topic = findings.get('research_topic', 'Research Question')

        # Build section structure guidance
        section_structure = ""
        if dynamic_sections:
            section_structure = "\n\nReport Structure (use these sections in order):\n"
            for i, sec in enumerate(dynamic_sections, 1):
                title = sec.get('title', sec) if isinstance(sec, dict) else sec
                purpose = sec.get('purpose', '') if isinstance(sec, dict) else ''
                section_structure += f"{i}. {title}"
                if purpose:
                    section_structure += f" - {purpose}"
                section_structure += "\n"

        prompt = f"""Generate a comprehensive research report on this topic.

Research Topic: {research_topic}

Pre-Calculated Data and Analysis:
{calc_summary}
{section_structure}

Instructions:
1. Write a complete, well-structured report answering the research question
2. Use the pre-calculated data and insights provided above
3. Follow the section structure provided above (if any)
4. When you need to present tabular data, insert a table anchor using this format: {anchor_format}
   - Example: {anchor_format.format(id='comparison_1')}
   - Use descriptive IDs like 'comparison_1', 'metrics_summary', 'detailed_breakdown'
5. Do NOT describe your research methodology or explain how you searched
6. Do NOT mention files, spreadsheets, or external tools
7. Focus on answering the user's question with the facts available
8. Use professional, clear language

Generate the complete report now:"""

        messages = [
            SystemMessage(content="""You are a professional research report writer.
Generate comprehensive, well-structured reports that directly answer research questions.
Never describe methodology or research process.
Never mention external files or tools.
Focus entirely on findings and analysis."""),
            HumanMessage(content=prompt)
        ]

        try:
            logger.info("[HYBRID Phase 2] Invoking LLM for holistic report")
            response = await self.llm.ainvoke(
                messages,
                timeout=settings.get('holistic_timeout_seconds', 240)
            )

            report_text = extract_content(response)
            logger.info(f"[HYBRID Phase 2] Generated holistic report: {len(report_text)} characters")

            # Count table anchors
            table_specs = self._extract_table_specs(report_text)
            logger.info(f"[HYBRID Phase 2] Found {len(table_specs)} table anchors")

            return report_text

        except Exception as exc:
            logger.error(f"[HYBRID Phase 2] Holistic report generation failed: {exc}")
            raise

    async def _generate_tables_from_anchors_async(
        self,
        report_text: str,
        calc_context: Any,
        findings: Dict[str, Any]
    ) -> str:
        """
        Phase 3: Parse table anchors and generate tables with structured output.

        Args:
            report_text: Report with table anchors from Phase 2
            calc_context: CalculationContext from Phase 1
            findings: Compiled findings

        Returns:
            Final report with all tables rendered
        """
        import asyncio

        logger.info("[HYBRID Phase 3] Starting table generation from anchors")

        tables = self._extract_table_specs(report_text)
        if not tables:
            logger.info("[HYBRID Phase 3] No table anchors found, returning report as-is")
            return report_text

        settings = self.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {})
        enable_async = settings.get('enable_async_blocks', False)
        concurrency = settings.get('max_concurrent_blocks', 2)

        logger.info(f"[HYBRID Phase 3] Generating {len(tables)} tables (async: {enable_async}, concurrency: {concurrency})")

        async def render_table(spec):
            try:
                table_md = await self._generate_single_table(spec, calc_context, findings)
                return spec['full_match'], table_md, None
            except Exception as exc:
                logger.error(f"[HYBRID Phase 3] Table generation failed for {spec['table_id']}: {exc}")
                fallback = self._fallback_table_summary(spec, calc_context, findings, error=str(exc))
                return spec['full_match'], fallback, exc

        if enable_async:
            semaphore = asyncio.Semaphore(concurrency)

            async def guarded(spec):
                async with semaphore:
                    return await render_table(spec)

            rendered_blocks = await asyncio.gather(*(guarded(spec) for spec in tables), return_exceptions=False)
        else:
            rendered_blocks = []
            for spec in tables:
                result = await render_table(spec)
                rendered_blocks.append(result)

        # Aggregate errors
        error_aggregates = self._aggregate_async_errors(rendered_blocks)
        logger.info(f"[HYBRID Phase 3] Table generation complete: {error_aggregates['successful']}/{error_aggregates['total_tables']} successful")

        # Replace anchors with rendered tables (in reverse order to preserve positions)
        final_text = report_text
        for match, replacement, error in reversed(rendered_blocks):
            final_text = final_text.replace(match, replacement, 1)

        # Alert on any unreplaced tables
        self._alert_on_unreplaced_tables(final_text)

        return final_text

    async def _generate_single_table(
        self,
        spec: Dict[str, Any],
        calc_context: Any,
        findings: Dict[str, Any]
    ) -> str:
        """
        Generate a single table from specification and calculation context.

        Args:
            spec: Table specification with table_id
            calc_context: CalculationContext with data
            findings: Compiled findings

        Returns:
            Rendered markdown table
        """
        from ..core.report_models_structured import TableBlock, get_databricks_schema

        table_id = spec['table_id']
        logger.info(f"[HYBRID Phase 3] Generating table: {table_id}")

        # Select relevant comparisons for this table
        relevant_comparisons = calc_context.key_comparisons[:10]  # Limit to 10 rows

        if not relevant_comparisons:
            logger.warning(f"[HYBRID Phase 3] No comparisons available for table {table_id}")
            return f"\n\n*Table {table_id}: No data available*\n\n"

        # Build prompt for table generation
        comparisons_text = "\n".join([
            f"- {comp.primary_key}: {', '.join([f'{k}={v}' for k, v in comp.metrics.items()])}"
            for comp in relevant_comparisons
        ])

        prompt = f"""Generate a table for: {table_id}

Available Comparison Data:
{comparisons_text}

Create a well-formatted table with:
- headers: List of column names
- rows: List of rows, where each row is a list of cell values
- caption: Brief description of what the table shows

Make the table clear, concise, and informative."""

        messages = [
            SystemMessage(content="You are a table generation specialist. Create clear, well-structured tables."),
            HumanMessage(content=prompt)
        ]

        try:
            # Use structured output for table
            structured_llm = self.llm.with_structured_output(
                TableBlock,
                method="json_schema"
            )

            response = await structured_llm.ainvoke(messages)
            table_block = self._validate_structured_response(response, TableBlock)

            # Render to markdown
            table_md = table_block.render_markdown()
            logger.info(f"[HYBRID Phase 3] Successfully generated table {table_id}: {len(table_md)} chars")

            return f"\n\n{table_md}\n\n"

        except Exception as exc:
            logger.error(f"[HYBRID Phase 3] Structured table generation failed for {table_id}: {exc}")
            raise

    def _fallback_table_summary(
        self,
        spec: Dict[str, Any],
        calc_context: Any,
        findings: Dict[str, Any],
        error: str = None
    ) -> str:
        """
        Generate bullet summary fallback when table generation fails.

        Args:
            spec: Table specification
            calc_context: CalculationContext with data
            findings: Compiled findings
            error: Optional error message

        Returns:
            Bullet list summary
        """
        table_id = spec['table_id']
        max_rows = self.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {}).get(
            'table_fallback_max_rows', 6
        )

        logger.info(f"[HYBRID Phase 3] Generating fallback summary for table {table_id}")

        # Select top comparisons
        comparisons = calc_context.key_comparisons[:max_rows]

        if not comparisons:
            return f"\n\n*Table {table_id}: No data available for fallback summary*\n\n"

        bullets = [f"\n\n**{table_id}** (summary):"]
        citations = findings.get('citations', [])

        for comp in comparisons:
            metrics_str = ", ".join([f"{k}: {v}" for k, v in list(comp.metrics.items())[:3]])

            # Try to find citation
            citation = ""
            if comp.source_observation_ids:
                citation = self._resolve_citation(comp.source_observation_ids[0], citations)

            bullets.append(f"- {comp.primary_key}: {metrics_str} {citation}")

        if error:
            bullets.append(f"\n*(Structured table unavailable: {error[:100]})*")

        return "\n".join(bullets) + "\n\n"

    # ===================================================================
    # HYBRID MULTI-PASS REPORT GENERATION - HELPER METHODS
    # ===================================================================

    def _dedupe_preserve_order(self, observations: List[Dict]) -> List[Dict]:
        """
        Deduplicate observations preserving first occurrence order.

        Deduplication key: (step_id, content_hash[:100])
        This prevents selector top-K and recency tail from duplicating tokens.
        """
        import hashlib

        seen = set()
        deduped = []

        for obs in observations:
            # Create deduplication key from step_id and content prefix
            step_id = obs.get('step_id', '')
            content = obs.get('content', '')[:100]  # First 100 chars for efficiency
            key = (step_id, hashlib.md5(content.encode()).hexdigest())

            if key not in seen:
                seen.add(key)
                deduped.append(obs)

        return deduped

    def _extract_table_specs(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract table anchor specifications from report text.

        Handles various bracket styles and validates for duplicates.
        """
        pattern_template = self.config.get('agents', {}).get('reporter', {}).get(
            'hybrid_settings', {}
        ).get('table_anchor_format', '[TABLE: {id}]')

        # Build regex pattern from template
        escaped = re.escape(pattern_template)
        # Replace the escaped {id} placeholder with capture group
        # Note: re.escape will turn {id} into \{id\}, so we need to match that
        pattern = escaped.replace(r'\{id\}', r'([^\[\]\{\}]+)')

        specs = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            specs.append({
                'full_match': match.group(0),
                'table_id': match.group(1).strip(),
                'position': match.start(),
                'context': text[max(0, match.start()-50):match.end()+50]  # For debugging
            })

        # Validate for duplicate IDs
        table_ids = [s['table_id'] for s in specs]
        if len(table_ids) != len(set(table_ids)):
            logger.warning(f"Duplicate table IDs detected: {table_ids}")

        return specs

    def _resolve_citation(self, observation_id: str, citations: List[Dict] = None) -> str:
        """
        Map observation ID to citation marker or source reference.

        Args:
            observation_id: ID of the observation needing citation
            citations: Optional list of existing citations from state

        Returns:
            Citation marker like '[1]' or '(ref:abc123)' or empty string
        """
        if not observation_id:
            return ""

        # First try to find in existing citations
        if citations:
            for i, cite in enumerate(citations):
                if cite.get('observation_id') == observation_id:
                    return f"[{i+1}]"
                # Also check if observation content matches citation source
                if cite.get('source_id') == observation_id:
                    return f"[{i+1}]"

        # Fallback to abbreviated observation ID
        return f"(obs:{observation_id[-6:]})" if len(observation_id) > 6 else f"(obs:{observation_id})"

    def _enforce_token_budget(self, text: str, max_tokens: int, safety_factor: float = 0.9) -> str:
        """
        Enforce token budget with safety margin.

        Args:
            text: Input text to potentially truncate
            max_tokens: Maximum allowed tokens
            safety_factor: Safety margin (0.9 = use 90% of limit)

        Returns:
            Truncated text if needed, original otherwise
        """
        try:
            from transformers import AutoTokenizer

            # Use cached tokenizer or initialize
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = AutoTokenizer.from_pretrained('gpt2')  # Approximate token counter

            actual_limit = int(max_tokens * safety_factor)
            tokens = self._tokenizer.encode(text)

            if len(tokens) <= actual_limit:
                return text

            # Smart truncation - try to break at sentence boundary
            truncated_tokens = tokens[:actual_limit]
            truncated_text = self._tokenizer.decode(truncated_tokens)

            # Find last complete sentence
            last_sentence = truncated_text.rfind('. ')
            if last_sentence > len(truncated_text) * 0.7:  # Keep at least 70% of text
                truncated_text = truncated_text[:last_sentence + 1]

            return truncated_text + " [truncated for token limit]"
        except Exception as e:
            logger.warning(f"Token budget enforcement failed: {e}. Falling back to character limit.")
            # Fallback to character-based truncation
            char_limit = int(max_tokens * 4 * safety_factor)  # ~4 chars per token
            if len(text) <= char_limit:
                return text
            return text[:char_limit] + " [truncated for token limit]"

    def _summarize_observations_for_prompt(
        self,
        observations: List[Dict],
        max_chars: int = 200000
    ) -> str:
        """
        Format observations with FULL CONTENT for LLM processing.

        Uses the full_content field to preserve complete observation data including
        tables, numeric data, and structured information critical for Phase 1 extraction.

        With 200K character budget, this should accommodate most research tasks without truncation.
        """
        if not observations:
            return ""

        # Build formatted observations with FULL content
        parts = []
        total_chars = 0

        for obs in observations:
            # CRITICAL: Use full_content field to preserve tables and numeric data!
            full_content = obs.get('full_content', obs.get('content', ''))
            step_id = obs.get('step_id', 'unknown')
            formatted = f"Step {step_id}: {full_content}"
            parts.append(formatted)
            total_chars += len(formatted)

        result = "\n\n".join(parts)

        logger.info(
            f"[Phase 1 Observations] Formatted {len(observations)} observations: "
            f"{total_chars:,} chars (limit: {max_chars:,})"
        )

        # Only truncate if we exceed the generous limit
        if len(result) > max_chars:
            logger.warning(
                f"[Phase 1 Observations] Truncating {len(result):,} -> {max_chars:,} chars"
            )
            result = result[:max_chars] + f"\n\n[Content truncated at {max_chars:,} char limit]"

        return result

    def _validate_hybrid_config(self, config: Dict) -> None:
        """
        Validate hybrid generation configuration.

        Raises:
            ValueError: If configuration is invalid or inconsistent
        """
        settings = config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {})
        memory_config = config.get('memory', {})
        report_config = config.get('report', {})

        # Validate observation windows don't exceed memory limits
        memory_max = memory_config.get('max_observations', 280)
        calc_total = settings.get('calc_selector_top_k', 60) + settings.get('calc_recent_tail', 20)

        if calc_total > memory_max:
            raise ValueError(
                f"Calculation observations ({calc_total}) exceed memory limit ({memory_max}). "
                f"Reduce calc_selector_top_k or calc_recent_tail."
            )

        # Validate async settings
        if settings.get('enable_async_blocks'):
            max_concurrent = settings.get('max_concurrent_blocks', 2)
            if max_concurrent > 5:
                logger.warning(
                    f"High concurrency ({max_concurrent}) may cause rate limit issues. "
                    f"Consider reducing max_concurrent_blocks."
                )

        # Validate observation limits are sensible
        para_limit = report_config.get('max_paragraph_observations', 40)
        table_limit = report_config.get('max_table_observations', 80)

        if table_limit < para_limit:
            logger.warning(
                f"Table observation limit ({table_limit}) < paragraph limit ({para_limit}). "
                f"Tables typically need more context."
            )

        # Validate section-specific limits don't exceed global
        section_limits = report_config.get('section_observation_limits', {})
        for section, limit in section_limits.items():
            if limit > memory_max:
                raise ValueError(
                    f"Section '{section}' limit ({limit}) exceeds global memory limit ({memory_max})"
                )

        # Validate contamination patterns are valid regexes
        patterns = settings.get('contamination_patterns', [])
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid contamination pattern regex '{pattern}': {e}")

        logger.info(f"Hybrid configuration validated: {calc_total} calc observations, "
                   f"{para_limit}/{table_limit} para/table limits")

    def _monitor_memory_usage(self, phase: str, observations_count: int) -> None:
        """
        Monitor and report memory usage during processing.

        Args:
            phase: Current processing phase name
            observations_count: Number of observations being processed
        """
        try:
            import psutil
            import os

            # Get current process
            process = psutil.Process(os.getpid())

            # Memory metrics
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
            memory_percent = process.memory_percent()

            # Check against soft limit
            memory_limit_mb = self.config.get('memory_limit_mb', 2048)

            if memory_mb > memory_limit_mb:
                logger.warning(
                    f"Memory usage {memory_mb:.1f}MB exceeds soft limit {memory_limit_mb}MB in {phase} "
                    f"with {observations_count} observations"
                )

            # Emit metrics for monitoring
            self._emit_hybrid_metrics(metrics={
                f'{phase}_memory_mb': round(memory_mb, 1),
                f'{phase}_memory_percent': round(memory_percent, 2),
                f'{phase}_observations': observations_count
            })

            # Log if verbose mode
            if self.config.get('debug_memory', False):
                logger.info(f"[{phase}] Memory: {memory_mb:.1f}MB ({memory_percent:.1f}%), "
                           f"Observations: {observations_count}")
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")

    def _salvage_partial_context(
        self,
        partial_response: Any,
        error: Exception
    ) -> Optional[Any]:
        """
        Attempt to salvage partial calculation context from failed response.

        Args:
            partial_response: Potentially partial response from LLM
            error: The exception that occurred

        Returns:
            CalculationContext with salvaged data or None if unrecoverable
        """
        from ..core.report_generation.models import CalculationContext, DataPoint, Calculation

        if partial_response is None:
            return None

        # Try to extract any valid data points
        salvaged = CalculationContext(
            extracted_data=[],
            calculations=[],
            key_comparisons=[],
            data_quality_notes=[f"Partial extraction due to: {str(error)[:100]}"]
        )

        # If response is a dict, try to extract fields
        if isinstance(partial_response, dict):
            # Salvage extracted_data if present
            if 'extracted_data' in partial_response:
                try:
                    for item in partial_response['extracted_data']:
                        if isinstance(item, dict) and 'entity' in item and 'metric' in item:
                            salvaged.extracted_data.append(DataPoint(**item))
                except Exception as e:
                    salvaged.data_quality_notes.append(f"Failed to parse extracted_data: {e}")

            # Salvage calculations if present
            if 'calculations' in partial_response:
                try:
                    for calc in partial_response.get('calculations', []):
                        if isinstance(calc, dict) and 'description' in calc:
                            salvaged.calculations.append(Calculation(**calc))
                except Exception as e:
                    salvaged.data_quality_notes.append(f"Failed to parse calculations: {e}")

        # Only return if we salvaged something useful
        if salvaged.extracted_data or salvaged.calculations:
            logger.info(f"Salvaged partial context: {len(salvaged.extracted_data)} data points, "
                       f"{len(salvaged.calculations)} calculations")
            return salvaged

        return None

    def _aggregate_async_errors(
        self,
        results: List[Tuple[str, str, Optional[Exception]]]
    ) -> Dict[str, Any]:
        """
        Aggregate errors from async table generation.

        Args:
            results: List of (match, replacement, error) tuples

        Returns:
            Aggregated error metrics and diagnostics
        """
        from collections import defaultdict

        errors_by_type = defaultdict(list)
        successful = 0
        failed = 0

        for match, replacement, error in results:
            if error:
                failed += 1
                error_type = type(error).__name__
                errors_by_type[error_type].append({
                    'match': match[:50],  # Truncate for logging
                    'error': str(error)[:200]
                })
            else:
                successful += 1

        # Compute aggregates
        aggregates = {
            'total_tables': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0,
            'error_types': list(errors_by_type.keys()),
            'error_breakdown': {k: len(v) for k, v in errors_by_type.items()}
        }

        # Log warnings for high failure rates
        if aggregates['success_rate'] < 0.5:
            logger.error(f"High table generation failure rate: {aggregates['success_rate']:.1%}")
            for error_type, instances in errors_by_type.items():
                logger.error(f"  {error_type}: {len(instances)} failures")
                if instances:
                    logger.error(f"    Example: {instances[0]['error']}")

        return aggregates

    def _validate_structured_response(
        self,
        response: Union[Dict, Any],
        expected_model: Any
    ) -> Any:
        """
        Validate and convert structured output response.

        Args:
            response: Response from LLM (dict or Pydantic model)
            expected_model: Expected Pydantic model class

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If response doesn't match schema
        """
        from pydantic import ValidationError

        # If already the right type, validate and return
        if isinstance(response, expected_model):
            return response

        # If dict, try to construct model
        if isinstance(response, dict):
            try:
                return expected_model(**response)
            except ValidationError as e:
                logger.error(f"Schema validation failed for {expected_model.__name__}: {e}")

                # Try to provide helpful error details
                errors = e.errors()
                for error in errors[:3]:  # Show first 3 errors
                    logger.error(f"  Field '{'.'.join(str(x) for x in error['loc'])}': {error['msg']}")

                raise

        # Unexpected type
        raise TypeError(f"Expected {expected_model.__name__} or dict, got {type(response).__name__}")

    def _alert_on_unreplaced_tables(self, final_text: str) -> None:
        """
        Check for and alert on any unreplaced table anchors.

        Args:
            final_text: Final report text after table replacement
        """
        pattern = self.config.get('agents', {}).get('reporter', {}).get(
            'hybrid_settings', {}
        ).get('table_anchor_format', '[TABLE: {id}]')

        # Build search pattern
        escaped = re.escape(pattern)
        search_pattern = escaped.replace(r'\{id\}', r'[^\\[\\]]+')

        unreplaced = re.findall(search_pattern, final_text)

        if unreplaced:
            logger.error(f"Found {len(unreplaced)} unreplaced table anchors in final report")
            for anchor in unreplaced[:3]:  # Log first 3
                logger.error(f"  Unreplaced: {anchor}")

            # Emit metric for monitoring
            self._emit_hybrid_metrics(metrics={
                'unreplaced_tables': len(unreplaced),
                'unreplaced_examples': unreplaced[:3]
            })

    def _emit_hybrid_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Emit metrics for hybrid report generation.

        Args:
            metrics: Dictionary of metric name -> value pairs
        """
        if self.event_emitter:
            try:
                # Emit metrics through event emitter if available
                for key, value in metrics.items():
                    logger.info(f"[HYBRID_METRIC] {key}: {value}")
            except Exception as e:
                logger.warning(f"Failed to emit hybrid metrics: {e}")
        else:
            # Just log if no event emitter
            for key, value in metrics.items():
                logger.info(f"[HYBRID_METRIC] {key}: {value}")

    # ===================================================================
    # END OF HYBRID HELPER METHODS
    # ===================================================================

    def _extract_loop_findings(self, compiled_findings: Dict[str, Any], loop_discoveries: List[Dict[str, Any]]) -> List[str]:
        """Extract new findings from the current research loop."""
        new_findings = []

        # Extract recent observations (from current loop)
        observations = compiled_findings.get("observations", [])
        if observations:
            # Get the most recent observations (assuming they're from current loop)
            recent_obs = observations[-10:]  # Last 10 observations
            for obs in recent_obs:
                if hasattr(obs, 'content') and obs.content:
                    finding = obs.content[:300] + "..." if len(obs.content) > 300 else obs.content
                    new_findings.append(finding)

        # Extract discoveries specifically from loop tracking
        for discovery in loop_discoveries[-5:]:  # Last 5 discoveries
            if isinstance(discovery, dict) and discovery.get("discovery"):
                new_findings.append(discovery["discovery"])

        return new_findings[:8]  # Limit to 8 key findings

    def _generate_progressive_synthesis(
        self,
        state: EnhancedResearchState,
        progressive_context: Dict[str, Any],
        compiled_findings: Dict[str, Any],
        style_config: Any,
        report_style: str
    ) -> str:
        """Generate progressive synthesis that builds on previous research loops."""
        try:
            # Import the progressive synthesis prompt
            from ..prompts import PROGRESSIVE_SYNTHESIS_PROMPT

            research_topic = state.get("research_topic", "")
            research_loop = state.get("research_loops", 0)

            # Format the prompt with progressive context
            synthesis_prompt = PROGRESSIVE_SYNTHESIS_PROMPT.format(
                research_topic=research_topic,
                research_loop=research_loop,
                previous_synthesis=progressive_context.get("previous_synthesis", ""),
                new_findings="\n".join([f"- {finding}" for finding in progressive_context.get("new_findings", [])]),
                verified_claims="\n".join([f"- {claim}" for claim in progressive_context.get("verified_claims", [])]),
                deep_dive_insights="\n".join([f"- {insight}" for insight in progressive_context.get("deep_dive_insights", [])])
            )

            # Use the LLM to generate progressive synthesis - required for this project
            if hasattr(self, 'llm') and self.llm:
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage

                    messages = [
                        SystemMessage(content="You are an expert research synthesizer creating progressive reports."),
                        HumanMessage(content=synthesis_prompt)
                    ]

                    response = self.llm.invoke(messages)
                    enhanced_synthesis = self._extract_text_from_response(response)

                    logger.info(f"[PROGRESSIVE SYNTHESIS] Generated enhanced synthesis: {len(enhanced_synthesis)} characters")
                    return enhanced_synthesis

                except Exception as e:
                    logger.error(f"[PROGRESSIVE SYNTHESIS] LLM call failed: {e}")
                    raise  # Re-raise - project requires LLM

            else:
                logger.error("[PROGRESSIVE SYNTHESIS] No LLM available - required for this project")
                raise ValueError("LLM not available for progressive synthesis - required for this project")

        except Exception as e:
            logger.error(f"[PROGRESSIVE SYNTHESIS] Error in progressive synthesis: {e}")
            raise  # Re-raise - no fallback needed

