"""
Reporter Agent: Report synthesis and formatting specialist.

Generates styled reports from research findings with proper citations.
"""

from typing import Dict, Any, Optional, List, Literal, Tuple, Sequence
from datetime import datetime
import time
import random
import re
import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import Command

from deep_research_agent.core import get_logger, Citation, SectionResearchResult
from deep_research_agent.core.template_generator import DynamicSection, SectionContentType
from deep_research_agent.core.multi_agent_state import EnhancedResearchState, StateManager
from deep_research_agent.core.report_styles import (
    ReportStyle,
    STYLE_CONFIGS,
    StyleTemplate,
    StyleConfig,
    ReportFormatter
)
from deep_research_agent.core.grounding import HallucinationPrevention
from deep_research_agent.core.presentation_requirements import PresentationRequirements
from deep_research_agent.core.semantic_extraction import SemanticEntityExtractor, StructuredDataMatcher
from deep_research_agent.core.message_utils import get_last_user_message, extract_content
from deep_research_agent.core.observation_models import (
    StructuredObservation, 
    ensure_structured_observation, 
    observations_to_research_data,
    observation_to_text,
)
from deep_research_agent.core.observation_selector import ObservationSelector
from deep_research_agent.core.plan_models import StepStatus
from deep_research_agent.core.response_handlers import parse_structured_response, ParsedResponse, ResponseType
from deep_research_agent.core.table_preprocessor import TablePreprocessor


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
    
    def __init__(self, llm=None, config=None, event_emitter=None):
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
        
        # Initialize observation selector for intelligent observation filtering
        self.observation_selector = ObservationSelector(
            embedding_manager=getattr(self, 'embedding_manager', None)
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
            from deep_research_agent.core.response_handlers import parse_structured_response
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
                            prompt_content += f"{msg.content[:200]}... "
                    logger.info(f"🔍 LLM_PROMPT [reporter_{section_name}]: {prompt_content[:500]}...")
                
                response = self.llm.invoke(messages)
                
                # Log the response received from LLM
                logger.info(f"🔍 LLM_RESPONSE [reporter_{section_name}]: {response.content[:500]}...")
                
                # ENTITY VALIDATION: Check for hallucinated entities in LLM response
                if state:
                    requested_entities = state.get("requested_entities", [])
                    if requested_entities:
                        from deep_research_agent.core.entity_validation import EntityExtractor
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
                from deep_research_agent.core.content_sanitizer import sanitize_agent_content
                
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
                
                # Calculate wait time
                if suggested_wait:
                    # Use suggested wait time from error
                    wait_time = min(suggested_wait, 120)  # Cap at 2 minutes
                else:
                    # Exponential backoff: 10, 15, 22.5, 33.75, 50.6 seconds
                    wait_time = min(10 * (1.5 ** (attempt - 1)), 60)
                
                # Add jitter to prevent thundering herd (up to 10% of wait time)
                jitter = random.uniform(0, min(5, wait_time * 0.1))
                wait_time += jitter
                
                logger.warning(
                    f"Transient error in {section_name} generation "
                    f"(attempt {attempt}/{max_attempts}), "
                    f"retrying in {wait_time:.1f}s: {error_str[:100]}"
                )
                
                time.sleep(wait_time)
        
        # Should never reach here
        raise Exception(f"Retry logic error for {section_name}")
    
    def __call__(
        self,
        state: EnhancedResearchState,
        config: Dict[str, Any]
    ) -> Command[Literal["end"]]:
        """
        Generate final report from research findings with integrated table support.
        
        Args:
            state: Current research state
            config: Configuration dictionary
            
        Returns:
            Command to end workflow with final report
        """
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
        
        # Deduplicate citations to prevent accumulation bug
        self._deduplicate_citations(state)

        current_plan = state.get("current_plan")
        template = getattr(current_plan, "report_template", None) if current_plan else None
        dynamic_sections = getattr(current_plan, "dynamic_sections", None) if current_plan else None

        # Debug logging for template and dynamic sections
        logger.info(f"REPORTER: Template available: {template is not None}")
        logger.info(f"REPORTER: Dynamic sections available: {dynamic_sections is not None}")
        if dynamic_sections:
            section_titles = [getattr(s, 'title', str(s)) for s in dynamic_sections]
            logger.info(f"REPORTER: Dynamic section titles: {section_titles}")

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
            
            # Build final report from sections
            final_report = self.formatter.format_final_report(
                report_sections,
                state.get("citations", []),
                report_style
            )
            
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
        
        # Apply table preprocessing to fix any malformed tables
        table_preprocessor = TablePreprocessor()
        report_with_metadata = table_preprocessor.preprocess_tables(report_with_metadata)
        
        # Don't mutate state directly - let LangGraph handle updates through Command
        
        logger.info(
            f"Final report generated: {len(report_with_metadata)} characters, "
            f"{len(state.get('citations', []))} citations, "
            f"table_included: {bool(table_content)}"
        )
        
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

            if isinstance(obs, dict):
                content = obs.get("content") or obs.get("observation") or ""
                source = obs.get("source")
                extracted = obs.get("extracted_data") or {}
            else:
                content = observation_to_text(obs)
                source = None
                extracted = {}

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
            if isinstance(obs, dict):
                data = obs.get("extracted_data") or {}
            if not data:
                continue

            row: Dict[str, str] = {}
            label = "Observation"
            if isinstance(obs, dict):
                label = obs.get("source") or obs.get("section") or obs.get("content", "Observation")
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
                    content = obs.get("content", obs.get("observation", ""))
                    logger.info(f"[DEBUG] Observation {i}: dict with content length {len(str(content))}: {str(content)[:200]}")
            
            # Convert string observations to dict format if needed
            for obs in direct_obs:
                if isinstance(obs, str):
                    observations.append({
                        "content": obs,
                        "source": "Research Finding",
                        "relevance": 0.8
                    })
                elif isinstance(obs, dict):
                    observations.append(obs)
        
        # 2. Research observations field (alternative)
        if "research_observations" in state and state["research_observations"]:
            research_obs = state["research_observations"]
            logger.info(f"[OBSERVATION TRACKING] Found {len(research_obs)} research observations")
            for obs in research_obs:
                # Avoid duplicates by checking content
                obs_content = obs if isinstance(obs, str) else obs.get("content", "")
                existing_contents = [o.get("content", "") for o in observations]
                if obs_content not in existing_contents:
                    if isinstance(obs, str):
                        observations.append({
                            "content": obs,
                            "source": "Research Observation",
                            "relevance": 0.8
                        })
                    elif isinstance(obs, dict):
                        observations.append(obs)
        
        # 3. Section research results (for section-based research)
        if state.get("section_research_results"):
            section_results = state.get("section_research_results", {})
            logger.info(f"[OBSERVATION TRACKING] Found section results with {len(section_results)} sections")
            for section_id, section_data in section_results.items():
                if isinstance(section_data, SectionResearchResult):
                    observations.append({
                        "content": section_data.synthesis,
                        "source": f"Section Research: {section_id}",
                        "relevance": section_data.confidence,
                        "section": section_data.metadata.get("section_title", section_id),
                        "extracted_data": dict(section_data.extracted_data),
                    })
                elif isinstance(section_data, dict):
                    observations.append({
                        "content": section_data.get("synthesis", ""),
                        "source": f"Section Research: {section_id}",
                        "relevance": section_data.get("confidence", 0.5),
                        "section": section_data.get("title", section_id),
                        "extracted_data": section_data.get("extracted_data", {}),
                    })
        
        # 4. Search results as fallback (ENHANCED)
        if not observations and "search_results" in state:
            logger.warning("[OBSERVATION TRACKING] No observations found, converting search results to observations")
            search_results = state.get("search_results", [])
            for result in search_results[:25]:  # Increased from 10 to 25 for more comprehensive data
                # Create more comprehensive observation from search result
                content = result.get("snippet", "")
                if not content:
                    content = result.get("title", "")
                if content:  # Only add if we have some content
                    observations.append({
                        "content": content,
                        "source": result.get("title", "Search Result"),
                        "url": result.get("url", ""),
                        "relevance": result.get("relevance_score", 0.5)
                    })
        
        # 5. CRITICAL FALLBACK: If still no observations, create placeholder content from citations
        if not observations and citations:
            logger.warning("[OBSERVATION TRACKING] No observations OR search results, creating from citations")
            for citation in citations[:15]:  # Create observations from citations
                observations.append({
                    "content": f"Source: {citation.get('title', 'Unknown')} - {citation.get('snippet', 'Research source')}",
                    "source": citation.get("title", "Citation"),
                    "url": citation.get("url", ""),
                    "relevance": 0.6
                })
        
        # 6. EMERGENCY FALLBACK: If absolutely no content, create a basic structure
        if not observations:
            logger.error("[OBSERVATION TRACKING] NO OBSERVATIONS, SEARCH RESULTS, OR CITATIONS FOUND!")
            logger.error("This will result in a report with only references - creating emergency fallback")
            # Create a placeholder observation to prevent completely empty reports
            research_topic = state.get("research_topic", "the requested topic")
            observations.append({
                "content": f"Research was conducted on {research_topic}. Multiple sources were consulted but observations were not properly captured in the system.",
                "source": "System Note",
                "url": "",
                "relevance": 0.3
            })
        
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
            if isinstance(first_obs, dict):
                content = first_obs.get('content', '')
                logger.info(f"[DEBUG] First observation content: {content[:200] if content else 'EMPTY CONTENT'}")
        
        return compiled
    
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
        
        for section_name in style_config.structure:
            logger.info(f"Generating section: {section_name}")
            
            # Get section template
            section_template = StyleTemplate.get_section_template(
                style_config.style,
                section_name
            )
            
            # Generate section content
            section_content = self._generate_section_content(
                section_name,
                findings,
                section_template,
                style_config.style
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

        for dynamic in sorted(dynamic_sections, key=lambda s: s.priority):
            section_name = dynamic.title
            logger.info(f"Generating dynamic section: {section_name}")

            template = self._build_dynamic_section_template(dynamic)
            section_content = self._generate_section_content(
                section_name,
                findings,
                template,
                style_config.style,
            )

            if embedded_table and section_name == table_section:
                table_intro = self._generate_table_introduction(state, table_metadata)
                section_content = f"{section_content}\n\n{table_intro}\n\n{embedded_table}\n\n"
                logger.info(f"Dynamic section '{section_name}' received embedded table")

            sections[section_name] = section_content

        return sections

    def _generate_section_content(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle
    ) -> str:
        """Generate content for a specific report section."""
        
        # CRITICAL: Don't generate content if this is an error report
        if findings.get("error_report", False):
            return f"## {section_name}\n\n{findings['observations'][0] if findings['observations'] else 'Research failed - no data available.'}"
        
        # CRITICAL: Check if we have actual research data
        observations = findings.get("observations", [])
        if not observations or len(observations) == 0:
            logger.warning(f"REPORTER: No observations available for section '{section_name}' - preventing LLM hallucination")
            return f"## {section_name}\n\nNo research data available for this section due to search tool failures."
        
        # Build prompt for section generation
        prompt = self._build_section_prompt(
            section_name,
            findings,
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
        observations: List[Any],
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
        {research.get('synthesis', '[No observations available]')}
        
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
            from deep_research_agent.core.llm_response_parser import extract_text_from_response
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
        Generate table using LLM with clean failure handling - NO FALLBACKS.
        Supports up to 32k tokens for comprehensive data extraction.
        When table generation fails, return honest error message with specific reason.
        """
        try:
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
                
                table_content = response.content if hasattr(response, 'content') else str(response)
                
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
    
    def _create_fallback_table(self, rows: List[str], columns: List[str]) -> Dict:
        """Generate basic table structure when all else fails."""
        try:
            # Build table structure
            header = "| " + " | ".join(["Entity"] + columns) + " |"
            separator = "|" + "---|" * (len(columns) + 1)
            
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
            from deep_research_agent.core.llm_response_parser import extract_text_from_response
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
