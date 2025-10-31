"""
Advanced utilities for deep_research_agent (V2 Refactoring).

This module provides domain-specific abstractions to replace duplicate patterns:
- TypeAwareListMergeManager: Type-aware list merging with Pydantic validation
- AsyncAwareBaseAgent: Base class for async/sync compatibility
- LLMInvocationManager: Structured output support with retry and metrics
- ObservationProcessor: Full observation processing pipeline
- StateUpdateBuilder: Type-safe state updates with builder pattern

These utilities implement the V2 refactoring plan with architecture awareness.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .utilities import ListMergeManager, StateExtractor
from .observation_models import StructuredObservation

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# TypeAwareListMergeManager - Enhanced list merging with type detection
# =============================================================================

class TypeAwareListMergeManager(ListMergeManager):
    """
    Enhanced list merger with type-aware limits.

    Uses proper type detection via isinstance and Pydantic validation
    instead of fragile dict key inspection.

    Integrates with MemoryOptimizedConfig for consistent limits.
    """

    def __init__(self):
        """Initialize with default max_items (will be overridden by type-specific limits)."""
        super().__init__(max_items=1000)
        self.type_limits = self._initialize_type_limits()

    def _initialize_type_limits(self) -> Dict[Type, int]:
        """Initialize type-specific limits from MemoryOptimizedConfig."""
        try:
            from .memory_config import MemoryOptimizedConfig

            return {
                StructuredObservation: MemoryOptimizedConfig.MAX_OBSERVATIONS,
                dict: MemoryOptimizedConfig.MAX_GENERAL_LIST_SIZE,
                str: MemoryOptimizedConfig.MAX_OBSERVATIONS
            }
        except ImportError:
            logger.warning("MemoryOptimizedConfig not found, using defaults")
            return {
                StructuredObservation: 500,
                dict: 1000,
                str: 500
            }

    def detect_type(self, item: Any) -> Type:
        """
        Detect item type for appropriate limits.

        Uses proper type checking instead of fragile dict key inspection.

        Args:
            item: Item to inspect

        Returns:
            Type class for limit lookup
        """
        # Check object types first (most specific)
        if isinstance(item, StructuredObservation):
            return StructuredObservation

        # For dicts, try Pydantic validation to determine type
        if isinstance(item, dict):
            # Try to validate as StructuredObservation
            try:
                # Don't actually convert, just validate structure
                if all(k in item for k in ['content']):
                    # Has observation-like structure
                    return StructuredObservation
            except Exception:
                pass

            # Check for SearchResult structure
            if all(k in item for k in ['url', 'score']):
                return dict  # SearchResult type

            # Check for Citation structure
            if all(k in item for k in ['source', 'snippet']):
                return dict  # Citation type

            # Generic dict
            return dict

        # String items
        if isinstance(item, str):
            return str

        # Default to dict for unknown types
        return dict

    def merge_with_type_limits(
        self,
        left: Union[List[Any], Any, None],
        right: Union[List[Any], Any, None],
        deduplicate: bool = False,
        key_fn: Optional[Callable] = None
    ) -> List[Any]:
        """
        Merge with type-specific limits.

        Args:
            left: First list (or single item, or None)
            right: Second list (or single item, or None)
            deduplicate: Whether to remove duplicates
            key_fn: Optional function to extract deduplication key

        Returns:
            Merged list with type-appropriate truncation
        """
        # Use parent merge_lists for basic merging
        merged = self.merge_lists(
            left=left,
            right=right,
            deduplicate=deduplicate,
            key_fn=key_fn
        )

        if not merged:
            return merged

        # Detect type and apply appropriate limit
        item_type = self.detect_type(merged[0])
        limit = self.type_limits.get(item_type, self.max_items)

        if len(merged) > limit:
            logger.warning(
                f"Truncating {item_type.__name__} list from {len(merged)} to {limit}"
            )
            merged = merged[-limit:]  # Keep most recent

        return merged


# =============================================================================
# AsyncAwareBaseAgent - Base class for async/sync compatibility
# =============================================================================

class AsyncAwareBaseAgent(ABC):
    """
    Base agent with async/sync compatibility.

    Automatically handles async/sync bridge for MLflow compatibility using AsyncExecutor.
    All agents should inherit from this class.
    """

    def __init__(
        self,
        name: str,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        event_emitter=None
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name (e.g., 'planner', 'researcher')
            llm: Language model instance
            config: Configuration dictionary
            event_emitter: Event emitter for progress tracking
        """
        self.name = name
        self.llm = llm
        self.event_emitter = event_emitter
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Import utilities here to avoid circular imports
        from .utilities import ConfigAccessor
        self.config_accessor = ConfigAccessor(config or {})

    @abstractmethod
    async def aprocess(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async processing method to be implemented by agents.

        Args:
            state: Current state dictionary

        Returns:
            State update dictionary
        """
        pass

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync wrapper for MLflow compatibility.

        Automatically bridges async code to sync using AsyncExecutor.

        Args:
            state: Current state dictionary

        Returns:
            State update dictionary
        """
        from .async_utils import AsyncExecutor

        return AsyncExecutor.run_async_safe(
            self.aprocess(state),
            timeout=300.0  # 5 minutes default
        )

    def extract_state(self, state: Dict[str, Any]) -> StateExtractor:
        """
        Get state extractor for current state.

        Args:
            state: State dictionary

        Returns:
            StateExtractor instance
        """
        return StateExtractor(state)

    def emit_progress(
        self,
        message: str,
        progress: float,
        metadata: Optional[Dict] = None
    ):
        """
        Emit progress event if emitter available.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
            metadata: Optional additional metadata
        """
        if self.event_emitter:
            try:
                self.event_emitter.emit('progress', {
                    'agent': self.name,
                    'message': message,
                    'progress': progress,
                    'metadata': metadata or {}
                })
            except Exception as e:
                self.logger.warning(f"Failed to emit progress: {e}")


# =============================================================================
# LLMInvocationManager - Structured output with retry and metrics
# =============================================================================

class LLMInvocationManager:
    """
    Manages LLM invocations with retry, structured output, and metrics.

    Provides sophisticated LLM interaction beyond basic error handling:
    - Structured output with Pydantic validation
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Temperature management by model tier
    """

    def __init__(
        self,
        llm,
        model_tier: str = "analytical",
        enable_metrics: bool = True
    ):
        """
        Initialize LLM invocation manager.

        Args:
            llm: Language model instance
            model_tier: Model tier (micro, simple, analytical, complex, structured)
            enable_metrics: Whether to track invocation metrics
        """
        self.llm = llm
        self.model_tier = model_tier
        self.enable_metrics = enable_metrics
        self.invocation_count = 0
        self.total_tokens = 0

    async def ainvoke_structured(
        self,
        prompt: Union[str, List],
        response_model: Type[BaseModel],
        max_retries: int = 3,
        temperature: Optional[float] = None
    ) -> BaseModel:
        """
        Invoke LLM with structured output validation.

        Args:
            prompt: Input prompt (string or message list)
            response_model: Pydantic model for response validation
            max_retries: Max retry attempts
            temperature: Override temperature

        Returns:
            Validated response model instance

        Raises:
            RuntimeError: If all retries fail
        """
        temp = temperature if temperature is not None else self._get_tier_temperature()

        for attempt in range(max_retries):
            try:
                # Invoke LLM
                result = await self.llm.ainvoke(
                    prompt,
                    temperature=temp
                )

                # Extract content
                content = self._extract_content(result)

                # Parse JSON and validate with Pydantic
                if isinstance(content, str):
                    # Try to extract JSON from markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    data = json.loads(content)
                else:
                    data = content

                validated = response_model(**data)

                # Track metrics
                if self.enable_metrics:
                    self.invocation_count += 1
                    if hasattr(result, 'usage'):
                        self.total_tokens += result.usage.get('total_tokens', 0)

                return validated

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except ValidationError as e:
                logger.warning(f"Validation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Invocation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to get structured response after {max_retries} attempts for {response_model.__name__}"
        )

    async def ainvoke_safe(
        self,
        prompt: Union[str, List],
        default_response: Optional[str] = None,
        max_retries: int = 3,
        temperature: Optional[float] = None
    ) -> str:
        """
        Safely invoke LLM with error handling.

        Args:
            prompt: Input prompt
            default_response: Fallback response on error
            max_retries: Max retry attempts
            temperature: Override temperature

        Returns:
            LLM response content or default_response on error
        """
        temp = temperature if temperature is not None else self._get_tier_temperature()

        for attempt in range(max_retries):
            try:
                result = await self.llm.ainvoke(prompt, temperature=temp)
                content = self._extract_content(result)

                if self.enable_metrics:
                    self.invocation_count += 1
                    if hasattr(result, 'usage'):
                        self.total_tokens += result.usage.get('total_tokens', 0)

                return content

            except Exception as e:
                logger.error(f"Invocation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return default_response or ""

    def _extract_content(self, result: Any) -> str:
        """
        Extract content from LLM result.

        Args:
            result: LLM result (str or object with content attribute)

        Returns:
            Extracted content string
        """
        if isinstance(result, str):
            return result
        if hasattr(result, 'content'):
            return result.content
        return str(result)

    def _get_tier_temperature(self) -> float:
        """Get temperature based on model tier."""
        tier_temps = {
            "micro": 0.5,
            "simple": 0.5,
            "analytical": 0.7,
            "complex": 0.7,
            "structured": 0.3
        }
        return tier_temps.get(self.model_tier, 0.7)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get invocation metrics.

        Returns:
            Dict with invocation_count and total_tokens
        """
        return {
            "invocation_count": self.invocation_count,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_invocation": (
                self.total_tokens / self.invocation_count if self.invocation_count > 0 else 0
            )
        }


# =============================================================================
# ReporterLLMInvocationManager - Reporter-specific LLM invocation with retry
# =============================================================================

class ReporterLLMInvocationManager:
    """
    Reporter-specific LLM invocation with entity validation and reasoning transformation.

    Extends basic LLM invocation with reporter-specific features:
    - Entity hallucination detection
    - Reasoning-to-report transformation
    - Content sanitization
    - Event emission for progress tracking
    - Databricks-specific error classification
    - Smart retry logic for transient errors

    Extracted from reporter.py to provide reusable reporter-specific LLM logic.
    """

    def __init__(
        self,
        llm,
        event_emitter=None,
        enable_entity_validation: bool = True,
        enable_reasoning_transform: bool = True,
        model_tier: str = "analytical"
    ):
        """
        Initialize reporter LLM invocation manager.

        Args:
            llm: Language model instance
            event_emitter: Optional event emitter for progress tracking
            enable_entity_validation: Whether to validate entities against query constraints
            enable_reasoning_transform: Whether to transform reasoning to report content
            model_tier: Model tier for temperature settings
        """
        self.llm = llm
        self.event_emitter = event_emitter
        self.enable_entity_validation = enable_entity_validation
        self.enable_reasoning_transform = enable_reasoning_transform
        self.model_tier = model_tier

    def invoke_with_smart_retry(
        self,
        messages: List,
        section_name: str,
        state: Optional[Dict] = None,
        max_attempts: int = 5
    ) -> str:
        """
        Invoke LLM with intelligent retry for transient errors only.

        Combines retry logic with reporter-specific processing:
        - Entity validation against query_constraints
        - Reasoning transformation if no proper content
        - Content sanitization
        - Event emission for progress tracking

        Args:
            messages: LLM messages to send
            section_name: Name of the section being generated (for logging)
            state: Optional state dict for entity validation
            max_attempts: Maximum retry attempts

        Returns:
            str: Generated content from LLM

        Raises:
            Exception: If all retries are exhausted or permanent error encountered
        """
        import time
        import random

        attempt = 0

        while attempt < max_attempts:
            try:
                # Log the prompt being sent to LLM
                if messages and len(messages) > 0:
                    prompt_content = ""
                    for msg in messages:
                        if hasattr(msg, 'content'):
                            prompt_content += f"{msg.content}... "
                    logger.info(f"🔍 LLM_PROMPT [reporter_{section_name}]: {prompt_content}...")

                # Invoke LLM
                response = self.llm.invoke(messages)
                logger.info(f"🔍 LLM_RESPONSE [reporter_{section_name}]: {response.content}...")

                # ENTITY VALIDATION (if enabled and state provided)
                if self.enable_entity_validation and state:
                    self._validate_entities(response.content, state, section_name)

                # Success - log if we had retries
                if attempt > 0:
                    logger.info(f"LLM call succeeded for {section_name} after {attempt} retries")

                # Parse response using universal response handler
                from .response_handlers import parse_structured_response
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

                # Emit reasoning event (if enabled)
                if reasoning_text and self.event_emitter:
                    self._emit_reasoning_event(reasoning_text, section_name)

                # Transform reasoning to report if needed (if enabled)
                if self.enable_reasoning_transform and reasoning_text:
                    content = self._maybe_transform_reasoning(
                        content, reasoning_text, section_name
                    )

                # Apply content sanitization
                if content:
                    content = self._sanitize_content(content, section_name)

                # Ensure non-empty content
                if not content:
                    logger.warning(f"No content extracted for {section_name}, using empty string")
                    content = ""

                return content

            except Exception as e:
                attempt += 1
                error_str = str(e)

                # Classify error (transient vs permanent)
                is_transient, suggested_wait = self._classify_databricks_error(e)

                if not is_transient:
                    logger.error(f"Permanent error in {section_name}: {e}")
                    raise

                if attempt >= max_attempts:
                    logger.error(f"Max retries ({max_attempts}) exceeded for {section_name}")
                    raise

                # Check for 429 (all endpoints exhausted)
                if "429" in error_str:
                    logger.error(
                        f"All endpoints exhausted for {section_name} (429 errors). "
                        "Not retrying - ModelSelector already tried all available endpoints."
                    )
                    raise

                # Calculate wait time
                if suggested_wait:
                    wait_time = min(suggested_wait, 30)
                else:
                    wait_time = min(5 * (2 ** (attempt - 1)), 30)

                # Add jitter (up to 10%)
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

    def _validate_entities(self, content: str, state: Dict, section_name: str):
        """
        Validate entities against query_constraints.

        Args:
            content: LLM response content
            state: State dict containing query_constraints
            section_name: Section name for logging
        """
        from .entity_validation import EntityExtractor

        constraints = state.get("query_constraints")
        requested_entities = constraints.entities if constraints else []

        if requested_entities:
            extractor = EntityExtractor()
            response_entities = extractor.extract_entities(content)
            hallucinated = response_entities - set(requested_entities)

            if hallucinated:
                logger.warning(
                    f"🚨 ENTITY_HALLUCINATION [reporter_{section_name}]: "
                    f"LLM mentioned entities not in original query: {hallucinated}"
                )
            else:
                logger.info(
                    f"✅ ENTITY_VALIDATION [reporter_{section_name}]: "
                    f"Response only mentions requested entities: {response_entities & set(requested_entities)}"
                )

    def _emit_reasoning_event(self, reasoning_text: str, section_name: str):
        """
        Emit reasoning event via event emitter.

        Args:
            reasoning_text: Reasoning text from LLM
            section_name: Section name for logging
        """
        try:
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

    def _maybe_transform_reasoning(
        self,
        content: str,
        reasoning_text: str,
        section_name: str
    ) -> str:
        """
        Transform reasoning to report if no proper content.

        Args:
            content: Current content
            reasoning_text: Reasoning text
            section_name: Section name

        Returns:
            Transformed or original content
        """
        if len(reasoning_text.strip()) > 100 and (not content or len(content.strip()) < 50):
            if not section_name.endswith("_transformed"):
                logger.warning(
                    f"🔄 REASONING_TO_REPORT: No proper content found for {section_name}, "
                    "transforming reasoning to report..."
                )
                transformed_content = self._transform_reasoning_to_report(
                    reasoning_text, section_name
                )
                logger.info(
                    f"🔄 REASONING_TO_REPORT: Transformation completed for {section_name} "
                    f"({len(transformed_content)} chars)"
                )
                return transformed_content
            else:
                logger.warning(
                    f"🔄 REASONING_TO_REPORT: Avoiding recursive transformation for {section_name}"
                )
        elif content and reasoning_text:
            logger.info(
                f"✅ PROPER_CONTENT: Using actual report content for {section_name} "
                f"({len(content)} chars), ignoring reasoning ({len(reasoning_text)} chars)"
            )

        return content

    def _transform_reasoning_to_report(
        self,
        reasoning_text: str,
        section_name: str,
        findings: Dict[str, Any] = None
    ) -> str:
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
            clean_reasoning = reasoning_text.replace(
                "I need to", "The analysis shows"
            ).replace(
                "Let me", ""
            ).replace(
                "I should", "The research indicates"
            )
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
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content="You are a professional report writer. Transform reasoning into clear, structured report content. Remove thinking process and present facts directly."),
                HumanMessage(content=transform_prompt)
            ]

            response = self.llm.invoke(messages)

            # Parse the transformation response
            from .response_handlers import parse_structured_response
            parsed = parse_structured_response(response)

            if parsed.content and len(parsed.content.strip()) > 50:
                logger.info(f"✅ TRANSFORMATION_SUCCESS: {section_name} ({len(parsed.content)} chars)")
                return parsed.content.strip()
            else:
                logger.warning(f"⚠️ TRANSFORMATION_FAILED: {section_name} - insufficient content generated")
                clean_reasoning = reasoning_text.replace(
                    "I need to", "The analysis shows"
                ).replace(
                    "Let me", ""
                ).replace(
                    "I should", "The research indicates"
                )
                return f"## {section_name}\n\n{clean_reasoning}"

        except Exception as e:
            logger.error(f"❌ TRANSFORMATION_ERROR: {section_name} - {e}")
            clean_reasoning = reasoning_text.replace(
                "I need to", "The analysis shows"
            ).replace(
                "Let me", ""
            ).replace(
                "I should", "The research indicates"
            )
            return f"## {section_name}\n\n{clean_reasoning}"

    def _sanitize_content(self, content: str, section_name: str) -> str:
        """
        Apply content sanitization.

        Args:
            content: Content to sanitize
            section_name: Section name for logging

        Returns:
            Sanitized content
        """
        from .content_sanitizer import sanitize_agent_content

        sanitization_result = sanitize_agent_content(content)
        if sanitization_result.sanitization_applied:
            logger.info(
                f"Applied content sanitization to {section_name}: "
                f"{len(content)} -> {len(sanitization_result.clean_content)} chars"
            )
            for warning in sanitization_result.warnings:
                logger.warning(f"Content sanitization warning for {section_name}: {warning}")

        return sanitization_result.clean_content

    def _classify_databricks_error(
        self,
        error: Exception
    ) -> Tuple[bool, Optional[float]]:
        """
        Classify Databricks errors and extract retry guidance.

        Args:
            error: Exception to classify

        Returns:
            Tuple[bool, Optional[float]]: (is_transient, suggested_wait_time)
        """
        import re

        error_str = str(error)

        # Transient errors that should be retried
        if 'TEMPORARILY_UNAVAILABLE' in error_str or '503' in error_str:
            # Extract wait time if provided
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
            return True, 30.0

        # Gateway errors
        if any(code in error_str for code in ['502', '504', 'gateway']):
            return True, 15.0

        # Permanent errors
        if any(err in error_str for err in ['401', '403', 'unauthorized', 'forbidden']):
            return False, None

        # Unknown errors - don't retry
        return False, None


# =============================================================================
# ObservationProcessor - Full observation processing pipeline
# =============================================================================

class ObservationProcessor:
    """
    Process, validate, filter, and enrich observations.

    Consolidates logic from reporter.py, researcher.py, and observation_selector.py.
    Provides full observation processing pipeline with metrics.
    """

    def __init__(
        self,
        query_constraints: Optional[Any] = None,
        enable_metrics: bool = True
    ):
        """
        Initialize observation processor.

        Args:
            query_constraints: Optional QueryConstraints for filtering
            enable_metrics: Whether to collect processing metrics
        """
        self.query_constraints = query_constraints
        self.enable_metrics = enable_metrics

    def process_batch(
        self,
        observations: List[Any],
        max_content_length: int = 50000,
        min_content_length: int = 50
    ) -> Tuple[List[StructuredObservation], Dict[str, Any]]:
        """
        Process observation batch with validation, filtering, and metrics.

        Args:
            observations: List of observations in any format
            max_content_length: Maximum content length
            min_content_length: Minimum content length

        Returns:
            Tuple of (processed_observations, metrics_dict)
        """
        # Normalize all observations
        normalized = []
        for obs in observations:
            try:
                if isinstance(obs, StructuredObservation):
                    normalized.append(obs)
                elif isinstance(obs, dict):
                    normalized.append(StructuredObservation(**obs))
                elif isinstance(obs, str):
                    normalized.append(StructuredObservation(content=obs))
                else:
                    logger.warning(f"Skipping unknown observation type: {type(obs)}")
            except Exception as e:
                logger.warning(f"Failed to normalize observation: {e}")

        # Apply content length filtering
        filtered = []
        for obs in normalized:
            content_len = len(obs.content or "")
            if min_content_length <= content_len <= max_content_length:
                filtered.append(obs)

        # Apply query constraints if available
        if self.query_constraints:
            filtered = self._apply_constraints(filtered)

        # Calculate metrics
        metrics = {}
        if self.enable_metrics:
            metrics = {
                "total_input": len(observations),
                "normalized": len(normalized),
                "filtered": len(filtered),
                "avg_content_length": (
                    sum(len(o.content or "") for o in filtered) / max(len(filtered), 1)
                ),
                "has_metrics": sum(1 for o in filtered if o.metric_values),
                "has_entities": sum(1 for o in filtered if o.entity_tags)
            }

        return filtered, metrics

    def _apply_constraints(
        self,
        observations: List[StructuredObservation]
    ) -> List[StructuredObservation]:
        """
        Apply query constraints to filter observations.

        Args:
            observations: List of observations to filter

        Returns:
            Filtered observations matching constraints
        """
        if not self.query_constraints:
            return observations

        filtered = []
        for obs in observations:
            # Check entity constraints
            if hasattr(self.query_constraints, 'allowed_entities') and self.query_constraints.allowed_entities:
                if not obs.entity_tags:
                    continue
                if not any(e in self.query_constraints.allowed_entities for e in obs.entity_tags):
                    continue

            # Check metric constraints
            if hasattr(self.query_constraints, 'required_metrics') and self.query_constraints.required_metrics:
                if not obs.metric_values:
                    continue
                if not all(m in obs.metric_values for m in self.query_constraints.required_metrics):
                    continue

            filtered.append(obs)

        return filtered

    @staticmethod
    def normalize(obs: Union[StructuredObservation, Dict, str, Any]) -> StructuredObservation:
        """
        Convert any observation format to StructuredObservation.

        Args:
            obs: Observation in any format

        Returns:
            StructuredObservation instance

        Raises:
            ValueError: If observation cannot be normalized
        """
        if isinstance(obs, StructuredObservation):
            return obs

        if isinstance(obs, dict):
            return StructuredObservation(**obs)

        if isinstance(obs, str):
            return StructuredObservation(content=obs)

        if hasattr(obs, '__dict__'):
            try:
                return StructuredObservation(**obs.__dict__)
            except Exception as e:
                logger.warning(f"Failed to normalize object: {e}")

        raise ValueError(f"Cannot normalize observation of type: {type(obs)}")

    @staticmethod
    def extract_content(obs: Union[StructuredObservation, Dict, Any]) -> str:
        """
        Safely extract content from any observation format.

        Args:
            obs: Observation in any format

        Returns:
            Content string (empty string if not found)
        """
        if isinstance(obs, StructuredObservation):
            return obs.content or ""

        if isinstance(obs, dict):
            return obs.get("content", "")

        if hasattr(obs, "content"):
            return obs.content or ""

        return str(obs)


# =============================================================================
# StateUpdateBuilder - Type-safe state updates with builder pattern
# =============================================================================

class StateUpdateBuilder:
    """
    Builder pattern for safe state updates.

    Replaces manual state dict updates with type-safe operations.
    Provides fluent interface for chaining updates.
    """

    def __init__(self, current_state: Optional[Dict[str, Any]] = None):
        """
        Initialize state update builder.

        Args:
            current_state: Optional current state for context
        """
        self.state = dict(current_state) if current_state else {}
        self.updates = {}

    def add_observation(self, observation: StructuredObservation) -> 'StateUpdateBuilder':
        """
        Add single observation.

        Args:
            observation: Observation to add

        Returns:
            Self for chaining
        """
        current = self.state.get("observations", [])
        merger = TypeAwareListMergeManager()
        self.updates["observations"] = merger.merge_observations(current, [observation])
        return self

    def add_observations(self, observations: List[StructuredObservation]) -> 'StateUpdateBuilder':
        """
        Add multiple observations.

        Args:
            observations: List of observations to add

        Returns:
            Self for chaining
        """
        current = self.state.get("observations", [])
        merger = TypeAwareListMergeManager()
        self.updates["observations"] = merger.merge_observations(current, observations)
        return self

    def set_plan(self, plan: Any) -> 'StateUpdateBuilder':
        """
        Set current plan.

        Args:
            plan: Plan object

        Returns:
            Self for chaining
        """
        self.updates["current_plan"] = plan
        return self

    def set_unified_plan(self, unified_plan: Any) -> 'StateUpdateBuilder':
        """
        Set unified plan from calculation agent.

        Args:
            unified_plan: UnifiedPlan object

        Returns:
            Self for chaining
        """
        self.updates["unified_plan"] = unified_plan
        return self

    def set_grounding_result(self, result: Any) -> 'StateUpdateBuilder':
        """
        Set grounding verification result.

        Args:
            result: GroundingResult object

        Returns:
            Self for chaining
        """
        self.updates["grounding_result"] = result
        return self

    def set_calculation_results(self, results: Dict[str, Any]) -> 'StateUpdateBuilder':
        """
        Set calculation results.

        Args:
            results: Calculation results dictionary

        Returns:
            Self for chaining
        """
        self.updates["calculation_results"] = results
        return self

    def add_agent_handoff(
        self,
        from_agent: str,
        to_agent: str,
        reason: str
    ) -> 'StateUpdateBuilder':
        """
        Add agent handoff record.

        Args:
            from_agent: Source agent name
            to_agent: Destination agent name
            reason: Reason for handoff

        Returns:
            Self for chaining
        """
        handoff = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        current = self.state.get("agent_handoffs", [])
        self.updates["agent_handoffs"] = current + [handoff]
        return self

    def set_field(self, key: str, value: Any) -> 'StateUpdateBuilder':
        """
        Set arbitrary state field.

        Args:
            key: Field name
            value: Field value

        Returns:
            Self for chaining
        """
        self.updates[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build final state update dict.

        Returns:
            State update dictionary
        """
        return self.updates


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'TypeAwareListMergeManager',
    'AsyncAwareBaseAgent',
    'LLMInvocationManager',
    'ObservationProcessor',
    'StateUpdateBuilder',
]
