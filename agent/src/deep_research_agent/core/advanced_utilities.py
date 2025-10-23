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
