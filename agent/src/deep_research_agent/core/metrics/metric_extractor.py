"""
LLM-based metric extraction using structured generation.

This module provides simple, reliable metric extraction using LLM structured
generation instead of complex regex patterns. It works with long web content
and handles missing data gracefully.
"""

from typing import Dict, List, Optional, Any
import logging

from .models import DataPoint, ExtractedMetric
from .unified_models import MetricSpec

logger = logging.getLogger(__name__)


class MetricExtractor:
    """
    Simple metric extraction using LLM structured generation.

    Uses a configurable (potentially smaller) model for extraction tasks.
    This approach is more reliable than regex patterns for long web content
    and handles context better.
    """

    def __init__(self, extraction_llm, config=None):
        """
        Initialize the metric extractor.

        Args:
            extraction_llm: LLM configured for extraction (can be smaller/faster model)
            config: Optional configuration dict with extraction settings
        """
        self.extraction_llm = extraction_llm
        self.config = config or {}
        self.max_context = self.config.get('max_context_length', 8000)

    async def extract_metric(
        self,
        observation: Dict,
        metric_spec: MetricSpec
    ) -> DataPoint:
        """
        Extract metric using LLM structured generation.

        Args:
            observation: Single observation dict to extract from
            metric_spec: MetricSpec with extraction hints and tags

        Returns:
            DataPoint with extracted value or None if not found/error
        """
        # Build extraction prompt
        entity = metric_spec.tags.get('entity', metric_spec.tags.get('country', 'any'))
        time_period = metric_spec.tags.get('time_period', metric_spec.tags.get('year', 'most recent'))

        prompt = f"""Extract the following specific metric from the text:

What to extract: {metric_spec.extraction_hint or metric_spec.data_id}
Entity: {entity}
Time period: {time_period}
Expected unit: {metric_spec.unit}

Text to search:
{(observation.content or '')[:self.max_context]}

If the metric is not found, set not_found=true.
Return the exact numeric value found in the text.
Include the source_text (the sentence where you found the value).
"""

        try:
            # Use structured generation to get ExtractedMetric
            # Import message type
            from langchain_core.messages import HumanMessage

            # Wrap LLM with structured output (uses project's custom StructuredOutputWrapper)
            structured_llm = self.extraction_llm.with_structured_output(
                schema=ExtractedMetric,
                method="json_schema"  # Strict Pydantic validation
            )

            # Convert prompt string to LangChain messages
            messages = [HumanMessage(content=prompt)]

            # Call ainvoke (async) with proper interface
            result: ExtractedMetric = await structured_llm.ainvoke(
                messages,
                temperature=0.0  # Deterministic extraction
            )

            # Convert to DataPoint
            if result.not_found:
                return DataPoint(
                    metric_id=metric_spec.data_id,
                    value=None,
                    unit=metric_spec.unit,
                    confidence=0.0,
                    # ✅ CRITICAL FIX: Use attribute access on StructuredObservation objects
                    source_observations=[observation.source_id or observation.step_id or 'unknown'],
                    extraction_method='llm_not_found',
                    error=result.error
                )

            return DataPoint(
                metric_id=metric_spec.data_id,
                value=result.value,
                unit=result.unit or metric_spec.unit,
                confidence=result.confidence,
                # ✅ CRITICAL FIX: Use attribute access on StructuredObservation objects
                source_observations=[observation.source_id or observation.step_id or 'unknown'],
                extraction_method='llm_structured',
                extraction_metadata={'source_text': result.source_text}
            )

        except Exception as e:
            import traceback
            # 🔍 DIAGNOSTIC: Log full traceback for debugging
            logger.error(
                f"Error extracting metric {metric_spec.data_id}: {e}\n"
                f"Error type: {type(e).__name__}\n"
                f"Full traceback:\n{traceback.format_exc()}"
            )
            return DataPoint(
                metric_id=metric_spec.data_id,
                value=None,
                unit=metric_spec.unit,
                confidence=0.0,
                # ✅ CRITICAL FIX: Use attribute access on StructuredObservation objects
                source_observations=[observation.source_id or observation.step_id or 'unknown'],
                extraction_method='llm_error',
                error=str(e)
            )

    def find_best_observation(
        self,
        observations: List[Dict],
        metric_spec: MetricSpec
    ) -> Optional[Dict]:
        """
        Find the best observation to extract from.

        Priority:
        1. Specified observation_id
        2. Fallback observation_ids
        3. Most recent observation with relevant content

        Args:
            observations: List of observation dicts
            metric_spec: MetricSpec with observation preferences

        Returns:
            Best observation dict or None if no suitable observation found
        """
        # Try primary observation
        if metric_spec.observation_id:
            for obs in observations:
                # ✅ CRITICAL FIX: Use attribute access on StructuredObservation objects
                if obs.source_id == metric_spec.observation_id or \
                   obs.step_id == metric_spec.observation_id:
                    logger.debug(f"Found primary observation {metric_spec.observation_id} for {metric_spec.data_id}")
                    return obs

        # Try fallback observations
        for fallback_id in metric_spec.fallback_observation_ids or []:
            for obs in observations:
                # ✅ CRITICAL FIX: Use attribute access on StructuredObservation objects
                if obs.source_id == fallback_id or \
                   obs.step_id == fallback_id:
                    logger.debug(f"Found fallback observation {fallback_id} for {metric_spec.data_id}")
                    return obs

        # If no specific observation, try to find most relevant
        # (This could be enhanced with relevance scoring or semantic search)
        if observations:
            logger.debug(f"No specific observation found, using most recent for {metric_spec.data_id}")
            return observations[-1]  # Default to most recent

        logger.warning(f"No observations available for {metric_spec.data_id}")
        return None


async def extract_metric_with_llm(
    metric_spec: MetricSpec,
    observations: List[Dict],
    extraction_llm,
    config=None
) -> DataPoint:
    """
    Extract a metric using LLM structured generation.

    This is a convenience function that creates an extractor, finds the best
    observation, and performs extraction.

    Args:
        metric_spec: MetricSpec describing what to extract
        observations: List of observation dicts
        extraction_llm: LLM to use for extraction
        config: Optional configuration dict

    Returns:
        DataPoint with extracted value or error

    Example:
        data_point = await extract_metric_with_llm(
            metric_spec=MetricSpec(
                data_id='spain_net_income',
                source_type='extract',
                extraction_hint='Spain net take-home salary in EUR',
                unit='EUR',
                tags={'country': 'spain', 'metric': 'net_income'}
            ),
            observations=state['observations'],
            extraction_llm=small_llm,
            config={'max_context_length': 8000}
        )
    """
    extractor = MetricExtractor(extraction_llm, config)

    # Find the best observation to extract from
    observation = extractor.find_best_observation(observations, metric_spec)

    if observation:
        return await extractor.extract_metric(observation, metric_spec)
    else:
        # No observation found
        return DataPoint(
            metric_id=metric_spec.data_id,
            value=None,
            unit=metric_spec.unit,
            confidence=0.0,
            source_observations=[],
            extraction_method='no_observation',
            error="No suitable observation found"
        )


__all__ = [
    'MetricExtractor',
    'extract_metric_with_llm',
]
