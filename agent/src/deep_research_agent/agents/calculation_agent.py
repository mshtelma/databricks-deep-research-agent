"""
Calculation Agent: Executes UnifiedPlan for metric extraction and calculation.

This agent sits between Researcher and Reporter, taking the UnifiedPlan created
during planning and executing it to extract/calculate all required metrics.
"""

import logging
from typing import Dict, Any, List, Optional

from ..core import get_logger
from ..core.multi_agent_state import EnhancedResearchState, StateManager
from ..core.metrics import (
    UnifiedPlan,
    MetricSpec,
    DataPoint,
    MetricExtractor,
    SimpleFormulaEvaluator,
    extract_metric_with_llm,
    execute_calculation_with_safety,
    SourceType,
)

logger = get_logger(__name__)


class CalculationAgent:
    """
    Agent responsible for executing UnifiedPlan to extract and calculate metrics.

    Responsibilities:
    - Execute UnifiedPlan from state
    - Extract metrics from observations using LLM structured generation
    - Calculate metrics using formulas
    - Handle missing data gracefully
    - Store results in state for Reporter
    """

    def __init__(self, extraction_llm=None, config=None, event_emitter=None):
        """
        Initialize the calculation agent.

        Args:
            extraction_llm: LLM for metric extraction (can be smaller/faster model)
            config: Configuration dictionary
            event_emitter: Optional event emitter for progress tracking
        """
        self.extraction_llm = extraction_llm
        self.config = config or {}
        self.event_emitter = event_emitter
        self.name = "CalculationAgent"

        # Extract calculation agent configuration
        calc_config = self.config.get('calculation_agent', {})
        self.enabled = calc_config.get('enabled', True)

        # Extraction configuration
        extraction_config = calc_config.get('metric_extraction', {})
        self.max_context_length = extraction_config.get('max_context_length', 8000)
        self.confidence_threshold = extraction_config.get('confidence_threshold', 0.7)
        self.enable_fallback_search = extraction_config.get('enable_fallback_search', True)

        # Calculation configuration
        calculation_config = calc_config.get('calculation', {})
        self.tolerance = calculation_config.get('tolerance', 0.01)
        self.max_retries = calculation_config.get('max_retries', 3)
        self.verify_results = calculation_config.get('verify_results', True)

        # Initialize components
        self.extractor = MetricExtractor(
            extraction_llm=self.extraction_llm,
            config={'max_context_length': self.max_context_length}
        )
        self.evaluator = SimpleFormulaEvaluator()

        logger.info(
            f"CalculationAgent initialized: enabled={self.enabled}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"tolerance={self.tolerance}"
        )

    async def execute(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """
        Execute the UnifiedPlan from state.

        Args:
            state: Current research state with UnifiedPlan and observations

        Returns:
            State updates with extracted/calculated metrics
        """
        if not self.enabled:
            logger.info("CalculationAgent disabled, skipping execution")
            return {"calculation_results": None}

        # Get UnifiedPlan from state
        unified_plan = state.get('unified_plan')
        if not unified_plan:
            logger.warning("No UnifiedPlan in state, skipping calculation")
            return {"calculation_results": None}

        # Get observations from state
        observations = state.get('observations', [])
        if not observations:
            logger.warning("No observations in state, cannot extract metrics")
            return {
                "calculation_results": {
                    "error": "No observations available for metric extraction"
                }
            }

        logger.info(
            f"Executing UnifiedPlan with {len(unified_plan.data_sources)} metrics "
            f"and {len(observations)} observations"
        )

        # Emit start event
        if self.event_emitter:
            await self.event_emitter.emit(
                "calculation_start",
                {
                    "metric_count": len(unified_plan.data_sources),
                    "observation_count": len(observations)
                }
            )

        # Execute extraction and calculation
        results = await self._execute_plan(unified_plan, observations)

        # Emit completion event
        if self.event_emitter:
            await self.event_emitter.emit(
                "calculation_complete",
                {
                    "extracted_count": results.get('extracted_count', 0),
                    "calculated_count": results.get('calculated_count', 0),
                    "failed_count": results.get('failed_count', 0)
                }
            )

        logger.info(
            f"Calculation complete: {results.get('extracted_count', 0)} extracted, "
            f"{results.get('calculated_count', 0)} calculated, "
            f"{results.get('failed_count', 0)} failed"
        )

        return {"calculation_results": results}

    async def _execute_plan(
        self,
        plan: UnifiedPlan,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the UnifiedPlan to extract and calculate metrics.

        Args:
            plan: UnifiedPlan with MetricSpecs
            observations: List of observation dicts

        Returns:
            Dict with extraction and calculation results
        """
        # Separate extraction and calculation specs
        extract_specs = [
            spec for spec in plan.data_sources
            if spec.source_type == SourceType.EXTRACT
        ]
        calculate_specs = [
            spec for spec in plan.data_sources
            if spec.source_type == SourceType.CALCULATE
        ]

        logger.info(
            f"Plan breakdown: {len(extract_specs)} to extract, "
            f"{len(calculate_specs)} to calculate"
        )

        # Phase 1: Extract metrics
        extracted_data = await self._extract_metrics(extract_specs, observations)

        # Phase 2: Calculate metrics using extracted data
        calculated_data = await self._calculate_metrics(calculate_specs, extracted_data)

        # Combine results
        all_data = {**extracted_data, **calculated_data}

        # Count successes and failures
        extracted_count = sum(
            1 for dp in extracted_data.values()
            if dp.value is not None
        )
        calculated_count = sum(
            1 for dp in calculated_data.values()
            if dp.value is not None
        )
        failed_count = len(all_data) - extracted_count - calculated_count

        return {
            "data_points": all_data,
            "extracted_count": extracted_count,
            "calculated_count": calculated_count,
            "failed_count": failed_count,
            "plan_id": getattr(plan, 'plan_id', 'unknown'),
        }

    async def _extract_metrics(
        self,
        specs: List[MetricSpec],
        observations: List[Dict[str, Any]]
    ) -> Dict[str, DataPoint]:
        """
        Extract metrics from observations using LLM.

        Args:
            specs: List of MetricSpecs to extract
            observations: List of observation dicts

        Returns:
            Dict mapping metric_id to DataPoint
        """
        results = {}

        for spec in specs:
            try:
                # Find best observation for this metric
                observation = self.extractor.find_best_observation(observations, spec)

                if observation:
                    # Extract using LLM
                    data_point = await self.extractor.extract_metric(observation, spec)
                    results[spec.data_id] = data_point

                    if data_point.value is not None:
                        logger.info(
                            f"Extracted {spec.data_id}: {data_point.value} {data_point.unit} "
                            f"(confidence: {data_point.confidence:.2f})"
                        )
                    else:
                        logger.warning(
                            f"Failed to extract {spec.data_id}: {data_point.error}"
                        )
                else:
                    # No suitable observation found
                    logger.warning(f"No observation found for {spec.data_id}")
                    results[spec.data_id] = DataPoint(
                        metric_id=spec.data_id,
                        value=None,
                        unit=spec.unit,
                        confidence=0.0,
                        source_observations=[],
                        extraction_method='no_observation',
                        error="No suitable observation found"
                    )

            except Exception as e:
                logger.error(f"Error extracting {spec.data_id}: {e}", exc_info=True)
                results[spec.data_id] = DataPoint(
                    metric_id=spec.data_id,
                    value=None,
                    unit=spec.unit,
                    confidence=0.0,
                    source_observations=[],
                    extraction_method='error',
                    error=str(e)
                )

        return results

    async def _calculate_metrics(
        self,
        specs: List[MetricSpec],
        extracted_data: Dict[str, DataPoint]
    ) -> Dict[str, DataPoint]:
        """
        Calculate metrics using formulas and extracted data.

        Args:
            specs: List of MetricSpecs to calculate
            extracted_data: Dict of already extracted DataPoints

        Returns:
            Dict mapping metric_id to calculated DataPoint
        """
        results = {}

        for spec in specs:
            try:
                if not spec.formula:
                    logger.warning(
                        f"Calculate spec {spec.data_id} has no formula, skipping"
                    )
                    results[spec.data_id] = DataPoint(
                        metric_id=spec.data_id,
                        value=None,
                        unit=spec.unit,
                        confidence=0.0,
                        source_observations=[],
                        extraction_method='missing_formula',
                        error="No formula provided"
                    )
                    continue

                # Collect inputs from extracted data
                inputs = {}
                missing_inputs = []
                for required_input in spec.required_inputs:
                    if required_input in extracted_data:
                        dp = extracted_data[required_input]
                        if dp.value is not None:
                            inputs[required_input] = dp.value
                        else:
                            missing_inputs.append(required_input)
                    elif required_input in results:
                        # Input might be from a previous calculation
                        dp = results[required_input]
                        if dp.value is not None:
                            inputs[required_input] = dp.value
                        else:
                            missing_inputs.append(required_input)
                    else:
                        missing_inputs.append(required_input)

                if missing_inputs:
                    logger.warning(
                        f"Cannot calculate {spec.data_id}: missing inputs {missing_inputs}"
                    )
                    results[spec.data_id] = DataPoint(
                        metric_id=spec.data_id,
                        value=None,
                        unit=spec.unit,
                        confidence=0.0,
                        source_observations=[],
                        extraction_method='missing_inputs',
                        error=f"Missing inputs: {missing_inputs}"
                    )
                    continue

                # Execute calculation
                evaluation = self.evaluator.evaluate(spec.formula, inputs)

                if evaluation.success:
                    # Collect source observations from inputs
                    source_obs = []
                    for input_id in spec.required_inputs:
                        if input_id in extracted_data:
                            source_obs.extend(extracted_data[input_id].source_observations)
                        elif input_id in results:
                            source_obs.extend(results[input_id].source_observations)

                    results[spec.data_id] = DataPoint(
                        metric_id=spec.data_id,
                        value=evaluation.value,
                        unit=spec.unit,
                        confidence=0.95,  # High confidence for successful calculation
                        source_observations=list(set(source_obs)),  # Deduplicate
                        extraction_method='calculation',
                        extraction_metadata={
                            'formula': spec.formula,
                            'inputs': evaluation.inputs_used
                        }
                    )

                    logger.info(
                        f"Calculated {spec.data_id}: {evaluation.value} {spec.unit} "
                        f"using formula: {spec.formula}"
                    )
                else:
                    logger.error(
                        f"Calculation failed for {spec.data_id}: {evaluation.error}"
                    )
                    results[spec.data_id] = DataPoint(
                        metric_id=spec.data_id,
                        value=None,
                        unit=spec.unit,
                        confidence=0.0,
                        source_observations=[],
                        extraction_method='calculation_failed',
                        error=evaluation.error
                    )

            except Exception as e:
                logger.error(
                    f"Error calculating {spec.data_id}: {e}",
                    exc_info=True
                )
                results[spec.data_id] = DataPoint(
                    metric_id=spec.data_id,
                    value=None,
                    unit=spec.unit,
                    confidence=0.0,
                    source_observations=[],
                    extraction_method='error',
                    error=str(e)
                )

        return results


__all__ = ['CalculationAgent']
