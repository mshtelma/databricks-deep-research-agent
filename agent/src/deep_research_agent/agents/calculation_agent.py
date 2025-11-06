"""
Calculation Agent: Executes UnifiedPlan for metric extraction and calculation.

This agent sits between Researcher and Reporter, taking the UnifiedPlan created
during planning and executing it to extract/calculate all required metrics.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from ..core import get_logger
from ..core.multi_agent_state import EnhancedResearchState, StateManager
from ..core.structured_models import ConstraintsOutput
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


def _parse_entity_and_metric(spec: MetricSpec) -> tuple[str, str]:
    """
    Extract entity and metric from MetricSpec.

    Args:
        spec: MetricSpec with data_id in format "{metric}_{entity}"
              or tags containing entity info

    Returns:
        Tuple of (entity, metric_name)

    Examples:
        "net_income_spain" -> ("Spain", "net income")
        "effective_tax_rate_france" -> ("France", "effective tax rate")
    """
    # Try extracting from tags first (most reliable)
    if spec.tags:
        # Look for entity-related tags
        entity = (
            spec.tags.get('country') or
            spec.tags.get('entity') or
            spec.tags.get('region') or
            spec.tags.get('company')
        )
        metric = spec.tags.get('metric')

        if entity and metric:
            return entity.title(), metric.replace('_', ' ')

    # Fallback: Parse data_id (format: "{metric}_{entity}")
    # Find last underscore to split entity from metric
    data_id = spec.data_id
    parts = data_id.rsplit('_', 1)  # Split from right, max 1 split

    if len(parts) == 2:
        metric_part, entity_part = parts
        # Clean up formatting
        entity = entity_part.replace('_', ' ').title()
        metric = metric_part.replace('_', ' ')
        return entity, metric

    # Last resort: use data_id as metric, "Unknown" as entity
    return "Unknown", data_id.replace('_', ' ')


def parse_scenario_string(scenario_str: str) -> Dict[str, Any]:
    """
    Parse a scenario string representation back into a dictionary.

    Example input:
    "id='s1' name='Single' description='...' parameters={'salary': 150000, 'rsu': 100000}"

    Returns properly structured dict with parameters as list of [key, value] pairs.
    """
    import re
    import ast

    result = {}

    # Extract id
    id_match = re.search(r"id='([^']+)'", scenario_str)
    if id_match:
        result['id'] = id_match.group(1)

    # Extract name
    name_match = re.search(r"name='([^']+)'", scenario_str)
    if name_match:
        result['name'] = name_match.group(1)

    # Extract description
    desc_match = re.search(r"description='((?:[^'\\]|\\.)*)'", scenario_str)
    if desc_match:
        result['description'] = desc_match.group(1)

    # Extract parameters - THE CRITICAL PART!
    params_dict_match = re.search(r"parameters=(\{[^}]+\})", scenario_str)
    params_list_match = re.search(r"parameters=(\[\[.*?\]\])", scenario_str)

    if params_dict_match:
        # Dictionary format: {'salary': 150000, 'rsu': 100000}
        params_str = params_dict_match.group(1)
        try:
            params = ast.literal_eval(params_str)
            # Convert dict to list format for consistency
            result['parameters'] = [[k, v] for k, v in params.items()]
        except (ValueError, SyntaxError):
            result['parameters'] = []
    elif params_list_match:
        # List format: [['salary', 150000], ['rsu', 100000]]
        params_str = params_list_match.group(1)
        try:
            result['parameters'] = ast.literal_eval(params_str)
        except (ValueError, SyntaxError):
            result['parameters'] = []
    else:
        result['parameters'] = []

    return result


def _ensure_pydantic_constraints(constraints: Union[Dict, ConstraintsOutput, None]) -> Optional[ConstraintsOutput]:
    """Convert dict query_constraints to Pydantic ConstraintsOutput at perimeter.

    Follows 'Pydantic at Perimeter' architecture:
    - LangGraph serializes Pydantic → dict between nodes
    - We convert dict → Pydantic at entry to calculation agent
    - Internal code always uses Pydantic objects

    Args:
        constraints: Dict or ConstraintsOutput from state

    Returns:
        ConstraintsOutput object or None
    """
    if constraints is None:
        return None

    if isinstance(constraints, ConstraintsOutput):
        # Already Pydantic object, pass through
        logger.debug("query_constraints is already ConstraintsOutput object")
        return constraints

    if isinstance(constraints, dict):
        try:
            logger.info(
                f"🔍 [PERIMETER] Converting dict query_constraints to ConstraintsOutput:\n"
                f"  - entities: {len(constraints.get('entities', []))}\n"
                f"  - metrics: {len(constraints.get('metrics', []))}\n"
                f"  - scenarios: {constraints.get('scenarios', [])}"
            )

            # Handle scenarios field - parse strings instead of discarding!
            scenarios = constraints.get('scenarios', [])
            if scenarios and isinstance(scenarios, list) and len(scenarios) > 0:
                # Check if first element is string (fixture artifact)
                if isinstance(scenarios[0], str):
                    logger.info(f"📝 [PERIMETER] Parsing {len(scenarios)} scenario strings from fixture")

                    # Parse each scenario string
                    parsed_scenarios = []
                    for scenario_str in scenarios:
                        parsed = parse_scenario_string(scenario_str)

                        # Log what we extracted
                        logger.info(
                            f"  ✅ Parsed scenario '{parsed.get('name', 'Unknown')}': "
                            f"{len(parsed.get('parameters', []))} parameters"
                        )

                        if parsed.get('parameters'):
                            for param in parsed['parameters']:
                                if isinstance(param, list) and len(param) >= 2:
                                    logger.debug(f"    - {param[0]}: {param[1]}")

                        parsed_scenarios.append(parsed)

                    # Replace string scenarios with parsed dicts
                    constraints_copy = dict(constraints)
                    constraints_copy['scenarios'] = parsed_scenarios

                    logger.info(
                        f"✅ [PERIMETER] Successfully parsed {len(parsed_scenarios)} scenarios "
                        f"with parameters intact!"
                    )
                else:
                    constraints_copy = constraints
            else:
                constraints_copy = constraints

            # Convert to Pydantic
            pydantic_constraints = ConstraintsOutput(**constraints_copy)
            logger.info("✅ [PERIMETER] Successfully converted to ConstraintsOutput")
            return pydantic_constraints

        except Exception as e:
            logger.error(
                f"❌ [PERIMETER] Failed to convert dict to ConstraintsOutput:\n"
                f"  - Exception: {type(e).__name__}: {str(e)}\n"
                f"  - Dict keys: {list(constraints.keys())}\n"
                f"  - Dict sample: {str(constraints)[:200]}...",
                exc_info=True
            )
            return None

    logger.warning(f"⚠️ [PERIMETER] Unexpected constraints type: {type(constraints)}")
    return None


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
        calc_config = self.config.get('agents', {}).get('calculation_agent', {})
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
        # ✅ CRITICAL FIX: Hydrate state to ensure observations are StructuredObservation objects
        # LangGraph serializes Pydantic objects to dicts between nodes
        from deep_research_agent.core.multi_agent_state import ensure_state_hydrated
        state = ensure_state_hydrated(state)

        if not self.enabled:
            logger.info("CalculationAgent disabled, skipping execution")
            existing_plan = state.get('unified_plan')
            return {
                "unified_plan": existing_plan.model_dump() if existing_plan and hasattr(existing_plan, 'model_dump') else existing_plan,
                "calculation_results": None
            }

        # Get UnifiedPlan from state
        unified_plan = state.get('unified_plan')

        # CRITICAL: Hydrate unified_plan if it's a dict (from serialization)
        # After model_dump(), LangGraph stores dicts, not Pydantic objects
        if unified_plan and isinstance(unified_plan, dict):
            try:
                unified_plan = UnifiedPlan(**unified_plan)
                logger.info(f"Hydrated UnifiedPlan from dict: {len(unified_plan.data_sources)} data sources")
            except Exception as e:
                logger.error(f"Failed to hydrate UnifiedPlan: {e}", exc_info=True)
                unified_plan = None

        if not unified_plan:
            # Check if we should create it (defensive check)
            if not self.config.get('metrics', {}).get('use_unified_planning', False):
                logger.warning("Unified planning disabled in config, skipping calculation")
                return {
                    "unified_plan": None,  # Explicitly None when disabled
                    "calculation_results": None
                }

            logger.info("No UnifiedPlan in state, attempting to create one...")

            # Get required inputs - convert dict to Pydantic at perimeter
            raw_constraints = state.get('query_constraints')
            constraints = _ensure_pydantic_constraints(raw_constraints)
            observations = state.get('observations', [])

            # Validate minimum requirements
            if not constraints:
                logger.warning("No query_constraints in state, cannot create UnifiedPlan")
                return {
                    "unified_plan": None,  # Can't create without constraints
                    "calculation_results": None
                }

            if not observations:
                logger.warning("No observations in state, cannot create UnifiedPlan")
                return {
                    "unified_plan": None,  # Can't create without data
                    "calculation_results": None
                }

            # Call hybrid planner directly (no router needed - config always forces hybrid mode)
            try:
                from ..core.metrics.hybrid_planner import create_unified_plan_hybrid

                # Limit observations to prevent excessive memory/token usage
                limited_observations = observations[:100] if len(observations) > 100 else observations

                logger.info(
                    f"🔍 [CALC AGENT] ATTEMPTING UnifiedPlan creation with:\n"
                    f"  - Observations: {len(limited_observations)}\n"
                    f"  - Entities: {len(constraints.entities) if hasattr(constraints, 'entities') else 0}\n"
                    f"  - Metrics: {len(constraints.metrics) if hasattr(constraints, 'metrics') else 0}\n"
                    f"  - Research topic: '{state.get('research_topic', '')}'\n"
                    f"  - LLM configured: {self.extraction_llm is not None}\n"
                    f"  - Config has metrics section: {'metrics' in self.config}"
                )

                # Log the actual constraints structure
                if constraints:
                    logger.info(
                        f"🔍 [CALC AGENT] QueryConstraints details:\n"
                        f"  - Entities: {[e.name if hasattr(e, 'name') else str(e) for e in (constraints.entities if hasattr(constraints, 'entities') else [])]}\n"
                        f"  - Metrics: {[m.name if hasattr(m, 'name') else str(m) for m in (constraints.metrics if hasattr(constraints, 'metrics') else [])]}"
                    )

                logger.error(
                    f"\n{'='*150}\n"
                    f"🚨🚨🚨 [CALC AGENT] ABOUT TO CALL create_unified_plan_hybrid() 🚨🚨🚨\n"
                    f"{'='*150}\n"
                    f"  - Function: {create_unified_plan_hybrid}\n"
                    f"  - Function module: {create_unified_plan_hybrid.__module__}\n"
                    f"  - Function file: {create_unified_plan_hybrid.__code__.co_filename if hasattr(create_unified_plan_hybrid, '__code__') else 'N/A'}\n"
                    f"  - Is coroutine: {asyncio.iscoroutinefunction(create_unified_plan_hybrid)}\n"
                    f"{'='*150}\n"
                )
                logger.info("🔍 [CALC AGENT] Calling create_unified_plan_hybrid() directly (simplified architecture)...")

                print("\n" + "="*150)
                print("🚨 CRITICAL: About to await create_unified_plan_hybrid()")
                print("="*150 + "\n")

                unified_plan = await create_unified_plan_hybrid(
                    user_request=state.get('research_topic', ''),
                    observations=limited_observations,
                    constraints=constraints,
                    llm=self.extraction_llm
                )

                print("\n" + "="*150)
                print(f"🚨 CRITICAL: create_unified_plan_hybrid() RETURNED: {type(unified_plan)}")
                print("="*150 + "\n")

                logger.error(
                    f"\n{'='*150}\n"
                    f"🚨🚨🚨 [CALC AGENT] create_unified_plan_hybrid() RETURNED 🚨🚨🚨\n"
                    f"{'='*150}\n"
                    f"  - Return type: {type(unified_plan)}\n"
                    f"  - Has metric_specs: {hasattr(unified_plan, 'metric_specs')}\n"
                    f"{'='*150}\n"
                )
                logger.info("🔍 [CALC AGENT] create_unified_plan_hybrid() returned successfully")

                # Store in state for Reporter to use
                state['unified_plan'] = unified_plan

                if unified_plan:
                    logger.info(
                        f"✅ [CALC AGENT] Created UnifiedPlan successfully:\n"
                        f"  - metric_specs: {len(unified_plan.metric_specs) if hasattr(unified_plan, 'metric_specs') else 'N/A'}\n"
                        f"  - response_tables: {len(unified_plan.response_tables) if hasattr(unified_plan, 'response_tables') else 'N/A'}\n"
                        f"  - data_sources: {len(unified_plan.data_sources) if hasattr(unified_plan, 'data_sources') else 'N/A'}"
                    )
                else:
                    logger.error("❌ [CALC AGENT] create_unified_plan_hybrid() returned None! This should NOT happen!")

            except ImportError as e:
                logger.error(
                    f"❌ [CALC AGENT] CRITICAL: Cannot import hybrid_planner module!\n"
                    f"  - Error: {e}\n"
                    f"  - This indicates a serious code issue - the module should exist"
                )
                logger.warning("Proceeding without calculations - reporter will use fallback strategy")
                return {
                    "unified_plan": None,
                    "calculation_results": None
                }
            except Exception as e:
                logger.error(
                    f"❌ [CALC AGENT] CRITICAL: Failed to create UnifiedPlan!\n"
                    f"  - Exception type: {type(e).__name__}\n"
                    f"  - Exception message: {str(e)}\n"
                    f"  - Full traceback below:",
                    exc_info=True
                )

                # Log additional diagnostic info
                logger.error(
                    f"❌ [CALC AGENT] Diagnostic information:\n"
                    f"  - LLM instance: {type(self.extraction_llm).__name__ if self.extraction_llm else 'None'}\n"
                    f"  - Config keys: {list(self.config.keys()) if self.config else 'None'}\n"
                    f"  - Constraints type: {type(constraints).__name__ if constraints else 'None'}\n"
                    f"  - Observations count: {len(limited_observations)}"
                )

                logger.warning("Proceeding without calculations - reporter will use fallback strategy")
                return {
                    "unified_plan": unified_plan.model_dump() if unified_plan and hasattr(unified_plan, 'model_dump') else unified_plan,
                    "calculation_results": None
                }

        # Get observations from state (may have been retrieved above)
        observations = state.get('observations', [])
        if not observations:
            logger.warning("No observations in state, cannot extract metrics")
            return {
                "unified_plan": unified_plan.model_dump() if unified_plan and hasattr(unified_plan, 'model_dump') else unified_plan,
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
            self.event_emitter.emit(
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
            self.event_emitter.emit(
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

        # CRITICAL: Return both unified_plan and calculation_results for LangGraph propagation
        # Direct state mutations don't propagate through Command(update=...) pattern
        # IMPORTANT: Serialize Pydantic models to dicts for proper LangGraph propagation
        return {
            "unified_plan": unified_plan.model_dump() if unified_plan and hasattr(unified_plan, 'model_dump') else unified_plan,
            "calculation_results": results
        }

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
            spec for spec in plan.data_sources.values()
            if spec.source_type == "extract"
        ]
        calculate_specs = [
            spec for spec in plan.data_sources.values()
            if spec.source_type == "calculate"
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

        # CRITICAL FIX (Bug #4): Return structure matching CalculationContext schema
        # This enables StateManager.hydrate_state() to convert dict → CalculationContext object
        # See: core/report_generation/models.py:CalculationContext for schema definition
        return {
            "extracted_data": [dp.model_dump() for dp in all_data.values()],  # ✅ FIX: Convert DataPoint objects to dicts
            "calculations": [],              # Required field (formulas/derived calculations)
            "key_comparisons": [],           # Required field (comparison entries for tables)
            "summary_insights": [],          # Required field (high-level insights)
            "data_quality_notes": [],        # Required field (data quality warnings)
            "metadata": {                    # Move stats to metadata dict
                "extracted_count": extracted_count,
                "calculated_count": calculated_count,
                "failed_count": failed_count,
                "plan_id": getattr(plan, 'plan_id', 'unknown'),
            }
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
                # ✅ CRITICAL FIX: Check if value already exists from UnifiedPlan
                # The hybrid planner extracts values during plan creation, so use them!
                if spec.value is not None and spec.confidence > 0.5:
                    logger.info(
                        f"✅ Using pre-extracted value for {spec.data_id}: {spec.value} "
                        f"(confidence: {spec.confidence:.2f}, source: unified_plan)"
                    )

                    # Create DataPoint from pre-extracted value
                    entity, metric = _parse_entity_and_metric(spec)
                    results[spec.data_id] = DataPoint(
                        # REQUIRED fields for reporter schema
                        entity=entity,
                        metric=metric,
                        value=spec.value,
                        # Presentation metadata
                        unit=spec.unit or "unitless",
                        confidence=spec.confidence,
                        # Technical provenance (optional)
                        metric_id=spec.data_id,
                        source_observations=[spec.observation_id] if spec.observation_id else [],
                        extraction_method='pre_extracted',
                        extraction_metadata={
                            'source': 'unified_plan',
                            'extraction_hint': spec.extraction_hint or ''
                        }
                    )
                    continue  # Skip re-extraction

                # Fallback: Extract from observations if value not pre-extracted
                logger.info(
                    f"🔍 No pre-extracted value for {spec.data_id}, attempting extraction "
                    f"(value: {spec.value}, confidence: {spec.confidence})"
                )

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
                    entity, metric = _parse_entity_and_metric(spec)
                    results[spec.data_id] = DataPoint(
                        # REQUIRED fields
                        entity=entity,
                        metric=metric,
                        value=None,
                        # Presentation metadata
                        unit=spec.unit or "unitless",
                        confidence=0.0,
                        # Technical provenance
                        metric_id=spec.data_id,
                        source_observations=[],
                        extraction_method='no_observation',
                        error="No suitable observation found"
                    )

            except Exception as e:
                logger.error(f"Error extracting {spec.data_id}: {e}", exc_info=True)
                entity, metric = _parse_entity_and_metric(spec)
                results[spec.data_id] = DataPoint(
                    # REQUIRED fields
                    entity=entity,
                    metric=metric,
                    value=None,
                    # Presentation metadata
                    unit=spec.unit or "unitless",
                    confidence=0.0,
                    # Technical provenance
                    metric_id=spec.data_id,
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
                    entity, metric = _parse_entity_and_metric(spec)
                    results[spec.data_id] = DataPoint(
                        # REQUIRED fields
                        entity=entity,
                        metric=metric,
                        value=None,
                        # Presentation metadata
                        unit=spec.unit or "unitless",
                        confidence=0.0,
                        # Technical provenance
                        metric_id=spec.data_id,
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
                    entity, metric = _parse_entity_and_metric(spec)
                    results[spec.data_id] = DataPoint(
                        # REQUIRED fields
                        entity=entity,
                        metric=metric,
                        value=None,
                        # Presentation metadata
                        unit=spec.unit or "unitless",
                        confidence=0.0,
                        # Technical provenance
                        metric_id=spec.data_id,
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

                    entity, metric = _parse_entity_and_metric(spec)
                    results[spec.data_id] = DataPoint(
                        # REQUIRED fields
                        entity=entity,
                        metric=metric,
                        value=evaluation.value,
                        # Presentation metadata
                        unit=spec.unit or "unitless",
                        confidence=0.95,  # High confidence for successful calculation
                        # Technical provenance
                        metric_id=spec.data_id,
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
                    entity, metric = _parse_entity_and_metric(spec)
                    results[spec.data_id] = DataPoint(
                        # REQUIRED fields
                        entity=entity,
                        metric=metric,
                        value=None,
                        # Presentation metadata
                        unit=spec.unit or "unitless",
                        confidence=0.0,
                        # Technical provenance
                        metric_id=spec.data_id,
                        source_observations=[],
                        extraction_method='calculation_failed',
                        error=evaluation.error
                    )

            except Exception as e:
                logger.error(
                    f"Error calculating {spec.data_id}: {e}",
                    exc_info=True
                )
                entity, metric = _parse_entity_and_metric(spec)
                results[spec.data_id] = DataPoint(
                    # REQUIRED fields
                    entity=entity,
                    metric=metric,
                    value=None,
                    # Presentation metadata
                    unit=spec.unit or "unitless",
                    confidence=0.0,
                    # Technical provenance
                    metric_id=spec.data_id,
                    source_observations=[],
                    extraction_method='error',
                    error=str(e)
                )

        return results


__all__ = ['CalculationAgent']
