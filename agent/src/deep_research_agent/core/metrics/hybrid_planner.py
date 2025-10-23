"""
Hybrid UnifiedPlan generation using structured LLM outputs.

Key improvements with structured generation:
1. No manual JSON parsing - Pydantic handles everything
2. Automatic validation of formulas and values
3. Type safety throughout
4. Cleaner error handling
5. No silent data loss - explicit limits and warnings
"""

from typing import Dict, List, Any, Optional
import logging
from ..constraint_system import QueryConstraints, ScenarioDefinition
from ..structured_models import (
    PatternExtractionOutput,
    CalculationFormula,
    EntityMetricsOutput
)
from .unified_models import (
    MetricSpec,
    SourceType,
    UnifiedPlan,
    ResponseTable,
    TableCell,
    RequestAnalysis
)
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Configuration constants - no silent data loss
# Phase 2: Fact Extraction
MAX_CONTENT_CHARS = 10000  # Per web page

# Phase 3a: Pattern Extraction (just identifying what exists)
OBS_SAMPLE_SIZE = 30        # How many observations to analyze
OBS_CONTENT_PREVIEW = 300   # Chars per observation for pattern detection

# Phase 3b: Value Extraction (actually extracting data)
OBS_CONTENT_FOR_EXTRACTION = 500  # More context needed for values
MAX_OBS_PER_ENTITY = 30           # Observations per entity


# ============================================================================
# Main Entry Point
# ============================================================================

async def create_unified_plan_hybrid(
    user_request: str,
    observations: List[Dict[str, Any]],
    constraints: QueryConstraints,
    llm: Any
) -> UnifiedPlan:
    """Create UnifiedPlan using hybrid approach with structured generation.

    Benefits of structured approach:
    1. Pattern extraction returns validated formulas
    2. Value extraction returns typed numeric values
    3. No manual JSON parsing or validation
    4. Automatic handling of edge cases via Pydantic
    """

    logger.info(
        f"Creating UnifiedPlan (hybrid with structured generation): "
        f"{len(observations)} observations, "
        f"{len(constraints.entities)} entities, "
        f"{len(constraints.metrics)} metrics, "
        f"{len(constraints.scenarios)} scenarios"
    )

    # Validate inputs
    if not constraints.entities:
        logger.warning("No entities in constraints")
        return UnifiedPlan(
            request_analysis=RequestAnalysis(
                what_user_wants=user_request,
                entities_to_compare=[],
                metrics_requested=[],
                comparison_dimensions=[]
            ),
            metric_specs={},
            response_tables=[]
        )

    # Step 1: Get patterns using structured generation
    patterns = await extract_patterns_structured(
        observations[:OBS_SAMPLE_SIZE],
        constraints,
        llm
    )

    # Step 2: Extract values using structured generation
    metric_specs = {}
    extraction_specs = await extract_values_structured(
        observations,
        constraints,
        llm,
        patterns
    )
    metric_specs.update(extraction_specs)

    # DEBUG: Log extraction results
    logger.debug(
        f"[HYBRID PLANNER] Step 2 extract_values_structured: "
        f"{len(extraction_specs)} metric_specs extracted"
    )
    if not extraction_specs:
        logger.warning(
            f"[HYBRID PLANNER] ⚠️ extract_values_structured returned EMPTY dict! "
            f"Observations: {len(observations)}, Entities: {len(constraints.entities)}"
        )

    # Step 3: Add scenario constants
    constant_specs = {}
    if constraints.has_scenarios():
        constant_specs = create_scenario_constants(constraints)
        metric_specs.update(constant_specs)
        logger.debug(
            f"[HYBRID PLANNER] Step 3 create_scenario_constants: "
            f"{len(constant_specs)} constant specs created"
        )

    # Step 4: Expand calculations programmatically
    calculation_specs = expand_calculations(
        patterns.calculation_formulas,  # Already validated CalculationFormula objects
        constraints,
        metric_specs
    )
    metric_specs.update(calculation_specs)

    # DEBUG: Log calculation expansion results
    logger.debug(
        f"[HYBRID PLANNER] Step 4 expand_calculations: "
        f"{len(calculation_specs)} calculation specs generated"
    )

    # Step 5: Build response tables
    tables = build_response_tables(constraints, metric_specs)

    # Step 6: Create request analysis
    analysis = RequestAnalysis(
        what_user_wants=user_request,
        entities_to_compare=constraints.entities,
        metrics_requested=constraints.metrics,
        comparison_dimensions=[s.name for s in constraints.scenarios] if constraints.has_scenarios() else []
    )

    # DEBUG: Log final metric_specs count before returning
    if not metric_specs:
        logger.error(
            f"[HYBRID PLANNER] CRITICAL: metric_specs is EMPTY after all phases! "
            f"extraction_specs={len(extraction_specs)}, "
            f"constant_specs={len(constant_specs)}, "
            f"calculation_specs={len(calculation_specs)}"
        )
    else:
        logger.info(
            f"[HYBRID PLANNER] Successfully created {len(metric_specs)} metric_specs "
            f"({len(extraction_specs)} extracted + {len(constant_specs)} constants + "
            f"{len(calculation_specs)} calculations)"
        )

    # Log statistics
    log_plan_statistics(metric_specs, tables, constraints)

    return UnifiedPlan(
        request_analysis=analysis,
        metric_specs=metric_specs,
        response_tables=tables,
        narrative_points=generate_narrative_points(constraints, metric_specs)
    )


# ============================================================================
# Step 1: Pattern Extraction with Structured Generation
# ============================================================================

async def extract_patterns_structured(
    sample_observations: List[Dict],
    constraints: QueryConstraints,
    llm: Any
) -> PatternExtractionOutput:
    """Extract patterns using structured generation for reliability."""

    system_prompt = """You are a data analysis expert. Analyze observations to identify metric patterns.

Your task:
1. Identify which metrics can be directly extracted from text (e.g., "tax_rate: 35%")
2. Identify which metrics need calculation (e.g., "net_income = salary - tax")
3. Provide formulas for calculated metrics
4. Provide extraction hints (synonyms, variations) for finding metrics

IMPORTANT: Return your analysis in JSON format with this exact structure:
{
  "extractable_metrics": ["metric1", "metric2", ...],
  "calculation_formulas": [
    {
      "metric": "calculated_metric_name",
      "formula": "input1 - input2",
      "inputs": ["input1", "input2"],
      "per_scenario": false
    }
  ],
  "extraction_hints": {
    "metric1": ["synonym1", "synonym2"],
    "metric2": ["variation1", "variation2"]
  }
}

If you identify that all requested metrics can be extracted directly, list them all in extractable_metrics."""

    # Build context for pattern identification
    # NOTE: We truncate observations here because we only need to identify PATTERNS,
    # not extract all values. Full observations will be used in value extraction phase.
    obs_sample = [
        o.get('content', '')[:OBS_CONTENT_PREVIEW]
        for o in sample_observations[:OBS_SAMPLE_SIZE]
    ]

    logger.info(
        f"Pattern extraction analyzing {len(obs_sample)} observations "
        f"({OBS_CONTENT_PREVIEW} chars each)"
    )

    scenario_context = ""
    if constraints.has_scenarios():
        scenario_context = f"""
Scenarios present: {len(constraints.scenarios)}
- {', '.join([f"{s.name} ({list(s.parameters.keys())})" for s in constraints.scenarios])}
Mark calculations as per_scenario=true if they use scenario parameters."""

    # WARNING: Don't truncate entity list - LLM needs to see full scope
    entity_context = f"- Entities ({len(constraints.entities)}): {', '.join(constraints.entities)}"
    if len(constraints.entities) > 20:
        logger.warning(f"Large entity list ({len(constraints.entities)}) may impact prompt size")

    human_prompt = f"""Analyze these observations to identify extraction and calculation patterns.
Return the results as JSON with the required schema.

Sample observations:
{chr(10).join(obs_sample[:10])}

Requirements:
{entity_context}
- Metrics needed: {', '.join(constraints.metrics)}
- Comparison type: {constraints.comparison_type}
{scenario_context}

Identify:
1. Which metrics can be directly extracted (tax_rate, price, etc.)
2. Which metrics need calculation (net_income, effective_rate, etc.)
3. Formulas for calculations
4. Extraction hints (synonyms) for finding metrics

Return your analysis in JSON format."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    try:
        # Use native structured generation
        structured_llm = llm.with_structured_output(
            schema=PatternExtractionOutput,
            method="json_mode"
        )

        response = await structured_llm.ainvoke(messages)

        # with_structured_output() should ALWAYS return validated Pydantic model
        if not isinstance(response, PatternExtractionOutput):
            logger.error(
                f"Unexpected response type for pattern extraction: {type(response)}. "
                "Using safe fallback."
            )
            # Return safe fallback - assume all metrics are extractable
            return PatternExtractionOutput(
                extractable_metrics=constraints.metrics,
                calculation_formulas=[],
                extraction_hints={}
            )

        patterns = response

        # Debug logging
        logger.info(
            f"✅ Pattern extraction response: "
            f"extractable_metrics={patterns.extractable_metrics}, "
            f"calculation_formulas={[f.metric for f in patterns.calculation_formulas]}, "
            f"extraction_hints keys={list(patterns.extraction_hints.keys())}"
        )

        # If LLM returned empty extractable_metrics, use fallback
        if not patterns.extractable_metrics:
            logger.warning(
                f"LLM returned empty extractable_metrics. "
                f"Using fallback: all {len(constraints.metrics)} metrics are extractable"
            )
            patterns.extractable_metrics = constraints.metrics.copy()

        logger.info(
            f"✅ Final patterns: "
            f"{len(patterns.extractable_metrics)} extractable metrics, "
            f"{len(patterns.calculation_formulas)} formulas"
        )

        return patterns

    except Exception as e:
        logger.error(f"❌ Pattern extraction failed: {e}")
        # Return safe fallback - assume all metrics are extractable
        return PatternExtractionOutput(
            extractable_metrics=constraints.metrics,
            calculation_formulas=[],
            extraction_hints={}
        )


# ============================================================================
# Step 2: Value Extraction with Structured Generation
# ============================================================================

async def extract_values_structured(
    observations: List[Dict],
    constraints: QueryConstraints,
    llm: Any,
    patterns: PatternExtractionOutput
) -> Dict[str, MetricSpec]:
    """Extract values using structured generation for each entity."""

    metric_specs = {}

    # Build metric keyword mapping for METRIC-AWARE filtering
    metric_keyword_map = {
        'social_contrib': ['social', 'security', 'contribution', 'ahv', 'iv', 'eo', 'mandatory'],
        'tax_rate': ['tax', 'rate', 'income', 'personal', 'marginal', 'pit', 'bracket'],
        'effective_tax_rate': ['effective', 'tax', 'rate', 'total', 'overall'],
        'net_take_home': ['net', 'take-home', 'takehome', 'after tax', 'disposable'],
        'exchange_rate': ['exchange', 'rate', 'currency', 'conversion', 'chf', 'gbp', 'eur'],
        'daycare': ['daycare', 'child care', 'nursery', 'kindergarten', 'kita'],
        'rent': ['rent', 'rental', 'housing', 'apartment', 'accommodation'],
        'rsu': ['rsu', 'restricted stock', 'equity', 'vesting', 'stock option'],
        'family_benefits': ['family', 'benefit', 'allowance', 'child benefit', 'credit'],
        'disposable_income': ['disposable', 'income', 'after', 'expenses', 'remaining']
    }

    def _get_metric_keywords(metric_name: str) -> List[str]:
        """Extract keywords for a metric to improve observation matching."""
        metric_lower = metric_name.lower()
        keywords = []

        # Check predefined patterns
        for pattern, kw_list in metric_keyword_map.items():
            if pattern in metric_lower:
                keywords.extend(kw_list)

        # Add metric name words as keywords
        words = metric_lower.replace('_', ' ').split()
        keywords.extend([w for w in words if len(w) > 3])

        return list(set(keywords))  # Deduplicate

    for entity in constraints.entities:
        entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

        # Get ALL metrics we need to extract for this entity
        metrics_for_entity = patterns.extractable_metrics
        all_metric_keywords = set()
        for metric in metrics_for_entity:
            all_metric_keywords.update(_get_metric_keywords(metric))

        logger.info(
            f"[METRIC-AWARE FILTER] Entity '{entity}' needs metrics: {metrics_for_entity}. "
            f"Metric keywords: {sorted(list(all_metric_keywords)[:10])}..."
        )

        # Filter observations for this entity WITH METRIC-AWARE scoring
        # NOTE: For value extraction, we need MORE context than pattern extraction
        entity_observations = []
        obs_id_map = {}  # Maps index -> observation_id for source tracking

        for obs in observations:
            content = obs.get('content', '')
            content_lower = content.lower()
            entity_tags = obs.get('entity_tags', [])

            # Check if observation is tagged with this entity
            entity_match = entity in entity_tags or constraints.matches_entity(content, entity)

            if not entity_match:
                continue

            # METRIC-AWARE SCORING: Check if observation contains relevant metric keywords
            metric_score = sum(1 for kw in all_metric_keywords if kw in content_lower)

            # Log scoring details for debugging
            obs_id = obs.get('step_id', obs.get('id', f'obs_{len(entity_observations)}'))

            # Only include observations with sufficient metric keyword matches
            # OR include all if we have very few observations
            if metric_score >= 2 or len(entity_observations) < 5:
                idx = len(entity_observations)

                # Track observation ID by index
                obs_id_map[idx] = obs_id

                entity_observations.append({
                    'index': idx,  # Include index for LLM reference
                    'content': content[:OBS_CONTENT_FOR_EXTRACTION],
                    'metric_values': obs.get('metric_values', {}),
                    'metric_score': metric_score  # Add score for debugging
                })

                logger.debug(
                    f"  ✓ Included {obs_id} for '{entity}' (metric_score={metric_score})"
                )
            else:
                logger.debug(
                    f"  ✗ Filtered out {obs_id} for '{entity}' (metric_score={metric_score} < 2) "
                    f"- content preview: {content[:100]}..."
                )

        if not entity_observations:
            logger.warning(f"[METRIC-AWARE FILTER] No observations found for {entity}")
            continue

        # Log metric scores distribution
        scores = [obs['metric_score'] for obs in entity_observations]
        logger.info(
            f"[METRIC-AWARE FILTER] Selected {len(entity_observations)} observations for '{entity}'. "
            f"Metric scores: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}"
        )

        # CRITICAL: Handle large observation sets
        total_obs = len(entity_observations)
        if total_obs > MAX_OBS_PER_ENTITY:
            # Sort by metric_score DESC to keep most relevant observations
            entity_observations.sort(key=lambda x: x['metric_score'], reverse=True)

            logger.warning(
                f"⚠️ {entity} has {total_obs} observations, keeping top {MAX_OBS_PER_ENTITY} by metric_score. "
                f"Score range of kept observations: {entity_observations[0]['metric_score']} to {entity_observations[MAX_OBS_PER_ENTITY-1]['metric_score']}"
            )
            entity_observations = entity_observations[:MAX_OBS_PER_ENTITY]
        else:
            logger.info(f"[METRIC-AWARE FILTER] Processing {total_obs} observations for {entity}")

        # PHASE 1: Extract values from pre-existing metric_values (FAST PATH)
        # This uses values already extracted by the researcher agent
        direct_extracted_count = 0
        direct_extracted_metrics = set()

        for metric in patterns.extractable_metrics:
            best_value = None
            best_obs_idx = None
            best_confidence = 0.0

            # Search through observations for this metric in metric_values
            for obs in entity_observations:
                metric_values = obs.get('metric_values', {})
                if metric in metric_values:
                    value = metric_values[metric]
                    if value is not None and value != 'N/A':
                        # Found a pre-extracted value!
                        # Use highest confidence (we'll use first found for now)
                        if best_value is None:
                            best_value = value
                            best_obs_idx = obs['index']
                            best_confidence = 0.95  # High confidence for pre-extracted values

            if best_value is not None:
                spec_id = f"{metric}_{entity_clean}"
                obs_id = obs_id_map.get(best_obs_idx, f'obs_{entity_clean}_{metric}')

                metric_specs[spec_id] = MetricSpec(
                    data_id=spec_id,
                    source_type=SourceType.EXTRACT,
                    value=float(best_value),
                    observation_id=obs_id,
                    extraction_path=f"{entity} {metric}: {best_value} (pre-extracted)",
                    tags=constraints.get_entity_tags(entity),
                    confidence=best_confidence
                )
                direct_extracted_count += 1
                direct_extracted_metrics.add(metric)
                logger.info(
                    f"  ✅ Direct extraction: {entity}.{metric} = {best_value} from {obs_id}"
                )

        logger.info(
            f"[FAST PATH] Directly extracted {direct_extracted_count}/{len(patterns.extractable_metrics)} "
            f"metrics for {entity} from pre-existing metric_values"
        )

        # PHASE 2: LLM extraction for remaining metrics (SLOW PATH - only if needed)
        metrics_needing_llm = [m for m in patterns.extractable_metrics if m not in direct_extracted_metrics]

        if not metrics_needing_llm:
            logger.info(f"[SKIP LLM] All metrics for {entity} were directly extracted, skipping LLM call")
            continue

        logger.info(
            f"[LLM EXTRACTION] Need LLM for {len(metrics_needing_llm)} metrics: {metrics_needing_llm[:5]}..."
        )

        # Build hints from patterns
        hints_text = ""
        for metric, hints in patterns.extraction_hints.items():
            if metric in metrics_needing_llm:
                hints_text += f"\n- {metric}: look for {', '.join(hints)}"

        # Enhanced observation format that includes metric_values for context
        obs_lines = []
        for o in entity_observations:
            obs_line = f"[{o['index']}]: {o['content']}"
            if o.get('metric_values'):
                obs_line += f"\n    Pre-extracted values: {o['metric_values']}"
            obs_lines.append(obs_line)

        system_prompt = f"""Extract specific metric values for {entity}.

Focus ONLY on these metrics (others were already found): {', '.join(metrics_needing_llm)}
{hints_text}

Return exact numeric values when found, null when not found.

NOTE: Some observations include pre-extracted values. Use them if they match the metrics you need.

CRITICAL: For each metric value you extract, specify which observation (by index number) it came from.
This is essential for tracking data sources.

IMPORTANT: Return your analysis in JSON format with this structure:
{{
  "entity": "{entity}",
  "extracted_values": {{
    "metric_name1": numeric_value_or_null,
    "metric_name2": numeric_value_or_null
  }},
  "confidence": {{
    "metric_name1": 0.9,
    "metric_name2": 0.8
  }},
  "observation_sources": {{
    "metric_name1": 0,  // Index of observation containing this value
    "metric_name2": 2   // Index of observation containing this value
  }}
}}

If you can't determine the exact observation, use the first observation (index 0)."""

        human_prompt = f"""Extract metric values for {entity} from these observations.

Each observation is numbered with an index. Note which observation each value comes from.

Observations:
{chr(10).join(obs_lines)}

Entity: {entity}
Total observations: {len(entity_observations)}
Metrics to extract: {', '.join(metrics_needing_llm)}

Return extracted values with observation sources in JSON format."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        try:
            # Use native structured generation
            structured_llm = llm.with_structured_output(
                schema=EntityMetricsOutput,
                method="json_mode"
            )

            response = await structured_llm.ainvoke(messages)

            # with_structured_output() should ALWAYS return validated Pydantic model
            if not isinstance(response, EntityMetricsOutput):
                logger.error(
                    f"Unexpected response type for {entity}: {type(response)}. "
                    "Skipping this entity."
                )
                continue

            entity_output = response

            # Set entity if it's empty (LLM didn't include it)
            if not entity_output.entity:
                entity_output.entity = entity

            # Create MetricSpecs from validated output WITH observation tracking
            for metric, value in entity_output.extracted_values.items():
                if value is not None and metric in metrics_needing_llm:
                    spec_id = f"{metric}_{entity_clean}"

                    # Get the observation ID from the observation_sources mapping
                    obs_index = entity_output.observation_sources.get(metric, 0)

                    # Validate index and get observation ID
                    if obs_index in obs_id_map:
                        obs_id = obs_id_map[obs_index]
                    else:
                        # Fallback to first observation if index is invalid
                        logger.warning(
                            f"Invalid observation index {obs_index} for {metric}, "
                            f"using first observation"
                        )
                        obs_id = obs_id_map.get(0, f'unknown_{entity_clean}')

                    metric_specs[spec_id] = MetricSpec(
                        data_id=spec_id,
                        source_type=SourceType.EXTRACT,
                        value=value,  # Already validated as float by Pydantic
                        observation_id=obs_id,  # ✅ NOW WE TRACK THE SOURCE!
                        extraction_path=f"{entity} {metric}: {value} (LLM-extracted)",  # ✅ ADD CONTEXT
                        tags=constraints.get_entity_tags(entity),
                        confidence=entity_output.confidence.get(metric, 0.9)
                    )

                    logger.debug(
                        f"✅ Extracted {metric}={value} for {entity} "
                        f"from observation {obs_id} (index {obs_index})"
                    )

            logger.info(
                f"Extracted {len(entity_output.extracted_values)} values for {entity}"
            )

        except Exception as e:
            logger.error(f"Failed to extract values for {entity}: {e}")
            continue

    return metric_specs


# ============================================================================
# Step 3: Scenario Constants (no LLM needed)
# ============================================================================

def create_scenario_constants(constraints: QueryConstraints) -> Dict[str, MetricSpec]:
    """Create MetricSpecs for scenario parameters."""

    constant_specs = {}

    for scenario in constraints.scenarios:
        for param_name, param_value in scenario.parameters.items():
            param_clean = param_name.lower().replace(' ', '_')
            spec_id = f"{param_clean}_{scenario.id}"

            # Ensure value is numeric
            if isinstance(param_value, (int, float)):
                numeric_value = float(param_value)
            else:
                # Try to convert string to float
                try:
                    numeric_value = float(param_value)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-numeric parameter {param_name}={param_value}")
                    continue

            constant_specs[spec_id] = MetricSpec(
                data_id=spec_id,
                source_type=SourceType.CONSTANT,
                value=numeric_value,
                tags={'scenario': scenario.id, 'scenario_name': scenario.name},
                confidence=1.0
            )

    logger.info(f"Created {len(constant_specs)} scenario constants")
    return constant_specs


# ============================================================================
# Step 4: Calculation Expansion (programmatic, uses validated formulas)
# ============================================================================

def expand_calculations(
    calculation_formulas: List[CalculationFormula],  # Already validated by Pydantic
    constraints: QueryConstraints,
    existing_specs: Dict[str, MetricSpec]
) -> Dict[str, MetricSpec]:
    """Expand validated calculation formulas to all combinations."""

    calculation_specs = {}

    for formula_obj in calculation_formulas:
        metric_name = formula_obj.metric
        formula_template = formula_obj.formula
        inputs_template = formula_obj.inputs
        per_scenario = formula_obj.per_scenario

        if per_scenario and constraints.has_scenarios():
            # Expand for entity × scenario
            for entity in constraints.entities:
                entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

                for scenario in constraints.scenarios:
                    spec_id = f"{metric_name}_{entity_clean}_{scenario.id}"

                    # Build actual inputs and formula
                    actual_inputs = []
                    actual_formula = formula_template

                    for input_name in inputs_template:
                        if input_name in scenario.parameters:
                            # Scenario parameter
                            input_id = f"{input_name}_{scenario.id}"
                        else:
                            # Extracted metric for this entity
                            input_id = f"{input_name}_{entity_clean}"

                        actual_inputs.append(input_id)
                        actual_formula = actual_formula.replace(
                            input_name,
                            f"@{input_id}"
                        )

                    # Create spec if inputs exist
                    inputs_exist = all(
                        inp in existing_specs or inp in calculation_specs
                        for inp in actual_inputs
                    )

                    if inputs_exist or len(actual_inputs) == 0:
                        calculation_specs[spec_id] = MetricSpec(
                            data_id=spec_id,
                            source_type=SourceType.CALCULATE,
                            formula=actual_formula,
                            required_inputs=actual_inputs,
                            tags={
                                **constraints.get_entity_tags(entity),
                                'scenario': scenario.id,
                                'scenario_name': scenario.name
                            },
                            confidence=0.95
                        )
        else:
            # Non-scenario calculations
            for entity in constraints.entities:
                entity_clean = entity.lower().replace(' ', '_').replace('-', '_')
                spec_id = f"{metric_name}_{entity_clean}"

                actual_inputs = []
                actual_formula = formula_template

                for input_name in inputs_template:
                    input_id = f"{input_name}_{entity_clean}"
                    actual_inputs.append(input_id)
                    actual_formula = actual_formula.replace(input_name, f"@{input_id}")

                # Create if inputs exist
                inputs_exist = all(
                    inp in existing_specs or inp in calculation_specs
                    for inp in actual_inputs
                )

                if inputs_exist:
                    calculation_specs[spec_id] = MetricSpec(
                        data_id=spec_id,
                        source_type=SourceType.CALCULATE,
                        formula=actual_formula,
                        required_inputs=actual_inputs,
                        tags=constraints.get_entity_tags(entity),
                        confidence=0.95
                    )

    logger.info(f"Created {len(calculation_specs)} calculation specs")
    return calculation_specs


# ============================================================================
# Step 5: Table Building
# ============================================================================

def build_response_tables(
    constraints: QueryConstraints,
    metric_specs: Dict[str, MetricSpec]
) -> List[ResponseTable]:
    """Build response tables from MetricSpecs."""

    tables = []

    if constraints.has_scenarios():
        # Create one table per scenario
        for scenario in constraints.scenarios:
            cells = []

            for entity in constraints.entities:
                entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

                for metric in constraints.metrics:
                    # Try scenario-specific metric first
                    spec_id = f"{metric}_{entity_clean}_{scenario.id}"

                    if spec_id not in metric_specs:
                        # Fall back to entity-only metric
                        spec_id = f"{metric}_{entity_clean}"

                    if spec_id in metric_specs:
                        spec = metric_specs[spec_id]
                        if spec.source_type == SourceType.CALCULATE:
                            value = "=CALC="
                        elif spec.value is not None:
                            value = str(spec.value)
                        else:
                            value = "N/A"
                        confidence = spec.confidence
                    else:
                        value = "N/A"
                        confidence = 0.0

                    # Create cell with unique ID
                    cell_id = f"{entity_clean}_{metric}_{scenario.id}"
                    cells.append(TableCell(
                        cell_id=cell_id,
                        row=entity,
                        column=metric,
                        data_id=spec_id if spec_id in metric_specs else "",
                        answers_user_question=f"{metric} for {entity} in {scenario.name} scenario"
                    ))

            table_id = f"scenario_{scenario.id}"
            title = f"{scenario.name}: {scenario.description}" if scenario.description else scenario.name

            tables.append(ResponseTable(
                table_id=table_id,
                title=title,
                purpose=f"Show metrics for {scenario.name} scenario",
                rows=constraints.entities,
                columns=constraints.metrics,
                cells=cells,
                context={'scenario_id': scenario.id}
            ))
    else:
        # Single table without scenarios
        cells = []

        for entity in constraints.entities:
            entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

            for metric in constraints.metrics:
                spec_id = f"{metric}_{entity_clean}"

                if spec_id in metric_specs:
                    spec = metric_specs[spec_id]
                    if spec.source_type == SourceType.CALCULATE:
                        value = "=CALC="
                    elif spec.value is not None:
                        value = str(spec.value)
                    else:
                        value = "N/A"
                    confidence = spec.confidence
                else:
                    value = "N/A"
                    confidence = 0.0

                # Create cell with unique ID
                cell_id = f"{entity_clean}_{metric}"
                cells.append(TableCell(
                    cell_id=cell_id,
                    row=entity,
                    column=metric,
                    data_id=spec_id if spec_id in metric_specs else "",
                    answers_user_question=f"{metric} for {entity}"
                ))

        title = f"Comparison: {' vs '.join(constraints.entities[:3])}"
        if len(constraints.entities) > 3:
            title += f" and {len(constraints.entities) - 3} more"

        tables.append(ResponseTable(
            table_id="main_comparison",
            title=title,
            purpose="Compare entities across requested metrics",
            rows=constraints.entities,
            columns=constraints.metrics,
            cells=cells
        ))

    return tables


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_expected_cells(constraints: QueryConstraints) -> int:
    """Calculate expected number of table cells."""
    base_cells = len(constraints.entities) * len(constraints.metrics)
    if constraints.has_scenarios():
        return base_cells * len(constraints.scenarios)
    return base_cells


def generate_narrative_points(
    constraints: QueryConstraints,
    metric_specs: Dict[str, MetricSpec]
) -> List[str]:
    """Generate key narrative points from the data."""
    points = []

    extraction_count = sum(1 for s in metric_specs.values()
                          if s.source_type == SourceType.EXTRACT and s.value is not None)
    calc_count = sum(1 for s in metric_specs.values()
                    if s.source_type == SourceType.CALCULATE)

    points.append(f"Extracted {extraction_count} data points from observations")
    points.append(f"Defined {calc_count} calculations for derived metrics")

    if constraints.has_scenarios():
        points.append(f"Analyzing {len(constraints.scenarios)} scenarios across {len(constraints.entities)} entities")

    return points


def log_plan_statistics(
    metric_specs: Dict[str, MetricSpec],
    tables: List[ResponseTable],
    constraints: QueryConstraints
) -> None:
    """Log statistics about the created plan."""

    extractions = sum(1 for s in metric_specs.values()
                     if s.source_type == SourceType.EXTRACT)
    calculations = sum(1 for s in metric_specs.values()
                      if s.source_type == SourceType.CALCULATE)
    constants = sum(1 for s in metric_specs.values()
                   if s.source_type == SourceType.CONSTANT)

    total_cells = sum(len(t.cells) for t in tables)
    expected_cells = calculate_expected_cells(constraints)
    coverage = (total_cells / expected_cells * 100) if expected_cells > 0 else 0

    # Count populated cells (not N/A or =CALC=)
    populated = sum(
        1 for t in tables for c in t.cells
        if c.data_id and c.data_id in metric_specs
    )

    logger.info(f"""
=== UnifiedPlan Statistics ===
MetricSpecs: {len(metric_specs)} total
  - Extractions: {extractions}
  - Calculations: {calculations}
  - Constants: {constants}

Tables: {len(tables)}
  - Total cells: {total_cells}/{expected_cells} ({coverage:.1f}% coverage)
  - Populated with data_ids: {populated}
  - Missing data_ids: {total_cells - populated}
""")
