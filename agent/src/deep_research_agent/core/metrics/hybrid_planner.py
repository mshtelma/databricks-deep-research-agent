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

    # ✅ CRITICAL FIX: Normalize observations to StructuredObservation objects
    # LangGraph may serialize them to dicts between nodes, so we normalize at entry
    from deep_research_agent.core.observation_converter import ObservationConverter
    observations = ObservationConverter.normalize_list(observations)

    # ✅ ULTRA-PROMINENT LOGGING FOR BUG 4 INVESTIGATION
    logger.error(
        f"\n{'#'*100}\n"
        f"{'#'*100}\n"
        f"🔥🔥🔥 [HYBRID PLANNER] create_unified_plan_hybrid() CALLED!!! 🔥🔥🔥\n"
        f"{'#'*100}\n"
        f"  - Observations: {len(observations)}\n"
        f"  - Entities: {len(constraints.entities)} → {constraints.entities}\n"
        f"  - Metrics: {len(constraints.metrics)} → {constraints.metrics}\n"
        f"  - Scenarios: {len(constraints.scenarios)}\n"
        f"  - User request: {user_request[:150]}...\n"
        f"{'#'*100}\n"
        f"{'#'*100}\n"
    )

    logger.info(
        f"🔍 [HYBRID PLANNER] ===== ENTRY POINT =====\n"
        f"  - Observations: {len(observations)}\n"
        f"  - Entities: {len(constraints.entities)} → {constraints.entities[:5]}{'...' if len(constraints.entities) > 5 else ''}\n"
        f"  - Metrics: {len(constraints.metrics)} → {constraints.metrics}\n"
        f"  - Scenarios: {len(constraints.scenarios)}\n"
        f"  - User request: {user_request[:100]}..."
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

    # 🔥🔥🔥 Step 4: Expand calculations programmatically 🔥🔥🔥
    logger.error(
        f"\n{'='*120}\n"
        f"🔥 [STEP 4] ABOUT TO CALL expand_calculations()!\n"
        f"{'='*120}\n"
        f"  - INPUT formulas: {len(patterns.calculation_formulas)}\n"
        f"  - INPUT formulas list: {[f.metric for f in patterns.calculation_formulas]}\n"
        f"  - INPUT entities: {len(constraints.entities)}\n"
        f"  - INPUT metric_specs available: {len(metric_specs)}\n"
        f"{'='*120}\n"
    )

    calculation_specs = expand_calculations(
        patterns.calculation_formulas,  # Already validated CalculationFormula objects
        constraints,
        metric_specs
    )

    logger.error(
        f"\n{'='*120}\n"
        f"🔥 [STEP 4] expand_calculations() RETURNED!\n"
        f"{'='*120}\n"
        f"  - OUTPUT calculation_specs: {len(calculation_specs)}\n"
        f"  - OUTPUT spec IDs: {list(calculation_specs.keys())[:10]}{'...' if len(calculation_specs) > 10 else ''}\n"
        f"{'='*120}\n"
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
    # ✅ CRITICAL FIX: Use attribute access after normalization (observations are StructuredObservation objects)
    obs_sample = [
        (o.content or '')[:OBS_CONTENT_PREVIEW]
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
        # ✅ CRITICAL FIX: Use json_schema for strict schema enforcement (was json_mode)
        structured_llm = llm.with_structured_output(
            schema=PatternExtractionOutput,
            method="json_schema"
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

        # 🔥🔥🔥 ULTRA-DETAILED LOGGING FOR BUG #2 INVESTIGATION 🔥🔥🔥
        logger.error(
            f"\n{'='*120}\n"
            f"🔥 [PATTERN EXTRACTION] LLM RESPONSE RECEIVED!\n"
            f"{'='*120}\n"
            f"📊 EXTRACTABLE METRICS: {len(patterns.extractable_metrics)}\n"
            f"   → {patterns.extractable_metrics}\n"
            f"\n"
            f"🧮 CALCULATION FORMULAS: {len(patterns.calculation_formulas)}\n"
        )

        if patterns.calculation_formulas:
            logger.error("   FORMULA DETAILS:")
            for idx, formula in enumerate(patterns.calculation_formulas, 1):
                logger.error(
                    f"   [{idx}] {formula.metric}\n"
                    f"       - formula: {formula.formula}\n"
                    f"       - inputs: {formula.inputs}\n"
                    f"       - per_scenario: {formula.per_scenario}"
                )
        else:
            logger.error("   ⚠️  NO FORMULAS RETURNED BY LLM!")

        logger.error(
            f"\n"
            f"💡 EXTRACTION HINTS: {len(patterns.extraction_hints)} metrics\n"
            f"   → {list(patterns.extraction_hints.keys())}\n"
            f"{'='*120}\n"
        )

        # Debug logging (original)
        logger.info(
            f"✅ Pattern extraction response: "
            f"extractable_metrics={patterns.extractable_metrics}, "
            f"calculation_formulas={[f.metric for f in patterns.calculation_formulas]}, "
            f"extraction_hints keys={list(patterns.extraction_hints.keys())}"
        )

        # 🔧 CRITICAL FIX: Ensure ALL metrics from constraints are extractable
        # The LLM sometimes classifies metrics as "calculation_formulas" only,
        # but we need to extract/derive their values first before calculating!
        #
        # Before fix: LLM returned extractable_metrics=['net_take_home', 'effective_tax_rate']
        #             but calculation_formulas included all 6 metrics
        # After fix:  extractable_metrics will include ALL 6 metrics
        if not patterns.extractable_metrics:
            logger.warning(
                f"LLM returned empty extractable_metrics. "
                f"Using fallback: all {len(constraints.metrics)} metrics are extractable"
            )
            patterns.extractable_metrics = constraints.metrics.copy()
        else:
            # Add any metrics from calculation_formulas that aren't in extractable_metrics
            formula_metrics = {f.metric for f in patterns.calculation_formulas}
            missing_metrics = formula_metrics - set(patterns.extractable_metrics)

            if missing_metrics:
                logger.warning(
                    f"🔧 [BUG FIX] LLM classified {len(missing_metrics)} metrics as calculation-only "
                    f"but they need extraction first! Adding to extractable_metrics: {missing_metrics}"
                )
                patterns.extractable_metrics.extend(list(missing_metrics))

            # Also ensure ALL constraint metrics are extractable (defensive)
            constraint_metrics_missing = set(constraints.metrics) - set(patterns.extractable_metrics)
            if constraint_metrics_missing:
                logger.warning(
                    f"🔧 [BUG FIX] {len(constraint_metrics_missing)} constraint metrics missing from extractable! "
                    f"Adding: {constraint_metrics_missing}"
                )
                patterns.extractable_metrics.extend(list(constraint_metrics_missing))

        logger.info(
            f"🔍 [HYBRID PLANNER] ===== PATTERN EXTRACTION COMPLETE =====\n"
            f"  - Extractable metrics ({len(patterns.extractable_metrics)}): {patterns.extractable_metrics}\n"
            f"  - Calculation formulas ({len(patterns.calculation_formulas)}): {[f.metric for f in patterns.calculation_formulas]}\n"
            f"  - Extraction hints keys: {list(patterns.extraction_hints.keys())[:10]}"
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

    # CRITICAL DIAGNOSTIC LOGGING - Entry point (WARNING level to ensure it shows in logs)
    logger.warning(
        f"\n🔍🔍🔍 [EXTRACT VALUES] ===== ENTRY POINT =====\n"
        f"  - Observations received: {len(observations)}\n"
        f"  - Observation types: {[type(o).__name__ for o in observations[:3]]}\n"
        f"  - Entities to process: {len(constraints.entities)} → {constraints.entities}\n"
        f"  - Extractable metrics: {patterns.extractable_metrics}\n"
        f"  - Calculation formulas: {[f.metric for f in patterns.calculation_formulas]}"
    )

    # 🔍 BUG 4 DIAGNOSTIC: Verify LLM type to understand why ERROR logs aren't appearing
    logger.error("=" * 100)
    logger.error(f"🔍🔍🔍 [BUG 4 DIAGNOSTIC] LLM TYPE VERIFICATION")
    logger.error(f"  - llm type: {type(llm).__name__}")
    logger.error(f"  - llm module: {type(llm).__module__}")
    logger.error(f"  - is RateLimitedChatModel: {'RateLimitedChatModel' in type(llm).__name__}")
    logger.error(f"  - has with_structured_output: {hasattr(llm, 'with_structured_output')}")
    if hasattr(llm, 'model_selector'):
        logger.error(f"  - has model_selector: {llm.model_selector is not None}")
    logger.error("=" * 100)

    metric_specs = {}

    # ✅ REMOVED: Unnecessary metric keyword filtering system
    # Observations are already tagged by researcher agent - we trust that tagging!

    for entity_idx, entity in enumerate(constraints.entities):
        entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

        logger.warning(
            f"\n🔍 [EXTRACT VALUES] ===== PROCESSING ENTITY {entity_idx + 1}/{len(constraints.entities)}: {entity} =====")

        logger.info(
            f"[ENTITY FILTER] Entity '{entity}' needs {len(patterns.extractable_metrics)} metrics: {patterns.extractable_metrics}"
        )

        # Filter observations for this entity by entity_tags
        # Trust the researcher agent's tagging - no keyword filtering needed!
        entity_observations = []
        obs_id_map = {}  # Maps index -> observation_id for source tracking

        for obs in observations:
            # ✅ Use attribute access (observations are StructuredObservation objects)
            content = obs.content or ''
            content_lower = content.lower()
            entity_tags = obs.entity_tags or []

            # Check if observation is tagged with this entity
            entity_match = entity in entity_tags or constraints.matches_entity(content, entity)

            if not entity_match:
                continue

            # ✅ CRITICAL FIX: TRUST THE ENTITY TAGS!
            # Observations are already tagged by the researcher agent as relevant to this entity.
            # We should NOT second-guess that tagging with keyword filtering.
            # If an observation is tagged with an entity, it's relevant - USE IT!

            # Track observation ID
            # ✅ Use attribute access (step_id or source_id)
            obs_id = obs.step_id or obs.source_id or f'obs_{len(entity_observations)}'
            idx = len(entity_observations)

            # Track observation ID by index
            obs_id_map[idx] = obs_id

            entity_observations.append({
                'index': idx,  # Include index for LLM reference
                'content': content[:OBS_CONTENT_FOR_EXTRACTION],
                # ✅ Use attribute access for metric_values
                'metric_values': obs.metric_values or {},
            })

            logger.debug(
                f"  ✓ Included {obs_id} for '{entity}' (entity-tagged observation)"
            )

        # CRITICAL DIAGNOSTIC: Log observation filtering results
        logger.warning(
            f"🔍🔍🔍 [EXTRACT VALUES] Observation filtering for '{entity}':\n"
            f"  - Total observations available: {len(observations)}\n"
            f"  - Observations matched: {len(entity_observations)}\n"
            f"  - First 3 observation samples: {[obs.get('content', '')[:50] for obs in entity_observations[:3]]}"
        )

        if not entity_observations:
            logger.warning(
                f"🔍 [EXTRACT VALUES] ⚠️ ZERO OBSERVATIONS matched for entity '{entity}'!\n"
                f"  - Total observations in state: {len(observations)}\n"
                f"  - Observations with entity_tags: {sum(1 for o in observations if o.entity_tags)}\n"
                f"  - This entity will have NO metric_specs!"
            )
            continue

        # Log selected observations count
        logger.info(
            f"🔍 [HYBRID PLANNER] Selected {len(entity_observations)} observations for '{entity}'\n"
            f"  - All entity-tagged observations included (trusting researcher agent tagging)"
        )

        # CRITICAL: Handle large observation sets
        total_obs = len(entity_observations)
        if total_obs > MAX_OBS_PER_ENTITY:
            # Keep first MAX_OBS_PER_ENTITY observations (already filtered by entity match)
            logger.warning(
                f"⚠️ {entity} has {total_obs} observations, keeping first {MAX_OBS_PER_ENTITY}. "
                f"Consider increasing MAX_OBS_PER_ENTITY if needed."
            )
            entity_observations = entity_observations[:MAX_OBS_PER_ENTITY]
        else:
            logger.info(f"[ENTITY FILTER] Processing {total_obs} observations for {entity}")

        # PHASE 1: Extract values from pre-existing metric_values (FAST PATH)
        # This uses values already extracted by the researcher agent
        direct_extracted_count = 0
        direct_extracted_metrics = set()

        # 🔍🔍🔍 ULTRA-DIAGNOSTIC: Log right before PHASE 1 loop
        logger.warning(
            f"🔍🔍🔍 [EXTRACT VALUES] ===== STARTING PHASE 1 LOOP for {entity} =====\n"
            f"  - entity_observations count: {len(entity_observations)}\n"
            f"  - entity_observations type: {type(entity_observations)}\n"
            f"  - patterns.extractable_metrics: {patterns.extractable_metrics}\n"
            f"  - About to iterate over {len(patterns.extractable_metrics)} metrics..."
        )

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

        logger.warning(
            f"🔍🔍🔍 [EXTRACT VALUES] PHASE 1 (FAST PATH) COMPLETE for {entity}:\n"
            f"  - Directly extracted: {direct_extracted_count}/{len(patterns.extractable_metrics)} metrics\n"
            f"  - Extracted metrics: {list(direct_extracted_metrics)}\n"
            f"  - Metric_specs created so far: {len([k for k in metric_specs.keys() if entity_clean in k])}"
        )

        # PHASE 2: LLM extraction for remaining metrics (SLOW PATH - only if needed)
        metrics_needing_llm = [m for m in patterns.extractable_metrics if m not in direct_extracted_metrics]

        if not metrics_needing_llm:
            logger.warning(f"🔍🔍🔍 [EXTRACT VALUES] SKIP LLM - All metrics for {entity} were directly extracted")
            continue

        logger.warning(
            f"🔍🔍🔍 [EXTRACT VALUES] PHASE 2 (SLOW PATH) STARTING for {entity}:\n"
            f"  - Metrics needing LLM: {len(metrics_needing_llm)} → {metrics_needing_llm}\n"
            f"  - Observations available: {len(entity_observations)}\n"
            f"  - Calling LLM with EntityMetricsOutput schema..."
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

        # 🔍 ULTRA DETAILED LOGGING - Log ALL observations with FULL content
        logger.error("=" * 120)
        logger.error(f"🔍🔍🔍 [EXTRACTION DEBUG] FULL OBSERVATIONS FOR {entity}")
        logger.error(f"  - Total observations: {len(entity_observations)}")
        logger.error(f"  - Metrics requested: {metrics_needing_llm}")
        logger.error("=" * 120)
        for i, o in enumerate(entity_observations[:5]):  # First 5 for readability
            logger.error(f"\n[Observation {i}]:")
            logger.error(f"  Index: {o.get('index', 'N/A')}")
            logger.error(f"  Content length: {len(o.get('content', ''))} chars")
            logger.error(f"  Content: {o.get('content', '')[:500]}...")  # First 500 chars
            if o.get('metric_values'):
                logger.error(f"  Pre-extracted values: {o['metric_values']}")
        logger.error("=" * 120)

        system_prompt = f"""Extract metric values for {entity} from the provided observations.

METRICS TO EXTRACT: {', '.join(metrics_needing_llm)}
{hints_text}

EXTRACTION GUIDELINES:
1. Look for NATURAL LANGUAGE descriptions that match the metric concept:
   - net_take_home: "net take-home", "net salary", "take-home pay", "net pay", "after-tax income"
   - effective_tax_rate: "effective tax rate", "overall tax rate", "total tax percentage"
   - annual_rent: "annual rent", "yearly rent", "rent per year"
   - disposable_income: "disposable income", "after-tax income", "spending money"
   - family_benefits: "family benefits", "child benefits", "family allowance"
   - daycare_cost: "daycare cost", "childcare cost", "nursery fees"

2. Extract from context (e.g., "leaving a net take-home of €X" → net_take_home: X)

3. If multiple values exist for a metric, prefer the most specific/detailed one

4. Extract ALL metrics you can find - it's better to extract with low confidence than return null

5. Return numeric values only (strip currency symbols like €, £, units, %)

6. For each value, specify which observation (by index number) it came from

EXAMPLES:
- "net take-home of €2,647" → {{"net_take_home": 2647, "confidence": 0.9}}
- "effective tax rate of 7.64%" → {{"effective_tax_rate": 7.64, "confidence": 0.9}}
- "annual rent of €18,600" → {{"annual_rent": 18600, "confidence": 0.9}}

OUTPUT FORMAT:
{{
  "entity": "{entity}",
  "extracted_values": {{
    "metric_name": numeric_value,
    ...
  }},
  "confidence": {{
    "metric_name": 0.0_to_1.0,
    ...
  }},
  "observation_sources": {{
    "metric_name": observation_index,
    ...
  }}
}}

IMPORTANT: Extract all metrics you can find. Don't leave extracted_values empty unless truly no metrics are found."""

        human_prompt = f"""TASK: Extract metric values for {entity}

OBSERVATIONS (numbered with index):
{chr(10).join(obs_lines)}

METRICS TO EXTRACT:
{', '.join(metrics_needing_llm)}

INSTRUCTIONS:
- Read through ALL {len(entity_observations)} observations
- Extract EVERY metric you can find
- Note the observation index for each extracted value
- Return JSON with extracted_values, confidence, and observation_sources
- Don't return empty - extract what you can find!"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # 🔍 ULTRA DETAILED LOGGING - Log FULL PROMPT being sent to LLM
        logger.error("=" * 120)
        logger.error(f"🔍🔍🔍 [EXTRACTION DEBUG] FULL PROMPT FOR {entity}")
        logger.error("=" * 120)
        logger.error("\n📋 SYSTEM PROMPT:")
        logger.error(system_prompt)
        logger.error("\n" + "=" * 120)
        logger.error("\n📋 HUMAN PROMPT:")
        logger.error(human_prompt)
        logger.error("\n" + "=" * 120)
        logger.error(f"\n📊 PROMPT STATS:")
        logger.error(f"  - System prompt length: {len(system_prompt)} chars")
        logger.error(f"  - Human prompt length: {len(human_prompt)} chars")
        logger.error(f"  - Total observations in prompt: {len(entity_observations)}")
        logger.error(f"  - Metrics to extract: {metrics_needing_llm}")
        logger.error("=" * 120)

        try:
            # Use native structured generation
            # ✅ CRITICAL FIX: Use json_schema for strict schema enforcement (was json_mode)
            structured_llm = llm.with_structured_output(
                schema=EntityMetricsOutput,
                method="json_schema"
            )

            # 🔍 BUG 4 DIAGNOSTIC: Verify structured_llm type
            logger.error("=" * 100)
            logger.error(f"🔍🔍🔍 [BUG 4 DIAGNOSTIC] STRUCTURED LLM TYPE VERIFICATION for {entity}")
            logger.error(f"  - structured_llm type: {type(structured_llm).__name__}")
            logger.error(f"  - structured_llm module: {type(structured_llm).__module__}")
            logger.error(f"  - is StructuredOutputWrapper: {'StructuredOutputWrapper' in type(structured_llm).__name__}")
            logger.error(f"  - has ainvoke: {hasattr(structured_llm, 'ainvoke')}")
            if hasattr(structured_llm, 'ainvoke'):
                import inspect
                try:
                    source_file = inspect.getfile(structured_llm.ainvoke)
                    logger.error(f"  - ainvoke source: {source_file}")
                except Exception as e:
                    logger.error(f"  - ainvoke source: Unable to determine ({e})")
            logger.error("=" * 100)

            response = await structured_llm.ainvoke(messages)

            # 🔍 ULTRA DETAILED LOGGING - Log FULL LLM RESPONSE
            logger.error("=" * 120)
            logger.error(f"🔍🔍🔍 [EXTRACTION DEBUG] LLM RESPONSE FOR {entity}")
            logger.error("=" * 120)
            logger.error(f"  Response type: {type(response)}")
            logger.error(f"  Response: {response}")
            if isinstance(response, EntityMetricsOutput):
                logger.error(f"\n📊 PARSED EntityMetricsOutput:")
                logger.error(f"  - entity: {response.entity}")
                logger.error(f"  - extracted_values: {response.extracted_values}")
                logger.error(f"  - confidence: {response.confidence}")
                logger.error(f"  - observation_sources: {response.observation_sources}")
                logger.error(f"  - Number of metrics extracted: {len(response.extracted_values)}")

                # 🔍 CRITICAL: If empty, log WHY
                if not response.extracted_values:
                    logger.error(f"\n❌ WARNING: NO METRICS EXTRACTED for {entity}!")
                    logger.error(f"  - This means LLM returned valid JSON but with empty extracted_values")
                    logger.error(f"  - Possible reasons:")
                    logger.error(f"    1. Observations don't contain the requested metrics")
                    logger.error(f"    2. Metric names don't match observation content")
                    logger.error(f"    3. Prompt is unclear or confusing")
                    logger.error(f"    4. LLM is being too conservative")
            logger.error("=" * 120)

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
                f"🔍 [HYBRID PLANNER] PHASE 2 (SLOW PATH) COMPLETE for {entity}:\n"
                f"  - LLM extracted: {len(entity_output.extracted_values)} values\n"
                f"  - Values: {entity_output.extracted_values}\n"
                f"  - Confidences: {entity_output.confidence}\n"
                f"  - Observation sources: {entity_output.observation_sources}\n"
                f"  - Total metric_specs for this entity: {len([k for k in metric_specs.keys() if entity_clean in k])}"
            )

        except Exception as e:
            logger.error(
                f"🔍 [HYBRID PLANNER] ❌ PHASE 2 FAILED for {entity}:\n"
                f"  - Exception: {type(e).__name__}: {str(e)}\n"
                f"  - This entity will have incomplete metric_specs!",
                exc_info=True
            )
            continue

    logger.info(
        f"🔍 [HYBRID PLANNER] ===== VALUE EXTRACTION COMPLETE =====\n"
        f"  - Total metric_specs created: {len(metric_specs)}\n"
        f"  - Metric_spec IDs: {list(metric_specs.keys())[:20]}{'...' if len(metric_specs) > 20 else ''}"
    )

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

    logger.error(
        f"\n{'*'*120}\n"
        f"🔥🔥🔥 [expand_calculations] FUNCTION ENTRY!\n"
        f"{'*'*120}\n"
        f"  - calculation_formulas LENGTH: {len(calculation_formulas)}\n"
        f"  - calculation_formulas TYPE: {type(calculation_formulas)}\n"
        f"  - existing_specs: {len(existing_specs)} specs available\n"
        f"  - entities: {len(constraints.entities)}\n"
        f"{'*'*120}\n"
    )

    if not calculation_formulas:
        logger.error(
            f"🔥🔥🔥 [expand_calculations] NO FORMULAS PROVIDED! "
            f"Returning empty dict immediately."
        )
        return {}

    calculation_specs = {}

    for idx, formula_obj in enumerate(calculation_formulas):
        logger.error(
            f"\n{'~'*100}\n"
            f"🔥 [expand_calculations] PROCESSING FORMULA {idx+1}/{len(calculation_formulas)}\n"
            f"{'~'*100}\n"
            f"  - metric: {formula_obj.metric}\n"
            f"  - formula: {formula_obj.formula}\n"
            f"  - inputs: {formula_obj.inputs}\n"
            f"  - per_scenario: {formula_obj.per_scenario}\n"
            f"{'~'*100}\n"
        )
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
