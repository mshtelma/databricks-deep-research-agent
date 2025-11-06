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
import time
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
# CRITICAL FIX: Explicitly ensure logger is enabled and visible
logger.setLevel(logging.DEBUG)
logger.propagate = True  # Ensure logs propagate to root logger

# Add a console handler if none exists (the real fix for missing logs!)
if not logger.handlers:
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s", '
        '"module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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

    # 🚨🚨🚨 PRINT BEFORE ANYTHING ELSE 🚨🚨🚨
    print("\n" + "="*200)
    print("🔥🔥🔥 [HYBRID PLANNER] FUNCTION ENTERED - VERY FIRST LINE!!!")
    print("="*200 + "\n")

    # ✅ CRITICAL FIX: Normalize observations to StructuredObservation objects
    # LangGraph may serialize them to dicts between nodes, so we normalize at entry
    from deep_research_agent.core.observation_converter import ObservationConverter

    print("\n" + "="*200)
    print("🔥🔥🔥 [HYBRID PLANNER] ABOUT TO NORMALIZE OBSERVATIONS...")
    print("="*200 + "\n")

    observations = ObservationConverter.normalize_list(observations)

    print("\n" + "="*200)
    print("🔥🔥🔥 [HYBRID PLANNER] OBSERVATIONS NORMALIZED! About to log with logger.error()...")
    print("="*200 + "\n")

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

    # ========================================
    # PHASE 1: Extract ALL Values (Per Entity)
    # ========================================
    logger.info("📊 [PHASE 1] Starting value extraction for all entities")

    # 🔬 DIAGNOSTIC: Dump ALL observations with tags to file for analysis
    # CRITICAL FIX: Wrap in try-except to prevent diagnostic code from breaking production
    try:
        import json
        from pathlib import Path
        diagnostic_dir = Path("/tmp/diagnostic_observations")
        diagnostic_dir.mkdir(exist_ok=True)

        diagnostic_data = {
            "total_observations": len(observations),
            "requested_metrics": constraints.metrics,
            "entities": constraints.entities,
            # FIX: Serialize Pydantic ScenarioDefinition objects before JSON dump
            "scenarios": [s.model_dump() for s in constraints.scenarios] if constraints.scenarios else [],
            "observations": []
        }

        for i, obs in enumerate(observations):
            obs_info = {
                "index": i,
                "content_preview": (obs.content[:300] if hasattr(obs, 'content') and obs.content else "NO CONTENT"),
                "full_content_length": len(obs.content) if hasattr(obs, 'content') and obs.content else 0,
                "entity_tags": obs.entity_tags if hasattr(obs, 'entity_tags') else [],
                "confidence": obs.confidence if hasattr(obs, 'confidence') else None,
                "relevance_score": obs.relevance_score if hasattr(obs, 'relevance_score') else None,
                "metadata": obs.metadata if hasattr(obs, 'metadata') else {}
            }
            diagnostic_data["observations"].append(obs_info)

        diagnostic_file = diagnostic_dir / "observations_with_tags.json"
        with open(diagnostic_file, 'w') as f:
            json.dump(diagnostic_data, f, indent=2)

        logger.error(f"🔬 [DIAGNOSTIC] Dumped all observations to: {diagnostic_file}")
        logger.error(f"🔬 [DIAGNOSTIC] Total observations: {len(observations)}")
        logger.error(f"🔬 [DIAGNOSTIC] Entities: {constraints.entities}")
        logger.error(f"🔬 [DIAGNOSTIC] Requested metrics: {constraints.metrics}")
    except Exception as diagnostic_error:
        # Diagnostic code should never break production - log and continue
        logger.warning(f"⚠️ [DIAGNOSTIC] Failed to write diagnostic file (non-critical): {diagnostic_error}")

    all_metric_specs = {}
    entity_available_metrics = {}  # Track what we found per entity

    for entity in constraints.entities:
        entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

        # Filter observations for this entity
        entity_observations = []
        for obs in observations:
            entity_tags = obs.entity_tags or []
            if entity in entity_tags or constraints.matches_entity(obs.content or '', entity):
                entity_observations.append(obs)

        if not entity_observations:
            logger.warning(f"No observations for {entity}, skipping")
            continue

        logger.info(
            f"🔍 [PHASE 1 - {entity}] Found {len(entity_observations)} entity-specific observations"
        )

        # 🔬 DIAGNOSTIC: Show sample observation content and tags
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Sample observations (first 3):")
        for i, obs in enumerate(entity_observations[:3]):
            content_preview = (obs.content[:200] if hasattr(obs, 'content') and obs.content else "NO CONTENT")
            tags = obs.entity_tags if hasattr(obs, 'entity_tags') else []
            logger.error(f"  Obs {i+1}: tags={tags}")
            logger.error(f"         content: '{content_preview}...'")

        # 🔬 DIAGNOSTIC: Check if requested metrics are mentioned in entity observations
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Checking for requested metrics in observations:")
        entity_obs_text = "\n".join([
            obs.content for obs in entity_observations
            if hasattr(obs, 'content') and obs.content
        ]).lower()

        for metric in constraints.metrics:
            metric_keyword = metric.replace('_', ' ')
            mentions_in_entity_obs = entity_obs_text.count(metric_keyword)
            logger.error(f"  '{metric}': mentioned {mentions_in_entity_obs} times in {entity} observations")

        # Extract all possible values for this entity
        entity_specs = await extract_all_values_for_entity(
            entity=entity,
            entity_observations=entity_observations,
            requested_metrics=constraints.metrics,
            llm=llm
        )

        all_metric_specs.update(entity_specs)

        # Track what metrics we successfully extracted for this entity
        entity_available_metrics[entity] = {
            spec_id.replace(f'_{entity_clean}', '')
            for spec_id in entity_specs.keys()
        }

        logger.info(
            f"✅ [PHASE 1 - {entity}] Extracted {len(entity_specs)} values: "
            f"{list(entity_available_metrics[entity])}"
        )

    # ========================================
    # PHASE 2: Iterative Multi-Pass Formula Discovery
    # ========================================
    logger.info("🧮 [PHASE 2] Starting iterative formula discovery")

    # Configuration
    MAX_ITERATIONS = 3
    ITERATION_TIMEOUT_SECONDS = 30

    all_calculation_formulas = []
    discovered_metric_names = set()
    iteration_start_time = time.time()

    for iteration in range(MAX_ITERATIONS):
        # ========================================
        # Step 1: Compute Current Availability
        # ========================================
        
        # Metrics extracted from observations (global availability)
        extracted_metrics = set().union(*entity_available_metrics.values()) if entity_available_metrics else set()
        
        # Scenario parameters (always available)
        scenario_param_names = set()
        if constraints.has_scenarios():
            for scenario in constraints.scenarios:
                scenario_param_names.update(scenario.parameters.keys())
        
        # Metrics discovered in previous iterations (now available as inputs)
        # This is the KEY to iterative discovery
        currently_available = extracted_metrics | scenario_param_names | discovered_metric_names
        
        # What user requested
        requested_metrics_set = set(constraints.metrics)
        
        # What's still missing
        still_missing = requested_metrics_set - currently_available
        
        # ========================================
        # Step 2: Logging (Detailed State)
        # ========================================
        
        logger.info(
            f"\n{'═' * 80}\n"
            f"📊 [PHASE 2 - Iteration {iteration}] State:\n"
            f"{'═' * 80}\n"
            f"  Requested metrics: {sorted(requested_metrics_set)}\n"
            f"  Extracted from observations: {sorted(extracted_metrics)}\n"
            f"  Scenario parameters: {sorted(scenario_param_names)}\n"
            f"  Discovered (prev iterations): {sorted(discovered_metric_names)}\n"
            f"  Currently available (total): {sorted(currently_available)} ({len(currently_available)} items)\n"
            f"  Still missing: {sorted(still_missing)} ({len(still_missing)} items)\n"
            f"{'═' * 80}"
        )
        
        # ========================================
        # Step 3: Termination Check - Complete
        # ========================================
        
        if not still_missing:
            logger.info(
                f"✅ [PHASE 2 - Iteration {iteration}] All metrics available! No further discovery needed."
            )
            break
        
        # ========================================
        # Step 4: Termination Check - Timeout
        # ========================================
        
        elapsed = time.time() - iteration_start_time
        if elapsed > ITERATION_TIMEOUT_SECONDS:
            logger.warning(
                f"⏱️  [PHASE 2 - Iteration {iteration}] Discovery timeout ({elapsed:.1f}s > {ITERATION_TIMEOUT_SECONDS}s). "
                f"Stopping with {len(still_missing)} metrics unresolved."
            )
            break
        
        # ========================================
        # Step 5: Build Data Catalog for LLM
        # ========================================
        
        data_catalog = {
            "available_metrics": currently_available,  # GROWS each iteration
            "entity_specific": entity_available_metrics,
            "requested_but_missing": still_missing
        }
        
        logger.info(
            f"🔍 [PHASE 2 - Iteration {iteration}] Calling discover_feasible_formulas()...\n"
            f"  Attempting to create formulas for: {sorted(still_missing)}\n"
            f"  Using available inputs: {sorted(currently_available)}"
        )
        
        # ========================================
        # Step 6: Discover Formulas (LLM Call)
        # ========================================
        
        iteration_llm_start = time.time()
        
        try:
            new_formulas = await discover_feasible_formulas(
                data_catalog=data_catalog,
                requested_metrics=list(still_missing),
                constraints=constraints,
                observations=observations,
                llm=llm
            )
        except Exception as e:
            logger.error(
                f"❌ [PHASE 2 - Iteration {iteration}] discover_feasible_formulas() raised exception: {e}"
            )
            break
        
        iteration_llm_time = time.time() - iteration_llm_start
        
        # ========================================
        # Step 7: Termination Check - No Progress
        # ========================================
        
        if not new_formulas or len(new_formulas) == 0:
            logger.warning(
                f"\n{'!' * 80}\n"
                f"⚠️  [PHASE 2 - Iteration {iteration}] No new formulas discovered!\n"
                f"{'!' * 80}\n"
                f"  Cannot derive: {sorted(still_missing)}\n"
                f"\n"
                f"  Possible reasons:\n"
                f"    1. Missing fundamental inputs (e.g., gross_salary not in observations or parameters)\n"
                f"    2. LLM unable to infer relationships from available data\n"
                f"    3. Metrics require external data not in observations\n"
                f"    4. Circular dependencies detected\n"
                f"\n"
                f"  Available for formulas: {sorted(currently_available)}\n"
                f"  Requested but unreachable: {sorted(still_missing)}\n"
                f"{'!' * 80}\n"
            )
            break
        
        # ========================================
        # Step 8: Accumulate Results
        # ========================================
        
        # Add to master list (preserves discovery order)
        all_calculation_formulas.extend(new_formulas)
        
        # Extract metric names
        newly_discovered_names = {f.metric for f in new_formulas}
        
        # Update tracking set (these become available for next iteration)
        discovered_metric_names.update(newly_discovered_names)
        
        # ========================================
        # Step 9: Success Logging
        # ========================================
        
        logger.info(
            f"\n{'═' * 80}\n"
            f"✅ [PHASE 2 - Iteration {iteration}] Discovered {len(new_formulas)} formula(s)!\n"
            f"{'═' * 80}\n"
            f"  LLM call time: {iteration_llm_time:.2f}s"
        )
        
        for formula in new_formulas:
            logger.info(
                f"  📐 {formula.metric} = {formula.formula}\n"
                f"     Inputs: {formula.inputs}\n"
                f"     Per-scenario: {formula.per_scenario}"
            )
        
        logger.info(
            f"{'═' * 80}\n"
            f"  Progress:\n"
            f"    Total formulas: {len(all_calculation_formulas)}\n"
            f"    Available metrics: {len(currently_available | newly_discovered_names)}\n"
            f"    Still missing: {len(still_missing - newly_discovered_names)}\n"
            f"{'═' * 80}\n"
        )
        
        # ========================================
        # Step 10: Check if Another Iteration Needed
        # ========================================
        
        # Recompute after adding new discoveries
        updated_missing = requested_metrics_set - (currently_available | discovered_metric_names)
        
        if not updated_missing:
            logger.info(
                f"✅ [PHASE 2 - Iteration {iteration}] All metrics now covered! No further iterations needed."
            )
            break
        
        logger.info(
            f"📊 [PHASE 2 - Iteration {iteration}] Still missing: {sorted(updated_missing)}. "
            f"Continuing to next iteration..."
        )

    # ========================================
    # Final Summary
    # ========================================

    calculation_formulas = all_calculation_formulas

    final_extracted = set().union(*entity_available_metrics.values()) if entity_available_metrics else set()
    final_derived = discovered_metric_names
    final_coverage = final_extracted | discovered_metric_names
    final_missing = set(constraints.metrics) - final_coverage

    total_time = time.time() - iteration_start_time

    logger.info(
        f"\n{'═' * 80}\n"
        f"🏁 [PHASE 2 COMPLETE] Formula Discovery Summary\n"
        f"{'═' * 80}\n"
        f"  Total iterations: {iteration + 1}\n"
        f"  Total time: {total_time:.2f}s\n"
        f"  Total formulas: {len(calculation_formulas)}\n"
        f"\n"
        f"  📊 Coverage Report:\n"
        f"    Extracted: {len(final_extracted)} {sorted(final_extracted)}\n"
        f"    Derived: {len(final_derived)} {sorted(final_derived)}\n"
        f"    Total coverage: {len(final_coverage)}/{len(constraints.metrics)} "
        f"({100 * len(final_coverage) / len(constraints.metrics) if constraints.metrics else 0:.0f}%)\n"
        f"    Missing: {len(final_missing)} {sorted(final_missing)}\n"
        f"\n"
        f"  📝 Discovered formulas:\n"
    )

    if calculation_formulas:
        for i, formula in enumerate(calculation_formulas, 1):
            logger.info(f"    {i}. {formula.metric} = {formula.formula}")
    else:
        logger.info("    (none)")

    logger.info(f"{'═' * 80}\n")

    # ========================================
    # PHASE 3: Add Scenario Constants
    # ========================================
    constant_specs = {}
    if constraints.has_scenarios():
        constant_specs = create_scenario_constants(constraints)
        all_metric_specs.update(constant_specs)
        logger.info(
            f"📐 [PHASE 3] Added {len(constant_specs)} scenario constants"
        )

    # ========================================
    # PHASE 4: Expand Calculations (Smart)
    # ========================================
    logger.info("🔧 [PHASE 4] Expanding calculations")

    calculation_specs = expand_calculations_smart(
        calculation_formulas=calculation_formulas,
        constraints=constraints,
        existing_specs=all_metric_specs,
        entity_available_metrics=entity_available_metrics
    )

    logger.info(
        f"✅ [PHASE 4] Created {len(calculation_specs)} calculation specs"
    )

    all_metric_specs.update(calculation_specs)

    # ========================================
    # PHASE 5: Build Response Tables
    # ========================================
    tables = build_response_tables(constraints, all_metric_specs)

    # Create request analysis
    analysis = RequestAnalysis(
        what_user_wants=user_request,
        entities_to_compare=constraints.entities,
        metrics_requested=constraints.metrics,
        comparison_dimensions=[s.name for s in constraints.scenarios] if constraints.has_scenarios() else []
    )

    # Log final stats
    extraction_count = len(all_metric_specs) - len(calculation_specs) - len(constant_specs)
    if not all_metric_specs:
        logger.error(
            f"[HYBRID PLANNER] CRITICAL: all_metric_specs is EMPTY after all phases!"
        )
    else:
        logger.info(
            f"[HYBRID PLANNER] Successfully created {len(all_metric_specs)} metric_specs "
            f"({extraction_count} extracted + {len(constant_specs)} constants + "
            f"{len(calculation_specs)} calculations)"
        )

    # Log statistics
    log_plan_statistics(all_metric_specs, tables, constraints)

    return UnifiedPlan(
        request_analysis=analysis,
        metric_specs=all_metric_specs,
        response_tables=tables,
        narrative_points=generate_narrative_points(constraints, all_metric_specs)
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
# PHASE 1: Per-Entity Value Extraction
# ============================================================================

async def extract_all_values_for_entity(
    entity: str,
    entity_observations: List[Dict],
    requested_metrics: List[str],
    llm: Any
) -> Dict[str, MetricSpec]:
    """
    Extract all possible values for a single entity.

    This function tries to extract ALL requested metrics from the entity's
    observations without needing to know formulas upfront.

    Args:
        entity: Entity name (e.g., "France", "Spain")
        entity_observations: Observations filtered for this entity
        requested_metrics: List of metrics to try extracting
        llm: Language model instance

    Returns:
        Dict mapping spec_id (metric_entity) to MetricSpec objects
    """
    entity_clean = entity.lower().replace(' ', '_').replace('-', '_')
    metric_specs = {}

    # Build prompt for entity-specific extraction
    system_prompt = f"""You are a data extraction expert. Extract metric values for {entity}.

Your task:
- Look for explicit values: "tax rate: 20%", "salary: €50,000"
- Look for implicit values: "after taxes: €45,000" (implies net income)
- Extract numbers even if not directly stated (e.g., calculate from percentages)
- Return confidence score based on how explicit the value is
"""

    obs_sample = []
    for obs in entity_observations[:30]:  # Limit to avoid huge prompts
        content = obs.content if hasattr(obs, 'content') else str(obs)
        if len(content) > 500:
            content = content[:500] + "..."
        obs_sample.append(content)

    human_prompt = f"""Extract values for these metrics from observations about {entity}:

Metrics to extract: {', '.join(requested_metrics)}

Observations:
{chr(10).join(f"- {obs}" for obs in obs_sample)}

For each metric you find, return:
{{
  "metric_name": "...",
  "value": <number>,
  "confidence": <0.0-1.0>,
  "source_text": "excerpt showing where value came from"
}}

Return JSON array of extracted metrics.
If you can't find a metric, don't include it (don't guess).
"""

    try:
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # 🔬 DIAGNOSTIC: Log FULL observation content being used
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] ========== FULL OBSERVATION CONTENT ==========")
        for i, obs in enumerate(entity_observations[:30]):
            obs_content = obs.content if hasattr(obs, 'content') else str(obs)
            logger.error(f"🔬 [DIAGNOSTIC - {entity}] Observation {i+1}:")
            logger.error(f"{obs_content}")
            logger.error(f"--- End Observation {i+1} ---")

        # 🔬 DIAGNOSTIC: Log the FULL prompt being sent
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] ========== FULL EXTRACTION PROMPT ==========")
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] System Prompt:")
        logger.error(f"{system_prompt}")
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Human Prompt:")
        logger.error(f"{human_prompt}")
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Observations count: {len(entity_observations[:30])}")

        # Call LLM
        response = await llm.ainvoke(messages)

        # 🔬 DIAGNOSTIC: Log the FULL raw LLM response
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] ========== FULL LLM RESPONSE ==========")
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Response type: {type(response)}")
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Full content ({len(str(response_content))} chars):")
        logger.error(f"{response_content}")
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] ========== END LLM RESPONSE ==========")

        # Parse response (handle both dict and string responses)
        if isinstance(response, dict):
            extracted_metrics = response.get('metrics', [])
        elif hasattr(response, 'content'):
            import json
            # Try to parse JSON from content
            content = response.content
            # Handle JSON wrapped in markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    extracted_metrics = parsed
                elif isinstance(parsed, dict) and 'metrics' in parsed:
                    extracted_metrics = parsed['metrics']
                else:
                    extracted_metrics = []
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from LLM response for {entity}")
                logger.error(f"🔬 [DIAGNOSTIC - {entity}] JSON parsing error: {str(e)}")
                logger.error(f"🔬 [DIAGNOSTIC - {entity}] Content that failed: {content[:500]}...")
                extracted_metrics = []
        else:
            extracted_metrics = []

        # 🔬 DIAGNOSTIC: Log parsing results
        logger.error(f"🔬 [DIAGNOSTIC - {entity}] Parsing results:")
        logger.error(f"  Extracted metrics count: {len(extracted_metrics)}")
        if extracted_metrics:
            logger.error(f"  Metrics found: {[m.get('metric_name') if isinstance(m, dict) else str(m) for m in extracted_metrics[:5]]}")
        else:
            logger.error(f"  No metrics extracted!")

        # Convert to MetricSpec objects
        for metric_data in extracted_metrics:
            if not isinstance(metric_data, dict):
                continue

            metric_name = metric_data.get('metric_name', '').lower().replace(' ', '_')
            value = metric_data.get('value')
            confidence = metric_data.get('confidence', 0.8)
            source_text = metric_data.get('source_text', '')

            if metric_name and value is not None:
                try:
                    spec_id = f"{metric_name}_{entity_clean}"
                    metric_specs[spec_id] = MetricSpec(
                        data_id=spec_id,
                        source_type=SourceType.EXTRACT,
                        value=float(value),
                        extraction_hint=f"{entity} {metric_name}: {source_text[:100]}",
                        tags={'entity': entity, 'metric': metric_name},
                        confidence=float(confidence)
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping metric {metric_name} for {entity}: {e}")

        logger.info(
            f"[extract_all_values_for_entity] {entity}: extracted {len(metric_specs)} metrics"
        )

    except Exception as e:
        logger.error(f"Error extracting values for {entity}: {e}")

    return metric_specs


# ============================================================================
# Step 2: Value Extraction with Structured Generation (LEGACY - Keep for reference)
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
            # Don't continue - still try Phase 3 derivation

        # PHASE 3: DERIVATION - When extraction fails, derive from available data
        # Check which metrics are still missing after Phase 1 + Phase 2
        extracted_metrics_for_entity = set()
        for spec_id in metric_specs.keys():
            if entity_clean in spec_id:
                # Extract metric name from spec_id (e.g., "net_take_home_spain" -> "net_take_home")
                parts = spec_id.split('_')
                if len(parts) > 1:
                    # Remove entity suffix
                    metric_name = '_'.join([p for p in parts if p != entity_clean])
                    extracted_metrics_for_entity.add(metric_name)

        # ✅ ULTRA-DETAILED LOGGING TO DEBUG WHY PHASE 3 ISN'T RUNNING!
        logger.error(
            f"\n{'='*120}\n"
            f"🔥🔥🔥 [HYBRID PLANNER] PHASE 3 DECISION POINT for {entity} 🔥🔥🔥\n"
            f"{'='*120}\n"
            f"  - patterns.extractable_metrics: {patterns.extractable_metrics}\n"
            f"  - extracted_metrics_for_entity: {extracted_metrics_for_entity}\n"
            f"  - constraints.metrics (ALL requested): {constraints.metrics}\n"
            f"{'='*120}\n"
        )

        still_missing_metrics = [m for m in patterns.extractable_metrics if m not in extracted_metrics_for_entity]

        # ✅ MORE ULTRA-DETAILED LOGGING!
        logger.error(
            f"\n{'='*120}\n"
            f"🔥🔥🔥 [HYBRID PLANNER] STILL_MISSING_METRICS CALCULATION for {entity} 🔥🔥🔥\n"
            f"{'='*120}\n"
            f"  - still_missing_metrics: {still_missing_metrics}\n"
            f"  - len(still_missing_metrics): {len(still_missing_metrics)}\n"
            f"  - entity_observations count: {len(entity_observations) if entity_observations else 0}\n"
            f"  - WILL RUN PHASE 3? {bool(still_missing_metrics and entity_observations)}\n"
            f"{'='*120}\n"
        )

        if still_missing_metrics and entity_observations:
            logger.info(
                f"🔍 [HYBRID PLANNER] PHASE 3 (DERIVATION) STARTING for {entity}:\n"
                f"  - Metrics still missing: {len(still_missing_metrics)} → {still_missing_metrics}\n"
                f"  - Will attempt formula derivation from available data..."
            )

            try:
                derived_specs = await _derive_missing_metrics(
                    entity=entity,
                    entity_clean=entity_clean,
                    missing_metrics=still_missing_metrics,
                    observations=entity_observations,
                    existing_specs=metric_specs,
                    constraints=constraints,
                    llm=llm
                )

                # Add derived specs to main metric_specs
                metric_specs.update(derived_specs)

                logger.info(
                    f"🔍 [HYBRID PLANNER] PHASE 3 COMPLETE for {entity}:\n"
                    f"  - Derived {len(derived_specs)} metrics\n"
                    f"  - Derived metrics: {list(derived_specs.keys())}"
                )

            except Exception as e:
                logger.error(
                    f"🔍 [HYBRID PLANNER] ❌ PHASE 3 FAILED for {entity}:\n"
                    f"  - Exception: {type(e).__name__}: {str(e)}",
                    exc_info=True
                )

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
# PHASE 2: Formula Discovery from Actual Data
# ============================================================================

async def discover_feasible_formulas(
    data_catalog: Dict,
    requested_metrics: List[str],
    constraints: QueryConstraints,
    observations: List[Dict],
    llm: Any
) -> List[CalculationFormula]:
    """
    Discover formulas that can actually work with available data.

    This is the key insight: instead of guessing formulas upfront,
    we discover them AFTER seeing what data we actually have.

    Args:
        data_catalog: Dict with 'available_metrics', 'entity_specific', 'requested_but_missing'
        requested_metrics: What the user wants
        constraints: Query constraints (entities, scenarios, etc.)
        observations: Original observations for context
        llm: Language model instance

    Returns:
        List of CalculationFormula objects that can be executed
    """
    from deep_research_agent.core.structured_models import CalculationFormula

    available = data_catalog.get('available_metrics', set())
    missing = data_catalog.get('requested_but_missing', set())

    if not missing:
        logger.info("[discover_feasible_formulas] All requested metrics extracted, no formulas needed")
        return []

    # Build context for LLM
    system_prompt = """You are a formula synthesis expert. Create calculation formulas based on AVAILABLE data.

CRITICAL RULES:
1. Only use inputs that are in the available_metrics list
2. If a metric cannot be calculated from available data, don't create a formula for it
3. Prefer simple formulas over complex ones
4. For scenario-dependent formulas, use scenario parameters (like num_children, marital_status)
5. Mark formulas as per_scenario=true if they use scenario parameters
"""

    scenario_context = ""
    if constraints.has_scenarios():
        params = {}
        for s in constraints.scenarios:
            params.update(s.parameters)
        scenario_context = f"""
SCENARIO PARAMETERS AVAILABLE: {list(params.keys())}
If a formula uses these parameters, mark per_scenario=true.
"""

    # Sample observations for context
    obs_sample = []
    for obs in observations[:20]:
        content = obs.content if hasattr(obs, 'content') else str(obs)
        if len(content) > 300:
            content = content[:300] + "..."
        obs_sample.append(content)

    human_prompt = f"""Create formulas for these missing metrics using ONLY available data.

AVAILABLE METRICS (extracted): {list(available)}
MISSING METRICS (need formulas): {list(missing)}
{scenario_context}

Context from observations:
{chr(10).join(f"- {obs}" for obs in obs_sample[:10])}

For each missing metric you can calculate, return:
{{
  "metric": "metric_name",
  "formula": "input1 - input2 + input3",
  "inputs": ["input1", "input2", "input3"],
  "description": "What this calculates",
  "per_scenario": false
}}

Return JSON array: {{"formulas": [...]}}

IMPORTANT: Only create formulas if ALL inputs are in AVAILABLE METRICS or SCENARIO PARAMETERS.
"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = await llm.ainvoke(messages)

        # Parse response
        formulas_data = []
        if isinstance(response, dict):
            formulas_data = response.get('formulas', [])
        elif hasattr(response, 'content'):
            import json
            content = response.content
            # Handle JSON wrapped in code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    formulas_data = parsed
                elif isinstance(parsed, dict):
                    formulas_data = parsed.get('formulas', [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse formula JSON from LLM")
                formulas_data = []

        # Validate and create CalculationFormula objects
        formulas = []
        for formula_dict in formulas_data:
            if not isinstance(formula_dict, dict):
                continue

            metric = formula_dict.get('metric', '').lower().replace(' ', '_')
            formula_str = formula_dict.get('formula', '')
            inputs = formula_dict.get('inputs', [])
            description = formula_dict.get('description', '')
            per_scenario = formula_dict.get('per_scenario', False)

            if not metric or not formula_str or not inputs:
                continue

            # Validate all inputs are available (or are scenario parameters)
            scenario_params = set()
            if constraints.has_scenarios():
                for s in constraints.scenarios:
                    scenario_params.update(s.parameters.keys())

            inputs_valid = all(
                inp in available or inp in scenario_params
                for inp in inputs
            )

            if inputs_valid:
                formulas.append(CalculationFormula(
                    metric=metric,
                    formula=formula_str,
                    inputs=inputs,
                    description=description,
                    per_scenario=per_scenario
                ))
                logger.info(
                    f"[discover_feasible_formulas] Created formula: {metric} = {formula_str}"
                )
            else:
                missing_inputs = [inp for inp in inputs if inp not in available and inp not in scenario_params]
                logger.warning(
                    f"[discover_feasible_formulas] Skipping formula for {metric}: "
                    f"missing inputs {missing_inputs}"
                )

        return formulas

    except Exception as e:
        logger.error(f"Error discovering formulas: {e}")
        return []


# ============================================================================
# PHASE 4: Smart Calculation Expansion
# ============================================================================

def expand_calculations_smart(
    calculation_formulas: List,
    constraints: QueryConstraints,
    existing_specs: Dict[str, MetricSpec],
    entity_available_metrics: Dict[str, set]
) -> Dict[str, MetricSpec]:
    """
    Expand calculations intelligently, only for entities that have the needed inputs.

    Key improvement over old expand_calculations:
    - Checks per-entity data availability
    - Skips entities that don't have required inputs
    - Supports scenarios properly

    Args:
        calculation_formulas: List of CalculationFormula objects
        constraints: Query constraints
        existing_specs: Already extracted/constant specs
        entity_available_metrics: Dict mapping entity -> set of available metrics

    Returns:
        Dict of calculation MetricSpec objects
    """
    from deep_research_agent.core.metrics.config import SourceType

    calculation_specs = {}

    for formula_obj in calculation_formulas:
        metric_name = formula_obj.metric
        formula_template = formula_obj.formula
        inputs_template = formula_obj.inputs

        logger.info(
            f"[expand_calculations_smart] Processing formula: {metric_name} = {formula_template}"
        )

        # Determine which entities can use this formula
        applicable_entities = []

        for entity in constraints.entities:
            entity_clean = entity.lower().replace(' ', '_').replace('-', '_')
            available_for_entity = entity_available_metrics.get(entity, set())

            # Check if entity has all required non-scenario inputs
            scenario_params = set()
            if constraints.has_scenarios():
                for s in constraints.scenarios:
                    scenario_params.update(s.parameters.keys())

            missing_inputs = []
            for inp in inputs_template:
                if inp in scenario_params:
                    continue  # Scenario param, always available

                # Check if this input exists for this entity
                input_id = f"{inp}_{entity_clean}"
                if input_id not in existing_specs and inp not in available_for_entity:
                    missing_inputs.append(inp)

            if not missing_inputs:
                applicable_entities.append(entity)
            else:
                logger.debug(
                    f"[expand_calculations_smart] Skipping {metric_name} for {entity}: "
                    f"missing {missing_inputs}"
                )

        if not applicable_entities:
            logger.warning(
                f"[expand_calculations_smart] Formula {metric_name} not applicable to any entity!"
            )
            continue

        logger.info(
            f"[expand_calculations_smart] Formula {metric_name} applicable to: {applicable_entities}"
        )

        # Create calculation specs for applicable entities
        for entity in applicable_entities:
            entity_clean = entity.lower().replace(' ', '_').replace('-', '_')

            if formula_obj.per_scenario and constraints.has_scenarios():
                # Create entity × scenario combinations
                for scenario in constraints.scenarios:
                    spec_id = f"{metric_name}_{entity_clean}_{scenario.id}"

                    # Build actual formula with entity + scenario suffixes
                    actual_formula = formula_template
                    actual_inputs = []

                    for inp in inputs_template:
                        if inp in scenario.parameters:
                            # Scenario parameter
                            input_id = f"{inp}_{scenario.id}"
                        else:
                            # Regular metric
                            input_id = f"{inp}_{entity_clean}"

                        actual_inputs.append(input_id)
                        actual_formula = actual_formula.replace(inp, f"@{input_id}")

                    calculation_specs[spec_id] = MetricSpec(
                        data_id=spec_id,
                        source_type=SourceType.CALCULATE,
                        formula=actual_formula,
                        required_inputs=actual_inputs,
                        tags={
                            'entity': entity,
                            'metric': metric_name,
                            'scenario': scenario.id,
                            'scenario_name': scenario.name
                        },
                        confidence=0.95
                    )

            else:
                # Single calculation for entity (no scenarios)
                spec_id = f"{metric_name}_{entity_clean}"

                actual_formula = formula_template
                actual_inputs = []

                for inp in inputs_template:
                    input_id = f"{inp}_{entity_clean}"
                    actual_inputs.append(input_id)
                    actual_formula = actual_formula.replace(inp, f"@{input_id}")

                calculation_specs[spec_id] = MetricSpec(
                    data_id=spec_id,
                    source_type=SourceType.CALCULATE,
                    formula=actual_formula,
                    required_inputs=actual_inputs,
                    tags={'entity': entity, 'metric': metric_name},
                    confidence=0.95
                )

    logger.info(f"[expand_calculations_smart] Created {len(calculation_specs)} calculation specs")
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


# ============================================================================
# Phase 3: Formula Derivation (NEW - Domain-Agnostic)
# ============================================================================

async def _derive_missing_metrics(
    entity: str,
    entity_clean: str,
    missing_metrics: List[str],
    observations: List[Dict],
    existing_specs: Dict[str, MetricSpec],
    constraints: QueryConstraints,
    llm: Any
) -> Dict[str, MetricSpec]:
    """
    Derive formulas for missing metrics using LLM-based data extraction and formula generation.
    This is DOMAIN-AGNOSTIC - works for tax, climate, medical, business, etc.

    Args:
        entity: Entity name (e.g., "Spain")
        entity_clean: Cleaned entity name for spec IDs (e.g., "spain")
        missing_metrics: List of metric names still missing after extraction
        observations: Filtered observations for this entity
        existing_specs: Already extracted/calculated metric specs
        constraints: Query constraints for context
        llm: LLM instance for generation

    Returns:
        Dict of new MetricSpec instances for derived metrics
    """
    import json

    logger.info(f"📊 [FORMULA DERIVATION] Starting for {entity}: {missing_metrics}")

    # Step 1: Use LLM to extract structured data (NOT hardcoded patterns!)
    structured_data = await _extract_structured_data_generic(
        observations=observations,
        missing_metrics=missing_metrics,
        entity=entity,
        llm=llm
    )

    if not structured_data.get('has_useful_data'):
        logger.info(f"📊 [FORMULA DERIVATION] No structured data found for {entity}, skipping derivation")
        return {}

    logger.info(
        f"📊 [FORMULA DERIVATION] Extracted structured data for {entity}:\n"
        f"  - Data points: {len(structured_data.get('data_points', []))}\n"
        f"  - Context: {structured_data.get('context', 'N/A')}"
    )

    # Step 2: Derive formulas for each missing metric
    derived_specs = {}

    for metric in missing_metrics:
        try:
            formula_info = await _derive_formula_generic(
                target_metric=metric,
                available_data=structured_data,
                domain_context=structured_data.get('context', ''),
                existing_specs=existing_specs,
                entity=entity,
                llm=llm
            )

            if not formula_info.get('code'):
                logger.warning(f"📊 [FORMULA DERIVATION] No code generated for {entity}.{metric}")
                continue

            logger.info(
                f"📊 [FORMULA DERIVATION] Generated formula for {entity}.{metric}:\n"
                f"  - Reasoning: {formula_info.get('reasoning', 'N/A')[:100]}...\n"
                f"  - Code length: {len(formula_info['code'])} chars"
            )

            # Step 3: Execute in sandbox
            from .sandbox import SafePythonExecutor

            sandbox = SafePythonExecutor(timeout_seconds=5.0)

            # Build execution context from structured data
            context = _build_execution_context(structured_data, existing_specs, entity_clean)

            logger.debug(
                f"📊 [FORMULA DERIVATION] Execution context for {entity}.{metric}:\n"
                f"  - Context keys: {list(context.keys())}\n"
                f"  - Context values: {context}"
            )

            # Execute derived formula
            result = sandbox.execute(
                code=formula_info['code'],
                context=context
            )

            if result.success and result.result is not None:
                spec_id = f"{metric}_{entity_clean}"

                derived_specs[spec_id] = MetricSpec(
                    data_id=spec_id,
                    source_type=SourceType.CALCULATE,  # Mark as calculated (derived)
                    value=float(result.result),
                    formula=formula_info.get('reasoning', 'Derived from available data'),
                    confidence=formula_info.get('confidence', 0.75) * 0.9,  # Lower confidence for derived
                    tags={'entity': entity, 'method': 'derived', **constraints.get_entity_tags(entity)},
                    code=formula_info['code']  # Store for audit
                )

                logger.info(
                    f"✅ [FORMULA DERIVATION] DERIVED {entity}.{metric} = {result.result:.2f}\n"
                    f"  - Reasoning: {formula_info.get('reasoning', '')[:150]}...\n"
                    f"  - Confidence: {derived_specs[spec_id].confidence:.2f}"
                )
            else:
                logger.warning(
                    f"❌ [FORMULA DERIVATION] Execution failed for {entity}.{metric}:\n"
                    f"  - Error type: {result.error_type}\n"
                    f"  - Error message: {result.error_message}"
                )

        except Exception as e:
            logger.error(
                f"❌ [FORMULA DERIVATION] Exception deriving {entity}.{metric}: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            continue

    return derived_specs


async def _extract_structured_data_generic(
    observations: List[Dict],
    missing_metrics: List[str],
    entity: str,
    llm: Any
) -> Dict[str, Any]:
    """
    Use LLM to extract structured data relevant to missing metrics.
    Domain-agnostic - works for tax, climate, medical, business, anything.

    CRITICAL: Uses full_content when available for complete data!
    """
    import json

    # Prepare observations (PREFER full_content for complete data)
    obs_text = []
    for obs in observations[:5]:  # Limit to top 5 for prompt size
        # CRITICAL: Use full_content when available for complete observations
        if 'full_content' in obs and obs['full_content']:
            content = obs['full_content']
        elif 'content' in obs:
            content = obs['content']
        else:
            content = str(obs)

        # Truncate for prompt (but we extracted from full content)
        obs_text.append(content[:2000] if len(content) > 2000 else content)

    if not obs_text:
        return {'has_useful_data': False}

    # Ask LLM to identify and extract available data
    prompt = f"""Analyze these observations about {entity} and extract structured data needed to calculate: {', '.join(missing_metrics)}

Observations:
{chr(10).join(f'[{i+1}] {text}' for i, text in enumerate(obs_text))}

TASK: Identify and extract ANY numerical data, rates, brackets, ranges, or formulas that could be used to derive the missing metrics.

Return a JSON object with this exact structure:
{{
    "data_points": [
        {{"type": "range", "name": "description", "values": [...]}},
        {{"type": "rate", "name": "description", "value": ...}},
        {{"type": "constant", "name": "description", "value": ...}},
        {{"type": "formula", "name": "description", "expression": "..."}}
    ],
    "context": "Brief description of the domain/scenario",
    "available_for_calculation": ["list", "of", "available", "data", "types"],
    "has_useful_data": true
}}

Be domain-agnostic. Extract whatever structured data is present:
- Tax: brackets, rates, deductions, thresholds
- Climate: temperatures, emissions, trends, projections
- Medical: dosages, success rates, thresholds, age adjustments
- Business: conversion rates, growth percentages, costs, revenues
- Real estate: prices, square footage, yields, cap rates

Return ONLY the JSON object, no additional text."""

    try:
        messages = [
            SystemMessage(content="You are a data extraction expert. Extract structured data from any domain. Return ONLY valid JSON."),
            HumanMessage(content=prompt)
        ]

        response = await llm.ainvoke(messages)

        # Parse response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # Try to parse as JSON
        data = json.loads(content)

        # Ensure has_useful_data is set
        if 'has_useful_data' not in data:
            data['has_useful_data'] = bool(data.get('data_points'))

        return data

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse structured data response as JSON: {e}")
        return {'has_useful_data': False}
    except Exception as e:
        logger.error(f"Structured data extraction failed: {type(e).__name__}: {str(e)}")
        return {'has_useful_data': False}


async def _derive_formula_generic(
    target_metric: str,
    available_data: Dict[str, Any],
    domain_context: str,
    existing_specs: Dict[str, MetricSpec],
    entity: str,
    llm: Any
) -> Dict[str, Any]:
    """
    Ask LLM to derive formula for ANY domain using its knowledge.
    Returns formula with reasoning and executable Python code.
    """
    import json

    # Format available data for prompt
    data_description = _format_available_data(available_data)

    # Format existing values (already calculated for this entity)
    existing_values = {}
    for spec_id, spec in existing_specs.items():
        if spec.value is not None:
            # Extract metric name from spec_id
            metric_name = spec_id.rsplit('_', 1)[0]  # Remove entity suffix
            existing_values[metric_name] = spec.value

    prompt = f"""You are a calculation expert with knowledge across all domains.

TARGET METRIC: {target_metric}
ENTITY: {entity}

AVAILABLE DATA:
{data_description}

ALREADY CALCULATED VALUES:
{json.dumps(existing_values, indent=2) if existing_values else "None"}

DOMAIN CONTEXT: {domain_context}

TASK: Generate Python code to calculate {target_metric} from the available data.

Consider domain-specific formulas:
- Tax: Progressive calculations, deductions, credits, brackets
- Climate: Trend analysis, projections, conversions, compound rates
- Medical: Dosage calculations, risk scores, age adjustments
- Business: ROI, conversion rates, growth calculations, margins
- Real estate: Cap rates, mortgage calculations, yields, price per sqft

Return a JSON object with this exact structure:
{{
    "reasoning": "Brief explanation of the calculation approach",
    "code": "# Python code to calculate the metric\\nresult = ...",
    "confidence": 0.85,
    "assumptions": ["any assumptions made"]
}}

IMPORTANT:
- The code MUST set a variable named 'result' with the final calculated value
- Use only basic Python operations (no imports)
- Keep the code simple and executable
- Return ONLY the JSON object, no additional text"""

    try:
        messages = [
            SystemMessage(content="You are a formula derivation expert across all domains. Return ONLY valid JSON."),
            HumanMessage(content=prompt)
        ]

        response = await llm.ainvoke(messages)

        # Parse response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # Try to parse as JSON
        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse formula response as JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Formula derivation failed: {type(e).__name__}: {str(e)}")
        return {}


def _format_available_data(data: Dict[str, Any]) -> str:
    """Format structured data for prompt in a readable way."""
    lines = []

    for point in data.get('data_points', []):
        point_type = point.get('type', 'unknown')
        name = point.get('name', 'unnamed')

        if point_type == 'range':
            values = point.get('values', [])
            lines.append(f"- {name} (ranges): {values}")
        elif point_type == 'rate':
            value = point.get('value', 'N/A')
            lines.append(f"- {name}: {value}")
        elif point_type == 'constant':
            value = point.get('value', 'N/A')
            lines.append(f"- {name}: {value}")
        elif point_type == 'formula':
            expression = point.get('expression', 'N/A')
            lines.append(f"- {name}: {expression}")

    return '\n'.join(lines) if lines else "No structured data available"


def _build_execution_context(
    structured_data: Dict[str, Any],
    existing_specs: Dict[str, MetricSpec],
    entity_clean: str
) -> Dict[str, Any]:
    """Build execution context for sandbox from structured data."""
    context = {}

    # Add data points to context
    for point in structured_data.get('data_points', []):
        point_type = point.get('type')
        name = point.get('name', '').lower().replace(' ', '_')

        if point_type == 'range' and name:
            # For ranges, use a descriptive name
            context[f"{name}_ranges"] = point.get('values', [])
            context[name] = point.get('values', [])  # Also add without suffix
        elif point_type in ['rate', 'constant'] and name:
            context[name] = point.get('value')

    # Add existing calculated values for this entity
    for spec_id, spec in existing_specs.items():
        if entity_clean in spec_id and spec.value is not None:
            # Extract metric name from spec_id (e.g., "net_take_home_spain" -> "net_take_home")
            parts = spec_id.split('_')
            if len(parts) > 1:
                metric_name = '_'.join([p for p in parts if p != entity_clean])
                context[metric_name] = spec.value

    return context
