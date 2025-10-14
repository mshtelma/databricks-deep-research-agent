"""Stage 1 components: extract metric specifications and structured data."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .. import get_logger
from ..observation_selector import ObservationSelector
from ..report_generation.models import (
    Calculation,
    CalculationContext,
    ComparisonEntry,
    DataPoint,
    Phase1AUnderstanding,
)
from .models import MetricSpecBundle


logger = get_logger(__name__)


class MetricSpecAnalyzer:
    """Generates metric specifications and calculation context from observations."""

    def __init__(
        self,
        reporter_like,
    ) -> None:
        """Initialize analyzer with dependencies borrowed from the reporter.

        Args:
            reporter_like: Object exposing utilities required for extraction. For now we
                reuse the existing reporter implementation until the extraction logic is fully
                modularised.
        """
        self._reporter = reporter_like

    async def analyze(
        self,
        findings: Dict[str, Any],
        *,
        include_debug_messages: bool = True,
    ) -> tuple[MetricSpecBundle, CalculationContext, List[AIMessage]]:
        """Extract specification bundle and calculation context.

        Returns:
            Tuple of (spec bundle, calculation context, instrumentation messages).
        """
        # Execute the migrated calculation-context workflow.
        calc_context = await _generate_calculation_context(self._reporter, findings)

        spec_bundle = MetricSpecBundle(
            structural_understanding=getattr(calc_context, "structural_understanding", ""),
            table_specifications=list(calc_context.table_specifications or []),
            observation_count=len(findings.get("observations", [])),
            truncated_observations=0,
            applied_plan_sections=None,
        )

        messages: List[AIMessage] = []
        if include_debug_messages:
            messages.append(
                AIMessage(
                    content=(
                        "MetricSpecAnalyzer generated calculation context with "
                        f"{len(calc_context.extracted_data)} data points, "
                        f"{len(calc_context.calculations)} calculations, and "
                        f"{len(calc_context.key_comparisons)} comparisons."
                    ),
                    name="metric_spec_analyzer",
                )
            )

        return spec_bundle, calc_context, messages


# === LEGACY REPORTER EXTRACTION ===

async def _understand_research_context(
    reporter,
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
        # Handle both dict and StructuredObservation objects
        if isinstance(obs, dict):
            full_content = obs.get('full_content', obs.get('content', ''))
            step_id = obs.get('step_id', 'unknown')
        else:
            full_content = getattr(obs, 'full_content', None) or getattr(obs, 'content', '')
            step_id = getattr(obs, 'step_id', 'unknown')
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

7. TABLE SPECIFICATIONS:
   For each table needed, specify EXACTLY what structure it should have.

**CRITICAL: Your entire response must be valid JSON matching this schema:**

```json
{{
  "narrative_understanding": "your detailed analysis addressing points 1-6 above",
  "table_specifications": [
{{
  "table_id": "descriptive_identifier",
  "purpose": "what this table shows",
  "row_entities": ["entity1", "entity2", "entity3"],
  "column_metrics": ["metric1", "metric2", "metric3"]
}}
  ]
}}
```

Example for country tax comparison:
```json
{{
  "narrative_understanding": "The user requests a comprehensive after-tax financial comparison across 7 European countries (Spain, France, UK, Switzerland, Germany, Poland, Bulgaria) for 3 family scenarios. The optimal structure uses countries as rows since we're comparing across entities, with columns representing different scenarios and metrics. Key metrics include net take-home income, effective tax rates, rent costs, daycare expenses, and disposable income.",
  "table_specifications": [
{{
  "table_id": "comprehensive_comparison",
  "purpose": "Compare after-tax finances across all countries and scenarios",
  "row_entities": ["Spain", "France", "UK", "Switzerland", "Germany", "Poland", "Bulgaria"],
  "column_metrics": ["net_income_single", "net_income_married", "net_income_married_child", "effective_tax_rate", "disposable_income"]
}}
  ]
}}
```

Respond with ONLY the JSON object - no other text."""
    logger.info(f"[Stage 1A] Generated prompt. Length: {len(prompt)}")

    # Import model for structured generation
        # Use proper LangChain message format (not dict format)
    messages = [
        SystemMessage(content="You are an expert research analyst specializing in data structuring and comparative analysis."),
        HumanMessage(content=prompt)
    ]

    try:
        # Use structured generation with proper message format
        structured_llm = reporter.llm.with_structured_output(Phase1AUnderstanding, method="json_schema")
        result = await structured_llm.ainvoke(messages)

        logger.info(
            f"[Stage 1A] Context understanding complete: "
            f"{len(result.narrative_understanding)} chars of analysis, "
            f"{len(result.table_specifications)} table specs extracted"
        )

        # Debug logging
        if reporter.debug_logger:
            reporter.debug_logger.log_stage(
                "Phase1A_Understanding",
                result.narrative_understanding,
                {
                    "observation_count": len(observations),
                    "total_chars": total_chars,
                    "understanding_length": len(result.narrative_understanding),
                    "table_specs_count": len(result.table_specifications)
                }
            )

        return {
            'understanding': result.narrative_understanding,
            'table_specifications': result.table_specifications,
            'observation_count': len(observations),
            'total_chars': total_chars
        }

    except Exception as e:
        logger.error(f"[Stage 1A] Failed to generate understanding: {e}")
        return {
            'understanding': f"Error during context understanding: {e}",
            'table_specifications': [],
            'observation_count': len(observations),
            'total_chars': total_chars
        }

async def _extract_with_understanding(
    reporter,
    research_topic: str,
    observations: List[Dict],
    understanding: str,
    table_specifications: List = None,
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
        table_specifications: Table structure specs from Phase 1A (for dynamic guidance)
        section_name: Current section being processed
        max_chars: Maximum characters for observations

    Returns:
        Dict with 'extraction' field containing structured JSON
    """
    logger.info(f"[Stage 1B] Extracting data with guided understanding for: {section_name}")

    # Format observations with FULL content
    obs_details = []
    for obs in observations:
        # Handle both dict and StructuredObservation objects
        if isinstance(obs, dict):
            full_content = obs.get('full_content', obs.get('content', ''))
            step_id = obs.get('step_id', 'unknown')
        else:
            full_content = getattr(obs, 'full_content', None) or getattr(obs, 'content', '')
            step_id = getattr(obs, 'step_id', 'unknown')
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
    # Generate dynamic extraction requirements from table specifications
    dynamic_requirements = reporter._build_dynamic_extraction_requirements(
        table_specifications or [],
        research_topic
    )

    logger.info(
        f"[Stage 1B] Generated dynamic requirements from {len(table_specifications or [])} table specs: "
        f"{len(dynamic_requirements)} chars"
    )

    # Build guidance from table specifications
    entity_guidance = ""
    entities_for_comparisons = []
    if table_specifications:
        # Extract entities from table specs to enforce in comparisons
        all_entities = set()
        for spec in table_specifications:
            if hasattr(spec, 'row_entities') and spec.row_entities:
                all_entities.update(spec.row_entities)

        if all_entities:
            entities_for_comparisons = sorted(list(all_entities))
            entity_list = ', '.join(entities_for_comparisons)
            entity_guidance = f"""

CRITICAL INSTRUCTION FROM TABLE ANALYSIS:
The table specifications identified these entities as rows: {entity_list}

Therefore, your key_comparisons MUST use these exact entity names as primary_key values.
Each entity should have its own comparison entry with all available metrics.

For example:
- ✅ CORRECT: "primary_key": "Spain"
- ✅ CORRECT: "primary_key": "France"
- ❌ WRONG: "primary_key": "Top Marginal Tax Rate"
- ❌ WRONG: "primary_key": "Tax Comparison"

You should create {len(all_entities)} comparison entries, one for each entity listed above.
"""

    example_output = {
        "extracted_data": [
            {"entity": "Country A", "metric": "corporate_tax_rate", "value": 21.0, "unit": "percent", "source_observation_id": "step_1"},
            {"entity": "Country B", "metric": "corporate_tax_rate", "value": 19.0, "unit": "percent", "source_observation_id": "step_2"}
        ],
        "calculations": [
            {"description": "Tax rate difference", "formula": "21.0 - 19.0", "inputs": {"rate_a": 21.0, "rate_b": 19.0}, "result": 2.0, "unit": "percentage_points"}
        ],
        "summary_insights": ["Country A has 2 percentage points higher corporate tax than Country B"],
        "data_quality_notes": []
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

TASK: Extract ONLY data points and calculations. Stage 1D will create comparisons from your extracted data.

Return JSON in this format:

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
  "summary_insights": ["Key insight 1", "Key insight 2"],
  "data_quality_notes": ["Any warnings about missing/conflicting data"]
}}

EXAMPLE:
{example_output}

IMPORTANT GUIDELINES:

1. **Extract ALL relevant data points** from observations:
{dynamic_requirements}
   - Focus on entities from table specifications: {', '.join(entities_for_comparisons) if entities_for_comparisons else "all identified entities"}
   - Include source_observation_id for traceability
   - Set confidence based on source quality
   - Extract data for EVERY entity (don't truncate!)

2. **Show your work** in calculations:
   - Explicit formulas with actual values
   - Named inputs for clarity
   - Include units

3. **Be honest about data quality**:
   - List missing data in data_quality_notes
   - Flag conflicting numbers
   - Note estimates vs precise values

CRITICAL VALUE FORMAT RULES (to ensure valid JSON):

✅ CORRECT value formats:
  - Simple numbers: "value": 21.0, "value": 19
  - Simple strings: "value": "progressive", "value": "married"
  - Percentage numbers: "value": 35.5, "unit": "percent"

❌ INCORRECT value formats (will break JSON parsing):
  - Compound values: "value": "0‑12,450 €:19%; 12,450‑20,200 €:24%" ← NO!
  - Multiple data in one: "value": "Spain 35%, France 40%, UK 45%" ← NO!
  - Complex strings: "value": "Rate varies: 10%-40% depending on income" ← NO!

If you find compound data, create SEPARATE data points:
  Instead of: {{"entity": "All", "metric": "rates", "value": "Spain:19%, France:20%"}}
  Do this: [
{{"entity": "Spain", "metric": "rate", "value": 19.0, "unit": "percent"}},
{{"entity": "France", "metric": "rate", "value": 20.0, "unit": "percent"}}
  ]

Return ONLY the JSON object, no additional text."""

    messages = [
        {"role": "system", "content": "You are a precise data extraction specialist. Extract structured data exactly as specified."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = await reporter.llm.ainvoke(messages)
        extraction_text = reporter._extract_text_from_response(response)

        # Try to extract JSON from code blocks or raw text
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', extraction_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try direct JSON parse
            json_str = extraction_text.strip()

        # Strategy 1: Try standard JSON parsing
        try:
            extraction_data = json.loads(json_str)
            logger.info(
                f"[Stage 1B] Standard JSON parse succeeded: "
                f"{len(extraction_data.get('extracted_data', []))} data points, "
                f"{len(extraction_data.get('calculations', []))} calculations"
            )

            # Debug logging
            if reporter.debug_logger:
                reporter.debug_logger.log_stage(
                    "Phase1B_Extraction_Success",
                    json.dumps(extraction_data, indent=2),
                    {
                        "strategy": "standard_json_parse",
                        "data_points": len(extraction_data.get('extracted_data', [])),
                        "calculations": len(extraction_data.get('calculations', [])),
                        "comparisons": len(extraction_data.get('key_comparisons', []))
                    }
                )

            return {
                'extraction': extraction_data,
                'raw_response': extraction_text
            }
        except json.JSONDecodeError as e:
            # Log detailed error context
            error_context = _get_json_error_context(json_str, e.lineno, e.colno)
            logger.warning(
                f"[Stage 1B] JSON parsing failed: {e.msg}\n"
                f"Error context:\n{error_context}"
            )

            # Strategy 2: Attempt JSON repair
            logger.info("[Stage 1B] Attempting JSON repair...")
            try:
                repaired_json = _repair_json_string(json_str)
                extraction_data = json.loads(repaired_json)
                logger.info(
                    f"[Stage 1B] JSON repair succeeded! Extracted "
                    f"{len(extraction_data.get('extracted_data', []))} data points"
                )
                return {
                    'extraction': extraction_data,
                    'raw_response': extraction_text,
                    'repair_applied': True
                }
            except json.JSONDecodeError as repair_error:
                logger.warning(f"[Stage 1B] JSON repair failed: {repair_error}")

                # Strategy 3: Fallback to regex-based partial extraction
                logger.info("[Stage 1B] Falling back to regex-based extraction...")
                extraction_data = _extract_partial_json_data(json_str)
                return {
                    'extraction': extraction_data,
                    'raw_response': extraction_text,
                    'fallback_used': True
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

def _repair_json_string(json_str: str) -> str:
    """
    Attempt to repair common JSON formatting issues that LLMs produce.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    import unicodedata

    repaired = json_str

    # Fix 1: Replace problematic Unicode characters
    # Non-breaking hyphens, special dashes, etc.
    repaired = repaired.replace('\u202f', ' ')  # Narrow no-break space
    repaired = repaired.replace('\u2011', '-')  # Non-breaking hyphen
    repaired = repaired.replace('\u2013', '-')  # En dash
    repaired = repaired.replace('\u2014', '-')  # Em dash

    # Fix 2: Remove trailing commas before closing brackets
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*\]', ']', repaired)

    # Fix 3: Fix common quote escaping issues in string values
    # Replace unescaped quotes inside string values (heuristic)
    def fix_unescaped_quotes(match):
        string_content = match.group(1)
        # Escape quotes that aren't already escaped
        fixed = re.sub(r'(?<!\\)"', '\\"', string_content)
        return f'"{fixed}"'

    # This is a simple heuristic - may need refinement
    # repaired = re.sub(r'"([^"]*)"', fix_unescaped_quotes, repaired)

    return repaired

def _extract_partial_json_data(broken_json: str) -> Dict[str, Any]:
    """
    Extract partial data from malformed JSON using regex fallback.

    This is a last-resort strategy when JSON parsing completely fails.

    Args:
        broken_json: Malformed JSON that couldn't be parsed

    Returns:
        Dict with partially extracted data
    """
    logger.info("[Stage 1B] Attempting regex fallback extraction from malformed JSON")

    extracted_data = []
    calculations = []
    key_comparisons = []

    # Pattern to match data point-like structures
    # Example: {"entity": "Spain", "metric": "rate", "value": 19, "unit": "percent"}
    data_point_pattern = r'\{[^}]*"entity"\s*:\s*"([^"]+)"[^}]*"metric"\s*:\s*"([^"]+)"[^}]*"value"\s*:\s*([^,}]+)[^}]*\}'

    for match in re.finditer(data_point_pattern, broken_json):
        try:
            entity = match.group(1)
            metric = match.group(2)
            value_str = match.group(3).strip().strip('"')

            # Try to parse value as number
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

            extracted_data.append({
                "entity": entity,
                "metric": metric,
                "value": value,
                "unit": "unknown",
                "confidence": 0.6  # Lower confidence for regex-extracted data
            })
        except Exception as e:
            logger.debug(f"[Stage 1B] Failed to parse regex match: {e}")
            continue

    logger.info(f"[Stage 1B] Regex fallback extracted {len(extracted_data)} data points")

    return {
        "extracted_data": extracted_data,
        "calculations": calculations,
        "key_comparisons": key_comparisons,
        "summary_insights": [],
        "data_quality_notes": [
            "JSON parsing failed - partial data extracted via regex fallback",
            f"Extracted {len(extracted_data)} data points with reduced confidence"
        ]
    }

def _get_json_error_context(
    json_str: str,
    line_num: int,
    col_num: int,
    window: int = 200
) -> str:
    """
    Extract context around JSON parsing error for debugging.

    Args:
        json_str: The JSON string that failed to parse
        line_num: Line number where error occurred
        col_num: Column number where error occurred
        window: Number of characters to show before/after error

    Returns:
        Formatted error context string
    """
    try:
        lines = json_str.split('\n')
        if line_num <= 0 or line_num > len(lines):
            return f"Line {line_num} out of range (total lines: {len(lines)})"

        # Get the error line (convert to 0-indexed)
        error_line = lines[line_num - 1]

        # Calculate character position in full string
        chars_before_line = sum(len(lines[i]) + 1 for i in range(line_num - 1))
        error_pos = chars_before_line + col_num - 1

        # Extract window around error
        start = max(0, error_pos - window)
        end = min(len(json_str), error_pos + window)
        context = json_str[start:end]

        # Mark error position
        error_indicator_pos = error_pos - start
        if 0 <= error_indicator_pos < len(context):
            context_with_marker = (
                context[:error_indicator_pos] +
                " <<<ERROR HERE>>> " +
                context[error_indicator_pos:]
            )
        else:
            context_with_marker = context

        return f"Line {line_num}, Column {col_num}:\n{context_with_marker}"

    except Exception as e:
        return f"Could not extract error context: {e}"

def _validate_extraction_quality(
    reporter,
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

    # Only check comparisons if they're required (min_comparisons > 0)
    # Stage 1B no longer creates comparisons - Stage 1D handles that
    if comparison_count < min_comparisons and min_comparisons > 0:
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
    # If comparisons not required (min_comparisons == 0), give full score
    if min_comparisons > 0:
        comp_score = min(30, (comparison_count / max(min_comparisons, 1)) * 30)
    else:
        comp_score = 30  # Full score when comparisons are created by Stage 1D
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
    # If min_comparisons is 0, don't check comparison count (Stage 1D will create them)
    is_valid = (
        (comparison_count >= min_comparisons or min_comparisons == 0) and
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
    reporter,
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

def _aggregate_into_entity_comparisons(
    reporter,
    data_points: List[Any],
    table_specifications: List[Any]
) -> List[Any]:
    """
    Stage 1D: Aggregate data points into entity-level comparisons.

    This method groups data points by entity (matching table row entities)
    and creates ComparisonEntry objects for each entity.

    Args:
        data_points: List of DataPoint objects with entity/metric/value
        table_specifications: List of TableSpec objects defining expected entities

    Returns:
        List of ComparisonEntry objects with entity-based primary keys
    """
    logger.info(
        f"[Stage 1D] Aggregating {len(data_points)} data points "
        f"for {len(table_specifications)} table specifications"
    )

    # Extract expected entities from table specifications
    expected_entities = set()
    for spec in table_specifications:
        if hasattr(spec, 'row_entities') and spec.row_entities:
            expected_entities.update(spec.row_entities)

    if not expected_entities:
        logger.warning(
            "[Stage 1D] No row entities found in table specifications, "
            "will use all entities from data"
        )

    # Group data points by entity
    entity_metrics = {}
    entity_sources = {}
    entities_found = set()

    for dp in data_points:
        # Handle both object and dict access patterns
        entity = dp.entity if hasattr(dp, 'entity') else dp.get('entity', 'Unknown')
        metric = dp.metric if hasattr(dp, 'metric') else dp.get('metric', 'unknown')
        value = dp.value if hasattr(dp, 'value') else dp.get('value', 'N/A')
        unit = dp.unit if hasattr(dp, 'unit') else dp.get('unit', '')
        source = dp.source_observation_id if hasattr(dp, 'source_observation_id') else dp.get('source_observation_id')

        # Skip if entity doesn't match expected entities (if we have them)
        if expected_entities and entity not in expected_entities:
            # Try fuzzy matching for common variations
            matched = False
            entity_lower = entity.lower()
            for expected in expected_entities:
                expected_lower = expected.lower()
                # Check for substring match or common variations
                if (expected_lower in entity_lower or
                    entity_lower in expected_lower or
                    entity_lower.replace(' ', '') == expected_lower.replace(' ', '')):
                    entity = expected  # Use canonical entity name
                    matched = True
                    break

            if not matched:
                logger.debug(f"[Stage 1D] Skipping entity '{entity}' not in expected list")
                continue

        entities_found.add(entity)

        # Initialize entity if needed
        if entity not in entity_metrics:
            entity_metrics[entity] = {}
            entity_sources[entity] = []

        # Store metric with unit if available
        metric_key = metric
        if unit:
            value_with_unit = f"{value} {unit}".strip()
        else:
            value_with_unit = str(value)

        entity_metrics[entity][metric_key] = value_with_unit

        # Track sources
        if source and source not in entity_sources[entity]:
            entity_sources[entity].append(source)

    # Create ComparisonEntry objects for each entity
    comparisons = []

    # First, create entries for entities with data
    for entity in sorted(entity_metrics.keys()):
        try:
            comp_entry = ComparisonEntry(
                primary_key=entity,
                metrics=entity_metrics[entity],
                source_observation_ids=entity_sources.get(entity, [])
            )
            comparisons.append(comp_entry)
            logger.debug(
                f"[Stage 1D] Created comparison for '{entity}' "
                f"with {len(entity_metrics[entity])} metrics"
            )
        except Exception as e:
            logger.warning(f"[Stage 1D] Failed to create ComparisonEntry for {entity}: {e}")

    # Add placeholder entries for missing expected entities
    if expected_entities:
        missing_entities = expected_entities - entities_found
        for entity in sorted(missing_entities):
            logger.info(f"[Stage 1D] Adding placeholder for missing entity: {entity}")
            comp_entry = ComparisonEntry(
                primary_key=entity,
                metrics={},  # Empty metrics - will show as N/A in tables
                source_observation_ids=[]
            )
            comparisons.append(comp_entry)

    logger.info(
        f"[Stage 1D Complete] Created {len(comparisons)} entity-level comparisons "
        f"({len(entities_found)} with data, {len(expected_entities - entities_found)} placeholders)"
    )

    return comparisons

async def _generate_calculation_context(
    reporter,
    findings: Dict[str, Any]
):
    """
    Phase 1: Three-stage calculation context generation.

    Mirrors the legacy reporter implementation while delegating to
    shared helpers within this module.
    """
    logger.info("[HYBRID Phase 1] Starting THREE-STAGE calculation context generation")

    settings = reporter.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {})
    all_observations = findings.get('observations', [])

    max_calc_chars = settings.get('max_calc_prompt_chars', 60000)
    logger.info(f"[HYBRID Phase 1] Using max_calc_prompt_chars: {max_calc_chars:,}")

    reporter._monitor_memory_usage('calc_context', len(all_observations))

    selector = reporter.observation_selector or ObservationSelector(reporter.embedding_manager)
    top_k = settings.get('calc_selector_top_k', 60)
    tail_k = settings.get('calc_recent_tail', 20)

    obs_dicts = []
    for obs in all_observations:
        if hasattr(obs, 'to_dict'):
            obs_dicts.append(obs.to_dict())
        elif isinstance(obs, dict):
            obs_dicts.append(obs)
        else:
            obs_dicts.append({'content': str(obs), 'step_id': 'unknown'})

    research_topic = findings.get('research_topic', 'Research Question')

    diversity_entities = selector.extract_key_entities_from_topic(research_topic)
    enable_diversity = len(diversity_entities) >= 2

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
        use_semantic=getattr(selector, 'embedding_manager', None) is not None,
        ensure_entity_diversity=enable_diversity,
        diversity_entities=diversity_entities if enable_diversity else None,
    )

    recent_tail = obs_dicts[-tail_k:] if len(obs_dicts) >= tail_k else obs_dicts
    merged = reporter._dedupe_preserve_order(scored + recent_tail)

    logger.info(
        f"[HYBRID Phase 1] Selected {len(merged)} observations "
        f"(top-k: {len(scored)}, tail: {len(recent_tail)})"
    )

    plan = findings.get('current_plan')

    try:
        understanding_result = await _understand_research_context(
            reporter,
            research_topic=research_topic,
            observations=merged,
            plan=plan,
            section_name="Calculation Context",
            max_chars=max_calc_chars,
        )

        understanding = understanding_result['understanding']
        table_specs = understanding_result.get('table_specifications', [])

        logger.info(
            f"[Stage 1A Complete] Generated {len(understanding)} chars of context understanding, "
            f"{len(table_specs)} table specifications"
        )

        extraction_result = await _extract_with_understanding(
            reporter,
            research_topic=research_topic,
            observations=merged,
            understanding=understanding,
            table_specifications=table_specs,
            section_name="Calculation Context",
            max_chars=max_calc_chars,
        )

        extraction_data = extraction_result['extraction']

        logger.info(
            f"[Stage 1B Complete] Extracted "
            f"{len(extraction_data.get('extracted_data', []))} data points, "
            f"{len(extraction_data.get('calculations', []))} calculations"
        )

        validation_result = _validate_extraction_quality(
            reporter=reporter,
            extraction=extraction_data,
            min_comparisons=0,
            min_data_points=2,
        )

        logger.info(
            f"[Stage 1C Complete] Quality Score: {validation_result['quality_score']:.2f}, "
            f"Valid: {validation_result['valid']}, "
            f"Issues: {len(validation_result['issues'])}"
        )

        if validation_result['issues']:
            logger.warning(f"[Stage 1C] Quality Issues: {validation_result['issues']}")
        if validation_result['recommendations']:
            logger.info(f"[Stage 1C] Recommendations: {validation_result['recommendations']}")

        data_points = []
        for dp in extraction_data.get('extracted_data', []):
            try:
                data_points.append(DataPoint(**dp))
            except Exception as exc:
                logger.warning(f"[Phase 1] Failed to create DataPoint from {dp}: {exc}")

        calculations = []
        for calc in extraction_data.get('calculations', []):
            try:
                calculations.append(Calculation(**calc))
            except Exception as exc:
                logger.warning(f"[Phase 1] Failed to create Calculation from {calc}: {exc}")

        comparisons = []
        for comp in extraction_data.get('key_comparisons', []):
            try:
                comparisons.append(ComparisonEntry(**comp))
            except Exception as exc:
                logger.warning(f"[Phase 1] Failed to create ComparisonEntry from {comp}: {exc}")

        calc_context = CalculationContext(
            extracted_data=data_points,
            calculations=calculations,
            key_comparisons=comparisons,
            summary_insights=extraction_data.get('summary_insights', []),
            data_quality_notes=extraction_data.get('data_quality_notes', []),
            metadata={
                'stage1_metrics': {
                    'data_points': len(data_points),
                    'calculations': len(calculations),
                    'comparisons': len(comparisons),
                    'quality_score': validation_result['quality_score'],
                    'quality_valid': validation_result['valid'],
                    'quality_summary': validation_result['summary'],
                    'table_specs_count': len(table_specs),
                }
            },
            table_specifications=table_specs,
            structural_understanding=understanding,
        )

        logger.info(
            f"[HYBRID Phase 1] THREE-STAGE PIPELINE COMPLETE: "
            f"{len(calc_context.extracted_data)} data points, "
            f"{len(calc_context.calculations)} calculations, "
            f"{len(calc_context.key_comparisons)} comparisons "
            f"(Quality: {validation_result['quality_score']:.2f})"
        )

        if calc_context.table_specifications and calc_context.extracted_data:
            logger.info(
                f"[Stage 1D] Creating entity-level comparisons from "
                f"{len(calc_context.extracted_data)} data points"
            )

            entity_comparisons = _aggregate_into_entity_comparisons(
                reporter,
                calc_context.extracted_data,
                calc_context.table_specifications,
            )

            if entity_comparisons:
                calc_context.key_comparisons = entity_comparisons
                logger.info(
                    f"[Stage 1D Complete] Created {len(entity_comparisons)} entity-level comparisons"
                )
            else:
                logger.warning("[Stage 1D] No entity comparisons created")
        else:
            logger.warning(
                f"[Stage 1D] Skipped: table_specifications={bool(calc_context.table_specifications)}, "
                f"data_points={len(calc_context.extracted_data)}"
            )

        if not validation_result['valid'] and not calc_context.key_comparisons and data_points:
            logger.warning(
                "[Phase 1] Quality check failed and no comparisons exist, attempting fallback to build comparisons"
            )
            fallback_comparisons = _build_comparisons_from_datapoints(
                reporter,
                data_points,
            )
            if fallback_comparisons:
                calc_context.key_comparisons = fallback_comparisons
                logger.info(
                    f"[Phase 1] Fallback generated {len(fallback_comparisons)} comparisons from data points"
                )
        elif calc_context.key_comparisons:
            logger.info(
                f"[Phase 1] Stage 1D created {len(calc_context.key_comparisons)} comparisons, skipping fallback"
            )

        return calc_context

    except Exception as exc:
        logger.error(f"[HYBRID Phase 1] THREE-STAGE PIPELINE FAILED: {exc}", exc_info=True)

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

async def generate_calculation_context(
    reporter,
    findings: Dict[str, Any],
):
    """Public wrapper used by reporter fallbacks."""

    return await _generate_calculation_context(reporter, findings)


__all__ = ["MetricSpecAnalyzer", "generate_calculation_context"]
