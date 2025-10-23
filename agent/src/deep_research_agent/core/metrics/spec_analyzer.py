"""Stage 1 components: extract metric specifications and structured data."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

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
    Phase 1: Calculation context generation.

    Supports two modes:
    1. Unified Planning (NEW): Single-pass user-request-driven planning with explicit data links
    2. Three-Stage Pipeline (LEGACY): Multi-stage extraction with metric matching

    Mode is controlled by metrics.use_unified_planning config flag.
    """
    # Check if unified planning is enabled
    use_unified = reporter.config.get('metrics', {}).get('use_unified_planning', False)

    if use_unified:
        logger.info("[UNIFIED PLANNING] Using user-request-driven planning mode")

        # Extract research topic
        research_topic = findings.get('research_topic', 'Research Question')
        all_observations = findings.get('observations', [])

        # Convert observations to dicts if needed
        obs_dicts = []
        for obs in all_observations:
            if hasattr(obs, 'to_dict'):
                obs_dicts.append(obs.to_dict())
            elif isinstance(obs, dict):
                obs_dicts.append(obs)
            else:
                obs_dicts.append({'content': str(obs), 'step_id': 'unknown'})

        # Select observations (reuse existing selection logic)
        settings = reporter.config.get('agents', {}).get('reporter', {}).get('hybrid_settings', {})
        selector = reporter.observation_selector or ObservationSelector(reporter.embedding_manager)
        top_k = settings.get('calc_selector_top_k', 60)
        tail_k = settings.get('calc_recent_tail', 20)

        diversity_entities = selector.extract_key_entities_from_topic(research_topic)
        enable_diversity = len(diversity_entities) >= 2

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
            f"[UNIFIED PLANNING] Selected {len(merged)} observations for planning "
            f"(top-k: {len(scored)}, tail: {len(recent_tail)})"
        )

        # Get query constraints if available (Phase 1 integration)
        constraints = findings.get('query_constraints')
        if constraints:
            logger.info(
                f"[UNIFIED PLANNING] Using QueryConstraints: "
                f"{len(constraints.entities)} entities, "
                f"{len(constraints.metrics)} metrics"
            )
        else:
            logger.info("[UNIFIED PLANNING] No QueryConstraints found, using LLM-based planning")

        # Try unified planning with fallback to legacy on failure
        try:
            # Create unified plan (hybrid if constraints available, else LLM-based)
            unified_plan = await _create_unified_plan(reporter, research_topic, merged, constraints)

            # Execute plan to generate calculation context
            calc_context = await _execute_unified_plan(reporter, unified_plan, merged)

            logger.info(
                f"[UNIFIED PLANNING] Complete: "
                f"{len(calc_context.calculations)} calculations, "
                f"{len(calc_context.key_comparisons)} comparisons"
            )

            # FIX #3: Validate that unified planning produced usable data
            # Check if we have comparisons with actual non-None values
            valid_comparisons = sum(
                1 for comp in calc_context.key_comparisons
                if comp.metrics and any(v is not None for v in comp.metrics.values())
            )

            if valid_comparisons == 0:
                logger.warning(
                    "[UNIFIED PLANNING] Produced 0 comparisons with valid data "
                    "(all values are None). Falling back to three-stage pipeline for better coverage."
                )
                # Fall through to legacy path below
                use_unified = False
            else:
                logger.info(
                    f"[UNIFIED PLANNING] Validation passed: "
                    f"{valid_comparisons}/{len(calc_context.key_comparisons)} comparisons have valid data"
                )
                return calc_context

        except Exception as e:
            logger.error(
                f"[UNIFIED PLANNING] Failed with exception, falling back to three-stage pipeline: {e}",
                exc_info=True
            )
            # Fall through to legacy path below
            use_unified = False

    # LEGACY PATH: Three-stage pipeline
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
                },
                'source_observations': merged  # Store for Planner's formula extraction
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


# === UNIFIED PLANNING (NEW ARCHITECTURE) ===

async def _create_unified_plan(
    reporter,
    user_request: str,
    observations: List[Dict],
    constraints=None
):
    """Create unified plan that directly answers user's request.

    Supports two approaches:
    1. Hybrid approach (NEW): Uses structured generation with QueryConstraints
    2. LLM-based approach (LEGACY): Single LLM call for planning

    Args:
        reporter: Reporter instance with LLM access
        user_request: The original user request/research topic
        observations: Selected research observations
        constraints: Optional QueryConstraints for hybrid approach

    Returns:
        UnifiedPlan with explicit data links
    """
    from .unified_models import UnifiedPlan, UserRequestAnalysis, DataSource, ResponseTable, TableCell
    from .hybrid_planner import create_unified_plan_hybrid

    # CRITICAL FIX: Check config to determine planner mode (hybrid is DEFAULT)
    use_legacy = reporter.config.get('metrics', {}).get('unified_planning', {}).get('use_legacy_llm_planner', False)
    force_hybrid = reporter.config.get('metrics', {}).get('unified_planning', {}).get('force_hybrid_planner', True)

    # Determine which planner to use
    use_hybrid_planner = (not use_legacy) or force_hybrid

    # Try hybrid approach if enabled and constraints are available
    if use_hybrid_planner and constraints:
        try:
            logger.info("[HYBRID PLANNER] Using structured generation with QueryConstraints (config-enabled)")
            unified_plan = await create_unified_plan_hybrid(
                user_request=user_request,
                observations=observations,
                constraints=constraints,
                llm=reporter.llm
            )
            logger.info(
                f"[HYBRID PLANNER] Successfully created plan: "
                f"{len(unified_plan.metric_specs)} specs, "
                f"{len(unified_plan.response_tables)} tables"
            )
            return unified_plan
        except Exception as e:
            logger.warning(
                f"[HYBRID PLANNER] Failed to create plan: {e}. "
                "Falling back to LLM-based approach"
            )
            # Fall through to legacy LLM-based approach
    elif use_hybrid_planner and not constraints:
        logger.warning(
            "[HYBRID PLANNER] Config requests hybrid mode but query_constraints not available! "
            "Falling back to legacy LLM planner. This indicates query_constraints weren't passed in findings."
        )

    # Legacy LLM-based approach
    logger.info(f"[LLM PLANNER] Using legacy single-call planning (use_legacy={use_legacy}, force_hybrid={force_hybrid}, has_constraints={constraints is not None})")

    logger.info(
        f"[UNIFIED PLAN] Creating user-request-driven plan for: '{user_request}' "
        f"with {len(observations)} observations"
    )

    # Format observations for prompt (limit to avoid token overflow)
    # CRITICAL FIX: Provide real observation IDs prominently
    obs_summaries = []
    obs_full_text = []

    for i, obs in enumerate(observations):
        step_id = obs.get('step_id', f'obs_{i}')
        content = obs.get('content', '')

        # Extract key entities and metrics from content for summary
        content_preview = content[:300]  # First 300 chars for entity/metric extraction

        # Build summary with actual ID
        obs_summaries.append(f"  - ID: '{step_id}' | Content preview: {content_preview}...")

        # Full observation for context
        obs_full_text.append(f"[Observation ID: '{step_id}']\n{content[:2000]}")

    # Create ID list section
    available_ids = "\n".join(obs_summaries[:20])  # Limit to first 20 for clarity
    obs_text = "\n\n".join(obs_full_text)

    prompt = f"""You are creating a complete plan to answer a user's research request.

USER REQUEST:
{user_request}

⚠️ CRITICAL - AVAILABLE OBSERVATION IDs ⚠️
You MUST use ONLY these exact observation IDs when creating data sources:
{available_ids}

DO NOT invent IDs like "obs_41" or "assumption" - use ONLY the IDs listed above!

RESEARCH OBSERVATIONS (first 2000 chars each):
{obs_text}

YOUR TASK:
Create a unified plan that DIRECTLY ANSWERS the user's request with explicit data links.

Step 1: Analyze the request
- What is the user asking for? (comparison? analysis? specific values?)
- What entities should be compared? (countries, products, scenarios)
- What metrics would answer their question?
- What dimensions are relevant? (scenarios, time periods)

Step 2: Design response structure
- What table(s) would best present the answer?
- What should be in rows vs columns?
- What narrative points would highlight key findings?

Step 3: Map each table cell to data
- For each cell: specify exact data_id, row, column
- For each data_id: specify how to get it:
  - "extract": from which observation and where to find the value
  - "calculate": formula and required inputs (other data_ids)

CRITICAL CONSTRAINTS:
- MAXIMUM 3 tables (focus on most important comparisons)
- MAXIMUM 7 rows per table (one per entity/country)
- MAXIMUM 6 columns per table (key metrics only)
- MAXIMUM 30 data sources total (extractions + calculations)
- Create EXPLICIT LINKS: User question → Metrics → Data sources → Table cells
- NO matching needed - everything is explicitly connected!
- Focus on data that DIRECTLY answers user's request - quality over quantity
- If data is missing from observations, note it but don't fabricate
- PREFER direct extractions over calculations whenever possible
- Each extraction_path MUST be specific enough to locate the exact value

**CRITICAL: Use these EXACT field names from the DataSource Pydantic schema!**

For EXTRACTION (getting value from observation):
{{
  "data_sources": {{
    "unique_data_id": {{
      "data_id": "unique_data_id",              # REQUIRED: Must match the key
      "source_type": "extract",                 # REQUIRED: Literal "extract"
      "observation_id": "gap_001",              # REQUIRED: Use EXACT ID from list above!
      "extraction_path": "Spain tax rate: 35%" # REQUIRED: Specific hint with entity + metric
    }}
  }}
}}

⚠️ CRITICAL: The observation_id MUST be one of the IDs listed in "AVAILABLE OBSERVATION IDs" above!
Do NOT make up IDs like "obs_41" or "assumption" - these will fail!

EXTRACTION_PATH GUIDELINES (FIX #4):
✅ GOOD extraction_path examples (specific context):
  - "Spain net take-home income: €172,000"
  - "France effective tax rate 31%"
  - "London average rent for 2-bedroom: £1,800/month"
  - "Germany child benefit: €2,628 annually"

❌ BAD extraction_path examples (too vague):
  - "tax rate" (which country?)
  - "income" (which metric? gross or net?)
  - "rent" (which city? which property type?)
  
GOLDEN RULE: extraction_path should mention BOTH the entity (country/city) AND the specific metric name.
This helps the extractor find the right value even in long observations.

For CALCULATION (computing from other data):
{{
  "data_sources": {{
    "calc_id": {{
      "data_id": "calc_id",                     # REQUIRED: Must match the key
      "source_type": "calculate",               # REQUIRED: Literal "calculate"
      "formula": "data_a - data_b",             # REQUIRED: Math expression
      "required_inputs": ["data_a", "data_b"]   # REQUIRED: List of input data_ids
    }}
  }}
}}

❌ WRONG field names (validation will fail):
- "type" instead of "source_type"
- "desc" instead of "data_id"
- Any other creative field names

✅ CORRECT field names (from Pydantic schema):
- "data_id" (string, unique identifier)
- "source_type" (literal "extract" or "calculate")
- "observation_id" (for extractions only)
- "extraction_path" (for extractions only)
- "formula" (for calculations only)
- "required_inputs" (for calculations only, list of strings)

Return structured data matching the UnifiedPlan schema with these fields:
- request_analysis: UserRequestAnalysis with what_user_wants, entities_to_compare, metrics_requested, comparison_dimensions
- data_sources: Dict of data_id to DataSource (source_type="extract" or "calculate")
- response_tables: List of ResponseTable with table_id, title, purpose, rows, columns, cells
- narrative_points: List of key insights

REMEMBER: Only include data that helps answer the user's SPECIFIC request.
Don't extract random metrics just because they're available."""

    messages = [
        SystemMessage(content="You are an expert at understanding user requests and creating structured data plans with explicit links. You MUST use exact field names from the Pydantic schema."),
        HumanMessage(content=prompt)
    ]

    try:
        # Get temperature from config (default to 0.2 for deterministic generation)
        unified_config = reporter.config.get('metrics', {}).get('unified_planning', {})
        planning_temp = unified_config.get('temperature', 0.2)

        logger.info(
            f"[UNIFIED PLAN] Using reporter's LLM "
            f"with temperature override {planning_temp} for structured generation"
        )

        # Use reporter's LLM (already configured with appropriate model tier)
        # Apply temperature override for more deterministic schema following
        planning_llm = reporter.llm
        if hasattr(planning_llm, 'temperature'):
            original_temp = planning_llm.temperature
            planning_llm.temperature = planning_temp
            logger.debug(
                f"[UNIFIED PLAN] Overrode temperature from {original_temp} to {planning_temp}"
            )

        # Use structured generation - NO JSON PARSING!
        structured_llm = planning_llm.with_structured_output(
            UnifiedPlan,
            method="json_schema"
        )

        # Direct Pydantic model output - guaranteed valid!
        unified_plan = await structured_llm.ainvoke(messages)

        # Detailed logging
        extraction_count = sum(1 for s in unified_plan.data_sources.values() if s.source_type == "extract")
        calculation_count = sum(1 for s in unified_plan.data_sources.values() if s.source_type == "calculate")

        logger.info(
            f"[UNIFIED PLAN] Successfully created plan: "
            f"{len(unified_plan.data_sources)} total data sources "
            f"({extraction_count} extractions, {calculation_count} calculations), "
            f"{len(unified_plan.response_tables)} tables, "
            f"{len(unified_plan.request_analysis.entities_to_compare)} entities to compare"
        )

        # CRITICAL: Validate plan has data sources
        if len(unified_plan.data_sources) == 0:
            logger.error(
                "[UNIFIED PLAN] ⚠️ CRITICAL: Plan has ZERO data sources! "
                "This will result in all N/A tables. Check: "
                "1) Pydantic field mapping (data_sources vs metric_specs), "
                "2) LLM response format, "
                "3) JSON parsing"
            )

        # Log request analysis details
        logger.debug(
            f"[UNIFIED PLAN] Request analysis: "
            f"User wants: '{unified_plan.request_analysis.what_user_wants}', "
            f"Entities: {unified_plan.request_analysis.entities_to_compare}, "
            f"Metrics: {unified_plan.request_analysis.metrics_requested}"
        )

        # Log table structure
        for table in unified_plan.response_tables:
            logger.debug(
                f"[UNIFIED PLAN] Table '{table.title}': "
                f"{len(table.rows)} rows × {len(table.columns)} columns = {len(table.cells)} cells"
            )

        return unified_plan

    except Exception as e:
        logger.error(
            f"[UNIFIED PLAN] Failed to create plan: {e}\n"
            f"Prompt length: {len(prompt)}, Observations: {len(observations)}",
            exc_info=True
        )
        # Re-raise to trigger fallback to legacy pipeline
        raise


def _extract_entities_from_path(extraction_path: str) -> List[str]:
    """Extract entity hints from extraction path for observation matching.

    Args:
        extraction_path: Hint about what to extract (e.g., "UK basic income tax rate 20%")

    Returns:
        List of entity variants to search for
    """
    # Country mappings with common variants
    country_map = {
        "united kingdom": ["united kingdom", "uk", "britain", "british"],
        "switzerland": ["switzerland", "swiss", "zug", "zurich"],
        "germany": ["germany", "german", "deutschland"],
        "france": ["france", "french", "français"],
        "spain": ["spain", "spanish", "españa"],
        "poland": ["poland", "polish", "polska"],
        "bulgaria": ["bulgaria", "bulgarian"]
    }

    entities = []
    path_lower = extraction_path.lower()

    for country, variants in country_map.items():
        if any(v in path_lower for v in variants):
            entities.extend(variants)
            break

    return entities


def _find_best_observation(
    source,
    obs_index: Dict,
    observations: List[Dict]
) -> tuple[Optional[Dict], str]:
    """Multi-strategy observation finder with entity-aware fallback.

    FIX: Layer 0 - When UnifiedPlan assigns wrong observation_id, this function
    uses entity extraction to find the correct observation.

    Args:
        source: MetricSpec with observation_id and extraction_path
        obs_index: Dict mapping step_id to observation
        observations: List of all observations

    Returns:
        Tuple of (observation, match_type) or (None, "not_found")
    """
    # Extract entity hints for validation (needed by all strategies)
    entity_hints = _extract_entities_from_path(source.extraction_path)

    # Strategy 1: Direct ID lookup WITH entity validation
    obs = obs_index.get(source.observation_id)
    if obs and entity_hints:
        # Validate that the observation actually contains the entity
        content = str(obs.get('content', '')).lower()
        entity_found = any(hint.lower() in content for hint in entity_hints)
        if entity_found:
            # Direct ID is correct - observation has the right entity
            return obs, "direct_id"
        else:
            # Direct ID found an observation, but wrong entity! Fall through to entity search
            logger.warning(
                f"[OBSERVATION FIX] direct_id={source.observation_id} does NOT contain "
                f"entity hints {entity_hints} for {source.data_id}. Searching for correct observation..."
            )
    elif obs:
        # No entity hints to validate - trust direct_id
        return obs, "direct_id"

    # Strategy 2: Entity-aware search (NEW - FIX for wrong observation_id)
    if entity_hints:
        for hint in entity_hints:
            for step_id, candidate in obs_index.items():
                content = str(candidate.get('content', '')).lower()
                if hint.lower() in content:
                    logger.info(
                        f"[OBSERVATION FIX] Found {hint} data in {step_id} "
                        f"(was looking in {source.observation_id})"
                    )
                    return candidate, "entity_match"

    # Strategy 3: Keyword search from extraction_path
    # Extract key terms from extraction path
    keywords = [word.lower() for word in source.extraction_path.split()
                if len(word) > 4 and word.lower() not in {'rate', 'income', 'euro', 'gross'}]

    if keywords:
        for obs in observations:
            content = str(obs.get('content', '')).lower()
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in content)
            if matches >= 2:  # At least 2 keyword matches
                logger.info(
                    f"[OBSERVATION FIX] Found observation with {matches} keyword matches "
                    f"for {source.data_id}"
                )
                return obs, "keyword_match"

    return None, "not_found"


def _parse_data_id(data_id: str) -> tuple[str, str]:
    """Parse entity and metric from data_id.

    Args:
        data_id: Identifier like 'spain_single_net_takehome' or 'uk_basic_income_tax_rate'

    Returns:
        (entity, metric) tuple, e.g., ('Spain', 'single_net_takehome')
    """
    # Normalize to lowercase and replace hyphens
    normalized = data_id.lower().replace('-', '_')
    parts = normalized.split('_')

    if not parts:
        return "Unknown", data_id

    # First part is typically the entity (country/region)
    entity = parts[0].capitalize()

    # Rest is the metric
    metric = '_'.join(parts[1:]) if len(parts) > 1 else data_id

    return entity, metric


async def _execute_unified_plan(
    reporter,
    plan,
    observations: List[Dict]
) -> CalculationContext:
    """Execute unified plan to generate response.

    Args:
        reporter: Reporter instance
        plan: UnifiedPlan with explicit data links
        observations: Research observations

    Returns:
        CalculationContext with calculations and comparisons
    """
    logger.info(
        f"[UNIFIED EXECUTION] Starting execution of plan with "
        f"{len(plan.data_sources)} data sources, {len(plan.response_tables)} tables"
    )

    # Build observation index for fast lookup
    obs_index = {}
    for i, obs in enumerate(observations):
        step_id = obs.get('step_id', f'obs_{i}')
        obs_index[step_id] = obs

    logger.debug(
        f"[UNIFIED EXECUTION] Built observation index with {len(obs_index)} entries"
    )

    # Step 1: Extract/calculate all data sources
    data_values = {}
    calculations = []
    extraction_count = 0
    calculation_count = 0
    failed_count = 0

    # FIX Layer 4: Track extraction analytics for monitoring
    extraction_stats = {
        "total_extractions": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "match_types": {
            "direct_id": 0,
            "entity_match": 0,
            "keyword_match": 0,
            "not_found": 0
        }
    }

    for data_id, source in plan.data_sources.items():
        logger.debug(
            f"[UNIFIED EXECUTION] Processing data source '{data_id}' "
            f"(type: {source.source_type})"
        )

        if source.source_type == "extract":
            # CRITICAL FIX: Use pre-extracted value from MetricSpec if it exists!
            # The hybrid planner already extracted values from observation.metric_values
            # Re-extraction is redundant and often fails due to truncated content
            if source.value is not None:
                # Trust the pre-extracted value from planner/researcher
                data_values[data_id] = source.value
                extraction_count += 1
                extraction_stats["successful_extractions"] += 1
                logger.info(
                    f"[UNIFIED EXECUTION] ✓ Using PRE-EXTRACTED value '{data_id}' = {source.value} "
                    f"(source: {source.observation_id}, path: '{source.extraction_path}')"
                )
            else:
                # Only attempt extraction if value is not pre-set (fallback for edge cases)
                logger.info(
                    f"[UNIFIED EXECUTION] ⚠️ No pre-extracted value for '{data_id}', "
                    f"attempting extraction from observation..."
                )

                # Extract from observation
                # FIX: Use multi-strategy observation finder (Layer 0)
                extraction_stats["total_extractions"] += 1
                obs, match_type = _find_best_observation(source, obs_index, observations)

                # FIX Layer 4: Track match type statistics
                extraction_stats["match_types"][match_type] += 1

                if obs:
                    logger.info(
                        f"[UNIFIED EXECUTION] ✓ Found observation for '{data_id}' "
                        f"via {match_type} (step_id: {obs.get('step_id', 'unknown')})"
                    )
                else:
                    logger.error(
                        f"[UNIFIED EXECUTION] ✗ No observation found for '{data_id}' "
                        f"after trying all strategies. observation_id={source.observation_id}, "
                        f"extraction_path='{source.extraction_path}'"
                    )

                if obs:
                    # FIX #2: Try LLM extraction first, fall back to regex if it fails
                    try:
                        value = await _extract_value_from_observation_llm(
                            reporter, obs, source.extraction_path, data_id
                        )
                    except Exception as e:
                        logger.warning(
                            f"[UNIFIED EXECUTION] LLM extraction failed for '{data_id}': {e}. "
                            "Falling back to regex extraction."
                        )
                        value = _extract_value_from_observation(obs, source.extraction_path)

                    data_values[data_id] = value

                    # FIX #5: Enhanced logging to track extraction success/failure
                    if value is not None:
                        extraction_count += 1
                        extraction_stats["successful_extractions"] += 1  # FIX Layer 4
                        logger.info(
                            f"[UNIFIED EXECUTION] ✓ Extracted '{data_id}' = {value} "
                            f"from observation {obs.get('step_id', 'unknown')} "
                            f"using path: '{source.extraction_path}'"
                        )
                    else:
                        failed_count += 1
                        extraction_stats["failed_extractions"] += 1  # FIX Layer 4
                        logger.warning(
                            f"[UNIFIED EXECUTION] ✗ Extraction FAILED for '{data_id}': "
                            f"no value found in observation {obs.get('step_id', 'unknown')} "
                            f"with extraction_path: '{source.extraction_path}'. "
                            f"This will result in N/A in the table."
                        )
                else:
                    logger.error(
                        f"[UNIFIED EXECUTION] ✗ Observation not found for data source '{data_id}' "
                        f"(tried ID: {source.observation_id}, semantic search also failed)"
                    )
                    data_values[data_id] = None
                    failed_count += 1
                    extraction_stats["failed_extractions"] += 1  # FIX Layer 4

        elif source.source_type == "calculate":
            # Calculate using formula
            try:
                # Start with standard variables that formulas commonly use
                inputs = {
                    'gross': 100000,  # Standard gross salary for ratio calculations
                    'hours_per_week': 40,
                    'weeks_per_year': 52,
                    'months_per_year': 12,
                    'working_days_per_year': 250
                }

                # Add input values from data_values (these override standard vars if same name)
                missing_inputs = []
                for input_id in source.required_inputs:
                    input_val = data_values.get(input_id)
                    inputs[input_id] = input_val
                    if input_val is None:
                        missing_inputs.append(input_id)

                if missing_inputs:
                    logger.warning(
                        f"[UNIFIED EXECUTION] Calculation '{data_id}' missing inputs: {missing_inputs}"
                    )

                # Evaluate formula with standard variables + extracted data
                result = _evaluate_formula(source.formula, inputs)
                data_values[data_id] = result

                # Create Calculation object
                calculations.append(Calculation(
                    description=f"Calculate {data_id}",
                    formula=source.formula,
                    inputs=inputs,
                    result=result,
                    unit="",
                    calculation_id=data_id
                ))

                calculation_count += 1
                logger.info(
                    f"[UNIFIED EXECUTION] Calculated '{data_id}' = {result} "
                    f"using formula: {source.formula}"
                )

            except Exception as e:
                logger.error(
                    f"[UNIFIED EXECUTION] Calculation failed for '{data_id}': {e}",
                    exc_info=True
                )
                data_values[data_id] = None
                failed_count += 1

    # Log execution summary
    logger.info(
        f"[UNIFIED EXECUTION] Data source processing complete: "
        f"{extraction_count} extracted, {calculation_count} calculated, "
        f"{failed_count} failed, {len(data_values)} total values"
    )

    # Step 2: Build response tables as ComparisonEntry objects
    comparisons = []

    for table in plan.response_tables:
        logger.info(
            f"[UNIFIED EXECUTION] Building table '{table.title}': "
            f"{len(table.cells)} cells across {len(table.rows)} rows"
        )

        # FIX: Extract actual column names from cells to detect mismatches
        actual_columns_in_cells = set()
        for cell in table.cells:
            actual_columns_in_cells.add(cell.column)
        
        # Compare with declared table.columns
        declared_columns = set(table.columns)
        
        if actual_columns_in_cells != declared_columns:
            logger.warning(
                f"[UNIFIED EXECUTION] Column mismatch in table '{table.title}': "
                f"declared={sorted(declared_columns)}, actual={sorted(actual_columns_in_cells)}"
            )
            # Use actual columns from cells (source of truth)
            consistent_columns = sorted(list(actual_columns_in_cells))
        else:
            consistent_columns = table.columns
            logger.info(
                f"[UNIFIED EXECUTION] Column names validated: {len(consistent_columns)} columns match"
            )

        # Group cells by row to create ComparisonEntry objects
        row_cells = {}
        for cell in table.cells:
            if cell.row not in row_cells:
                row_cells[cell.row] = {}
            cell_value = data_values.get(cell.data_id)
            row_cells[cell.row][cell.column] = cell_value
            logger.debug(
                f"[UNIFIED EXECUTION] Cell ({cell.row}, {cell.column}): "
                f"data_id='{cell.data_id}', value={cell_value}"
            )

        # Create ComparisonEntry for each row
        row_start_idx = len(comparisons)
        for row, metrics in row_cells.items():
            comparisons.append(ComparisonEntry(
                primary_key=row,
                metrics=metrics,
                source_observation_ids=[]
            ))
            logger.debug(
                f"[UNIFIED EXECUTION] Created comparison for '{row}' "
                f"with {len(metrics)} metrics"
            )
        
        # FIX: Log what keys are actually in the ComparisonEntry objects
        for comp in comparisons[row_start_idx:]:
            logger.info(
                f"[UNIFIED EXECUTION] ComparisonEntry '{comp.primary_key}': "
                f"metrics keys={sorted(comp.metrics.keys())}"
            )
        
        # Store consistent_columns for later use in TableSpec
        table._consistent_columns = consistent_columns

    # FIX #1: Convert unified plan tables to TableSpec format for pipeline compatibility
    # Without this, the StructuredReportPipeline can't build tables (it iterates over table_specifications)
    from ..report_generation.models import TableSpec
    
    table_specs = []
    for table in plan.response_tables:
        # FIX: Use consistent_columns (validated from actual cells) instead of table.columns
        consistent_columns = getattr(table, '_consistent_columns', table.columns)
        table_specs.append(TableSpec(
            table_id=table.table_id,
            purpose=table.title,
            row_entities=table.rows,
            column_metrics=consistent_columns
        ))
        logger.info(
            f"[UNIFIED EXECUTION] Created TableSpec '{table.table_id}' with "
            f"{len(table.rows)} rows × {len(consistent_columns)} columns (validated)"
        )

    # FIX #5: Log data quality metrics for debugging
    populated_values = sum(1 for v in data_values.values() if v is not None)
    none_values = len(data_values) - populated_values
    valid_comparisons = sum(
        1 for comp in comparisons
        if comp.metrics and any(v is not None for v in comp.metrics.values())
    )
    
    logger.info(
        f"[UNIFIED EXECUTION] Data Quality: "
        f"{populated_values}/{len(data_values)} values populated ({none_values} None), "
        f"{valid_comparisons}/{len(comparisons)} comparisons have valid data"
    )
    
    if populated_values == 0:
        logger.error(
            "[UNIFIED EXECUTION] CRITICAL: All extracted values are None! "
            "This will result in empty tables with N/A values."
        )

    # FIX Layer 4: Log extraction analytics for monitoring and debugging
    success_rate = (
        extraction_stats["successful_extractions"] / extraction_stats["total_extractions"] * 100
        if extraction_stats["total_extractions"] > 0 else 0
    )
    logger.info(
        f"[UNIFIED EXECUTION] FIX Layer 4 Analytics: "
        f"Success Rate: {success_rate:.1f}% "
        f"({extraction_stats['successful_extractions']}/{extraction_stats['total_extractions']} extractions)"
    )
    logger.info(
        f"[UNIFIED EXECUTION] Match Strategy Distribution: "
        f"direct_id={extraction_stats['match_types']['direct_id']}, "
        f"entity_match={extraction_stats['match_types']['entity_match']}, "
        f"keyword_match={extraction_stats['match_types']['keyword_match']}, "
        f"not_found={extraction_stats['match_types']['not_found']}"
    )

    # FIX Layer 2: Build extracted_data from data_values for data flow consistency
    # This ensures table rendering and other components have access to extracted values
    extracted_data_points = []
    for data_id, value in data_values.items():
        if value is not None:
            # Parse entity and metric from data_id
            entity, metric = _parse_data_id(data_id)

            # Create DataPoint with extracted value
            extracted_data_points.append(DataPoint(
                entity=entity,
                metric=metric,
                value=value,
                confidence=0.9,  # High confidence from successful extraction
                source_observation_id=None,  # FIXED: Use correct field name (was source_id)
                unit=""  # Unit info may be in source.unit if available
            ))

            logger.debug(
                f"[UNIFIED EXECUTION] Created DataPoint: entity='{entity}', "
                f"metric='{metric}', value={value}"
            )

    logger.info(
        f"[UNIFIED EXECUTION] FIX Layer 2: Converted {len(extracted_data_points)} "
        f"extracted values to DataPoints (was hardcoded to 0)"
    )

    # Create CalculationContext
    calc_context = CalculationContext(
        extracted_data=extracted_data_points,  # FIX Layer 2: Populated from successful extractions
        calculations=calculations,
        key_comparisons=comparisons,
        summary_insights=plan.narrative_points,
        data_quality_notes=[],
        metadata={
            'unified_plan': True,
            'request_analysis': plan.request_analysis.model_dump(),
            'tables': [t.model_dump() for t in plan.response_tables]
        },
        table_specifications=table_specs,  # FIX #1: Populate instead of leaving empty
        structural_understanding=plan.request_analysis.what_user_wants
    )

    logger.info(
        f"[UNIFIED EXECUTION] Complete: "
        f"{len(calculations)} calculations, {len(comparisons)} comparison entries, "
        f"{len(plan.narrative_points)} narrative points"
    )

    # FIX Layer 2: Validate that extracted_data matches expected count
    actual_data_points = len(calc_context.extracted_data)
    if actual_data_points != populated_values:
        logger.warning(
            f"[UNIFIED EXECUTION] Data point count mismatch: "
            f"expected {populated_values}, got {actual_data_points}"
        )
    else:
        logger.info(
            f"[UNIFIED EXECUTION] ✓ FIX Layer 2 validation passed: "
            f"{actual_data_points} data points in extracted_data"
        )

    return calc_context


async def _extract_value_from_observation_llm(
    reporter,
    obs: Dict,
    extraction_path: str,
    data_id: str
) -> Any:
    """Use LLM to intelligently extract values from observations.

    FIX #2: LLM-assisted extraction as specified in the plan.
    Uses structured output to extract specific values with context awareness.

    FIX #7 (NEW): Enhanced error logging and fallback handling.
    - Detailed logging for LLM failures to aid debugging
    - Graceful degradation on LLM errors
    - Timeout handling for slow LLM responses

    Args:
        reporter: Reporter instance with LLM access
        obs: Observation dictionary
        extraction_path: Hint about what to find (e.g., "Spain net income: €XX")
        data_id: Identifier for what we're extracting (e.g., "spain_net_income")

    Returns:
        Extracted value or None
    """
    from pydantic import BaseModel, Field
    from langchain_core.messages import SystemMessage, HumanMessage

    content = obs.get('content', '')
    step_id = obs.get('step_id', 'unknown')

    # Validate inputs
    if not content:
        logger.warning(
            f"[LLM EXTRACTION] Empty content for data_id='{data_id}', step_id={step_id}"
        )
        return None

    if not extraction_path:
        logger.warning(
            f"[LLM EXTRACTION] Empty extraction_path for data_id='{data_id}', step_id={step_id}"
        )
        return None

    # Check if reporter has LLM configured
    if not hasattr(reporter, 'llm') or reporter.llm is None:
        logger.error(
            f"[LLM EXTRACTION] Reporter has no LLM configured! "
            f"Cannot perform LLM extraction for '{data_id}'. "
            f"Falling back to regex extraction."
        )
        return None

    # Parse data_id to extract context
    # e.g., "spain_single_net_takehome" -> entity="Spain", metric="net_takehome"
    parts = data_id.lower().replace('-', '_').split('_')
    entity = parts[0] if parts else "entity"
    metric = '_'.join(parts[1:]) if len(parts) > 1 else "value"

    # Define structured output model
    class ExtractionResult(BaseModel):
        value: Optional[str] = Field(
            default=None,
            description="The extracted value as a string (number, currency, percentage, etc.)"
        )
        found: bool = Field(
            description="True if the value was found in the observation, False otherwise"
        )
        context: Optional[str] = Field(
            default=None,
            description="Brief snippet showing where the value was found"
        )
        reasoning: Optional[str] = Field(
            default=None,
            description="Brief explanation of why this value was selected (for debugging)"
        )

    # Truncate content if too long (keep first 2000 chars for context)
    content_excerpt = content[:2000] if len(content) > 2000 else content
    content_length = len(content)

    logger.debug(
        f"[LLM EXTRACTION] Starting extraction for '{data_id}': "
        f"entity={entity}, metric={metric}, "
        f"content_length={content_length}, excerpt_length={len(content_excerpt)}"
    )

    prompt = f"""Extract a specific value from the research observation below.

WHAT TO EXTRACT:
- Data ID: {data_id}
- Entity: {entity.capitalize()}
- Metric: {metric.replace('_', ' ')}
- Hint: {extraction_path}

OBSERVATION (step {step_id}):
{content_excerpt}

INSTRUCTIONS:
1. Find the value that matches the hint for the specified entity/metric
2. Return ONLY the numeric/text value (remove currency symbols, units, etc.)
3. For percentages, return just the number (e.g., "31.5" not "31.5%")
4. For currency, return just the number (e.g., "150000" not "€150,000")
5. If the value is genuinely not present, set found=False
6. If multiple values exist, choose the one closest to the hint keywords

EXAMPLES:
- Hint: "Spain net income: €172,000" → value: "172000", found: true, reasoning: "Found '€172,000' after 'net income' in text"
- Hint: "France tax rate 31%" → value: "31", found: true, reasoning: "Found '31%' near 'tax rate'"
- Hint: "London rent" (if not in observation) → value: null, found: false, reasoning: "No mention of London or rent in observation"
- Hint: "France social contributions €7,557 for €35,000 gross" → value: "7557", found: true, reasoning: "Found €7,557 directly after 'pays' and before 'in social contributions'"

CRITICAL: When multiple numeric values exist, select the one that appears NEAR the metric keywords (e.g., "social contributions", "tax rate").
"""

    try:
        # Use reporter's LLM with structured output
        logger.debug(
            f"[LLM EXTRACTION] Calling LLM with structured output for '{data_id}'"
        )

        structured_llm = reporter.llm.with_structured_output(
            ExtractionResult,
            method="json_schema"
        )

        messages = [
            SystemMessage(content="You are a precise data extractor. Extract only the requested value, nothing more."),
            HumanMessage(content=prompt)
        ]

        # Make the LLM call
        result = await structured_llm.ainvoke(messages)

        # Log the full result for debugging
        logger.debug(
            f"[LLM EXTRACTION] LLM response for '{data_id}': "
            f"found={result.found}, value={result.value}, "
            f"reasoning={result.reasoning}, context={result.context}"
        )

        if result.found and result.value:
            # Try to convert to float if it looks numeric
            try:
                numeric_value = float(result.value.replace(',', ''))
                logger.info(
                    f"[LLM EXTRACTION] ✓ Extracted '{data_id}' = {numeric_value} "
                    f"from observation {step_id} "
                    f"(reasoning: {result.reasoning or 'N/A'})"
                )
                return numeric_value
            except (ValueError, AttributeError) as conv_err:
                # Return as string if not numeric
                logger.info(
                    f"[LLM EXTRACTION] ✓ Extracted '{data_id}' = '{result.value}' (string) "
                    f"from observation {step_id} "
                    f"(could not convert to numeric: {conv_err})"
                )
                return result.value
        else:
            # LLM explicitly said not found
            logger.warning(
                f"[LLM EXTRACTION] ✗ LLM could not find value for '{data_id}' "
                f"in observation {step_id}. "
                f"Reasoning: {result.reasoning or 'None provided'}. "
                f"Extraction_path: '{extraction_path}'. "
                f"Content preview: '{content[:200]}...'"
            )
            return None

    except TimeoutError as timeout_err:
        logger.error(
            f"[LLM EXTRACTION] ⏱ Timeout for '{data_id}': {timeout_err}. "
            f"LLM took too long to respond. Falling back to regex extraction."
        )
        return None

    except Exception as e:
        # Detailed error logging
        error_type = type(e).__name__
        error_msg = str(e)

        logger.error(
            f"[LLM EXTRACTION] ❌ Exception during extraction for '{data_id}': "
            f"{error_type}: {error_msg}. "
            f"Observation: {step_id}, "
            f"Extraction_path: '{extraction_path}', "
            f"Content_length: {content_length}. "
            f"Falling back to regex extraction.",
            exc_info=True  # Include full stack trace
        )
        return None


def _extract_value_from_observation(obs: Dict, extraction_path: str) -> Any:
    """Extract value from observation using enhanced pattern matching.

    FIX #2: Improved extraction using smarter context-aware pattern matching.
    Previously, this only looked for the first number in the entire observation,
    ignoring context. Now it searches near the extraction_path hint for better accuracy.

    FIX #6 (NEW): Context-aware value selection instead of blindly taking first match.
    When multiple values exist (e.g., "earning €35,000...pays €7,557..."), we now:
    1. Identify target metric keywords from extraction_path
    2. Score each match based on proximity to those keywords
    3. Return the best match, not just the first match

    FIX #8 (NEW): Unicode normalization for consistent character handling.

    Args:
        obs: Observation dictionary
        extraction_path: Hint about where to find the value (e.g., "Spain net income: €XX")

    Returns:
        Extracted value or None
    """
    content = obs.get('content', '')
    step_id = obs.get('step_id', 'unknown')

    logger.debug(
        f"[VALUE EXTRACTION] Extracting from observation {step_id}, "
        f"path hint: '{extraction_path}', content length: {len(content)}"
    )

    if not extraction_path or not content:
        logger.warning(f"[VALUE EXTRACTION] Missing extraction_path or content")
        return None

    # FIX #8: Normalize Unicode in both extraction_path and content
    extraction_path_normalized = _normalize_unicode_text(extraction_path)
    content_normalized = _normalize_unicode_text(content)

    # Parse extraction_path to identify metric keywords and target value hint
    # Example: "France social contributions €7,557 for €35,000 gross (2024) – used as rate proxy"
    # -> metric_keywords: ["france", "social", "contributions"]
    # -> value_hint: "7557" (if present)
    metric_keywords, value_hint = _parse_extraction_path(extraction_path_normalized)

    logger.debug(
        f"[VALUE EXTRACTION] Parsed extraction_path: "
        f"metric_keywords={metric_keywords}, value_hint={value_hint}"
    )

    # Strategy 1: If extraction_path contains a specific value hint, look for that exact amount first
    if value_hint:
        value = _find_exact_value_in_content(content_normalized, value_hint)
        if value is not None:
            logger.info(
                f"[VALUE EXTRACTION] ✓ Found exact value hint {value_hint} -> {value}"
            )
            return value

    # Strategy 2: Context-aware pattern matching with proximity scoring
    # Find all numeric values in the content
    all_matches = _find_all_numeric_matches(content_normalized)

    if not all_matches:
        logger.warning(
            f"[VALUE EXTRACTION] No numeric values found in observation {step_id}"
        )
        return None

    # Score each match based on proximity to metric keywords
    scored_matches = _score_matches_by_proximity(
        all_matches, metric_keywords, content_normalized.lower()
    )

    if not scored_matches:
        logger.warning(
            f"[VALUE EXTRACTION] No scored matches for keywords {metric_keywords}"
        )
        return None

    # Return the best match (highest score)
    best_match = scored_matches[0]
    best_value = best_match['value']
    best_score = best_match['score']

    logger.info(
        f"[VALUE EXTRACTION] ✓ Selected best match: {best_value} "
        f"(score: {best_score:.2f}, raw: '{best_match['raw_text']}', "
        f"position: {best_match['position']})"
    )

    # Log runner-up matches for debugging
    if len(scored_matches) > 1:
        runner_ups = [
            f"{m['raw_text']} (score: {m['score']:.2f})"
            for m in scored_matches[1:3]
        ]
        logger.debug(
            f"[VALUE EXTRACTION] Runner-ups: {', '.join(runner_ups)}"
        )

    return best_value


def _normalize_unicode_text(text: str) -> str:
    """Normalize Unicode text to handle special characters consistently.

    FIX #8 (NEW): Normalize special Unicode characters to prevent matching issues.
    Handles:
    - Non-breaking hyphens (U+2011 ‑) -> regular hyphen (-)
    - En-dashes (U+2013 –) -> regular hyphen (-)
    - Em-dashes (U+2014 —) -> regular hyphen (-)
    - Non-breaking spaces (U+00A0, U+202F) -> regular space ( )
    - Various other Unicode variants

    Args:
        text: Input text with potential Unicode variants

    Returns:
        Normalized text with consistent ASCII equivalents
    """
    if not text:
        return text

    # NFKC normalization: Compatibility decomposition, then canonical composition
    # This converts variant forms to their canonical equivalents
    normalized = unicodedata.normalize('NFKC', text)

    # Additional manual replacements for common problematic characters
    replacements = {
        '\u2011': '-',  # Non-breaking hyphen
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u00A0': ' ',  # Non-breaking space
        '\u202F': ' ',  # Narrow no-break space
        '\u2010': '-',  # Hyphen
        '\u2212': '-',  # Minus sign
        '\u00AD': '',   # Soft hyphen (remove)
    }

    for unicode_char, replacement in replacements.items():
        normalized = normalized.replace(unicode_char, replacement)

    return normalized


def _parse_extraction_path(extraction_path: str) -> Tuple[List[str], Optional[str]]:
    """Parse extraction_path to identify metric keywords and value hints.

    Args:
        extraction_path: Hint string like "France social contributions €7,557 for €35,000 gross"

    Returns:
        Tuple of (metric_keywords, value_hint)
        - metric_keywords: List of significant words (length > 3, no stopwords)
        - value_hint: Numeric value mentioned in path (e.g., "7557"), or None
    """
    # FIX #8: Normalize Unicode before parsing
    extraction_path = _normalize_unicode_text(extraction_path)

    # Remove metadata suffixes like "– used as rate proxy"
    clean_path = re.sub(r'\s*[–—-]\s*used as.*$', '', extraction_path, flags=re.IGNORECASE)
    clean_path = re.sub(r'\s*\(.*?\)\s*', ' ', clean_path)  # Remove parenthetical notes

    # Extract value hint if present (e.g., "€7,557" -> "7557")
    value_hint = None
    value_patterns = [
        r'[€$£]\s*([\d,]+(?:\.\d+)?)',  # Currency amounts
        r'(\d+(?:\.\d+)?)\s*%',          # Percentages
        r'\b([\d,]+(?:\.\d+)?)\b'        # Plain numbers
    ]

    for pattern in value_patterns:
        match = re.search(pattern, extraction_path)
        if match:
            # Extract numeric part only
            value_hint = match.group(1).replace(',', '')
            break

    # Extract metric keywords (significant words, no numbers/symbols)
    # Stopwords to exclude
    stopwords = {'for', 'the', 'and', 'or', 'in', 'at', 'to', 'of', 'a', 'an', 'as', 'is', 'on', 'by'}

    words = clean_path.lower().split()
    metric_keywords = [
        word for word in words
        if len(word) > 3  # Significant length
        and word not in stopwords
        and not re.match(r'^[\d\W]+$', word)  # Not just numbers/symbols
    ]

    return metric_keywords, value_hint


def _find_exact_value_in_content(content: str, value_hint: str) -> Optional[float]:
    """Find exact value hint in content (e.g., "7557" from "€7,557").

    Args:
        content: Observation content
        value_hint: Numeric value to find (e.g., "7557")

    Returns:
        Float value if found, None otherwise
    """
    # Try to find the value with various formatting
    # e.g., "7557" could appear as "€7,557" or "7557" or "7,557"
    target_float = float(value_hint)

    # Pattern to match currency amounts or plain numbers
    pattern = r'[€$£]?\s*[\d,]+(?:\.\d+)?'
    matches = re.findall(pattern, content)

    for match in matches:
        # Clean and parse
        clean = match.replace('€', '').replace('$', '').replace('£', '').replace(',', '').strip()
        try:
            value = float(clean)
            # Check if this matches our target (allowing small floating point differences)
            if abs(value - target_float) < 0.01:
                return value
        except ValueError:
            continue

    return None


def _find_all_numeric_matches(content: str) -> List[Dict[str, Any]]:
    """Find all numeric values in content with their positions and raw text.

    Args:
        content: Observation content

    Returns:
        List of dicts with keys: 'value' (float), 'raw_text' (str), 'position' (int)
    """
    matches = []

    # Patterns in priority order (most specific first)
    patterns = [
        (r'[€$£]\s*[\d,]+(?:\.\d+)?', lambda m: m.replace('€', '').replace('$', '').replace('£', '').replace(',', '')),  # Currency
        (r'\d+(?:\.\d+)?%', lambda m: m.replace('%', '')),  # Percentages
        (r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', lambda m: m.replace(',', '')),  # Thousands
        (r'\d+\.\d+', lambda m: m),  # Decimals
        (r'\b\d+\b', lambda m: m),  # Integers
    ]

    for pattern, cleaner in patterns:
        for match in re.finditer(pattern, content):
            raw_text = match.group()
            position = match.start()

            try:
                clean_str = cleaner(raw_text)
                value = float(clean_str)

                matches.append({
                    'value': value,
                    'raw_text': raw_text,
                    'position': position
                })
            except ValueError:
                continue

    # Sort by position (left to right in text)
    matches.sort(key=lambda x: x['position'])

    # Deduplicate overlapping matches (keep more specific patterns)
    deduplicated = []
    used_positions = set()

    for match in matches:
        pos = match['position']
        # Check if this position overlaps with any used position
        overlap = False
        for used_pos in used_positions:
            if abs(pos - used_pos) < 5:  # Within 5 chars = overlap
                overlap = True
                break

        if not overlap:
            deduplicated.append(match)
            used_positions.add(pos)

    return deduplicated


def _score_matches_by_proximity(
    matches: List[Dict[str, Any]],
    metric_keywords: List[str],
    content_lower: str
) -> List[Dict[str, Any]]:
    """Score numeric matches based on proximity to metric keywords.

    Scoring logic:
    - Higher score = better match
    - Proximity: Closer to metric keywords = higher score
    - Order: Earlier in text = slightly higher score (tie-breaker)

    Args:
        matches: List of match dicts from _find_all_numeric_matches
        metric_keywords: Keywords identifying the target metric
        content_lower: Lowercase content for keyword search

    Returns:
        Matches sorted by score (best first), with added 'score' field
    """
    scored = []

    for match in matches:
        pos = match['position']

        # Calculate minimum distance to any metric keyword
        min_distance = float('inf')

        for keyword in metric_keywords:
            # Find all occurrences of this keyword
            keyword_positions = [
                m.start() for m in re.finditer(re.escape(keyword), content_lower)
            ]

            for kw_pos in keyword_positions:
                distance = abs(pos - kw_pos)
                min_distance = min(min_distance, distance)

        # Score based on proximity (inverse of distance)
        # Closer = higher score
        if min_distance < float('inf'):
            # Score: 1000 / (distance + 1)
            # This gives high scores for close matches, decreasing as distance increases
            proximity_score = 1000.0 / (min_distance + 1)
        else:
            # No keywords found - give low score
            proximity_score = 1.0

        # Add small position bonus (earlier in text = slightly higher)
        # This is a tie-breaker when distances are similar
        position_bonus = (1000 - min(pos, 1000)) / 1000.0

        total_score = proximity_score + position_bonus

        scored.append({
            **match,
            'score': total_score,
            'min_distance': min_distance
        })

    # Sort by score (highest first)
    scored.sort(key=lambda x: x['score'], reverse=True)

    return scored


def _evaluate_formula(formula: str, inputs: Dict[str, Any]) -> Any:
    """Safely evaluate formula with inputs.

    Args:
        formula: Mathematical formula string
        inputs: Dictionary of variable name -> value

    Returns:
        Calculation result or None
    """
    logger.debug(
        f"[FORMULA EVAL] Evaluating formula: '{formula}' "
        f"with inputs: {inputs}"
    )

    # Replace variable names with values
    expr = formula
    for var, value in inputs.items():
        if value is not None:
            expr = expr.replace(var, str(value))

    # Check if all variables were replaced
    missing_vars = [inp for inp in inputs.keys() if inp in expr]
    if missing_vars:
        # Some inputs missing
        logger.warning(
            f"[FORMULA EVAL] Missing or None inputs in formula '{formula}': {missing_vars}"
        )
        return None

    try:
        # Safe eval (only math operations, no builtins)
        logger.debug(f"[FORMULA EVAL] Evaluating expression: '{expr}'")
        result = eval(expr, {"__builtins__": {}}, {})
        final_result = float(result) if isinstance(result, (int, float)) else result
        logger.info(
            f"[FORMULA EVAL] Successfully evaluated '{formula}' = {final_result}"
        )
        return final_result
    except Exception as e:
        logger.error(
            f"[FORMULA EVAL] Evaluation failed for expression '{expr}' "
            f"(original: '{formula}'): {e}",
            exc_info=True
        )
        return None


__all__ = ["MetricSpecAnalyzer", "generate_calculation_context"]
