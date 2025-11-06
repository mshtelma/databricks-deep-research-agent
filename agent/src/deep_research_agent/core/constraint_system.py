"""
Abstract Constraint Extraction and Enforcement System

This module provides a flexible, domain-agnostic system for extracting and enforcing
constraints from user queries. It handles entities (countries, companies, products),
metrics (numbers, rates, percentages), comparisons, and data format requirements.

The system is designed to work with ANY type of query, not just specific use cases.

IMPORTANT: QueryConstraints and ScenarioDefinition now live in structured_models.py as
Pydantic models (ConstraintsOutput and ScenarioDefinition). This module provides type
aliases for backward compatibility during migration.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import re
import json
from .structured_output import extract_json
from .structured_models import (
    ConstraintsOutput,
    ScenarioOutput,
    ScenarioDefinition,
    ComparisonType
)

logger = logging.getLogger(__name__)

# DIAGNOSTIC: Verify implementation version is loaded
logger.error("="*100)
logger.error("🔥 CONSTRAINT_SYSTEM MODULE LOADED - IMPLEMENTATION V2 (2025-11-03)")
logger.error("  - ScenarioParameterValidator: ACTIVE")
logger.error("  - LLM Correction Loop: ACTIVE (MAX_RETRY_ATTEMPTS=2)")
logger.error("="*100)
print("="*100)
print("🔥 CONSTRAINT_SYSTEM MODULE LOADED - IMPLEMENTATION V2 (2025-11-03)")
print("  - ScenarioParameterValidator: ACTIVE")
print("  - LLM Correction Loop: ACTIVE (MAX_RETRY_ATTEMPTS=2)")
print("="*100)

# Type alias for backward compatibility - ConstraintsOutput is the new QueryConstraints
QueryConstraints = ConstraintsOutput


# ============================================================================
# Dataclasses DELETED - Now Pydantic models in structured_models.py
# ============================================================================
# ScenarioDefinition → structured_models.ScenarioDefinition (Pydantic)
# QueryConstraints → structured_models.ConstraintsOutput (Pydantic, aliased above)


# Removed old dataclass definitions - see structured_models.py for Pydantic versions
# This section intentionally left with comment to show what was removed


# ============================================================================
# ConstraintExtractor (kept, but updated to return Pydantic)
# ============================================================================

# DELETED OLD DATACLASS - START MARKER
# (Was: @dataclass class QueryConstraints with all fields and methods)
# Now: ConstraintsOutput in structured_models.py (Pydantic)
# DELETED OLD DATACLASS - END MARKER


class ScenarioParameterValidator:
    """Validates scenario parameters without performing extraction.
    
    This validator DETECTS whether numeric values should be present,
    but does NOT extract them. Extraction is done by LLM only.
    """
    
    @staticmethod
    def contains_numeric_mentions(text: str) -> bool:
        """
        Detect if text contains numeric patterns that should be extracted.
        This is detection only - does NOT extract values.
        
        Returns:
            True if text likely contains extractable numerics
        """
        # Patterns indicating numbers (detection, not extraction)
        patterns = [
            r'[€$£¥]\s*\d+[,\d]*',  # Currency amounts
            r'\d+[,\d]*\s*[€$£¥]',  # Amounts with trailing currency
            r'\d+k\b',  # Shorthand like 150k
            r'\d+\s*child',  # Child counts
            r'\d+\s*bedroom',  # Property details
            r'\d+/month',  # Monthly amounts
            r'annual.*?\d+',  # Annual amounts
            r'salary.*?\d+|\d+.*?salary',  # Salary mentions
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def validate_scenario(scenario) -> dict:  # Remove type hint to accept both types
        """
        Validate scenario parameters against description.
        
        Handles both ScenarioOutput (list parameters) and ScenarioDefinition (dict parameters).
        
        Returns:
            dict with keys: 'valid', 'has_numerics', 'has_parameters', 'warning'
        """
        has_numerics = ScenarioParameterValidator.contains_numeric_mentions(
            scenario.description
        )
        
        # Handle both list (ScenarioOutput) and dict (ScenarioDefinition) parameters
        if isinstance(scenario.parameters, list):
            has_parameters = bool(scenario.parameters and len(scenario.parameters) > 0)
        elif isinstance(scenario.parameters, dict):
            has_parameters = bool(scenario.parameters and len(scenario.parameters) > 0)
        else:
            # Unknown type - log error
            logger.error(f"[validate_scenario] Unknown parameters type: {type(scenario.parameters)}")
            has_parameters = False
        
        valid = True
        warning = None
        
        if has_numerics and not has_parameters:
            valid = False
            warning = (
                f"Scenario '{scenario.name}' description contains numeric values "
                f"but parameters are empty. Description: {scenario.description[:100]}"
            )
        
        return {
            'valid': valid,
            'has_numerics': has_numerics,
            'has_parameters': has_parameters,
            'warning': warning,
            'param_type': type(scenario.parameters).__name__,  # For debugging
            'param_value': str(scenario.parameters)[:100]  # For debugging
        }


class ConstraintExtractor:
    """Extract constraints from user queries using LLM or pattern matching."""

    def __init__(self, llm=None, config=None):
        self.llm = llm
        self.config = config or {}
        self.entity_patterns = self._build_entity_patterns()

    def _build_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for common entity detection."""
        return {
            "countries": re.compile(
                r'\b(?:Spain|France|Germany|Poland|Bulgaria|UK|United Kingdom|'
                r'Switzerland|Italy|Netherlands|Belgium|Denmark|Sweden|Norway|'
                r'Austria|Portugal|Greece|Ireland|Finland|Czech Republic|Hungary|'
                r'Romania|Croatia|Slovenia|Slovakia|Luxembourg|Malta|Cyprus|'
                r'United States|USA|Canada|Mexico|Brazil|Argentina|Chile|'
                r'China|Japan|India|Australia|New Zealand|South Korea|Singapore)\b',
                re.IGNORECASE
            ),
            "currencies": re.compile(r'[€$£¥₹]\d+(?:,\d{3})*(?:\.\d+)?(?:[KMB])?'),
            "percentages": re.compile(r'\d+(?:\.\d+)?%'),
            "years": re.compile(r'\b(?:19|20)\d{2}\b'),
            "comparisons": re.compile(
                r'\b(?:compare|comparison|versus|vs\.?|against|between|'
                r'difference|contrast|relative to)\b',
                re.IGNORECASE
            )
        }

    def extract_constraints(self, query: str, state: Dict = None) -> QueryConstraints:
        """
        Extract all constraints from user query.

        Args:
            query: The user's query text
            state: Optional state dict with additional context

        Returns:
            QueryConstraints object with all extracted constraints
        """
        # DIAGNOSTIC: Entry point logging
        print("="*100)
        print(f"🔍 [EXTRACT_CONSTRAINTS] ENTRY POINT")
        print(f"  - query length: {len(query) if query else 0}")
        print(f"  - self.llm: {self.llm}")
        print(f"  - self.llm type: {type(self.llm)}")
        print("="*100)
        logger.error("="*100)
        logger.error("🔍 [EXTRACT_CONSTRAINTS] ENTRY POINT")
        logger.error(f"  - query length: {len(query) if query else 0}")
        logger.error(f"  - self.llm: {self.llm}")
        logger.error(f"  - self.llm type: {type(self.llm)}")
        logger.error("="*100)
        
        if not query:
            print("❌ [EXTRACT_CONSTRAINTS] Query is empty, returning empty constraints")
            logger.error("❌ [EXTRACT_CONSTRAINTS] Query is empty, returning empty constraints")
            return QueryConstraints()

        # Try LLM extraction first for best results
        if self.llm:
            print(f"✅ [EXTRACT_CONSTRAINTS] LLM is set, calling _extract_with_llm()")
            logger.error(f"✅ [EXTRACT_CONSTRAINTS] LLM is set, calling _extract_with_llm()")
            try:
                result = self._extract_with_llm(query, state)
                print(f"✅ [EXTRACT_CONSTRAINTS] _extract_with_llm() returned successfully")
                logger.error(f"✅ [EXTRACT_CONSTRAINTS] _extract_with_llm() returned successfully")
                return result
            except Exception as e:
                print(f"❌ [EXTRACT_CONSTRAINTS] _extract_with_llm() raised exception: {type(e).__name__}: {e}")
                logger.error(f"❌ [EXTRACT_CONSTRAINTS] _extract_with_llm() raised exception: {type(e).__name__}: {e}")
                logger.warning(f"LLM constraint extraction failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
        else:
            print("❌ [EXTRACT_CONSTRAINTS] self.llm is None, using pattern-based fallback")
            logger.error("❌ [EXTRACT_CONSTRAINTS] self.llm is None, using pattern-based fallback")

        # Fallback to pattern-based extraction
        print("🔄 [EXTRACT_CONSTRAINTS] Using pattern-based extraction fallback")
        logger.error("🔄 [EXTRACT_CONSTRAINTS] Using pattern-based extraction fallback")
        return self._extract_with_patterns(query)

    def _extract_with_llm(self, query: str, state: Dict = None) -> QueryConstraints:
        """Extract constraints using native structured generation.

        Benefits over function calling:
        1. More efficient for rate-limited models
        2. Native response_format support
        3. Guaranteed structure - no parsing errors
        4. Automatic validation - Pydantic handles types
        5. Less overhead than function calling
        """
        print("🔍 [_EXTRACT_WITH_LLM] ENTERED")
        logger.error("🔍 [_EXTRACT_WITH_LLM] ENTERED")

        # Import json locally to avoid UnboundLocalError
        import json as json_module
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = """You are a constraint extraction expert analyzing research queries.

TASK: Extract ALL entities, metrics, and scenarios with COMPLETE numeric parameters.

════════════════════════════════════════════════════════════════
🚨 CRITICAL: EXTRACT ALL SCENARIOS FROM THE QUERY 🚨
════════════════════════════════════════════════════════════════

The query may contain MULTIPLE scenarios in various formats:
- Numbered lists (1. Single 2. Married)
- Bulleted lists (• Single • Married)
- Descriptive text (single person... married couple...)

YOU MUST EXTRACT EVERY SCENARIO MENTIONED!

Common scenario patterns to look for:
- "1. single: ..." AND "2. married: ..." → Extract BOTH scenarios
- "Single earner" AND "Married couple" → Extract BOTH scenarios
- "first scenario" AND "second scenario" → Extract BOTH scenarios

════════════════════════════════════════════════════════════════
PART 1: SCENARIOS WITH NUMERIC PARAMETERS (HIGHEST PRIORITY)
════════════════════════════════════════════════════════════════

🚨🚨🚨 CRITICAL REQUIREMENTS:
1. Extract ALL scenarios mentioned (not just the first one!)
2. Extract ALL numeric parameters from each scenario
3. Parameters MUST be list of [key, value] pairs: [["salary", 150000]]

📋 PARAMETER EXTRACTION RULES:

For EACH scenario, extract these parameters:
1. Salary/Income → [["salary", <number>]]
2. RSU/Stock/Bonus → [["rsu", <number>]]
3. Spouse income → [["salary_spouse", <number>]]
4. Children count → [["children", <number>]]
5. Rent → [["rent_monthly", <number>]]

📐 NUMBER CONVERSION (CRITICAL):
- "€150,000" or "€150k" → 150000
- "€100,000" or "€100k" → 100000
- "$350,000" or "$350k" → 350000
- "2 children" → 2
- "€2000/month" → 2000

✅ COMPLETE EXAMPLE - Tax comparison with 2 scenarios:

Query: "Compare taxes between Spain and Poland for:
1. single: €150,000 annual gross salary + €100,000 annual RSUs
2. married, no child: same primary earner (€150k + €100k RSUs) + spouse with €100,000 salary"

CORRECT Output (BOTH scenarios with parameters):
{
  "entities": ["Spain", "Poland"],
  "metrics": ["net_take_home", "effective_tax_rate", "disposable_income"],
  "scenarios": [
    {
      "id": "s1",
      "name": "Single",
      "description": "Single earner with €150,000 annual gross salary plus €100,000 annual RSUs",
      "parameters": [["salary", 150000], ["rsu", 100000]]
    },
    {
      "id": "s2",
      "name": "Married, no child",
      "description": "Married couple with no children; primary earner €150,000 salary + €100,000 RSUs, spouse €100,000 salary",
      "parameters": [["salary", 150000], ["rsu", 100000], ["salary_spouse", 100000]]
    }
  ],
  "comparison_type": "country",
  "topics": ["taxation"],
  "monetary_values": ["€150000", "€100000"],
  "data_format": "table"
}

❌ WRONG - Only extracting first scenario:
{
  "scenarios": [
    {"id": "s1", "name": "Single", "parameters": [["salary", 150000]]}
  ]
  // MISSING THE MARRIED SCENARIO!
}

❌ WRONG - Empty parameters when numbers present:
{
  "scenarios": [
    {"id": "s1", "name": "Single", "description": "€150k salary", "parameters": []}
  ]
  // PARAMETERS CANNOT BE EMPTY WHEN NUMBERS ARE IN DESCRIPTION!
}

════════════════════════════════════════════════════════════════
PART 2: ENTITIES (What to compare)
════════════════════════════════════════════════════════════════

Extract ALL countries, cities, companies mentioned for comparison

════════════════════════════════════════════════════════════════
PART 3: METRICS (What to measure)
════════════════════════════════════════════════════════════════

Extract measurable values: tax_rate, net_take_home, rent, disposable_income

════════════════════════════════════════════════════════════════
REMEMBER:
1. Extract ALL scenarios (not just the first one)
2. Parameters as list of [key, value] pairs
3. Convert currency shorthand (€150k → 150000)
════════════════════════════════════════════════════════════════
"""

        human_prompt = f"""Extract all constraints from this query:
"{query}"

CRITICAL REMINDERS:
1. Extract ALL scenarios (e.g., if query mentions "1. single" AND "2. married", extract BOTH)
2. Extract numeric parameters for EACH scenario as [["key", value]] pairs
3. Look for numbered lists, bullet points, or any mention of different cases/scenarios

Be comprehensive - extract ALL entities, metrics, and scenarios mentioned.
Respond with valid JSON matching the schema above."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # DEBUG: Log the prompt being sent to LLM
        logger.error("="*100)
        logger.error("🔍 [CONSTRAINT EXTRACTION] SENDING INITIAL PROMPT TO LLM")
        logger.error(f"  Query length: {len(query)}")
        logger.error(f"  Query preview: {query[:200]}...")
        logger.error(f"  System prompt length: {len(system_prompt)}")
        logger.error(f"  Human prompt length: {len(human_prompt)}")
        logger.error("="*100)
        print("="*100)
        print("🔍 [CONSTRAINT EXTRACTION] SENDING INITIAL PROMPT TO LLM")
        print(f"  Query: {query[:200]}...")
        print("="*100)

        try:
            # CRITICAL FIX: Use json_schema method for strict enforcement
            # json_schema: Full schema validation with Databricks response_format
            # json_mode: Generic JSON object (less strict, can cause validation errors)
            logger.error("🔍 [CONSTRAINT EXTRACTION] Calling LLM.with_structured_output()...")
            print("🔍 [CONSTRAINT EXTRACTION] Calling LLM.with_structured_output()...")

            response = self.llm.with_structured_output(
                schema=ConstraintsOutput,
                method="json_schema",  # CHANGED: Strict schema enforcement (was json_mode)
                include_raw=False
            ).invoke(messages)

            logger.error(f"🔍 [CONSTRAINT EXTRACTION] LLM returned response type: {type(response)}")
            print(f"🔍 [CONSTRAINT EXTRACTION] LLM returned response type: {type(response)}")

            # AGGRESSIVE LOGGING: Log the actual response
            if isinstance(response, ConstraintsOutput):
                logger.error(f"🔍 [CONSTRAINT EXTRACTION] Response scenarios count: {len(response.scenarios)}")
                print(f"🔍 [CONSTRAINT EXTRACTION] Response scenarios count: {len(response.scenarios)}")
                for i, scenario in enumerate(response.scenarios):
                    logger.error(f"  Scenario {i}: {scenario.name} - params: {scenario.parameters}")
                    print(f"  Scenario {i}: {scenario.name} - params: {scenario.parameters}")

            # with_structured_output() should ALWAYS return a validated Pydantic model
            # If not, something is seriously wrong (not a JSON formatting issue)
            if not isinstance(response, ConstraintsOutput):
                logger.error(
                    f"Structured generation returned unexpected type: {type(response)}. "
                    "This should never happen with response_format. Falling back to pattern-based extraction."
                )
                return self._create_fallback_constraints(query)

            constraints_output = response

            logger.info(
                f"✅ Extracted via structured generation: "
                f"{len(constraints_output.entities)} entities, "
                f"{len(constraints_output.metrics)} metrics, "
                f"{len(constraints_output.scenarios)} scenarios"
            )

            # DIAGNOSTIC: Log scenarios before validation
            logger.error("="*100)
            logger.error(f"🔍 [DIAGNOSTIC] SCENARIOS BEFORE VALIDATION (count={len(constraints_output.scenarios)})")
            print("="*100)
            print(f"🔍 [DIAGNOSTIC] SCENARIOS BEFORE VALIDATION (count={len(constraints_output.scenarios)})")
            for idx, scenario in enumerate(constraints_output.scenarios):
                logger.error(f"  Scenario {idx}: {type(scenario).__name__}")
                logger.error(f"    - name: {scenario.name}")
                logger.error(f"    - description: {scenario.description[:80]}...")
                logger.error(f"    - parameters type: {type(scenario.parameters)}")
                logger.error(f"    - parameters value: {scenario.parameters}")
                logger.error(f"    - parameters len: {len(scenario.parameters) if hasattr(scenario.parameters, '__len__') else 'N/A'}")
                logger.error(f"    - parameters bool: {bool(scenario.parameters)}")
                print(f"  Scenario {idx}: {type(scenario).__name__}, name={scenario.name}, param_type={type(scenario.parameters)}, param_len={len(scenario.parameters) if hasattr(scenario.parameters, '__len__') else 'N/A'}, params={scenario.parameters}")
            logger.error("="*100)
            print("="*100)

            # NEW: Validate and retry if needed
            # Get MAX_RETRY_ATTEMPTS from config (default to 3 if not set)
            MAX_RETRY_ATTEMPTS = self.config.get('metrics', {}).get('scenario_parameter_correction', {}).get('max_retry_attempts', 3)
            logger.error(f"🔥🔥🔥 [CORRECTION] MAX_RETRY_ATTEMPTS from config = {MAX_RETRY_ATTEMPTS}")
            print(f"🔥🔥🔥 [CORRECTION] MAX_RETRY_ATTEMPTS from config = {MAX_RETRY_ATTEMPTS}")

            validator = ScenarioParameterValidator()
            retry_needed = []

            for scenario in constraints_output.scenarios:
                validation = validator.validate_scenario(scenario)
                logger.error(f"[DIAGNOSTIC] Validation result for '{scenario.name}': {validation}")
                print(f"[DIAGNOSTIC] Validation result for '{scenario.name}': {validation}")
                if not validation['valid']:
                    retry_needed.append({
                        'scenario': scenario,
                        'warning': validation['warning']
                    })
                    logger.error(f"❌ [extract_constraints] Validation FAILED: {validation['warning']}")
                    logger.error(f"   - has_numerics: {validation['has_numerics']}")
                    logger.error(f"   - has_parameters: {validation['has_parameters']}")
                    logger.error(f"   - param_type: {validation.get('param_type', 'unknown')}")
                    print(f"❌ [extract_constraints] Validation FAILED: {validation['warning']}")
                    print(f"   - has_numerics: {validation['has_numerics']}, has_parameters: {validation['has_parameters']}, param_type: {validation.get('param_type', 'unknown')}")

            # CRITICAL DEBUG: Log retry_needed state BEFORE condition check
            logger.error("="*100)
            logger.error(f"🔥🔥🔥 [CORRECTION] BEFORE CONDITION CHECK:")
            logger.error(f"  - retry_needed length: {len(retry_needed)}")
            logger.error(f"  - retry_needed bool: {bool(retry_needed)}")
            logger.error(f"  - MAX_RETRY_ATTEMPTS: {MAX_RETRY_ATTEMPTS}")
            logger.error(f"  - MAX_RETRY_ATTEMPTS > 0: {MAX_RETRY_ATTEMPTS > 0}")
            logger.error(f"  - Combined condition: {bool(retry_needed) and MAX_RETRY_ATTEMPTS > 0}")
            if retry_needed:
                logger.error(f"  - retry_needed scenarios: {[item['scenario'].name for item in retry_needed]}")
            logger.error("="*100)
            print("="*100)
            print(f"🔥🔥🔥 [CORRECTION] retry_needed={len(retry_needed)}, MAX_RETRY_ATTEMPTS={MAX_RETRY_ATTEMPTS}, will_enter_loop={bool(retry_needed) and MAX_RETRY_ATTEMPTS > 0}")
            print("="*100)

            # Retry loop for invalid scenarios
            if retry_needed and MAX_RETRY_ATTEMPTS > 0:
                logger.error("🔥🔥🔥 [CORRECTION] ✅ ENTERING CORRECTION LOOP!")
                print("🔥🔥🔥 [CORRECTION] ✅ ENTERING CORRECTION LOOP!")
                logger.info(
                    f"[extract_constraints] {len(retry_needed)} scenario(s) need parameter correction. "
                    f"Attempting targeted retry..."
                )
                
                for attempt in range(MAX_RETRY_ATTEMPTS):
                    logger.error(f"🔥 [CORRECTION] Loop iteration {attempt + 1}/{MAX_RETRY_ATTEMPTS}")
                    print(f"🔥 [CORRECTION] Loop iteration {attempt + 1}/{MAX_RETRY_ATTEMPTS}")

                    if not retry_needed:
                        logger.error(f"🔥 [CORRECTION] retry_needed is now empty, breaking out of loop")
                        print(f"🔥 [CORRECTION] retry_needed is now empty, breaking out of loop")
                        break

                    logger.error(f"🔥 [CORRECTION] Attempting to fix {len(retry_needed)} scenarios")
                    print(f"🔥 [CORRECTION] Attempting to fix {len(retry_needed)} scenarios")

                    # Build correction prompt
                    scenarios_to_fix = [item['scenario'] for item in retry_needed]

                    # Prepare JSON strings outside f-string to avoid UnboundLocalError
                    entities_json = json_module.dumps(constraints_output.entities)
                    metrics_json = json_module.dumps(constraints_output.metrics)
                    scenarios_info = [{'name': s.name, 'description': s.description, 'parameters': s.parameters} for s in scenarios_to_fix]

                    correction_prompt = f"""
The following scenarios were extracted but are missing numeric parameters.

Original query: "{query}"

Already extracted entities: {entities_json}
Already extracted metrics: {metrics_json}

Scenarios needing correction:
{scenarios_info}

TASK: For EACH scenario above, extract the numeric parameters mentioned in the description.

CRITICAL PARAMETER FORMAT - Use list of [key, value] pairs:
- "€150,000 annual gross salary" → [["salary", 150000]]
- "€100,000 annual RSUs" → [["rsu", 100000]]
- "spouse with €100,000 salary" → [["salary_spouse", 100000]]
- "1 child" → [["children", 1]]
- "€2000/month rent" → [["rent_monthly", 2000]]

COMPLETE EXAMPLE - Look at the descriptions and extract ALL numeric values:

For "Single earner with €150,000 annual gross salary plus €100,000 annual RSUs":
parameters: [["salary", 150000], ["rsu", 100000]]

For "Married couple; primary €150,000 salary + €100,000 RSUs, spouse €100,000 salary":
parameters: [["salary", 150000], ["rsu", 100000], ["salary_spouse", 100000]]

Return the complete ConstraintsOutput in JSON format:
{{
  "entities": {entities_json},
  "metrics": {metrics_json},
  "scenarios": [
    {{"id": "s1", "name": "Single", "description": "...", "parameters": [["salary", 150000], ["rsu", 100000]]}},
    {{"id": "s2", "name": "Married, no child", "description": "...", "parameters": [["salary", 150000], ["rsu", 100000], ["salary_spouse", 100000]]}}
  ]
}}
"""

                    logger.error(f"🔥 [CORRECTION] Sending correction prompt to LLM...")
                    print(f"🔥 [CORRECTION] Sending correction prompt to LLM...")

                    correction_messages = [
                        SystemMessage(content="""You are a numeric parameter extraction specialist.
Your task is to extract numeric values from scenario descriptions and return a complete ConstraintsOutput.

CRITICAL RULES:
1. Parameters MUST be a list of [key, value] pairs: [["salary", 150000], ["rsu", 100000]]
2. Extract ALL numeric values mentioned in descriptions
3. Convert currency shorthand: €150k → 150000, €100k → 100000
4. Use standard keys: salary, rsu, salary_spouse, rsu_spouse, children
5. Values must be numbers, not strings: 150000 not "150000"

Example scenario extraction:
"Single earner with €150,000 annual gross salary plus €100,000 annual RSUs"
→ parameters: [["salary", 150000], ["rsu", 100000]]

"Married couple; primary €150k salary + €100k RSUs, spouse €100k"
→ parameters: [["salary", 150000], ["rsu", 100000], ["salary_spouse", 100000]]"""),
                        HumanMessage(content=correction_prompt)
                    ]

                    try:
                        logger.error(f"🔥 [CORRECTION] Calling LLM with structured output...")
                        print(f"🔥 [CORRECTION] Calling LLM with structured output...")

                        # Retry extraction with focused prompt
                        correction_response = self.llm.with_structured_output(
                            schema=ConstraintsOutput,  # Reuse schema
                            method="json_schema",
                            include_raw=False
                        ).invoke(correction_messages)

                        logger.error(f"🔥 [CORRECTION] LLM returned response with {len(correction_response.scenarios)} scenarios")
                        print(f"🔥 [CORRECTION] LLM returned response with {len(correction_response.scenarios)} scenarios")
                        
                        # Replace scenarios with corrected versions
                        corrected_count = 0
                        still_invalid = []
                        
                        for i, original_scenario in enumerate(constraints_output.scenarios):
                            # Find matching corrected scenario
                            corrected = next(
                                (s for s in correction_response.scenarios if s.id == original_scenario.id),
                                None
                            )
                            
                            if corrected:
                                # Validate correction
                                validation = validator.validate_scenario(corrected)
                                if validation['valid'] or validation['has_parameters']:
                                    constraints_output.scenarios[i] = corrected
                                    corrected_count += 1
                                    logger.info(
                                        f"[extract_constraints] Corrected scenario '{corrected.name}': "
                                        f"parameters={dict(corrected.parameters) if hasattr(corrected, 'parameters') else corrected.parameters}"
                                    )
                                else:
                                    still_invalid.append({'scenario': corrected, 'warning': validation['warning']})
                        
                        retry_needed = still_invalid
                        
                        logger.info(
                            f"[extract_constraints] Retry attempt {attempt + 1}: "
                            f"Corrected {corrected_count}, still invalid {len(still_invalid)}"
                        )
                        
                    except Exception as e:
                        logger.error(f"[extract_constraints] Correction attempt {attempt + 1} failed: {e}")
                        logger.error(f"[extract_constraints] Exception details: {type(e).__name__}: {str(e)}")
                        logger.error(f"🔥 [CORRECTION] Exception during correction: {e}")
                        print(f"🔥 [CORRECTION] Exception during correction: {e}")

                        # Enhanced debugging for entity validation errors
                        if "entities" in str(e) and "at least one entity" in str(e).lower():
                            logger.error(f"🔥 [CORRECTION] Schema validation failed - LLM returned empty entities")
                            logger.error(f"🔥 [CORRECTION] Original entities passed to LLM: {constraints_output.entities}")
                            logger.error(f"🔥 [CORRECTION] Original metrics passed to LLM: {constraints_output.metrics}")
                            logger.error(f"🔥 [CORRECTION] Check correction_prompt includes entities/metrics correctly")

                        break
            else:
                # Log when correction loop is NOT entered
                logger.error("🔥🔥🔥 [CORRECTION] ❌ NOT ENTERING CORRECTION LOOP!")
                logger.error(f"  Reason: retry_needed={bool(retry_needed)} ({len(retry_needed)} items), MAX_RETRY_ATTEMPTS={MAX_RETRY_ATTEMPTS}")
                print("🔥🔥🔥 [CORRECTION] ❌ NOT ENTERING CORRECTION LOOP!")
                print(f"  Reason: retry_needed={bool(retry_needed)} ({len(retry_needed)} items), MAX_RETRY_ATTEMPTS={MAX_RETRY_ATTEMPTS}")

            # Final validation logging
            final_invalid = []
            for scenario in constraints_output.scenarios:
                validation = validator.validate_scenario(scenario)
                if not validation['valid']:
                    final_invalid.append(scenario.name)
                    logger.error(
                        f"[extract_constraints] FINAL: Scenario '{scenario.name}' still has empty parameters "
                        f"after {MAX_RETRY_ATTEMPTS} retry attempts. Description: {scenario.description[:100]}"
                    )

            if final_invalid:
                logger.warning(
                    f"[extract_constraints] Parameter extraction incomplete for {len(final_invalid)} scenario(s): "
                    f"{final_invalid}. Downstream formula discovery may be limited."
                )

            # Convert ScenarioOutput (LLM format) to ScenarioDefinition (internal format)
            # This is CRITICAL: LLM returns ScenarioOutput with list parameters (Databricks-compatible)
            # We convert to ScenarioDefinition with dict parameters (ergonomic API)
            # NOTE: The model_validator in ConstraintsOutput may have already converted parameters to dict,
            # in which case scenarios are already ScenarioDefinition objects. Handle both cases.
            hydrated_scenarios = [
                s if isinstance(s, ScenarioDefinition) else ScenarioDefinition.from_scenario_output(s)
                for s in constraints_output.scenarios
            ]

            # Return ConstraintsOutput (Pydantic) with hydrated scenarios
            # Note: QueryConstraints is now an alias for ConstraintsOutput
            # We use model_copy to create new instance with updated scenarios
            return constraints_output.model_copy(update={'scenarios': hydrated_scenarios})

        except Exception as e:
            # Enhanced error logging with detailed validation errors
            print(f"❌ [_EXTRACT_WITH_LLM] Exception occurred: {type(e).__name__}: {e}")
            logger.error(f"❌ [_EXTRACT_WITH_LLM] Exception occurred: {type(e).__name__}: {e}")
            logger.error(f"❌ Structured generation failed: {e}")
            
            import traceback
            traceback.print_exc()

            # If it's a ValidationError from Pydantic, log the detailed errors
            if hasattr(e, '__class__') and 'ValidationError' in e.__class__.__name__:
                try:
                    if hasattr(e, 'errors'):
                        errors_detail = json_module.dumps(e.errors(), indent=2)
                        logger.error(f"📋 Detailed validation errors:\n{errors_detail}")
                        print(f"📋 Detailed validation errors:\n{errors_detail}")
                except Exception as log_err:
                    logger.debug(f"Could not format validation errors: {log_err}")

            # Use pattern-based fallback (doesn't need LLM)
            print("🔄 [_EXTRACT_WITH_LLM] Returning fallback constraints")
            logger.error("🔄 [_EXTRACT_WITH_LLM] Returning fallback constraints")
            return self._create_fallback_constraints(query)

    def _create_fallback_constraints(self, query: str) -> QueryConstraints:
        """Fallback pattern-based extraction when LLM fails.

        NOTE: This parses the ORIGINAL USER QUERY with regex, not a failed LLM response.
        LLM failure could be timeout/rate-limit, not that the query is unparseable.
        """
        logger.warning("Using fallback pattern-based constraint extraction")

        entities = []

        # Pattern: "X vs Y vs Z"
        vs_pattern = r'(\w+(?:\s+\w+)?)\s+vs\.?\s+(\w+(?:\s+\w+)?)'
        matches = re.findall(vs_pattern, query, re.IGNORECASE)
        for match in matches:
            entities.extend(match)

        # Pattern: "Compare X, Y, and Z"
        compare_pattern = r'compare\s+([\w\s,]+?)(?:\s+for|\s+across|\.)'
        match = re.search(compare_pattern, query, re.IGNORECASE)
        if match:
            entity_str = match.group(1)
            parts = re.split(r',\s*|\s+and\s+', entity_str)
            entities.extend([p.strip() for p in parts if p.strip()])

        # Deduplicate
        unique_entities = list(dict.fromkeys([e.strip().title() for e in entities if e.strip()]))[:10]

        # Return Pydantic ConstraintsOutput (QueryConstraints is alias)
        return ConstraintsOutput(
            entities=unique_entities if unique_entities else ["Unknown"],  # Pydantic validator requires at least one
            metrics=["information"],  # Generic fallback
            comparison_type=ComparisonType.ENTITY,  # Use enum, not string
            scenarios=[],
            topics=[],
            monetary_values=[],
            comparisons=[],
            data_format="text",
            specifics={},
            time_constraints=[]
        )

    def _extract_with_patterns(self, query: str) -> QueryConstraints:
        """Pattern-based fallback extraction for when LLM is unavailable."""

        # Extract countries/entities
        country_matches = self.entity_patterns["countries"].findall(query)
        entities = list(set(country_matches))

        # Extract monetary amounts - CRITICAL: Store in both metrics and monetary_values
        currency_matches = self.entity_patterns["currencies"].findall(query)
        metrics = list(currency_matches)
        monetary_values = currency_matches  # Store exact amounts

        # Extract percentages
        percent_matches = self.entity_patterns["percentages"].findall(query)
        metrics.extend(percent_matches)

        # Extract years
        year_matches = self.entity_patterns["years"].findall(query)
        time_constraints = list(set(year_matches))

        # Determine data format
        data_format = "text"
        comparisons = []
        if self.entity_patterns["comparisons"].search(query):
            data_format = "table"
            if len(entities) > 1:
                comparisons = [f"{entities[0]} vs {' vs '.join(entities[1:])}"]

        # Check for table keywords
        table_keywords = ["table", "chart", "comparison", "compare", "list", "breakdown"]
        if any(keyword in query.lower() for keyword in table_keywords):
            data_format = "table"

        # Extract scenarios (create structured ScenarioDefinition objects)
        import re
        scenarios = []
        scenario_match = re.search(r'(\d+)\s+scenarios?', query, re.IGNORECASE)
        if scenario_match:
            num_scenarios = int(scenario_match.group(1))
            # Create simple scenario definitions
            for i in range(num_scenarios):
                scenarios.append(ScenarioDefinition(
                    id=f"s{i+1}",
                    name=f"Scenario {i+1}",
                    description=f"Scenario {i+1} from pattern extraction",
                    parameters={}
                ))

        # Determine comparison type (use enum)
        if any(c.lower() in ["spain", "france", "germany", "uk"] for c in entities):
            comparison_type = ComparisonType.COUNTRY
        else:
            comparison_type = ComparisonType.ENTITY

        # Determine topics
        topics = []
        if "tax" in query.lower():
            topics.append("taxation")
        if "price" in query.lower() or "cost" in query.lower():
            topics.append("pricing")
        if "performance" in query.lower():
            topics.append("performance")

        # Return Pydantic ConstraintsOutput (QueryConstraints is alias)
        # Ensure entities list is not empty (Pydantic validator requires at least one)
        return ConstraintsOutput(
            entities=entities if entities else ["Unknown"],
            metrics=metrics if metrics else ["information"],
            scenarios=scenarios,
            comparison_type=comparison_type,
            topics=topics,
            monetary_values=monetary_values,
            comparisons=comparisons,
            data_format=data_format,
            specifics={},
            time_constraints=time_constraints
        )


class ConstraintEnforcer:
    """Enforce constraints throughout the research pipeline."""

    @staticmethod
    def validate_content(content: str, constraints: QueryConstraints) -> Tuple[bool, List[str]]:
        """
        Validate that content adheres to constraints.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        if not constraints.entities:
            return True, []  # No entity constraints to enforce

        content_lower = content.lower()

        # Build list of allowed entity variations
        allowed_patterns = set()
        for entity in constraints.entities:
            allowed_patterns.add(entity.lower())
            # Add common variations
            if entity.lower() == "united kingdom":
                allowed_patterns.update(["uk", "britain", "british", "england"])
            elif entity.lower() == "united states":
                allowed_patterns.update(["usa", "us", "america", "american"])
            # Add more variations as needed

        # Check for common forbidden entities (countries not in the constraint list)
        common_countries = [
            "belgium", "hungary", "denmark", "sweden", "norway", "netherlands",
            "italy", "portugal", "greece", "ireland", "finland", "austria",
            "czech republic", "romania", "croatia", "slovenia", "slovakia",
            "luxembourg", "malta", "cyprus", "estonia", "latvia", "lithuania"
        ]

        for country in common_countries:
            if country in content_lower and country not in allowed_patterns:
                # Check if it's a significant mention (not just passing reference)
                if content_lower.count(country) > 1 or f"{country} tax" in content_lower:
                    violations.append(f"Mentions forbidden entity: {country.title()}")

        # Check if required entities are present
        for entity in constraints.entities:
            if entity.lower() not in content_lower:
                violations.append(f"Missing required entity: {entity}")

        return len(violations) == 0, violations

    @staticmethod
    def filter_observations(observations: List[Any],
                          constraints: QueryConstraints) -> List[Any]:
        """
        Filter observations to only those matching constraints.

        This is a gentler filtering - we keep observations that are relevant
        but might not mention all entities.
        """
        if not constraints.entities:
            return observations

        filtered = []
        for obs in observations:
            obs_text = str(obs) if not isinstance(obs, str) else obs
            obs_lower = obs_text.lower()

            # Keep if mentions ANY of the constrained entities
            if any(entity.lower() in obs_lower for entity in constraints.entities):
                filtered.append(obs)
            # Also keep if mentions key metrics
            elif constraints.metrics and any(metric in obs_text for metric in constraints.metrics):
                filtered.append(obs)
            else:
                logger.debug(f"Filtered observation: no entity/metric match")

        return filtered

    @staticmethod
    def enhance_prompt_with_constraints(prompt: str, constraints: QueryConstraints) -> str:
        """
        Add constraint instructions to any prompt.

        This ensures LLMs respect the constraints at every step.
        """
        if not constraints or not (constraints.entities or constraints.metrics):
            return prompt

        constraint_block = f"""
{'='*60}
MANDATORY CONSTRAINTS FOR THIS RESPONSE:
{constraints.to_prompt_instructions()}
{'='*60}

"""
        return constraint_block + prompt


# Global constraint management
_global_constraints: Optional[QueryConstraints] = None

def set_global_constraints(constraints: QueryConstraints):
    """Set constraints for the current request."""
    global _global_constraints
    _global_constraints = constraints
    logger.info(f"Set global constraints: {len(constraints.entities)} entities, "
                f"{len(constraints.metrics)} metrics")

def get_global_constraints() -> Optional[QueryConstraints]:
    """Get current request constraints."""
    return _global_constraints

def clear_global_constraints():
    """Clear constraints after request completion."""
    global _global_constraints
    _global_constraints = None
    logger.info("Cleared global constraints")

def enforce_constraints_on_response(response: str) -> str:
    """
    Post-process a response to ensure constraint compliance.

    This is a last-resort filter to catch any constraint violations.
    """
    constraints = get_global_constraints()
    if not constraints:
        return response

    is_valid, violations = ConstraintEnforcer.validate_content(response, constraints)

    if not is_valid:
        logger.warning(f"Response has constraint violations: {violations}")
        # Could potentially filter or modify response here
        # For now, just log the warning

    return response