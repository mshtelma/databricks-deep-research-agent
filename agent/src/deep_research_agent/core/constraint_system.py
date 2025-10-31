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


class ConstraintExtractor:
    """Extract constraints from user queries using LLM or pattern matching."""

    def __init__(self, llm=None):
        self.llm = llm
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
        if not query:
            return QueryConstraints()

        # Try LLM extraction first for best results
        if self.llm:
            try:
                return self._extract_with_llm(query, state)
            except Exception as e:
                logger.warning(f"LLM constraint extraction failed: {e}, using fallback")

        # Fallback to pattern-based extraction
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
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = """You are a constraint extraction expert. Extract ALL constraints from user queries.

CRITICAL: Identify what the user wants to COMPARE, not technical details ABOUT the comparison.

ENTITIES - What is being compared:
✓ Countries: "Spain", "France", "Germany", "United Kingdom"
✓ Companies: "Apple", "Google", "Microsoft"
✓ Products: "iPhone 15", "Galaxy S24", "Pixel 8"
✓ Services: "Netflix", "Disney+", "Amazon Prime"
✗ NOT technical concepts: "At Vest", "Sale", "Ordinary Income", "Capital Gains"
✗ NOT descriptions: "Compare Spain", "France Tax", "Upper-middle-class"

METRICS - What to measure about each entity:
✓ Measurable outcomes: "tax_rate", "net_take_home", "effective_tax_rate", "price", "rent"
✓ Derived values: "disposable_income", "daycare_cost", "family_benefits"
✗ NOT generic: "information", "data", "details"
✗ NOT technical terms: "RSU_treatment", "tax_rules" (these are implementation details)

SCENARIOS - Specific cases with parameters:
✓ With financial amounts: "Single: €150,000 salary + €100,000 RSUs"
✓ With family details: "Married, 1 child"
✗ Empty if just mentioned as "3 scenarios" without details

CRITICAL: Scenario parameters should be NUMERIC values (salary, rsu, rent, etc.).
Descriptive attributes (marital_status, employment_type) should go in name/description, NOT parameters.

GOOD scenario:
  "name": "Married with 1 child",
  "description": "Married couple with one child, primary earner €150k salary",
  "parameters": [["salary", 150000], ["children", 1]]

BAD scenario:
  "parameters": [["marital_status", "married"], ["employment_type", "full-time"]]

REQUIRED OUTPUT FORMAT (JSON):
{
  "entities": ["Spain", "France", "Germany"],  // Clean names of things being compared
  "metrics": ["net_take_home", "effective_tax_rate", "rent"],  // Measurable values
  "scenarios": [
    {
      "id": "s1",
      "name": "Single",
      "description": "Single earner with €150k salary + €100k RSUs",
      "parameters": [["salary", 150000], ["rsu", 100000]]  // Numeric values only
    }
  ],
  "comparison_type": "country",  // Type of entities (country/product/company/service)
  "topics": ["taxation", "after-tax finances"],  // General themes
  "monetary_values": ["€150000", "€100000"],  // Amounts mentioned
  "data_format": "table"  // "table" if comparative table requested, else "text"
}

Convert shorthand: "€150k" → "€150000", "$100K" → "$100000"
"""

        human_prompt = f"""Extract all constraints from this query:
"{query}"

Be comprehensive - extract ALL entities, metrics, and scenarios mentioned.
Respond with valid JSON matching the schema above."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        try:
            # CRITICAL FIX: Use json_schema method for strict enforcement
            # json_schema: Full schema validation with Databricks response_format
            # json_mode: Generic JSON object (less strict, can cause validation errors)
            response = self.llm.with_structured_output(
                schema=ConstraintsOutput,
                method="json_schema",  # CHANGED: Strict schema enforcement (was json_mode)
                include_raw=False
            ).invoke(messages)

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
            logger.error(f"❌ Structured generation failed: {e}")

            # If it's a ValidationError from Pydantic, log the detailed errors
            if hasattr(e, '__class__') and 'ValidationError' in e.__class__.__name__:
                try:
                    import json
                    if hasattr(e, 'errors'):
                        errors_detail = json.dumps(e.errors(), indent=2)
                        logger.error(f"📋 Detailed validation errors:\n{errors_detail}")
                except Exception as log_err:
                    logger.debug(f"Could not format validation errors: {log_err}")

            # Use pattern-based fallback (doesn't need LLM)
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