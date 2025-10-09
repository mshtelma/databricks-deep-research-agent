"""
Abstract Constraint Extraction and Enforcement System

This module provides a flexible, domain-agnostic system for extracting and enforcing
constraints from user queries. It handles entities (countries, companies, products),
metrics (numbers, rates, percentages), comparisons, and data format requirements.

The system is designed to work with ANY type of query, not just specific use cases.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging
import re
import json
from .structured_output import extract_json

logger = logging.getLogger(__name__)


@dataclass
class QueryConstraints:
    """
    Abstract representation of all constraints extracted from a user query.

    This is designed to be flexible and work with any domain:
    - Financial queries (salaries, taxes, investments)
    - Geographic queries (countries, cities, regions)
    - Product comparisons (features, prices, specs)
    - Time-based queries (years, quarters, trends)
    - Any other domain-specific constraints
    """
    entities: List[str] = field(default_factory=list)          # Countries, companies, products, people
    metrics: List[str] = field(default_factory=list)           # Numbers, rates, percentages, amounts
    comparisons: List[str] = field(default_factory=list)       # Things being compared
    data_format: str = "text"                                  # "table", "list", "comparison", "text"
    specifics: Dict[str, Any] = field(default_factory=dict)    # Specific values/requirements
    time_constraints: List[str] = field(default_factory=list)  # Years, dates, periods
    scenarios: List[str] = field(default_factory=list)        # Scenarios requested (e.g., "3 scenarios")
    monetary_values: List[str] = field(default_factory=list)   # Specific monetary amounts mentioned

    def to_prompt_instructions(self) -> str:
        """Convert constraints to clear, enforceable LLM instructions."""
        instructions = []

        # Entity constraints - MOST CRITICAL
        if self.entities:
            entities_str = ", ".join(self.entities)
            instructions.append(f"🔴 CRITICAL ENTITY CONSTRAINT:")
            instructions.append(f"   - ONLY include information about: {entities_str}")
            instructions.append(f"   - DO NOT mention ANY other entities besides: {entities_str}")
            instructions.append(f"   - If data for any of these entities is missing, explicitly say so")
            instructions.append("")

        # Metric constraints
        if self.metrics:
            instructions.append(f"📊 METRIC REQUIREMENTS:")
            instructions.append(f"   - Focus on these specific values: {', '.join(self.metrics)}")
            instructions.append(f"   - Use exact values when mentioned (not averages/medians)")
            instructions.append("")

        # Specific requirements
        if self.specifics:
            instructions.append(f"🎯 SPECIFIC REQUIREMENTS:")
            for key, value in self.specifics.items():
                instructions.append(f"   - {key}: {value}")
            instructions.append("")

        # Monetary values - CRITICAL for accuracy
        if self.monetary_values:
            instructions.append(f"💰 MONETARY VALUES:")
            instructions.append(f"   - Use EXACTLY these values: {', '.join(self.monetary_values)}")
            instructions.append(f"   - DO NOT use average, median, or example values")
            instructions.append("")

        # Scenarios
        if self.scenarios:
            instructions.append(f"📊 SCENARIOS REQUIRED:")
            for scenario in self.scenarios:
                instructions.append(f"   - {scenario}")
            instructions.append("")

        # Comparison requirements
        if self.comparisons:
            instructions.append(f"⚖️ COMPARISON REQUIREMENTS:")
            instructions.append(f"   - Compare/contrast: {', '.join(self.comparisons)}")
            instructions.append("")

        # Time constraints
        if self.time_constraints:
            instructions.append(f"🕐 TIME CONSTRAINTS:")
            instructions.append(f"   - Focus on: {', '.join(self.time_constraints)}")
            instructions.append("")

        # Data format requirements
        if self.data_format == "table":
            instructions.append(f"📋 FORMAT: Present data in a clear comparison table")

        return "\n".join(instructions)

    def to_validation_rules(self) -> Dict[str, Any]:
        """Convert constraints to validation rules for content checking."""
        return {
            "allowed_entities": self.entities,
            "required_metrics": self.metrics,
            "required_comparisons": self.comparisons,
            "forbidden_entities": [],  # Will be populated dynamically
            "strict_mode": len(self.entities) > 0  # Strict if entities specified
        }


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
        """Extract constraints using LLM for maximum flexibility."""
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = """You are a constraint extraction expert. Extract ALL constraints from the query.

Return a JSON object with:
- entities: List of specific entities mentioned (countries, companies, products, people, etc.)
- metrics: List of specific numbers, amounts, rates mentioned
- comparisons: List of things being compared
- data_format: "table" if table/comparison requested, "list" if list requested, else "text"
- specifics: Object with any specific key-value requirements
- time_constraints: List of years, dates, or time periods mentioned
- scenarios: List of scenarios mentioned (e.g., ["3 scenarios"], ["multiple cases"], ["different situations"])
- monetary_values: List of EXACT monetary amounts mentioned (e.g., ["€210,000"], ["$100K"])

BE VERY PRECISE - only extract what is EXPLICITLY mentioned.
CRITICAL: For monetary_values, extract the EXACT amount mentioned, not generic terms.

Examples:
"I would like a table showing the tax comparison. 7 countries: Spain, France, Germany, Poland, Bulgaria, UK, Switzerland for 3 scenarios for a salary of €210,000"
→ {
    "entities": ["Spain", "France", "Germany", "Poland", "Bulgaria", "UK", "Switzerland"],
    "metrics": ["tax", "€210,000"],
    "comparisons": ["tax comparison across 7 countries"],
    "data_format": "table",
    "specifics": {"salary": "€210,000", "metric_type": "tax", "num_countries": 7, "num_scenarios": 3},
    "time_constraints": [],
    "scenarios": ["3 scenarios"],
    "monetary_values": ["€210,000"]
}

"How does Apple stock performance compare to Microsoft in Q3?"
→ {
    "entities": ["Apple", "Microsoft"],
    "metrics": ["stock performance"],
    "comparisons": ["Apple vs Microsoft stock"],
    "data_format": "text",
    "specifics": {"metric_type": "stock_performance", "period": "Q3"},
    "time_constraints": ["Q3"]
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract constraints from: {query}")
        ]

        response = self.llm.invoke(messages)

        # Parse response using unified parser
        try:
            # ✅ NEW: Use unified JSON extraction (replaces duplicated markdown/JSON parsing)
            data = extract_json(response, repair=True)

            if data is None:
                raise ValueError("LLM did not return valid JSON for constraint extraction")

            return QueryConstraints(
                entities=data.get("entities", []),
                metrics=data.get("metrics", []),
                comparisons=data.get("comparisons", []),
                data_format=data.get("data_format", "text"),
                specifics=data.get("specifics", {}),
                time_constraints=data.get("time_constraints", []),
                scenarios=data.get("scenarios", []),
                monetary_values=data.get("monetary_values", [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise

    def _extract_with_patterns(self, query: str) -> QueryConstraints:
        """Pattern-based fallback extraction for when LLM is unavailable."""
        constraints = QueryConstraints()

        # Extract countries/entities
        country_matches = self.entity_patterns["countries"].findall(query)
        constraints.entities = list(set(country_matches))

        # Extract monetary amounts - CRITICAL: Store in both metrics and monetary_values
        currency_matches = self.entity_patterns["currencies"].findall(query)
        constraints.metrics.extend(currency_matches)
        constraints.monetary_values = currency_matches  # Store exact amounts

        # Extract percentages
        percent_matches = self.entity_patterns["percentages"].findall(query)
        constraints.metrics.extend(percent_matches)

        # Extract years
        year_matches = self.entity_patterns["years"].findall(query)
        constraints.time_constraints = list(set(year_matches))

        # Check if comparison is requested
        if self.entity_patterns["comparisons"].search(query):
            constraints.data_format = "table"
            if len(constraints.entities) > 1:
                constraints.comparisons = [f"{constraints.entities[0]} vs {' vs '.join(constraints.entities[1:])}"]

        # Check for table keywords
        table_keywords = ["table", "chart", "comparison", "compare", "list", "breakdown"]
        if any(keyword in query.lower() for keyword in table_keywords):
            constraints.data_format = "table"

        # Extract scenarios
        import re
        scenario_match = re.search(r'(\d+)\s+scenarios?', query, re.IGNORECASE)
        if scenario_match:
            constraints.scenarios = [f"{scenario_match.group(1)} scenarios"]
            constraints.specifics["num_scenarios"] = int(scenario_match.group(1))

        # Store salary in specifics if found
        for monetary in constraints.monetary_values:
            if "salary" in query.lower():
                constraints.specifics["salary"] = monetary
                break

        return constraints


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