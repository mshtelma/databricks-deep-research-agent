"""
Pydantic models for structured LLM outputs.

These models ensure type safety and validation for all LLM responses,
eliminating manual JSON parsing and reducing error-prone code.

CRITICAL NOTE on JSON Repair:
- Rate-limited models (ChatDatabricks) with with_structured_output() already use json_repair internally
- If structured generation fails, additional repair attempts won't help
- Best strategy: Use safe fallbacks (pattern-based or empty) when structured generation fails
- DO NOT try to repair what already failed internal repair mechanisms
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional
from enum import Enum
import json

# Base model WITHOUT extra='forbid' to avoid additionalProperties in schema
# Databricks rejects schemas with "additionalProperties" keyword completely
class StrictBaseModel(BaseModel):
    """Base model for Databricks compatibility - no additionalProperties in schema."""

    @classmethod
    def model_json_schema(cls, **kwargs) -> Dict[str, Any]:
        """Generate JSON schema without additionalProperties keyword.

        Databricks requires schemas with no 'additionalProperties' keyword at all.
        """
        schema = super().model_json_schema(**kwargs)
        # Recursively remove all additionalProperties from schema
        _remove_additional_properties(schema)
        return schema

def _remove_additional_properties(obj):
    """Recursively remove problematic keywords from schema dict.

    Databricks rejects schemas with:
    - additionalProperties (any value)
    - minItems/maxItems on array types
    - prefixItems (tuple validation)
    """
    if isinstance(obj, dict):
        # Remove problematic keys
        obj.pop('additionalProperties', None)
        obj.pop('minItems', None)
        obj.pop('maxItems', None)
        obj.pop('prefixItems', None)  # Tuple validation not supported
        # Recurse into nested dicts
        for value in obj.values():
            _remove_additional_properties(value)
    elif isinstance(obj, list):
        # Recurse into lists
        for item in obj:
            _remove_additional_properties(item)

# ============================================================================
# Phase 1: Constraint Extraction Models
# ============================================================================

class ScenarioOutput(StrictBaseModel):
    """Structured scenario from LLM."""
    id: str = Field(description="Unique identifier like s1, s2, etc.")
    name: str = Field(description="Human readable name like 'Single earner'")
    description: str = Field(default="", description="Full scenario description")
    # Use List of tuples instead of Dict to avoid additionalProperties in JSON schema
    # Databricks rejects schemas with additionalProperties keyword
    parameters: List[tuple[str, Any]] = Field(
        default_factory=list,
        description="List of [parameter_name, value] pairs. Values can be numeric (salary=150000), string (status='married'), or boolean. Prefer numeric values when possible."
    )

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Scenario ID cannot be empty")
        return v.strip()

    @field_validator('parameters', mode='before')
    @classmethod
    def validate_parameters(cls, v):
        # Handle empty string (LLM sometimes returns "" instead of [])
        if isinstance(v, str) and v == "":
            return []

        # Validate parameter structure and try to convert numeric strings to numbers
        validated = []
        for item in v:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Each parameter must be a [name, value] pair, got {item}")
            key, value = item

            # Try to convert numeric strings to float, but keep non-numeric values as-is
            if isinstance(value, str):
                # Try numeric conversion
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    # Keep as string if not numeric
                    pass

            validated.append((str(key), value))
        return validated

    def get_parameters_dict(self) -> Dict[str, Any]:
        """Convert parameter list to dict for easy access."""
        return dict(self.parameters)


class ScenarioDefinition(StrictBaseModel):
    """
    Internal scenario representation with dict parameters for ergonomic API.

    This is converted from ScenarioOutput (LLM output format) which uses list of tuples
    to avoid Databricks additionalProperties issues. ScenarioDefinition uses dict
    for easy access patterns throughout the codebase.
    """
    id: str = Field(description="Unique scenario identifier (s1, s2, etc.)")
    name: str = Field(description="Human-readable scenario name")
    description: str = Field(default="", description="Full scenario description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scenario-specific parameters (salary, RSU, etc.) as dict"
    )

    @field_validator('parameters', mode='before')
    @classmethod
    def validate_parameters(cls, v):
        # Handle empty string (LLM sometimes returns "" instead of {})
        if isinstance(v, str) and v == "":
            return {}
        return v

    @classmethod
    def from_scenario_output(cls, output: 'ScenarioOutput') -> 'ScenarioDefinition':
        """Create from Pydantic ScenarioOutput model (LLM output format)."""
        return cls(
            id=output.id,
            name=output.name,
            description=output.description,
            parameters=output.get_parameters_dict()  # Convert list→dict
        )

    def get_unique_key(self) -> str:
        """Generate unique key for this scenario."""
        return f"{self.id}_{self.name.replace(' ', '_').lower()}"

    def get_param_value(self, param_name: str, default: float = 0.0) -> float:
        """Get parameter value with default."""
        return self.parameters.get(param_name, default)


class ComparisonType(str, Enum):
    """Valid comparison types."""
    COUNTRY = "country"
    CLOUD_PROVIDER = "cloud_provider"
    PRODUCT = "product"
    COMPANY = "company"
    SERVICE = "service"
    ENTITY = "entity"

class ConstraintsOutput(StrictBaseModel):
    """
    Structured constraints extraction output - single source of truth for query constraints.

    This replaces the old QueryConstraints dataclass with a type-safe Pydantic model.
    Enhanced with structured scenarios and metadata helpers for observation enrichment.

    Designed to be flexible and work with any domain:
    - Financial queries (salaries, taxes, investments)
    - Geographic queries (countries, cities, regions)
    - Product comparisons (features, prices, specs)
    - Time-based queries (years, quarters, trends)
    """
    # Core fields
    entities: List[str] = Field(
        description="List of entities to compare (countries, companies, products, people)"
    )
    metrics: List[str] = Field(
        description="List of metrics to extract (tax_rate, price, performance, numbers, rates, percentages)"
    )
    scenarios: List['ScenarioDefinition'] = Field(
        default_factory=list,
        description="Structured scenarios with dict parameters (uses ScenarioDefinition, not ScenarioOutput)"
    )

    @model_validator(mode='before')
    @classmethod
    def convert_scenario_parameters(cls, data: Any) -> Any:
        """Convert scenario parameters from list format to dict format.

        LLM generates: {"parameters": [["salary", 150000], ["rsu", 100000]]}
        We need: {"parameters": {"salary": 150000, "rsu": 100000}}

        This handles the mismatch between ScenarioOutput (LLM format with list)
        and ScenarioDefinition (internal format with dict).
        """
        if isinstance(data, dict) and 'scenarios' in data:
            scenarios = data['scenarios']
            if isinstance(scenarios, list):
                converted_scenarios = []
                for scenario in scenarios:
                    if isinstance(scenario, dict) and 'parameters' in scenario:
                        params = scenario['parameters']
                        # Convert list of lists/tuples to dict
                        if isinstance(params, list) and params:
                            if isinstance(params[0], (list, tuple)) and len(params[0]) >= 2:
                                # Convert [["key", value], ...] to {"key": value, ...}
                                scenario['parameters'] = {str(k): v for k, v in params}
                    converted_scenarios.append(scenario)
                data['scenarios'] = converted_scenarios
        return data

    # Enhanced fields for metadata enrichment
    comparison_type: ComparisonType = Field(
        default=ComparisonType.ENTITY,
        description="Type of comparison being performed (country, cloud_provider, product, company)"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="General topics like taxation, pricing, performance"
    )
    monetary_values: List[str] = Field(
        default_factory=list,
        description="Specific monetary amounts mentioned in query"
    )

    # Additional fields from QueryConstraints dataclass
    comparisons: List[str] = Field(
        default_factory=list,
        description="Things being compared (legacy/compatibility field)"
    )
    data_format: str = Field(
        default="text",
        description="Requested format: table, list, comparison, or text"
    )
    specifics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific values/requirements from query"
    )
    time_constraints: List[str] = Field(
        default_factory=list,
        description="Years, dates, periods mentioned"
    )

    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v):
        if not v:
            raise ValueError("At least one entity required")
        # Clean and deduplicate
        return list(dict.fromkeys([e.strip() for e in v if e.strip()]))

    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v):
        # Clean and deduplicate
        return list(dict.fromkeys([m.strip() for m in v if m.strip()]))

    # ========================================================================
    # Helper Methods (migrated from QueryConstraints dataclass)
    # ========================================================================

    def has_scenarios(self) -> bool:
        """Check if query has scenarios."""
        return bool(self.scenarios)

    def get_entity_tags(self, entity: str) -> Dict[str, Any]:
        """
        Get metadata tags for an entity.

        This is CRITICAL for populating observation metadata!
        Used in Phase 2 to enrich observations with tags.
        """
        tags = {
            'entity': entity,
            'entity_type': str(self.comparison_type.value) if isinstance(self.comparison_type, ComparisonType) else self.comparison_type,
        }

        if self.topics:
            tags['topics'] = ','.join(self.topics)

        # Add entity-specific metadata based on comparison_type
        comp_type = str(self.comparison_type.value) if isinstance(self.comparison_type, ComparisonType) else self.comparison_type
        if comp_type == 'country':
            tags['entity_category'] = 'geographic'
        elif comp_type == 'cloud_provider':
            tags['entity_category'] = 'technology'
        elif comp_type == 'company':
            tags['entity_category'] = 'business'
        elif comp_type == 'product':
            tags['entity_category'] = 'product'

        return tags

    def matches_entity(self, text: str, entity: str) -> bool:
        """
        Check if text mentions the entity.

        Used for filtering observations by entity.
        Handles common variations and abbreviations.
        """
        if not text or not entity:
            return False

        # Case-insensitive matching
        text_lower = text.lower()
        entity_lower = entity.lower()

        # Direct match
        if entity_lower in text_lower:
            return True

        # Handle common variations (e.g., "U.S." for "United States")
        entity_parts = entity_lower.split()
        if len(entity_parts) > 1:
            # Check abbreviation (first letters)
            abbrev = ''.join(p[0] for p in entity_parts)
            if abbrev in text_lower or f"{abbrev}." in text_lower:
                return True

            # Check if all parts present
            if all(part in text_lower for part in entity_parts):
                return True

        return False

    def extract_metric_value(self, text: str, metric: str) -> Optional[float]:
        """
        Extract numeric value for a metric from text.

        Returns None if not found.
        Used for extracting structured data from observations.
        """
        import re

        # Common patterns for metrics
        patterns = {
            'tax_rate': r'tax.*?(\d+(?:\.\d+)?)\s*%',
            'price': r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'number': r'(\d+(?:,\d{3})*(?:\.\d+)?)',
        }

        # Try specific pattern for metric
        pattern = patterns.get(metric, patterns['number'])
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            try:
                # Remove commas and convert to float
                value_str = match.group(1).replace(',', '')
                return float(value_str)
            except (ValueError, TypeError):
                pass

        return None

    def get_scenario_by_id(self, scenario_id: str) -> Optional['ScenarioDefinition']:
        """Get scenario by ID."""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None

    def validate(self) -> List[str]:
        """Validate constraints and return any issues."""
        issues = []

        if not self.entities:
            issues.append("No entities specified")

        if not self.metrics:
            issues.append("No metrics specified")

        # Check for duplicate scenario IDs
        if self.scenarios:
            seen_ids = set()
            for scenario in self.scenarios:
                if scenario.id in seen_ids:
                    issues.append(f"Duplicate scenario ID: {scenario.id}")
                seen_ids.add(scenario.id)

        return issues

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


# ============================================================================
# Phase 2: Fact Extraction Models
# ============================================================================

class FactWithMetadata(StrictBaseModel):
    """Structured fact extraction output."""
    content: str = Field(
        description="Complete fact statement, 20-200 characters",
        alias="statement"  # Accept both 'content' and 'statement' from LLM
    )
    entity: str = Field(
        default="",  # Allow empty entity (for facts not specific to an entity)
        description="Entity this fact relates to (must be from entity list)"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Numeric metric values found in this fact"
    )
    # ✅ BUG #4 FIX: Removed ge=/le= (Databricks doesn't support minimum/maximum in JSON schema)
    # Runtime validation added via @field_validator below
    confidence: float = Field(
        default=0.9,
        description="Confidence score: 0.95 for direct facts, 0.85 for clear, 0.70 for inferred"
    )

    model_config = {
        "populate_by_name": True  # Allow populating by field name OR alias
    }

    @model_validator(mode='before')
    @classmethod
    def remap_field_names(cls, data: Any) -> Any:
        """Remap 'fact' to 'content' to handle LLM variations."""
        if isinstance(data, dict):
            # Handle 'fact' as alternative to 'content' or 'statement'
            if 'fact' in data and 'content' not in data and 'statement' not in data:
                data['content'] = data.pop('fact')
        return data

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if len(v) < 20:
            raise ValueError("Fact content too short (min 20 chars)")
        if len(v) > 200:
            raise ValueError("Fact content too long (max 200 chars)")
        return v

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v):
        """Clamp confidence to [0.0, 1.0] range (Databricks JSON schema fix)."""
        # Silently clamp to valid range (LLM should respect hints, but be defensive)
        return max(0.0, min(1.0, v))

    @field_validator('metrics')
    @classmethod
    def clean_metric_values(cls, v):
        """Ensure all metric values are numeric."""
        cleaned = {}
        for key, value in v.items():
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
            elif isinstance(value, str):
                # Try to parse strings like "35%", "€50,000"
                try:
                    # Remove common symbols
                    clean_val = value.replace(',', '').replace('€', '').replace('$', '').replace('%', '').strip()
                    if clean_val:
                        cleaned[key] = float(clean_val)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    pass
        return cleaned

class FactExtractionOutput(StrictBaseModel):
    """Container for extracted facts."""
    facts: List[FactWithMetadata] = Field(
        description="List of extracted facts with metadata"
    )

    @field_validator('facts')
    @classmethod
    def validate_facts(cls, v):
        # Remove duplicates based on content
        seen = set()
        unique_facts = []
        for fact in v:
            if fact.content not in seen:
                seen.add(fact.content)
                unique_facts.append(fact)
        return unique_facts

# ============================================================================
# Phase 3: Pattern Extraction Models
# ============================================================================

class CalculationFormula(StrictBaseModel):
    """Structured calculation formula."""
    metric: str = Field(
        description="Name of the calculated metric"
    )
    description: str = Field(
        default="",
        description="What this calculation computes"
    )
    formula: str = Field(
        description="Formula expression using input variables"
    )
    inputs: List[str] = Field(
        default_factory=list,
        description="List of input metrics/parameters required"
    )
    per_scenario: bool = Field(
        default=False,
        description="Whether this varies by scenario"
    )

    @field_validator('formula')
    @classmethod
    def validate_formula(cls, v):
        # Check for dangerous operations
        dangerous = ['__', 'import', 'exec', 'eval', 'open', 'file', 'os.', 'sys.']
        for d in dangerous:
            if d in v.lower():
                raise ValueError(f"Formula contains dangerous operation: {d}")
        return v

class PatternExtractionOutput(StrictBaseModel):
    """Patterns extracted from observations."""
    extractable_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics that can be directly extracted from text"
    )
    calculation_formulas: List[CalculationFormula] = Field(
        default_factory=list,
        description="Formulas for computed metrics"
    )
    extraction_hints: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Synonyms and variations for each metric"
    )

    @field_validator('extractable_metrics')
    @classmethod
    def clean_metrics(cls, v):
        return list(dict.fromkeys([m.strip() for m in v if m.strip()]))

# ============================================================================
# Phase 3: Value Extraction Models
# ============================================================================

class EntityMetricsOutput(StrictBaseModel):
    """Metrics extracted for a specific entity.

    CRITICAL: extracted_values is REQUIRED to prevent silent failures.
    If LLM doesn't return metrics, we want a ValidationError, not an empty dict.
    This ensures parsing failures surface immediately rather than silently succeeding.
    """
    entity: str = Field(
        description="Entity name these metrics belong to"
    )
    extracted_values: Dict[str, Optional[float]] = Field(
        ...,  # ✅ REQUIRED: No default - force validation to fail if missing
        description="Metric name to numeric value mapping. MUST be present even if empty {}."
    )
    confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence score for each metric"
    )
    observation_sources: Dict[str, int] = Field(
        default_factory=dict,
        description="Maps metric name to observation index (0-based) where value was found"
    )

    @model_validator(mode='before')
    @classmethod
    def remap_field_names(cls, data: Any) -> Any:
        """Remap capitalized field names and handle flat metric format from LLM."""
        if isinstance(data, dict):
            # Handle Entity -> entity
            if 'Entity' in data and 'entity' not in data:
                data['entity'] = data.pop('Entity')
            # Handle Metrics -> extracted_values
            if 'Metrics' in data and 'extracted_values' not in data:
                data['extracted_values'] = data.pop('Metrics')
            # Handle Confidence -> confidence (if provided)
            if 'Confidence' in data and 'confidence' not in data:
                data['confidence'] = data.pop('Confidence')

            # Handle flat format: LLM returned metrics directly without wrapper
            # e.g., {"tax_rate": 45, "social_security": 6.35}
            # Convert to: {"entity": "", "extracted_values": {"tax_rate": 45, ...}}
            if 'entity' not in data and 'extracted_values' not in data:
                # All keys except known fields are metrics
                known_fields = {'entity', 'extracted_values', 'confidence', 'Entity', 'Metrics', 'Confidence'}
                metric_keys = [k for k in data.keys() if k not in known_fields]

                if metric_keys:
                    # This is a flat metrics dict, restructure it
                    extracted_values = {k: data.pop(k) for k in metric_keys}
                    data['extracted_values'] = extracted_values
                    # Entity will be set from context in the caller
                    if 'entity' not in data:
                        data['entity'] = ""  # Placeholder

        return data

    @field_validator('extracted_values')
    @classmethod
    def clean_values(cls, v):
        """Ensure values are numeric or None."""
        cleaned = {}
        for key, value in v.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (int, float)):
                cleaned[key] = float(value)
            else:
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    cleaned[key] = None
        return cleaned

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v, info):
        """Ensure confidence scores are in valid range."""
        if 'extracted_values' in info.data:
            for key in info.data['extracted_values']:
                if key not in v:
                    v[key] = 0.9  # Default confidence
                elif not 0.0 <= v[key] <= 1.0:
                    v[key] = max(0.0, min(1.0, v[key]))
        return v

    @model_validator(mode='after')
    def validate_has_metrics(self) -> 'EntityMetricsOutput':
        """
        Validate that at least one metric was extracted.

        This catches cases where:
        1. LLM returned {"extracted_values": {}} (empty but present)
        2. All values were None and filtered out by clean_values()
        3. Parsing succeeded but with no useful data

        We log a warning instead of raising to allow graceful degradation,
        but the warning will surface in logs for debugging.
        """
        if not self.extracted_values or len(self.extracted_values) == 0:
            # This is suspicious - entity processed but NO metrics extracted
            from . import get_logger
            logger = get_logger(__name__)

            logger.error(
                f"🚨 EntityMetricsOutput VALIDATION WARNING for entity='{self.entity}'\n"
                f"   - extracted_values is EMPTY (0 metrics)\n"
                f"   - This indicates LLM did not extract any values\n"
                f"   - OR all extracted values were invalid (None/non-numeric)\n"
                f"   - This entity will have NO metric_specs!\n"
                f"   - Check:\n"
                f"      1. LLM prompt clarity\n"
                f"      2. Observation quality for this entity\n"
                f"      3. Schema compliance in LLM response\n"
                f"   - Response details:\n"
                f"      - entity: {self.entity}\n"
                f"      - confidence: {self.confidence}\n"
                f"      - observation_sources: {self.observation_sources}"
            )

            # Graceful degradation - log warning but allow
            # This allows partial results (e.g., 6 of 7 entities succeed)
            logger.warning(
                f"⚠️  Allowing EntityMetricsOutput with 0 metrics for '{self.entity}' "
                "(graceful degradation - this entity will have no metric_specs)"
            )

        else:
            # Success - at least one metric extracted
            from . import get_logger
            logger = get_logger(__name__)
            logger.debug(
                f"✅ EntityMetricsOutput validation passed for '{self.entity}': "
                f"{len(self.extracted_values)} metrics extracted"
            )

        return self

# ============================================================================
# Utility Functions for Structured Generation
# ============================================================================

# NOTE: Rate-limited models (ChatDatabricks) with with_structured_output() already:
# 1. Uses native response_format parameter
# 2. Applies JSON schema validation
# 3. Uses json_repair internally for error recovery
# 4. Retries with backoff
#
# Therefore: NO additional repair needed - if it fails, it fails
# Just use safe fallbacks (pattern-based or empty results)
