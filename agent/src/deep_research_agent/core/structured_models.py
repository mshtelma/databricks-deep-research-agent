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

    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v):
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

class ComparisonType(str, Enum):
    """Valid comparison types."""
    COUNTRY = "country"
    CLOUD_PROVIDER = "cloud_provider"
    PRODUCT = "product"
    COMPANY = "company"
    SERVICE = "service"
    ENTITY = "entity"

class ConstraintsOutput(StrictBaseModel):
    """Structured constraints extraction output."""
    entities: List[str] = Field(
        description="List of entities to compare (countries, companies, products)"
    )
    metrics: List[str] = Field(
        description="List of metrics to extract (tax_rate, price, performance)"
    )
    scenarios: List[ScenarioOutput] = Field(
        default_factory=list,
        description="Structured scenarios with parameters"
    )
    comparison_type: ComparisonType = Field(
        default=ComparisonType.ENTITY,
        description="Type of comparison being performed"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="General topics like taxation, pricing, performance"
    )
    monetary_values: List[str] = Field(
        default_factory=list,
        description="Raw monetary values found in query"
    )
    data_format: str = Field(
        default="text",
        description="Requested format: table, list, or text"
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
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
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
    """Metrics extracted for a specific entity."""
    entity: str = Field(
        description="Entity name these metrics belong to"
    )
    extracted_values: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Metric name to numeric value mapping"
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
