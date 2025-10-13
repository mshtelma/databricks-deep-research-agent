"""
Pydantic models for hybrid multi-pass report generation.

These models support the pre-calculation phase where data points and calculations
are extracted before narrative generation begins.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Union, Optional


class DataPoint(BaseModel):
    """
    A single extracted data point from research observations.

    Represents a quantitative or qualitative fact that can be used
    in calculations or comparisons.
    """
    entity: str = Field(
        ...,
        description="Abstract entity this data describes (e.g., 'Country A', 'Product X')"
    )
    metric: str = Field(
        ...,
        description="What attribute is measured (e.g., 'tax rate', 'market share')"
    )
    value: Union[float, str] = Field(
        ...,
        description="Numeric or categorical value"
    )
    unit: str = Field(
        default="unitless",
        description="Unit of measurement (e.g., 'USD', 'percent', 'count')"
    )
    source_observation_id: Optional[str] = Field(
        default=None,
        description="Traceability to source observation for citation"
    )
    confidence: float = Field(
        default=0.8,
        description="Confidence level in this value based on source quality (0.0-1.0)"
    )

    class Config:
        extra = "forbid"

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class Calculation(BaseModel):
    """
    A derived calculation from extracted data points.

    Makes calculations explicit and traceable, preventing hallucinated numbers.
    """
    description: str = Field(
        ...,
        description="Human-readable description of what is being calculated"
    )
    formula: str = Field(
        ...,
        description="Mathematical formula with source values (e.g., '$23,000 × 0.35')"
    )
    inputs: Dict[str, Union[float, str]] = Field(
        ...,
        description="Named input values used in calculation"
    )
    result: Union[float, str] = Field(
        ...,
        description="Calculated result"
    )
    unit: str = Field(
        default="unitless",
        description="Unit of measurement for the result"
    )

    class Config:
        extra = "forbid"


class ComparisonEntry(BaseModel):
    """
    A single row in a comparison table.

    Represents one entity with multiple metrics for tabular display.
    """
    primary_key: str = Field(
        ...,
        description="Primary identifier for this entry (e.g., entity name)"
    )
    metrics: Dict[str, Union[float, str, None]] = Field(
        ...,
        description="Dictionary of metric_name -> value pairs (None allowed for missing data)"
    )
    source_observation_ids: List[str] = Field(
        default_factory=list,
        description="List of observation IDs supporting these metrics"
    )

    class Config:
        extra = "forbid"


class TableSpec(BaseModel):
    """
    Specification for a table determined from understanding user intent.

    This model preserves the LLM's understanding of what structure a table
    should have, based on the user's research question and the data available.
    """
    table_id: str = Field(
        ...,
        description="Descriptive identifier for this table (e.g., 'country_comparison')"
    )
    purpose: str = Field(
        ...,
        description="What this table is meant to show (e.g., 'Compare after-tax income across countries')"
    )
    row_entities: List[str] = Field(
        default_factory=list,
        description="List of entities that should appear as rows (e.g., ['Spain', 'France', 'UK'])"
    )
    column_metrics: List[str] = Field(
        default_factory=list,
        description="List of metrics that should appear as columns (e.g., ['net_income', 'tax_rate'])"
    )

    class Config:
        extra = "forbid"


class Phase1AUnderstanding(BaseModel):
    """
    Dual-output structured result from Phase 1A.

    Combines narrative understanding with table specifications
    in a single structured response. This ensures reliable extraction
    without JSON parsing errors.
    """
    narrative_understanding: str = Field(
        ...,
        description="Deep analysis of structure, relationships, and data patterns in the research"
    )
    table_specifications: List[TableSpec] = Field(
        default_factory=list,
        description="Extracted table structure specifications based on user intent and research data"
    )

    class Config:
        extra = "forbid"


class CalculationContext(BaseModel):
    """
    Complete context from pre-calculation phase.

    This is passed to subsequent phases to inform narrative generation
    and table rendering with grounded, traceable data.
    """
    extracted_data: List[DataPoint] = Field(
        default_factory=list,
        description="All data points extracted from observations"
    )
    calculations: List[Calculation] = Field(
        default_factory=list,
        description="All derived calculations performed"
    )
    key_comparisons: List[ComparisonEntry] = Field(
        default_factory=list,
        description="Comparison entries suitable for table generation"
    )
    summary_insights: List[str] = Field(
        default_factory=list,
        description="High-level insights derived from the data"
    )
    data_quality_notes: List[str] = Field(
        default_factory=list,
        description="Warnings or notes about data quality issues"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (processing time, observation count, etc.)"
    )
    table_specifications: List[TableSpec] = Field(
        default_factory=list,
        description="Table structure specifications determined from Phase 1A understanding"
    )
    structural_understanding: str = Field(
        default="",
        description="Deep understanding text from Phase 1A about user intent and optimal structure"
    )

    class Config:
        extra = "forbid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalculationContext:
        """Create from dictionary."""
        return cls(**data)


# ===== PHASE 1: Lightweight Metadata Helpers (Added per revised plan) =====
# These helpers work with existing Table/ContentBlock models instead of duplicating them


class TableRenderMetadata(BaseModel):
    """
    Metadata needed to materialize a Table from structured comparisons.

    This is a lightweight helper that works with existing Table models,
    not a replacement for them.
    """
    source_spec_id: str = Field(
        ...,
        description="ID of the TableSpec this table was built from"
    )
    title: str = Field(
        ...,
        description="Table title/purpose"
    )
    alignments: Optional[List[str]] = Field(
        default=None,
        description="Column alignments (left/center/right)"
    )
    footnotes: Optional[List[str]] = Field(
        default=None,
        description="Table footnotes with calculation details"
    )
    referenced_calculation_ids: List[str] = Field(
        default_factory=list,
        description="IDs of calculations referenced in this table"
    )

    class Config:
        extra = "forbid"


class SectionDraft(BaseModel):
    """
    Structured section before final rendering.

    Used during Phase 2 generation to capture narrative without inline tables.
    Paragraphs stay as text, tables are referenced by ID.
    """
    title: str = Field(
        ...,
        description="Section title"
    )
    paragraphs: List[str] = Field(
        default_factory=list,
        description="Narrative paragraphs (no tables)"
    )
    table_refs: List[str] = Field(
        default_factory=list,
        description="IDs of tables to insert after generation"
    )

    class Config:
        extra = "forbid"
