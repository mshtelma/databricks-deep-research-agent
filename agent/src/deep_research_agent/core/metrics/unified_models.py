"""Unified planning models for user-request-driven calculation pipeline.

This module defines the data structures for unified planning that eliminates
the need for metric matching by creating explicit links between user requests,
data sources, and response tables.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from .config import SourceType


class UserRequestAnalysis(BaseModel):
    """Analysis of what the user is asking for."""

    what_user_wants: str = Field(
        description="Core question or comparison the user is asking for"
    )
    entities_to_compare: List[str] = Field(
        default_factory=list,
        description="Entities being compared (countries, products, scenarios, etc.)"
    )
    metrics_requested: List[str] = Field(
        default_factory=list,
        description="Specific metrics that would answer the user's question"
    )
    comparison_dimensions: List[str] = Field(
        default_factory=list,
        description="Dimensions for comparison (scenarios, time periods, etc.)"
    )


class MetricSpec(BaseModel):
    """Specification for obtaining a single metric value.

    Note: Renamed from DataSource to avoid confusion with actual source observations.
    A MetricSpec describes HOW to obtain a metric, not the source itself.
    """

    data_id: str = Field(description="Unique identifier for this data point")
    source_type: str = Field(description="'extract', 'calculate', or 'constant'")

    # Extracted/constant value
    value: Optional[float] = Field(
        default=None,
        description="Extracted or constant value (for extract/constant source types)"
    )

    # Confidence score
    confidence: float = Field(
        default=0.9,
        description="Confidence score for this metric (0.0 to 1.0)"
    )

    # For extraction from observations
    observation_id: Optional[str] = Field(
        default=None,
        description="ID of observation to extract from"
    )
    fallback_observation_ids: List[str] = Field(
        default_factory=list,
        description="Alternative observations to try if primary fails"
    )

    # Robust extraction (IMPROVED - replaces brittle extraction_path)
    extraction_hint: Optional[str] = Field(
        default=None,
        description="Human-readable hint for LLM extraction (e.g., 'Spain net take-home in EUR')",
        alias="extraction_path"  # Accept both "extraction_path" and "extraction_hint"
    )
    extraction_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns to try for extraction (with fallbacks)"
    )

    # For calculations
    formula: Optional[str] = Field(
        default=None,
        description="Mathematical formula for calculation"
    )
    required_inputs: List[str] = Field(
        default_factory=list,
        description="List of data_ids needed for this calculation",
        alias="inputs"  # Accept both "required_inputs" and "inputs"
    )

    # Grouping and context (NEW - enables multi-scenario support)
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Tags for filtering and grouping (e.g., country, metric, scenario, year)"
    )

    # Data availability (NEW - simple missing data handling)
    availability: str = Field(
        default="available",
        description="Data availability status: 'available', 'missing', 'not_applicable'"
    )
    fallback_note: Optional[str] = Field(
        default=None,
        description="Explanation if using fallback or missing data"
    )

    # Unit tracking (NEW - enables validation)
    unit: str = Field(
        default="",
        description="Unit of measurement (EUR, USD, %, days, etc.)"
    )
    original_unit: Optional[str] = Field(
        default=None,
        description="Original unit if converted (e.g., 'GBP' if converted to EUR)"
    )

    # Quality metadata
    source_quality: Optional[str] = Field(
        default=None,
        description="Quality indicator: 'primary', 'synthesized', 'estimated'"
    )
    requires_verification: bool = Field(
        default=False,
        description="Whether this metric requires additional verification"
    )

    # FIX: Updated to Pydantic v2 ConfigDict syntax
    model_config = ConfigDict(
        populate_by_name=True,  # Accept both "extraction_hint" and "extraction_path"
    )

    @property
    def extraction_path(self) -> Optional[str]:
        """Backward compatibility property for extraction_path."""
        return self.extraction_hint

    @property
    def inputs(self) -> List[str]:
        """Backward compatibility property for inputs."""
        return self.required_inputs


# Keep DataSource as alias for backward compatibility
DataSource = MetricSpec

# Keep RequestAnalysis as alias for UserRequestAnalysis (hybrid planner compatibility)
RequestAnalysis = UserRequestAnalysis


class TableCell(BaseModel):
    """Single cell in a response table."""

    cell_id: str = Field(description="Unique identifier for this cell")
    row: str = Field(description="Row label/entity")
    column: str = Field(description="Column label/metric")
    data_id: str = Field(description="ID of the data source for this cell's value")
    answers_user_question: str = Field(
        description="Explanation of how this cell helps answer the user's request"
    )


class ResponseTable(BaseModel):
    """Table structure that directly answers the user's request."""

    table_id: str = Field(description="Unique identifier for this table")
    # FIX: Renamed from 'title' to 'table_title' to avoid Pydantic v2 reserved field conflict
    table_title: str = Field(
        description="Human-readable title",
        validation_alias="title"  # Accept 'title' as input for backward compatibility
    )
    purpose: str = Field(description="What specific question this table answers")

    rows: List[str] = Field(description="List of row labels (entities)")
    columns: List[str] = Field(description="List of column labels (metrics)")
    cells: List[TableCell] = Field(description="Cell specifications with data links")

    # NEW: Simple context dict for filtering (enables multi-scenario support)
    context: Dict[str, str] = Field(
        default_factory=dict,
        description="Context tags for this table (e.g., scenario, year, region)"
    )

    # Backward compatibility property for .title access
    @property
    def title(self) -> str:
        """Backward compatibility - use table_title instead."""
        return self.table_title


class UnifiedPlan(BaseModel):
    """Complete plan to answer a user's request with explicit data links.

    This unified plan eliminates the need for metric matching by creating
    explicit links between:
    1. User's request → Required metrics
    2. Metrics → MetricSpecs (extract or calculate)
    3. MetricSpecs → Table cells
    4. Table cells → Response presentation
    """

    request_analysis: UserRequestAnalysis = Field(
        description="Understanding of what the user is asking for"
    )

    metric_specs: Dict[str, MetricSpec] = Field(
        default_factory=dict,
        description="Map of data_id to metric specification",
        alias="data_sources"  # Accept both "data_sources" and "metric_specs"
    )

    response_tables: List[ResponseTable] = Field(
        default_factory=list,
        description="Tables that directly present the answer to user's request"
    )

    narrative_points: List[str] = Field(
        default_factory=list,
        description="Key insights that answer the user's specific questions"
    )

    # Backward compatibility
    @property
    def data_sources(self) -> Dict[str, MetricSpec]:
        """Alias for metric_specs (backward compatibility)."""
        return self.metric_specs

    # FIX: Updated to Pydantic v2 ConfigDict syntax
    model_config = ConfigDict(
        populate_by_name=True,  # Accept both "metric_specs" and "data_sources"
    )

    def get_all_calculation_ids(self) -> List[str]:
        """Get IDs of all calculated metric specs."""
        return [
            data_id
            for data_id, spec in self.metric_specs.items()
            if spec.source_type == "calculate"
        ]

    def get_all_extraction_ids(self) -> List[str]:
        """Get IDs of all extracted metric specs."""
        return [
            data_id
            for data_id, spec in self.metric_specs.items()
            if spec.source_type == "extract"
        ]

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for calculations."""
        graph = {}
        for data_id, spec in self.metric_specs.items():
            if spec.source_type == "calculate":
                graph[data_id] = spec.required_inputs
        return graph


__all__ = [
    "UserRequestAnalysis",
    "RequestAnalysis",  # Alias for hybrid planner compatibility
    "MetricSpec",
    "DataSource",  # Backward compatibility alias
    "TableCell",
    "ResponseTable",
    "UnifiedPlan",
    "SourceType",  # Re-export from config
]
