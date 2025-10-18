"""Unified planning models for user-request-driven calculation pipeline.

This module defines the data structures for unified planning that eliminates
the need for metric matching by creating explicit links between user requests,
data sources, and response tables.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


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


class DataSource(BaseModel):
    """Specification for how to obtain a single piece of data."""

    data_id: str = Field(description="Unique identifier for this data point")
    source_type: str = Field(description="'extract' or 'calculate'")

    # For extraction from observations
    observation_id: Optional[str] = Field(
        default=None,
        description="ID of observation to extract from"
    )
    extraction_path: Optional[str] = Field(
        default=None,
        description="Hint about where to find the value in the observation"
    )

    # For calculations
    formula: Optional[str] = Field(
        default=None,
        description="Mathematical formula for calculation"
    )
    required_inputs: List[str] = Field(
        default_factory=list,
        description="List of data_ids needed for this calculation"
    )


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
    title: str = Field(description="Human-readable title")
    purpose: str = Field(description="What specific question this table answers")

    rows: List[str] = Field(description="List of row labels (entities)")
    columns: List[str] = Field(description="List of column labels (metrics)")
    cells: List[TableCell] = Field(description="Cell specifications with data links")


class UnifiedPlan(BaseModel):
    """Complete plan to answer a user's request with explicit data links.

    This unified plan eliminates the need for metric matching by creating
    explicit links between:
    1. User's request → Required metrics
    2. Metrics → Data sources (extract or calculate)
    3. Data sources → Table cells
    4. Table cells → Response presentation
    """

    request_analysis: UserRequestAnalysis = Field(
        description="Understanding of what the user is asking for"
    )

    data_sources: Dict[str, DataSource] = Field(
        default_factory=dict,
        description="Map of data_id to data source specification"
    )

    response_tables: List[ResponseTable] = Field(
        default_factory=list,
        description="Tables that directly present the answer to user's request"
    )

    narrative_points: List[str] = Field(
        default_factory=list,
        description="Key insights that answer the user's specific questions"
    )

    def get_all_calculation_ids(self) -> List[str]:
        """Get IDs of all calculated data sources."""
        return [
            data_id
            for data_id, source in self.data_sources.items()
            if source.source_type == "calculate"
        ]

    def get_all_extraction_ids(self) -> List[str]:
        """Get IDs of all extracted data sources."""
        return [
            data_id
            for data_id, source in self.data_sources.items()
            if source.source_type == "extract"
        ]

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for calculations."""
        graph = {}
        for data_id, source in self.data_sources.items():
            if source.source_type == "calculate":
                graph[data_id] = source.required_inputs
        return graph


__all__ = [
    "UserRequestAnalysis",
    "DataSource",
    "TableCell",
    "ResponseTable",
    "UnifiedPlan",
]
