"""Query configuration schemas for Vector Search.

This module defines Pydantic models for configuring query settings
per Vector Search data source, including:
- Query type selection (ANN, HYBRID, FULL_TEXT)
- Filter expressions (SQL-like or dictionary syntax)
- Result settings (num_results, score_threshold)
- Reranking configuration

Supports functional requirements FR-138 through FR-147.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryType(StrEnum):
    """Supported query types for Vector Search.

    ANN: Approximate Nearest Neighbor - fast vector search (default)
    HYBRID: Combines vector + keyword search (max 200 results)
    FULL_TEXT: Keyword-only search, no vectors (max 200 results, beta)
    """

    ANN = "ANN"
    HYBRID = "HYBRID"
    FULL_TEXT = "FULL_TEXT"


class FilterOperator(StrEnum):
    """Supported filter operators for Vector Search.

    Operators vary by column data type:
    - Comparison (all types): =, !=, <, <=, >, >=
    - String patterns: LIKE, NOT_LIKE
    - List membership: IN
    """

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"


class FilterSyntax(StrEnum):
    """Filter syntax options for Vector Search queries.

    SQL: SQL-like syntax for storage-optimized endpoints
         e.g., "category = 'docs' AND date > '2024-01-01'"

    DICT: Dictionary syntax for standard endpoints
          e.g., {"category": "docs", "date >": "2024-01-01"}
    """

    SQL = "sql"
    DICT = "dict"


# Maximum IDs per IN clause (Databricks limitation)
MAX_IN_CLAUSE_IDS = 1024


class FilterExpression(BaseModel):
    """A single filter expression for Vector Search queries.

    Examples:
        - FilterExpression(column="category", operator="=", value="docs")
        - FilterExpression(column="date", operator=">", value="2024-01-01")
        - FilterExpression(column="id", operator="IN", value=[1, 2, 3])
    """

    column: str = Field(..., min_length=1, description="Column name to filter on")

    operator: FilterOperator = Field(
        ...,
        description="Filter operator",
    )

    value: str | int | float | list[str | int | float] = Field(
        ...,
        description="Filter value (single value or list for IN operator)",
    )

    @field_validator("value")
    @classmethod
    def validate_in_clause_limit(cls, v: Any, _info: Any) -> Any:
        """Validate IN clause doesn't exceed ID limit."""
        if isinstance(v, list) and len(v) > MAX_IN_CLAUSE_IDS:
            raise ValueError(
                f"IN filter exceeds {MAX_IN_CLAUSE_IDS} ID limit (got {len(v)}). "
                "Consider batching using OR clauses."
            )
        return v

    def to_sql(self) -> str:
        """Convert filter expression to SQL-like string.

        Returns:
            SQL-like filter string (e.g., "category = 'docs'")
        """
        if self.operator == FilterOperator.IN:
            if isinstance(self.value, list):
                values = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v) for v in self.value
                )
                return f"{self.column} IN ({values})"
            return f"{self.column} IN ('{self.value}')"

        elif self.operator in (FilterOperator.LIKE, FilterOperator.NOT_LIKE):
            return f"{self.column} {self.operator.value} '{self.value}'"

        else:
            val = f"'{self.value}'" if isinstance(self.value, str) else self.value
            return f"{self.column} {self.operator.value} {val}"

    def to_dict(self) -> dict[str, Any]:
        """Convert filter expression to dictionary format.

        Returns:
            Dictionary filter (e.g., {"category": "docs"} or {"date >": "2024-01-01"})
        """
        if self.operator == FilterOperator.EQ:
            return {self.column: self.value}
        return {f"{self.column} {self.operator.value}": self.value}


class VectorSearchQueryConfig(BaseModel):
    """Query configuration for a Vector Search index.

    Defines how queries should be executed against a specific index,
    including query type, filters, result settings, and reranking.

    Example:
        config = VectorSearchQueryConfig(
            query_type=QueryType.HYBRID,
            num_results=20,
            filters=[
                FilterExpression(column="category", operator="=", value="docs"),
            ],
            enable_reranking=True,
            columns_to_rerank=["content", "title"],
        )
    """

    # Query type selection
    query_type: QueryType = Field(
        default=QueryType.ANN,
        description="Query type: ANN (default), HYBRID, or FULL_TEXT",
    )

    # Result settings
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return (1-100)",
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0-1.0)",
    )

    columns: list[str] | None = Field(
        default=None,
        description="Columns to return (None = all columns)",
    )

    # Reranking settings
    enable_reranking: bool = Field(
        default=False,
        description="Enable reranking for improved relevance",
    )

    columns_to_rerank: list[str] | None = Field(
        default=None,
        description="Text columns to use for reranking",
    )

    # Filter settings
    filters: list[FilterExpression] = Field(
        default_factory=list,
        description="Filter expressions to apply",
    )

    filter_syntax: FilterSyntax = Field(
        default=FilterSyntax.SQL,
        description="Filter syntax (sql or dict)",
    )

    @field_validator("num_results")
    @classmethod
    def validate_results_limit(cls, v: int, _info: Any) -> int:
        """Validate result limit for HYBRID and FULL_TEXT queries."""
        # This is a soft validation - actual enforcement happens at query time
        # when we know the query type
        return v

    def build_filters_sql(self) -> str | None:
        """Build SQL-like filter string from expressions.

        Returns:
            SQL filter string or None if no filters.
        """
        if not self.filters:
            return None
        return " AND ".join(f.to_sql() for f in self.filters)

    def build_filters_dict(self) -> dict[str, Any] | None:
        """Build dictionary filters from expressions.

        Returns:
            Filter dictionary or None if no filters.
        """
        if not self.filters:
            return None

        result: dict[str, Any] = {}
        for f in self.filters:
            result.update(f.to_dict())
        return result


class QueryConfigValidationResult(BaseModel):
    """Result of validating query config against index capabilities."""

    is_valid: bool
    """Whether the configuration is valid."""

    errors: list[str] = Field(default_factory=list)
    """List of validation errors."""

    warnings: list[str] = Field(default_factory=list)
    """List of validation warnings (config will work but may not be optimal)."""


def validate_query_config(
    config: VectorSearchQueryConfig,
    supported_query_types: list[QueryType],
    filter_columns: list[str],
    supports_reranking: bool = False,
) -> QueryConfigValidationResult:
    """Validate query configuration against index capabilities.

    Args:
        config: Query configuration to validate.
        supported_query_types: Query types supported by the index.
        filter_columns: Columns available for filtering.
        supports_reranking: Whether reranking is supported.

    Returns:
        QueryConfigValidationResult with validation status.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check query type is supported
    if config.query_type not in supported_query_types:
        errors.append(
            f"Query type {config.query_type.value} not supported by index. "
            f"Supported types: {[t.value for t in supported_query_types]}"
        )

    # Check result limit for HYBRID/FULL_TEXT
    if config.query_type in (QueryType.HYBRID, QueryType.FULL_TEXT) and config.num_results > 200:
        errors.append(
            f"{config.query_type.value} queries limited to 200 results "
            f"(requested: {config.num_results})"
        )

    # Check reranking
    if config.enable_reranking and not supports_reranking:
        errors.append("Reranking not supported by index (no text columns available)")

    if config.columns_to_rerank:
        missing_cols = set(config.columns_to_rerank) - set(filter_columns)
        if missing_cols:
            errors.append(f"Reranking columns not found in index: {missing_cols}")

    # Check filter columns
    for f in config.filters:
        if f.column not in filter_columns:
            warnings.append(
                f"Filter column '{f.column}' may not exist in index. "
                f"Available columns: {filter_columns}"
            )

        # Check IN clause limit
        if f.operator == FilterOperator.IN and isinstance(f.value, list) and len(f.value) > MAX_IN_CLAUSE_IDS:
                errors.append(
                    f"IN filter on '{f.column}' exceeds {MAX_IN_CLAUSE_IDS} ID limit "
                    f"(got {len(f.value)})"
                )

    return QueryConfigValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# API Request/Response Schemas
# =============================================================================


class UpdateQueryConfigRequest(BaseModel):
    """Request to update query configuration for a data source."""

    query_type: QueryType | None = None
    num_results: int | None = Field(None, ge=1, le=100)
    score_threshold: float | None = Field(None, ge=0.0, le=1.0)
    columns: list[str] | None = None
    enable_reranking: bool | None = None
    columns_to_rerank: list[str] | None = None
    filters: list[FilterExpression] | None = None
    filter_syntax: FilterSyntax | None = None


class QueryConfigResponse(BaseModel):
    """Response schema for query configuration."""

    source_id: str
    """Source ID this config applies to."""

    config: VectorSearchQueryConfig
    """The query configuration."""

    validation: QueryConfigValidationResult | None = None
    """Validation result if validation was performed."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
