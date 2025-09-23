"""
Core requirement models for instruction-aware planning system.

This module defines Pydantic models for capturing and validating user requirements
extracted from natural language instructions.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


class OutputFormatType(str, Enum):
    """Types of output formats that can be requested."""
    TABLE = "table"
    NARRATIVE = "narrative"
    VISUALIZATION = "visualization"
    MIXED = "mixed"
    LIST = "list"
    APPENDIX = "appendix"


class TableStructureType(str, Enum):
    """Types of table structures."""
    COMPARATIVE = "comparative"
    SUMMARY = "summary" 
    DATA_MATRIX = "data_matrix"
    TIMELINE = "timeline"
    RANKING = "ranking"


class ConstraintType(str, Enum):
    """Types of constraints that can be applied."""
    WORD_COUNT = "word_count"
    SECTION_COUNT = "section_count"
    FORMAT_REQUIREMENT = "format_requirement"
    DATA_REQUIREMENT = "data_requirement"
    STYLE_REQUIREMENT = "style_requirement"


class TableSpecification(BaseModel):
    """Detailed specification for table requirements."""
    
    structure_type: TableStructureType = Field(
        description="Type of table structure requested"
    )
    
    rows_represent: str = Field(
        description="What the rows represent (e.g., 'countries', 'scenarios', 'time periods')"
    )
    
    columns_represent: str = Field(
        description="What the columns represent (e.g., 'scenarios', 'metrics', 'categories')"
    )
    
    required_data_points: List[str] = Field(
        default_factory=list,
        description="Specific data points that must be included in the table"
    )
    
    format_specification: str = Field(
        default="markdown",
        description="Output format (markdown, csv, json, html)"
    )
    
    must_include_totals: bool = Field(
        default=False,
        description="Whether totals/summaries are required"
    )
    
    sort_by: Optional[str] = Field(
        default=None,
        description="How to sort the table (if specified)"
    )


class NarrativeSpecification(BaseModel):
    """Detailed specification for narrative requirements."""
    
    word_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of words allowed"
    )
    
    required_sections: List[str] = Field(
        default_factory=list,
        description="Sections that must be included"
    )
    
    style: str = Field(
        default="professional",
        description="Writing style (professional, academic, casual, etc.)"
    )
    
    must_highlight: List[str] = Field(
        default_factory=list,
        description="Key points that must be highlighted"
    )
    
    numbered_format: bool = Field(
        default=False,
        description="Whether to use numbered format"
    )


class VisualizationSpecification(BaseModel):
    """Specification for visualization requirements."""
    
    chart_type: str = Field(
        description="Type of chart/visualization requested"
    )
    
    data_series: List[str] = Field(
        description="Data series to be visualized"
    )
    
    format: str = Field(
        default="description",
        description="Output format (description, vega-lite, matplotlib, etc.)"
    )


class Constraint(BaseModel):
    """A specific constraint on the output."""
    
    type: ConstraintType = Field(
        description="Type of constraint"
    )
    
    value: Union[int, str, bool] = Field(
        description="The constraint value"
    )
    
    description: str = Field(
        description="Human-readable description of the constraint"
    )
    
    is_hard_requirement: bool = Field(
        default=True,
        description="Whether this is a hard requirement or preference"
    )


class RequiredDataPoint(BaseModel):
    """A specific data point that must be included in the output."""
    
    name: str = Field(
        description="Name/identifier for the data point"
    )
    
    description: str = Field(
        description="What this data point represents"
    )
    
    format_requirement: Optional[str] = Field(
        default=None,
        description="Specific format requirement (percentage, currency, etc.)"
    )
    
    is_critical: bool = Field(
        default=True,
        description="Whether this data point is critical for success"
    )
    
    fallback_strategy: Optional[str] = Field(
        default=None,
        description="What to do if this data point cannot be obtained"
    )


class AssumptionRequirement(BaseModel):
    """An assumption that must be made explicit in the output."""
    
    category: str = Field(
        description="Category of assumption (tax, calculation, data, etc.)"
    )
    
    description: str = Field(
        description="The assumption that should be stated"
    )
    
    rationale: Optional[str] = Field(
        default=None,
        description="Why this assumption is necessary"
    )


class OutputFormat(BaseModel):
    """Specification for a single output format."""
    
    type: OutputFormatType = Field(
        description="Type of output format"
    )
    
    table_spec: Optional[TableSpecification] = Field(
        default=None,
        description="Table specification if type is TABLE"
    )
    
    narrative_spec: Optional[NarrativeSpecification] = Field(
        default=None,
        description="Narrative specification if type is NARRATIVE"
    )
    
    visualization_spec: Optional[VisualizationSpecification] = Field(
        default=None,
        description="Visualization specification if type is VISUALIZATION"
    )
    
    priority: int = Field(
        default=1,
        description="Priority order for this format (1 = highest)"
    )


class RequirementSet(BaseModel):
    """Complete set of requirements extracted from user instructions."""
    
    # Core output requirements
    output_formats: List[OutputFormat] = Field(
        description="All requested output formats"
    )
    
    # Data requirements
    required_data_points: List[RequiredDataPoint] = Field(
        default_factory=list,
        description="All data points that must be included"
    )
    
    # Constraints
    constraints: List[Constraint] = Field(
        default_factory=list,
        description="All constraints that must be satisfied"
    )
    
    # Assumptions to make explicit
    assumptions: List[AssumptionRequirement] = Field(
        default_factory=list,
        description="Assumptions that must be stated"
    )
    
    # Success criteria
    success_criteria: List[str] = Field(
        default_factory=list,
        description="What defines success for this request"
    )
    
    # Meta information
    complexity_level: Literal["simple", "moderate", "complex"] = Field(
        default="moderate",
        description="Assessed complexity of the requirements"
    )
    
    estimated_research_steps: int = Field(
        default=5,
        description="Estimated number of research steps needed"
    )
    
    original_instruction: str = Field(
        description="The original user instruction"
    )
    
    extraction_confidence: float = Field(
        default=0.8,
        description="Confidence in requirement extraction (0.0-1.0)"
    )

    def get_primary_output_format(self) -> OutputFormat:
        """Get the primary (highest priority) output format."""
        if not self.output_formats:
            # Default fallback
            return OutputFormat(type=OutputFormatType.NARRATIVE)
        
        return min(self.output_formats, key=lambda x: x.priority)
    
    def requires_table(self) -> bool:
        """Check if any output format requires a table."""
        return any(fmt.type == OutputFormatType.TABLE for fmt in self.output_formats)
    
    def requires_narrative(self) -> bool:
        """Check if any output format requires a narrative."""
        return any(fmt.type == OutputFormatType.NARRATIVE for fmt in self.output_formats)
    
    def get_word_limit(self) -> Optional[int]:
        """Get word limit constraint if specified."""
        for constraint in self.constraints:
            if constraint.type == ConstraintType.WORD_COUNT:
                return int(constraint.value)
        return None
    
    def get_critical_data_points(self) -> List[str]:
        """Get list of critical data point names."""
        return [dp.name for dp in self.required_data_points if dp.is_critical]
    
    def validate_completeness(self) -> tuple[bool, List[str]]:
        """Validate that requirements are complete and consistent."""
        issues = []
        
        if not self.output_formats:
            issues.append("No output formats specified")
        
        # Check table specs
        for fmt in self.output_formats:
            if fmt.type == OutputFormatType.TABLE and not fmt.table_spec:
                issues.append("Table format specified but no table specification provided")
            if fmt.type == OutputFormatType.NARRATIVE and not fmt.narrative_spec:
                issues.append("Narrative format specified but no narrative specification provided")
        
        # Check for conflicting constraints
        word_limits = [c for c in self.constraints if c.type == ConstraintType.WORD_COUNT]
        if len(word_limits) > 1:
            issues.append("Multiple conflicting word count constraints")
        
        return len(issues) == 0, issues


class RequirementExtractionResult(BaseModel):
    """Result of requirement extraction process."""
    
    requirements: RequirementSet = Field(
        description="The extracted requirements"
    )
    
    extraction_method: str = Field(
        description="Method used for extraction (llm, rules, hybrid)"
    )
    
    confidence_score: float = Field(
        description="Overall confidence in the extraction"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about potential issues"
    )
    
    fallback_applied: bool = Field(
        default=False,
        description="Whether fallback extraction was used"
    )
    
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation errors found"
    )