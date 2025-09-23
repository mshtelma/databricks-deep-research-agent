"""
Requirement validator for multi-stage validation of extracted requirements.

Provides comprehensive validation of requirements extracted from user instructions,
including consistency checks, feasibility assessment, and completeness verification.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .requirements import (
    RequirementSet, RequiredDataPoint, OutputFormat, TableSpecification, 
    NarrativeSpecification, Constraint, AssumptionRequirement,
    OutputFormatType, TableStructureType, ConstraintType
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue with requirement."""
    
    severity: str  # "error", "warning", "info"
    category: str  # "consistency", "feasibility", "completeness", "format"
    message: str
    suggestion: Optional[str] = None
    affected_component: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of requirement validation."""
    
    is_valid: bool
    issues: List[ValidationIssue]
    confidence_score: float
    validation_method: str
    enhanced_requirements: Optional[RequirementSet] = None
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == "error"]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]
    
    def has_blocking_errors(self) -> bool:
        """Check if there are blocking errors that prevent execution."""
        return len(self.get_errors()) > 0


class RequirementValidator:
    """
    Multi-stage validator for requirement consistency, feasibility, and completeness.
    
    Validation stages:
    1. Structure validation - Check basic requirement structure
    2. Consistency validation - Check for conflicting requirements  
    3. Feasibility validation - Check if requirements are achievable
    4. Completeness validation - Check for missing critical requirements
    5. Enhancement - Add implied requirements and optimizations
    """
    
    def __init__(self):
        """Initialize the requirement validator."""
        self.validation_history: Dict[str, ValidationResult] = {}
    
    def validate(
        self, 
        requirements: RequirementSet,
        enhance: bool = True
    ) -> ValidationResult:
        """
        Perform comprehensive validation of requirements.
        
        Args:
            requirements: The requirement set to validate
            enhance: Whether to enhance requirements with implied needs
            
        Returns:
            ValidationResult with all findings and enhanced requirements
        """
        logger.info("REQUIREMENT_VALIDATOR: Starting comprehensive validation")
        
        all_issues = []
        validation_stages = [
            ("structure", self._validate_structure),
            ("consistency", self._validate_consistency), 
            ("feasibility", self._validate_feasibility),
            ("completeness", self._validate_completeness)
        ]
        
        # Run validation stages
        for stage_name, validator_func in validation_stages:
            logger.info(f"REQUIREMENT_VALIDATOR: Running {stage_name} validation")
            stage_issues = validator_func(requirements)
            all_issues.extend(stage_issues)
            logger.info(f"REQUIREMENT_VALIDATOR: {stage_name} validation found {len(stage_issues)} issues")
        
        # Check for blocking errors
        errors = [issue for issue in all_issues if issue.severity == "error"]
        is_valid = len(errors) == 0
        
        # Enhance requirements if requested and no blocking errors
        enhanced_requirements = None
        if enhance and is_valid:
            logger.info("REQUIREMENT_VALIDATOR: Enhancing requirements")
            enhanced_requirements = self._enhance_requirements(requirements, all_issues)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(requirements, all_issues)
        
        result = ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            confidence_score=confidence,
            validation_method="multi_stage",
            enhanced_requirements=enhanced_requirements or requirements
        )
        
        logger.info(f"REQUIREMENT_VALIDATOR: Validation complete. Valid: {is_valid}, "
                   f"Issues: {len(all_issues)}, Confidence: {confidence:.2f}")
        
        return result
    
    def _validate_structure(self, requirements: RequirementSet) -> List[ValidationIssue]:
        """Validate basic requirement structure."""
        issues = []
        
        # Check if any output formats exist
        if not requirements.output_formats:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="No output formats specified",
                suggestion="Add at least one output format (table, narrative, etc.)",
                affected_component="output_formats"
            ))
        
        # Validate each output format
        for i, output_format in enumerate(requirements.output_formats):
            format_issues = self._validate_output_format(output_format, f"output_formats[{i}]")
            issues.extend(format_issues)
        
        # Check for duplicate constraints
        constraint_types = [c.type for c in requirements.constraints]
        for constraint_type in set(constraint_types):
            if constraint_types.count(constraint_type) > 1:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="structure",
                    message=f"Duplicate constraint type: {constraint_type}",
                    suggestion="Consider combining or removing duplicate constraints",
                    affected_component="constraints"
                ))
        
        return issues
    
    def _validate_output_format(self, output_format: OutputFormat, path: str) -> List[ValidationIssue]:
        """Validate a single output format specification."""
        issues = []
        
        # Table-specific validation
        if output_format.type == OutputFormatType.TABLE:
            if not output_format.table_spec:
                issues.append(ValidationIssue(
                    severity="error",
                    category="structure",
                    message="Table format specified but no table specification provided",
                    suggestion="Add TableSpecification with rows_represent and columns_represent",
                    affected_component=path
                ))
            else:
                table_issues = self._validate_table_spec(output_format.table_spec, f"{path}.table_spec")
                issues.extend(table_issues)
        
        # Narrative-specific validation
        elif output_format.type == OutputFormatType.NARRATIVE:
            if output_format.narrative_spec:
                narrative_issues = self._validate_narrative_spec(
                    output_format.narrative_spec, f"{path}.narrative_spec"
                )
                issues.extend(narrative_issues)
        
        return issues
    
    def _validate_table_spec(self, table_spec: TableSpecification, path: str) -> List[ValidationIssue]:
        """Validate table specification."""
        issues = []
        
        if not table_spec.rows_represent.strip():
            issues.append(ValidationIssue(
                severity="error",
                category="structure", 
                message="Table rows_represent cannot be empty",
                suggestion="Specify what the table rows represent (e.g., 'countries', 'scenarios')",
                affected_component=path
            ))
        
        if not table_spec.columns_represent.strip():
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message="Table columns_represent cannot be empty", 
                suggestion="Specify what the table columns represent (e.g., 'metrics', 'time periods')",
                affected_component=path
            ))
        
        # Check if table structure makes sense
        if table_spec.rows_represent.lower() == table_spec.columns_represent.lower():
            issues.append(ValidationIssue(
                severity="warning",
                category="consistency",
                message="Table rows and columns represent the same thing",
                suggestion="Ensure rows and columns represent different dimensions",
                affected_component=path
            ))
        
        return issues
    
    def _validate_narrative_spec(self, narrative_spec: NarrativeSpecification, path: str) -> List[ValidationIssue]:
        """Validate narrative specification."""
        issues = []
        
        # Check word limit reasonableness
        if narrative_spec.word_limit:
            if narrative_spec.word_limit < 50:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="feasibility",
                    message=f"Very low word limit: {narrative_spec.word_limit}",
                    suggestion="Consider if such a short response can be meaningful",
                    affected_component=path
                ))
            elif narrative_spec.word_limit > 5000:
                issues.append(ValidationIssue(
                    severity="warning", 
                    category="feasibility",
                    message=f"Very high word limit: {narrative_spec.word_limit}",
                    suggestion="Consider breaking into multiple sections or reducing scope",
                    affected_component=path
                ))
        
        return issues
    
    def _validate_consistency(self, requirements: RequirementSet) -> List[ValidationIssue]:
        """Check for conflicting or inconsistent requirements."""
        issues = []
        
        # Check word count constraints vs narrative specs
        word_count_constraints = [
            c for c in requirements.constraints 
            if c.type == ConstraintType.WORD_COUNT
        ]
        
        narrative_formats = [
            fmt for fmt in requirements.output_formats 
            if fmt.type == OutputFormatType.NARRATIVE and fmt.narrative_spec
        ]
        
        if word_count_constraints and narrative_formats:
            constraint_limit = int(word_count_constraints[0].value)
            for narrative_format in narrative_formats:
                if narrative_format.narrative_spec.word_limit:
                    narrative_limit = narrative_format.narrative_spec.word_limit
                    if constraint_limit != narrative_limit:
                        issues.append(ValidationIssue(
                            severity="warning",
                            category="consistency",
                            message=f"Word limit mismatch: constraint={constraint_limit}, narrative={narrative_limit}",
                            suggestion="Align word limits between constraints and narrative specifications",
                            affected_component="word_limits"
                        ))
        
        # Check for conflicting format requirements
        table_formats = [fmt for fmt in requirements.output_formats if fmt.type == OutputFormatType.TABLE]
        narrative_formats = [fmt for fmt in requirements.output_formats if fmt.type == OutputFormatType.NARRATIVE]
        
        if table_formats and narrative_formats:
            # This is actually fine - mixed format
            pass
        
        # Check data point requirements vs output formats
        if requirements.required_data_points:
            data_point_names = [dp.name for dp in requirements.required_data_points]
            
            # Check if table data points are mentioned in table specs
            for table_format in table_formats:
                if table_format.table_spec and table_format.table_spec.required_data_points:
                    table_data_points = set(table_format.table_spec.required_data_points)
                    required_data_points = set(data_point_names)
                    
                    missing_in_table = required_data_points - table_data_points
                    if missing_in_table:
                        issues.append(ValidationIssue(
                            severity="warning",
                            category="consistency",
                            message=f"Data points not mentioned in table spec: {list(missing_in_table)}",
                            suggestion="Ensure all required data points are included in table specification",
                            affected_component="table_data_points"
                        ))
        
        return issues
    
    def _validate_feasibility(self, requirements: RequirementSet) -> List[ValidationIssue]:
        """Check if requirements are achievable with current capabilities."""
        issues = []
        
        # Check table complexity
        table_formats = [fmt for fmt in requirements.output_formats if fmt.type == OutputFormatType.TABLE]
        for table_format in table_formats:
            if table_format.table_spec:
                data_points = len(table_format.table_spec.required_data_points)
                if data_points > 20:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="feasibility",
                        message=f"Very complex table with {data_points} data points",
                        suggestion="Consider breaking into multiple smaller tables",
                        affected_component="table_complexity"
                    ))
        
        # Check total data point requirements
        total_data_points = len(requirements.required_data_points)
        if total_data_points > 30:
            issues.append(ValidationIssue(
                severity="warning",
                category="feasibility",
                message=f"Very high number of required data points: {total_data_points}",
                suggestion="Consider prioritizing the most critical data points",
                affected_component="data_point_count"
            ))
        
        # Check multiple output format feasibility
        if len(requirements.output_formats) > 3:
            issues.append(ValidationIssue(
                severity="warning",
                category="feasibility",
                message=f"Multiple output formats requested: {len(requirements.output_formats)}",
                suggestion="Consider if all formats are necessary or could be combined",
                affected_component="output_format_count"
            ))
        
        return issues
    
    def _validate_completeness(self, requirements: RequirementSet) -> List[ValidationIssue]:
        """Check for missing critical requirements."""
        issues = []
        
        # Check if table format is missing critical specifications
        table_formats = [fmt for fmt in requirements.output_formats if fmt.type == OutputFormatType.TABLE]
        for table_format in table_formats:
            if table_format.table_spec:
                spec = table_format.table_spec
                if not spec.required_data_points:
                    issues.append(ValidationIssue(
                        severity="info",
                        category="completeness",
                        message="Table format has no specific data points listed",
                        suggestion="Consider adding specific data points that must be included in the table",
                        affected_component="table_data_points"
                    ))
        
        # Check if success criteria are defined
        if not requirements.success_criteria:
            issues.append(ValidationIssue(
                severity="info",
                category="completeness",
                message="No success criteria defined",
                suggestion="Success criteria will be auto-generated from requirements",
                affected_component="success_criteria"
            ))
        
        # Check if critical data points have fallback strategies
        critical_data_points = [dp for dp in requirements.required_data_points if dp.is_critical]
        for data_point in critical_data_points:
            if not data_point.fallback_strategy:
                issues.append(ValidationIssue(
                    severity="info",
                    category="completeness",
                    message=f"Critical data point '{data_point.name}' has no fallback strategy",
                    suggestion="Consider what to do if this data point cannot be obtained",
                    affected_component="fallback_strategies"
                ))
        
        return issues
    
    def _enhance_requirements(
        self, 
        requirements: RequirementSet, 
        validation_issues: List[ValidationIssue]
    ) -> RequirementSet:
        """Enhance requirements with implied needs and optimizations."""
        
        logger.info("REQUIREMENT_VALIDATOR: Enhancing requirements with implied needs")
        
        # Create a copy to modify
        enhanced = RequirementSet(
            output_formats=requirements.output_formats.copy(),
            required_data_points=requirements.required_data_points.copy(),
            constraints=requirements.constraints.copy(),
            assumptions=requirements.assumptions.copy(),
            success_criteria=requirements.success_criteria.copy(),
            complexity_level=requirements.complexity_level,
            estimated_research_steps=requirements.estimated_research_steps,
            original_instruction=requirements.original_instruction,
            extraction_confidence=requirements.extraction_confidence
        )
        
        # Add implied constraints based on output formats
        if enhanced.requires_table() and not any(c.type == ConstraintType.FORMAT_REQUIREMENT for c in enhanced.constraints):
            enhanced.constraints.append(Constraint(
                type=ConstraintType.FORMAT_REQUIREMENT,
                value="markdown",
                description="Use markdown format for table output",
                is_hard_requirement=False
            ))
        
        # Add common assumptions for research tasks
        if not enhanced.assumptions:
            enhanced.assumptions.extend([
                AssumptionRequirement(
                    category="data",
                    description="Data sources are current as of search date"
                ),
                AssumptionRequirement(
                    category="calculation",
                    description="Calculations use standard methodologies unless otherwise specified"
                )
            ])
        
        # Enhance success criteria if empty
        if not enhanced.success_criteria:
            enhanced.success_criteria = self._generate_success_criteria(enhanced)
        
        # Add data quality requirements for complex requests
        if enhanced.complexity_level == "complex" and len(enhanced.required_data_points) > 5:
            enhanced.required_data_points.append(RequiredDataPoint(
                name="source_quality_assessment",
                description="Assessment of data source reliability",
                is_critical=False,
                fallback_strategy="Note any data quality limitations"
            ))
        
        logger.info(f"REQUIREMENT_VALIDATOR: Enhanced requirements - added {len(enhanced.assumptions)} assumptions, "
                   f"{len(enhanced.success_criteria)} success criteria")
        
        return enhanced
    
    def _generate_success_criteria(self, requirements: RequirementSet) -> List[str]:
        """Generate success criteria based on requirements."""
        criteria = []
        
        if requirements.requires_table():
            criteria.append("Generate table with correct structure and all required data")
        
        if requirements.requires_narrative():
            word_limit = requirements.get_word_limit()
            if word_limit:
                criteria.append(f"Keep narrative content within {word_limit} word limit")
        
        if requirements.required_data_points:
            critical_count = len([dp for dp in requirements.required_data_points if dp.is_critical])
            criteria.append(f"Include all {critical_count} critical data points")
        
        criteria.append("Provide accurate and well-sourced information")
        criteria.append("Format output according to specifications")
        
        return criteria
    
    def _calculate_confidence_score(
        self, 
        requirements: RequirementSet, 
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate confidence score based on requirements quality and issues."""
        
        base_confidence = requirements.extraction_confidence
        
        # Penalty for errors and warnings
        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity == "warning"]
        
        error_penalty = len(errors) * 0.2
        warning_penalty = len(warnings) * 0.05
        
        # Bonus for completeness
        completeness_bonus = 0.0
        if requirements.output_formats:
            completeness_bonus += 0.05
        if requirements.required_data_points:
            completeness_bonus += 0.05
        if requirements.constraints:
            completeness_bonus += 0.05
        
        final_confidence = base_confidence - error_penalty - warning_penalty + completeness_bonus
        return max(0.0, min(1.0, final_confidence))