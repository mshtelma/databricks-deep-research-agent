"""Manual Step schemas for user-defined research workflows.

These models allow power users to define precise research workflows
with specific sources per step, bypassing or augmenting the planner.

Part of 007-enterprise-data-sources feature (T050).
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceConstraintType(StrEnum):
    """How strictly to enforce source constraints."""

    REQUIRED = "required"
    """Sources in the list MUST be queried."""

    PREFERRED = "preferred"
    """Sources in the list should be tried first, but others are allowed."""

    EXCLUSIVE = "exclusive"
    """ONLY sources in the list may be used."""


class SourceConstraint(BaseModel):
    """Constraints on which sources can be used for a step.

    Allows fine-grained control over source selection:
    - Require specific sources to be queried
    - Exclude certain sources
    - Restrict to specific source types
    """

    allowed_types: list[str] | None = None
    """If set, only these source types are allowed (vector_search, genie, etc.)."""

    allowed_sources: list[str] | None = None
    """If set, only these specific sources (by name) are allowed."""

    required_sources: list[str] = Field(default_factory=list)
    """Sources that MUST be queried in this step."""

    excluded_sources: list[str] = Field(default_factory=list)
    """Sources that must NOT be queried in this step."""

    constraint_type: SourceConstraintType = SourceConstraintType.PREFERRED
    """How strictly to enforce the constraints."""

    min_sources: int = 1
    """Minimum number of sources to query (default: 1)."""

    max_sources: int | None = None
    """Maximum number of sources to query (None = unlimited)."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def is_source_allowed(self, source_name: str, source_type: str) -> bool:
        """Check if a source is allowed by these constraints.

        Args:
            source_name: Name of the source.
            source_type: Type of the source.

        Returns:
            True if the source is allowed.
        """
        # Check exclusions first
        if source_name in self.excluded_sources:
            return False

        # Check type restrictions
        if self.allowed_types is not None and source_type not in self.allowed_types:
            return False

        # Check source restrictions (if exclusive)
        return not (
            self.constraint_type == SourceConstraintType.EXCLUSIVE
            and self.allowed_sources is not None
            and source_name not in self.allowed_sources
        )

    def get_missing_required(self, queried_sources: set[str]) -> list[str]:
        """Get list of required sources that haven't been queried.

        Args:
            queried_sources: Set of source names already queried.

        Returns:
            List of required sources not yet queried.
        """
        return [s for s in self.required_sources if s not in queried_sources]


class StepSourceAttachment(BaseModel):
    """A specific source attached to a manual step.

    Provides step-level customization of how a source should be used:
    - Custom prompt/query hint for this step
    - Specific filters to apply
    - Priority within the step
    """

    source_name: str
    """Name of the data source."""

    source_type: str
    """Type of the source (vector_search, genie, knowledge_assistant, web_search)."""

    custom_prompt: str | None = None
    """Custom query or instruction for this source in this step."""

    filters: dict[str, Any] | None = None
    """Filters to apply (for vector search sources)."""

    priority: int = Field(default=2, ge=1, le=3)
    """Priority within step: 1=primary, 2=secondary, 3=supplementary."""

    is_required: bool = False
    """Whether this source MUST be queried in this step."""


class ManualStepDefinition(BaseModel):
    """A user-defined research step with explicit source configuration.

    Manual steps allow users to specify exactly:
    - What to research (title, objective)
    - Which sources to use (attachments)
    - How to constrain source selection (constraints)
    """

    id: str
    """Unique identifier for this step (generated or user-provided)."""

    title: str
    """Short title for the step (displayed in UI)."""

    objective: str
    """Detailed description of what this step should accomplish."""

    sources: list[StepSourceAttachment] = Field(default_factory=list)
    """Specific sources attached to this step with their configurations."""

    constraints: SourceConstraint | None = None
    """Optional constraints on source selection beyond explicit attachments."""

    order: int
    """Execution order (1-based)."""

    is_required: bool = True
    """Whether this step must be executed (vs. optional)."""

    source_scope: str | None = None
    """Per-step source scope override: 'enterprise_only', 'web_only', 'all', or None (inherit)."""

    expected_output: str | None = None
    """Description of expected output format or content."""

    depends_on: list[str] = Field(default_factory=list)
    """IDs of steps that must complete before this one."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def get_source_names(self) -> list[str]:
        """Get all source names attached to this step."""
        return [s.source_name for s in self.sources]

    def get_primary_sources(self) -> list[StepSourceAttachment]:
        """Get sources with priority 1 (primary)."""
        return [s for s in self.sources if s.priority == 1]

    def get_required_sources(self) -> list[StepSourceAttachment]:
        """Get sources marked as required."""
        return [s for s in self.sources if s.is_required]


class ManualWorkflowDefinition(BaseModel):
    """Complete manual research workflow with multiple steps.

    A workflow is an ordered sequence of manual steps that can either:
    - Replace the planner entirely (MANUAL mode)
    - Prepend to planner steps (HYBRID mode)
    """

    name: str
    """Name for this workflow."""

    description: str | None = None
    """Description of what this workflow accomplishes."""

    steps: list[ManualStepDefinition]
    """Ordered list of steps to execute."""

    default_scope: str | None = None
    """Default source scope for steps without explicit constraints."""

    allow_additional_steps: bool = False
    """If True, planner can add steps after manual steps (HYBRID mode)."""

    max_additional_steps: int = 3
    """Maximum steps planner can add in HYBRID mode."""

    def get_step_by_id(self, step_id: str) -> ManualStepDefinition | None:
        """Get a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_steps_in_order(self) -> list[ManualStepDefinition]:
        """Get steps sorted by order."""
        return sorted(self.steps, key=lambda s: s.order)

    def validate_dependencies(self) -> list[str]:
        """Validate that all step dependencies exist.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        step_ids = {s.id for s in self.steps}

        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(
                        f"Step '{step.id}' depends on non-existent step '{dep_id}'"
                    )

        return errors
