"""Prompt template schemas for the custom template library.

This module defines Pydantic models for:
- TemplateType: Types of prompt templates
- TemplateVariable: Variable definition with metadata
- Request/Response schemas for template API endpoints

Part of US5 - Custom Prompt Template Library (T069).
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from deep_research.schemas.common import BaseSchema


class TemplateType(str, Enum):
    """Types of prompt templates.

    Each type serves a different purpose in the research pipeline:
    - SYSTEM: System-level instructions for the entire agent
    - STEP: Instructions for individual research steps
    - SYNTHESIS: Final report generation prompts
    - QUERY: Query formulation/rewriting templates
    """

    SYSTEM = "system"
    STEP = "step"
    SYNTHESIS = "synthesis"
    QUERY = "query"


class TemplateVisibility(str, Enum):
    """Visibility levels for prompt templates."""

    PRIVATE = "private"
    WORKSPACE = "workspace"


class VariableType(str, Enum):
    """Types of template variables for validation."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


# =============================================================================
# Variable Definition Schema
# =============================================================================


class TemplateVariable(BaseModel):
    """Definition of a template variable with metadata.

    Used to describe available variables in a template and
    validate inputs during rendering.
    """

    name: str = Field(..., min_length=1, max_length=100)
    """Variable name (used in {{name}} placeholders)."""

    type: VariableType = VariableType.STRING
    """Type of the variable for validation."""

    required: bool = True
    """Whether this variable must be provided during rendering."""

    default: Any | None = None
    """Default value if not provided (only used if not required)."""

    description: str | None = None
    """Human-readable description for documentation."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


# =============================================================================
# API Request Schemas
# =============================================================================


class CreateTemplateRequest(BaseSchema):
    """Request to create a new prompt template."""

    name: str = Field(..., min_length=1, max_length=255)
    """Display name for the template."""

    type: TemplateType
    """Type of template (system, step, synthesis, query)."""

    content: str = Field(..., min_length=1, max_length=100000)
    """Template content with {{variable}} placeholders."""

    description: str | None = Field(None, max_length=5000)
    """Optional description for the template."""

    variables: list[TemplateVariable] = Field(default_factory=list)
    """Variable definitions with metadata."""

    tags: list[str] = Field(default_factory=list, max_length=20)
    """Tags for filtering and organization."""

    visibility: TemplateVisibility = TemplateVisibility.PRIVATE
    """Visibility level (private or workspace)."""

    is_default: bool = False
    """Whether to set as default template for this type."""


class UpdateTemplateRequest(BaseSchema):
    """Request to update an existing template."""

    name: str | None = Field(None, min_length=1, max_length=255)
    """Updated display name."""

    content: str | None = Field(None, min_length=1, max_length=100000)
    """Updated template content."""

    description: str | None = Field(None, max_length=5000)
    """Updated description."""

    variables: list[TemplateVariable] | None = None
    """Updated variable definitions."""

    tags: list[str] | None = Field(None, max_length=20)
    """Updated tags."""

    visibility: TemplateVisibility | None = None
    """Updated visibility level."""

    is_default: bool | None = None
    """Whether to set as default template for this type."""


class RenderTemplateRequest(BaseSchema):
    """Request to render a template with variable values."""

    variables: dict[str, Any] = Field(default_factory=dict)
    """Variable name to value mapping."""


# =============================================================================
# API Response Schemas
# =============================================================================


class TemplateResponse(BaseSchema):
    """Response schema for a single prompt template."""

    id: UUID
    """Unique template identifier."""

    owner_id: str
    """ID of the user who created this template."""

    name: str
    """Display name for the template."""

    type: TemplateType
    """Type of template."""

    content: str
    """Template content with {{variable}} placeholders."""

    description: str | None = None
    """Optional description for the template."""

    variables: list[TemplateVariable]
    """Variable definitions with metadata."""

    tags: list[str]
    """Tags for filtering."""

    visibility: TemplateVisibility
    """Visibility level."""

    is_default: bool
    """Whether this is the default template for its type."""

    origin: str = "user"
    """Origin of the template (user, system, plugin)."""

    created_at: datetime
    """When the template was created."""

    updated_at: datetime
    """When the template was last modified."""

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        use_enum_values = True


class TemplateListResponse(BaseSchema):
    """Response schema for listing templates."""

    templates: list[TemplateResponse]
    """List of templates."""

    total: int
    """Total number of matching templates."""

    user_templates: int
    """Number of templates owned by the user."""

    workspace_templates: int
    """Number of workspace-visible templates from others."""


class RenderTemplateResponse(BaseSchema):
    """Response from rendering a template."""

    rendered_content: str
    """The template content with variables substituted."""

    missing_variables: list[str]
    """List of required variables that were not provided."""

    used_defaults: list[str]
    """List of variables that used default values."""
