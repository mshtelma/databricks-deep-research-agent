"""PromptTemplate SQLAlchemy model for customizable prompt templates.

Allows users to create and manage custom prompt templates for various
agent components (system prompts, step prompts, synthesis prompts, queries).

Part of US5 - Custom Prompt Template Library (T065).
"""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import Base, UUIDMixin


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

    PRIVATE = "private"  # Only creator can see/use
    WORKSPACE = "workspace"  # All workspace users can see/use


class PromptTemplate(Base, UUIDMixin):
    """User-defined prompt template for research components.

    Stores customizable templates with variable substitution support.
    Templates can be marked as defaults for their type and shared
    with workspace users.

    Attributes:
        owner_id: Databricks workspace user ID who created this template.
        name: Display name for the template (unique per owner).
        type: Type of template (system, step, synthesis, query).
        content: Template content with {{variable}} placeholders.
        variables: JSONB metadata about template variables.
        tags: JSONB array of tags for filtering.
        visibility: Who can see/use this template.
        is_default: Whether this is the default template for its type.
        created_at: Timestamp when template was created.
        updated_at: Timestamp when template was last modified.
    """

    __tablename__ = "prompt_templates"

    # Owner identification (Databricks workspace user ID)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Template name (unique per owner for clarity)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Template type
    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Template content with {{variable}} placeholders
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional description
    description: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    # Variable metadata (JSONB array of variable definitions)
    # Each variable: {name: str, type: str, required: bool, default: Any, description: str}
    variables: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        server_default="[]",
        nullable=False,
    )

    # Tags for filtering and organization
    tags: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        server_default="[]",
        nullable=False,
    )

    # Visibility level
    visibility: Mapped[str] = mapped_column(
        String(20),
        default=TemplateVisibility.PRIVATE.value,
        server_default="private",
        nullable=False,
    )

    # Default template flag
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Indexes
    __table_args__ = (
        # Fast lookup by owner
        Index("idx_prompt_templates_owner", "owner_id"),
        # Fast lookup by type
        Index("idx_prompt_templates_type", "type"),
        # Fast lookup for workspace-visible templates
        Index("idx_prompt_templates_visibility", "visibility"),
        # Composite for listing accessible templates
        Index("idx_prompt_templates_owner_visibility", "owner_id", "visibility"),
        # Fast lookup for default templates
        Index("idx_prompt_templates_type_default", "type", "is_default"),
        # Unique name per owner to prevent confusion
        Index(
            "uq_prompt_templates_owner_name",
            "owner_id",
            "name",
            unique=True,
        ),
    )

    @property
    def template_type(self) -> TemplateType:
        """Get type as enum."""
        return TemplateType(self.type)

    @property
    def visibility_level(self) -> TemplateVisibility:
        """Get visibility as enum."""
        return TemplateVisibility(self.visibility)

    @property
    def is_workspace_visible(self) -> bool:
        """Check if template is visible to workspace users."""
        return self.visibility == TemplateVisibility.WORKSPACE.value

    def get_variable_names(self) -> list[str]:
        """Get list of variable names from metadata."""
        return [var.get("name", "") for var in (self.variables or []) if var.get("name")]

    def get_required_variables(self) -> list[str]:
        """Get list of required variable names."""
        return [
            var.get("name", "")
            for var in (self.variables or [])
            if var.get("required", False) and var.get("name")
        ]
