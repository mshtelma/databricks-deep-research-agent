"""CustomAgent and AgentPresetStep SQLAlchemy models for custom agent configuration.

Allows users to create custom research agents with specific configurations:
- Custom system/synthesis prompt templates
- Source scope restrictions
- Preset research steps for manual workflows
- Default research depth and workflow mode

Part of US6 - Custom Agent Configurations (T074, T075).
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import BaseModel


class AgentVisibility(StrEnum):
    """Visibility levels for custom agents."""

    PRIVATE = "private"  # Only creator can see/use
    WORKSPACE = "workspace"  # All workspace users can see/use
    SYSTEM = "system"  # System-provided agents (read-only for users)


class AgentSourceScope(StrEnum):
    """Source scope options for custom agents."""

    ALL = "all"  # Use all available sources
    ENTERPRISE_ONLY = "enterprise_only"  # Only Databricks enterprise sources
    WEB_ONLY = "web_only"  # Only web search


class AgentWorkflowMode(StrEnum):
    """Workflow mode options for custom agents."""

    PLANNER = "planner"  # Use AI planner to generate steps
    MANUAL = "manual"  # Use preset steps only
    HYBRID = "hybrid"  # Preset steps + planner can add more


class AgentOutputFormat(StrEnum):
    """Output format options for custom agents."""

    MARKDOWN = "markdown"  # Standard markdown report
    JSON = "json"  # Structured JSON output


class AgentResearchDepth(StrEnum):
    """Research depth options for custom agents."""

    LIGHT = "light"  # Quick research (1-3 steps)
    MEDIUM = "medium"  # Standard research (3-5 steps)
    EXTENDED = "extended"  # Deep research (5-10 steps)


if TYPE_CHECKING:
    from deep_research.models.prompt_template import PromptTemplate


class CustomAgent(BaseModel):
    """User-defined custom research agent.

    Custom agents bundle together configuration options:
    - Prompt templates for system instructions and synthesis
    - Source scope restrictions
    - Preset research steps (for manual/hybrid modes)
    - Default research settings

    Attributes:
        owner_id: Databricks workspace user ID who created this agent.
        name: Display name for the agent (unique per owner).
        description: Human-readable description.
        avatar_url: Optional URL for agent avatar image.
        system_prompt_template_id: FK to template for system instructions.
        synthesis_template_id: FK to template for synthesis.
        source_scope: Default source scope (all, enterprise_only, web_only).
        enabled_sources: JSONB array of source names to enable.
        disabled_sources: JSONB array of source names to disable.
        use_planner: Whether to use AI planner for step generation.
        default_depth: Default research depth (light, medium, extended).
        default_mode: Default workflow mode (planner, manual, hybrid).
        enable_clarification: Whether to enable clarification questions.
        output_format: Default output format (markdown, json).
        output_schema: JSONB schema for structured JSON output.
        visibility: Who can see/use this agent (private, workspace, system).
    """

    __tablename__ = "custom_agents"

    # Owner identification (Databricks workspace user ID)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Agent name (unique per owner for clarity)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Description for UI and documentation
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Avatar URL for UI display
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Template references (nullable FKs to prompt_templates)
    system_prompt_template_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("prompt_templates.id", ondelete="SET NULL"),
        nullable=True,
    )
    synthesis_template_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("prompt_templates.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Source scope configuration
    source_scope: Mapped[str] = mapped_column(
        String(50),
        default=AgentSourceScope.ALL.value,
        server_default="all",
        nullable=False,
    )

    # Explicit source enable/disable lists (JSONB arrays)
    enabled_sources: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    disabled_sources: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        server_default="[]",
        nullable=False,
    )

    # Workflow configuration
    use_planner: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False,
    )

    default_depth: Mapped[str] = mapped_column(
        String(20),
        default=AgentResearchDepth.MEDIUM.value,
        server_default="medium",
        nullable=False,
    )

    default_mode: Mapped[str] = mapped_column(
        String(20),
        default=AgentWorkflowMode.PLANNER.value,
        server_default="planner",
        nullable=False,
    )

    enable_clarification: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False,
    )

    # Output configuration
    output_format: Mapped[str] = mapped_column(
        String(20),
        default=AgentOutputFormat.MARKDOWN.value,
        server_default="markdown",
        nullable=False,
    )

    output_schema: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Visibility level
    visibility: Mapped[str] = mapped_column(
        String(20),
        default=AgentVisibility.PRIVATE.value,
        server_default="private",
        nullable=False,
    )

    # Per-agent model tier overrides (009-custom-agent-config)
    # Maps tier name (e.g. "complex") to endpoint identifier (e.g. "databricks-haiku")
    model_overrides: Mapped[dict[str, str] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Per-agent domain filtering (009-custom-agent-config)
    domain_filter_mode: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
    include_domains: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    exclude_domains: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Per-source query configuration (overrides per enterprise source)
    source_query_configs: Mapped[dict[str, dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Plugin workflow reference (012-workflow-provider)
    workflow_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True, default=None,
    )

    # Relationships
    system_prompt_template: Mapped["PromptTemplate | None"] = relationship(
        "PromptTemplate",
        foreign_keys=[system_prompt_template_id],
        lazy="selectin",
    )
    synthesis_template: Mapped["PromptTemplate | None"] = relationship(
        "PromptTemplate",
        foreign_keys=[synthesis_template_id],
        lazy="selectin",
    )
    preset_steps: Mapped[list["AgentPresetStep"]] = relationship(
        "AgentPresetStep",
        back_populates="agent",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AgentPresetStep.order",
    )

    # Indexes
    __table_args__ = (
        # Fast lookup by owner
        Index("idx_custom_agents_owner", "owner_id"),
        # Fast lookup for workspace/system visible agents
        Index("idx_custom_agents_visibility", "visibility"),
        # Composite for listing accessible agents
        Index("idx_custom_agents_owner_visibility", "owner_id", "visibility"),
        # Unique name per owner to prevent confusion
        Index(
            "uq_custom_agents_owner_name",
            "owner_id",
            "name",
            unique=True,
        ),
    )

    @property
    def visibility_level(self) -> AgentVisibility:
        """Get visibility as enum."""
        return AgentVisibility(self.visibility)

    @property
    def scope(self) -> AgentSourceScope:
        """Get source scope as enum."""
        return AgentSourceScope(self.source_scope)

    @property
    def workflow_mode(self) -> AgentWorkflowMode:
        """Get workflow mode as enum."""
        return AgentWorkflowMode(self.default_mode)

    @property
    def research_depth(self) -> AgentResearchDepth:
        """Get research depth as enum."""
        return AgentResearchDepth(self.default_depth)

    @property
    def format(self) -> AgentOutputFormat:
        """Get output format as enum."""
        return AgentOutputFormat(self.output_format)

    @property
    def is_workspace_visible(self) -> bool:
        """Check if agent is visible to workspace users."""
        return self.visibility in (
            AgentVisibility.WORKSPACE.value,
            AgentVisibility.SYSTEM.value,
        )

    @property
    def is_system_agent(self) -> bool:
        """Check if this is a system-provided agent."""
        return self.visibility == AgentVisibility.SYSTEM.value

    def has_preset_steps(self) -> bool:
        """Check if agent has preset research steps configured."""
        return len(self.preset_steps) > 0

    def get_ordered_preset_steps(self) -> list["AgentPresetStep"]:
        """Get preset steps sorted by order."""
        return sorted(self.preset_steps, key=lambda s: s.order)


class AgentPresetStep(BaseModel):
    """Preset research step for a custom agent.

    Preset steps allow users to define specific research steps
    that execute when use_planner=False or in hybrid mode.

    Attributes:
        agent_id: FK to parent custom agent.
        title: Short title for the step (displayed in UI).
        description: Detailed description of what this step should accomplish.
        order: Execution order (1-based).
        is_required: Whether this step must be executed (vs. optional).
        source_hints: JSONB with hints for source selection.
        source_scope: Optional override for source scope in this step.
    """

    __tablename__ = "agent_preset_steps"

    # Parent agent reference
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("custom_agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Step title (short display name)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # Step description (detailed objective)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Execution order (1-based)
    order: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Whether this step is required
    is_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False,
    )

    # Source hints for this step (JSONB)
    # Example: {"preferred_sources": ["vector_search_1"], "search_queries": ["topic X"]}
    source_hints: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Optional source scope override for this step
    source_scope: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Relationship back to agent
    agent: Mapped["CustomAgent"] = relationship(
        "CustomAgent",
        back_populates="preset_steps",
    )

    # Indexes
    __table_args__ = (
        # Fast lookup by agent
        Index("idx_agent_preset_steps_agent", "agent_id"),
        # Ordering within agent
        Index("idx_agent_preset_steps_agent_order", "agent_id", "order"),
    )

    @property
    def step_scope(self) -> AgentSourceScope | None:
        """Get source scope as enum if set."""
        if self.source_scope:
            return AgentSourceScope(self.source_scope)
        return None

    def get_preferred_sources(self) -> list[str]:
        """Get list of preferred source names from hints."""
        if self.source_hints and "preferred_sources" in self.source_hints:
            result = self.source_hints.get("preferred_sources", [])
            return list(result) if result else []
        return []

    def get_search_queries(self) -> list[str]:
        """Get suggested search queries from hints."""
        if self.source_hints and "search_queries" in self.source_hints:
            result = self.source_hints.get("search_queries", [])
            return list(result) if result else []
        return []
