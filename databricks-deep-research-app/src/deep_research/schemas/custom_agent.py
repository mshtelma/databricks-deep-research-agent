"""Custom agent schemas for the custom agent configuration API.

This module defines Pydantic models for:
- Custom agent visibility, source scope, workflow mode, output format
- Request/Response schemas for custom agent API endpoints
- Preset step schemas

Part of US6 - Custom Agent Configurations (T079).
"""

import re
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from deep_research.schemas.common import BaseSchema

_DOMAIN_PATTERN_RE = re.compile(r"^[a-zA-Z0-9.*\-]+$")


def _validate_domain_filter_fields(
    mode: str | None,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
) -> None:
    """Validate domain filter field consistency.

    Args:
        mode: Domain filter mode (include/exclude/both) or None.
        include_domains: Whitelist patterns.
        exclude_domains: Blacklist patterns.

    Raises:
        ValueError: If fields are inconsistent.
    """
    if mode is None:
        return

    valid_modes = {"include", "exclude", "both"}
    if mode not in valid_modes:
        msg = f"domain_filter_mode must be one of {valid_modes}, got '{mode}'"
        raise ValueError(msg)

    if mode in ("include", "both") and not include_domains:
        msg = "include_domains must be non-empty when domain_filter_mode is 'include' or 'both'"
        raise ValueError(msg)

    if mode in ("exclude", "both") and not exclude_domains:
        msg = "exclude_domains must be non-empty when domain_filter_mode is 'exclude' or 'both'"
        raise ValueError(msg)

    # Validate pattern format
    for patterns, label in [
        (include_domains or [], "include_domains"),
        (exclude_domains or [], "exclude_domains"),
    ]:
        for pattern in patterns:
            if not _DOMAIN_PATTERN_RE.match(pattern):
                msg = f"Invalid domain pattern in {label}: '{pattern}'"
                raise ValueError(msg)


class AgentVisibility(StrEnum):
    """Visibility levels for custom agents."""

    PRIVATE = "private"
    WORKSPACE = "workspace"
    SYSTEM = "system"


class AgentSourceScope(StrEnum):
    """Source scope options for custom agents."""

    ALL = "all"
    ENTERPRISE_ONLY = "enterprise_only"
    WEB_ONLY = "web_only"


class AgentWorkflowMode(StrEnum):
    """Workflow mode options for custom agents."""

    PLANNER = "planner"
    MANUAL = "manual"
    HYBRID = "hybrid"


class AgentOutputFormat(StrEnum):
    """Output format options for custom agents."""

    MARKDOWN = "markdown"
    JSON = "json"


class AgentResearchDepth(StrEnum):
    """Research depth options for custom agents."""

    LIGHT = "light"
    MEDIUM = "medium"
    EXTENDED = "extended"


# =============================================================================
# Preset Step Schemas
# =============================================================================


class SourceHints(BaseModel):
    """Source hints for a preset step.

    Provides guidance to the researcher on which sources to use.
    """

    preferred_sources: list[str] = Field(
        default_factory=list,
        description="List of preferred source names to use for this step",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Suggested search queries for this step",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional filters for vector search sources",
    )


class CreatePresetStepRequest(BaseSchema):
    """Request to create a preset step within an agent."""

    title: str = Field(..., min_length=1, max_length=255)
    """Short title for the step."""

    description: str | None = Field(None, max_length=5000)
    """Detailed description of what this step should accomplish."""

    order: int = Field(default=1, ge=1)
    """Execution order (1-based)."""

    is_required: bool = True
    """Whether this step must be executed."""

    source_hints: SourceHints | None = None
    """Hints for source selection."""

    source_scope: AgentSourceScope | None = None
    """Optional scope override for this step."""


class UpdatePresetStepRequest(BaseSchema):
    """Request to update a preset step."""

    title: str | None = Field(None, min_length=1, max_length=255)
    """Updated title."""

    description: str | None = Field(None, max_length=5000)
    """Updated description."""

    order: int | None = Field(None, ge=1)
    """Updated execution order."""

    is_required: bool | None = None
    """Updated required flag."""

    source_hints: SourceHints | None = None
    """Updated source hints."""

    source_scope: AgentSourceScope | None = None
    """Updated source scope override."""


class PresetStepResponse(BaseSchema):
    """Response schema for a preset step."""

    id: UUID
    """Unique step identifier."""

    agent_id: UUID
    """Parent agent ID."""

    title: str
    """Step title."""

    description: str | None
    """Step description."""

    order: int
    """Execution order."""

    is_required: bool
    """Whether this step is required."""

    source_hints: dict[str, Any] | None
    """Source hints for this step."""

    source_scope: AgentSourceScope | None
    """Source scope override."""

    created_at: datetime
    """When the step was created."""

    updated_at: datetime
    """When the step was last modified."""

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        use_enum_values = True


# =============================================================================
# Custom Agent Request Schemas
# =============================================================================


class CreateCustomAgentRequest(BaseSchema):
    """Request to create a new custom agent."""

    name: str = Field(..., min_length=1, max_length=255)
    """Display name for the agent."""

    description: str | None = Field(None, max_length=5000)
    """Human-readable description."""

    avatar_url: str | None = Field(None, max_length=500)
    """URL for agent avatar image."""

    system_prompt_template_id: UUID | None = None
    """ID of template for system instructions."""

    synthesis_template_id: UUID | None = None
    """ID of template for synthesis."""

    source_scope: AgentSourceScope = AgentSourceScope.ALL
    """Default source scope."""

    enabled_sources: list[str] | None = None
    """Explicit list of source names to enable."""

    disabled_sources: list[str] = Field(default_factory=list)
    """List of source names to disable."""

    use_planner: bool = True
    """Whether to use AI planner for step generation."""

    default_depth: AgentResearchDepth = AgentResearchDepth.MEDIUM
    """Default research depth."""

    default_mode: AgentWorkflowMode = AgentWorkflowMode.PLANNER
    """Default workflow mode."""

    enable_clarification: bool = True
    """Whether to enable clarification questions."""

    output_format: AgentOutputFormat = AgentOutputFormat.MARKDOWN
    """Default output format."""

    output_schema: dict[str, Any] | None = None
    """JSON schema for structured output (when output_format=json)."""

    visibility: AgentVisibility = AgentVisibility.PRIVATE
    """Visibility level."""

    # Per-agent model tier overrides (009-custom-agent-config)
    model_overrides: dict[str, str] | None = None
    """Mapping of tier name to endpoint identifier override."""

    # Per-agent domain filtering (009-custom-agent-config)
    domain_filter_mode: str | None = None
    """Domain filter mode: include, exclude, or both."""

    include_domains: list[str] | None = None
    """Domain whitelist patterns (e.g. ['*.gov', '*.edu'])."""

    exclude_domains: list[str] | None = None
    """Domain blacklist patterns."""

    # Per-source query configuration (009-custom-agent-config M5)
    source_query_configs: dict[str, dict[str, Any]] | None = None
    """Per-source query configs. Keys are source names, values are query config overrides."""

    # Plugin workflow reference (012-workflow-provider)
    workflow_ref: str | None = None
    """Named workflow to resolve from plugins instead of config_translator."""

    # Optionally create preset steps inline
    preset_steps: list[CreatePresetStepRequest] = Field(default_factory=list)
    """Preset steps to create with the agent."""

    @model_validator(mode="after")
    def _validate_domain_filter(self) -> "CreateCustomAgentRequest":
        """Validate domain filter consistency."""
        _validate_domain_filter_fields(
            self.domain_filter_mode, self.include_domains, self.exclude_domains
        )
        return self


class UpdateCustomAgentRequest(BaseSchema):
    """Request to update an existing custom agent."""

    name: str | None = Field(None, min_length=1, max_length=255)
    """Updated display name."""

    description: str | None = Field(None, max_length=5000)
    """Updated description."""

    avatar_url: str | None = Field(None, max_length=500)
    """Updated avatar URL."""

    system_prompt_template_id: UUID | None = None
    """Updated system prompt template ID."""

    synthesis_template_id: UUID | None = None
    """Updated synthesis template ID."""

    source_scope: AgentSourceScope | None = None
    """Updated source scope."""

    enabled_sources: list[str] | None = None
    """Updated enabled sources list."""

    disabled_sources: list[str] | None = None
    """Updated disabled sources list."""

    use_planner: bool | None = None
    """Updated use_planner flag."""

    default_depth: AgentResearchDepth | None = None
    """Updated default depth."""

    default_mode: AgentWorkflowMode | None = None
    """Updated default mode."""

    enable_clarification: bool | None = None
    """Updated clarification flag."""

    output_format: AgentOutputFormat | None = None
    """Updated output format."""

    output_schema: dict[str, Any] | None = None
    """Updated JSON schema."""

    visibility: AgentVisibility | None = None
    """Updated visibility."""

    # Per-agent model tier overrides (009-custom-agent-config)
    model_overrides: dict[str, str] | None = None
    """Updated model tier overrides."""

    domain_filter_mode: str | None = None
    """Updated domain filter mode."""

    include_domains: list[str] | None = None
    """Updated include domain patterns."""

    exclude_domains: list[str] | None = None
    """Updated exclude domain patterns."""

    # Per-source query configuration (009-custom-agent-config M5)
    source_query_configs: dict[str, dict[str, Any]] | None = None
    """Updated per-source query configs."""

    # Plugin workflow reference (012-workflow-provider)
    workflow_ref: str | None = None
    """Updated workflow ref."""

    @model_validator(mode="after")
    def _validate_domain_filter(self) -> "UpdateCustomAgentRequest":
        """Validate domain filter consistency."""
        _validate_domain_filter_fields(
            self.domain_filter_mode, self.include_domains, self.exclude_domains
        )
        return self


# =============================================================================
# Custom Agent Response Schemas
# =============================================================================


class CustomAgentResponse(BaseSchema):
    """Response schema for a custom agent."""

    id: UUID
    """Unique agent identifier."""

    owner_id: str
    """ID of the user who created this agent."""

    name: str
    """Display name."""

    description: str | None
    """Description."""

    avatar_url: str | None
    """Avatar URL."""

    system_prompt_template_id: UUID | None
    """System prompt template ID."""

    synthesis_template_id: UUID | None
    """Synthesis template ID."""

    source_scope: AgentSourceScope
    """Source scope."""

    enabled_sources: list[str] | None
    """Enabled sources list."""

    disabled_sources: list[str]
    """Disabled sources list."""

    use_planner: bool
    """Whether to use AI planner."""

    default_depth: AgentResearchDepth
    """Default research depth."""

    default_mode: AgentWorkflowMode
    """Default workflow mode."""

    enable_clarification: bool
    """Whether clarification is enabled."""

    output_format: AgentOutputFormat
    """Output format."""

    output_schema: dict[str, Any] | None
    """JSON schema for structured output."""

    visibility: AgentVisibility
    """Visibility level."""

    # Per-agent model tier overrides (009-custom-agent-config)
    model_overrides: dict[str, str] | None = None
    """Model tier overrides."""

    domain_filter_mode: str | None = None
    """Domain filter mode."""

    include_domains: list[str] | None = None
    """Domain whitelist patterns."""

    exclude_domains: list[str] | None = None
    """Domain blacklist patterns."""

    # Per-source query configuration (009-custom-agent-config M5)
    source_query_configs: dict[str, dict[str, Any]] | None = None
    """Per-source query configs."""

    # Plugin workflow reference (012-workflow-provider)
    workflow_ref: str | None = None
    """Named workflow to resolve from plugins."""

    model_override_warnings: list[dict[str, str]] = Field(default_factory=list)
    """Warnings for model overrides referencing unknown endpoints."""

    preset_steps: list[PresetStepResponse]
    """Preset steps for this agent."""

    created_at: datetime
    """When the agent was created."""

    updated_at: datetime
    """When the agent was last modified."""

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        use_enum_values = True


class CustomAgentSummary(BaseSchema):
    """Summary response for listing agents."""

    id: UUID
    """Unique agent identifier."""

    owner_id: str
    """ID of the user who created this agent."""

    name: str
    """Display name."""

    description: str | None
    """Description."""

    avatar_url: str | None
    """Avatar URL."""

    visibility: AgentVisibility
    """Visibility level."""

    source_scope: AgentSourceScope
    """Source scope."""

    default_mode: AgentWorkflowMode
    """Default workflow mode."""

    default_depth: AgentResearchDepth
    """Default research depth."""

    preset_step_count: int
    """Number of preset steps."""

    has_model_overrides: bool = False
    """Whether the agent has model tier overrides configured."""

    has_domain_filter: bool = False
    """Whether the agent has domain filtering configured."""

    has_source_config: bool = False
    """Whether the agent defines source scope or enabled sources."""

    capabilities: list[str] = Field(default_factory=list)
    """Computed capability tags for display."""

    created_at: datetime
    """When the agent was created."""

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        use_enum_values = True


class CustomAgentListResponse(BaseSchema):
    """Response schema for listing custom agents."""

    agents: list[CustomAgentSummary]
    """List of agent summaries."""

    total: int
    """Total number of matching agents."""

    user_agents: int
    """Number of agents owned by the user."""

    workspace_agents: int
    """Number of workspace-visible agents from others."""

    system_agents: int
    """Number of system-provided agents."""


# =============================================================================
# Agent Resolution Request
# =============================================================================


class ResolveAgentRequest(BaseSchema):
    """Request to resolve an agent by ID or name."""

    agent_id: UUID | None = None
    """Agent ID to resolve."""

    agent_name: str | None = None
    """Agent name to resolve (owner's agents + workspace + system)."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
