"""Research request schemas for API input validation.

Defines the schema for initiating research requests with
source scope controls and plan review options.

Part of 007-enterprise-data-sources feature (US12, T041).
"""

from pydantic import Field

from deep_research.models.research_session import ResearchDepth
from deep_research.schemas.common import BaseSchema
from deep_research.schemas.source_scope import SourceScope


class ResearchRequest(BaseSchema):
    """Request schema for initiating a research session.

    Extends the basic query with source scope controls and
    plan review options for enterprise data source integration.
    """

    content: str = Field(..., min_length=1, description="The user's research query")

    research_depth: ResearchDepth = Field(
        default=ResearchDepth.AUTO,
        description="Research depth level: auto, light, medium, or extended",
    )

    # =========================================================================
    # Source Scope Configuration (007-enterprise-data-sources)
    # =========================================================================

    source_scope: SourceScope | None = Field(
        default=None,
        description="Scope of data sources to use (enterprise_only, web_only, all)",
    )

    enabled_sources: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of source names to enable. "
            "When set, only these sources will be used (whitelist)."
        ),
    )

    disabled_sources: list[str] = Field(
        default_factory=list,
        description=(
            "List of source names to disable. "
            "These sources will be excluded even if otherwise enabled."
        ),
    )

    # =========================================================================
    # Plan Review Configuration (007-enterprise-data-sources)
    # =========================================================================

    enable_plan_review: bool = Field(
        default=False,
        description=(
            "If True, pause after plan creation and emit a PlanReviewEvent. "
            "The user can then approve, modify, or reject the plan."
        ),
    )

    require_plan_approval: bool = Field(
        default=False,
        description=(
            "If True, research will not proceed until user explicitly approves the plan. "
            "If False, auto-proceed after timeout. Only relevant if enable_plan_review=True."
        ),
    )

    plan_review_timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description=(
            "Timeout in seconds before auto-proceeding with the plan. "
            "Only applies when enable_plan_review=True and require_plan_approval=False."
        ),
    )


class ResearchRequestWithChat(ResearchRequest):
    """Research request that includes chat context.

    Used when initiating research within an existing chat conversation.
    """

    include_chat_history: bool = Field(
        default=True,
        description="Whether to include previous chat messages as context",
    )

    max_history_messages: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum number of previous messages to include as context",
    )
