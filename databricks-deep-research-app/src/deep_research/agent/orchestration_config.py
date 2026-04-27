"""Orchestration configuration types and helpers.

Extracted from orchestrator.py to keep the public API surface thin.
All config types and config-mutation logic lives here; orchestrator.py
re-exports for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.agent.config import get_coordinator_config, get_planner_config
from deep_research.core.app_config import DomainFilterConfig, DomainFilterMode
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.agent.state import ResearchState
    from deep_research.models.custom_agent import CustomAgent
    from deep_research.schemas.streaming import StreamEvent

logger = get_logger(__name__)


# =============================================================================
# Config types
# =============================================================================


def get_default_orchestration_config() -> OrchestrationConfig:
    """Create OrchestrationConfig with defaults from central app config."""
    planner_config = get_planner_config()
    coordinator_config = get_coordinator_config()
    return OrchestrationConfig(
        max_plan_iterations=planner_config.max_plan_iterations,
        enable_clarification=coordinator_config.enable_clarification,
    )


@dataclass
class OrchestrationConfig:
    """Configuration for research orchestration.

    Defaults are loaded from central app.yaml config.
    Override by passing explicit values to constructor.
    """

    max_plan_iterations: int = 3
    max_steps_per_plan: int = 10
    enable_background_investigation: bool = True
    enable_clarification: bool = True
    timeout_seconds: int = 300  # 5 minutes
    # Query mode configuration (tiered query modes feature)
    query_mode: str = "deep_research"  # simple, web_search, deep_research
    research_depth: str = "auto"  # auto, light, medium, extended (deep_research only)
    system_instructions: str | None = None  # User's custom system instructions
    # Persistence context (for claim/citation storage)
    message_id: UUID | None = None  # Agent message ID for claims
    research_session_id: UUID | None = None  # Research session ID for sources
    # Draft chat support - True if chat doesn't exist in DB yet
    is_draft: bool = False
    # Citation verification toggle - when False, use classical synthesis
    verify_sources: bool = True
    # Session pre-created - True if JobManager already created the session
    # When True, orchestrator skips session creation to avoid duplicate key error
    session_pre_created: bool = False
    # Structured output configuration
    output_format: str = "markdown"  # "markdown" or "json"
    output_schema: type | dict[str, Any] | None = None  # Pydantic model or JSON schema for output

    # Synthesis mode and post-verification configuration
    synthesis_mode: str = "simple"  # "simple" or "reclaim"
    enable_post_verification: bool = False  # Run stages 4-6 after simple generation

    # Custom prompts for structured synthesis (plugin can override)
    structured_system_prompt: str | None = None
    structured_user_prompt: str | None = None

    # =========================================================================
    # Manual Workflow Mode (007-enterprise-data-sources, T052)
    # =========================================================================
    workflow_mode: str = "planner"  # WorkflowMode: planner, manual, hybrid
    manual_steps: list[Any] | None = None  # ManualStepDefinition list

    # =========================================================================
    # Plan Review Configuration (007-enterprise-data-sources, US12, T040)
    # =========================================================================
    enable_plan_review: bool = False
    """If True, pause after plan creation and yield PlanReviewEvent."""

    require_plan_approval: bool = False
    """If True, do not auto-proceed; wait indefinitely for user response."""

    plan_review_timeout_seconds: int = 300
    """Timeout in seconds before auto-proceeding (when require_plan_approval=False)."""

    # =========================================================================
    # Source Scope Configuration (007-enterprise-data-sources, US12, T041)
    # =========================================================================
    source_scope: str | None = None
    """SourceScope value: 'enterprise_only', 'web_only', or 'all'."""

    enabled_sources: list[str] | None = None
    """Explicit whitelist of source names to enable."""

    disabled_sources: list[str] | None = None
    """List of source names to disable."""

    # =========================================================================
    # OBO Authentication (007-enterprise-data-sources, Phase 2)
    # =========================================================================
    user_token: str | None = None
    """User OAuth token for OBO authentication with enterprise data sources."""

    approval_broker: Any | None = None
    """HITL approval broker instance. Set from app.state.approval_broker in FastAPI handlers."""

    # =========================================================================
    # File Upload and Custom Agent Support
    # =========================================================================
    file_ids: list[str] | None = None
    """Uploaded file IDs to include in research context."""

    agent_id: str | None = None
    """Custom agent ID to use for this research job."""

    # =========================================================================
    # Per-Agent Overrides (009-custom-agent-config)
    # =========================================================================
    model_overrides: dict[str, str] | None = None
    """Per-tier model endpoint overrides from custom agent."""

    domain_filter: Any | None = None  # DomainFilterConfig
    """Per-agent domain filter overrides from custom agent."""

    # =========================================================================
    # Research Session Timeout (H1)
    # =========================================================================
    research_timeout_seconds: int = 1800  # 30 minutes default
    """Maximum time in seconds before a research job is terminated."""

    # =========================================================================
    # Plugin Workflow Reference (012-workflow-provider)
    # =========================================================================
    workflow_ref: str | None = None
    """When set, resolves a named workflow from plugins instead of
    building one via config_translator. None = existing flow unchanged."""


@dataclass
class OrchestrationResult:
    """Result from orchestration."""

    state: ResearchState
    events: list[StreamEvent] = field(default_factory=list)
    total_duration_ms: float = 0
    steps_executed: int = 0
    steps_skipped: int = 0


# =============================================================================
# Config mutation helpers
# =============================================================================


def _convert_preset_steps_to_manual_steps(
    preset_steps: list[Any],
) -> list[Any]:
    """Convert AgentPresetStep models to ManualStepDefinition-like objects.

    Used when applying a custom agent's preset steps to the orchestration config.

    Args:
        preset_steps: List of AgentPresetStep model instances.

    Returns:
        List of objects compatible with ManualStepDefinition expectations.
    """
    from deep_research.schemas.manual_step import (
        ManualStepDefinition,
        SourceConstraint,
        StepSourceAttachment,
    )

    manual_steps: list[ManualStepDefinition] = []

    for idx, step in enumerate(sorted(preset_steps, key=lambda s: s.order)):
        # Build source attachments from hints
        sources: list[StepSourceAttachment] = []
        if step.source_hints:
            preferred = step.source_hints.get("preferred_sources", [])
            for source_name in preferred:
                sources.append(
                    StepSourceAttachment(
                        source_name=source_name,
                        source_type="web_search",  # Default, will be resolved
                        priority=1,
                    )
                )

        # Build constraint if step has scope override
        constraint = None
        if step.source_scope:
            constraint = SourceConstraint(
                allowed_types=None,
                allowed_sources=None,
            )

        manual_step = ManualStepDefinition(
            id=str(step.id)[:8],
            title=step.title,
            objective=step.description or step.title,
            sources=sources,
            constraints=constraint,
            order=idx + 1,
            is_required=step.is_required,
            source_scope=getattr(step, "source_scope", None),
        )
        manual_steps.append(manual_step)

    return manual_steps


def apply_custom_agent_to_config(
    config: OrchestrationConfig,
    agent: CustomAgent,
    query_overrides: dict[str, Any] | None = None,
) -> OrchestrationConfig:
    """Apply custom agent settings to an orchestration config.

    This helper merges a CustomAgent's configuration into an OrchestrationConfig,
    respecting per-query overrides when provided.

    The merge priority is (lowest to highest):
    1. Agent defaults
    2. Query-level overrides

    Applied settings include:
    - Source scope (enterprise_only, web_only, all)
    - Enabled/disabled sources
    - Research depth
    - Workflow mode
    - Clarification toggle
    - Output format and schema
    - Preset steps (when use_planner=False)

    Args:
        config: Base orchestration config to modify.
        agent: CustomAgent model instance with settings.
        query_overrides: Optional per-query overrides that take precedence.

    Returns:
        Modified OrchestrationConfig with agent settings applied.

    Part of US6 - Custom Agent Configurations (T080).
    """
    overrides = query_overrides or {}

    # Source scope configuration
    # Agent default, then query override
    if "source_scope" in overrides:
        config.source_scope = overrides["source_scope"]
    elif agent.source_scope:
        config.source_scope = agent.source_scope

    # Enabled/disabled sources
    # Note: Source existence can't be validated here — enterprise tools aren't
    # loaded until stream_research(). Invalid source references are silently
    # skipped by the tool factory at runtime (T050).
    if "enabled_sources" in overrides:
        config.enabled_sources = overrides["enabled_sources"]
    elif agent.enabled_sources:
        config.enabled_sources = agent.enabled_sources
        if len(agent.enabled_sources) > 0:
            logger.info(
                "AGENT_ENABLED_SOURCES_SET",
                extra={
                    "agent_id": str(agent.id),
                    "source_count": len(agent.enabled_sources),
                },
            )

    if "disabled_sources" in overrides:
        config.disabled_sources = overrides["disabled_sources"]
    elif agent.disabled_sources:
        config.disabled_sources = agent.disabled_sources

    # Research depth
    config.research_depth = overrides.get("research_depth", agent.default_depth)

    # Workflow mode
    config.workflow_mode = overrides.get("workflow_mode", agent.default_mode)

    # Clarification toggle
    config.enable_clarification = overrides.get(
        "enable_clarification", agent.enable_clarification
    )

    # Output format and schema
    config.output_format = overrides.get("output_format", agent.output_format)

    if "output_schema" in overrides:
        config.output_schema = overrides["output_schema"]
    elif agent.output_schema:
        config.output_schema = agent.output_schema

    # Handle preset steps in MANUAL or HYBRID mode
    if config.workflow_mode in ("manual", "hybrid") and agent.preset_steps:
        config.manual_steps = _convert_preset_steps_to_manual_steps(agent.preset_steps)

    # Model overrides (009-custom-agent-config)
    if "model_overrides" not in overrides and agent.model_overrides:
        from deep_research.core.app_config import get_app_config

        app_config = get_app_config()
        validated: dict[str, str] = {}
        for tier_name, endpoint_id in agent.model_overrides.items():
            if not endpoint_id or not endpoint_id.strip():
                continue
            endpoint_id = endpoint_id.strip()
            validated[tier_name] = endpoint_id
            if endpoint_id not in app_config.endpoints:
                logger.info(
                    "AGENT_MODEL_OVERRIDE_DIRECT_ENDPOINT",
                    extra={
                        "agent_id": str(agent.id),
                        "tier": tier_name,
                        "endpoint": endpoint_id,
                        "note": "not in YAML, will be used as direct endpoint identifier",
                    },
                )
        if validated:
            config.model_overrides = validated
            logger.info(
                "AGENT_MODEL_OVERRIDE_APPLIED",
                extra={
                    "agent_id": str(agent.id),
                    "override_count": len(validated),
                    "tiers": list(validated.keys()),
                },
            )

    # Domain filter (009-custom-agent-config)
    if "domain_filter" not in overrides and agent.domain_filter_mode:
        try:
            domain_filter = DomainFilterConfig(
                mode=DomainFilterMode(agent.domain_filter_mode),
                include_domains=agent.include_domains or [],
                exclude_domains=agent.exclude_domains or [],
            )
            config.domain_filter = domain_filter
            logger.info(
                "AGENT_DOMAIN_FILTER_APPLIED",
                extra={
                    "agent_id": str(agent.id),
                    "mode": agent.domain_filter_mode,
                    "include_count": len(agent.include_domains or []),
                    "exclude_count": len(agent.exclude_domains or []),
                },
            )
        except ValueError:
            logger.warning(
                "AGENT_DOMAIN_FILTER_INVALID",
                extra={
                    "agent_id": str(agent.id),
                    "mode": agent.domain_filter_mode,
                },
            )

    # Workflow ref (012-workflow-provider)
    if "workflow_ref" not in overrides and agent.workflow_ref:
        config.workflow_ref = agent.workflow_ref

    # System instructions from template
    if agent.system_prompt_template and agent.system_prompt_template.content:
        config.system_instructions = agent.system_prompt_template.content
    if agent.synthesis_template and agent.synthesis_template.content:
        config.structured_system_prompt = agent.synthesis_template.content

    logger.info(
        "Applied custom agent config",
        extra={
            "agent_id": str(agent.id),
            "agent_name": agent.name,
            "source_scope": config.source_scope,
            "workflow_mode": config.workflow_mode,
            "research_depth": config.research_depth,
            "output_format": config.output_format,
            "has_preset_steps": len(agent.preset_steps) > 0 if agent.preset_steps else False,
            "has_model_overrides": config.model_overrides is not None,
            "has_domain_filter": config.domain_filter is not None,
        },
    )

    return config
