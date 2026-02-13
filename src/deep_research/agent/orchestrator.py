"""Multi-agent orchestrator - coordinates the 5-agent research workflow."""

import asyncio
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.agent.config import (
    get_coordinator_config,
    get_planner_config,
    get_query_mode_config,
    get_researcher_config_for_depth,
)
from deep_research.agent.nodes.background import run_background_investigator
from deep_research.agent.nodes.citation_synthesizer import (
    run_citation_synthesizer,
    stream_synthesis_with_citations,
)
from deep_research.agent.nodes.coordinator import handle_simple_query, run_coordinator
from deep_research.agent.nodes.planner import run_planner
from deep_research.agent.nodes.react_researcher import ReactResearchEvent, run_react_researcher
from deep_research.agent.nodes.reflector import run_reflector
from deep_research.agent.nodes.researcher import run_researcher
from deep_research.agent.nodes.synthesizer import (
    post_verify_structured_output,
    run_structured_synthesizer,
    run_synthesizer,
    stream_synthesis,
)
from deep_research.agent.state import (
    Plan,
    PlanStep,
    ReflectionDecision,
    ReflectionResult,
    ResearchState,
    StepStatus,
    StepType,
)
from deep_research.agent.tools.web_crawler import WebCrawler
from deep_research.core.app_config import DomainFilterConfig, DomainFilterMode, ResearcherMode

if TYPE_CHECKING:
    from deep_research.models.custom_agent import CustomAgent
from deep_research.core.exceptions import StructuredSynthesisError
from deep_research.core.logging_utils import (
    get_logger,
    log_agent_phase,
    log_agent_transition,
    truncate,
)
from deep_research.core.tracing import log_research_config, safe_mlflow_run, safe_tool_span, safe_update_trace
from deep_research.schemas.research import PlanStepSummary
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    CitationCorrectedEvent,
    ClaimVerifiedEvent,
    CustomPhaseModeStartedEvent,
    NumericClaimDetectedEvent,
    PersistenceCompletedEvent,
    PhaseCompletedEvent,
    PhaseErrorEvent,
    PhaseSkippedEvent,
    PhaseStartedEvent,
    PlanCreatedEvent,
    PlanForReview,
    PlanReviewEvent,
    PlanReviewTimeoutEvent,
    PlanStepForReview,
    ReflectionDecisionEvent,
    ResearchCompletedEvent,
    ResearchStartedEvent,
    StepCompletedEvent,
    StepStartedEvent,
    StreamErrorEvent,
    StreamEvent,
    SynthesisProgressEvent,
    SynthesisStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolSkippedEvent,
    VerificationSummaryEvent,
)
from deep_research.services.llm.client import LLMClient
from deep_research.services.research_event_buffer import EventBuffer
from deep_research.services.search.brave import BraveSearchClient

# Import database session type for persistence (optional dependency)
try:
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:
    AsyncSession = None  # type: ignore[misc, assignment]

# Import PluginManager and PipelineCustomization for custom phase mode
try:
    from deep_research.plugins.manager import PluginManager
except ImportError:
    PluginManager = None  # type: ignore[misc, assignment]

try:
    from deep_research.agent.pipeline.protocols import PipelineCustomization
except ImportError:
    PipelineCustomization = None  # type: ignore[misc, assignment]

logger = get_logger(__name__)


def _convert_react_event(react_event: ReactResearchEvent) -> StreamEvent | None:
    """Convert internal ReactResearchEvent to a public StreamEvent.

    Shared by all researcher modes (ReAct, classic, background)
    to avoid duplicating event conversion logic.
    """
    if react_event.event_type == "tool_call":
        return ToolCallEvent(
            tool_name=react_event.data.get("tool", ""),
            tool_args=react_event.data.get("args", {}),
            call_number=react_event.data.get("call_number", 0),
            source_type=react_event.data.get("source_type"),
        )
    elif react_event.event_type == "tool_result":
        return ToolResultEvent(
            tool_name=react_event.data.get("tool", ""),
            result_preview=react_event.data.get("result_preview", "")[:200],
            sources_crawled=react_event.data.get("high_quality_count", 0),
            sources_added=react_event.data.get("sources_added", 0),
            source_type=react_event.data.get("source_type"),
        )
    return None


def _get_schema_name(output_schema: type | dict | None) -> str | None:
    """Get a display name for the output schema.

    Handles both Pydantic class types (with __name__) and JSON schema dicts.

    Args:
        output_schema: Either a class type or a JSON schema dict

    Returns:
        Schema name string, or None if no schema
    """
    if output_schema is None:
        return None
    if isinstance(output_schema, dict):
        return output_schema.get("title", "dict_schema")
    return getattr(output_schema, "__name__", str(type(output_schema)))


def _get_tool_name(tool: Any) -> str | None:
    """Safely extract tool definition name from a ResearchTool-like object."""
    definition = getattr(tool, "definition", None)
    name = getattr(definition, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _append_unique_tools(target: list[Any], incoming: list[Any]) -> None:
    """Append tools to target, deduplicating by tool definition name."""
    known_names = {_get_tool_name(tool) for tool in target}
    for tool in incoming:
        tool_name = _get_tool_name(tool)
        if tool_name in known_names:
            continue
        target.append(tool)
        known_names.add(tool_name)


def _has_non_file_tools(tools: list[Any]) -> bool:
    """Return True if there is at least one tool other than file_search."""
    for tool in tools:
        if _get_tool_name(tool) != "file_search":
            return True
    return False


def _get_default_orchestration_config() -> "OrchestrationConfig":
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
    output_schema: type | None = None  # Pydantic model for JSON output

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


def _convert_manual_steps_to_plan(
    manual_steps: list[Any],
    query: str,
) -> Plan:
    """Convert manual step definitions to a Plan object.

    Used when workflow_mode is MANUAL or HYBRID to create a plan
    from user-defined steps instead of using the AI planner.

    Args:
        manual_steps: List of ManualStepDefinition objects.
        query: The original user query.

    Returns:
        Plan object with steps converted from manual definitions.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    plan_steps: list[PlanStep] = []

    for manual_step in manual_steps:
        # Extract fields from ManualStepDefinition
        step_id = getattr(manual_step, "id", str(uuid4())[:8])
        title = getattr(manual_step, "title", "Manual Step")
        objective = getattr(manual_step, "objective", "")

        # Determine step type based on sources
        sources = getattr(manual_step, "sources", [])
        needs_search = any(
            getattr(s, "source_type", "") in ("web_search", "vector_search", "genie")
            for s in sources
        )

        plan_step = PlanStep(
            id=step_id,
            title=title,
            description=objective,
            step_type=StepType.RESEARCH if needs_search else StepType.ANALYSIS,
            needs_search=needs_search,
            status=StepStatus.PENDING,
        )
        plan_steps.append(plan_step)

    return Plan(
        id=str(uuid4())[:8],
        title=f"Manual Research Plan: {query[:50]}...",
        thought="User-defined manual research workflow",
        steps=plan_steps,
        has_enough_context=False,
        iteration=1,
        created_at=datetime.now(UTC),
    )


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
    config: "OrchestrationConfig",
    agent: "CustomAgent",
    query_overrides: dict[str, Any] | None = None,
) -> "OrchestrationConfig":
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


@dataclass
class OrchestrationResult:
    """Result from orchestration."""

    state: ResearchState
    events: list[StreamEvent] = field(default_factory=list)
    total_duration_ms: float = 0
    steps_executed: int = 0
    steps_skipped: int = 0


async def run_research(
    query: str,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    crawler: WebCrawler,
    conversation_history: list[dict[str, str]] | None = None,
    session_id: UUID | None = None,
    user_id: str | None = None,
    chat_id: str | None = None,
    config: OrchestrationConfig | None = None,
) -> OrchestrationResult:
    """Run the complete multi-agent research workflow.

    Args:
        query: User's research query.
        llm: LLM client for completions.
        brave_client: Brave Search client for web searches.
        crawler: Web crawler for fetching page content.
        conversation_history: Previous messages for context.
        session_id: Optional session ID for tracking.
        user_id: Optional user ID for MLflow trace grouping.
        chat_id: Optional chat ID for MLflow trace session grouping.
        config: Orchestration configuration.

    Returns:
        OrchestrationResult with final state and events.
    """
    config = config or _get_default_orchestration_config()
    start_time = time.perf_counter()

    # Initialize state
    state = ResearchState(
        query=query,
        conversation_history=conversation_history or [],
        max_plan_iterations=config.max_plan_iterations,
        enable_clarification=config.enable_clarification,
        query_mode=config.query_mode,
        research_depth=config.research_depth,
        system_instructions=config.system_instructions,
        enable_citation_verification=config.verify_sources,
        output_format=config.output_format,
        output_schema=config.output_schema,
        synthesis_mode=config.synthesis_mode,
        enable_post_verification=config.enable_post_verification,
        structured_system_prompt=config.structured_system_prompt,
        structured_user_prompt=config.structured_user_prompt,
        # Workflow mode configuration (007-enterprise-data-sources)
        workflow_mode=config.workflow_mode,
        manual_steps=config.manual_steps or [],
    )
    if session_id:
        state.session_id = session_id

    # Wire source scope from OrchestrationConfig to ResearchState (008-data-source-selection)
    if config.source_scope:
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

        try:
            scope_enum = SourceScope(config.source_scope)
            state.source_scope_config = SourceScopeConfig(
                scope=scope_enum,
                enabled_sources=config.enabled_sources,
                disabled_sources=config.disabled_sources or [],
            )
            logger.info(
                "ORCHESTRATION_SOURCE_SCOPE_SET",
                scope=config.source_scope,
                enabled_count=len(config.enabled_sources or []),
                disabled_count=len(config.disabled_sources or []),
            )
        except ValueError as e:
            logger.warning(
                "ORCHESTRATION_INVALID_SOURCE_SCOPE",
                scope=config.source_scope,
                error=str(e),
            )
            # Default to ALL scope on invalid input
            state.source_scope_config = SourceScopeConfig(scope=SourceScope.ALL)

    # Wire user_token for OBO authentication (007-enterprise-data-sources Phase 2)
    if config.user_token:
        state.user_token = config.user_token

    # Wire file_ids and agent_id from config to state
    if config.file_ids:
        state.file_ids = config.file_ids
        logger.warning(
            "ORCHESTRATION_FILE_SEARCH_UNAVAILABLE",
            reason="run_research has no database session for file_search",
            file_count=len(config.file_ids),
        )
    if config.agent_id:
        state.agent_id = config.agent_id
    if config.model_overrides:
        state.model_overrides = config.model_overrides
    if config.domain_filter:
        state.domain_filter = config.domain_filter

    # Create enterprise tools from discovery cache (run_research has no db param)
    if (
        not state.enterprise_tools
        and state.source_scope_config
        and state.source_scope_config.enabled_sources
        and state.is_enterprise_search_allowed()
    ):
        try:
            from deep_research.agent.tools.factory import create_tools_from_discovered_sources
            from deep_research.services.discovery_cache import get_discovery_cache

            cache = get_discovery_cache()
            cached_sources = await cache.get(user_id=user_id)

            if cached_sources:
                enabled_ids = set(state.source_scope_config.enabled_sources)
                matching = [s for s in cached_sources if s.source_id in enabled_ids]

                if matching:
                    type_filtered = [
                        s for s in matching
                        if state.source_scope_config.is_type_enabled(s.source_type)
                    ]

                    if type_filtered:
                        discovery_tools = await create_tools_from_discovered_sources(
                            type_filtered
                        )
                        state.enterprise_tools = discovery_tools
                        logger.info(
                            "ORCHESTRATION_ENTERPRISE_TOOLS_FROM_DISCOVERY",
                            tool_count=len(discovery_tools),
                            tool_names=[t.definition.name for t in discovery_tools],
                            source_ids=[s.source_id for s in type_filtered],
                        )
                else:
                    logger.warning(
                        "ORCHESTRATION_NO_MATCHING_DISCOVERY_SOURCES",
                        enabled_ids=list(enabled_ids)[:5],
                        cached_count=len(cached_sources),
                    )
            else:
                logger.warning(
                    "ORCHESTRATION_DISCOVERY_CACHE_EMPTY",
                    user_id=user_id,
                )
        except Exception as e:
            logger.error(
                "ORCHESTRATION_DISCOVERY_TOOLS_FAILED",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

    # Handle manual workflow mode: convert manual steps to plan
    if state.is_manual_mode() and state.manual_steps:
        state.current_plan = _convert_manual_steps_to_plan(
            state.manual_steps, query
        )
        # Also set source constraints from manual steps
        for manual_step in state.manual_steps:
            step_id = getattr(manual_step, "id", "")
            constraint = getattr(manual_step, "constraints", None)
            if step_id and constraint:
                state.set_source_constraint(step_id, constraint)

    events: list[StreamEvent] = []
    steps_executed = 0
    steps_skipped = 0

    # Log orchestration start
    log_agent_phase(
        logger,
        "ORCHESTRATION_START",
        {
            "session_id": str(state.session_id)[:8],
            "query": truncate(query, 100),
            "max_iterations": config.max_plan_iterations,
        },
    )

    try:
        # Create MLflow run to associate trace with params
        with safe_mlflow_run(f"research_{str(state.session_id)[:8]}"):
            async with safe_tool_span("research_orchestration", "CHAIN", {
                "research.session_id": str(state.session_id),
                "research.query": truncate(query, 200),
                "research.max_iterations": config.max_plan_iterations,
                "research.enable_background": config.enable_background_investigation,
                "research.enable_clarification": config.enable_clarification,
            }) as root_span:

                # Group traces by user and chat session for MLflow trace correlation
                if user_id or chat_id:
                    trace_metadata: dict[str, str] = {}
                    if user_id:
                        trace_metadata["mlflow.trace.user"] = user_id
                    if chat_id:
                        trace_metadata["mlflow.trace.session"] = chat_id
                    safe_update_trace(trace_metadata)

                # Phase 1: Coordinator - Query Classification
                log_agent_transition(logger, from_agent=None, to_agent="coordinator")
                log_agent_phase(logger, "COORDINATOR_START")
                events.append(_agent_started("coordinator", "simple"))
                agent_start = time.perf_counter()

                state = await run_coordinator(state, llm)

                coordinator_ms = (time.perf_counter() - agent_start) * 1000
                log_agent_phase(
                    logger,
                    "COORDINATOR_COMPLETE",
                    {
                        "is_simple": state.is_simple_query,
                        "complexity": state.query_classification.complexity if state.query_classification else "unknown",
                        "duration_ms": round(coordinator_ms, 1),
                    },
                )
                events.append(_agent_completed("coordinator", agent_start))

                # Log research configuration to MLflow run (after coordinator resolves depth)
                log_research_config(depth=state.resolve_depth())

                # Handle simple queries directly
                if state.is_simple_query and state.direct_response:
                    logger.info(
                        "SIMPLE_QUERY_HANDLED",
                        response_len=len(state.direct_response),
                    )
                    state.complete(state.direct_response)
                    return OrchestrationResult(
                        state=state,
                        events=events,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                    )

                # Phase 2: Background Investigation (optional)
                if config.enable_background_investigation:
                    log_agent_transition(logger, from_agent="coordinator", to_agent="background_investigator")
                    log_agent_phase(logger, "BACKGROUND_START")
                    events.append(_agent_started("background_investigator", "simple"))
                    agent_start = time.perf_counter()

                    async for react_event in run_background_investigator(state, llm, brave_client):
                        stream_evt = _convert_react_event(react_event)
                        if stream_evt:
                            events.append(stream_evt)

                    background_ms = (time.perf_counter() - agent_start) * 1000
                    log_agent_phase(
                        logger,
                        "BACKGROUND_COMPLETE",
                        {
                            "context_len": len(state.background_investigation_results) if state.background_investigation_results else 0,
                            "duration_ms": round(background_ms, 1),
                        },
                    )
                    events.append(_agent_completed("background_investigator", agent_start))

                # Phase 3: Planning and Research Loop
                log_agent_phase(logger, "RESEARCH_LOOP_START")
                while state.plan_iterations < config.max_plan_iterations:
                    if state.is_cancelled:
                        logger.info("RESEARCH_CANCELLED")
                        break

                    # Plan
                    prev_agent = "background_investigator" if config.enable_background_investigation else "coordinator"
                    log_agent_transition(
                        logger,
                        from_agent=prev_agent,
                        to_agent="planner",
                        reason=f"iteration {state.plan_iterations + 1}",
                    )
                    log_agent_phase(
                        logger,
                        "PLANNER_START",
                        {"iteration": state.plan_iterations + 1},
                    )
                    events.append(_agent_started("planner", "analytical"))
                    agent_start = time.perf_counter()

                    state = await run_planner(state, llm)

                    planner_ms = (time.perf_counter() - agent_start) * 1000
                    if state.current_plan:
                        log_agent_phase(
                            logger,
                            "PLANNER_COMPLETE",
                            {
                                "plan_title": truncate(state.current_plan.title, 60),
                                "steps": len(state.current_plan.steps),
                                "has_enough_context": state.current_plan.has_enough_context,
                                "duration_ms": round(planner_ms, 1),
                            },
                        )
                    events.append(_agent_completed("planner", agent_start))

                    if state.current_plan:
                        events.append(_plan_created(state))

                        # Skip research if planner says we have enough context
                        if state.current_plan.has_enough_context:
                            logger.info("SKIPPING_RESEARCH", reason="has_enough_context")
                            break

                        # Execute steps with reflection after each
                        total_steps = len(state.current_plan.steps)
                        while state.has_more_steps() and not state.is_cancelled:
                            step = state.get_current_step()
                            if not step:
                                break

                            log_agent_phase(
                                logger,
                                "STEP_START",
                                {
                                    "step": f"{state.current_step_index + 1}/{total_steps}",
                                    "title": truncate(step.title, 60),
                                    "type": step.step_type.value,
                                },
                            )

                            # Emit step started
                            events.append(_step_started(state))

                            # Research step - switch between modes based on depth config
                            log_agent_transition(
                                logger,
                                from_agent="planner",
                                to_agent="researcher",
                                reason=f"step {state.current_step_index + 1}",
                            )
                            events.append(_agent_started("researcher", "analytical"))
                            agent_start = time.perf_counter()

                            # Get researcher mode for current depth
                            depth = state.resolve_depth()
                            researcher_config = get_researcher_config_for_depth(depth)

                            if researcher_config.mode == ResearcherMode.REACT:
                                # ReAct mode: LLM controls the research loop
                                researcher_gen = run_react_researcher(
                                    state, llm, crawler, brave_client
                                )
                            else:
                                # Classic mode: single-pass fixed searches/crawls
                                researcher_gen = run_researcher(
                                    state, llm, crawler, brave_client
                                )

                            async for react_event in researcher_gen:
                                stream_evt = _convert_react_event(react_event)
                                if stream_evt:
                                    events.append(stream_evt)
                                elif react_event.event_type == "research_complete":
                                    logger.info(
                                        "REACT_RESEARCH_COMPLETE",
                                        reason=react_event.data.get("reason", ""),
                                        tool_calls=react_event.data.get("tool_calls", 0),
                                        high_quality=react_event.data.get("high_quality_sources", 0),
                                    )

                            researcher_ms = (time.perf_counter() - agent_start) * 1000
                            log_agent_phase(
                                logger,
                                "STEP_COMPLETE",
                                {
                                    "step": f"{state.current_step_index + 1}/{total_steps}",
                                    "sources_found": len(state.sources),
                                    "observation_len": len(state.last_observation) if state.last_observation else 0,
                                    "duration_ms": round(researcher_ms, 1),
                                },
                            )
                            events.append(_agent_completed("researcher", agent_start))
                            steps_executed += 1

                            # Emit step completed
                            events.append(_step_completed(state))

                            # Reflect
                            log_agent_transition(logger, from_agent="researcher", to_agent="reflector")
                            events.append(_agent_started("reflector", "simple"))
                            agent_start = time.perf_counter()

                            state = await run_reflector(state, llm)

                            reflector_ms = (time.perf_counter() - agent_start) * 1000
                            events.append(_agent_completed("reflector", agent_start))

                            if state.last_reflection:
                                log_agent_phase(
                                    logger,
                                    "REFLECTION_DECISION",
                                    {
                                        "decision": state.last_reflection.decision.value,
                                        "reasoning": truncate(state.last_reflection.reasoning, 80),
                                        "duration_ms": round(reflector_ms, 1),
                                    },
                                )
                                events.append(_reflection_decision(state))

                                if state.last_reflection.decision == ReflectionDecision.COMPLETE:
                                    # Check minimum steps enforcement
                                    min_steps = state.get_min_steps()
                                    completed = len(state.get_completed_steps())

                                    if completed < min_steps:
                                        # Override early completion - minimum steps not reached
                                        logger.warning(
                                            "OVERRIDE_EARLY_COMPLETE",
                                            completed=completed,
                                            minimum=min_steps,
                                            reason="Minimum steps not reached",
                                        )
                                        state.last_reflection = ReflectionResult(
                                            decision=ReflectionDecision.CONTINUE,
                                            reasoning=f"Override: {completed}/{min_steps} minimum steps completed",
                                        )
                                    else:
                                        # Allow completion - mark remaining steps as skipped
                                        while state.has_more_steps():
                                            state.advance_step()
                                            steps_skipped += 1
                                        logger.info("EARLY_COMPLETION", steps_skipped=steps_skipped)
                                        break

                                if state.last_reflection.decision == ReflectionDecision.ADJUST:
                                    preserved_count = len(state.get_completed_steps())
                                    logger.info(
                                        "ADJUSTING_PLAN",
                                        reason="reflection_decision",
                                        preserving_completed_steps=preserved_count,
                                    )
                                    # Go back to planning (completed steps will be preserved)
                                    break

                            # Advance to next step
                            state.advance_step()

                        # Check if we should replan or finish
                        if state.last_reflection and state.last_reflection.decision == ReflectionDecision.ADJUST:
                            continue  # Back to planning loop
                        break  # Done with research

                # Phase 4: Synthesis
                log_agent_transition(logger, from_agent="reflector", to_agent="synthesizer")
                log_agent_phase(
                    logger,
                    "SYNTHESIS_START",
                    {
                        "observations": len(state.all_observations),
                        "sources": len(state.sources),
                    },
                )
                events.append(
                    SynthesisStartedEvent(
                        total_observations=len(state.all_observations),
                        total_sources=len(state.sources),
                    )
                )

                events.append(_agent_started("synthesizer", "complex"))
                agent_start = time.perf_counter()

                # Use structured synthesizer if JSON output requested
                if state.output_format == "json" and state.output_schema:
                    state = await run_structured_synthesizer(state, llm)
                    # Run post-verification if enabled (requires verify_sources=True)
                    if state.enable_post_verification and state.enable_citation_verification:
                        state = await post_verify_structured_output(state, llm)
                # Use citation-aware synthesizer if enabled
                elif state.enable_citation_verification:
                    state = await run_citation_synthesizer(state, llm)
                else:
                    state = await run_synthesizer(state, llm)

                synthesis_ms = (time.perf_counter() - agent_start) * 1000
                log_agent_phase(
                    logger,
                    "SYNTHESIS_COMPLETE",
                    {
                        "report_len": len(state.final_report) if state.final_report else 0,
                        "duration_ms": round(synthesis_ms, 1),
                    },
                )
                events.append(_agent_completed("synthesizer", agent_start))

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(
            "ORCHESTRATION_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        events.append(
            StreamErrorEvent(
                error_code="ORCHESTRATION_ERROR",
                error_message=str(e),
                recoverable=False,
                stack_trace=tb,
                error_type=type(e).__name__,
            )
        )

    total_duration_ms = (time.perf_counter() - start_time) * 1000

    # Log final orchestration summary
    log_agent_phase(
        logger,
        "ORCHESTRATION_COMPLETE",
        {
            "session_id": str(state.session_id)[:8],
            "steps_executed": steps_executed,
            "steps_skipped": steps_skipped,
            "plan_iterations": state.plan_iterations,
            "total_sources": len(state.sources),
            "total_duration_ms": round(total_duration_ms, 1),
        },
    )

    events.append(
        ResearchCompletedEvent(
            session_id=state.session_id,
            total_steps_executed=steps_executed,
            total_steps_skipped=steps_skipped,
            plan_iterations=state.plan_iterations,
            total_duration_ms=int(total_duration_ms),
            final_report=state.final_report,
            structured_output=(
                state.final_report_structured.model_dump()
                if state.final_report_structured else None
            ),
        )
    )

    return OrchestrationResult(
        state=state,
        events=events,
        total_duration_ms=total_duration_ms,
        steps_executed=steps_executed,
        steps_skipped=steps_skipped,
    )


# --- File content loading thresholds ---
_FILE_INLINE_THRESHOLD = 15_000    # ~4K tokens -> full text in prompts
_FILE_HYBRID_THRESHOLD = 50_000    # ~13K tokens -> preview + file_search
# Above hybrid -> retrieval only (metadata + file_search)

# Preview chunk count for hybrid strategy
_HYBRID_PREVIEW_CHUNKS = 3


def _determine_strategy(char_count: int) -> str:
    """Determine file loading strategy from character count."""
    if char_count <= _FILE_INLINE_THRESHOLD:
        return "inline"
    elif char_count <= _FILE_HYBRID_THRESHOLD:
        return "hybrid"
    else:
        return "retrieval"


def _build_prompt_content(
    strategy: str,
    chunks: list[Any],
    char_count: int,
    chunk_count: int,
) -> str:
    """Build the content string for prompt injection based on strategy.

    Args:
        strategy: One of "inline", "hybrid", "retrieval".
        chunks: Loaded chunk objects (all for inline, first N for hybrid, empty for retrieval).
        char_count: Total extracted character count.
        chunk_count: Total chunk count (from uploaded_file.chunk_count).

    Returns:
        Content string appropriate for the strategy.
    """
    if strategy == "inline":
        return "\n\n".join(c.content for c in chunks)
    elif strategy == "hybrid":
        preview = "\n\n".join(c.content for c in chunks)
        preview_len = len(preview)
        remaining = char_count - preview_len
        if remaining > 0:
            return (
                preview
                + f"\n\n[... ~{remaining:,} more chars — "
                f"use file_search for details.]"
            )
        # All content fit in preview chunks — treat as effectively inline
        return preview
    else:  # retrieval
        return (
            f"[Large file: ~{char_count:,} chars across {chunk_count} chunks. "
            f"Use file_search to query specific sections.]"
        )


async def _load_file_contents(
    state: ResearchState,
    db: "AsyncSession",
    user_id: str,
) -> None:
    """Load file contents into state and register SourceInfo entries.

    Uses a two-path strategy for efficiency:
    - Fast path: When total_extracted_chars is available in metadata,
      decides strategy BEFORE loading chunks. Loads only what's needed:
      all chunks (inline), first 3 (hybrid), or none (retrieval).
    - Fallback path: For legacy files without total_extracted_chars,
      loads all chunks to compute char_count (same as previous behavior).
    """
    if not state.file_ids:
        return

    from uuid import UUID as _UUID

    from deep_research.agent.state import SourceInfo
    from deep_research.services.file_upload_service import FileUploadService

    service = FileUploadService(db)

    for file_id_str in state.file_ids:
        try:
            file_id = _UUID(file_id_str)
            uploaded_file = await service.get_for_user(file_id, user_id)
            if not uploaded_file or not uploaded_file.is_ready:
                logger.warning(
                    "FILE_CONTENT_LOAD_SKIP",
                    file_id=file_id_str,
                    reason="not_found_or_not_ready",
                )
                continue

            # --- Determine strategy and load chunks ---
            precomputed_chars = uploaded_file.total_extracted_chars

            if precomputed_chars is not None:
                # FAST PATH: strategy known before loading any chunks
                strategy = _determine_strategy(precomputed_chars)
                char_count = precomputed_chars

                if strategy == "inline":
                    chunks = await service.get_file_chunks(file_id, limit=500)
                elif strategy == "hybrid":
                    chunks = await service.get_file_chunks(
                        file_id, limit=_HYBRID_PREVIEW_CHUNKS
                    )
                else:  # retrieval
                    chunks = []

                if strategy == "inline" and not chunks:
                    # File marked ready but has no chunks — skip
                    logger.warning(
                        "FILE_CONTENT_LOAD_SKIP",
                        file_id=file_id_str,
                        reason="no_chunks_despite_ready",
                    )
                    continue

            else:
                # FALLBACK PATH: legacy file without precomputed chars
                chunks = await service.get_file_chunks(file_id, limit=500)
                if not chunks:
                    logger.warning(
                        "FILE_CONTENT_LOAD_SKIP",
                        file_id=file_id_str,
                        reason="no_chunks",
                    )
                    continue

                full_content = "\n\n".join(c.content for c in chunks)
                char_count = len(full_content)
                strategy = _determine_strategy(char_count)

                logger.info(
                    "FILE_CONTENT_STRATEGY_FALLBACK",
                    file_id=file_id_str,
                    char_count=char_count,
                    strategy=strategy,
                )

            # --- Build prompt content ---
            content_for_prompt = _build_prompt_content(
                strategy, chunks, char_count, uploaded_file.chunk_count,
            )

            state.file_contents.append({
                "file_id": file_id_str,
                "filename": uploaded_file.filename,
                "file_type": uploaded_file.file_type,
                "file_size": uploaded_file.file_size,
                "content": content_for_prompt,
                "strategy": strategy,
                "char_count": char_count,
            })

            # --- Register SourceInfo for citation tracking ---
            if strategy == "inline":
                source_content = content_for_prompt
                snippet = content_for_prompt[:300] if content_for_prompt else None
            elif strategy == "hybrid":
                source_content = content_for_prompt
                snippet = content_for_prompt[:300] if content_for_prompt else None
            else:
                source_content = None
                snippet = None

            state.add_source(
                SourceInfo(
                    url=f"uploaded-file://{file_id_str}",
                    title=uploaded_file.filename,
                    snippet=snippet,
                    content=source_content[:50_000] if source_content else None,
                    content_type="uploaded_file",
                )
            )

            logger.info(
                "FILE_CONTENT_LOADED",
                file_id=file_id_str,
                filename=uploaded_file.filename,
                file_size_bytes=uploaded_file.file_size,
                total_extracted_chars=char_count,
                strategy=strategy,
                chunks_loaded=len(chunks),
                fast_path=precomputed_chars is not None,
            )

        except Exception as e:
            logger.error("FILE_CONTENT_LOAD_ERROR", file_id=file_id_str, error=str(e))


async def stream_research(
    query: str,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    crawler: WebCrawler,
    conversation_history: list[dict[str, str]] | None = None,
    session_id: UUID | None = None,
    user_id: str | None = None,
    chat_id: str | None = None,
    config: OrchestrationConfig | None = None,
    db: "AsyncSession | None" = None,
    plugin_manager: "PluginManager | None" = None,
    plugin_data: dict[str, Any] | None = None,
) -> AsyncGenerator[StreamEvent | str, None]:
    """Stream the research workflow with real-time events.

    Now supports custom phase mode when a plugin disables the planner
    and provides custom phases via PhaseProvider protocol.

    Args:
        query: User's research query.
        llm: LLM client for completions.
        brave_client: Brave Search client for web searches.
        crawler: Web crawler for fetching page content.
        conversation_history: Previous messages for context.
        session_id: Optional session ID.
        user_id: Optional user ID for MLflow trace grouping.
        chat_id: Optional chat ID for MLflow trace session grouping.
        config: Orchestration configuration.
        db: Optional database session for persisting claims/citations.
        plugin_manager: Optional PluginManager for custom phase mode.
        plugin_data: Optional structured context data (account_name, company_name, etc.)
            to pass to custom phases. When provided, bypasses query extraction.

    Yields:
        StreamEvent objects and synthesis content chunks.
    """
    config = config or _get_default_orchestration_config()
    start_time = time.perf_counter()

    # Initialize state
    state = ResearchState(
        query=query,
        conversation_history=conversation_history or [],
        max_plan_iterations=config.max_plan_iterations,
        enable_clarification=config.enable_clarification,
        query_mode=config.query_mode,
        research_depth=config.research_depth,
        system_instructions=config.system_instructions,
        enable_citation_verification=config.verify_sources,
        output_format=config.output_format,
        output_schema=config.output_schema,
        synthesis_mode=config.synthesis_mode,
        enable_post_verification=config.enable_post_verification,
        structured_system_prompt=config.structured_system_prompt,
        structured_user_prompt=config.structured_user_prompt,
    )
    if session_id:
        state.session_id = session_id

    # Wire source scope from OrchestrationConfig to ResearchState (008-data-source-selection)
    if config.source_scope:
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

        try:
            scope_enum = SourceScope(config.source_scope)
            state.source_scope_config = SourceScopeConfig(
                scope=scope_enum,
                enabled_sources=config.enabled_sources,
                disabled_sources=config.disabled_sources or [],
            )
            logger.info(
                "ORCHESTRATION_SOURCE_SCOPE_SET",
                scope=config.source_scope,
                enabled_count=len(config.enabled_sources or []),
                disabled_count=len(config.disabled_sources or []),
            )
        except ValueError as e:
            logger.warning(
                "ORCHESTRATION_INVALID_SOURCE_SCOPE",
                scope=config.source_scope,
                error=str(e),
            )
            # Default to ALL scope on invalid input
            state.source_scope_config = SourceScopeConfig(scope=SourceScope.ALL)

    # Wire user_token for OBO authentication (007-enterprise-data-sources Phase 2)
    if config.user_token:
        state.user_token = config.user_token

    # Wire file_ids and agent_id from config to state
    if config.file_ids:
        state.file_ids = config.file_ids
        if db is not None and user_id:
            try:
                from deep_research.agent.tools.file_search import create_file_search_tool

                file_search_tool = create_file_search_tool(
                    session=db,
                    owner_id=user_id,
                    file_ids=config.file_ids,
                )
                _append_unique_tools(state.enterprise_tools, [file_search_tool])
                logger.info(
                    "ORCHESTRATION_FILE_SEARCH_TOOL_ATTACHED",
                    file_count=len(config.file_ids),
                )
            except Exception as e:
                logger.error(
                    "ORCHESTRATION_FILE_SEARCH_TOOL_FAILED",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
        else:
            logger.warning(
                "ORCHESTRATION_FILE_SEARCH_TOOL_SKIPPED",
                reason="missing_db_or_user_id",
                has_db=db is not None,
                has_user_id=bool(user_id),
                file_count=len(config.file_ids),
            )

    elif db is not None and user_id and chat_id:
        # AUTO-DISCOVERY: Look up files for this chat when frontend didn't
        # pass file_ids (e.g., component re-mount lost the state).
        try:
            from uuid import UUID as _UUID

            from deep_research.services.file_upload_service import FileUploadService

            discovery_service = FileUploadService(db)
            chat_files, _ = await discovery_service.get_session_files(
                user_id, _UUID(chat_id), limit=20
            )
            ready_files = [f for f in chat_files if f.is_ready]

            if ready_files:
                state.file_ids = [str(f.id) for f in ready_files]
                logger.info(
                    "ORCHESTRATION_FILE_IDS_AUTO_DISCOVERED",
                    file_count=len(state.file_ids),
                    chat_id=chat_id,
                )

                # Create file_search tool (same as explicit path)
                try:
                    from deep_research.agent.tools.file_search import create_file_search_tool

                    file_search_tool = create_file_search_tool(
                        session=db,
                        owner_id=user_id,
                        file_ids=state.file_ids,
                    )
                    _append_unique_tools(state.enterprise_tools, [file_search_tool])
                    logger.info(
                        "ORCHESTRATION_FILE_SEARCH_TOOL_ATTACHED",
                        file_count=len(state.file_ids),
                        source="auto_discovery",
                    )
                except Exception as e:
                    logger.error(
                        "ORCHESTRATION_FILE_SEARCH_TOOL_FAILED",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
        except Exception as e:
            logger.warning(
                "ORCHESTRATION_FILE_DISCOVERY_FAILED",
                error=str(e),
                chat_id=chat_id,
            )

    if config.agent_id:
        state.agent_id = config.agent_id
    if config.model_overrides:
        state.model_overrides = config.model_overrides
    if config.domain_filter:
        state.domain_filter = config.domain_filter

    # Load file contents for inline injection and citation tracking
    if state.file_ids and db is not None and user_id:
        try:
            await _load_file_contents(state, db, user_id)
            if state.file_contents:
                strategies = [fc["strategy"] for fc in state.file_contents]
                logger.info(
                    "ORCHESTRATION_FILE_CONTENTS_LOADED",
                    file_count=len(state.file_contents),
                    strategies=strategies,
                )
        except Exception as e:
            logger.error("ORCHESTRATION_FILE_CONTENT_LOAD_FAILED", error=str(e))

    # Optimization: remove file_search when all files are inline
    # (their full content is already in every prompt)
    if state.file_contents and state.enterprise_tools:
        all_inline = all(fc.get("strategy") == "inline" for fc in state.file_contents)
        if all_inline:
            state.enterprise_tools = [
                t for t in state.enterprise_tools
                if getattr(getattr(t, "definition", None), "name", None) != "file_search"
            ]
            logger.info(
                "ORCHESTRATION_FILE_SEARCH_REMOVED_ALL_INLINE",
                inline_count=len(state.file_contents),
            )

    # Load enterprise tools from user data sources (007-enterprise-data-sources Phase 2)
    # Only load if enterprise search is allowed and we have db and user_id
    if state.is_enterprise_search_allowed() and db is not None and user_id:
        try:
            from deep_research.agent.tools.factory import get_enabled_tools_for_user

            enterprise_tools = await get_enabled_tools_for_user(
                user_id=user_id,
                user_token=state.user_token,
                session=db,
            )
            _append_unique_tools(state.enterprise_tools, enterprise_tools)
            logger.info(
                "ORCHESTRATION_ENTERPRISE_TOOLS_LOADED",
                loaded_count=len(enterprise_tools),
                total_count=len(state.enterprise_tools),
                tool_names=[t.definition.name for t in state.enterprise_tools],
            )
        except Exception as e:
            logger.error(
                "ORCHESTRATION_ENTERPRISE_TOOLS_FAILED",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Continue without enterprise tools - not critical

    # Fallback: create tools from discovered sources if DB-based loading returned empty.
    # This handles the case where user selected sources from discovery UI
    # but hasn't saved them as UserDataSource records in the DB.
    if (
        not _has_non_file_tools(state.enterprise_tools)
        and state.source_scope_config
        and state.source_scope_config.enabled_sources
        and state.is_enterprise_search_allowed()
    ):
        try:
            from deep_research.agent.tools.factory import create_tools_from_discovered_sources
            from deep_research.services.discovery_cache import get_discovery_cache

            cache = get_discovery_cache()
            cached_sources = await cache.get(user_id=user_id)

            if cached_sources:
                enabled_ids = set(state.source_scope_config.enabled_sources)
                matching = [s for s in cached_sources if s.source_id in enabled_ids]

                if matching:
                    # Filter by source-type toggles (e.g., enable_vector_search=False)
                    type_filtered = [
                        s for s in matching
                        if state.source_scope_config.is_type_enabled(s.source_type)
                    ]

                    if type_filtered:
                        discovery_tools = await create_tools_from_discovered_sources(
                            type_filtered
                        )
                        _append_unique_tools(state.enterprise_tools, discovery_tools)
                        logger.info(
                            "ORCHESTRATION_ENTERPRISE_TOOLS_FROM_DISCOVERY",
                            loaded_count=len(discovery_tools),
                            total_count=len(state.enterprise_tools),
                            tool_names=[t.definition.name for t in state.enterprise_tools],
                            source_ids=[s.source_id for s in type_filtered],
                        )
                else:
                    logger.warning(
                        "ORCHESTRATION_NO_MATCHING_DISCOVERY_SOURCES",
                        enabled_ids=list(enabled_ids)[:5],
                        cached_count=len(cached_sources),
                    )
            else:
                logger.warning(
                    "ORCHESTRATION_DISCOVERY_CACHE_EMPTY",
                    user_id=user_id,
                )
        except Exception as e:
            logger.error(
                "ORCHESTRATION_DISCOVERY_TOOLS_FAILED",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

    # Last resort: create tools directly from source IDs (no cache/DB needed).
    # The source_id format (e.g. "assistant:endpoint_name") encodes enough
    # information to construct the tool without any API calls.
    if (
        not _has_non_file_tools(state.enterprise_tools)
        and state.source_scope_config
        and state.source_scope_config.enabled_sources
        and state.is_enterprise_search_allowed()
    ):
        try:
            from deep_research.agent.tools.factory import create_tools_from_source_ids

            direct_tools = create_tools_from_source_ids(
                state.source_scope_config.enabled_sources
            )
            _append_unique_tools(state.enterprise_tools, direct_tools)
            if direct_tools:
                logger.info(
                    "ORCHESTRATION_ENTERPRISE_TOOLS_FROM_SOURCE_IDS",
                    loaded_count=len(direct_tools),
                    total_count=len(state.enterprise_tools),
                    tool_names=[t.definition.name for t in direct_tools],
                    source_ids=state.source_scope_config.enabled_sources[:5],
                )
        except Exception as e:
            logger.error(
                "ORCHESTRATION_SOURCE_ID_TOOLS_FAILED",
                error=str(e)[:200],
                error_type=type(e).__name__,
            )

    # DIAGNOSTIC: Log state creation for structured output debugging
    logger.info(
        "ORCHESTRATION_STATE_CREATED",
        output_format=state.output_format,
        output_schema=_get_schema_name(state.output_schema),
        structured_system_prompt_len=len(state.structured_system_prompt) if state.structured_system_prompt else 0,
        structured_user_prompt_len=len(state.structured_user_prompt) if state.structured_user_prompt else 0,
        enable_citation_verification=state.enable_citation_verification,
        enable_post_verification=state.enable_post_verification,
    )

    # Emit lifecycle hook: synthesis_config (if plugin manager available)
    if plugin_manager and state.session_id:
        try:
            from deep_research.plugins.lifecycle import EventEmitter

            emitter = EventEmitter(plugin_manager)
            await emitter.synthesis_config(
                job_id=state.session_id,
                output_type=config.output_format or "generic",
                model_tier="synthesis",  # Default tier
                temperature=0.7,  # Default from config
                max_tokens=8000,  # Default from config
                query_preview=query[:200] if query else "",
                schema_name=_get_schema_name(state.output_schema),
                schema_fields=(
                    list(state.output_schema.model_json_schema().get("properties", {}).keys())
                    if state.output_schema else None
                ),
                schema_required_fields=(
                    state.output_schema.model_json_schema().get("required", [])
                    if state.output_schema else None
                ),
                verify_sources=state.enable_citation_verification,
                enable_post_verification=state.enable_post_verification,
            )
        except Exception as e:
            logger.warning(
                "LIFECYCLE_HOOK_EMISSION_FAILED",
                hook="on_synthesis_config",
                error=str(e)[:200],
            )

    # Load existing sources from chat source pool for follow-ups
    # This enables citing previous research without re-crawling
    chat_source_pool = None
    if db is not None and chat_id and conversation_history:
        try:
            from deep_research.services.chat_source_pool_service import ChatSourcePoolService
            from deep_research.services.llm.embedder import get_embedder

            chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id
            embedder = get_embedder()
            chat_source_pool = ChatSourcePoolService(db, embedder=embedder)

            # Load existing sources from this chat
            existing_sources = await chat_source_pool.get_all_sources(chat_id_uuid)

            if existing_sources:
                # Pre-populate state with existing sources for follow-up context
                from deep_research.agent.state import SourceInfo

                for src in existing_sources:
                    state.sources.append(
                        SourceInfo(
                            url=src.url,
                            title=src.title,
                            snippet=src.snippet,
                            content=src.content,
                            relevance_score=src.relevance_score,
                        )
                    )

                # Build searchable index for researcher to use
                await chat_source_pool.build_search_index(chat_id_uuid)

                logger.info(
                    "CHAT_SOURCES_LOADED",
                    chat_id=str(chat_id_uuid),
                    count=len(existing_sources),
                )
        except Exception as e:
            logger.warning(
                "CHAT_SOURCE_POOL_LOAD_FAILED",
                error=str(e)[:200],
            )
            # Continue without existing sources - not critical

    steps_executed = 0
    steps_skipped = 0
    event_buffer: EventBuffer | None = None  # Initialize before try block for exception handler

    # Create MLflow run to associate trace with params
    with safe_mlflow_run(f"research_{str(state.session_id)[:8]}"):
        # Create span INSIDE the run context so trace is properly nested under run
        async with safe_tool_span("stream_research_orchestration", "CHAIN", {
            "research.session_id": str(state.session_id),
            "research.query": truncate(query, 200),
            "research.max_iterations": config.max_plan_iterations,
            "research.streaming": True,
            "research.enable_background": config.enable_background_investigation,
            "research.enable_clarification": config.enable_clarification,
        }) as root_span:

            # Group traces by user and chat session for MLflow trace correlation
            if user_id or chat_id:
                trace_metadata: dict[str, str] = {}
                if user_id:
                    trace_metadata["mlflow.trace.user"] = user_id
                if chat_id:
                    trace_metadata["mlflow.trace.session"] = chat_id
                safe_update_trace(trace_metadata)

            try:
                # =============================================================
                # Query Context Extraction (Plugin-based)
                # =============================================================
                # If a plugin provides ExtractionConfigProvider, extract structured
                # context from the query using LLM. This provides plugin_data for
                # custom phases (e.g., company_name, attendees, meeting_purpose).
                # =============================================================

                extracted_context: dict[str, Any] | None = None

                if plugin_manager and not plugin_data:
                    # Only extract if plugin_data not already provided
                    try:
                        from deep_research.agent.extraction import extract_query_context

                        logger.info(f"EXTRACTION_START query_len={len(query)}")

                        extracted_context = await extract_query_context(
                            query=query,
                            llm=llm,
                            plugin_manager=plugin_manager,
                        )

                        if extracted_context:
                            logger.info(
                                f"EXTRACTION_SUCCESS keys={list(extracted_context.keys())} company={extracted_context.get('company_name', 'N/A')} attendees_count={len(extracted_context.get('attendees', []))}"
                            )
                            plugin_data = extracted_context
                        else:
                            logger.info("EXTRACTION_NO_RESULT query may not match plugin extraction pattern")

                    except Exception as e:
                        logger.warning(
                            f"EXTRACTION_FAILED error={str(e)[:200]} type={type(e).__name__}"
                        )
                        # Continue without extraction - not a fatal error

                    # Fallback to regex-based extraction if LLM failed
                    if not extracted_context or not extracted_context.get("company_name"):
                        logger.info("EXTRACTION_FALLBACK_REGEX attempting pattern matching")

                        regex_data = _extract_context_from_query(query)

                        if regex_data:
                            logger.info(
                                f"EXTRACTION_FALLBACK_SUCCESS keys={list(regex_data.keys())} company={regex_data.get('company_name', 'N/A')}"
                            )
                            extracted_context = regex_data
                            plugin_data = extracted_context

                elif plugin_data:
                    logger.info(
                        f"EXTRACTION_SKIPPED_PROVIDED_DATA keys={list(plugin_data.keys())} company={plugin_data.get('company_name', 'N/A')}"
                    )

                # =============================================================
                # Custom Phase Mode Check (Plugin-provided phases)
                # =============================================================
                # Check if any plugin has disabled planner and defined custom phases
                # If so, route to custom phase execution instead of planner
                # =============================================================
                use_custom_phases = False
                customization = None

                # DIAGNOSTIC: Track plugin_manager instance and state
                if plugin_manager:
                    logger.info(
                        f"ORCHESTRATOR_PLUGIN_MANAGER_CHECK instance_id={id(plugin_manager)} has_method={hasattr(plugin_manager, 'has_custom_phase_mode')} num_plugins={len(plugin_manager) if hasattr(plugin_manager, '__len__') else 0} num_phases={len(plugin_manager.get_all_phases()) if hasattr(plugin_manager, 'get_all_phases') else 0}"
                    )

                    # DIAGNOSTIC: Check customization state
                    customization_result = plugin_manager.get_pipeline_customization()
                    logger.info(
                        f"ORCHESTRATOR_CUSTOMIZATION_CHECK has_customization={customization_result is not None} disabled_agents={list(customization_result.disabled_agents) if customization_result else []} num_insertions={len(customization_result.phase_insertions) if customization_result else 0}"
                    )

                    # DIAGNOSTIC: Check has_custom_phase_mode components
                    if hasattr(plugin_manager, 'has_custom_phase_mode'):
                        result = plugin_manager.has_custom_phase_mode()
                        logger.info(
                            f"ORCHESTRATOR_CUSTOM_PHASE_CHECK result={result}"
                        )
                    else:
                        logger.error("ORCHESTRATOR_MISSING_METHOD plugin_manager lacks has_custom_phase_mode()")
                else:
                    logger.warning("ORCHESTRATOR_NO_PLUGIN_MANAGER plugin_manager is None")

                if plugin_manager and plugin_manager.has_custom_phase_mode():
                    use_custom_phases = True
                    customization = plugin_manager.get_pipeline_customization()
                    logger.info(
                        "CUSTOM_PHASE_MODE_DETECTED",
                        phases=list(plugin_manager.get_all_phases().keys()),
                        disabled_agents=list(customization.disabled_agents) if customization else [],
                    )

                # Create event buffer early for custom phase mode
                # Session is pre-created by JobManager, so FK constraint is satisfied
                if use_custom_phases and config.session_pre_created and config.research_session_id:
                    event_buffer = EventBuffer(config.research_session_id)
                    logger.info(
                        "EVENT_BUFFER_CREATED_FOR_CUSTOM_PHASES",
                        session_id=str(config.research_session_id)[:8],
                    )
                    # Emit research_started event for frontend to show progress
                    started_event = ResearchStartedEvent(
                        message_id=str(config.message_id) if config.message_id else "",
                        research_session_id=str(config.research_session_id),
                    )
                    yield started_event
                    await event_buffer.add_event(started_event)

                # =============================================================
                # Query Mode Routing (Tiered Query Modes feature)
                # =============================================================
                # SIMPLE mode: Direct LLM response, skip coordinator entirely
                # WEB_SEARCH mode: Lightweight pipeline (handled below in T022)
                # DEEP_RESEARCH mode: Full pipeline (existing flow)
                # CUSTOM_PHASE mode: Plugin-provided phases (checked above)
                # =============================================================

                if config.query_mode == "simple":
                    # Simple mode: Direct LLM response with full memory access
                    # Skip coordinator, no web search, but has access to sources/observations
                    logger.info(
                        "SIMPLE_MODE_START",
                        query=truncate(query, 100),
                        sources_count=len(state.sources),
                        observations_count=len(state.all_observations),
                    )

                    yield SynthesisStartedEvent(
                        total_observations=len(state.all_observations),
                        total_sources=len(state.sources),
                    )
                    yield _agent_started("synthesizer", "simple")
                    agent_start = time.perf_counter()

                    # Use handle_simple_query with memory access (sources + observations)
                    simple_chunks: list[str] = []
                    async for chunk in handle_simple_query(
                        state, llm, chat_source_pool=chat_source_pool
                    ):
                        simple_chunks.append(chunk)
                        yield SynthesisProgressEvent(content_chunk=chunk)

                    full_report = "".join(simple_chunks)
                    yield _agent_completed("synthesizer", agent_start)
                    state.complete(full_report)

                    # Emit completion event
                    total_duration_ms = (time.perf_counter() - start_time) * 1000
                    yield ResearchCompletedEvent(
                        session_id=state.session_id,
                        total_steps_executed=0,
                        total_steps_skipped=0,
                        plan_iterations=0,
                        total_duration_ms=int(total_duration_ms),
                        final_report=state.final_report,
                        structured_output=(
                            state.final_report_structured.model_dump()
                            if state.final_report_structured else None
                        ),
                    )

                    # Persist chat + message for simple mode (no research session)
                    # Use asyncio.shield with independent session to survive cancellation
                    if (
                        db is not None
                        and config.message_id is not None
                        and chat_id is not None
                        and user_id is not None
                    ):
                        from deep_research.agent.persistence import (
                            persist_simple_message_independent,
                            persist_simple_message_update_independent,
                        )

                        try:
                            chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id

                            # Check if session was pre-created (e.g., by JobManager)
                            # If so, use UPDATE path; otherwise use INSERT path
                            if config.session_pre_created:
                                counts = await asyncio.shield(
                                    persist_simple_message_update_independent(
                                        message_id=config.message_id,
                                        content=full_report,
                                    )
                                )
                            else:
                                counts = await asyncio.shield(
                                    persist_simple_message_independent(
                                        chat_id=chat_id_uuid,
                                        user_id=user_id,
                                        user_query=query,
                                        message_id=config.message_id,
                                        content=full_report,
                                    )
                                )
                            logger.info(
                                "SIMPLE_MODE_PERSISTED",
                                message_id=str(config.message_id),
                                content_len=len(full_report),
                                session_pre_created=config.session_pre_created,
                            )

                            # Emit persistence_completed event for frontend
                            chat_title = query[:47] + "..." if len(query) > 50 else query
                            yield PersistenceCompletedEvent(
                                chat_id=str(chat_id_uuid),
                                message_id=str(config.message_id),
                                research_session_id=None,  # No research session for simple mode
                                chat_title=chat_title,
                                was_draft=config.is_draft,
                                counts=counts,
                            )
                        except asyncio.CancelledError:
                            # INTENTIONAL: Not re-raising CancelledError here.
                            # asyncio.shield() ensures persistence completes even if client
                            # disconnects. We swallow the exception to allow graceful
                            # degradation - data is saved, just the confirmation event
                            # couldn't be sent to the disconnected client.
                            logger.warning(
                                "SIMPLE_MODE_PERSISTENCE_CANCELLED",
                                detail="Persistence cancelled but may have completed",
                            )
                        except Exception as e:
                            logger.warning(
                                "SIMPLE_MODE_PERSISTENCE_FAILED",
                                error=str(e)[:200],
                                message_id=str(config.message_id) if config.message_id else None,
                            )
                    else:
                        logger.warning(
                            "SIMPLE_MODE_PERSISTENCE_SKIPPED",
                            db_available=db is not None,
                            message_id=config.message_id,
                            chat_id=chat_id,
                            user_id=user_id,
                        )
                        yield StreamErrorEvent(
                            error_code="PERSISTENCE_SKIPPED",
                            error_message="Simple mode response completed but could not persist to database",
                            recoverable=True,
                            stack_trace="".join(traceback.format_stack()),
                        )

                    return

                if config.query_mode == "web_search":
                    # =============================================================
                    # Web Search mode: Lightweight pipeline with 2-5 sources
                    # Reuses existing researcher + synthesizer with minimal config
                    # Includes 15-second timeout with fallback to Simple mode
                    # =============================================================
                    logger.info(
                        "WEB_SEARCH_MODE_START",
                        query=truncate(query, 100),
                    )

                    # Get web search mode config (includes timeout_seconds and max_retries)
                    mode_config = get_query_mode_config("web_search")
                    web_search_timeout = getattr(mode_config, "timeout_seconds", 20)
                    web_search_max_retries = getattr(mode_config, "max_retries", 5)

                    # Track start time for timeout
                    web_search_start = time.perf_counter()

                    # For pre-created sessions (JobManager), create event buffer and emit started event
                    # Use buffer_size=1 for web_search mode to ensure immediate event visibility
                    # (prevents "waiting for activity" in UI while events accumulate)
                    if config.session_pre_created and config.research_session_id is not None:
                        event_buffer = EventBuffer(config.research_session_id, buffer_size=1)
                        logger.info(
                            "WEB_SEARCH_USING_PRE_CREATED_SESSION",
                            session_id=str(config.research_session_id)[:8],
                        )
                        # Emit research_started event for frontend
                        started_event = ResearchStartedEvent(
                            message_id=str(config.message_id) if config.message_id else "",
                            research_session_id=str(config.research_session_id),
                        )
                        yield started_event
                        await event_buffer.add_event(started_event)
                        # Explicit flush for immediate visibility (user sees "started" right away)
                        await event_buffer.flush()

                    try:
                        # 1. Create minimal 1-step plan programmatically
                        plan_id = str(uuid4())
                        step_id = str(uuid4())
                        state.current_plan = Plan(
                            id=plan_id,
                            title="Quick Web Search",
                            thought="Answering query with quick web search",
                            steps=[
                                PlanStep(
                                    id=step_id,
                                    title="Search and answer",
                                    description=f"Find information about: {query}",
                                    step_type=StepType.RESEARCH,
                                    needs_search=True,
                                    status=StepStatus.PENDING,
                                )
                            ],
                            has_enough_context=False,
                            iteration=1,
                        )
                        ws_evt: StreamEvent = _plan_created(state)
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)

                        # 2. Run researcher with minimal configuration and timeout
                        ws_evt = _step_started(state)
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)
                        log_agent_transition(logger, from_agent=None, to_agent="researcher")
                        ws_evt = _agent_started("researcher", "analytical")
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)
                        agent_start = time.perf_counter()

                        # Use classic researcher - it loads limits from depth config
                        # For web search, we set effective_depth to 'light' to get minimal limits
                        state.effective_depth = "light"  # Override to use light depth limits

                        # Wrap researcher in timeout with retry logic
                        researcher_succeeded = False
                        for retry_attempt in range(web_search_max_retries):
                            try:
                                # Drain the generator within the timeout
                                async def _run_researcher_to_completion() -> None:
                                    async for react_event in run_researcher(
                                        state, llm, crawler, brave_client
                                    ):
                                        stream_evt = _convert_react_event(react_event)
                                        if stream_evt:
                                            yield_buf.append(stream_evt)

                                yield_buf: list[StreamEvent] = []
                                await asyncio.wait_for(
                                    _run_researcher_to_completion(),
                                    timeout=web_search_timeout,
                                )
                                # Emit buffered events
                                for buffered_evt in yield_buf:
                                    yield buffered_evt
                                    await _buffer_event(buffered_evt, event_buffer)
                                researcher_succeeded = True
                                break  # Success - exit retry loop
                            except TimeoutError:
                                logger.warning(
                                    "WEB_SEARCH_RESEARCHER_TIMEOUT_RETRY",
                                    attempt=retry_attempt + 1,
                                    max_retries=web_search_max_retries,
                                    elapsed_seconds=time.perf_counter() - web_search_start,
                                    timeout_seconds=web_search_timeout,
                                )
                                # Continue to next retry attempt

                        if not researcher_succeeded:
                            # All retries exhausted - raise to trigger fallback
                            logger.error(
                                "WEB_SEARCH_RESEARCHER_ALL_RETRIES_EXHAUSTED",
                                attempts=web_search_max_retries,
                                total_elapsed_seconds=time.perf_counter() - web_search_start,
                            )
                            raise TimeoutError("Web search researcher timed out after all retries")

                        ws_evt = _agent_completed("researcher", agent_start)
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)

                        # Mark step as complete (skip reflector - always COMPLETE for web search)
                        state.mark_step_complete(state.last_observation)
                        ws_evt = _step_completed(state)
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)
                        state.advance_step()
                        steps_executed = 1

                        # 3. Synthesize with natural mode ([1], [2] citations)
                        ws_evt = SynthesisStartedEvent(
                            total_observations=1,
                            total_sources=len(state.sources),
                        )
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)
                        log_agent_transition(logger, from_agent="researcher", to_agent="synthesizer")
                        ws_evt = _agent_started("synthesizer", "analytical")
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)
                        agent_start = time.perf_counter()

                        # DIAGNOSTIC: Log synthesis mode decision
                        will_use_structured = state.output_format == "json" and state.output_schema is not None
                        logger.info(
                            "SYNTHESIS_MODE_DECISION",
                            output_format=state.output_format,
                            output_schema=_get_schema_name(state.output_schema),
                            enable_citation_verification=state.enable_citation_verification,
                            will_use_structured=will_use_structured,
                        )

                        # Check for structured JSON output first (non-streaming)
                        if state.output_format == "json" and state.output_schema:
                            # Non-streaming structured output - run_structured_synthesizer
                            # handles state.complete() internally
                            state = await run_structured_synthesizer(state, llm)
                            # Run post-verification if enabled (requires verify_sources=True)
                            if state.enable_post_verification and state.enable_citation_verification:
                                state = await post_verify_structured_output(state, llm)
                        else:
                            # Use citation synthesizer - returns dict events, convert to StreamEvents
                            web_search_chunks: list[str] = []
                            async for event_dict in stream_synthesis_with_citations(state, llm):
                                event_type = event_dict.get("type")
                                if event_type == "content":
                                    chunk = event_dict.get("chunk", "")
                                    web_search_chunks.append(chunk)
                                    ws_evt = SynthesisProgressEvent(content_chunk=chunk)
                                    yield ws_evt
                                    await _buffer_event(ws_evt, event_buffer)
                                elif event_type == "claim_verified":
                                    # Convert claim_id to UUID - generate new one if not valid UUID
                                    raw_claim_id = event_dict.get("claim_id")
                                    if isinstance(raw_claim_id, UUID):
                                        claim_id_uuid = raw_claim_id
                                    else:
                                        # Generate new UUID - claim_id from synthesis may be
                                        # an index or non-UUID string
                                        claim_id_uuid = uuid4()
                                    ws_evt = ClaimVerifiedEvent(
                                        claim_id=claim_id_uuid,
                                        claim_text=event_dict.get("claim_text", ""),
                                        position_start=event_dict.get("position_start", 0),
                                        position_end=event_dict.get("position_end", 0),
                                        verdict=event_dict.get("verdict", "unsupported"),
                                        confidence_level=event_dict.get("confidence", "medium"),
                                        evidence_preview=event_dict.get("evidence_preview", ""),
                                        reasoning=event_dict.get("reasoning"),
                                        citation_key=event_dict.get("citation_key"),
                                        citation_keys=event_dict.get("citation_keys"),
                                    )
                                    yield ws_evt
                                    await _buffer_event(ws_evt, event_buffer)
                                elif event_type == "verification_summary":
                                    ws_evt = VerificationSummaryEvent(
                                        message_id=config.message_id or uuid4(),
                                        total_claims=event_dict.get("total_claims", 0),
                                        supported=event_dict.get("supported", 0),
                                        partial=event_dict.get("partial", 0),
                                        unsupported=event_dict.get("unsupported", 0),
                                        contradicted=event_dict.get("contradicted", 0),
                                        abstained_count=event_dict.get("abstained_count", 0),
                                        citation_corrections=event_dict.get("citation_corrections", 0),
                                        warning=event_dict.get("warning", False),
                                    )
                                    yield ws_evt
                                    await _buffer_event(ws_evt, event_buffer)

                            # Note: stream_synthesis_with_citations() already calls
                            # state.complete() internally via the synthesis pipeline.
                            # (Same pattern as orchestrator.py deep_research path)

                        ws_evt = _agent_completed("synthesizer", agent_start)
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)

                        # Emit completion event
                        total_duration_ms = (time.perf_counter() - start_time) * 1000
                        ws_evt = ResearchCompletedEvent(
                            session_id=state.session_id,
                            total_steps_executed=steps_executed,
                            total_steps_skipped=0,
                            plan_iterations=1,
                            total_duration_ms=int(total_duration_ms),
                            final_report=state.final_report,
                            structured_output=(
                                state.final_report_structured.model_dump()
                                if state.final_report_structured else None
                            ),
                        )
                        yield ws_evt
                        await _buffer_event(ws_evt, event_buffer)

                        # Persist web search session (lightweight - sources only)
                        # Use asyncio.shield with independent session to prevent cancellation
                        # when client disconnects and request-scoped session is cleaned up
                        if (
                            db is not None
                            and config.research_session_id
                            and config.message_id
                            and chat_id
                            and user_id
                        ):
                            from deep_research.agent.persistence import (
                                persist_complete_research_independent,
                                persist_research_session_complete_update_independent,
                            )

                            try:
                                # Check if session was pre-created (e.g., by JobManager)
                                # If so, use UPDATE path; otherwise use INSERT path
                                if config.session_pre_created:
                                    counts = await asyncio.shield(
                                        persist_research_session_complete_update_independent(
                                            chat_id=UUID(chat_id),
                                            research_session_id=config.research_session_id,
                                            agent_message_id=config.message_id,
                                            state=state,
                                        )
                                    )
                                else:
                                    counts = await asyncio.shield(
                                        persist_complete_research_independent(
                                            chat_id=UUID(chat_id),
                                            user_id=user_id,
                                            user_query=query,
                                            message_id=config.message_id,
                                            research_session_id=config.research_session_id,
                                            research_depth="light",  # Web search uses light depth
                                            state=state,
                                        )
                                    )
                            except asyncio.CancelledError:
                                # INTENTIONAL: Not re-raising CancelledError here.
                                # asyncio.shield() ensures persistence completes even if client
                                # disconnects. We swallow the exception to allow graceful
                                # degradation - data is saved, just the confirmation event
                                # couldn't be sent to the disconnected client.
                                logger.warning(
                                    "WEB_SEARCH_PERSISTENCE_CANCELLED",
                                    detail="Persistence cancelled but may have completed",
                                )
                                counts = {"sources": len(state.sources)}  # Fallback count
                            ws_evt = PersistenceCompletedEvent(
                                chat_id=chat_id,
                                message_id=str(config.message_id),
                                research_session_id=str(config.research_session_id),
                                chat_title=query[:50] + "..." if len(query) > 50 else query,
                                was_draft=not config.session_pre_created,
                                counts=counts,
                            )
                            yield ws_evt
                            await _buffer_event(ws_evt, event_buffer)
                        else:
                            # Log warning when persistence conditions not met
                            logger.warning(
                                "WEB_SEARCH_PERSISTENCE_SKIPPED",
                                db_available=db is not None,
                                message_id=config.message_id,
                                research_session_id=config.research_session_id,
                                chat_id=chat_id,
                                user_id=user_id,
                            )
                            ws_evt = StreamErrorEvent(
                                error_code="PERSISTENCE_SKIPPED",
                                error_message="Web search completed but could not persist to database",
                                recoverable=True,
                                stack_trace="".join(traceback.format_stack()),
                            )
                            yield ws_evt
                            await _buffer_event(ws_evt, event_buffer)

                        # Flush event buffer to ensure all events are persisted
                        if event_buffer:
                            await event_buffer.flush()
                        return

                    except TimeoutError:
                        # Web search timed out - fall back to Simple mode
                        logger.warning(
                            "WEB_SEARCH_TIMEOUT_FALLBACK",
                            elapsed_seconds=time.perf_counter() - web_search_start,
                            timeout_seconds=web_search_timeout,
                            query=truncate(query, 100),
                        )

                        # Notify frontend of fallback
                        fallback_evt: StreamEvent = StreamErrorEvent(
                            error_code="WEB_SEARCH_TIMEOUT",
                            error_message="Web search timed out, falling back to direct answer",
                            recoverable=True,
                            stack_trace="".join(traceback.format_stack()),
                        )
                        yield fallback_evt
                        await _buffer_event(fallback_evt, event_buffer)

                        # Fall back to Simple mode (direct LLM response)
                        fallback_evt = SynthesisStartedEvent(total_observations=0, total_sources=0)
                        yield fallback_evt
                        await _buffer_event(fallback_evt, event_buffer)
                        fallback_evt = _agent_started("synthesizer", "simple")
                        yield fallback_evt
                        await _buffer_event(fallback_evt, event_buffer)
                        fallback_start = time.perf_counter()

                        fallback_chunks: list[str] = []
                        async for chunk in handle_simple_query(state, llm):
                            fallback_chunks.append(chunk)
                            fallback_evt = SynthesisProgressEvent(content_chunk=chunk)
                            yield fallback_evt
                            await _buffer_event(fallback_evt, event_buffer)

                        full_report = "".join(fallback_chunks)
                        fallback_evt = _agent_completed("synthesizer", fallback_start)
                        yield fallback_evt
                        await _buffer_event(fallback_evt, event_buffer)
                        if state.completed_at is None:
                            state.complete(full_report)
                        else:
                            logger.warning(
                                "TIMEOUT_FALLBACK_SKIP_ALREADY_COMPLETE",
                                completed_at=str(state.completed_at),
                                existing_report_len=len(state.final_report) if state.final_report else 0,
                                fallback_report_len=len(full_report),
                            )

                        # Emit completion event
                        total_duration_ms = (time.perf_counter() - start_time) * 1000
                        fallback_evt = ResearchCompletedEvent(
                            session_id=state.session_id,
                            total_steps_executed=0,
                            total_steps_skipped=0,
                            plan_iterations=0,
                            total_duration_ms=int(total_duration_ms),
                            final_report=state.final_report,
                            structured_output=(
                                state.final_report_structured.model_dump()
                                if state.final_report_structured else None
                            ),
                        )
                        yield fallback_evt
                        await _buffer_event(fallback_evt, event_buffer)

                        # Persist fallback response (same pattern as simple mode)
                        # Use asyncio.shield with independent session to survive cancellation
                        if (
                            db is not None
                            and config.message_id is not None
                            and chat_id is not None
                            and user_id is not None
                        ):
                            from deep_research.agent.persistence import (
                                persist_research_session_complete_update_independent,
                                persist_simple_message_independent,
                            )

                            try:
                                chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id

                                # Check if session was pre-created (e.g., by JobManager)
                                # If so, use UPDATE path; otherwise use INSERT path
                                if config.session_pre_created and config.research_session_id:
                                    # state.final_report is already set via state.complete(full_report)
                                    counts = await asyncio.shield(
                                        persist_research_session_complete_update_independent(
                                            chat_id=chat_id_uuid,
                                            research_session_id=config.research_session_id,
                                            agent_message_id=config.message_id,
                                            state=state,
                                        )
                                    )
                                    research_session_id_str: str | None = str(config.research_session_id)
                                else:
                                    counts = await asyncio.shield(
                                        persist_simple_message_independent(
                                            chat_id=chat_id_uuid,
                                            user_id=user_id,
                                            user_query=query,
                                            message_id=config.message_id,
                                            content=full_report,
                                        )
                                    )
                                    research_session_id_str = None

                                logger.info(
                                    "WEB_SEARCH_FALLBACK_PERSISTED",
                                    message_id=str(config.message_id),
                                    content_len=len(full_report),
                                    session_pre_created=config.session_pre_created,
                                )

                                # Emit persistence_completed event for frontend
                                chat_title = query[:47] + "..." if len(query) > 50 else query
                                fallback_evt = PersistenceCompletedEvent(
                                    chat_id=str(chat_id_uuid),
                                    message_id=str(config.message_id),
                                    research_session_id=research_session_id_str,
                                    chat_title=chat_title,
                                    was_draft=not config.session_pre_created,
                                    counts=counts,
                                )
                                yield fallback_evt
                                await _buffer_event(fallback_evt, event_buffer)
                            except asyncio.CancelledError:
                                # INTENTIONAL: Not re-raising CancelledError here.
                                # asyncio.shield() ensures persistence completes even if client
                                # disconnects. We swallow the exception to allow graceful
                                # degradation - data is saved, just the confirmation event
                                # couldn't be sent to the disconnected client.
                                logger.warning(
                                    "WEB_SEARCH_FALLBACK_PERSISTENCE_CANCELLED",
                                    detail="Persistence cancelled but may have completed",
                                )
                            except Exception as e:
                                logger.warning(
                                    "WEB_SEARCH_FALLBACK_PERSISTENCE_FAILED",
                                    error=str(e)[:200],
                                    message_id=str(config.message_id) if config.message_id else None,
                                )
                        else:
                            logger.warning(
                                "WEB_SEARCH_FALLBACK_PERSISTENCE_SKIPPED",
                                db_available=db is not None,
                                message_id=config.message_id,
                                chat_id=chat_id,
                                user_id=user_id,
                            )
                            fallback_evt = StreamErrorEvent(
                                error_code="PERSISTENCE_SKIPPED",
                                error_message="Web search fallback response could not persist to database",
                                recoverable=True,
                                stack_trace="".join(traceback.format_stack()),
                            )
                            yield fallback_evt
                            await _buffer_event(fallback_evt, event_buffer)

                        # Flush event buffer
                        if event_buffer:
                            await event_buffer.flush()

                    return

                # =============================================================
                # Custom Phase Mode: Execute plugin-provided phases
                # =============================================================
                # Activated when a plugin:
                # 1. Disables the planner via disabled_agents={"planner"}
                # 2. Provides custom phases via PhaseProvider protocol
                # 3. Specifies phase execution order via phase_insertions
                # =============================================================
                if use_custom_phases and customization and plugin_manager:
                    logger.info(
                        f"CUSTOM_PHASE_EXECUTION_START num_phases={len(plugin_manager.get_all_phases())} insertions={len(customization.phase_insertions)}"
                    )

                    try:
                        async for event in _stream_research_with_custom_phases(
                            query=query,
                            state=state,
                            llm=llm,
                            brave_client=brave_client,
                            crawler=crawler,
                            plugin_manager=plugin_manager,
                            customization=customization,
                            config=config,
                            db=db,
                            chat_id=chat_id,
                            user_id=user_id,
                            start_time=start_time,
                            plugin_data=plugin_data,
                        ):
                            if event_buffer:
                                await event_buffer.add_event(event)
                            yield event

                        logger.info("CUSTOM_PHASE_EXECUTION_COMPLETE")

                        # Flush event buffer to ensure all events are persisted
                        if event_buffer:
                            await event_buffer.flush()
                            logger.info(
                                "EVENT_BUFFER_FLUSHED_CUSTOM_PHASES",
                                session_id=str(config.research_session_id)[:8] if config.research_session_id else None,
                            )

                        # Persist session completion (update status to COMPLETED and save final_report)
                        # This is critical - without this, the frontend won't see the report!
                        if (
                            db is not None
                            and config.research_session_id
                            and config.message_id
                            and chat_id
                            and user_id
                            and state.final_report
                        ):
                            from deep_research.agent.persistence import (
                                persist_research_session_complete_update_independent,
                            )

                            try:
                                counts = await asyncio.shield(
                                    persist_research_session_complete_update_independent(
                                        chat_id=UUID(chat_id),
                                        research_session_id=config.research_session_id,
                                        agent_message_id=config.message_id,
                                        state=state,
                                    )
                                )
                                logger.info(
                                    "CUSTOM_PHASE_SESSION_COMPLETED",
                                    session_id=str(config.research_session_id)[:8],
                                    final_report_length=len(state.final_report) if state.final_report else 0,
                                    counts=counts,
                                )
                            except asyncio.CancelledError:
                                # asyncio.shield ensures persistence completes even if client disconnects
                                logger.warning(
                                    "CUSTOM_PHASE_SESSION_PERSIST_CANCELLED",
                                    session_id=str(config.research_session_id)[:8],
                                )
                            except Exception as persist_err:
                                logger.error(
                                    "CUSTOM_PHASE_SESSION_PERSIST_FAILED",
                                    error=str(persist_err)[:200],
                                    session_id=str(config.research_session_id)[:8],
                                )
                        else:
                            logger.warning(
                                "CUSTOM_PHASE_SESSION_PERSIST_SKIPPED",
                                has_db=db is not None,
                                has_session_id=config.research_session_id is not None,
                                has_message_id=config.message_id is not None,
                                has_chat_id=chat_id is not None,
                                has_user_id=user_id is not None,
                                has_final_report=state.final_report is not None and len(state.final_report) > 0 if state.final_report else False,
                            )

                        return  # Exit after custom phases

                    except Exception as e:
                        logger.error(
                            f"CUSTOM_PHASE_EXECUTION_FAILED error={str(e)[:500]} type={type(e).__name__}",
                            exc_info=True,
                        )
                        raise  # Re-raise to trigger error handling
                else:
                    # DIAGNOSTIC: Why didn't custom phases run?
                    logger.info(
                        f"CUSTOM_PHASE_EXECUTION_SKIPPED use_custom_phases={use_custom_phases} has_customization={customization is not None} has_plugin_manager={plugin_manager is not None}"
                    )

                # =============================================================
                # Deep Research mode continues here (existing full pipeline)
                # =============================================================

                # Pre-generate user message UUID for session start
                user_message_id = uuid4()

                # =============================================================
                # Two-Phase Persistence: Create session at START for crash resilience
                # =============================================================
                # This enables:
                # - Events to be persisted during streaming (FK to session satisfied)
                # - Frontend to reconnect if browser reloads mid-research
                # - Session marked FAILED on error instead of orphaned
                # =============================================================
                # Skip if session was already created by JobManager
                if (
                    db is not None
                    and config.research_session_id is not None
                    and config.message_id is not None
                    and chat_id is not None
                    and user_id is not None
                    and not config.session_pre_created  # Skip if JobManager already created
                ):
                    from deep_research.agent.persistence import (
                        persist_research_session_start_independent,
                    )

                    chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id
                    try:
                        await persist_research_session_start_independent(
                            chat_id=chat_id_uuid,
                            user_id=user_id,
                            user_query=query,
                            user_message_id=user_message_id,
                            agent_message_id=config.message_id,
                            research_session_id=config.research_session_id,
                            research_depth=config.research_depth,
                            query_mode=config.query_mode,
                        )
                        logger.info(
                            "RESEARCH_SESSION_CREATED_AT_START",
                            session_id=str(config.research_session_id)[:8],
                            chat_id=str(chat_id_uuid)[:8],
                        )

                        # Create event buffer now that session exists (FK satisfied)
                        event_buffer = EventBuffer(config.research_session_id)

                        # Emit research_started event for frontend
                        started_event = ResearchStartedEvent(
                            message_id=str(config.message_id),
                            research_session_id=str(config.research_session_id),
                        )
                        yield started_event
                        if event_buffer:
                            await event_buffer.add_event(started_event)

                    except Exception as e:
                        logger.warning(
                            "RESEARCH_SESSION_START_FAILED",
                            error=str(e)[:200],
                            session_id=str(config.research_session_id)[:8] if config.research_session_id else None,
                        )
                        # Continue without event buffering - old behavior as fallback
                elif config.session_pre_created and config.research_session_id is not None:
                    # Session was pre-created by JobManager, just set up event buffer
                    event_buffer = EventBuffer(config.research_session_id)
                    logger.info(
                        "USING_PRE_CREATED_SESSION",
                        session_id=str(config.research_session_id)[:8],
                    )
                    # Emit research_started event for frontend
                    started_event = ResearchStartedEvent(
                        message_id=str(config.message_id) if config.message_id else "",
                        research_session_id=str(config.research_session_id),
                    )
                    yield started_event
                    if event_buffer:
                        await event_buffer.add_event(started_event)

                # Phase 1: Coordinator
                log_agent_transition(logger, from_agent=None, to_agent="coordinator")
                evt: StreamEvent = _agent_started("coordinator", "simple")
                yield evt
                await _buffer_event(evt, event_buffer)
                agent_start = time.perf_counter()

                state = await run_coordinator(state, llm)

                evt = _agent_completed("coordinator", agent_start)
                yield evt
                await _buffer_event(evt, event_buffer)

                # Log research configuration to MLflow run (after coordinator resolves depth)
                log_research_config(depth=state.resolve_depth())

                # Handle simple queries (coordinator-detected, not user-selected mode)
                if state.is_simple_query:
                    yield SynthesisStartedEvent(total_observations=0, total_sources=0)
                    yield _agent_started("synthesizer", "simple")

                    # Accumulate chunks locally to avoid state mutation during iteration
                    chunks: list[str] = []
                    async for chunk in handle_simple_query(state, llm):
                        chunks.append(chunk)
                        yield SynthesisProgressEvent(content_chunk=chunk)

                    # Update state only after successful completion
                    full_report = "".join(chunks)
                    yield _agent_completed("synthesizer", agent_start)
                    state.complete(full_report)

                else:
                    # Phase 2: Background Investigation
                    if config.enable_background_investigation:
                        log_agent_transition(logger, from_agent="coordinator", to_agent="background_investigator")
                        evt = _agent_started("background_investigator", "simple")
                        yield evt
                        await _buffer_event(evt, event_buffer)
                        agent_start = time.perf_counter()
                        async for react_event in run_background_investigator(state, llm, brave_client):
                            stream_evt = _convert_react_event(react_event)
                            if stream_evt:
                                yield stream_evt
                                await _buffer_event(stream_evt, event_buffer)
                        evt = _agent_completed("background_investigator", agent_start)
                        yield evt
                        await _buffer_event(evt, event_buffer)

                    # Phase 3: Planning and Research Loop
                    while state.plan_iterations < config.max_plan_iterations:
                        if state.is_cancelled:
                            break

                        prev_agent = "background_investigator" if config.enable_background_investigation else "coordinator"
                        log_agent_transition(
                            logger,
                            from_agent=prev_agent,
                            to_agent="planner",
                            reason=f"iteration {state.plan_iterations + 1}",
                        )
                        evt = _agent_started("planner", "analytical")
                        yield evt
                        await _buffer_event(evt, event_buffer)
                        agent_start = time.perf_counter()
                        state = await run_planner(state, llm)
                        evt = _agent_completed("planner", agent_start)
                        yield evt
                        await _buffer_event(evt, event_buffer)

                        if state.current_plan:
                            evt = _plan_created(state)
                            yield evt
                            await _buffer_event(evt, event_buffer)

                            if state.current_plan.has_enough_context:
                                break

                            while state.has_more_steps() and not state.is_cancelled:
                                step = state.get_current_step()
                                if not step:
                                    break

                                evt = _step_started(state)
                                yield evt
                                await _buffer_event(evt, event_buffer)

                                log_agent_transition(
                                    logger,
                                    from_agent="planner",
                                    to_agent="researcher",
                                    reason=f"step {state.current_step_index + 1}",
                                )
                                evt = _agent_started("researcher", "analytical")
                                yield evt
                                await _buffer_event(evt, event_buffer)
                                agent_start = time.perf_counter()

                                # Get researcher mode for current depth
                                depth = state.resolve_depth()
                                researcher_config = get_researcher_config_for_depth(depth)

                                if researcher_config.mode == ResearcherMode.REACT:
                                    # ReAct mode: LLM controls the research loop
                                    researcher_gen = run_react_researcher(
                                        state, llm, crawler, brave_client
                                    )
                                else:
                                    # Classic mode: single-pass fixed searches/crawls
                                    researcher_gen = run_researcher(
                                        state, llm, crawler, brave_client
                                    )

                                async for react_event in researcher_gen:
                                    stream_evt = _convert_react_event(react_event)
                                    if stream_evt:
                                        yield stream_evt
                                        await _buffer_event(stream_evt, event_buffer)
                                    elif react_event.event_type == "research_complete":
                                        logger.info(
                                            "REACT_RESEARCH_COMPLETE",
                                            reason=react_event.data.get("reason", ""),
                                            tool_calls=react_event.data.get("tool_calls", 0),
                                            high_quality=react_event.data.get("high_quality_sources", 0),
                                        )

                                evt = _agent_completed("researcher", agent_start)
                                yield evt
                                await _buffer_event(evt, event_buffer)
                                steps_executed += 1

                                evt = _step_completed(state)
                                yield evt
                                await _buffer_event(evt, event_buffer)

                                log_agent_transition(logger, from_agent="researcher", to_agent="reflector")
                                evt = _agent_started("reflector", "simple")
                                yield evt
                                await _buffer_event(evt, event_buffer)
                                agent_start = time.perf_counter()
                                state = await run_reflector(state, llm)
                                evt = _agent_completed("reflector", agent_start)
                                yield evt
                                await _buffer_event(evt, event_buffer)

                                if state.last_reflection:
                                    evt = _reflection_decision(state)
                                    yield evt
                                    await _buffer_event(evt, event_buffer)

                                    if state.last_reflection.decision == ReflectionDecision.COMPLETE:
                                        while state.has_more_steps():
                                            state.advance_step()
                                            steps_skipped += 1
                                        break

                                    if state.last_reflection.decision == ReflectionDecision.ADJUST:
                                        preserved_count = len(state.get_completed_steps())
                                        logger.info(
                                            "ADJUSTING_PLAN",
                                            reason="reflection_decision",
                                            preserving_completed_steps=preserved_count,
                                        )
                                        break

                                state.advance_step()

                            if state.last_reflection and state.last_reflection.decision == ReflectionDecision.ADJUST:
                                continue
                            break

                    # Phase 4: Streaming Synthesis
                    log_agent_transition(logger, from_agent="reflector", to_agent="synthesizer")
                    evt = SynthesisStartedEvent(
                        total_observations=len(state.all_observations),
                        total_sources=len(state.sources),
                    )
                    yield evt
                    await _buffer_event(evt, event_buffer)

                    evt = _agent_started("synthesizer", "complex")
                    yield evt
                    await _buffer_event(evt, event_buffer)
                    agent_start = time.perf_counter()

                    # Emit lifecycle hook: synthesis_started
                    if plugin_manager and state.session_id:
                        try:
                            from deep_research.plugins.lifecycle import EventEmitter

                            emitter = EventEmitter(plugin_manager)
                            await emitter.synthesis_started(
                                job_id=state.session_id,
                                first_event_type="synthesis_started",
                                elapsed_ms=(time.perf_counter() - start_time) * 1000,
                            )
                        except Exception as e:
                            logger.warning(
                                "LIFECYCLE_HOOK_EMISSION_FAILED hook=on_synthesis_started error=%s",
                                str(e)[:200],
                            )

                    # Use citation-aware synthesizer if enabled
                    # Collect content chunks for persistence
                    content_chunks: list[str] = []
                    structured_synthesis_failed = False
                    if state.output_format == "json" and state.output_schema:
                        # Non-streaming structured output - run_structured_synthesizer
                        # handles state.complete() internally
                        try:
                            state = await run_structured_synthesizer(state, llm)
                            # Run post-verification if enabled (requires verify_sources=True)
                            if state.enable_post_verification and state.enable_citation_verification:
                                state = await post_verify_structured_output(state, llm)
                        except StructuredSynthesisError as e:
                            logger.warning(
                                "STRUCTURED_SYNTHESIS_FALLBACK",
                                error=str(e)[:200],
                                falling_back_to="streaming",
                            )

                            # Emit lifecycle hook: validation_error
                            if plugin_manager and state.session_id:
                                try:
                                    from deep_research.plugins.lifecycle import EventEmitter

                                    emitter = EventEmitter(plugin_manager)
                                    # StructuredSynthesisError wraps the validation error
                                    await emitter.validation_error(
                                        job_id=state.session_id,
                                        error=e.original_error if hasattr(e, "original_error") else e,
                                        raw_output=str(e)[:1000],
                                    )
                                except Exception as hook_error:
                                    logger.warning(
                                        "LIFECYCLE_HOOK_EMISSION_FAILED hook=on_validation_error error=%s",
                                        str(hook_error)[:200],
                                    )

                            state = e.state
                            structured_synthesis_failed = True
                            # Fall back to streaming synthesis - emit chunks for frontend
                            async for chunk in stream_synthesis(state, llm):
                                content_chunks.append(chunk)
                                yield SynthesisProgressEvent(content_chunk=chunk)
                    elif state.enable_citation_verification:
                        async for synth_evt in stream_synthesis_with_citations(state, llm):
                            synth_event_type = synth_evt.get("type", "")
                            if synth_event_type == "content":
                                chunk = synth_evt.get("chunk", "")
                                content_chunks.append(chunk)
                                yield SynthesisProgressEvent(content_chunk=chunk)
                            # Yield verification events to frontend for real-time display
                            elif synth_event_type == "claim_verified":
                                yield ClaimVerifiedEvent(
                                    claim_id=_to_claim_uuid(synth_evt.get("claim_id")),
                                    claim_text=synth_evt.get("claim_text", ""),
                                    position_start=synth_evt.get("position_start", 0),
                                    position_end=synth_evt.get("position_end", 0),
                                    verdict=synth_evt.get("verdict", ""),
                                    confidence_level=synth_evt.get("confidence_level", ""),
                                    evidence_preview=synth_evt.get("evidence_preview", ""),
                                    reasoning=synth_evt.get("reasoning"),
                                    citation_key=synth_evt.get("citation_key"),
                                    citation_keys=synth_evt.get("citation_keys"),
                                )
                            elif synth_event_type == "verification_summary":
                                yield VerificationSummaryEvent(
                                    message_id=config.message_id or UUID(int=0),
                                    total_claims=synth_evt.get("total_claims", 0),
                                    supported=synth_evt.get("supported", 0),
                                    partial=synth_evt.get("partial", 0),
                                    unsupported=synth_evt.get("unsupported", 0),
                                    contradicted=synth_evt.get("contradicted", 0),
                                    abstained_count=synth_evt.get("abstained_count", 0),
                                    citation_corrections=synth_evt.get("citation_corrections", 0),
                                    warning=synth_evt.get("warning") or False,
                                )
                            elif synth_event_type == "citation_corrected":
                                yield CitationCorrectedEvent(
                                    claim_id=_to_claim_uuid(synth_evt.get("claim_id")),
                                    correction_type=synth_evt.get("correction_type", ""),
                                    reasoning=synth_evt.get("reasoning"),
                                )
                            elif synth_event_type == "numeric_claim_detected":
                                # Convert normalized_value to string - schema expects str, not float
                                raw_normalized = synth_evt.get("normalized_value")
                                normalized_str = str(raw_normalized) if raw_normalized is not None else None
                                yield NumericClaimDetectedEvent(
                                    claim_id=_to_claim_uuid(synth_evt.get("claim_id")),
                                    raw_value=synth_evt.get("raw_value", ""),
                                    normalized_value=normalized_str,
                                    unit=synth_evt.get("unit"),
                                    derivation_type=synth_evt.get("derivation_type", "direct"),
                                    qa_verified=synth_evt.get("qa_verified", False),
                                )
                            elif synth_event_type == "correction_metrics":
                                # Log metrics, no need to send to frontend
                                logger.debug(
                                    "CITATION_CORRECTION_METRICS",
                                    total_corrected=synth_evt.get("total_corrected", 0),
                                    kept=synth_evt.get("kept", 0),
                                    replaced=synth_evt.get("replaced", 0),
                                    removed=synth_evt.get("removed", 0),
                                )
                    else:
                        async for chunk in stream_synthesis(state, llm):
                            content_chunks.append(chunk)
                            yield SynthesisProgressEvent(content_chunk=chunk)

                    # Note: Streaming synthesis functions (stream_synthesis, stream_synthesis_with_citations)
                    # already call state.complete() internally, so we don't call it again here.
                    # The streaming functions also validate non-empty content internally.

                    evt = _agent_completed("synthesizer", agent_start)
                    yield evt
                    await _buffer_event(evt, event_buffer)

                    # =============================================================
                    # Two-Phase Persistence: Update session to COMPLETED at END
                    # =============================================================
                    # Flush event buffer first to ensure all events are persisted
                    if event_buffer:
                        try:
                            await event_buffer.flush()
                            logger.debug(
                                "EVENT_BUFFER_FINAL_FLUSH",
                                total_flushed=event_buffer.total_flushed,
                            )
                        except Exception as e:
                            logger.warning(
                                "EVENT_BUFFER_FLUSH_FAILED",
                                error=str(e)[:200],
                            )

                    # Use update function if session was created at START (two-phase)
                    # Otherwise fall back to old create function (backward compat)
                    if (
                        db is not None
                        and config.message_id is not None
                        and config.research_session_id is not None
                        and state.final_report
                        and chat_id is not None
                        and user_id is not None
                    ):
                        chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id

                        # Check if session was created at START (event_buffer exists)
                        if event_buffer is not None:
                            # Two-phase: Update existing session to COMPLETED
                            from deep_research.agent.persistence import (
                                persist_research_session_complete_update_independent,
                            )

                            try:
                                counts = await asyncio.shield(
                                    persist_research_session_complete_update_independent(
                                        chat_id=chat_id_uuid,
                                        research_session_id=config.research_session_id,
                                        agent_message_id=config.message_id,
                                        state=state,
                                    )
                                )
                                logger.info(
                                    "RESEARCH_SESSION_COMPLETED",
                                    message_id=str(config.message_id),
                                    research_session_id=str(config.research_session_id),
                                    report_len=len(state.final_report),
                                    claims=counts.get("claims", 0),
                                    citations=counts.get("citations", 0),
                                    sources=counts.get("sources", 0),
                                )

                                # Emit persistence_completed event
                                chat_title = query[:47] + "..." if len(query) > 50 else query
                                yield PersistenceCompletedEvent(
                                    chat_id=str(chat_id_uuid),
                                    message_id=str(config.message_id),
                                    research_session_id=str(config.research_session_id),
                                    chat_title=chat_title,
                                    was_draft=config.is_draft,
                                    counts=counts,
                                )
                            except Exception as e:
                                logger.warning(
                                    "RESEARCH_SESSION_COMPLETE_FAILED",
                                    error=str(e)[:200],
                                    message_id=str(config.message_id) if config.message_id else None,
                                )
                                # Mark session as FAILED
                                from deep_research.agent.persistence import (
                                    persist_research_session_failed_independent,
                                )
                                try:
                                    await persist_research_session_failed_independent(
                                        research_session_id=config.research_session_id,
                                        agent_message_id=config.message_id,
                                        error_message=str(e)[:500],
                                    )
                                except Exception:
                                    pass  # Best effort
                        else:
                            # Fallback: Old single-phase persistence (session not created at START)
                            from deep_research.agent.persistence import (
                                persist_complete_research_independent,
                            )

                            try:
                                counts = await asyncio.shield(
                                    persist_complete_research_independent(
                                        chat_id=chat_id_uuid,
                                        user_id=user_id,
                                        user_query=query,
                                        message_id=config.message_id,
                                        research_session_id=config.research_session_id,
                                        research_depth=config.research_depth,
                                        state=state,
                                    )
                                )
                                logger.info(
                                    "RESEARCH_DATA_PERSISTED_LEGACY",
                                    message_id=str(config.message_id),
                                    research_session_id=str(config.research_session_id),
                                    report_len=len(state.final_report),
                                    chat_created=counts.get("chat_created", 0),
                                    claims=counts.get("claims", 0),
                                    citations=counts.get("citations", 0),
                                    sources=counts.get("sources", 0),
                                )

                                chat_title = query[:47] + "..." if len(query) > 50 else query
                                yield PersistenceCompletedEvent(
                                    chat_id=str(chat_id_uuid),
                                    message_id=str(config.message_id),
                                    research_session_id=str(config.research_session_id),
                                    chat_title=chat_title,
                                    was_draft=config.is_draft,
                                    counts=counts,
                                )
                            except Exception as e:
                                logger.warning(
                                    "RESEARCH_PERSISTENCE_FAILED",
                                    error=str(e)[:200],
                                    message_id=str(config.message_id) if config.message_id else None,
                                )
                    else:
                        # Log warning when persistence conditions not met
                        logger.warning(
                            "DEEP_RESEARCH_PERSISTENCE_SKIPPED",
                            db_available=db is not None,
                            message_id=config.message_id,
                            research_session_id=config.research_session_id,
                            has_final_report=bool(state.final_report),
                            chat_id=chat_id,
                            user_id=user_id,
                        )
                        yield StreamErrorEvent(
                            error_code="PERSISTENCE_SKIPPED",
                            error_message="Deep research completed but could not persist to database",
                            recoverable=True,
                            stack_trace="".join(traceback.format_stack()),
                        )

            except Exception as e:
                tb = traceback.format_exc()
                logger.exception(
                    "STREAM_ORCHESTRATION_ERROR",
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )
                yield StreamErrorEvent(
                    error_code="ORCHESTRATION_ERROR",
                    error_message=str(e),
                    recoverable=False,
                    stack_trace=tb,
                    error_type=type(e).__name__,
                )

                # Mark session as FAILED if it was created at START
                if (
                    event_buffer is not None
                    and config.research_session_id is not None
                    and config.message_id is not None
                ):
                    from deep_research.agent.persistence import (
                        persist_research_session_failed_independent,
                    )

                    try:
                        # Flush buffer first to preserve any events collected
                        await event_buffer.flush()
                    except Exception:
                        pass  # Best effort

                    try:
                        await persist_research_session_failed_independent(
                            research_session_id=config.research_session_id,
                            agent_message_id=config.message_id,
                            error_message=str(e)[:500],
                        )
                        logger.info(
                            "RESEARCH_SESSION_MARKED_FAILED",
                            session_id=str(config.research_session_id)[:8],
                            error=str(e)[:100],
                        )
                    except Exception as fail_err:
                        logger.warning(
                            "RESEARCH_SESSION_FAIL_MARK_FAILED",
                            error=str(fail_err)[:200],
                        )

            total_duration_ms = (time.perf_counter() - start_time) * 1000

            yield ResearchCompletedEvent(
                session_id=state.session_id,
                total_steps_executed=steps_executed,
                total_steps_skipped=steps_skipped,
                plan_iterations=state.plan_iterations,
                total_duration_ms=int(total_duration_ms),
                final_report=state.final_report,
                structured_output=(
                    state.final_report_structured.model_dump()
                    if state.final_report_structured else None
                ),
            )


# =============================================================================
# Plan Review Functions (007-enterprise-data-sources, US12, T040)
# =============================================================================


def _create_plan_for_review(state: ResearchState) -> PlanForReview:
    """Create a PlanForReview from current state for user review.

    Converts the internal Plan dataclass to the PlanForReview schema
    that includes source hints for display to the user.

    Args:
        state: Current research state with plan.

    Returns:
        PlanForReview schema ready for serialization.

    Raises:
        ValueError: If no plan exists in state.
    """
    plan = state.current_plan
    if not plan:
        raise ValueError("No plan in state for review")

    # Convert steps to PlanStepForReview with source hints
    steps_for_review: list[PlanStepForReview] = []
    for step in plan.steps:
        # Check if we have source constraints for this step
        constraint = state.get_source_constraint(step.id)

        source_hints: list[dict[str, Any]] = []
        exclude_sources: list[str] = []

        if constraint:
            # Extract source hints from constraint if available
            if hasattr(constraint, "source_hints"):
                source_hints = [
                    h.to_dict() if hasattr(h, "to_dict") else dict(h)
                    for h in constraint.source_hints
                ]
            if hasattr(constraint, "exclude_sources"):
                exclude_sources = list(constraint.exclude_sources)

        steps_for_review.append(
            PlanStepForReview(
                id=step.id,
                title=step.title,
                description=step.description,
                step_type=step.step_type.value,
                needs_search=step.needs_search,
                source_hints=source_hints,
                exclude_sources=exclude_sources,
            )
        )

    return PlanForReview(
        id=plan.id,
        title=plan.title,
        thought=plan.thought,
        steps=steps_for_review,
        iteration=plan.iteration,
        data_landscape_summary=None,  # TODO: Add if available in state
    )


def apply_user_edits(
    state: ResearchState,
    edited_plan: PlanForReview,
) -> ResearchState:
    """Apply user edits from plan review to state.

    Updates the current plan in state with user modifications
    from the PlanReviewResponseEvent.

    Args:
        state: Current research state.
        edited_plan: User-modified plan from review.

    Returns:
        Updated state with modified plan.
    """
    if not state.current_plan:
        logger.warning("APPLY_USER_EDITS_NO_PLAN", detail="No plan to modify")
        return state

    # Update plan metadata
    state.current_plan.title = edited_plan.title
    state.current_plan.thought = edited_plan.thought

    # Update steps - match by ID
    existing_step_map = {s.id: s for s in state.current_plan.steps}
    new_steps: list[PlanStep] = []

    for edited_step in edited_plan.steps:
        if edited_step.id in existing_step_map:
            # Update existing step
            existing = existing_step_map[edited_step.id]
            existing.title = edited_step.title
            existing.description = edited_step.description
            existing.needs_search = edited_step.needs_search
            new_steps.append(existing)
        else:
            # Create new step from user addition
            new_steps.append(
                PlanStep(
                    id=edited_step.id,
                    title=edited_step.title,
                    description=edited_step.description,
                    step_type=StepType(edited_step.step_type),
                    needs_search=edited_step.needs_search,
                    status=StepStatus.PENDING,
                )
            )

        # Update source constraints from edited step
        if edited_step.source_hints or edited_step.exclude_sources:
            # Create a simple constraint dict to store
            constraint_dict = {
                "source_hints": edited_step.source_hints,
                "exclude_sources": edited_step.exclude_sources,
            }
            state.set_source_constraint(edited_step.id, constraint_dict)

    state.current_plan.steps = new_steps

    logger.info(
        "PLAN_EDITS_APPLIED",
        step_count=len(new_steps),
        edited_plan_id=edited_plan.id,
    )

    return state


async def execute_with_plan_review(
    state: ResearchState,
    config: OrchestrationConfig,
    response_queue: "asyncio.Queue[dict[str, Any]]",
) -> AsyncGenerator[StreamEvent, None]:
    """Wait for user review of plan with optional timeout.

    This generator yields PlanReviewEvent and waits for user response
    via the response_queue. Supports timeout with auto-proceed.

    Args:
        state: Current research state with plan.
        config: Orchestration config with review settings.
        response_queue: Queue to receive user response events.

    Yields:
        PlanReviewEvent, then PlanReviewTimeoutEvent if timeout occurs.

    Note:
        The caller is responsible for:
        1. Setting up the response_queue
        2. Feeding user responses into the queue
        3. Applying edits via apply_user_edits() if user edited the plan
    """
    if not config.enable_plan_review:
        return

    if not state.current_plan:
        logger.warning("PLAN_REVIEW_NO_PLAN", detail="No plan to review")
        return

    review_id = str(uuid4())
    plan_for_review = _create_plan_for_review(state)

    # Emit plan review event
    yield PlanReviewEvent(
        plan=plan_for_review,
        timeout_seconds=config.plan_review_timeout_seconds,
        review_id=review_id,
        require_approval=config.require_plan_approval,
        available_sources=[],  # TODO: Populate from data landscape
    )

    # Wait for response or timeout
    timeout = None if config.require_plan_approval else config.plan_review_timeout_seconds

    try:
        if timeout is not None:
            response = await asyncio.wait_for(
                response_queue.get(),
                timeout=float(timeout),
            )
        else:
            # No timeout - wait indefinitely
            response = await response_queue.get()

        # Process response
        action = response.get("action", "")
        response_review_id = response.get("review_id", "")

        if response_review_id != review_id:
            logger.warning(
                "PLAN_REVIEW_ID_MISMATCH",
                expected=review_id,
                received=response_review_id,
            )
            # Continue with original plan

        if action == "reject":
            logger.info(
                "PLAN_REVIEW_REJECTED",
                review_id=review_id,
                reason=response.get("rejection_reason", ""),
            )
            # Set state to indicate rejection
            state.is_cancelled = True
            return

        if action == "approve_with_edits":
            edited_plan_dict = response.get("edited_plan")
            if edited_plan_dict:
                # Parse and apply edits
                edited_plan = PlanForReview.model_validate(edited_plan_dict)
                apply_user_edits(state, edited_plan)
                logger.info(
                    "PLAN_REVIEW_EDITS_APPLIED",
                    review_id=review_id,
                )

        # action == "approve" or edits applied - continue with research
        logger.info(
            "PLAN_REVIEW_APPROVED",
            review_id=review_id,
            action=action,
        )

    except TimeoutError:
        logger.info(
            "PLAN_REVIEW_TIMEOUT",
            review_id=review_id,
            timeout_seconds=config.plan_review_timeout_seconds,
        )
        yield PlanReviewTimeoutEvent(
            review_id=review_id,
            timeout_seconds=config.plan_review_timeout_seconds,
        )
        # Continue with original plan


# Helper functions for creating events
async def _buffer_event(
    event: StreamEvent, event_buffer: "EventBuffer | None"
) -> None:
    """Add event to buffer if available (for database persistence)."""
    if event_buffer is not None:
        await event_buffer.add_event(event)


def _agent_started(agent: str, tier: str) -> AgentStartedEvent:
    return AgentStartedEvent(agent=agent, model_tier=tier)


def _agent_completed(agent: str, start_time: float) -> AgentCompletedEvent:
    return AgentCompletedEvent(
        agent=agent,
        duration_ms=int((time.perf_counter() - start_time) * 1000),
    )


def _plan_id_to_uuid(plan_id: str) -> UUID:
    """Convert plan ID to UUID, handling non-UUID strings."""
    try:
        return UUID(plan_id)
    except ValueError:
        # Convert non-UUID string to deterministic UUID using uuid5
        from uuid import NAMESPACE_DNS, uuid5
        return uuid5(NAMESPACE_DNS, plan_id)


def _to_claim_uuid(value: int | str | UUID | None) -> UUID:
    """Convert claim_id (may be int, str, or UUID) to UUID.

    The citation pipeline uses id(claim) which returns a Python object ID (integer).
    This function converts any such identifier to a deterministic UUID for the
    event schema which expects UUID type.
    """
    from uuid import NAMESPACE_DNS, uuid5

    if isinstance(value, UUID):
        return value
    if value is None:
        return UUID(int=0)
    # Convert int or str to deterministic UUID
    return uuid5(NAMESPACE_DNS, str(value))


def _plan_created(state: ResearchState) -> PlanCreatedEvent:
    plan = state.current_plan
    if not plan:
        raise ValueError("No plan in state")

    return PlanCreatedEvent(
        plan_id=_plan_id_to_uuid(plan.id),
        title=plan.title,
        thought=plan.thought,
        steps=[
            PlanStepSummary(
                id=s.id,
                title=s.title,
                step_type=s.step_type.value,
                needs_search=s.needs_search,
            )
            for s in plan.steps
        ],
        iteration=plan.iteration,
    )


def _step_started(state: ResearchState) -> StepStartedEvent:
    step = state.get_current_step()
    if not step:
        raise ValueError("No current step")

    return StepStartedEvent(
        step_index=state.current_step_index,
        step_id=step.id,
        step_title=step.title,
        step_type=step.step_type.value,
    )


def _step_completed(state: ResearchState) -> StepCompletedEvent:
    # Safe access with bounds check
    step_id = ""
    if (
        state.current_plan
        and 0 <= state.current_step_index < len(state.current_plan.steps)
    ):
        step_id = state.current_plan.steps[state.current_step_index].id

    file_source_count = sum(
        1 for s in state.sources
        if s.url and s.url.startswith("uploaded-file://")
    )

    return StepCompletedEvent(
        step_index=state.current_step_index,
        step_id=step_id,
        observation_summary=state.last_observation[:200] if state.last_observation else "",
        sources_found=len(state.sources),
        file_sources_found=file_source_count,
    )


def _reflection_decision(state: ResearchState) -> ReflectionDecisionEvent:
    if not state.last_reflection:
        raise ValueError("No reflection in state")

    return ReflectionDecisionEvent(
        decision=state.last_reflection.decision.value,
        reasoning=state.last_reflection.reasoning,
        suggested_changes=state.last_reflection.suggested_changes,
    )


async def _stream_research_with_custom_phases(
    query: str,
    state: ResearchState,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    crawler: WebCrawler,
    plugin_manager: "PluginManager",
    customization: "PipelineCustomization",
    config: OrchestrationConfig,
    db: "AsyncSession | None",
    chat_id: str | None,
    user_id: str | None,
    start_time: float,
    plugin_data: dict[str, Any] | None = None,
) -> AsyncGenerator[StreamEvent | str, None]:
    """Execute research using custom phases instead of planner.

    This mode is activated when a plugin:
    1. Disables the planner via disabled_agents={"planner"}
    2. Provides custom phases via PhaseProvider protocol
    3. Specifies phase execution order via phase_insertions

    Args:
        query: User's research query
        state: Initial research state
        llm: LLM client for completions
        brave_client: Brave Search client
        crawler: Web crawler
        plugin_manager: PluginManager with registered phases
        customization: PipelineCustomization from plugin
        config: Orchestration configuration
        db: Optional database session
        chat_id: Optional chat ID
        user_id: Optional user ID
        start_time: Start time for duration tracking
        plugin_data: Optional pre-extracted context data. When provided,
            bypasses query extraction. Pass structured data from SFDC here.

    Yields:
        StreamEvent objects for custom phase execution
    """
    from deep_research.agent.pipeline.phase_executor import PhaseExecutor
    from deep_research.agent.tools.base import ResearchContext

    phases = plugin_manager.get_all_phases()

    # Emit custom phase mode started
    yield CustomPhaseModeStartedEvent(
        total_phases=len(phases),
        phase_names=list(phases.keys()),
    )

    # Use provided plugin_data if available, otherwise extract from query
    if plugin_data:
        # Structured data passed from caller (e.g., SFDC account info)
        # This is the preferred path - no regex, no guessing
        resolved_plugin_data = plugin_data
        logger.info(
            "CUSTOM_PHASE_MODE_USING_PROVIDED_PLUGIN_DATA",
            keys=list(plugin_data.keys()),
            company_name=plugin_data.get("company_name") or plugin_data.get("account_name"),
        )
    else:
        # Fallback: extract from query (backward compatibility)
        resolved_plugin_data = _extract_context_from_query(query)
        logger.info(
            "CUSTOM_PHASE_MODE_EXTRACTED_PLUGIN_DATA",
            keys=list(resolved_plugin_data.keys()),
            company_name=resolved_plugin_data.get("company_name"),
        )

    # Inject original query into plugin_data so phases can access it
    # This allows phases to see the full user intent, not just extracted fields
    resolved_plugin_data["original_query"] = query

    # Create research context with plugin_data and services
    context = ResearchContext(
        chat_id=state.session_id,
        user_id=user_id or "system",
        research_type=state.resolve_depth(),
        plugin_data=resolved_plugin_data,
        llm=llm,
        brave_client=brave_client,
        crawler=crawler,
    )

    steps_executed = 0

    # Run coordinator first (query classification)
    yield _agent_started("coordinator", "simple")
    agent_start = time.perf_counter()
    state = await run_coordinator(state, llm)
    yield _agent_completed("coordinator", agent_start)

    # Handle simple queries
    if state.is_simple_query and state.direct_response:
        state.complete(state.direct_response)
        yield ResearchCompletedEvent(
            session_id=state.session_id,
            total_steps_executed=0,
            total_steps_skipped=0,
            plan_iterations=0,
            total_duration_ms=int((time.perf_counter() - start_time) * 1000),
            final_report=state.final_report,
        )
        return

    # Optional: Background investigation
    if config.enable_background_investigation:
        yield _agent_started("background_investigator", "simple")
        agent_start = time.perf_counter()
        async for react_event in run_background_investigator(state, llm, brave_client):
            stream_evt = _convert_react_event(react_event)
            if stream_evt:
                yield stream_evt
        yield _agent_completed("background_investigator", agent_start)

    # Execute custom phases
    executor = PhaseExecutor(phases=phases, customization=customization)

    async for phase_event, state in executor.execute_all(context, state):
        if phase_event.event_type == "started":
            phase_description = ""
            if phase_event.phase_name in phases:
                phase_description = phases[phase_event.phase_name].description
            yield PhaseStartedEvent(
                phase_name=phase_event.phase_name,
                description=phase_description,
            )
        elif phase_event.event_type == "completed":
            steps_executed += 1
            yield PhaseCompletedEvent(
                phase_name=phase_event.phase_name,
                duration_ms=phase_event.duration_ms,
                sources_count=phase_event.sources_count,
            )
        elif phase_event.event_type == "skipped":
            yield PhaseSkippedEvent(phase_name=phase_event.phase_name)
        elif phase_event.event_type == "error":
            yield PhaseErrorEvent(
                phase_name=phase_event.phase_name,
                error=phase_event.error or "Unknown error",
            )

    # Apply synthesizer override if specified
    synthesizer_prompt = None
    if "synthesizer" in customization.agent_overrides:
        override = customization.agent_overrides["synthesizer"]
        synthesizer_prompt = override.get("prompt_override")
        if synthesizer_prompt:
            state.structured_system_prompt = synthesizer_prompt
            logger.info("Applied custom synthesizer prompt from plugin")

    # Run synthesis
    yield SynthesisStartedEvent(
        total_observations=len(state.all_observations),
        total_sources=len(state.sources),
    )
    yield _agent_started("synthesizer", "complex")
    agent_start = time.perf_counter()

    # Use appropriate synthesizer based on output format
    if state.output_format == "json" and state.output_schema:
        state = await run_structured_synthesizer(state, llm)
        # Run post-verification if enabled
        if state.enable_post_verification and state.enable_citation_verification:
            state = await post_verify_structured_output(state, llm)
    else:
        # Streaming synthesis
        # Note: stream_synthesis() internally calls state.complete()
        # so we don't need to call it again here
        async for chunk in stream_synthesis(state, llm):
            yield SynthesisProgressEvent(content_chunk=chunk)

    yield _agent_completed("synthesizer", agent_start)

    # Emit completion
    total_duration_ms = (time.perf_counter() - start_time) * 1000
    yield ResearchCompletedEvent(
        session_id=state.session_id,
        total_steps_executed=steps_executed,
        total_steps_skipped=0,
        plan_iterations=0,
        total_duration_ms=int(total_duration_ms),
        final_report=state.final_report,
        structured_output=(
            state.final_report_structured.model_dump()
            if state.final_report_structured else None
        ),
    )


def _extract_context_from_query(query: str) -> dict[str, str | list | None]:
    """Extract company and attendee context from query.

    Parses patterns like:
    - "meeting with Head of AI of Depop"
    - "prep for Acme Corp"
    - "company: Microsoft"

    Args:
        query: User's research query

    Returns:
        Dict with company_name, account_name, and attendees
    """
    import re

    context: dict[str, str | list | None] = {
        "company_name": None,
        "account_name": None,
        "attendees": [],
    }

    # Extract company name patterns
    patterns = [
        r"(?:meeting|prep)\s+(?:with|for)\s+(?:[\w\s]+?\s+(?:of|at|from)\s+)?(\w+)",
        r"(?:company|customer|account):\s*(\w+)",
        r"with\s+(\w+)\s+(?:team|leadership|executives)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            company = match.group(1)
            # Filter out common false positives
            if company.lower() not in {
                "the", "a", "an", "our", "their", "head", "vp", "director"
            }:
                context["company_name"] = company
                context["account_name"] = company
                break

    # Extract attendee titles
    attendee_pattern = r"((?:Head|VP|Director|CTO|CEO|CDO|Chief)\s+(?:of\s+)?\w+)"
    matches = re.findall(attendee_pattern, query, re.IGNORECASE)
    attendees: list[dict[str, str]] = []
    for match in matches:
        attendees.append({"title": match})
    context["attendees"] = attendees

    return context
