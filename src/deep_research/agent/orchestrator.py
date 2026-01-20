"""Multi-agent orchestrator - coordinates the 5-agent research workflow."""

import asyncio
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import mlflow

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
from deep_research.agent.nodes.react_researcher import run_react_researcher
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
from deep_research.core.app_config import ResearcherMode
from deep_research.core.exceptions import StructuredSynthesisError
from deep_research.core.logging_utils import (
    get_logger,
    log_agent_phase,
    log_agent_transition,
    truncate,
)
from deep_research.core.tracing import log_research_config
from deep_research.schemas.research import PlanStepSummary
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    CitationCorrectedEvent,
    ClaimVerifiedEvent,
    NumericClaimDetectedEvent,
    PersistenceCompletedEvent,
    PlanCreatedEvent,
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
    VerificationSummaryEvent,
)
from deep_research.services.llm.client import LLMClient
from deep_research.services.search.brave import BraveSearchClient
from deep_research.services.research_event_buffer import EventBuffer

# Import database session type for persistence (optional dependency)
try:
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:
    AsyncSession = None  # type: ignore[misc, assignment]

logger = get_logger(__name__)


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
    )
    if session_id:
        state.session_id = session_id

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
        with mlflow.start_run(run_name=f"research_{str(state.session_id)[:8]}", nested=True):
            with mlflow.start_span(name="research_orchestration", span_type="CHAIN") as root_span:
                # Add research context to root span for trace correlation
                root_span.set_attributes({
                    "research.session_id": str(state.session_id),
                    "research.query": truncate(query, 200),
                    "research.max_iterations": config.max_plan_iterations,
                    "research.enable_background": config.enable_background_investigation,
                    "research.enable_clarification": config.enable_clarification,
                })

                # Group traces by user and chat session for MLflow trace correlation
                if user_id or chat_id:
                    trace_metadata: dict[str, str] = {}
                    if user_id:
                        trace_metadata["mlflow.trace.user"] = user_id
                    if chat_id:
                        trace_metadata["mlflow.trace.session"] = chat_id
                    mlflow.update_current_trace(metadata=trace_metadata)

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

                    state = await run_background_investigator(state, llm, brave_client)

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
                                async for react_event in run_react_researcher(
                                    state, llm, crawler, brave_client
                                ):
                                    if react_event.event_type == "tool_call":
                                        events.append(
                                            ToolCallEvent(
                                                tool_name=react_event.data.get("tool", ""),
                                                tool_args=react_event.data.get("args", {}),
                                                call_number=react_event.data.get("call_number", 0),
                                            )
                                        )
                                    elif react_event.event_type == "tool_result":
                                        events.append(
                                            ToolResultEvent(
                                                tool_name=react_event.data.get("tool", ""),
                                                result_preview=react_event.data.get("result_preview", "")[:200],
                                                sources_crawled=react_event.data.get("high_quality_count", 0),
                                            )
                                        )
                                    elif react_event.event_type == "research_complete":
                                        logger.info(
                                            "REACT_RESEARCH_COMPLETE",
                                            reason=react_event.data.get("reason", ""),
                                            tool_calls=react_event.data.get("tool_calls", 0),
                                            high_quality=react_event.data.get("high_quality_sources", 0),
                                        )
                            else:
                                # Classic mode: single-pass fixed searches/crawls
                                state = await run_researcher(
                                    state, llm, crawler, brave_client
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
) -> AsyncGenerator[StreamEvent | str, None]:
    """Stream the research workflow with real-time events.

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
    with mlflow.start_run(run_name=f"research_{str(state.session_id)[:8]}", nested=True):
        # Create span INSIDE the run context so trace is properly nested under run
        with mlflow.start_span(name="stream_research_orchestration", span_type="CHAIN") as root_span:
            # Add research context to root span for trace correlation
            root_span.set_attributes({
                "research.session_id": str(state.session_id),
                "research.query": truncate(query, 200),
                "research.max_iterations": config.max_plan_iterations,
                "research.streaming": True,
                "research.enable_background": config.enable_background_investigation,
                "research.enable_clarification": config.enable_clarification,
            })

            # Group traces by user and chat session for MLflow trace correlation
            if user_id or chat_id:
                trace_metadata: dict[str, str] = {}
                if user_id:
                    trace_metadata["mlflow.trace.user"] = user_id
                if chat_id:
                    trace_metadata["mlflow.trace.session"] = chat_id
                mlflow.update_current_trace(metadata=trace_metadata)

            try:
                # =============================================================
                # Query Mode Routing (Tiered Query Modes feature)
                # =============================================================
                # SIMPLE mode: Direct LLM response, skip coordinator entirely
                # WEB_SEARCH mode: Lightweight pipeline (handled below in T022)
                # DEEP_RESEARCH mode: Full pipeline (existing flow)
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
                        from deep_research.agent.persistence import persist_simple_message_independent

                        try:
                            chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id
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

                    # Get web search mode config (includes timeout_seconds)
                    mode_config = get_query_mode_config("web_search")
                    web_search_timeout = getattr(mode_config, "timeout_seconds", 15)

                    # Track start time for timeout
                    web_search_start = time.perf_counter()

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
                        yield _plan_created(state)

                        # 2. Run researcher with minimal configuration and timeout
                        yield _step_started(state)
                        log_agent_transition(logger, from_agent=None, to_agent="researcher")
                        yield _agent_started("researcher", "analytical")
                        agent_start = time.perf_counter()

                        # Use classic researcher - it loads limits from depth config
                        # For web search, we set effective_depth to 'light' to get minimal limits
                        state.effective_depth = "light"  # Override to use light depth limits

                        # Wrap researcher in timeout
                        try:
                            state = await asyncio.wait_for(
                                run_researcher(
                                    state,
                                    llm,
                                    crawler,
                                    brave_client,
                                ),
                                timeout=web_search_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "WEB_SEARCH_RESEARCHER_TIMEOUT",
                                elapsed_seconds=time.perf_counter() - web_search_start,
                                timeout_seconds=web_search_timeout,
                            )
                            raise  # Re-raise to trigger fallback

                        yield _agent_completed("researcher", agent_start)

                        # Mark step as complete (skip reflector - always COMPLETE for web search)
                        state.mark_step_complete(state.last_observation)
                        yield _step_completed(state)
                        state.advance_step()
                        steps_executed = 1

                        # 3. Synthesize with natural mode ([1], [2] citations)
                        yield SynthesisStartedEvent(
                            total_observations=1,
                            total_sources=len(state.sources),
                        )
                        log_agent_transition(logger, from_agent="researcher", to_agent="synthesizer")
                        yield _agent_started("synthesizer", "analytical")
                        agent_start = time.perf_counter()

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
                                    yield SynthesisProgressEvent(content_chunk=chunk)
                                elif event_type == "claim_verified":
                                    # Convert claim_id to UUID - generate new one if not valid UUID
                                    raw_claim_id = event_dict.get("claim_id")
                                    if isinstance(raw_claim_id, UUID):
                                        claim_id_uuid = raw_claim_id
                                    else:
                                        # Generate new UUID - claim_id from synthesis may be
                                        # an index or non-UUID string
                                        claim_id_uuid = uuid4()
                                    yield ClaimVerifiedEvent(
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
                                elif event_type == "verification_summary":
                                    yield VerificationSummaryEvent(
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

                            full_report = "".join(web_search_chunks)
                            state.complete(full_report)

                        yield _agent_completed("synthesizer", agent_start)

                        # Emit completion event
                        total_duration_ms = (time.perf_counter() - start_time) * 1000
                        yield ResearchCompletedEvent(
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
                            from deep_research.agent.persistence import persist_complete_research_independent

                            try:
                                await asyncio.shield(
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
                            yield PersistenceCompletedEvent(
                                chat_id=chat_id,
                                message_id=str(config.message_id),
                                research_session_id=str(config.research_session_id),
                                chat_title=query[:50] + "..." if len(query) > 50 else query,
                                was_draft=True,  # Web search always creates new session
                                counts={"sources": len(state.sources)},
                            )
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
                            yield StreamErrorEvent(
                                error_code="PERSISTENCE_SKIPPED",
                                error_message="Web search completed but could not persist to database",
                                recoverable=True,
                                stack_trace="".join(traceback.format_stack()),
                            )

                        return

                    except asyncio.TimeoutError:
                        # Web search timed out - fall back to Simple mode
                        logger.warning(
                            "WEB_SEARCH_TIMEOUT_FALLBACK",
                            elapsed_seconds=time.perf_counter() - web_search_start,
                            timeout_seconds=web_search_timeout,
                            query=truncate(query, 100),
                        )

                        # Notify frontend of fallback
                        yield StreamErrorEvent(
                            error_code="WEB_SEARCH_TIMEOUT",
                            error_message="Web search timed out, falling back to direct answer",
                            recoverable=True,
                            stack_trace="".join(traceback.format_stack()),
                        )

                        # Fall back to Simple mode (direct LLM response)
                        yield SynthesisStartedEvent(total_observations=0, total_sources=0)
                        yield _agent_started("synthesizer", "simple")
                        fallback_start = time.perf_counter()

                        fallback_chunks: list[str] = []
                        async for chunk in handle_simple_query(state, llm):
                            fallback_chunks.append(chunk)
                            yield SynthesisProgressEvent(content_chunk=chunk)

                        full_report = "".join(fallback_chunks)
                        yield _agent_completed("synthesizer", fallback_start)
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

                        # Persist fallback response (same pattern as simple mode)
                        # Use asyncio.shield with independent session to survive cancellation
                        if (
                            db is not None
                            and config.message_id is not None
                            and chat_id is not None
                            and user_id is not None
                        ):
                            from deep_research.agent.persistence import persist_simple_message_independent

                            try:
                                chat_id_uuid = UUID(chat_id) if isinstance(chat_id, str) else chat_id
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
                                    "WEB_SEARCH_FALLBACK_PERSISTED",
                                    message_id=str(config.message_id),
                                    content_len=len(full_report),
                                )

                                # Emit persistence_completed event for frontend
                                chat_title = query[:47] + "..." if len(query) > 50 else query
                                yield PersistenceCompletedEvent(
                                    chat_id=str(chat_id_uuid),
                                    message_id=str(config.message_id),
                                    research_session_id=None,  # No research session for fallback
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
                            yield StreamErrorEvent(
                                error_code="PERSISTENCE_SKIPPED",
                                error_message="Web search fallback response could not persist to database",
                                recoverable=True,
                                stack_trace="".join(traceback.format_stack()),
                            )

                    return

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
                    from deep_research.agent.persistence import persist_research_session_start_independent

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
                        state = await run_background_investigator(state, llm, brave_client)
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
                                    async for react_event in run_react_researcher(
                                        state, llm, crawler, brave_client
                                    ):
                                        if react_event.event_type == "tool_call":
                                            evt = ToolCallEvent(
                                                tool_name=react_event.data.get("tool", ""),
                                                tool_args=react_event.data.get("args", {}),
                                                call_number=react_event.data.get("call_number", 0),
                                            )
                                            yield evt
                                            await _buffer_event(evt, event_buffer)
                                        elif react_event.event_type == "tool_result":
                                            evt = ToolResultEvent(
                                                tool_name=react_event.data.get("tool", ""),
                                                result_preview=react_event.data.get("result_preview", "")[:200],
                                                sources_crawled=react_event.data.get("high_quality_count", 0),
                                            )
                                            yield evt
                                            await _buffer_event(evt, event_buffer)
                                        elif react_event.event_type == "research_complete":
                                            logger.info(
                                                "REACT_RESEARCH_COMPLETE",
                                                reason=react_event.data.get("reason", ""),
                                                tool_calls=react_event.data.get("tool_calls", 0),
                                                high_quality=react_event.data.get("high_quality_sources", 0),
                                            )
                                else:
                                    # Classic mode: single-pass fixed searches/crawls
                                    state = await run_researcher(
                                        state, llm, crawler, brave_client
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

                    # Validate and complete - only for streaming modes
                    # (structured output is validated by Pydantic and completed internally)
                    # Also validate when structured synthesis failed and fell back to streaming
                    if state.output_format != "json" or structured_synthesis_failed:
                        # Validate synthesis produced content - empty synthesis is an error
                        if not content_chunks:
                            logger.error(
                                "SYNTHESIS_EMPTY: Synthesis produced no content chunks",
                                enable_citation_verification=state.enable_citation_verification,
                                total_observations=len(state.all_observations),
                                total_sources=len(state.sources),
                            )
                            raise RuntimeError("Synthesis produced no content")

                        # Aggregate content and update state
                        final_content = "".join(content_chunks)
                        if not final_content.strip():
                            logger.error(
                                "SYNTHESIS_WHITESPACE_ONLY: Synthesis produced only whitespace content",
                                content_len=len(final_content),
                            )
                            raise RuntimeError("Synthesis produced empty or whitespace-only content")

                        state.complete(final_content)

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
                            from deep_research.agent.persistence import persist_research_session_complete_update_independent

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
                                from deep_research.agent.persistence import persist_research_session_failed_independent
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
                            from deep_research.agent.persistence import persist_complete_research_independent

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
                    from deep_research.agent.persistence import persist_research_session_failed_independent

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

    return StepCompletedEvent(
        step_index=state.current_step_index,
        step_id=step_id,
        observation_summary=state.last_observation[:200] if state.last_observation else "",
        sources_found=len(state.sources),
    )


def _reflection_decision(state: ResearchState) -> ReflectionDecisionEvent:
    if not state.last_reflection:
        raise ValueError("No reflection in state")

    return ReflectionDecisionEvent(
        decision=state.last_reflection.decision.value,
        reasoning=state.last_reflection.reasoning,
        suggested_changes=state.last_reflection.suggested_changes,
    )
