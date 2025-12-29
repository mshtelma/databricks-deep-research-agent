"""Multi-agent orchestrator - coordinates the 5-agent research workflow."""

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from uuid import UUID

import mlflow

from src.agent.config import get_coordinator_config, get_planner_config
from src.agent.nodes.background import run_background_investigator
from src.agent.nodes.coordinator import handle_simple_query, run_coordinator
from src.agent.nodes.planner import run_planner
from src.agent.nodes.react_researcher import ReactResearchEvent, run_react_researcher
from src.agent.nodes.reflector import run_reflector
from src.agent.nodes.researcher import run_researcher
from src.agent.nodes.citation_synthesizer import (
    run_citation_synthesizer,
    stream_synthesis_with_citations,
)
from src.agent.nodes.synthesizer import run_synthesizer, stream_synthesis
from src.agent.state import ReflectionDecision, ReflectionResult, ResearchState
from src.agent.tools.web_crawler import WebCrawler
from src.core.logging_utils import (
    get_logger,
    log_agent_phase,
    log_agent_transition,
    truncate,
)
from src.schemas.research import PlanStepSummary
from src.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    ClaimVerifiedEvent,
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    ResearchCompletedEvent,
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
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient

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
    research_depth: str = "auto"  # auto, light, medium, extended
    system_instructions: str | None = None  # User's custom system instructions
    # Persistence context (for claim/citation storage)
    message_id: UUID | None = None  # Agent message ID for claims
    research_session_id: UUID | None = None  # Research session ID for sources


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
        research_depth=config.research_depth,
        system_instructions=config.system_instructions,
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

                        # Research step - use ReAct researcher with agentic tool use
                        log_agent_transition(
                            logger,
                            from_agent="planner",
                            to_agent="researcher",
                            reason=f"step {state.current_step_index + 1}",
                        )
                        events.append(_agent_started("researcher", "analytical"))
                        agent_start = time.perf_counter()

                        # Run ReAct researcher and collect events
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

            # Use citation-aware synthesizer if enabled
            if state.enable_citation_verification:
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
        )
    )

    return OrchestrationResult(
        state=state,
        events=events,
        total_duration_ms=total_duration_ms,
        steps_executed=steps_executed,
        steps_skipped=steps_skipped,
    )


@mlflow.trace(name="stream_research_orchestration", span_type="CHAIN")
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
        research_depth=config.research_depth,
        system_instructions=config.system_instructions,
    )
    if session_id:
        state.session_id = session_id

    steps_executed = 0
    steps_skipped = 0

    # Set root span attributes for trace correlation (span created by @mlflow.trace decorator)
    root_span = mlflow.get_current_active_span()
    if root_span:
        root_span.set_attributes({
            "research.session_id": str(state.session_id),
            "research.query": truncate(query, 200),
            "research.max_iterations": config.max_plan_iterations,
            "research.streaming": True,
            "research.enable_background": config.enable_background_investigation,
            "research.enable_clarification": config.enable_clarification,
        })

    # Group traces by user and chat session for MLflow trace correlation
    # Must be called after span is created by decorator
    if user_id or chat_id:
        trace_metadata: dict[str, str] = {}
        if user_id:
            trace_metadata["mlflow.trace.user"] = user_id
        if chat_id:
            trace_metadata["mlflow.trace.session"] = chat_id
        mlflow.update_current_trace(metadata=trace_metadata)

    try:
        # Phase 1: Coordinator
        log_agent_transition(logger, from_agent=None, to_agent="coordinator")
        yield _agent_started("coordinator", "simple")
        agent_start = time.perf_counter()

        state = await run_coordinator(state, llm)

        yield _agent_completed("coordinator", agent_start)

        # Handle simple queries
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
                yield _agent_started("background_investigator", "simple")
                agent_start = time.perf_counter()
                state = await run_background_investigator(state, llm, brave_client)
                yield _agent_completed("background_investigator", agent_start)

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
                yield _agent_started("planner", "analytical")
                agent_start = time.perf_counter()
                state = await run_planner(state, llm)
                yield _agent_completed("planner", agent_start)

                if state.current_plan:
                    yield _plan_created(state)

                    if state.current_plan.has_enough_context:
                        break

                    while state.has_more_steps() and not state.is_cancelled:
                        step = state.get_current_step()
                        if not step:
                            break

                        yield _step_started(state)

                        log_agent_transition(
                            logger,
                            from_agent="planner",
                            to_agent="researcher",
                            reason=f"step {state.current_step_index + 1}",
                        )
                        yield _agent_started("researcher", "analytical")
                        agent_start = time.perf_counter()

                        # Run ReAct researcher with agentic tool use
                        async for react_event in run_react_researcher(
                            state, llm, crawler, brave_client
                        ):
                            if react_event.event_type == "tool_call":
                                yield ToolCallEvent(
                                    tool_name=react_event.data.get("tool", ""),
                                    tool_args=react_event.data.get("args", {}),
                                    call_number=react_event.data.get("call_number", 0),
                                )
                            elif react_event.event_type == "tool_result":
                                yield ToolResultEvent(
                                    tool_name=react_event.data.get("tool", ""),
                                    result_preview=react_event.data.get("result_preview", "")[:200],
                                    sources_crawled=react_event.data.get("high_quality_count", 0),
                                )
                            elif react_event.event_type == "research_complete":
                                logger.info(
                                    "REACT_RESEARCH_COMPLETE",
                                    reason=react_event.data.get("reason", ""),
                                    tool_calls=react_event.data.get("tool_calls", 0),
                                    high_quality=react_event.data.get("high_quality_sources", 0),
                                )

                        yield _agent_completed("researcher", agent_start)
                        steps_executed += 1

                        yield _step_completed(state)

                        log_agent_transition(logger, from_agent="researcher", to_agent="reflector")
                        yield _agent_started("reflector", "simple")
                        agent_start = time.perf_counter()
                        state = await run_reflector(state, llm)
                        yield _agent_completed("reflector", agent_start)

                        if state.last_reflection:
                            yield _reflection_decision(state)

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
            yield SynthesisStartedEvent(
                total_observations=len(state.all_observations),
                total_sources=len(state.sources),
            )

            yield _agent_started("synthesizer", "complex")
            agent_start = time.perf_counter()

            # Use citation-aware synthesizer if enabled
            # Collect content chunks for persistence
            content_chunks: list[str] = []
            if state.enable_citation_verification:
                async for event in stream_synthesis_with_citations(state, llm):
                    event_type = event.get("type", "")
                    if event_type == "content":
                        chunk = event.get("chunk", "")
                        content_chunks.append(chunk)
                        yield SynthesisProgressEvent(content_chunk=chunk)
                    # Log verification events (frontend fetches claims via API)
                    elif event_type in ("claim_verified", "verification_summary", "citation_corrected"):
                        logger.debug(
                            f"CITATION_EVENT_{event_type.upper()}",
                            **{k: str(v)[:50] for k, v in event.items() if k != "type"},
                        )
            else:
                async for chunk in stream_synthesis(state, llm):
                    content_chunks.append(chunk)
                    yield SynthesisProgressEvent(content_chunk=chunk)

            # Aggregate content and update state
            if content_chunks:
                final_content = "".join(content_chunks)
                state.complete(final_content)

            yield _agent_completed("synthesizer", agent_start)

            # Persist all research data if db session available and we have content
            if (
                db is not None
                and config.message_id is not None
                and config.research_session_id is not None
                and state.final_report
                and chat_id is not None
            ):
                try:
                    from src.agent.persistence import persist_complete_research

                    counts = await persist_complete_research(
                        db=db,
                        chat_id=UUID(chat_id) if isinstance(chat_id, str) else chat_id,
                        user_query=query,
                        message_id=config.message_id,
                        research_session_id=config.research_session_id,
                        research_depth=config.research_depth,
                        state=state,
                    )
                    logger.info(
                        "RESEARCH_DATA_PERSISTED",
                        message_id=str(config.message_id),
                        research_session_id=str(config.research_session_id),
                        report_len=len(state.final_report),
                        claims=counts.get("claims", 0),
                        citations=counts.get("citations", 0),
                        sources=counts.get("sources", 0),
                    )
                except Exception as e:
                    logger.warning(
                        "RESEARCH_PERSISTENCE_FAILED",
                        error=str(e)[:200],
                        message_id=str(config.message_id) if config.message_id else None,
                    )

    except Exception as e:
        logger.exception(
            "STREAM_ORCHESTRATION_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        yield StreamErrorEvent(
            error_code="ORCHESTRATION_ERROR",
            error_message=str(e),
            recoverable=False,
        )

    total_duration_ms = (time.perf_counter() - start_time) * 1000

    yield ResearchCompletedEvent(
        session_id=state.session_id,
        total_steps_executed=steps_executed,
        total_steps_skipped=steps_skipped,
        plan_iterations=state.plan_iterations,
        total_duration_ms=int(total_duration_ms),
    )


# Helper functions for creating events
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
