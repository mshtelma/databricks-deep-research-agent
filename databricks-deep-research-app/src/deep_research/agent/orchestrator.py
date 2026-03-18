"""Multi-agent orchestrator — public API surface.

Delegates all work to ``framework_orchestrator`` via the framework runtime.
Config types live in ``orchestration_config`` and are re-exported here for
backward compatibility.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.agent.orchestration_config import (
    OrchestrationConfig,
    OrchestrationResult,
    apply_custom_agent_to_config,
    get_default_orchestration_config,
)
from deep_research.agent.state import ResearchState
from deep_research.schemas.streaming import ResearchCompletedEvent, StreamEvent
from deep_research.services.llm.client import LLMClient
from deep_research.services.search.brave import BraveSearchClient

if TYPE_CHECKING:
    from deep_research.agent.tools.web_crawler import WebCrawler

# Backward-compat alias (was the private name in the monolith)
_get_default_orchestration_config = get_default_orchestration_config

# Re-export so existing ``from deep_research.agent.orchestrator import …``
# statements keep working without edits.
__all__ = [
    "OrchestrationConfig",
    "OrchestrationResult",
    "apply_custom_agent_to_config",
    "get_default_orchestration_config",
    "_get_default_orchestration_config",  # backward-compat alias; callers use the private name
    "run_research",
    "stream_research",
]


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
    db: Any | None = None,
    plugin_manager: Any | None = None,
    plugin_data: dict[str, Any] | None = None,
) -> AsyncGenerator[StreamEvent | str, None]:
    """Stream the research workflow with real-time events.

    Delegates entirely to the framework orchestrator.

    Yields:
        StreamEvent objects and synthesis content chunks.
    """
    config = config or get_default_orchestration_config()

    from deep_research.agent.framework_orchestrator import (
        stream_research_via_framework,
    )

    async for event in stream_research_via_framework(
        query=query,
        llm=llm,
        brave_client=brave_client,
        crawler=crawler,
        conversation_history=conversation_history,
        session_id=session_id,
        user_id=user_id,
        chat_id=chat_id,
        config=config,
        db=db,
        plugin_manager=plugin_manager,
        plugin_data=plugin_data,
    ):
        yield event


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

    Delegates to ``stream_research()`` and collects the results
    into an ``OrchestrationResult``.

    Returns:
        OrchestrationResult with final state and events.
    """
    config = config or get_default_orchestration_config()
    start_time = time.perf_counter()

    events: list[StreamEvent] = []
    steps_executed = 0
    steps_skipped = 0
    final_report: str | None = None
    plan_iterations = 0

    async for event in stream_research(
        query=query,
        llm=llm,
        brave_client=brave_client,
        crawler=crawler,
        conversation_history=conversation_history,
        session_id=session_id,
        user_id=user_id,
        chat_id=chat_id,
        config=config,
    ):
        if isinstance(event, str):
            # Synthesis chunk
            continue
        if isinstance(event, StreamEvent):
            events.append(event)
            if isinstance(event, ResearchCompletedEvent):
                steps_executed = event.total_steps_executed
                steps_skipped = event.total_steps_skipped
                plan_iterations = event.plan_iterations
                final_report = event.final_report

    # Build a minimal state for OrchestrationResult compatibility
    state = ResearchState(query=query)
    if final_report:
        state.complete(final_report)
    if session_id:
        state.session_id = session_id

    return OrchestrationResult(
        state=state,
        events=events,
        total_duration_ms=(time.perf_counter() - start_time) * 1000,
        steps_executed=steps_executed,
        steps_skipped=steps_skipped,
    )
