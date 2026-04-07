"""Databricks Agent handlers.

Implements @invoke and @stream handlers that wrap the research orchestrator
for Databricks Agent Server deployment.
"""

import asyncio
import logging
from collections.abc import Generator
from typing import Any
from uuid import uuid4

from deep_research.agent.orchestrator import (
    OrchestrationConfig,
    run_research,
    stream_research,
)
from deep_research.agent_server.utils import (
    extract_messages,
)

logger = logging.getLogger(__name__)


def _create_services() -> tuple[Any, Any, Any]:
    """Create shared services (LLM, Brave, Crawler) for agent handlers.

    Returns:
        Tuple of (LLMClient, BraveSearchClient, WebCrawler).
    """
    from deep_research.agent.tools.web_crawler import WebCrawler
    from deep_research.services.llm.client import LLMClient
    from deep_research.services.llm.config import ModelConfig
    from deep_research.services.search.brave import BraveSearchClient

    model_config = ModelConfig()
    llm = LLMClient(model_config)
    brave_client = BraveSearchClient()
    crawler = WebCrawler()
    return llm, brave_client, crawler


def invoke(
    request: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke handler for synchronous research.

    This handler is called for non-streaming requests.

    Args:
        request: Request containing messages array.
        context: Optional context with user info, trace ID, etc.

    Returns:
        Response dict with content and metadata.
    """
    # Extract user query and conversation history
    messages = extract_messages(request)
    if not messages:
        return {
            "content": "No query provided. Please ask a research question.",
            "metadata": {"error": "empty_query"},
        }

    # Get the latest user message as query
    query = messages[-1].get("content", "")
    conversation_history = messages[:-1] if len(messages) > 1 else []

    # Extract context info for logging
    session_id = uuid4()
    user_id = context.get("user_id") if context else None
    logger.info(f"Invoke request from user={user_id}, session={session_id}")

    # Configure orchestration
    config = OrchestrationConfig(
        max_plan_iterations=3,
    )

    logger.info(f"Invoke: processing query '{query[:100]}...'")

    # Create shared services
    llm, brave_client, crawler = _create_services()

    # Run research synchronously
    try:
        result = asyncio.run(
            run_research(
                query=query,
                llm=llm,
                brave_client=brave_client,
                crawler=crawler,
                conversation_history=conversation_history,
                session_id=session_id,
                config=config,
            )
        )

        # Format sources from state
        sources = [
            {
                "url": s.url,
                "title": s.title,
                "snippet": s.snippet,
            }
            for s in result.state.sources[:10]
        ]

        return {
            "content": result.state.final_report or "",
            "metadata": {
                "session_id": str(session_id),
                "steps_executed": result.steps_executed,
                "sources_count": len(result.state.sources),
                "sources": sources,
                "total_duration_ms": result.total_duration_ms,
            },
        }

    except Exception as e:
        logger.error(f"Research failed: {e}")
        return {
            "content": f"Research failed: {e!s}",
            "metadata": {"error": str(e)},
        }


def stream(
    request: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,  # noqa: ARG001 - reserved for future use
) -> Generator[dict[str, Any], None, None]:
    """Stream handler for real-time research.

    This handler is called for streaming requests and yields
    events as the research progresses.

    Args:
        request: Request containing messages array.
        context: Optional context with user info, trace ID, etc.

    Yields:
        Event dicts compatible with Databricks Agent protocol.
    """
    # Extract user query and conversation history
    messages = extract_messages(request)
    if not messages:
        yield {
            "type": "error",
            "content": "No query provided. Please ask a research question.",
        }
        return

    # Get the latest user message as query
    query = messages[-1].get("content", "")
    conversation_history = messages[:-1] if len(messages) > 1 else []

    # Extract context info
    session_id = uuid4()

    # Configure orchestration
    config = OrchestrationConfig(
        max_plan_iterations=3,
    )

    logger.info(f"Stream: processing query '{query[:100]}...'")

    # Create shared services
    llm, brave_client, crawler = _create_services()

    # Run async generator synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        agen = stream_research(
            query=query,
            llm=llm,
            brave_client=brave_client,
            crawler=crawler,
            conversation_history=conversation_history,
            session_id=session_id,
            config=config,
        )
        while True:
            try:
                event = loop.run_until_complete(agen.__anext__())
                # Convert event to dict for Databricks protocol
                if isinstance(event, str):
                    yield {"type": "text", "content": event}
                elif hasattr(event, "model_dump"):
                    yield event.model_dump()
                else:
                    yield {"type": "event", "data": str(event)}
            except StopAsyncIteration:
                break
    except Exception as e:
        logger.error(f"Stream research failed: {e}")
        yield {
            "type": "error",
            "content": f"Research failed: {e!s}",
        }
    finally:
        loop.close()
