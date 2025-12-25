"""Databricks Agent handlers.

Implements @invoke and @stream handlers that wrap the research orchestrator
for Databricks Agent Server deployment.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any
from uuid import uuid4

from src.agent.orchestrator import (
    OrchestrationConfig,
    run_research,
    stream_research,
)
from src.agent_server.utils import (
    extract_messages,
    transform_event_to_databricks,
)

logger = logging.getLogger(__name__)


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
        max_concurrent_steps=2,
    )

    logger.info(f"Invoke: processing query '{query[:100]}...'")

    # Run research synchronously
    try:
        result = asyncio.run(
            run_research(
                query=query,
                conversation_history=conversation_history,
                session_id=session_id,
                config=config,
            )
        )

        # Format sources
        sources = [
            {
                "url": s.url,
                "title": s.title,
                "snippet": s.snippet,
            }
            for s in result.sources[:10]
        ]

        return {
            "content": result.final_report,
            "metadata": {
                "session_id": str(result.session_id),
                "steps_completed": result.steps_completed,
                "sources_count": len(result.sources),
                "sources": sources,
                "research_time_seconds": result.research_time,
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
        max_concurrent_steps=2,
    )

    logger.info(f"Stream: processing query '{query[:100]}...'")

    # Create async generator wrapper
    async def async_stream() -> AsyncGenerator[dict[str, Any], None]:
        try:
            async for event in stream_research(
                query=query,
                conversation_history=conversation_history,
                session_id=session_id,
                config=config,
            ):
                yield transform_event_to_databricks(event)
        except Exception as e:
            logger.error(f"Stream research failed: {e}")
            yield {
                "type": "error",
                "content": f"Research failed: {e!s}",
            }

    # Run async generator synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        agen = async_stream()
        while True:
            try:
                event = loop.run_until_complete(agen.__anext__())
                yield event
            except StopAsyncIteration:
                break
    finally:
        loop.close()
