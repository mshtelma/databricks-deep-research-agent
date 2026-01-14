"""Coordinator agent - query classification and simple query handling."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel

from src.agent.prompts.coordinator import (
    COORDINATOR_SYSTEM_PROMPT,
    COORDINATOR_USER_PROMPT,
    SIMPLE_QUERY_SYSTEM_PROMPT,
    SIMPLE_QUERY_TOOLS,
)
from src.agent.state import QueryClassification, ResearchState
from src.core.logging_utils import get_logger, log_agent_decision, truncate
from src.core.tracing_constants import (
    ATTR_DECISION_COMPLEXITY,
    ATTR_DECISION_REASONING,
    ATTR_DECISION_VALUE,
    PHASE_CLASSIFY,
    research_span_name,
    truncate_for_attr,
)
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

if TYPE_CHECKING:
    from src.services.chat_source_pool_service import ChatSourcePoolService

logger = get_logger(__name__)


class CoordinatorOutput(BaseModel):
    """Output schema for Coordinator agent."""

    complexity: str
    follow_up_type: str
    is_ambiguous: bool
    clarifying_questions: list[str] = []
    recommended_depth: str = "auto"
    reasoning: str
    is_simple_query: bool = False
    direct_response: str | None = None


async def run_coordinator(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Coordinator agent to classify the query.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Returns:
        Updated state with query classification.
    """
    span_name = research_span_name(PHASE_CLASSIFY, "coordinator")

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        span.set_attributes({
            "query": truncate_for_attr(state.query, 150),
            "conversation_history_length": len(state.conversation_history),
        })

        logger.info(
            "COORDINATOR_ANALYZING",
            query=truncate(state.query, 100),
            history_len=len(state.conversation_history),
        )

        # Format conversation history
        history_str = ""
        if state.conversation_history:
            history_str = "\n".join(
                f"{msg['role'].upper()}: {msg['content'][:200]}"
                for msg in state.conversation_history[-5:]  # Last 5 messages
            )
        else:
            history_str = "(No previous conversation)"

        # Build messages
        messages = [
            {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": COORDINATOR_USER_PROMPT.format(
                    query=state.query,
                    conversation_history=history_str,
                ),
            },
        ]

        # Get classification
        try:
            response = await llm.complete(
                messages=messages,
                tier=ModelTier.BULK_ANALYSIS,  # Use Gemini for classification
                structured_output=CoordinatorOutput,
            )

            if response.structured:
                output = response.structured
            else:
                # Parse JSON manually
                output = CoordinatorOutput.model_validate_json(response.content)

            # Update state
            state.query_classification = QueryClassification(
                complexity=output.complexity,
                follow_up_type=output.follow_up_type,
                is_ambiguous=output.is_ambiguous,
                clarifying_questions=output.clarifying_questions,
                recommended_depth=output.recommended_depth,
                reasoning=output.reasoning,
            )

            state.is_simple_query = output.is_simple_query

            # Handle simple query with direct response
            if output.is_simple_query and output.direct_response:
                state.direct_response = output.direct_response
                logger.info(
                    "SIMPLE_QUERY_DETECTED",
                    response_len=len(output.direct_response),
                )

            # Set decision attributes for trace
            span.set_attributes({
                ATTR_DECISION_VALUE: "simple_query" if output.is_simple_query else "research",
                ATTR_DECISION_COMPLEXITY: output.complexity,
                ATTR_DECISION_REASONING: truncate_for_attr(output.reasoning, 300),
                "decision.is_simple_query": output.is_simple_query,
                "decision.is_ambiguous": output.is_ambiguous,
                "decision.follow_up_type": output.follow_up_type,
                "decision.recommended_depth": output.recommended_depth,
            })

            log_agent_decision(
                logger,
                decision="classify",
                reasoning=output.reasoning,
                complexity=output.complexity,
                is_simple=output.is_simple_query,
                is_ambiguous=output.is_ambiguous,
                follow_up_type=output.follow_up_type,
            )

        except Exception as e:
            logger.error(
                "COORDINATOR_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            span.set_attributes({
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            })
            # Default classification on error
            state.query_classification = QueryClassification(
                complexity="moderate",
                follow_up_type="new_topic",
                is_ambiguous=False,
                reasoning=f"Default classification due to error: {e}",
            )

        return state


async def _handle_search_sources(
    query: str,
    chat_source_pool: ChatSourcePoolService,
) -> str:
    """Execute search_sources tool against chat source pool.

    Args:
        query: Search query from the model.
        chat_source_pool: Service for searching sources.

    Returns:
        Formatted search results or error message.
    """
    try:
        # Use hybrid search (BM25 + semantic)
        results = await chat_source_pool.search(query, limit=5)

        if not results:
            return "No relevant content found in sources."

        # Format results
        output = []
        for i, result in enumerate(results, 1):
            source_title = result.title or "Untitled"
            snippet = result.content[:500] if result.content else result.snippet or ""
            output.append(f"[Result {i}] From: {source_title}\n{snippet}")

        return "\n\n".join(output)
    except Exception as e:
        logger.warning(f"search_sources failed: {e}")
        return f"Search failed: {str(e)}"


async def _simple_query_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    llm: LLMClient,
    chat_source_pool: ChatSourcePoolService,
    max_tool_calls: int = 3,
) -> AsyncGenerator[str, None]:
    """ReAct loop for simple query with search tool.

    Args:
        messages: Initial messages for the LLM.
        tools: Tool definitions.
        llm: LLM client.
        chat_source_pool: Service for searching sources.
        max_tool_calls: Maximum number of tool calls allowed.

    Yields:
        Response chunks for streaming.
    """
    tool_call_count = 0
    current_messages: list[dict[str, Any]] = list(messages)  # Copy to avoid mutation

    while tool_call_count < max_tool_calls:
        # Use stream_with_tools to get content and/or tool calls
        pending_tool_calls = []
        content_chunks = []

        async for chunk in llm.stream_with_tools(
            messages=current_messages,
            tools=tools,
            tier=ModelTier.COMPLEX,
        ):
            # Collect content chunks
            if chunk.content:
                content_chunks.append(chunk.content)

            # Collect tool calls
            if chunk.tool_calls:
                pending_tool_calls.extend(chunk.tool_calls)

            # If done and no tool calls, we can yield content
            if chunk.is_done:
                break

        # If we got tool calls, execute them
        if pending_tool_calls:
            for tool_call in pending_tool_calls:
                if tool_call.name == "search_sources":
                    query = tool_call.arguments.get("query", "")
                    result = await _handle_search_sources(query, chat_source_pool)

                    # Add assistant message with tool call (omit content field - API rejects empty content)
                    current_messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }
                        ],
                    })

                    # Add tool result
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                    tool_call_count += 1

                    logger.info(
                        "SIMPLE_QUERY_TOOL_CALL",
                        tool="search_sources",
                        query=query[:100],
                        result_len=len(result),
                    )
        else:
            # No tool calls - yield the final content
            for chunk_content in content_chunks:
                yield chunk_content
            return

    # Max tool calls reached - generate final response without tools
    # Convert messages to string-only format for regular stream
    # Skip tool messages and messages with no content (tool-call-only assistant messages)
    stream_messages: list[dict[str, str]] = [
        {"role": m["role"], "content": m["content"]}
        for m in current_messages
        if m.get("role") != "tool" and m.get("content")
    ]
    async for final_chunk in llm.stream(
        messages=stream_messages, tier=ModelTier.COMPLEX
    ):
        yield final_chunk


async def handle_simple_query(
    state: ResearchState,
    llm: LLMClient,
    chat_source_pool: ChatSourcePoolService | None = None,
) -> AsyncGenerator[str, None]:
    """Handle simple query with full memory access (hybrid approach).

    Phase 1: Include source summaries + observations in context
    Phase 2: Use search_sources tool for deep retrieval when needed

    Args:
        state: Research state with simple query.
        llm: LLM client for completions.
        chat_source_pool: Optional service for searching sources.

    Yields:
        Response chunks for streaming.
    """
    if state.direct_response:
        yield state.direct_response
        return

    # === PHASE 1: Static Context ===

    # 1a. Build sources summary (titles + URLs for awareness)
    sources_summary = ""
    if state.sources:
        sources_summary = "\n\n## Available Sources from Previous Research\n"
        for i, src in enumerate(state.sources[:20], 1):
            title = src.title or "Untitled"
            sources_summary += f"[{i}] {title}\n    URL: {src.url}\n"
            if src.snippet:
                sources_summary += f"    Summary: {src.snippet[:200]}...\n"

    # 1b. Build observations context (key findings from research)
    observations_context = ""
    if state.all_observations:
        observations_context = "\n\n## Key Findings from Previous Research\n"
        for i, obs in enumerate(state.all_observations[-5:], 1):
            # Truncate long observations
            obs_preview = obs[:1000] + "..." if len(obs) > 1000 else obs
            observations_context += f"\n### Finding {i}\n{obs_preview}\n"

    # 1c. Enhanced system prompt with memory
    system_prompt = SIMPLE_QUERY_SYSTEM_PROMPT
    if sources_summary:
        system_prompt += sources_summary
    if observations_context:
        system_prompt += observations_context

    # === PHASE 2: Tool-Based Search (if source pool available) ===
    tools = None
    if chat_source_pool and state.sources:
        tools = SIMPLE_QUERY_TOOLS

    # Build messages with conversation history
    from src.agent.utils.conversation import build_messages_with_history

    messages = build_messages_with_history(
        system_prompt=system_prompt,
        user_query=state.query,
        history=state.conversation_history,
        max_history_messages=10,  # Increased for richer context
    )

    logger.info(
        "SIMPLE_QUERY_CONTEXT",
        history_messages=len(state.conversation_history),
        sources_count=len(state.sources),
        observations_count=len(state.all_observations),
        has_search_tool=tools is not None,
    )

    # Use COMPLEX tier - simple mode means "no web search", not "dumb model"
    if tools and chat_source_pool:
        # ReAct loop with search tool
        async for chunk in _simple_query_with_tools(
            messages, tools, llm, chat_source_pool
        ):
            yield chunk
    else:
        # Direct response (no tools available)
        async for chunk in llm.stream(messages=messages, tier=ModelTier.COMPLEX):
            yield chunk
