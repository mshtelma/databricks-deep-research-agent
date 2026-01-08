"""Coordinator agent - query classification and simple query handling."""

from collections.abc import AsyncGenerator

import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel

from src.agent.prompts.coordinator import (
    COORDINATOR_SYSTEM_PROMPT,
    COORDINATOR_USER_PROMPT,
    SIMPLE_QUERY_SYSTEM_PROMPT,
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
                tier=ModelTier.SIMPLE,
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


async def handle_simple_query(
    state: ResearchState,
    llm: LLMClient,
) -> AsyncGenerator[str, None]:
    """Handle a simple query with direct LLM response.

    Includes full conversation history so follow-ups can reference
    previous research reports (e.g., "Tell me more about the first point").

    Args:
        state: Research state with simple query.
        llm: LLM client for completions.

    Yields:
        Response chunks for streaming.
    """
    if state.direct_response:
        yield state.direct_response
        return

    # Include FULL conversation history for follow-up context
    # This enables referencing previous 13K+ word reports
    from src.agent.utils.conversation import build_messages_with_history

    messages = build_messages_with_history(
        system_prompt=SIMPLE_QUERY_SYSTEM_PROMPT,
        user_query=state.query,
        history=state.conversation_history,
        max_history_messages=5,  # Balance context with token limits
    )

    async for chunk in llm.stream(messages=messages, tier=ModelTier.SIMPLE):
        yield chunk
