"""Synthesizer agent - generates final research report."""

from collections.abc import AsyncGenerator

import mlflow

from src.agent.prompts.synthesizer import (
    STREAMING_SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_PROMPT,
)
from src.agent.state import ResearchState
from src.core.logging_utils import get_logger, truncate
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

logger = get_logger(__name__)


@mlflow.trace(name="synthesizer", span_type="AGENT")
async def run_synthesizer(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Synthesizer agent to create final report.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Returns:
        Updated state with final report.
    """
    logger.info(
        "SYNTHESIZER_START",
        observations=len(state.all_observations),
        sources=len(state.sources),
        query=truncate(state.query, 60),
    )

    # Format observations
    observations_str = ""
    if state.all_observations:
        observations_str = "\n\n---\n\n".join(
            f"**Observation {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
        )
    else:
        observations_str = "(No research observations available)"

    # Format sources
    sources_list = ""
    for i, source in enumerate(state.sources[:20]):  # Limit to 20 sources
        title = source.title or "Untitled"
        sources_list += f"[{i + 1}] {title}\n    URL: {source.url}\n"
        if source.snippet:
            sources_list += f"    Snippet: {source.snippet[:200]}...\n"

    # Calculate research stats
    steps_executed = sum(
        1 for s in (state.current_plan.steps if state.current_plan else [])
        if s.status.value in ("completed", "skipped")
    )

    # Determine research depth label
    depth_label = "medium"
    if state.query_classification:
        depth_label = state.query_classification.recommended_depth

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SYNTHESIZER_USER_PROMPT.format(
                query=state.query,
                research_depth=depth_label,
                plan_iterations=state.plan_iterations,
                steps_executed=steps_executed,
                sources_count=len(state.sources),
                all_observations=observations_str,
                sources_list=sources_list or "(No sources collected)",
            ),
        },
    ]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.COMPLEX,
            max_tokens=4000,
        )

        state.complete(response.content)
        logger.info(
            "SYNTHESIZER_COMPLETE",
            report_len=len(response.content),
            report_preview=truncate(response.content, 150),
        )

    except Exception as e:
        logger.error(
            "SYNTHESIZER_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        # Fallback: use collected observations
        fallback_report = f"## Research Summary\n\n{observations_str}"
        state.complete(fallback_report)

    return state


async def stream_synthesis(state: ResearchState, llm: LLMClient) -> AsyncGenerator[str, None]:
    """Stream the synthesis output for real-time display.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Yields:
        Content chunks as they are generated.
    """
    logger.info(
        "SYNTHESIZER_STREAM_START",
        observations=len(state.all_observations),
        sources=len(state.sources),
    )

    # Format observations
    observations_str = ""
    if state.all_observations:
        observations_str = "\n\n---\n\n".join(
            f"**Observation {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
        )

    # Format sources
    sources_list = ""
    for source in state.sources[:15]:
        title = source.title or "Untitled"
        sources_list += f"- [{title}]({source.url})\n"

    messages = [
        {"role": "system", "content": STREAMING_SYNTHESIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Create a comprehensive research report.

## Query
{state.query}

## Research Findings
{observations_str}

## Available Sources
{sources_list}

Provide a well-structured markdown response with inline citations.""",
        },
    ]

    full_content = ""
    try:
        async for chunk in llm.stream(
            messages=messages,
            tier=ModelTier.COMPLEX,
            max_tokens=4000,
        ):
            full_content += chunk
            yield chunk

        # Update state with full content
        state.complete(full_content)

    except Exception as e:
        logger.error(
            "SYNTHESIZER_STREAM_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
            content_len=len(full_content),
        )
        error_msg = f"\n\n*Error during synthesis: {e}*"
        yield error_msg
        state.complete(full_content + error_msg)
