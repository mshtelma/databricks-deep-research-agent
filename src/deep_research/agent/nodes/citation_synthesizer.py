"""Citation-aware Synthesizer - Generates final report with claim-level attribution.

This module extends the standard synthesizer with the 6-stage citation verification
pipeline to provide claim-level citations and verification.
"""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import mlflow

from deep_research.agent.config import get_report_limits
from deep_research.agent.prompts.synthesizer import (
    STREAMING_SYNTHESIZER_SYSTEM_PROMPT,
)
from deep_research.agent.prompts.utils import build_system_prompt
from deep_research.agent.state import ResearchState
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

if TYPE_CHECKING:
    from deep_research.services.citation import CitationVerificationPipeline, VerificationEvent

logger = get_logger(__name__)


async def stream_synthesis_with_citations(
    state: ResearchState,
    llm: LLMClient,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream synthesis with citation verification pipeline.

    Integrates the 6-stage citation verification pipeline:
    1. Pre-selects evidence from sources
    2. Generates claims constrained by evidence
    3. Classifies claim confidence
    4. Verifies claims in isolation
    5. Emits verification events for real-time display

    Args:
        state: Current research state with sources and observations.
        llm: LLM client for completions.

    Yields:
        Dict events with either:
        - {"type": "content", "chunk": str} for synthesis content
        - {"type": "claim_verified", ...} for verification events
        - {"type": "verification_summary", ...} for final summary
    """
    logger.info(
        "CITATION_SYNTHESIZER_START",
        observations=len(state.all_observations),
        sources=len(state.sources),
        citations_enabled=state.enable_citation_verification,
    )

    # Check if citation verification is enabled
    if not state.enable_citation_verification:
        # Fall back to standard streaming synthesis
        async for chunk in _standard_stream_synthesis(state, llm):
            yield {"type": "content", "chunk": chunk}
        return

    # Lazy import to avoid circular dependency
    from deep_research.services.citation import CitationVerificationPipeline, VerificationEvent

    # Get effective research depth from state
    depth_label = state.resolve_depth()

    # Get word count and token limits from centralized research_types config
    limits = get_report_limits(depth_label)
    min_words = limits.min_words
    max_words = limits.max_words
    max_tokens = limits.max_tokens

    logger.info(
        "CITATION_SYNTHESIZER_DEPTH",
        depth=depth_label,
        min_words=min_words,
        max_words=max_words,
        max_tokens=max_tokens,
    )

    # Run the full citation verification pipeline with per-depth config
    pipeline = CitationVerificationPipeline(llm, depth=depth_label)

    try:
        async for item in pipeline.run_full_pipeline(
            state,
            target_word_count=max_words,  # Use max_words as the upper bound target
            max_tokens=max_tokens,
        ):
            if isinstance(item, str):
                # Content chunk
                yield {"type": "content", "chunk": item}
            elif isinstance(item, VerificationEvent):
                # Convert VerificationEvent to dict for streaming
                yield {
                    "type": item.event_type,
                    **item.data,
                }
    except Exception as e:
        logger.error(
            "CITATION_SYNTHESIZER_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        # Fall back to standard synthesis on error
        yield {
            "type": "error",
            "message": f"Citation verification failed: {e}. Falling back to standard synthesis.",
            "recoverable": True,
        }
        async for chunk in _standard_stream_synthesis(state, llm):
            yield {"type": "content", "chunk": chunk}

    logger.info(
        "CITATION_SYNTHESIZER_COMPLETE",
        claims=len(state.claims),
        report_len=len(state.final_report),
    )


async def _standard_stream_synthesis(
    state: ResearchState,
    llm: LLMClient,
) -> AsyncGenerator[str, None]:
    """Standard streaming synthesis without citation verification.

    Falls back to this when citation verification is disabled or fails.

    Args:
        state: Current research state.
        llm: LLM client.

    Yields:
        Content chunks as strings.
    """
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

    # Build system prompt with user's custom instructions
    system_prompt = build_system_prompt(
        STREAMING_SYNTHESIZER_SYSTEM_PROMPT,
        state.system_instructions,
    )

    messages = [
        {"role": "system", "content": system_prompt},
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
            "STANDARD_SYNTHESIS_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
            content_len=len(full_content),
        )
        error_msg = f"\n\n*Error during synthesis: {e}*"
        yield error_msg
        state.complete(full_content + error_msg)


@mlflow.trace(name="research.synthesis.citation_synthesizer", span_type="AGENT")
async def run_citation_synthesizer(
    state: ResearchState,
    llm: LLMClient,
) -> ResearchState:
    """Non-streaming version of citation-aware synthesis.

    Collects all content and verifies claims before returning.

    Args:
        state: Current research state.
        llm: LLM client.

    Returns:
        Updated state with final report and verified claims.
    """
    logger.info(
        "CITATION_SYNTHESIZER_RUN",
        observations=len(state.all_observations),
        sources=len(state.sources),
    )

    full_content = ""
    async for event in stream_synthesis_with_citations(state, llm):
        if event.get("type") == "content":
            full_content += event.get("chunk", "")

    # State is updated by the pipeline
    if not state.final_report:
        state.complete(full_content)

    logger.info(
        "CITATION_SYNTHESIZER_RUN_COMPLETE",
        report_len=len(state.final_report),
        claims=len(state.claims),
    )

    return state
