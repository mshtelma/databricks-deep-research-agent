"""ReAct-based researcher agent with agentic tool use.

This module implements a ReAct (Reasoning + Acting) pattern for research,
where the LLM decides:
- Which URLs to search for
- Which URLs to crawl for full content
- Whether content quality is sufficient
- When to stop and synthesize

The key difference from the standard researcher is:
- LLM controls the research loop via tool calls
- No fixed number of URLs crawled
- Quality-based decisions on content
- Stops when sufficient high-quality content is collected
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import SpanType

from src.agent.config import get_researcher_config, get_researcher_config_for_depth
from src.agent.state import ResearchState, SourceInfo
from src.agent.tools.research_tools import RESEARCH_TOOLS
from src.agent.tools.url_registry import UrlRegistry
from src.agent.tools.web_crawler import WebCrawler, web_crawl
from src.agent.tools.web_search import format_search_results_indexed, web_search
from src.core.logging_utils import get_logger, truncate
from src.core.tracing_constants import (
    ATTR_STEP_INDEX,
    ATTR_STEP_TITLE,
    PHASE_EXECUTE,
    research_span_name,
    truncate_for_attr,
)
from src.services.llm.types import ModelTier, ToolCall
from src.services.search.brave import BraveSearchClient

if TYPE_CHECKING:
    from src.services.llm.client import LLMClient

logger = get_logger(__name__)


# ReAct system prompt for agentic research
# Only 2 tools: web_search and web_crawl. LLM judges quality naturally.
# URLs are hidden - LLM uses indices for security.
REACT_SYSTEM_PROMPT = """You are a research agent with web search and crawl tools.

## Goal
Find 3-5 high-quality sources with SPECIFIC facts (numbers, quotes, data) to answer the query.
Quality > Quantity: 3 excellent sources beats 10 mediocre ones.

## Tools
- **web_search**: Search the web. Returns numbered results with titles and snippets.
- **web_crawl**: Fetch full content using the INDEX number (0, 1, 2, etc.) from search results.

## Research Loop
1. Search for relevant information
2. Review snippets - identify sources by INDEX that look promising (specific facts, not overviews)
3. Crawl promising sources using their INDEX numbers to get full content
4. Read the content - YOU decide if it's high quality
5. Repeat until you have 3+ sources with good specific content

## Quality Judgment (YOU decide after reading content)
- **GOOD**: Specific numbers, exact quotes, research findings, detailed analysis
- **BAD**: Abstract only, paywall, navigation text, vague overview, "click to read more"

## Stopping
Stop calling tools when:
- You have 3+ sources with high-quality specific content
- OR after 10+ crawls, use best available

CRITICAL: When satisfied, respond WITHOUT calling any tools. No tool calls = done.
"""


@dataclass
class ReactResearchEvent:
    """Event emitted during ReAct research loop."""

    event_type: str  # tool_call, tool_result, quality_update, research_complete
    data: dict[str, Any]


@dataclass
class ReactResearchState:
    """Internal state for the ReAct research loop."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    high_quality_sources: list[str] = field(default_factory=list)  # URLs
    low_quality_sources: list[str] = field(default_factory=list)  # URLs
    tool_call_count: int = 0
    crawled_content: dict[str, str] = field(default_factory=dict)  # URL -> content
    url_registry: UrlRegistry = field(default_factory=UrlRegistry)  # Index -> URL mapping


async def run_react_researcher(
    state: ResearchState,
    llm: "LLMClient",
    crawler: WebCrawler,
    brave_client: BraveSearchClient,
    max_tool_calls: int | None = None,
) -> AsyncGenerator[ReactResearchEvent, None]:
    """Run the ReAct research loop with agentic tool use.

    The LLM controls the research process by deciding:
    - What to search for
    - Which URLs to crawl
    - When content quality is sufficient
    - When to stop researching

    Args:
        state: Current research state.
        llm: LLM client with tool support.
        crawler: Web crawler for fetching page content.
        brave_client: Brave search client.
        max_tool_calls: Maximum tool calls before stopping. If None,
            uses per-depth config from research_types.

    Yields:
        ReactResearchEvent for each tool call and result.
    """
    # Load global researcher configuration
    config = get_researcher_config()

    # Get per-depth researcher settings (max_tool_calls)
    depth = state.resolve_depth()
    depth_config = get_researcher_config_for_depth(depth)

    # Use provided max_tool_calls or get from per-depth config
    effective_max_tool_calls = max_tool_calls or depth_config.max_tool_calls
    step = state.get_current_step()

    if not step:
        logger.warning("REACT_RESEARCHER_NO_STEP")
        return

    # Build span name with step context
    step_number = state.current_step_index + 1
    span_name = research_span_name(PHASE_EXECUTE, "react_researcher", step=step_number)

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        span.set_attributes({
            ATTR_STEP_INDEX: step_number,
            ATTR_STEP_TITLE: truncate_for_attr(step.title, 100),
            "max_tool_calls": effective_max_tool_calls,
            "depth": depth,
        })

        logger.info(
            "REACT_RESEARCHER_START",
            step_title=truncate(step.title, 60),
            query=truncate(state.query, 80),
            max_tool_calls=effective_max_tool_calls,
            depth=depth,
        )

        # Initialize ReAct state
        react_state = ReactResearchState()
        react_state.messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Research this topic and find high-quality sources:

**Research Query:** {state.query}

**Current Step:** {step.title}
{step.description or ''}

Find sources with specific facts, numbers, and quotes that can be cited.
Start by searching for relevant information.""",
            },
        ]

        # ReAct loop - continues until LLM stops calling tools or max reached
        while react_state.tool_call_count < effective_max_tool_calls:
            # Call LLM with tools
            tool_calls_this_turn = []
            accumulated_content = ""

            try:
                async for chunk in llm.stream_with_tools(
                    messages=react_state.messages,
                    tools=RESEARCH_TOOLS,
                    tier=ModelTier.ANALYTICAL,
                    max_tokens=2000,
                ):
                    if chunk.content:
                        accumulated_content += chunk.content

                    if chunk.is_done:
                        if chunk.tool_calls:
                            tool_calls_this_turn = chunk.tool_calls
                        break

            except Exception as e:
                logger.error(
                    "REACT_RESEARCHER_LLM_ERROR",
                    error=str(e)[:200],
                )
                yield ReactResearchEvent(
                    event_type="error",
                    data={"error": str(e)[:200]},
                )
                break

            # If no tool calls, LLM is done researching (implicit stop)
            if not tool_calls_this_turn:
                logger.info(
                    "REACT_RESEARCHER_IMPLICIT_STOP",
                    tool_calls=react_state.tool_call_count,
                    high_quality_sources=len(react_state.high_quality_sources),
                    reasoning=truncate(accumulated_content, 200),
                )

                # Add assistant response to message history
                if accumulated_content:
                    react_state.messages.append({
                        "role": "assistant",
                        "content": accumulated_content,
                    })

                yield ReactResearchEvent(
                    event_type="research_complete",
                    data={
                        "reason": "llm_decided",
                        "tool_calls": react_state.tool_call_count,
                        "high_quality_sources": len(react_state.high_quality_sources),
                        "summary": accumulated_content,
                    },
                )
                break

            # Add assistant response with tool calls to message history
            react_state.messages.append({
                "role": "assistant",
                "content": accumulated_content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _serialize_args(tc.arguments),
                        },
                    }
                    for tc in tool_calls_this_turn
                ],
            })

            # Execute each tool call
            for tc in tool_calls_this_turn:
                react_state.tool_call_count += 1

                yield ReactResearchEvent(
                    event_type="tool_call",
                    data={
                        "tool": tc.name,
                        "args": tc.arguments,
                        "call_number": react_state.tool_call_count,
                    },
                )

                # Execute tool
                tool_result = await _execute_tool(
                    tc,
                    state,
                    react_state,
                    crawler,
                    brave_client,
                    config,
                )

                # Add tool result to message history
                react_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

                yield ReactResearchEvent(
                    event_type="tool_result",
                    data={
                        "tool": tc.name,
                        "result_preview": truncate(tool_result, 200),
                        "high_quality_count": len(react_state.high_quality_sources),
                    },
                )

            # Check if we have enough high-quality sources
            if len(react_state.high_quality_sources) >= 5:
                logger.info(
                    "REACT_RESEARCHER_QUALITY_THRESHOLD",
                    high_quality_sources=len(react_state.high_quality_sources),
                )
                yield ReactResearchEvent(
                    event_type="quality_threshold",
                    data={"count": len(react_state.high_quality_sources)},
                )
                # Let LLM make one more turn to potentially stop

        # If we hit max tool calls
        if react_state.tool_call_count >= effective_max_tool_calls:
            logger.warning(
                "REACT_RESEARCHER_MAX_CALLS",
                tool_calls=react_state.tool_call_count,
                high_quality_sources=len(react_state.high_quality_sources),
            )
            yield ReactResearchEvent(
                event_type="research_complete",
                data={
                    "reason": "max_tool_calls",
                    "tool_calls": react_state.tool_call_count,
                    "high_quality_sources": len(react_state.high_quality_sources),
                },
            )

        # Store high-quality content in state sources
        for url in react_state.high_quality_sources:
            if url in react_state.crawled_content:
                for source in state.sources:
                    if source.url == url:
                        source.content = react_state.crawled_content[url]
                        break

        # Generate observation from crawled content for reflector/synthesizer
        # Build a summary of what was found
        observation_parts = []
        step_title = step.title if step else "Research"
        observation_parts.append(f"## Research Step: {step_title}\n")
        observation_parts.append(f"Searched and crawled {len(react_state.high_quality_sources)} sources.\n")

        for url in react_state.high_quality_sources[:5]:  # Top 5 sources
            content = react_state.crawled_content.get(url, "")
            if content:
                # Get source title
                title = url
                for s in state.sources:
                    if s.url == url:
                        title = s.title or url
                        break
                # Include first 500 chars as summary
                preview = content[:500].replace("\n", " ").strip()
                observation_parts.append(f"\n### {title}\n{preview}...\n")

        observation = "\n".join(observation_parts)
        state.last_observation = observation
        state.all_observations.append(observation)

        # Update span with final metrics
        span.set_attributes({
            "total_tool_calls": react_state.tool_call_count,
            "high_quality_sources": len(react_state.high_quality_sources),
            "low_quality_sources": len(react_state.low_quality_sources),
        })

        logger.info(
            "REACT_RESEARCHER_COMPLETE",
            total_tool_calls=react_state.tool_call_count,
            high_quality_sources=len(react_state.high_quality_sources),
            low_quality_sources=len(react_state.low_quality_sources),
            state_sources=len(state.sources),
            observation_len=len(observation),
        )


async def _execute_tool(
    tc: ToolCall,
    state: ResearchState,
    react_state: ReactResearchState,
    crawler: WebCrawler,
    brave_client: BraveSearchClient,
    config: Any,
) -> str:
    """Execute a tool call and return the result string.

    Args:
        tc: The tool call to execute.
        state: Main research state (for adding sources).
        react_state: ReAct loop state.
        crawler: Web crawler.
        brave_client: Search client.
        config: Researcher configuration.

    Returns:
        Tool result as a string for the LLM.
    """
    try:
        if tc.name == "web_search":
            query = tc.arguments.get("query", "")
            count = min(tc.arguments.get("count", 5), 10)

            results = await web_search(query=query, count=count, client=brave_client)

            # Add sources to state
            for r in results.results:
                state.add_source(
                    SourceInfo(
                        url=r.url,
                        title=r.title,
                        snippet=r.snippet,
                        relevance_score=r.relevance_score,
                    )
                )

            # Format results with indices (URLs hidden from LLM)
            return format_search_results_indexed(results, react_state.url_registry)

        elif tc.name == "web_crawl":
            # Accept index instead of URL (security: LLM cannot hallucinate URLs)
            index = tc.arguments.get("index")

            if index is None:
                return "Error: 'index' parameter required. Use the index number from search results."

            # Resolve index to URL via registry
            url = react_state.url_registry.get_url(index)
            if url is None:
                return f"Error: Invalid index {index}. Use an index from search results (0, 1, 2, etc.)."

            crawl_output = await web_crawl(urls=[url], crawler=crawler)

            if not crawl_output.results:
                return f"Failed to crawl source [{index}]"

            result = crawl_output.results[0]
            if not result.success or not result.content:
                return f"Failed to extract content from source [{index}]: {result.error or 'empty content'}"

            # Store content for later
            content = result.content[:config.content_storage_length]
            react_state.crawled_content[url] = content

            # Track as high quality (LLM selected it for crawling)
            if url not in react_state.high_quality_sources:
                react_state.high_quality_sources.append(url)

            # Update source in state
            for s in state.sources:
                if s.url == url:
                    s.content = content
                    break

            # Return preview for LLM to judge quality naturally
            # NOTE: Title is shown, but URL is not - only index reference
            preview_len = min(3000, len(content))
            return (
                f"**[{index}] {result.title or 'Source'}** ({len(content)} chars total)\n\n"
                f"{content[:preview_len]}"
                f"{'...' if len(content) > preview_len else ''}\n\n"
                f"Sources crawled: {len(react_state.high_quality_sources)}"
            )

        else:
            return f"Unknown tool: {tc.name}"

    except Exception as e:
        logger.warning(
            "REACT_TOOL_ERROR",
            tool=tc.name,
            error=str(e)[:200],
        )
        return f"Tool error: {str(e)[:200]}"


def _serialize_args(args: dict[str, Any]) -> str:
    """Serialize tool arguments to JSON string."""
    import json
    return json.dumps(args)
