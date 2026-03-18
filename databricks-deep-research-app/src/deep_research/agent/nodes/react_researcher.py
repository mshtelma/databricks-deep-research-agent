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

Parallel Tool Execution (007-enterprise-data-sources):
- Tools from different sources (web, vector_search, genie) can execute in parallel
- Same-source tools are batched but serialized by rate limiters
- Dependencies (web_crawl -> web_search) are respected
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlflow.entities import SpanType

from deep_research.agent.config import (
    get_endpoint_override,
    get_parallel_tool_execution_config,
    get_researcher_config,
    get_researcher_config_for_depth,
)
from deep_research.agent.state import ResearchState, SourceInfo
from deep_research.agent.tools.research_tools import RESEARCH_TOOLS
from deep_research.agent.tools.url_registry import UrlRegistry
from deep_research.agent.tools.web_crawler import WebCrawler, web_crawl
from deep_research.agent.tools.web_search import format_search_results_indexed, web_search
from deep_research.services.search.domain_filter import DomainFilter
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing import safe_tool_span
from deep_research.core.tracing_constants import (
    ATTR_STEP_INDEX,
    ATTR_STEP_TITLE,
    PHASE_EXECUTE,
    research_span_name,
    truncate_for_attr,
)
from deep_research.services.llm.types import ModelTier, ToolCall
from deep_research.services.search.brave import BraveSearchClient

if TYPE_CHECKING:
    from deep_research.services.llm.client import LLMClient

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
- **file_search** (if available): Search user-uploaded files for relevant passages.

When enterprise tools are available (query_genie_*, search_*, ask_*):
- These provide authoritative internal data — use results as primary evidence
- Genie tools query enterprise SQL databases and return structured data
- Vector search tools return semantically relevant enterprise documents
- Knowledge assistant tools return expert answers with citations
- Always cite enterprise data in your observations

## Research Loop
1. **Check uploaded files first**: If file content is provided in the user message, use it directly as evidence
2. If file_search is available, search uploaded files for additional passages
3. Search for relevant information from web/enterprise sources
4. Review snippets - identify sources by INDEX that look promising (specific facts, not overviews)
5. Crawl promising sources using their INDEX numbers to get full content
6. Read the content - YOU decide if it's high quality
7. Repeat until you have 3+ sources with good specific content

## Quality Judgment (YOU decide after reading content)
- **GOOD**: Specific numbers, exact quotes, research findings, detailed analysis
- **BAD**: Abstract only, paywall, navigation text, vague overview, "click to read more"

## Stopping
Stop calling tools when:
- You have 3+ sources with high-quality specific content
- OR after 10+ crawls, use best available

CRITICAL: When satisfied, respond WITHOUT calling any tools. No tool calls = done.
"""


def _is_uploaded_file_search_enabled(state: ResearchState) -> bool:
    """Check whether uploaded-file search is enabled in source scope."""
    if state.source_scope_config is None:
        return True
    return state.source_scope_config.is_type_enabled("uploaded_file")


def _get_scope_allowed_non_web_tools(state: ResearchState) -> list[Any]:
    """Return non-web tools currently allowed by source-scope settings."""
    if not state.enterprise_tools:
        return []

    allowed: list[Any] = []
    for tool in state.enterprise_tools:
        tool_name = tool.definition.name
        if tool_name == "file_search":
            if _is_uploaded_file_search_enabled(state):
                allowed.append(tool)
            continue

        if state.is_enterprise_search_allowed():
            allowed.append(tool)

    return allowed


@dataclass
class ReactResearchEvent:
    """Event emitted during ReAct research loop."""

    event_type: str  # tool_call, tool_result, quality_update, research_complete
    data: dict[str, Any]


@dataclass
class ReactResearchState:
    """Internal state for the ReAct research loop.

    Thread Safety (007-enterprise-data-sources):
    Uses asyncio.Lock for state accessed during parallel tool execution.
    CRITICAL: Use asyncio.Lock (NOT threading.RLock) because tool execution
    is async. threading.RLock would block the event loop and cause deadlocks.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    high_quality_sources: list[str] = field(default_factory=list)  # URLs
    low_quality_sources: list[str] = field(default_factory=list)  # URLs
    tool_call_count: int = 0
    crawled_content: dict[str, str] = field(default_factory=dict)  # URL -> content
    url_registry: UrlRegistry = field(default_factory=UrlRegistry)  # Index -> URL mapping

    # Token Optimization: Early stopping tracking
    # Track information gain per tool call to detect diminishing returns
    _info_gain_history: list[float] = field(default_factory=list)
    _last_high_quality_count: int = 0
    _last_content_length: int = 0
    _consecutive_low_gain_calls: int = 0

    # =========================================================================
    # Async Locks for Parallel Tool Execution
    # =========================================================================
    # Granular locks to reduce contention when tools update different collections
    _tool_count_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _content_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _sources_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def increment_tool_count(self) -> int:
        """Atomically increment tool call count and return new value (async-safe)."""
        async with self._tool_count_lock:
            self.tool_call_count += 1
            return self.tool_call_count

    async def add_crawled_content(self, url: str, content: str) -> None:
        """Add crawled content (async-safe)."""
        async with self._content_lock:
            self.crawled_content[url] = content

    async def add_high_quality_source(self, url: str) -> None:
        """Add high quality source (async-safe, deduplicates)."""
        async with self._sources_lock:
            if url not in self.high_quality_sources:
                self.high_quality_sources.append(url)

    async def add_low_quality_source(self, url: str) -> None:
        """Add low quality source (async-safe, deduplicates)."""
        async with self._sources_lock:
            if url not in self.low_quality_sources:
                self.low_quality_sources.append(url)

    async def record_tool_call_outcome(self) -> None:
        """Record the outcome of a tool call for early stopping analysis (async-safe)."""
        async with self._sources_lock:
            current_hq_count = len(self.high_quality_sources)
        async with self._content_lock:
            current_content_len = sum(len(c) for c in self.crawled_content.values())

        new_sources = current_hq_count - self._last_high_quality_count
        new_content = current_content_len - self._last_content_length

        info_gain = 0.0
        if new_sources > 0:
            info_gain += 0.5
        if new_content > 1000:
            info_gain += min(0.5, new_content / 10000)

        self._info_gain_history.append(info_gain)

        if info_gain < 0.1:
            self._consecutive_low_gain_calls += 1
        else:
            self._consecutive_low_gain_calls = 0

        self._last_high_quality_count = current_hq_count
        self._last_content_length = current_content_len

    async def should_stop_early(
        self,
        min_calls: int = 5,
        min_sources: int = 3,
        max_low_gain_calls: int = 5,
    ) -> tuple[bool, str]:
        """Determine if ReAct loop should stop early (async-safe)."""
        async with self._tool_count_lock:
            tc = self.tool_call_count
        if tc < min_calls:
            return False, ""

        if self._consecutive_low_gain_calls >= max_low_gain_calls:
            async with self._sources_lock:
                hq_count = len(self.high_quality_sources)
            if hq_count >= min_sources:
                return True, f"diminishing_returns_after_{self._consecutive_low_gain_calls}_low_gain_calls"

        async with self._sources_lock:
            hq_count = len(self.high_quality_sources)
        if hq_count >= min_sources + 2 and len(self._info_gain_history) >= 3:
            recent_gain = sum(self._info_gain_history[-3:])
            if recent_gain < 0.3:
                return True, f"high_coverage_{hq_count}_sources_low_recent_gain"

        return False, ""


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

    async with safe_tool_span(span_name, SpanType.AGENT, {
        ATTR_STEP_INDEX: step_number,
        ATTR_STEP_TITLE: truncate_for_attr(step.title, 100),
        "max_tool_calls": effective_max_tool_calls,
        "depth": depth,
    }) as span:

        logger.info(
            "REACT_RESEARCHER_START",
            step_title=truncate(step.title, 60),
            query=truncate(state.query, 80),
            max_tool_calls=effective_max_tool_calls,
            depth=depth,
        )

        # Initialize ReAct state
        react_state = ReactResearchState()

        # Build dynamic tool list including enterprise and uploaded-file tools.
        available_tools = list(RESEARCH_TOOLS)  # Start with web tools
        non_web_tools = _get_scope_allowed_non_web_tools(state)

        # Source-type-specific query guidance for tool descriptions
        # This costs zero latency — the LLM naturally generates better queries
        _QUERY_GUIDANCE: dict[str, str] = {
            "vector_search": (
                "\n\nQuery guidance: Provide natural language sentences, not keywords. "
                "Describe what the ideal document would say about your topic."
            ),
            "genie": (
                "\n\nQuery guidance: Be specific about metrics, time periods, and entity names. "
                "Ask a clear data question."
            ),
            "knowledge_assistant": (
                "\n\nQuery guidance: Ask a single focused question. "
                "Include context from your prior findings."
            ),
        }

        if non_web_tools:
            for tool in non_web_tools:
                desc = tool.definition.description
                guidance = _QUERY_GUIDANCE.get(tool.definition.source_type, "")
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.definition.name,
                        "description": desc + guidance,
                        "parameters": tool.definition.parameters,
                    },
                })
            logger.info(
                "REACT_NON_WEB_TOOLS_ADDED",
                tool_count=len(non_web_tools),
                tool_names=[t.definition.name for t in non_web_tools],
                total_tools=len(available_tools),
            )

        # Build file context for the initial prompt
        file_context = state.get_file_context_for_prompt(max_chars=8_000)
        file_instruction = ""
        if file_context:
            file_instruction = (
                f"\n\n{file_context}\n\n"
                "The user uploaded these files. Use this content directly as evidence where relevant. "
                "For large files, use file_search tool for specific lookups."
            )

        react_state.messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research this topic and find high-quality sources:\n\n"
                    f"**Research Query:** {state.query}\n\n"
                    f"**Current Step:** {step.title}\n"
                    f"{step.description or ''}"
                    f"{file_instruction}\n\n"
                    f"Find sources with specific facts, numbers, and quotes that can be cited.\n"
                    f"Start by searching for relevant information."
                ),
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
                    tools=available_tools,  # Dynamic tools including enterprise
                    tier=ModelTier.ANALYTICAL,
                    endpoint_override=get_endpoint_override(state, ModelTier.ANALYTICAL),
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

            # Execute tool calls - parallel when multiple tools and enabled, sequential otherwise
            # Parallel execution provides 20-40% latency reduction for cross-source queries
            parallel_config = get_parallel_tool_execution_config()
            use_parallel = (
                parallel_config.enabled
                and len(tool_calls_this_turn) > 1
            )

            if use_parallel:
                # PARALLEL EXECUTION: Execute tools concurrently
                logger.debug(
                    "REACT_PARALLEL_TOOLS",
                    tool_count=len(tool_calls_this_turn),
                    tools=[tc.name for tc in tool_calls_this_turn],
                )

                # Emit tool_call events first (for UI streaming)
                for tc in tool_calls_this_turn:
                    yield ReactResearchEvent(
                        event_type="tool_call",
                        data={
                            "tool": tc.name,
                            "args": tc.arguments,
                            "call_number": react_state.tool_call_count + 1,
                            "parallel": True,
                        },
                    )

                # Execute tools in parallel with cross-source parallelism
                tool_results: dict[str, str] = {}
                async for tc, result, sources_added in execute_tools_parallel(
                    tool_calls_this_turn,
                    state,
                    react_state,
                    crawler,
                    brave_client,
                    config,
                    tool_timeout_seconds=parallel_config.tool_timeout_seconds,
                    batch_timeout_seconds=parallel_config.batch_timeout_seconds,
                    llm=llm,
                ):
                    # Increment tool count (async-safe)
                    await react_state.increment_tool_count()
                    tool_results[tc.id] = result

                    # Emit result event for UI streaming
                    yield ReactResearchEvent(
                        event_type="tool_result",
                        data={
                            "tool": tc.name,
                            "result_preview": truncate(result, 200),
                            "high_quality_count": len(react_state.high_quality_sources),
                            "sources_added": sources_added,
                            "parallel": True,
                        },
                    )

                # Add all results to message history in original order (for LLM context)
                for tc in tool_calls_this_turn:
                    react_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_results.get(tc.id, "Error: Missing result"),
                    })

                # Track info gain for early stopping (once after batch)
                await react_state.record_tool_call_outcome()

            else:
                # SEQUENTIAL EXECUTION: Single tool, no parallelism benefit
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
                    sources_before = len(state.sources)
                    tool_result = await _execute_tool(
                        tc,
                        state,
                        react_state,
                        crawler,
                        brave_client,
                        config,
                        llm=llm,
                    )
                    sources_added = max(0, len(state.sources) - sources_before)

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
                            "sources_added": sources_added,
                        },
                    )

                    # TOKEN OPTIMIZATION: Track info gain for early stopping
                    await react_state.record_tool_call_outcome()

            # TOKEN OPTIMIZATION: Check for early stopping (diminishing returns)
            should_stop, stop_reason = await react_state.should_stop_early(
                min_calls=5,  # At least 5 calls before considering early stop
                min_sources=3,  # Need at least 3 high-quality sources
                max_low_gain_calls=5,  # Stop after 5 consecutive low-gain calls
            )
            if should_stop:
                logger.info(
                    "REACT_RESEARCHER_EARLY_STOP",
                    reason=stop_reason,
                    tool_calls=react_state.tool_call_count,
                    high_quality_sources=len(react_state.high_quality_sources),
                    info_gain_history=react_state._info_gain_history[-5:],
                )
                yield ReactResearchEvent(
                    event_type="research_complete",
                    data={
                        "reason": f"early_stop_{stop_reason}",
                        "tool_calls": react_state.tool_call_count,
                        "high_quality_sources": len(react_state.high_quality_sources),
                    },
                )
                break

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
        if span:
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
    llm: "LLMClient | None" = None,
) -> str:
    """Execute a tool call and return the result string.

    Args:
        tc: The tool call to execute.
        state: Main research state (for adding sources).
        react_state: ReAct loop state.
        crawler: Web crawler.
        brave_client: Search client.
        config: Researcher configuration.
        llm: LLM client (needed for enterprise query rewriting).

    Returns:
        Tool result as a string for the LLM.
    """
    # Build domain filter once for all searches in this tool execution
    _agent_domain_filter = DomainFilter(state.domain_filter) if state.domain_filter else None

    try:
        if tc.name == "web_search":
            # Check source scope before web search (008-data-source-selection)
            if not state.is_web_search_allowed():
                logger.info(
                    "REACT_SKIP_WEB_SEARCH",
                    query=truncate(tc.arguments.get("query", ""), 60),
                    scope=state.get_active_scope(),
                    tool_call_number=react_state.tool_call_count,
                )
                return (
                    "[Web search skipped: Source scope is set to 'enterprise_only'. "
                    "Enterprise data source integration is coming in a future update.]"
                )

            query = tc.arguments.get("query", "")
            count = min(tc.arguments.get("count", 5), 10)

            results = await web_search(query=query, count=count, client=brave_client, domain_filter=_agent_domain_filter)

            # Add sources to state
            for r in results.results:
                await state.add_source_async(
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
            # Check source scope before web crawl (008-data-source-selection)
            if not state.is_web_search_allowed():
                logger.info(
                    "REACT_SKIP_WEB_CRAWL",
                    url_index=tc.arguments.get("index"),
                    scope=state.get_active_scope(),
                )
                return "[Web crawl skipped: Source scope restricts web sources.]"

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
            # Check if it's an enterprise tool (007-enterprise-data-sources Phase 2)
            enterprise_tool = None
            for tool in state.enterprise_tools:
                if tool.definition.name == tc.name:
                    enterprise_tool = tool
                    break

            if enterprise_tool:
                from deep_research.agent.tools.base import ResearchContext
                from uuid import uuid4

                context = ResearchContext(
                    chat_id=state.session_id or uuid4(),
                    user_id="",  # Not tracked in ReAct researcher
                    user_token=state.user_token,
                )

                # Pre-execution query rewriting (enterprise query optimization)
                from deep_research.agent.config import get_query_rewrite_config
                from deep_research.agent.tools.query_rewriter import rewrite_for_source_type

                rewrite_config = get_query_rewrite_config(enterprise_tool.definition.source_type)
                if rewrite_config and rewrite_config.enabled and llm is not None:
                    # Determine the primary argument key (query or question)
                    tool_params = enterprise_tool.definition.parameters
                    required_keys = tool_params.get("required", [])
                    arg_key = required_keys[0] if required_keys else "query"
                    original_query = tc.arguments.get(arg_key, "")

                    step = state.get_current_step()
                    rewritten = await rewrite_for_source_type(
                        llm=llm,
                        source_type=enterprise_tool.definition.source_type,
                        step_title=step.title if step else "",
                        step_description=step.description if step else "",
                        original_query=original_query,
                        source_description=enterprise_tool.definition.description,
                        config=rewrite_config,
                        previous_observations=state.all_observations[-3:] if state.all_observations else None,
                    )
                    tc.arguments[arg_key] = rewritten.primary_query
                    # Store alternates for multi-query execution (M2)
                    tc.arguments["_alternate_queries"] = rewritten.alternate_queries
                    logger.info(
                        "REACT_QUERY_REWRITTEN",
                        tool=tc.name,
                        strategy=rewritten.strategy_used,
                        original_len=len(original_query),
                        rewritten_len=len(rewritten.primary_query),
                    )

                # Validate before execution
                validation_errors = enterprise_tool.validate_arguments(tc.arguments)
                if validation_errors:
                    logger.warning(
                        "REACT_ENTERPRISE_TOOL_VALIDATION_FAILED",
                        tool=tc.name,
                        errors=validation_errors,
                        arguments_keys=list(tc.arguments.keys()),
                    )
                    return f"Validation error: {'; '.join(validation_errors)}"

                result = await enterprise_tool.execute(
                    arguments=tc.arguments,
                    context=context,
                )

                if result.success:
                    fallback_url = f"enterprise://{tc.name}"

                    # Determine the canonical URL — must match what goes into state.sources
                    # so the post-processing loop can link content back
                    primary_url = fallback_url
                    if result.sources:
                        first_url = result.sources[0].get("url")
                        if first_url:
                            primary_url = first_url

                    await react_state.add_high_quality_source(primary_url)
                    await react_state.add_crawled_content(primary_url, result.content)

                    # Add sources to main state for citation tracking
                    ent_source_type = enterprise_tool.definition.source_type
                    if result.sources:
                        for src in result.sources:
                            await state.add_source_async(
                                SourceInfo(
                                    url=src.get("url", fallback_url),
                                    title=src.get("title", tc.name),
                                    snippet=src.get("content", "")[:500],
                                    content=src.get("content"),
                                    source_type=src.get("type", ent_source_type),
                                )
                            )
                    else:
                        # No structured sources — add generic entry so content can be linked
                        await state.add_source_async(
                            SourceInfo(
                                url=primary_url,
                                title=tc.name,
                                snippet=result.content[:500] if result.content else "",
                                content=result.content,
                                source_type=ent_source_type,
                            )
                        )

                    logger.info(
                        "REACT_ENTERPRISE_TOOL_SUCCESS",
                        tool=tc.name,
                        content_len=len(result.content),
                        primary_url=primary_url,
                        source_count=len(result.sources) if result.sources else 0,
                    )

                    # Heuristic quality signal (no LLM cost)
                    content_len = len(result.content) if result.content else 0
                    if content_len == 0:
                        quality_signal = "empty"
                    elif content_len < 100:
                        quality_signal = "low_content"
                    else:
                        quality_signal = "good"
                    state.record_source_quality(tc.name, quality_signal)

                    return result.content
                else:
                    logger.error(
                        "REACT_ENTERPRISE_TOOL_FAILED",
                        tool=tc.name,
                        error=result.error if result.error else "unknown",
                    )
                    state.record_source_quality(tc.name, "error")
                    return f"Tool error: {result.error or 'Unknown error'}"

            return f"Unknown tool: {tc.name}"

    except Exception as e:
        logger.error(
            "REACT_TOOL_ERROR",
            tool=tc.name,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return f"Tool error: {str(e)[:200]}"


def _serialize_args(args: dict[str, Any]) -> str:
    """Serialize tool arguments to JSON string."""
    import json
    return json.dumps(args)


# =============================================================================
# Parallel Tool Execution (007-enterprise-data-sources)
# =============================================================================

# Tool source mapping - tools from different sources can run in parallel
# Same-source tools may be serialized by rate limiters anyway
TOOL_SOURCE_MAPPING: dict[str, str] = {
    "web_search": "web",
    "web_crawl": "web",  # Depends on web_search completing first
    "vector_search": "vector",
    "genie_query": "genie",
    "file_search": "uploaded_file",
}

# Dependencies: tools that must complete before others can start
# web_crawl needs web_search to populate the URL registry
TOOL_DEPENDENCIES: dict[str, list[str]] = {
    "web_crawl": ["web_search"],
}


@dataclass
class ToolBatch:
    """A batch of tools that can execute in parallel."""

    tool_types: list[str]
    tool_calls: list[ToolCall]

    def __repr__(self) -> str:
        return f"ToolBatch(types={self.tool_types}, count={len(self.tool_calls)})"


def _get_tool_source(tool_name: str) -> str:
    """Get source type for a tool (for parallel execution grouping).

    Enterprise tools are identified by naming patterns:
    - query_genie_* -> genie
    - search_* or *_vector_search -> vector
    - *_assistant or knowledge_* -> assistant

    Args:
        tool_name: Name of the tool.

    Returns:
        Source type string for grouping.
    """
    if tool_name in TOOL_SOURCE_MAPPING:
        return TOOL_SOURCE_MAPPING[tool_name]

    # Enterprise tools: identify by naming patterns
    if tool_name.startswith("query_genie_") or "genie" in tool_name.lower():
        return "genie"
    if tool_name.startswith("search_") or "vector" in tool_name.lower():
        return "vector"
    if "assistant" in tool_name.lower() or "knowledge" in tool_name.lower():
        return "assistant"

    # Default: use tool name as source type (unique source per unknown tool)
    return tool_name


def _group_tools_by_source(tool_calls: list[ToolCall]) -> dict[str, list[ToolCall]]:
    """Group tool calls by source type for parallel execution.

    Different sources (web, vector_search, genie) can run in parallel.
    Same-source tools may be serialized by rate limiters anyway.

    Args:
        tool_calls: List of tool calls from the LLM.

    Returns:
        Dict mapping source type to list of tool calls.
    """
    from collections import defaultdict

    groups: dict[str, list[ToolCall]] = defaultdict(list)
    for tc in tool_calls:
        source = _get_tool_source(tc.name)
        groups[source].append(tc)
    return dict(groups)


def _get_execution_batches(tool_calls: list[ToolCall]) -> list[ToolBatch]:
    """Build execution batches respecting tool dependencies.

    web_crawl depends on web_search, so if both are present:
    - Batch 1: all web_search calls
    - Batch 2: everything else (including web_crawl)

    If no dependencies, all tools go in one batch.

    Args:
        tool_calls: List of tool calls from the LLM.

    Returns:
        List of ToolBatch (earlier batches must complete before later ones).
    """
    # Check for dependency conflicts
    has_crawl = any(tc.name == "web_crawl" for tc in tool_calls)
    has_search = any(tc.name == "web_search" for tc in tool_calls)

    batches: list[ToolBatch] = []

    if has_search and has_crawl:
        # Split: search first, then everything else (including crawl)
        search_calls = [tc for tc in tool_calls if tc.name == "web_search"]
        other_calls = [tc for tc in tool_calls if tc.name != "web_search"]

        if search_calls:
            batches.append(ToolBatch(
                tool_types=["web_search"],
                tool_calls=search_calls,
            ))
        if other_calls:
            batches.append(ToolBatch(
                tool_types=list(set(tc.name for tc in other_calls)),
                tool_calls=other_calls,
            ))
    else:
        # No dependencies - all tools in one batch
        if tool_calls:
            batches.append(ToolBatch(
                tool_types=list(set(tc.name for tc in tool_calls)),
                tool_calls=tool_calls,
            ))

    return batches


async def _execute_tool_with_timeout(
    tc: ToolCall,
    state: ResearchState,
    react_state: ReactResearchState,
    crawler: WebCrawler,
    brave_client: BraveSearchClient,
    config: Any,
    timeout_seconds: float = 30.0,
    llm: "LLMClient | None" = None,
) -> tuple[str, str, int]:
    """Execute a single tool with timeout.

    Args:
        tc: Tool call to execute.
        state: Main research state.
        react_state: ReAct loop state.
        crawler: Web crawler.
        brave_client: Search client.
        config: Researcher configuration.
        timeout_seconds: Per-tool timeout.
        llm: LLM client (needed for enterprise query rewriting).

    Returns:
        Tuple of (tool_call_id, result_string, sources_added).
    """
    try:
        sources_before = len(state.sources)
        result = await asyncio.wait_for(
            _execute_tool(tc, state, react_state, crawler, brave_client, config, llm=llm),
            timeout=timeout_seconds,
        )
        sources_added = max(0, len(state.sources) - sources_before)
        return (tc.id, result, sources_added)
    except TimeoutError:
        logger.warning(
            "PARALLEL_TOOL_TIMEOUT",
            tool=tc.name,
            tool_call_id=tc.id,
            timeout_seconds=timeout_seconds,
        )
        return (tc.id, f"Error: {tc.name} timed out after {timeout_seconds}s", 0)
    except Exception as e:
        logger.error(
            "PARALLEL_TOOL_ERROR",
            tool=tc.name,
            tool_call_id=tc.id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return (tc.id, f"Error executing {tc.name}: {str(e)[:200]}", 0)


async def execute_tools_parallel(
    tool_calls: list[ToolCall],
    state: ResearchState,
    react_state: ReactResearchState,
    crawler: WebCrawler,
    brave_client: BraveSearchClient,
    config: Any,
    tool_timeout_seconds: float = 30.0,
    batch_timeout_seconds: float = 60.0,
    llm: "LLMClient | None" = None,
) -> AsyncGenerator[tuple[ToolCall, str, int], None]:
    """Execute tool calls with cross-source parallelism.

    Different sources (web, vector_search, genie) execute in parallel.
    Dependencies (web_crawl -> web_search) are respected.

    Args:
        tool_calls: List of tool calls from the LLM.
        state: Main research state.
        react_state: ReAct loop state.
        crawler: Web crawler.
        brave_client: Search client.
        config: Researcher configuration.
        tool_timeout_seconds: Per-tool timeout.
        batch_timeout_seconds: Per-batch timeout.
        llm: LLM client (needed for enterprise query rewriting).

    Yields:
        (tool_call, result, sources_added) tuples as they complete.
    """
    if not tool_calls:
        return

    # Build execution batches respecting dependencies
    batches = _get_execution_batches(tool_calls)

    logger.info(
        "PARALLEL_TOOL_EXECUTION_START",
        total_tools=len(tool_calls),
        batches=len(batches),
        batch_details=[str(b) for b in batches],
    )

    results_by_id: dict[str, str] = {}

    for batch_idx, batch in enumerate(batches):
        logger.debug(
            "PARALLEL_BATCH_START",
            batch_index=batch_idx,
            tool_types=batch.tool_types,
            tool_count=len(batch.tool_calls),
        )

        # Create tasks for all tools in this batch
        tasks: dict[str, asyncio.Task[tuple[str, str, int]]] = {
            tc.id: asyncio.create_task(
                _execute_tool_with_timeout(
                    tc, state, react_state, crawler, brave_client, config,
                    timeout_seconds=tool_timeout_seconds,
                    llm=llm,
                )
            )
            for tc in batch.tool_calls
        }

        # Process results as they complete (for streaming)
        try:
            for coro in asyncio.as_completed(list(tasks.values()), timeout=batch_timeout_seconds):
                try:
                    tc_id, result, sources_added = await coro
                    results_by_id[tc_id] = result

                    # Find the tool call and yield immediately
                    tc = next(t for t in batch.tool_calls if t.id == tc_id)
                    yield (tc, result, sources_added)

                except Exception as e:
                    logger.error(
                        "PARALLEL_TOOL_EXECUTION_ERROR",
                        batch_index=batch_idx,
                        error=str(e)[:200],
                    )

        except TimeoutError:
            logger.warning(
                "PARALLEL_BATCH_TIMEOUT",
                batch_index=batch_idx,
                timeout_seconds=batch_timeout_seconds,
            )
            # Mark remaining tasks as timed out and cancel them
            for tc_id, task in tasks.items():
                if tc_id not in results_by_id:
                    task.cancel()
                    results_by_id[tc_id] = f"Error: Batch timed out after {batch_timeout_seconds}s"
                    # Yield timeout result
                    matching_tc = next((t for t in batch.tool_calls if t.id == tc_id), None)
                    if matching_tc is not None:
                        yield (matching_tc, results_by_id[tc_id], 0)

        logger.debug(
            "PARALLEL_BATCH_COMPLETE",
            batch_index=batch_idx,
            results_count=len([r for r in results_by_id.values() if not r.startswith("Error:")]),
        )

    logger.info(
        "PARALLEL_TOOL_EXECUTION_COMPLETE",
        total_tools=len(tool_calls),
        successful=len([r for r in results_by_id.values() if not r.startswith("Error:")]),
        failed=len([r for r in results_by_id.values() if r.startswith("Error:")]),
    )
