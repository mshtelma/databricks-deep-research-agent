"""Background Investigator agent - quick context gathering before planning.

This module provides two modes of background investigation:
1. Legacy: `run_background_investigator()` - web-only context gathering
2. Enterprise: `run_background_discovery()` - multi-source discovery for US10

The enterprise discovery mode (T029-T034) explores ALL enabled data sources
in parallel to build a DataLandscape for source-aware planning.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from deep_research.agent.config import get_background_config
from deep_research.agent.prompts.background import (
    BACKGROUND_SEARCH_PROMPT,
    BACKGROUND_SYSTEM_PROMPT,
    BACKGROUND_USER_PROMPT,
)
from deep_research.agent.nodes.react_researcher import ReactResearchEvent
from deep_research.agent.state import ResearchState, SourceInfo
from deep_research.agent.tools.web_search import web_search
from deep_research.services.search.domain_filter import DomainFilter
from deep_research.core.app_config import get_app_config
from deep_research.core.logging_utils import get_logger, log_tool_call, truncate
from deep_research.core.tracing import safe_tool_span
from deep_research.core.tracing_constants import PHASE_BACKGROUND, research_span_name
from deep_research.schemas.data_landscape import DataLandscape, SourceDiscoveryResult
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier
from deep_research.services.obo_client import OBODatabricksClient
from deep_research.services.search.brave import BraveSearchClient

logger = get_logger(__name__)

# Discovery timeout in seconds (FR-088)
DISCOVERY_TIMEOUT_SECONDS = 5.0

# Default sample results limit for discovery
DEFAULT_SAMPLE_RESULTS = 3


# Pydantic models for structured LLM output
class BackgroundQueriesOutput(BaseModel):
    """Output from background search query generation."""

    queries: list[str] = Field(
        default_factory=list,
        description="List of 2-3 focused search queries"
    )


async def _generate_search_queries(
    llm: LLMClient,
    query: str,
    max_queries: int = 3,
) -> list[str]:
    """Generate focused search queries from user query.

    Args:
        llm: LLM client for completions.
        query: Original user query.
        max_queries: Maximum queries to generate.

    Returns:
        List of focused search queries.
    """
    messages = [
        {"role": "user", "content": BACKGROUND_SEARCH_PROMPT.format(query=query)}
    ]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.SIMPLE,
            max_tokens=200,
            structured_output=BackgroundQueriesOutput,
        )

        if response.structured:
            output: BackgroundQueriesOutput = response.structured
            result = [q for q in output.queries[:max_queries] if q.strip()]
            if result:
                logger.info(
                    "BACKGROUND_QUERIES_GENERATED",
                    count=len(result),
                    queries=[truncate(q, 60) for q in result],
                )
                return result

    except Exception as e:
        logger.warning(
            "BACKGROUND_QUERY_GEN_FAILED",
            error_type=type(e).__name__,
            error=str(e)[:100],
        )

    # Fallback: truncate original query to safe length
    fallback = query[:200] if len(query) > 200 else query
    logger.info("BACKGROUND_USING_FALLBACK_QUERY", query=truncate(fallback, 60))
    return [fallback]


async def run_background_investigator(
    state: ResearchState,
    llm: LLMClient,
    brave_client: BraveSearchClient,
) -> AsyncGenerator[ReactResearchEvent, None]:
    """Run Background Investigator to gather context before planning.

    Yields ReactResearchEvent for each web search, allowing
    the orchestrator to emit real-time progress events.

    Args:
        state: Current research state (mutated in-place).
        llm: LLM client for completions.
        brave_client: Brave Search client for web searches.

    Yields:
        ReactResearchEvent for each tool call/result.
    """
    span_name = research_span_name(PHASE_BACKGROUND, "investigator")

    async with safe_tool_span(span_name, SpanType.AGENT) as span:
        config = get_background_config()

        # Build domain filter once for all searches in this function
        _agent_domain_filter = DomainFilter(state.domain_filter) if state.domain_filter else None

        logger.info(
            "BACKGROUND_GATHERING_CONTEXT",
            query=truncate(state.query, 80),
        )

        try:
            # Generate focused search queries using LLM
            search_queries = await _generate_search_queries(
                llm, state.query, max_queries=config.max_search_queries
            )

            # Check source scope before web search (008-data-source-selection)
            all_results = []
            call_number = 0
            if not state.is_web_search_allowed():
                logger.info(
                    "BACKGROUND_INVESTIGATOR_SKIP_WEB",
                    scope=state.get_active_scope(),
                    reason="source_scope restricts web search",
                )
                # Skip web search but continue to LLM summarization with empty results
            else:
                # Perform searches for each generated query
                for sq in search_queries:
                    call_number += 1
                    yield ReactResearchEvent(event_type="tool_call", data={
                        "tool": "web_search",
                        "args": {"query": sq},
                        "call_number": call_number,
                    })
                    log_tool_call(
                        logger,
                        tool_name="web_search",
                        params={"query": sq, "count": config.max_results_per_query},
                    )
                    try:
                        output = await web_search(
                            query=sq,
                            count=config.max_results_per_query,
                            client=brave_client,
                            domain_filter=_agent_domain_filter,
                        )
                        all_results.extend(output.results)
                        yield ReactResearchEvent(event_type="tool_result", data={
                            "tool": "web_search",
                            "result_preview": f"Found {len(output.results)} results",
                            "high_quality_count": len(all_results),
                        })
                    except Exception as e:
                        logger.warning(
                            "BACKGROUND_SEARCH_FAILED",
                            query=truncate(sq, 60),
                            error=str(e)[:100],
                        )
                        yield ReactResearchEvent(event_type="tool_result", data={
                            "tool": "web_search",
                            "result_preview": f"Search failed: {str(e)[:100]}",
                            "high_quality_count": 0,
                        })

            # Deduplicate by URL
            seen_urls: set[str] = set()
            unique_results = []
            for r in all_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)

            # Use first N unique results
            final_results = unique_results[: config.max_total_results]

            # Format search results
            search_results = "\n\n".join(
                f"**{r.title}**\n{r.url}\n{r.snippet}"
                for r in final_results
            )

            # Add sources to state
            for r in final_results:
                state.add_source(
                    SourceInfo(
                        url=r.url,
                        title=r.title,
                        snippet=r.snippet,
                        relevance_score=r.relevance_score,
                    )
                )

            # Format conversation history
            history_str = ""
            if state.conversation_history:
                history_str = "\n".join(
                    f"{msg['role'].upper()}: {msg['content'][:100]}"
                    for msg in state.conversation_history[-3:]
                )
            else:
                history_str = "(No previous conversation)"

            # Include file content in background context so the summary
            # reflects uploaded file information for the planner
            file_context = state.get_file_context_for_prompt(max_chars=5000)
            if file_context:
                search_results = (search_results or "") + "\n\n---\n\n" + file_context

            # Get LLM to summarize findings
            messages = [
                {"role": "system", "content": BACKGROUND_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": BACKGROUND_USER_PROMPT.format(
                        query=state.query,
                        conversation_history=history_str,
                        search_results=search_results if search_results else "(No search results)",
                    ),
                },
            ]

            response = await llm.complete(
                messages=messages,
                tier=ModelTier.SIMPLE,
                max_tokens=500,
            )

            state.background_investigation_results = response.content
            logger.info(
                "BACKGROUND_COMPLETE",
                result_len=len(response.content),
                sources_added=len(final_results),
                result_preview=truncate(response.content, 150),
            )

            # Add span attributes
            if span:
                span.set_attributes({
                    "sources_added": len(final_results),
                    "search_queries_count": len(search_queries),
                })

        except Exception as e:
            logger.error(
                "BACKGROUND_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            state.background_investigation_results = f"Background investigation unavailable: {e}"


# =============================================================================
# Enterprise Data Source Discovery (T029-T034)
# =============================================================================


class ExploratoryQueriesOutput(BaseModel):
    """Output from exploratory query generation for discovery."""

    queries: list[str] = Field(
        default_factory=list,
        description="List of 3 exploratory queries covering different aspects"
    )


async def generate_exploratory_queries(
    llm: LLMClient,
    user_query: str,
    max_queries: int = 3,
) -> list[str]:
    """Generate exploratory queries for data source discovery (T030).

    Generates 3 query variants covering different aspects of the user's
    research question. Used during background discovery to probe each
    data source for relevance.

    Args:
        llm: LLM client for query decomposition.
        user_query: Original user research query.
        max_queries: Maximum number of queries to generate.

    Returns:
        List of exploratory query strings.
    """
    prompt = f"""Generate {max_queries} exploratory queries to discover relevant data sources \
for this research question.

Research Question: {user_query}

Requirements:
- Each query should explore a different aspect of the topic
- Queries should be simple and broad (good for discovery, not deep search)
- Include: 1) core concept query, 2) related entities query, 3) recent events query
- Keep each query under 80 characters
- Use natural language suitable for both semantic search and SQL

Respond with a JSON object containing a "queries" array:
{{"queries": ["query 1", "query 2", "query 3"]}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.SIMPLE,
            max_tokens=200,
            structured_output=ExploratoryQueriesOutput,
        )

        if response.structured:
            output: ExploratoryQueriesOutput = response.structured
            result = [q.strip() for q in output.queries[:max_queries] if q.strip()]
            if result:
                logger.info(
                    "DISCOVERY_QUERIES_GENERATED",
                    count=len(result),
                    queries=[truncate(q, 60) for q in result],
                )
                return result

    except Exception as e:
        logger.warning(
            "DISCOVERY_QUERY_GEN_FAILED",
            error_type=type(e).__name__,
            error=str(e)[:100],
        )

    # Fallback: use simplified versions of original query
    fallback = user_query[:100] if len(user_query) > 100 else user_query
    logger.info("DISCOVERY_USING_FALLBACK_QUERY", query=truncate(fallback, 60))
    return [fallback, f"latest {fallback}", f"overview {fallback}"]


async def explore_vector_search(
    endpoint_name: str,  # noqa: ARG001  # Kept for API compatibility with callers
    index_name: str,
    exploratory_queries: list[str],
    obo_client: OBODatabricksClient,
    user_token: str | None,
    source_name: str,
    num_results: int = DEFAULT_SAMPLE_RESULTS,
) -> SourceDiscoveryResult:
    """Explore a Vector Search source for relevance (T031).

    Performs lightweight queries to assess source relevance and extract
    available filter columns.

    Uses WorkspaceClient.vector_search_indexes.query_index() for consistent
    authentication (replaces VectorSearchClient).

    Args:
        endpoint_name: Vector Search endpoint name.
        index_name: Fully qualified index name (catalog.schema.index).
        exploratory_queries: Queries to try against the source.
        obo_client: OBO client for authentication.
        user_token: User's OAuth token for OBO access.
        source_name: Display name for this source.
        num_results: Number of results to fetch per query.

    Returns:
        SourceDiscoveryResult with relevance assessment.
    """
    from deep_research.services.vector_search_query import VectorSearchQueryService

    start_time = time.time()
    result = SourceDiscoveryResult(
        source_name=source_name,
        source_type="vector_search",
        relevance_score=0.0,
    )

    try:
        # Get OBO-authenticated WorkspaceClient
        client = await obo_client.get_client(user_token)

        # Try the first exploratory query
        query = exploratory_queries[0] if exploratory_queries else "sample query"
        result.query_used = query

        # Execute lightweight search via shared service
        service = VectorSearchQueryService()
        search_results = await service.query(
            client=client,
            index_name=index_name,
            query_text=query,
            columns=[],  # Return all columns for discovery
            num_results=num_results,
        )

        if search_results:
            result.has_results = True
            result.sample_results = [
                {
                    "id": r.id,
                    "title": r.title,
                    "content": r.content[:200] if r.content else "",
                    "score": r.score,
                }
                for r in search_results[:num_results]
            ]

            # Get relevance score from top result
            top_score = search_results[0].score
            result.relevance_score = float(top_score) if top_score else 0.5

            # Extract available filter columns from metadata keys
            if search_results[0].metadata:
                filter_columns = list(search_results[0].metadata.keys())
                result.available_filters = filter_columns

            # Suggest queries based on sample results
            result.suggested_queries = exploratory_queries[:2]

        logger.info(
            "DISCOVERY_VECTOR_SEARCH_COMPLETE",
            source=source_name,
            has_results=result.has_results,
            relevance=result.relevance_score,
            filter_columns=len(result.available_filters),
        )

    except Exception as e:
        error_msg = str(e)
        if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            result.error_message = f"Permission denied for index '{index_name}'"
        elif "NOT_FOUND" in error_msg or "404" in error_msg:
            result.error_message = f"Index '{index_name}' not found"
        else:
            result.error_message = error_msg[:200]

        # Set low relevance for failed sources
        result.relevance_score = 0.1

        logger.error(
            "DISCOVERY_VECTOR_SEARCH_ERROR",
            source=source_name,
            error=result.error_message,
            error_type=type(e).__name__,
            exc_info=True,
        )

    result.response_time_ms = (time.time() - start_time) * 1000
    return result


async def explore_genie(
    space_id: str,
    exploratory_queries: list[str],
    obo_client: OBODatabricksClient,
    user_token: str | None,
    source_name: str,
) -> SourceDiscoveryResult:
    """Explore a Genie source for relevance (T032).

    Implements FR-089: First tries a metadata query ("What data do you have
    related to..."), falls back to a sample query if ambiguous.

    Args:
        space_id: Genie space ID.
        exploratory_queries: Queries to try against the source.
        obo_client: OBO client for authentication.
        user_token: User's OAuth token for OBO access.
        source_name: Display name for this source.

    Returns:
        SourceDiscoveryResult with relevance assessment.
    """
    start_time = time.time()
    result = SourceDiscoveryResult(
        source_name=source_name,
        source_type="genie",
        relevance_score=0.0,
    )

    try:
        # Get OBO-authenticated client
        client = await obo_client.get_client(user_token)
        loop = asyncio.get_event_loop()

        # FR-089: Try metadata query first
        topic = exploratory_queries[0] if exploratory_queries else "this research topic"
        metadata_query = f"What data do you have related to {topic}?"
        result.query_used = metadata_query

        # Check if Genie API is available
        if not hasattr(client, "genie"):
            result.error_message = "Genie API not available in SDK"
            result.relevance_score = 0.1
            logger.warning(
                "DISCOVERY_GENIE_API_UNAVAILABLE",
                source=source_name,
            )
            result.response_time_ms = (time.time() - start_time) * 1000
            return result

        genie = client.genie

        try:
            # Start conversation with metadata query using start_conversation_and_wait
            # FIX: start_conversation() returns Wait[GenieMessage], not GenieMessage
            # Using start_conversation_and_wait() returns GenieMessage directly and handles polling
            message = await loop.run_in_executor(
                None,
                lambda: genie.start_conversation_and_wait(
                    space_id=space_id,
                    content=metadata_query,
                    timeout=timedelta(seconds=30),  # Discovery timeout
                ),
            )

            # Message is now a completed GenieMessage
            # Check status
            status_value = ""
            if hasattr(message, "status") and message.status:
                status_value = str(getattr(message.status, "value", str(message.status)))

            if "COMPLETED" in status_value or status_value == "COMPLETED":
                # Extract result summary from completed message
                # FIX: Changed 'status' to 'message' throughout
                # Check for content/narrative (GenieMessage uses 'content' field)
                if hasattr(message, "content") and message.content:
                    result.has_results = True
                    result.relevance_score = 0.7  # Good relevance if narrative returned
                    result.sample_results = [{"narrative": str(message.content)[:500]}]
                    result.suggested_queries = exploratory_queries[:2]
                # Check for query_result (GenieMessage structure)
                elif hasattr(message, "query_result") and message.query_result:
                    result.has_results = True
                    result.relevance_score = 0.6
                    query_result = message.query_result
                    if hasattr(query_result, "columns") and query_result.columns:
                        result.available_filters = [
                            getattr(col, "name", str(col)) for col in query_result.columns
                        ]

            elif "FAILED" in status_value or "CANCELLED" in status_value:
                # FR-089: Fall back to sample query
                # FIX: GenieMessage uses 'error' field with 'message' attribute
                error_obj = getattr(message, "error", None)
                error_msg_attr = getattr(error_obj, "message", str(error_obj)) if error_obj else ""
                if "ambiguous" in str(error_msg_attr).lower():
                    fallback_query = (
                        exploratory_queries[0]
                        if exploratory_queries
                        else "show me sample data"
                    )
                    result.query_used = fallback_query
                    # Could retry with fallback, but for discovery we just note ambiguity
                    result.error_message = "Query was ambiguous - try more specific questions"
                    result.relevance_score = 0.4  # Medium relevance - data exists but unclear
                else:
                    result.error_message = error_msg_attr if error_msg_attr else "Query failed"
                    result.relevance_score = 0.2
            else:
                # Unknown status - treat as partial success
                result.relevance_score = 0.3

        except Exception as query_error:
            error_str = str(query_error)
            if "ambiguous" in error_str.lower():
                result.error_message = "Query was ambiguous - Genie has relevant data"
                result.relevance_score = 0.4
            else:
                raise

        logger.info(
            "DISCOVERY_GENIE_COMPLETE",
            source=source_name,
            has_results=result.has_results,
            relevance=result.relevance_score,
        )

    except AttributeError:
        result.error_message = "Genie API not available in SDK version"
        result.relevance_score = 0.1
        logger.warning("DISCOVERY_GENIE_SDK_ERROR", source=source_name)

    except Exception as e:
        error_msg = str(e)
        if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            result.error_message = f"Permission denied for Genie space '{space_id}'"
        elif "NOT_FOUND" in error_msg or "404" in error_msg:
            result.error_message = f"Genie space '{space_id}' not found"
        else:
            result.error_message = error_msg[:200]

        result.relevance_score = 0.1

        logger.warning(
            "DISCOVERY_GENIE_ERROR",
            source=source_name,
            error=result.error_message[:100],
        )

    result.response_time_ms = (time.time() - start_time) * 1000
    return result


async def explore_web_source(
    exploratory_queries: list[str],
    brave_client: BraveSearchClient,
    source_name: str = "web_search",
    num_results: int = DEFAULT_SAMPLE_RESULTS,
    domain_filter: DomainFilter | None = None,
) -> SourceDiscoveryResult:
    """Explore web search for relevance (T033).

    Uses existing Brave search with limited results to assess
    web source relevance for the research topic.

    Args:
        exploratory_queries: Queries to try against web search.
        brave_client: Brave Search client.
        source_name: Display name for this source.
        num_results: Number of results to fetch.

    Returns:
        SourceDiscoveryResult with relevance assessment.
    """
    start_time = time.time()
    result = SourceDiscoveryResult(
        source_name=source_name,
        source_type="web_search",
        relevance_score=0.0,
    )

    try:
        # Use first exploratory query
        query = exploratory_queries[0] if exploratory_queries else "sample query"
        result.query_used = query

        log_tool_call(
            logger, tool_name="web_search", params={"query": query, "count": num_results}
        )

        output = await web_search(
            query=query,
            count=num_results,
            client=brave_client,
            domain_filter=domain_filter,
        )

        if output.results:
            result.has_results = True
            result.sample_results = [
                {
                    "title": r.title,
                    "snippet": r.snippet[:200],
                    "relevance_score": r.relevance_score,
                }
                for r in output.results[:num_results]
            ]

            # Calculate relevance from top results
            if output.results:
                avg_relevance = sum(
                    r.relevance_score for r in output.results[:3]
                ) / min(3, len(output.results))
                result.relevance_score = avg_relevance
            else:
                result.relevance_score = 0.3  # Low default for no results

            result.suggested_queries = exploratory_queries[:2]

        logger.info(
            "DISCOVERY_WEB_COMPLETE",
            source=source_name,
            has_results=result.has_results,
            relevance=result.relevance_score,
            results_count=len(output.results) if output.results else 0,
        )

    except Exception as e:
        result.error_message = str(e)[:200]
        result.relevance_score = 0.1

        logger.warning(
            "DISCOVERY_WEB_ERROR",
            source=source_name,
            error=result.error_message[:100],
        )

    result.response_time_ms = (time.time() - start_time) * 1000
    return result


def build_data_landscape(
    query: str,
    discovery_results: list[SourceDiscoveryResult],
    total_time_ms: float,
) -> DataLandscape:
    """Build DataLandscape from discovery results (T034).

    Aggregates discovery results, ranks sources by relevance,
    and builds a capabilities map per source type.

    Args:
        query: Original user query.
        discovery_results: Results from exploring each source.
        total_time_ms: Total discovery time in milliseconds.

    Returns:
        DataLandscape with aggregated discovery information.
    """
    # Sort results by relevance score (descending)
    sorted_results = sorted(
        discovery_results,
        key=lambda r: r.relevance_score,
        reverse=True,
    )

    # Build top sources list (top 5 by relevance)
    top_sources = [r.source_name for r in sorted_results[:5] if r.relevance_score > 0.2]

    # Build capabilities map (source_type -> source_names)
    capabilities_map: dict[str, list[str]] = {}

    for result in discovery_results:
        source_type = result.source_type
        if source_type not in capabilities_map:
            capabilities_map[source_type] = []

        # Only include sources with reasonable relevance
        if result.relevance_score >= 0.2 or result.has_results:
            capabilities_map[source_type].append(result.source_name)

    # Add capability tags based on source types
    capability_tags: dict[str, list[str]] = {
        "semantic_search": [],
        "sql_analytics": [],
        "current_events": [],
        "domain_expertise": [],
    }

    for result in discovery_results:
        if result.source_type == "vector_search" and result.has_results:
            capability_tags["semantic_search"].append(result.source_name)
        elif result.source_type == "genie" and result.has_results:
            capability_tags["sql_analytics"].append(result.source_name)
        elif result.source_type == "web_search" and result.has_results:
            capability_tags["current_events"].append(result.source_name)
        elif result.source_type == "knowledge_assistant" and result.has_results:
            capability_tags["domain_expertise"].append(result.source_name)

    # Merge capability tags into capabilities_map
    for capability, sources in capability_tags.items():
        if sources:
            capabilities_map[capability] = sources

    # Count statistics
    sources_queried = len(discovery_results)
    sources_with_results = sum(1 for r in discovery_results if r.has_results)

    landscape = DataLandscape(
        query=query,
        discovery_results=sorted_results,
        top_sources=top_sources,
        capabilities_map=capabilities_map,
        total_discovery_time_ms=total_time_ms,
        sources_queried=sources_queried,
        sources_with_results=sources_with_results,
        discovered_at=datetime.now(UTC),
    )

    logger.info(
        "DATA_LANDSCAPE_BUILT",
        sources_queried=sources_queried,
        sources_with_results=sources_with_results,
        top_sources=top_sources[:3],
        total_time_ms=total_time_ms,
    )

    return landscape


async def run_background_discovery(
    state: ResearchState,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    obo_client: OBODatabricksClient | None = None,
) -> ResearchState:
    """Run background discovery across all enabled data sources (T029).

    This is the main entry point for enterprise data source discovery.
    It queries ALL enabled sources in parallel to build a DataLandscape
    that informs source-aware planning.

    Implements:
    - FR-083: Run discovery across ALL enabled sources before planning
    - FR-084: Generate exploratory queries dynamically from user prompt
    - FR-085: Query sources in parallel to minimize latency
    - FR-088: Complete within 5 seconds timeout

    Args:
        state: Current research state.
        llm: LLM client for query generation.
        brave_client: Brave Search client for web searches.
        obo_client: OBO client for enterprise source access.

    Returns:
        Updated state with data_landscape populated.
    """
    span_name = research_span_name(PHASE_BACKGROUND, "discovery")
    start_time = time.time()

    async with safe_tool_span(span_name, SpanType.AGENT) as span:
        # Build domain filter once for all discovery searches
        _agent_domain_filter = DomainFilter(state.domain_filter) if state.domain_filter else None

        logger.info(
            "BACKGROUND_DISCOVERY_START",
            query=truncate(state.query, 80),
        )

        try:
            # Step 1: Generate exploratory queries
            exploratory_queries = await generate_exploratory_queries(
                llm=llm,
                user_query=state.query,
                max_queries=3,
            )

            # Step 2: Collect discovery tasks for all enabled sources
            discovery_tasks: list[Any] = []
            source_identifiers: list[str] = []  # Track which source each task is for

            # Get app configuration for enabled sources
            config = get_app_config()

            # Include web search ONLY if scope allows it (008-data-source-selection)
            if state.is_web_search_allowed():
                discovery_tasks.append(
                    explore_web_source(
                        exploratory_queries=exploratory_queries,
                        brave_client=brave_client,
                        source_name="web_search",
                        domain_filter=_agent_domain_filter,
                    )
                )
                source_identifiers.append("web_search")
            else:
                logger.info(
                    "BACKGROUND_DISCOVERY_SKIP_WEB",
                    scope=state.get_active_scope(),
                    reason="source_scope restricts web search",
                )

            # Add Vector Search sources if enabled
            if config.vector_search.enabled and obo_client:
                for name, endpoint_config in config.vector_search.endpoints.items():
                    if getattr(endpoint_config, "enabled", True):
                        discovery_tasks.append(
                            explore_vector_search(
                                endpoint_name=endpoint_config.endpoint_name,
                                index_name=endpoint_config.index_name,
                                exploratory_queries=exploratory_queries,
                                obo_client=obo_client,
                                user_token=state.user_token,
                                source_name=name,
                            )
                        )
                        source_identifiers.append(f"vs:{name}")

            # Note: Genie sources would be added here when Genie config is available
            # For now, Genie exploration requires explicit configuration
            # Example (when genie config exists):
            # if hasattr(config, 'genie') and config.genie.enabled and obo_client:
            #     for space_id, space_config in config.genie.spaces.items():
            #         discovery_tasks.append(
            #             explore_genie(
            #                 space_id=space_id,
            #                 exploratory_queries=exploratory_queries,
            #                 obo_client=obo_client,
            #                 user_token=state.user_token,
            #                 source_name=space_config.name or space_id,
            #             )
            #         )

            # Step 3: Execute all discovery tasks in parallel with timeout (FR-088)
            logger.info(
                "BACKGROUND_DISCOVERY_PARALLEL_START",
                task_count=len(discovery_tasks),
                sources=source_identifiers,
            )

            try:
                # Use asyncio.wait_for with gather to enforce timeout
                # return_exceptions=True means exceptions are returned as results, not raised
                gathered_results: list[SourceDiscoveryResult | BaseException] = (
                    await asyncio.wait_for(
                        asyncio.gather(*discovery_tasks, return_exceptions=True),
                        timeout=DISCOVERY_TIMEOUT_SECONDS,
                    )
                )

                # Handle any exceptions that were returned
                discovery_results: list[SourceDiscoveryResult] = []
                for idx, gathered_result in enumerate(gathered_results):
                    if isinstance(gathered_result, BaseException):
                        # Create low-relevance result for failed source
                        source_id = (
                            source_identifiers[idx]
                            if idx < len(source_identifiers)
                            else f"unknown_{idx}"
                        )
                        discovery_results.append(
                            SourceDiscoveryResult(
                                source_name=source_id,
                                source_type="unknown",
                                relevance_score=0.1,
                                error_message=str(gathered_result)[:200],
                            )
                        )
                        logger.error(
                            "DISCOVERY_TASK_EXCEPTION",
                            source=source_id,
                            error=str(gathered_result),
                            error_type=type(gathered_result).__name__,
                            exc_info=gathered_result,
                        )
                    else:
                        discovery_results.append(gathered_result)

            except TimeoutError:
                # Timeout reached - use partial results
                logger.warning(
                    "BACKGROUND_DISCOVERY_TIMEOUT",
                    timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
                    completed_tasks="partial",
                )
                # Create empty results for timed-out discovery
                discovery_results = [
                    SourceDiscoveryResult(
                        source_name="web_search",
                        source_type="web_search",
                        relevance_score=0.5,  # Default relevance for web
                        error_message="Discovery timed out",
                    )
                ]

            # Step 4: Build DataLandscape from results
            total_time_ms = (time.time() - start_time) * 1000

            data_landscape = build_data_landscape(
                query=state.query,
                discovery_results=discovery_results,
                total_time_ms=total_time_ms,
            )

            # Update state with data landscape
            state.data_landscape = data_landscape

            # Also run legacy background investigation for compatibility
            # This provides the background_investigation_results string
            # that existing code may rely on
            await _run_legacy_background_summary(
                state=state,
                llm=llm,
                data_landscape=data_landscape,
            )

            # Add span attributes
            if span:
                span.set_attributes({
                    "sources_queried": data_landscape.sources_queried,
                    "sources_with_results": data_landscape.sources_with_results,
                    "discovery_time_ms": total_time_ms,
                    "top_sources": ",".join(data_landscape.top_sources[:3]),
                })

            logger.info(
                "BACKGROUND_DISCOVERY_COMPLETE",
                sources_queried=data_landscape.sources_queried,
                sources_with_results=data_landscape.sources_with_results,
                total_time_ms=total_time_ms,
            )

        except Exception as e:
            logger.error(
                "BACKGROUND_DISCOVERY_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            # Create minimal data landscape on error
            state.data_landscape = DataLandscape(
                query=state.query,
                discovered_at=datetime.now(UTC),
            )
            state.background_investigation_results = f"Background discovery unavailable: {e}"

        return state


async def _run_legacy_background_summary(
    state: ResearchState,
    llm: LLMClient,  # noqa: ARG001  # Reserved for future LLM summarization
    data_landscape: DataLandscape,
) -> None:
    """Generate legacy background_investigation_results from DataLandscape.

    Provides backward compatibility by generating a text summary
    that existing planner code can consume.

    Args:
        state: Research state to update.
        llm: LLM client for summarization.
        data_landscape: Discovery results to summarize.
    """
    # Build summary from discovery results
    summary_parts: list[str] = []

    summary_parts.append("## Data Sources Discovery Summary")
    summary_parts.append(f"Discovered {data_landscape.sources_with_results} relevant sources "
                        f"out of {data_landscape.sources_queried} queried.")
    summary_parts.append("")

    if data_landscape.top_sources:
        summary_parts.append("**Top Sources:**")
        for source_name in data_landscape.top_sources[:5]:
            source_result = data_landscape.get_source_by_name(source_name)
            if source_result:
                summary_parts.append(
                    f"- {source_name} ({source_result.source_type}): "
                    f"relevance={source_result.relevance_score:.2f}"
                )
        summary_parts.append("")

    # Add sample insights from top results
    top_results = data_landscape.get_relevant_sources(min_score=0.5)
    if top_results:
        summary_parts.append("**Sample Results:**")
        for result in top_results[:3]:
            if result.sample_results:
                sample = result.sample_results[0]
                if isinstance(sample, dict):
                    if "snippet" in sample:
                        snippet = sample["snippet"][:150]
                        summary_parts.append(f"- [{result.source_name}] {snippet}...")
                    elif "narrative" in sample:
                        narrative = sample["narrative"][:150]
                        summary_parts.append(f"- [{result.source_name}] {narrative}...")

    state.background_investigation_results = "\n".join(summary_parts)
