"""Researcher agent - executes individual research steps."""

import json
from typing import TYPE_CHECKING

import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from src.agent.config import get_researcher_config, get_researcher_config_for_depth
from src.agent.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_PROMPT,
    SEARCH_QUERY_PROMPT,
)
from src.agent.state import ResearchState, SourceInfo, StepStatus
from src.agent.tools.web_crawler import WebCrawler, web_crawl
from src.agent.tools.web_search import web_search
from src.core.logging_utils import (
    get_logger,
    log_search_queries_generated,
    log_tool_call,
    log_urls_selected,
    truncate,
)
from src.core.tracing_constants import (
    ATTR_CRAWL_SUCCESSFUL,
    ATTR_CRAWL_URLS_COUNT,
    ATTR_SEARCH_QUERY,
    ATTR_SEARCH_RESULTS_COUNT,
    ATTR_STEP_INDEX,
    ATTR_STEP_TITLE,
    ATTR_STEP_TYPE,
    PHASE_EXECUTE,
    research_span_name,
    truncate_for_attr,
)
from src.services.llm.types import ModelTier
from src.services.search.brave import BraveSearchClient

if TYPE_CHECKING:
    from src.services.llm.client import LLMClient

logger = get_logger(__name__)


class ResearcherOutput(BaseModel):
    """Output schema for Researcher agent."""

    search_queries: list[str] = []
    observation: str = "No observation could be generated from the available search results."
    key_points: list[str] = []
    sources_used: list[str] = []


class SearchQueriesOutput(BaseModel):
    """Output schema for search query generation."""

    queries: list[str] = Field(
        description="2-3 specific search queries to find information for this research step"
    )


async def run_researcher(
    state: ResearchState,
    llm: "LLMClient",
    crawler: WebCrawler,
    brave_client: BraveSearchClient,
) -> ResearchState:
    """Run the Researcher agent to execute a single research step.

    Args:
        state: Current research state.
        llm: LLM client for completions.
        crawler: Web crawler for fetching page content.
        brave_client: Brave search client for web searches.

    Returns:
        Updated state with step observation.
    """
    # Load global researcher configuration
    config = get_researcher_config()

    # Get per-depth researcher settings (max_search_queries, max_urls_to_crawl)
    depth = state.resolve_depth()
    depth_config = get_researcher_config_for_depth(depth)

    step = state.get_current_step()
    if not step:
        logger.warning("RESEARCHER_NO_STEP")
        return state

    # Use 1-based indexing for user-facing span names
    step_number = state.current_step_index + 1
    span_name = research_span_name(PHASE_EXECUTE, "researcher", step=step_number)

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        span.set_attributes({
            ATTR_STEP_INDEX: step_number,
            ATTR_STEP_TITLE: truncate_for_attr(step.title, 100),
            ATTR_STEP_TYPE: step.step_type.value,
            "step.needs_search": step.needs_search,
        })

        logger.info(
            "RESEARCHER_EXECUTING_STEP",
            step_title=truncate(step.title, 60),
            step_type=step.step_type.value,
            needs_search=step.needs_search,
        )

        # Mark step as in progress
        step.status = StepStatus.IN_PROGRESS

        search_results_text = ""
        page_contents_text = ""
        search_queries_used: list[str] = []
        search_results_count = 0
        crawl_successful = 0

        try:
            # If step needs search, perform web search
            if step.needs_search:
                # Generate search queries
                search_queries = await _generate_search_queries(
                    llm, step.title, step.description, state.query, config.max_generated_queries
                )

                # Log generated queries
                log_search_queries_generated(logger, step_title=step.title, queries=search_queries)

                # Perform searches (limit from per-depth config)
                all_results = []
                for query in search_queries[: depth_config.max_search_queries]:
                    search_queries_used.append(query)
                    try:
                        log_tool_call(logger, tool_name="web_search", params={"query": query, "count": 5})
                        results = await web_search(query=query, count=5, client=brave_client)
                        all_results.extend(results.results)
                    except Exception as e:
                        logger.warning(
                            "RESEARCHER_SEARCH_FAILED",
                            query=truncate(query, 60),
                            error=str(e)[:100],
                        )

                search_results_count = len(all_results)

                # Format search results
                if all_results:
                    search_results_text = "\n\n".join(
                        f"**{r.title}**\n{r.url}\n{r.snippet}"
                        for r in all_results[: config.max_search_results]
                    )

                    # Add sources to state
                    for r in all_results[: config.max_search_results]:
                        state.add_source(
                            SourceInfo(
                                url=r.url,
                                title=r.title,
                                snippet=r.snippet,
                                relevance_score=r.relevance_score,
                            )
                        )

                    # Crawl top URLs for content (limit from per-depth config)
                    top_urls = [r.url for r in all_results[: depth_config.max_urls_to_crawl]]
                    log_urls_selected(
                        logger, purpose="crawl", urls=top_urls, from_total=len(all_results)
                    )
                    try:
                        log_tool_call(logger, tool_name="web_crawl", params={"urls": top_urls})
                        crawl_output = await web_crawl(urls=top_urls, crawler=crawler)
                        for result in crawl_output.results:
                            if result.success and result.content:
                                crawl_successful += 1
                                page_contents_text += (
                                    f"\n\n---\n**{result.title or result.url}**\n"
                                    f"{result.content[: config.content_preview_length]}"
                                )
                                # Update source with content
                                for s in state.sources:
                                    if s.url == result.url:
                                        s.content = result.content[: config.content_storage_length]
                                        break
                    except Exception as e:
                        logger.warning(
                            "RESEARCHER_CRAWL_FAILED",
                            urls=len(top_urls),
                            error=str(e)[:100],
                        )

                    # Update span with search/crawl stats
                    span.set_attributes({
                        ATTR_SEARCH_QUERY: truncate_for_attr(", ".join(search_queries_used), 200),
                        ATTR_SEARCH_RESULTS_COUNT: search_results_count,
                        ATTR_CRAWL_URLS_COUNT: len(top_urls),
                        ATTR_CRAWL_SUCCESSFUL: crawl_successful,
                    })

                    # Log source content statistics for debugging citation pipeline issues
                    sources_with_content = sum(1 for s in state.sources if s.content)
                    sample_content_lengths = [len(s.content or "") for s in state.sources[:5]]
                    logger.info(
                        "RESEARCHER_CRAWL_COMPLETE",
                        total_sources=len(state.sources),
                        sources_with_content=sources_with_content,
                        sample_content_lengths=sample_content_lengths,
                    )

            # Format previous observations
            prev_observations = ""
            if state.all_observations:
                prev_observations = "\n\n".join(
                    f"Step {i + 1}: {obs[:500]}..."
                    for i, obs in enumerate(state.all_observations[-config.max_previous_observations :])
                )

            # Build messages for observation
            messages = [
                {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": RESEARCHER_USER_PROMPT.format(
                        step_title=step.title,
                        step_description=step.description,
                        step_type=step.step_type.value,
                        query=state.query,
                        previous_observations=prev_observations or "(No previous observations)",
                        search_results=search_results_text or "(No search performed)",
                        page_contents=page_contents_text[: config.page_contents_limit]
                        or "(No page contents)",
                    ),
                },
            ]

            response = await llm.complete(
                messages=messages,
                tier=ModelTier.ANALYTICAL,
                max_tokens=1500,
                structured_output=ResearcherOutput,
            )

            if response.structured:
                output = response.structured
            else:
                output = ResearcherOutput.model_validate_json(response.content)

            # Update step and state
            state.mark_step_complete(output.observation)

            # Set output attributes
            span.set_attributes({
                "output.key_points_count": len(output.key_points),
                "output.sources_used_count": len(output.sources_used),
                "output.observation_length": len(output.observation),
            })

            logger.info(
                "RESEARCHER_STEP_COMPLETED",
                key_points=len(output.key_points),
                sources_used=len(output.sources_used),
                observation_len=len(output.observation),
                observation_preview=truncate(output.observation, 150),
            )

        except Exception as e:
            logger.error(
                "RESEARCHER_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            span.set_attributes({
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            })
            # Mark step complete with error observation
            state.mark_step_complete(f"Step failed due to error: {e}")

        return state


async def _generate_search_queries(
    llm: "LLMClient",
    step_title: str,
    step_description: str,
    query: str,
    max_generated_queries: int,
) -> list[str]:
    """Generate search queries for a research step.

    Args:
        llm: LLM client for completions.
        step_title: Title of the current research step.
        step_description: Description of the current research step.
        query: Original user query.
        max_generated_queries: Maximum number of queries to generate.

    Returns:
        List of search queries.
    """
    messages = [
        {
            "role": "user",
            "content": SEARCH_QUERY_PROMPT.format(
                step_title=step_title,
                step_description=step_description,
                query=query,
            ),
        },
    ]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.SIMPLE,
            max_tokens=500,  # Increased from 200 to prevent truncation
            structured_output=SearchQueriesOutput,
        )

        # Use structured output if available
        if response.structured:
            return list(response.structured.queries[:max_generated_queries])

        # Fallback: parse JSON manually for non-structured endpoints
        queries = json.loads(response.content)
        if isinstance(queries, list):
            return list(queries[:max_generated_queries])
        if isinstance(queries, dict) and "queries" in queries:
            return list(queries["queries"][:max_generated_queries])
    except Exception as e:
        logger.warning(
            "QUERY_GENERATION_FAILED",
            step_title=truncate(step_title, 40),
            error=str(e)[:100],
        )

    # Fallback: use step description as query
    return [f"{step_title}: {step_description[:100]}"]
