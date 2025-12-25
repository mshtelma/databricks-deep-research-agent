"""Background Investigator agent - quick context gathering before planning."""

import json

import mlflow

from src.agent.config import get_background_config
from src.agent.prompts.background import (
    BACKGROUND_SEARCH_PROMPT,
    BACKGROUND_SYSTEM_PROMPT,
    BACKGROUND_USER_PROMPT,
)
from src.agent.state import ResearchState, SourceInfo
from src.agent.tools.web_search import web_search
from src.core.logging_utils import get_logger, log_tool_call, truncate
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier
from src.services.search.brave import BraveSearchClient

logger = get_logger(__name__)


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
        )

        # Parse JSON array from response
        content = response.content.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        queries = json.loads(content)
        if isinstance(queries, list):
            result = [q for q in queries[:max_queries] if isinstance(q, str) and q.strip()]
            if result:
                logger.info(
                    "BACKGROUND_QUERIES_GENERATED",
                    count=len(result),
                    queries=[truncate(q, 60) for q in result],
                )
                return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "BACKGROUND_QUERY_GEN_FAILED",
            error_type=type(e).__name__,
            error=str(e)[:100],
        )

    # Fallback: truncate original query to safe length
    fallback = query[:200] if len(query) > 200 else query
    logger.info("BACKGROUND_USING_FALLBACK_QUERY", query=truncate(fallback, 60))
    return [fallback]


@mlflow.trace(name="background_investigator", span_type="AGENT")
async def run_background_investigator(
    state: ResearchState,
    llm: LLMClient,
    brave_client: BraveSearchClient,
) -> ResearchState:
    """Run Background Investigator to gather context before planning.

    Args:
        state: Current research state.
        llm: LLM client for completions.
        brave_client: Brave Search client for web searches.

    Returns:
        Updated state with background investigation results.
    """
    config = get_background_config()

    logger.info(
        "BACKGROUND_GATHERING_CONTEXT",
        query=truncate(state.query, 80),
    )

    try:
        # Generate focused search queries using LLM
        search_queries = await _generate_search_queries(
            llm, state.query, max_queries=config.max_search_queries
        )

        # Perform searches for each generated query
        all_results = []
        for sq in search_queries:
            log_tool_call(
                logger, tool_name="web_search", params={"query": sq, "count": config.max_results_per_query}
            )
            try:
                output = await web_search(
                    query=sq,
                    count=config.max_results_per_query,
                    client=brave_client,
                )
                all_results.extend(output.results)
            except Exception as e:
                logger.warning(
                    "BACKGROUND_SEARCH_FAILED",
                    query=truncate(sq, 60),
                    error=str(e)[:100],
                )

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

    except Exception as e:
        logger.error(
            "BACKGROUND_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        state.background_investigation_results = f"Background investigation unavailable: {e}"

    return state
