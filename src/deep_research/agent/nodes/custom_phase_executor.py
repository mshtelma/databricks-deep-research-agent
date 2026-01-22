"""
Custom Phase Executor - Executes plugin-provided research phases.

Provides the execution logic for BaseResearchPhase implementations.
Uses the researcher agent pattern with web search and crawl tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import SpanType

from deep_research.agent.state import SourceInfo
from deep_research.agent.tools.web_crawler import WebCrawler, web_crawl
from deep_research.agent.tools.web_search import web_search
from deep_research.core.logging_utils import (
    get_logger,
    log_tool_call,
    truncate,
)
from deep_research.core.tracing_constants import (
    ATTR_SEARCH_QUERY,
    ATTR_SEARCH_RESULTS_COUNT,
    PHASE_EXECUTE,
    research_span_name,
    truncate_for_attr,
)
from deep_research.services.llm.types import ModelTier
from deep_research.services.search.brave import BraveSearchClient

if TYPE_CHECKING:
    from deep_research.agent.state import ResearchState
    from deep_research.agent.tools.base import ResearchContext
    from deep_research.services.llm.client import LLMClient

logger = get_logger(__name__)


@dataclass
class PhaseResult:
    """Result from custom phase execution."""

    output: str
    """Main output text from phase execution."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    """Sources used during research."""

    success: bool = True
    """Whether phase completed successfully."""

    error: str | None = None
    """Error message if success=False."""


async def execute_custom_phase(
    phase_name: str,
    prompt: str,
    tools: list[str],
    context: "ResearchContext",
    state: "ResearchState",
    config: dict[str, Any],
) -> PhaseResult:
    """Execute a custom research phase.

    Uses the researcher agent pattern:
    1. Generate search queries from prompt
    2. Execute web searches
    3. Crawl top results
    4. Generate observation from gathered content

    Args:
        phase_name: Name of the phase being executed
        prompt: The research prompt for this phase
        tools: Tool names this phase can use
        context: Research context with session info
        state: Current research state
        config: Phase-specific configuration

    Returns:
        PhaseResult with output and sources
    """
    span_name = research_span_name(PHASE_EXECUTE, f"phase:{phase_name}")

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        span.set_attributes({
            "phase.name": phase_name,
            "phase.tools": ",".join(tools),
            "phase.prompt_length": len(prompt),
        })

        logger.info(
            "CUSTOM_PHASE_EXECUTING",
            phase_name=phase_name,
            tools=tools,
            prompt_preview=truncate(prompt, 100),
        )

        try:
            # Get services from app context
            from deep_research.app import get_app_services

            services = get_app_services()
            llm: LLMClient = services.llm
            crawler: WebCrawler = services.crawler
            brave_client: BraveSearchClient = services.brave_client

            # Gather content based on available tools
            search_results_text = ""
            page_contents_text = ""
            sources_collected: list[dict[str, Any]] = []

            # Execute web search if allowed
            if "web_search" in tools:
                search_queries = await _generate_search_queries(
                    llm, phase_name, prompt, max_queries=3
                )

                all_results = []
                for query in search_queries[:3]:
                    try:
                        log_tool_call(logger, tool_name="web_search", params={"query": query})
                        results = await web_search(query=query, count=5, client=brave_client)
                        all_results.extend(results.results)
                    except Exception as e:
                        logger.warning(
                            "PHASE_SEARCH_FAILED",
                            phase_name=phase_name,
                            query=truncate(query, 60),
                            error=str(e)[:100],
                        )

                span.set_attributes({
                    ATTR_SEARCH_QUERY: truncate_for_attr(", ".join(search_queries), 200),
                    ATTR_SEARCH_RESULTS_COUNT: len(all_results),
                })

                if all_results:
                    search_results_text = "\n\n".join(
                        f"**{r.title}**\n{r.url}\n{r.snippet}"
                        for r in all_results[:10]
                    )

                    # Add to sources
                    for r in all_results[:10]:
                        state.add_source(
                            SourceInfo(
                                url=r.url,
                                title=r.title,
                                snippet=r.snippet,
                                relevance_score=r.relevance_score,
                            )
                        )
                        sources_collected.append({
                            "type": "web",
                            "url": r.url,
                            "title": r.title,
                        })

                    # Crawl if allowed
                    if "web_crawl" in tools:
                        top_urls = [r.url for r in all_results[:3]]
                        try:
                            log_tool_call(logger, tool_name="web_crawl", params={"urls": top_urls})
                            crawl_output = await web_crawl(urls=top_urls, crawler=crawler)
                            for result in crawl_output.results:
                                if result.success and result.content:
                                    page_contents_text += (
                                        f"\n\n---\n**{result.title or result.url}**\n"
                                        f"{result.content[:5000]}"
                                    )
                                    # Update source with content
                                    for s in state.sources:
                                        if s.url == result.url:
                                            s.content = result.content[:10000]
                                            break
                        except Exception as e:
                            logger.warning(
                                "PHASE_CRAWL_FAILED",
                                phase_name=phase_name,
                                error=str(e)[:100],
                            )

            # Generate observation from gathered content
            system_prompt = f"""You are a research analyst executing the "{phase_name}" phase.

Your task is to analyze the search results and page content to extract insights
relevant to the research prompt. Focus on specific, actionable findings with
exact quotes and source attribution.

RESEARCH PROMPT:
{prompt}

GUIDELINES:
1. Cite sources for all findings
2. Prefer specific facts over general statements
3. Include relevant quotes from executives or official sources
4. Note any gaps in the available information
"""

            user_prompt = f"""Based on the search results and page content below, provide your research findings.

SEARCH RESULTS:
{search_results_text or "(No search results available)"}

PAGE CONTENT:
{page_contents_text[:15000] or "(No page content available)"}

Provide structured findings with source attribution."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await llm.complete(
                messages=messages,
                tier=ModelTier.ANALYTICAL,
                max_tokens=2000,
            )

            output = response.content

            span.set_attributes({
                "output.length": len(output),
                "output.sources_count": len(sources_collected),
            })

            logger.info(
                "CUSTOM_PHASE_COMPLETED",
                phase_name=phase_name,
                output_length=len(output),
                sources_count=len(sources_collected),
            )

            return PhaseResult(
                output=output,
                sources=sources_collected,
                success=True,
            )

        except Exception as e:
            logger.error(
                "CUSTOM_PHASE_ERROR",
                phase_name=phase_name,
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            span.set_attributes({
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            })
            return PhaseResult(
                output=f"Phase {phase_name} failed: {e}",
                success=False,
                error=str(e),
            )


async def _generate_search_queries(
    llm: "LLMClient",
    phase_name: str,
    prompt: str,
    max_queries: int = 3,
) -> list[str]:
    """Generate search queries from phase prompt.

    Args:
        llm: LLM client for completions
        phase_name: Name of the phase
        prompt: Research prompt to analyze
        max_queries: Maximum queries to generate

    Returns:
        List of search queries
    """
    messages = [
        {
            "role": "user",
            "content": f"""Generate {max_queries} specific search queries to research this topic.

RESEARCH PHASE: {phase_name}

PROMPT:
{prompt[:2000]}

Return ONLY a JSON array of search query strings, no explanation.
Example: ["query 1", "query 2", "query 3"]""",
        },
    ]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.FAST,
            max_tokens=300,
        )

        import json

        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if "```" in content:
                content = content.rsplit("```", 1)[0]
            content = content.strip()

        queries = json.loads(content)
        if isinstance(queries, list):
            return [str(q) for q in queries[:max_queries]]
    except Exception as e:
        logger.warning(
            "QUERY_GENERATION_FAILED",
            phase_name=phase_name,
            error=str(e)[:100],
        )

    # Fallback: extract key terms from prompt
    return [f"{phase_name} research"]


__all__ = ["execute_custom_phase", "PhaseResult"]
