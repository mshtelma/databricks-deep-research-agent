"""Researcher agent - executes individual research steps."""

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from deep_research.agent.config import get_endpoint_override, get_researcher_config, get_researcher_config_for_depth
from deep_research.agent.nodes.react_researcher import ReactResearchEvent
from deep_research.agent.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_PROMPT,
    SEARCH_QUERY_PROMPT,
)
from deep_research.agent.state import ResearchState, SourceInfo, StepStatus
from deep_research.agent.tools.source_routing import (
    prompt_for_required_sources,
    validate_required_sources_consulted,
)
from deep_research.agent.tools.web_crawler import WebCrawler, web_crawl
from deep_research.agent.tools.web_search import web_search
from deep_research.services.search.domain_filter import DomainFilter
from deep_research.core.logging_utils import (
    get_logger,
    log_search_queries_generated,
    log_tool_call,
    log_urls_selected,
    truncate,
)
from deep_research.core.tracing import safe_tool_span
from deep_research.core.tracing_constants import (
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
from deep_research.schemas.manual_step import SourceConstraint
from deep_research.services.llm.types import ModelTier
from deep_research.services.search.brave import BraveSearchClient

if TYPE_CHECKING:
    from deep_research.agent.state import PlanStep
    from deep_research.services.llm.client import LLMClient

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


async def _execute_enterprise_tools(
    state: ResearchState,
    step: "PlanStep",
    tools: list[Any],
    llm: "LLMClient | None" = None,
) -> AsyncGenerator[tuple[str, dict[str, str], str, int], None]:
    """Execute enterprise tools one at a time, yielding per-tool results.

    Yields (tool_name, arguments, result_text, source_count) per tool.
    Accumulates sources on state as a side effect.
    The caller is responsible for emitting tool_call/tool_result events
    and accumulating result texts.

    Args:
        state: Current research state (provides identity and query context).
        step: Current plan step (provides title, description for query building).
        tools: Scope-allowed non-web tools to execute.

    Yields:
        Tuple of (tool_name, arguments, formatted_result_text, source_count).
    """
    if not tools:
        return

    from uuid import uuid4

    from deep_research.agent.tools.base import ResearchContext

    context = ResearchContext(
        chat_id=state.session_id or uuid4(),
        user_id="",
        user_token=state.user_token,
    )

    for tool in tools:
        tool_name = tool.definition.name

        # Determine correct argument key from tool's own schema
        tool_params = tool.definition.parameters
        required_keys = tool_params.get("required", [])
        arg_key = required_keys[0] if required_keys else "question"

        # Enterprise query rewriting: generate source-optimized query
        from deep_research.agent.config import get_query_rewrite_config
        from deep_research.agent.tools.query_rewriter import rewrite_for_source_type

        rewrite_config = get_query_rewrite_config(tool.definition.source_type)
        if rewrite_config and rewrite_config.enabled and llm is not None:
            try:
                rewritten = await rewrite_for_source_type(
                    llm=llm,
                    source_type=tool.definition.source_type,
                    step_title=step.title,
                    step_description=step.description or "",
                    original_query=state.query,
                    source_description=tool.definition.description,
                    config=rewrite_config,
                    previous_observations=state.all_observations[-3:] if state.all_observations else None,
                )
                query = rewritten.primary_query
                logger.info(
                    "RESEARCHER_QUERY_REWRITTEN",
                    tool=tool_name,
                    strategy=rewritten.strategy_used,
                    query_len=len(query),
                )
            except Exception as e:
                logger.warning(
                    "RESEARCHER_QUERY_REWRITE_FAILED",
                    tool=tool_name,
                    error=str(e)[:200],
                )
                query = f"{step.title}: {step.description or state.query}"
        else:
            query = f"{step.title}: {step.description or state.query}"

        arguments = {arg_key: query}

        try:
            # Validate arguments before execution
            validation_errors = tool.validate_arguments(arguments)
            if validation_errors:
                logger.warning(
                    "RESEARCHER_ENTERPRISE_TOOL_VALIDATION_FAILED",
                    tool=tool_name,
                    arg_key=arg_key,
                    errors=validation_errors,
                )
                yield (tool_name, arguments, "", 0)
                continue

            result = await tool.execute(
                arguments=arguments,
                context=context,
            )
            if result.success and result.content:
                source_count = 0
                source_type = tool.definition.source_type
                # Add sources for citation tracking
                if result.sources:
                    for src in result.sources:
                        state.add_source(
                            SourceInfo(
                                url=src.get(
                                    "url",
                                    f"enterprise://{tool_name}",
                                ),
                                title=src.get(
                                    "title", tool_name
                                ),
                                snippet=src.get("content", "")[:500],
                                content=src.get("content"),
                                source_type=src.get("type", source_type),
                            )
                        )
                        source_count += 1
                # Readable display name for the header
                _ENTERPRISE_DISPLAY_NAMES = {
                    "genie": "Enterprise Database (Genie)",
                    "vector_search": "Enterprise Document Search",
                    "knowledge_assistant": "Knowledge Assistant",
                }
                display_name = _ENTERPRISE_DISPLAY_NAMES.get(source_type, tool_name)
                logger.info(
                    "RESEARCHER_ENTERPRISE_TOOL_SUCCESS",
                    tool=tool_name,
                    source_type=source_type,
                    content_len=len(result.content),
                    source_count=len(result.sources) if result.sources else 0,
                )
                yield (tool_name, arguments, f"### {display_name}\n{result.content}", source_count)
            elif not result.success:
                logger.error(
                    "RESEARCHER_ENTERPRISE_TOOL_FAILED",
                    tool=tool_name,
                    error=result.error if result.error else "no error message",
                )
                yield (tool_name, arguments, "", 0)
            else:
                logger.info(
                    "RESEARCHER_ENTERPRISE_TOOL_EMPTY",
                    tool=tool_name,
                )
                yield (tool_name, arguments, "", 0)
        except TimeoutError:
            logger.error(
                "RESEARCHER_ENTERPRISE_TOOL_TIMEOUT",
                tool=tool_name,
            )
            yield (tool_name, arguments, "Tool execution timed out", 0)
        except ConnectionError as e:
            logger.error(
                "RESEARCHER_ENTERPRISE_TOOL_CONNECTION_ERROR",
                tool=tool_name,
                error=str(e)[:200],
            )
            yield (tool_name, arguments, f"Connection failed: {e}", 0)
        except Exception as e:
            logger.error(
                "RESEARCHER_ENTERPRISE_TOOL_UNEXPECTED_ERROR",
                tool=tool_name,
                error=str(e)[:200],
                error_type=type(e).__name__,
                exc_info=True,
            )
            yield (tool_name, arguments, "", 0)


async def run_researcher(
    state: ResearchState,
    llm: "LLMClient",
    crawler: WebCrawler,
    brave_client: BraveSearchClient,
) -> AsyncGenerator[ReactResearchEvent, None]:
    """Run the Researcher agent to execute a single research step.

    Yields ReactResearchEvent for each tool call and result, allowing
    the orchestrator to emit real-time progress events.

    Args:
        state: Current research state (mutated in-place).
        llm: LLM client for completions.
        crawler: Web crawler for fetching page content.
        brave_client: Brave search client for web searches.

    Yields:
        ReactResearchEvent for each tool call/result.
    """
    # Load global researcher configuration
    config = get_researcher_config()

    # Build domain filter once for all searches in this step
    _agent_domain_filter = DomainFilter(state.domain_filter) if state.domain_filter else None

    # Get per-depth researcher settings (max_search_queries, max_urls_to_crawl)
    depth = state.resolve_depth()
    depth_config = get_researcher_config_for_depth(depth)

    step = state.get_current_step()
    if not step:
        logger.warning("RESEARCHER_NO_STEP")
        return

    # Use 1-based indexing for user-facing span names
    step_number = state.current_step_index + 1
    span_name = research_span_name(PHASE_EXECUTE, "researcher", step=step_number)

    async with safe_tool_span(span_name, SpanType.AGENT, {
        ATTR_STEP_INDEX: step_number,
        ATTR_STEP_TITLE: truncate_for_attr(step.title, 100),
        ATTR_STEP_TYPE: step.step_type.value,
        "step.needs_search": step.needs_search,
    }) as span:
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
        call_number = 0

        try:
            if step.needs_search:
                non_web_tools = _get_scope_allowed_non_web_tools(state)
                # ------- ENTERPRISE TOOLS (runs when allowed AND tools available) -------
                if non_web_tools:
                    enterprise_parts: list[str] = []
                    # Build tool_name -> source_type lookup for event emission
                    _tool_source_types = {
                        t.definition.name: t.definition.source_type for t in non_web_tools
                    }
                    async for tool_name, arguments, result_text, src_count in _execute_enterprise_tools(
                        state, step, non_web_tools, llm=llm
                    ):
                        call_number += 1
                        ent_st = _tool_source_types.get(tool_name)
                        yield ReactResearchEvent(event_type="tool_call", data={
                            "tool": tool_name, "args": arguments, "call_number": call_number,
                            "source_type": ent_st,
                        })
                        yield ReactResearchEvent(event_type="tool_result", data={
                            "tool": tool_name,
                            "result_preview": result_text[:200] if result_text else "No results",
                            "high_quality_count": src_count,
                            "source_type": ent_st,
                        })
                        if result_text:
                            enterprise_parts.append(result_text)
                    enterprise_text = "\n\n".join(enterprise_parts)
                    if enterprise_text:
                        search_results_text = "## Enterprise Data Source Results\n\n" + enterprise_text
                        logger.info(
                            "RESEARCHER_ENTERPRISE_RESULTS",
                            source_count=sum(1 for p in enterprise_parts if p),
                            text_len=len(enterprise_text),
                        )

                # ------- WEB SEARCH (runs when allowed) -------
                if state.is_web_search_allowed():
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
                        call_number += 1
                        yield ReactResearchEvent(event_type="tool_call", data={
                            "tool": "web_search", "args": {"query": query}, "call_number": call_number,
                            "source_type": "web_search",
                        })
                        try:
                            log_tool_call(logger, tool_name="web_search", params={"query": query, "count": 5})
                            results = await web_search(query=query, count=5, client=brave_client, domain_filter=_agent_domain_filter)
                            all_results.extend(results.results)
                            yield ReactResearchEvent(event_type="tool_result", data={
                                "tool": "web_search",
                                "result_preview": f"Found {len(results.results)} results",
                                "high_quality_count": len(all_results),
                                "source_type": "web_search",
                            })
                        except Exception as e:
                            logger.warning(
                                "RESEARCHER_SEARCH_FAILED",
                                query=truncate(query, 60),
                                error=str(e)[:100],
                            )
                            yield ReactResearchEvent(event_type="tool_result", data={
                                "tool": "web_search",
                                "result_preview": f"Search failed: {str(e)[:100]}",
                                "high_quality_count": 0,
                                "source_type": "web_search",
                            })

                    search_results_count = len(all_results)

                    # Format search results
                    if all_results:
                        web_results_text = "\n\n".join(
                            f"**{r.title}**\n{r.url}\n{r.snippet}"
                            for r in all_results[: config.max_search_results]
                        )

                        # Combine with enterprise results if both ran
                        if search_results_text:
                            search_results_text += "\n\n---\n\n## Web Search Results\n" + web_results_text
                        else:
                            search_results_text = web_results_text

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
                        call_number += 1
                        yield ReactResearchEvent(event_type="tool_call", data={
                            "tool": "web_crawl",
                            "args": {"urls": top_urls},
                            "call_number": call_number,
                            "source_type": "web_crawl",
                        })
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
                            yield ReactResearchEvent(event_type="tool_result", data={
                                "tool": "web_crawl",
                                "result_preview": f"Crawled {crawl_successful} pages",
                                "high_quality_count": len(state.sources),
                                "source_type": "web_crawl",
                            })
                        except Exception as e:
                            logger.warning(
                                "RESEARCHER_CRAWL_FAILED",
                                urls=len(top_urls),
                                error=str(e)[:100],
                            )
                            yield ReactResearchEvent(event_type="tool_result", data={
                                "tool": "web_crawl",
                                "result_preview": f"Crawl failed: {str(e)[:100]}",
                                "high_quality_count": 0,
                                "source_type": "web_crawl",
                            })

                        # Update span with search/crawl stats
                        if span:
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
                elif not state.is_web_search_allowed():
                    logger.info(
                        "RESEARCHER_SKIP_WEB_SEARCH",
                        step=step.title,
                        scope=state.get_active_scope(),
                        reason="source_scope restricts web search",
                    )

                # Inject inline file content into search results
                file_context = state.get_file_context_for_prompt(max_chars=8_000)
                if file_context:
                    file_evidence = "## Uploaded File Evidence\n" + file_context + "\n\n---\n\n"
                    search_results_text = file_evidence + (search_results_text or "")

                # ------- FALLBACK: nothing found from any source -------
                if not search_results_text:
                    if not state.is_web_search_allowed() and not non_web_tools:
                        search_results_text = (
                            "[Web search skipped: Source scope is set to 'enterprise_only'. "
                            "No enterprise data sources are configured.]"
                        )
                    else:
                        search_results_text = "[No search results found from any source.]"
                    logger.warning(
                        "RESEARCHER_NO_RESULTS_FROM_ANY_SOURCE",
                        scope=state.get_active_scope(),
                        has_enterprise_tools=bool(non_web_tools),
                        web_allowed=state.is_web_search_allowed(),
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
                endpoint_override=get_endpoint_override(state, ModelTier.ANALYTICAL),
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
            if span:
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
            if span:
                span.set_attributes({
                    "error": str(e)[:200],
                    "error_type": type(e).__name__,
                })
            # Mark step complete with error observation
            state.mark_step_complete(f"Step failed due to error: {e}")


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
            tier=ModelTier.FAST,  # Use GPT 5.2 for non-structured query generation
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


# =============================================================================
# Source Constraint Integration (007-enterprise-data-sources, T053)
# =============================================================================


def check_step_source_constraint(
    state: ResearchState,
    step_id: str,
    queried_sources: set[str],
) -> tuple[bool, list[str], str]:
    """Check if source constraints are satisfied for a step.

    This function is used to validate that required sources from a
    SourceConstraint have been consulted during step execution.

    Args:
        state: Current research state.
        step_id: ID of the step to check.
        queried_sources: Set of source names that have been queried.

    Returns:
        Tuple of (is_satisfied, missing_sources, prompt_text):
        - is_satisfied: True if all constraints are satisfied
        - missing_sources: List of required sources not yet queried
        - prompt_text: Prompt text for unconsulted required sources
    """
    constraint = state.get_source_constraint(step_id)

    # No constraint means always satisfied
    if constraint is None:
        return True, [], ""

    # Ensure constraint is proper type (might be dict from JSON deserialization)
    if isinstance(constraint, dict):
        constraint = SourceConstraint.model_validate(constraint)

    # Check for missing required sources
    missing_sources = validate_required_sources_consulted(constraint, queried_sources)

    if not missing_sources:
        return True, [], ""

    # Get step info for prompt generation
    step = state.get_current_step()
    step_title = step.title if step else "Current Step"
    step_objective = step.description if step else ""

    prompt_text = prompt_for_required_sources(
        missing_sources=missing_sources,
        step_title=step_title,
        step_objective=step_objective,
    )

    return False, missing_sources, prompt_text


def get_step_queried_sources(state: ResearchState) -> set[str]:
    """Get set of source names queried in current research session.

    This extracts source information from state.source_results which
    tracks which sources have been queried.

    Args:
        state: Current research state.

    Returns:
        Set of source names that have been queried.
    """
    return set(state.source_results.keys())
