"""Bounded "live web search" escape hatch for follow-up turns (spec §4.7, Tier 3).

When a follow-up question cannot be answered from the chat's prior research pool
but is a focused, web-searchable lookup, this runs a SMALL, capped live web
search (a handful of results, a short timeout) and answers from those fresh
sources — instead of either answering from an empty pool or paying for a full
multi-step research re-run.

Hard bounds (never unbounded):
- ``max_results`` caps the number of live results admitted (~5).
- ``timeout_seconds`` caps the wall-clock spent in the search call.

Provenance is surfaced, never silent: the live sources are attributed via the
normal ``ToolCallEvent`` / ``ToolResultEvent`` / ``StepCompletedEvent`` stream
(distinct from the prior pool) before the grounded answer streams.

Graceful fallback: on timeout, an empty result set, or any search error this
yields a single :class:`LiveSearchUnavailable` sentinel and NO content events,
so the caller can fall through to its normal research path. Like
``chat_answer``, this module only *streams content/provenance events*;
persistence is owned by the caller (``framework_orchestrator``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.agent.nodes.coordinator import handle_simple_query
from deep_research.agent.state import ResearchState, SourceInfo
from deep_research.core.logging_utils import get_logger
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    StepCompletedEvent,
    StepStartedEvent,
    StreamEvent,
    SynthesisProgressEvent,
    SynthesisStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from deep_research.services.llm.client import LLMClient

logger = get_logger(__name__)

# The bounded search uses a single research "step" so the provenance surfaces
# through the same step/tool events the deep-research path uses.
_LIVE_STEP_INDEX = 0
_LIVE_STEP_ID = "followup_live_search"


@dataclass
class LiveSearchUnavailable:
    """Sentinel yielded when the bounded live search produced no usable answer.

    The caller treats this as "fall back to the normal research path" — it is
    NOT a stream event and must never be forwarded to the frontend.
    """

    reason: str


def _as_uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _result_to_source(result: Any) -> SourceInfo:
    """Convert a framework ``SearchResult`` (duck-typed) into a ``SourceInfo``."""
    return SourceInfo(
        url=str(getattr(result, "url", "") or ""),
        title=getattr(result, "title", None),
        snippet=getattr(result, "snippet", None),
        content=getattr(result, "content", None),
        relevance_score=getattr(result, "relevance_score", None),
        source_type="web",
    )


async def _run_bounded_search(
    *,
    web_search_client: Any,
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> list[SourceInfo]:
    """Run one capped, time-boxed search; return ``[]`` on timeout/empty/error.

    The result count is capped twice: ``count=max_results`` asks the backend for
    a small page, and the returned list is sliced to ``max_results`` as a hard
    belt-and-braces cap regardless of what the backend returns.
    """
    cap = max(1, max_results)
    try:
        raw = await asyncio.wait_for(
            web_search_client.search(query, count=cap),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "FOLLOWUP_LIVE_SEARCH_TIMEOUT",
            timeout_s=timeout_seconds,
            query=query[:100],
        )
        return []
    except Exception as exc:  # noqa: BLE001 — bounded search is best-effort
        logger.warning("FOLLOWUP_LIVE_SEARCH_ERROR", error=str(exc)[:200])
        return []

    results = list(raw or [])[:cap]
    return [_result_to_source(r) for r in results if getattr(r, "url", None)]


async def stream_live_search_answer(
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    chat_id: str | UUID,
    llm: LLMClient,
    web_search_client: Any | None,
    max_results: int,
    timeout_seconds: float,
    prior_findings_summary: str = "",
) -> AsyncGenerator[StreamEvent | LiveSearchUnavailable, None]:
    """Stream a grounded answer over a small, bounded set of fresh web sources.

    Emits provenance for the live sources (a research step + tool call/result
    events) and then the same synthesis events a simple answer uses. On any
    failure to gather live sources, yields exactly one
    :class:`LiveSearchUnavailable` and nothing else, so the caller falls back to
    its normal research path without surfacing a half-finished answer.
    """
    cid = _as_uuid(chat_id)

    if web_search_client is None:
        logger.info("FOLLOWUP_LIVE_SEARCH_NO_CLIENT", chat_id=str(cid))
        yield LiveSearchUnavailable(reason="no_web_search_client")
        return

    logger.info(
        "FOLLOWUP_LIVE_SEARCH_START",
        chat_id=str(cid),
        max_results=max_results,
        timeout_s=timeout_seconds,
    )

    # -- Provenance: announce the bounded live-search step + tool call. --
    yield StepStartedEvent(
        step_index=_LIVE_STEP_INDEX,
        step_id=_LIVE_STEP_ID,
        step_title="Live web search",
        step_type="research",
    )
    yield ToolCallEvent(
        tool_name="web_search",
        tool_args={"query": query, "count": max(1, max_results)},
        call_number=1,
        source_type="web_search",
    )

    sources = await _run_bounded_search(
        web_search_client=web_search_client,
        query=query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )

    if not sources:
        # Graceful fallback. No content events were emitted; the StepStarted /
        # ToolCall above are harmless progress markers, and the caller will run
        # the full research path next.
        yield ToolResultEvent(
            tool_name="web_search",
            result_preview="No live results within the bounded search budget.",
            sources_crawled=0,
            sources_added=0,
            source_type="web_search",
        )
        logger.info("FOLLOWUP_LIVE_SEARCH_EMPTY", chat_id=str(cid))
        yield LiveSearchUnavailable(reason="no_live_results")
        return

    # -- Provenance: attribute the fresh sources (distinct from the prior pool). --
    preview = "; ".join(
        (s.title or s.url or "")[:80] for s in sources[:3]
    )
    yield ToolResultEvent(
        tool_name="web_search",
        result_preview=f"Live web search returned {len(sources)} source(s): {preview}",
        sources_crawled=len(sources),
        sources_added=len(sources),
        source_type="web_search",
    )
    yield StepCompletedEvent(
        step_index=_LIVE_STEP_INDEX,
        step_id=_LIVE_STEP_ID,
        observation_summary=f"Gathered {len(sources)} live web source(s) for this follow-up.",
        sources_found=len(sources),
        file_sources_found=0,
    )

    # -- Answer grounded ONLY on the fresh sources (plus any prior findings). --
    observations = [prior_findings_summary] if prior_findings_summary else []
    state = ResearchState(
        query=query,
        conversation_history=list(conversation_history or []),
        sources=sources,
        all_observations=observations,
    )

    yield SynthesisStartedEvent(
        total_observations=len(observations), total_sources=len(sources)
    )
    yield AgentStartedEvent(agent="synthesizer", model_tier="simple")
    # No source pool is passed: the bounded live sources are injected directly
    # into the state's system context by handle_simple_query (no deep retrieval).
    async for chunk in handle_simple_query(state, llm, None):
        if chunk:
            yield SynthesisProgressEvent(content_chunk=chunk)
    yield AgentCompletedEvent(agent="synthesizer", duration_ms=0)

    logger.info(
        "FOLLOWUP_LIVE_SEARCH_DONE",
        chat_id=str(cid),
        sources=len(sources),
    )
