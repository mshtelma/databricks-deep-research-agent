"""Merged web-research tool — search + auto-crawl top results in ONE call.

Replaces the LLM-orchestrated ``web_search`` → ``web_crawl`` pair for the
common research path. The LLM gets a single tool that:

1. Runs a search against a :class:`SearchClient` (default: Brave).
2. **Automatically crawls the top K results** so the agent ALWAYS has full
   page bodies — no more "Potemkin research" failure mode (Class C1 defect).
3. Returns the remaining results as snippet-only **candidates** so the LLM
   can still selectively fetch more via the standalone ``web_crawl`` tool.

This is the merged-tool design proposed in the designer-hardening plan and
matches user feedback: "we must not rely on llm here, both tools go
together" — except the pair is now *inside* the tool, not exposed as
separate calls. Selectivity is preserved via the candidates list +
``web_crawl`` follow-up.

Provider differences (Brave vs Exa):
- **Brave** returns snippets only. ``SearchResult.content`` is ``None``, so
  the tool calls the ``ContentCrawler`` for each top-K URL.
- **Exa / Jina** can return full content inline. ``SearchResult.content``
  is populated, so the tool skips the crawl step for those results.

Either way, the LLM sees a uniform tool interface: ``fetched`` (with
bodies) + ``candidates`` (snippets only).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from databricks_deep_research.tools.builtins.web_crawl import (
    ContentCrawler,
    _default_crawl,
)
from databricks_deep_research.tools.builtins.web_search import (
    SearchClient,
    SearchResult,
)
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = ["WebResearchTool"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


# Auto-fetch top K results' bodies; matches "balanced" profile from the design.
# Researchers default to this; coordinators can request quick=3.
_DEFAULT_AUTO_FETCH_TOP_K = 5

# How many total results to surface (auto-fetched + candidates).
_DEFAULT_TOTAL_RESULTS = 10

# Hard ceiling on both — matches the underlying Brave API limit.
_MAX_RESULT_COUNT = 20

# Minimum body length (chars) for a fetched source to count as substantive.
# 404 pages, paywalled stubs, and crawler errors all produce sub-500 char
# bodies; dropping them prevents the LLM from citing them.
_MIN_BODY_CHARS = 300

# Per-fetch timeout. The crawler itself has internal timeouts; this is a
# wall-clock cap so a slow URL can't block the whole research call.
_FETCH_TIMEOUT_SECONDS = 25.0

# Truncate per-source body content for token budget. The full body is
# crawled then truncated to this limit before being returned to the LLM.
_MAX_BODY_CHARS = 8000

# Brave-compatible freshness values. Any other value is silently dropped.
_VALID_FRESHNESS = {"pd", "pw", "pm", "py", "any"}


# ---------------------------------------------------------------------------
# WebResearchTool
# ---------------------------------------------------------------------------


class WebResearchTool:
    """One-call web research: search + auto-crawl top K + return candidates.

    Constructor DI parallels :class:`WebSearchTool` and :class:`WebCrawlTool`.

    Parameters:
        search_client: any object satisfying :class:`SearchClient` (default:
            ``BraveSearchAdapter``).
        crawler: optional ``async (url) -> (text, title)`` callable. When
            ``None``, the tool uses :func:`_default_crawl` (httpx+trafilatura).
        auto_fetch_top_k: number of top results to crawl automatically.
        total_results: number of search hits to request (auto-fetched +
            candidates).
        max_body_chars: per-source body length cap.
    """

    def __init__(
        self,
        search_client: SearchClient,
        *,
        crawler: ContentCrawler | None = None,
        auto_fetch_top_k: int = _DEFAULT_AUTO_FETCH_TOP_K,
        total_results: int = _DEFAULT_TOTAL_RESULTS,
        max_body_chars: int = _MAX_BODY_CHARS,
    ) -> None:
        self._search_client = search_client
        self._crawler = crawler
        self._auto_fetch_top_k = max(1, min(auto_fetch_top_k, _MAX_RESULT_COUNT))
        self._total_results = max(self._auto_fetch_top_k, min(total_results, _MAX_RESULT_COUNT))
        self._max_body_chars = max_body_chars

        self._definition = ToolDefinition(
            name="web_research",
            description=(
                "Search the web AND fetch the top results' content in ONE "
                "call. Returns 'fetched' (top N results with full extracted "
                "body) plus 'candidates' (snippet-only results you can "
                "selectively crawl via web_crawl). PREFER this tool over "
                "calling web_search then web_crawl separately — it guarantees "
                "you get real source content even on the first call. Use "
                "web_crawl ONLY to follow up on specific 'candidates' the "
                "snippets show as relevant but you need full bodies for."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific, focused search query. Include "
                            "entities, dates, or metrics for best results. "
                            "Example: 'Snowflake Q4 FY2025 revenue earnings'"
                        ),
                    },
                    "auto_fetch_top_k": {
                        "type": "integer",
                        "description": (
                            f"How many top results to auto-crawl. "
                            f"Default {self._auto_fetch_top_k}. "
                            f"Use 3 for quick lookups, 5 for balanced research, "
                            f"8 for deep analysis. Max {_MAX_RESULT_COUNT}."
                        ),
                        "default": self._auto_fetch_top_k,
                    },
                    "total_results": {
                        "type": "integer",
                        "description": (
                            f"Total search hits to return (auto-fetched + "
                            f"candidates). Default {self._total_results}. "
                            f"Max {_MAX_RESULT_COUNT}."
                        ),
                        "default": self._total_results,
                    },
                    "freshness": {
                        "type": "string",
                        "description": (
                            "Time filter: 'pd' (past day), 'pw' (past week), "
                            "'pm' (past month), 'py' (past year), or 'any'."
                        ),
                        "enum": sorted(_VALID_FRESHNESS),
                    },
                },
                "required": ["query"],
            },
            source_type="web_research",
            source_kind="web",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    # -- argument validation -------------------------------------------------

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("web_research requires non-empty 'query'")
        # Coerce numeric strings ("5") to ints; clamp to limits.
        out: dict[str, Any] = {"query": query.strip()}
        for key, default in (
            ("auto_fetch_top_k", self._auto_fetch_top_k),
            ("total_results", self._total_results),
        ):
            raw = arguments.get(key, default)
            try:
                value = int(raw) if raw is not None else default
            except (TypeError, ValueError):
                value = default
            out[key] = max(1, min(value, _MAX_RESULT_COUNT))
        # auto_fetch_top_k cannot exceed total_results
        if out["auto_fetch_top_k"] > out["total_results"]:
            out["total_results"] = out["auto_fetch_top_k"]
        freshness = arguments.get("freshness")
        if isinstance(freshness, str) and freshness in _VALID_FRESHNESS:
            out["freshness"] = None if freshness == "any" else freshness
        return out

    # -- execution -----------------------------------------------------------

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query: str = arguments["query"]
        top_k: int = arguments["auto_fetch_top_k"]
        total: int = arguments["total_results"]
        freshness: str | None = arguments.get("freshness")

        # 1. Search
        try:
            hits = await self._search_client.search(
                query, count=total, freshness=freshness
            )
        except Exception as exc:  # noqa: BLE001 — surface as tool error
            logger.warning("WEB_RESEARCH_SEARCH_FAILED query=%r err=%s", query, exc)
            return ToolResult(
                content=json.dumps({
                    "fetched": [],
                    "candidates": [],
                    "metadata": {
                        "query": query,
                        "error": f"search failed: {exc}",
                    },
                }),
                success=False,
                error=f"search failed: {exc}",
            )

        if not hits:
            return ToolResult(
                content=json.dumps({
                    "fetched": [],
                    "candidates": [],
                    "metadata": {
                        "query": query,
                        "auto_fetched": 0,
                        "total_results": 0,
                        "next_step_hint": (
                            "Search returned no results. Try a broader or "
                            "different query."
                        ),
                    },
                }),
            )

        # 2. Partition: top K → auto-fetch group, rest → candidates.
        top_hits = hits[:top_k]
        candidates_hits = hits[top_k:]

        # 3. For each top hit without content, crawl in parallel.
        fetch_tasks = []
        for hit in top_hits:
            if hit.content:
                # Provider returned content inline (Exa/Jina path).
                fetch_tasks.append(self._return_inline(hit))
            else:
                # Brave-style: snippet only, crawl required.
                fetch_tasks.append(self._fetch_one(hit, context))

        fetched_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # 4. Admission filter — drop fetch errors and stubs below threshold.
        fetched: list[dict[str, Any]] = []
        admission_dropped: int = 0
        for idx, item in enumerate(fetched_results):
            if isinstance(item, BaseException):
                admission_dropped += 1
                logger.info(
                    "WEB_RESEARCH_FETCH_FAILED url=%s err=%s",
                    top_hits[idx].url, item,
                )
                continue
            if not item or len(item.get("body", "")) < _MIN_BODY_CHARS:
                admission_dropped += 1
                continue
            fetched.append(item)

        # 5. Build candidates payload (snippets only).
        candidates: list[dict[str, Any]] = []
        for offset, hit in enumerate(candidates_hits):
            candidates.append({
                "url": hit.url,
                "title": hit.title,
                "snippet": hit.snippet,
                "rank": top_k + offset + 1,
                "relevance_score": hit.relevance_score,
            })

        # 6. Register URLs with the URL registry so the LLM can reference
        # candidate URLs by integer index in follow-up web_crawl calls.
        url_indices: list[int] = []
        if context.url_registry is not None:
            for hit in hits:
                try:
                    idx = context.url_registry.register(hit.url)
                    url_indices.append(idx)
                except Exception:  # pragma: no cover — defensive
                    url_indices.append(-1)

        # 7. Return both layers + metadata + sources for citation tracking.
        next_step_hint = (
            "Scan the 'candidates' list. If any look essential and you need "
            "their full bodies, call web_crawl(urls=[...]) with those URLs. "
            "Cite specific facts back to the 'url' field of each fetched/"
            "candidate source."
            if candidates
            else "All requested results were auto-fetched; you have everything."
        )

        payload = {
            "fetched": fetched,
            "candidates": candidates,
            "metadata": {
                "query": query,
                "auto_fetched": len(fetched),
                "candidates_offered": len(candidates),
                "total_results": len(hits),
                "admission_dropped": admission_dropped,
                "next_step_hint": next_step_hint,
            },
        }

        sources = [
            SourceInfo(
                url=item["url"],
                title=item.get("title") or item["url"],
                snippet=item.get("snippet") or "",
                content=item.get("body") or "",
                relevance_score=item.get("relevance_score", 0.5),
                source_type="web_research",
            )
            for item in fetched
        ]

        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False),
            data={"web_research_query": query},
            sources=sources,
        )

    # ----------------------------------------------------------------------
    # Internal fetchers
    # ----------------------------------------------------------------------

    async def _return_inline(self, hit: SearchResult) -> dict[str, Any]:
        """Provider supplied content inline (Exa/Jina). No crawl needed."""
        body = (hit.content or "").strip()
        if len(body) > self._max_body_chars:
            body = body[: self._max_body_chars] + "…"
        return {
            "url": hit.url,
            "title": hit.title,
            "snippet": hit.snippet,
            "body": body,
            "relevance_score": hit.relevance_score,
            "source": "provider_inline",
        }

    async def _fetch_one(
        self, hit: SearchResult, context: ToolContext
    ) -> dict[str, Any]:
        """Crawl one URL via the injected crawler (or the default httpx path).

        Wrapped in a wall-clock timeout so a single slow page doesn't block
        the entire research call. Returns ``{}`` on failure — the caller
        admission-filters those out.
        """
        try:
            async with asyncio.timeout(_FETCH_TIMEOUT_SECONDS):
                if self._crawler is not None:
                    text, title_from_page = await self._crawler(hit.url)
                else:
                    text, title_from_page = await _default_crawl(
                        hit.url, timeout=10.0
                    )
        except (asyncio.TimeoutError, TimeoutError):
            logger.info("WEB_RESEARCH_FETCH_TIMEOUT url=%s", hit.url)
            return {}
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "WEB_RESEARCH_FETCH_ERROR url=%s err=%s", hit.url, exc
            )
            return {}

        body = (text or "").strip()
        if len(body) > self._max_body_chars:
            body = body[: self._max_body_chars] + "…"

        # Prefer the search snippet's title; fall back to the page-extracted
        # one when the snippet didn't have a usable title.
        title = hit.title or title_from_page or hit.url

        return {
            "url": hit.url,
            "title": title,
            "snippet": hit.snippet,
            "body": body,
            "relevance_score": hit.relevance_score,
            "source": "crawled",
        }
