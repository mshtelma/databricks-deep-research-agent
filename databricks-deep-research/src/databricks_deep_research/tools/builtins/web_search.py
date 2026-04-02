"""Builtin web search tool implementing the ``ResearchTool`` protocol.

Uses constructor dependency injection so the actual HTTP/search backend can be
swapped for testing.  The only contract is the :class:`SearchClient` protocol.

Usage::

    from databricks_deep_research.tools.builtins.web_search import WebSearchTool

    tool = WebSearchTool(my_brave_client, max_results=10)
    args = tool.validate_arguments({"query": "NVIDIA revenue 2025"})
    result = await tool.execute(args, context)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = ["WebSearchTool"]

logger = logging.getLogger(__name__)

# Maximum query length accepted by the tool.
_MAX_QUERY_LENGTH = 500

# Freshness values understood by Brave (and most search APIs).
_VALID_FRESHNESS = {"pd", "pw", "pm"}

# Hard ceiling on result count (Brave API limit).
_MAX_RESULT_COUNT = 20


# ---------------------------------------------------------------------------
# Search client protocol — the only dependency the tool needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A single result returned by a :class:`SearchClient`.

    The *content* field carries full page text when the search provider
    returns it (e.g. Jina).  Snippet-only providers leave it ``None``.
    """

    url: str
    title: str
    snippet: str
    relevance_score: float = 0.5
    content: str | None = None


@runtime_checkable
class SearchClient(Protocol):
    """Minimal async search backend contract.

    Any concrete implementation (Brave, Bing, mock) only needs to satisfy
    this single method.  The framework never touches the underlying HTTP
    client directly.
    """

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
    ) -> list[SearchResult]:
        """Execute a search and return results.

        Args:
            query: The search query string.
            count: Maximum number of results to return.
            freshness: Optional time filter (``"pd"``, ``"pw"``, ``"pm"``).

        Returns:
            A list of :class:`SearchResult` objects.
        """
        ...


# ---------------------------------------------------------------------------
# Domain filter helper
# ---------------------------------------------------------------------------


def _domain_matches(url: str, domains: list[str]) -> bool:
    """Return ``True`` if *url*'s host matches any entry in *domains*.

    Supports simple wildcard prefixes (``*.gov``, ``*.edu``).
    """
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False

    if not host:
        return False

    for pattern in domains:
        pattern = pattern.lower().strip()
        if pattern.startswith("*."):
            suffix = pattern[2:]  # e.g. "gov" from "*.gov"
            if host == suffix or host.endswith("." + suffix):
                return True
        elif host == pattern:
            return True
    return False


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Web search tool implementing :class:`~databricks_deep_research.tools.protocol.ResearchTool`.

    Dependencies are constructor-injected:

    * ``search_client`` — any object satisfying the :class:`SearchClient` protocol.
    * ``domain_filter`` — optional list of allowed domain patterns (e.g.
      ``["*.gov", "reuters.com"]``).  When set, results whose URL does not
      match any pattern are dropped.
    * ``max_results`` — default result count when the LLM does not specify one.

    The tool registers discovered URLs with the shared
    :class:`~databricks_deep_research.tools.protocol.UrlRegistry` so downstream
    tools (like ``web_crawl``) can resolve integer indices back to URLs without
    the LLM ever seeing raw URLs.
    """

    def __init__(
        self,
        search_client: SearchClient,
        *,
        domain_filter: list[str] | None = None,
        max_results: int = 5,
        max_content_per_result: int = 5000,
    ) -> None:
        self._client = search_client
        self._domain_filter = domain_filter or []
        self._max_results = min(max(max_results, 1), _MAX_RESULT_COUNT)
        self._max_content_per_result = max_content_per_result

        self._definition = ToolDefinition(
            name="web_search",
            description=(
                "Search the web for information. Returns numbered results with "
                "titles and snippets. Use the result INDEX numbers (0, 1, 2, …) "
                "to select sources for crawling."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific, focused search query. "
                            "Include entities, dates, or metrics for best results. "
                            "Example: 'Apple Q4 2024 revenue earnings report'"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            f"Number of results to return "
                            f"(default: {max_results}, max: {_MAX_RESULT_COUNT})"
                        ),
                        "default": max_results,
                    },
                    "freshness": {
                        "type": "string",
                        "description": (
                            "Time filter: 'pd' (past day), 'pw' (past week), "
                            "'pm' (past month), or omit for any time"
                        ),
                        "enum": list(_VALID_FRESHNESS),
                    },
                },
                "required": ["query"],
            },
            source_type="web_search",
            source_kind="web",
        )

    # -- ResearchTool protocol ------------------------------------------------

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean raw LLM arguments.

        Returns a canonical ``dict`` ready for :meth:`execute`.

        Raises:
            ValueError: If required arguments are missing or malformed.
        """
        errors: list[str] = []

        # -- query (required) -------------------------------------------------
        query = arguments.get("query")
        if not query:
            errors.append("'query' is required")
        elif not isinstance(query, str):
            errors.append("'query' must be a string")
        elif len(query) > _MAX_QUERY_LENGTH:
            errors.append(f"'query' must be {_MAX_QUERY_LENGTH} characters or less")

        # -- count (optional) -------------------------------------------------
        raw_count = arguments.get("count")
        if raw_count is not None:
            if not isinstance(raw_count, int):
                errors.append("'count' must be an integer")
            elif raw_count < 1 or raw_count > _MAX_RESULT_COUNT:
                errors.append(f"'count' must be between 1 and {_MAX_RESULT_COUNT}")

        # -- freshness (optional) ---------------------------------------------
        freshness = arguments.get("freshness")
        if freshness is not None and freshness not in _VALID_FRESHNESS:
            errors.append(f"'freshness' must be one of: {sorted(_VALID_FRESHNESS)}")

        if errors:
            raise ValueError("; ".join(errors))

        # Build cleaned dict with defaults applied.
        count: int = raw_count if isinstance(raw_count, int) else self._max_results
        count = min(max(count, 1), _MAX_RESULT_COUNT)

        cleaned: dict[str, Any] = {
            "query": str(query).strip(),
            "count": count,
        }
        if freshness is not None:
            cleaned["freshness"] = freshness
        return cleaned

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the web search and return a :class:`ToolResult`.

        Args:
            arguments: Validated arguments (from :meth:`validate_arguments`).
            context: Per-call context carrying the shared URL registry.

        Returns:
            ``ToolResult`` with formatted content, source metadata, and
            structured data (query, counts).
        """
        query: str = arguments["query"]
        count: int = arguments.get("count", self._max_results)
        freshness: str | None = arguments.get("freshness")

        try:
            raw_results = await self._client.search(
                query,
                count=count,
                freshness=freshness,
            )

            # Apply domain filtering if configured.
            if self._domain_filter:
                results = [
                    r for r in raw_results
                    if _domain_matches(r.url, self._domain_filter)
                ]
                filtered_count = len(raw_results) - len(results)
                if filtered_count > 0:
                    logger.debug(
                        "WEB_SEARCH_DOMAIN_FILTERED count_before=%d count_after=%d",
                        len(raw_results),
                        len(results),
                    )
            else:
                results = list(raw_results)

            # Build formatted output and source list.
            sources: list[SourceInfo] = []
            formatted_lines: list[str] = []
            registry = context.url_registry

            for result in results:
                # Register URL in shared registry if available.
                if registry is not None:
                    idx = registry.register(result.url)
                else:
                    idx = len(formatted_lines)

                # Truncate content when present.
                source_content: str | None = None
                if result.content:
                    source_content = result.content[: self._max_content_per_result]

                sources.append(
                    SourceInfo(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        content=source_content,
                        source_type="web",
                    )
                )

                # LLM sees indices only — never raw URLs.
                if source_content:
                    formatted_lines.append(
                        f"[{idx}] **{result.title}**\n{source_content}"
                    )
                else:
                    formatted_lines.append(
                        f"[{idx}] **{result.title}**\n    {result.snippet}"
                    )

            if not formatted_lines:
                content = "No search results found. Try a different query."
            else:
                content = "\n\n".join(formatted_lines)

            logger.info(
                "WEB_SEARCH_COMPLETE query=%s results=%d",
                query[:80],
                len(results),
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "query": query,
                    "total_results": len(results),
                    "count": count,
                },
            )

        except Exception as e:
            logger.error("WEB_SEARCH_ERROR query=%s error=%s", query[:80], e)
            return ToolResult(
                content=f"Search failed: {e}",
                success=False,
                error=str(e),
            )
