"""Tool adapter — creates framework tools from app config.

Constructor DI factory: creates framework-level ``ResearchTool``
implementations once per workflow execution, injecting all dependencies
(search clients, crawlers, user tokens) at construction time.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceKind,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

_APP_SOURCE_TYPE_TO_KIND: dict[str, str] = {
    "vector_search": SourceKind.vector_index,
    "genie": SourceKind.sql_analytics,
    "knowledge_assistant": SourceKind.qa_assistant,
    "file_search": SourceKind.file,
    "uploaded_file": SourceKind.file,
    "web_search": SourceKind.web,
    "web_crawl": SourceKind.web,
}


def _truncate_text(value: Any, limit: int = 500) -> str | None:
    """Convert *value* to text and truncate it for source snippets."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text[:limit]


def _normalize_source(source: Any) -> Any | None:
    """Convert app tool source payloads into app SourceInfo-like objects."""
    from deep_research.agent.state import SourceInfo as AppSourceInfo

    if source is None:
        return None

    if isinstance(source, dict):
        url = source.get("url")
        if not url:
            return None
        title = source.get("title") or source.get("filename") or source.get("source_name")
        content = _truncate_text(source.get("content"), 20000)
        snippet = (
            _truncate_text(source.get("snippet"))
            or _truncate_text(source.get("highlight"))
            or _truncate_text(content)
        )
        return AppSourceInfo(
            url=str(url),
            title=_truncate_text(title, 1000),
            snippet=snippet,
            content=content,
            relevance_score=source.get("relevance_score"),
            source_type=source.get("source_type") or source.get("type") or "web",
        )

    url = getattr(source, "url", None)
    if not url:
        return None

    content = _truncate_text(getattr(source, "content", None), 20000)
    snippet = (
        _truncate_text(getattr(source, "snippet", None))
        or _truncate_text(getattr(source, "highlight", None))
        or _truncate_text(content)
    )

    return AppSourceInfo(
        url=str(url),
        title=_truncate_text(
            getattr(source, "title", None)
            or getattr(source, "filename", None)
            or getattr(source, "source_name", None),
            1000,
        ),
        snippet=snippet,
        content=content,
        relevance_score=getattr(source, "relevance_score", None),
        source_type=(
            getattr(source, "source_type", None)
            or getattr(source, "type", None)
            or "web"
        ),
    )


def _normalize_sources(sources: Any) -> list[Any]:
    """Normalize structured tool sources while dropping malformed entries."""
    if not sources:
        return []

    normalized = []
    for source in sources:
        app_source = _normalize_source(source)
        if app_source is not None:
            normalized.append(app_source)
    return normalized


class EnterpriseToolAdapter:
    """Wraps an existing app enterprise tool as a framework ``ResearchTool``.

    The adapter delegates ``execute()`` to the underlying app tool while
    conforming to the framework's ``ResearchTool`` protocol.
    """

    def __init__(
        self,
        app_tool: Any,
        *,
        user_token: str | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._tool = app_tool
        self._user_token = user_token
        self._chat_id = chat_id
        self._user_id = user_id

    @property
    def definition(self) -> ToolDefinition:
        """Build framework ToolDefinition from the app tool."""
        app_definition = getattr(self._tool, "definition", None)
        name = (
            getattr(app_definition, "name", None)
            or getattr(self._tool, "name", None)
            or type(self._tool).__name__
        )
        description = (
            getattr(app_definition, "description", None)
            or getattr(self._tool, "description", None)
            or f"Enterprise tool: {name}"
        )
        params = (
            getattr(app_definition, "parameters", None)
            or getattr(self._tool, "parameters", None)
            or {"type": "object", "properties": {}}
        )
        source_type_str = (
            getattr(app_definition, "source_type", None)
            or getattr(self._tool, "source_type", None)
            or "enterprise"
        )
        source_kind_val = _APP_SOURCE_TYPE_TO_KIND.get(
            source_type_str, SourceKind.vector_index
        )
        metadata: dict[str, Any] = {
            "source_name": (
                getattr(self._tool, "_source_name", None)
                or getattr(self._tool, "source_name", None)
                or name
            ),
            "source_description": description,
            "backend": type(self._tool).__name__,
            "index_name": getattr(self._tool, "_index_name", None),
            "endpoint_name": getattr(self._tool, "_endpoint_name", None),
            "source_type": (
                getattr(app_definition, "source_type", None)
                or getattr(self._tool, "source_type", None)
                or "enterprise"
            ),
        }
        # VS tools: passthrough sends LLM query directly to embedding API
        if source_kind_val == SourceKind.vector_index:
            metadata["query_policy"] = "passthrough"

        return ToolDefinition(
            name=name,
            description=description,
            parameters=params,
            source_type=source_type_str,
            source_kind=source_kind_val,
            metadata=metadata,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Pass-through validation (app tools handle their own validation)."""
        return arguments

    async def execute(self, arguments: dict[str, Any], _context: Any = None) -> ToolResult:
        """Execute the underlying app tool with correct calling convention.

        App enterprise tools follow the protocol:
            execute(arguments: dict, context: ResearchContext)
        NOT **kwargs unpacking.
        """
        try:
            from uuid import UUID as _UUID
            from uuid import uuid4 as _uuid4

            from deep_research.agent.tools.base import ResearchContext as AppResearchContext

            app_context = AppResearchContext(
                chat_id=_UUID(self._chat_id) if self._chat_id else _uuid4(),
                user_id=self._user_id or "framework",
                user_token=self._user_token,
            )
            app_result = await self._tool.execute(
                arguments=arguments, context=app_context,
            )
            content = app_result.content if hasattr(app_result, "content") else str(app_result)
            success = app_result.success if hasattr(app_result, "success") else True
            sources = _normalize_sources(getattr(app_result, "sources", None))
            raw_data = getattr(app_result, "data", None)
            data = raw_data if isinstance(raw_data, dict) else {}
            error = getattr(app_result, "error", None)
            return ToolResult(
                content=content,
                success=success,
                sources=sources,
                data=data,
                error=error,
            )
        except Exception as e:
            logger.warning(
                "ENTERPRISE_TOOL_ERROR tool=%s error=%s",
                self.definition.name,
                str(e)[:200],
            )
            return ToolResult(content=f"Error: {e}", success=False)


class CrawlerAdapter:
    """Adapts app WebCrawler to framework ContentCrawler protocol.

    Framework expects: async __call__(url: str) -> (text, title | None)
    App provides:      async _fetch_url(url: str) -> CrawlResult(content, title, success, error)
    """

    def __init__(self, crawler: Any) -> None:
        self._crawler = crawler

    async def __call__(self, url: str) -> tuple[str, str | None]:
        result = await self._crawler._fetch_url(url)
        if not result.success:
            raise RuntimeError(result.error or f"Crawl failed for {url}")
        return result.content, result.title


class BraveSearchAdapter:
    """Adapts app BraveSearchClient to framework SearchClient protocol.

    Solves two integration issues:
    1. Return type: BraveSearchClient.search() returns SearchResponse (non-iterable
       dataclass), but framework WebSearchTool expects list[SearchResult].
    2. Per-agent domain filtering: applies custom agent DomainFilterConfig on top of
       the global domain filter already applied inside BraveSearchClient.
    """

    def __init__(
        self,
        client: Any,  # BraveSearchClient
        domain_filter_config: Any | None = None,  # DomainFilterConfig
    ) -> None:
        self._client = client
        self._per_agent_filter: Any | None = None
        if domain_filter_config is not None:
            from deep_research.services.search.domain_filter import DomainFilter

            self._per_agent_filter = DomainFilter(domain_filter_config)

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
    ) -> list[Any]:
        """Search via BraveSearchClient and return framework-compatible list.

        The global domain filter (from app.yaml search config) is applied
        inside BraveSearchClient.search(). The per-agent filter is applied
        here as an additional layer, matching the legacy orchestrator behavior
        where both filters are stacked.
        """
        response = await self._client.search(
            query, count=count, freshness=freshness,
        )
        results = response.results

        logger.info(
            "BRAVE_ADAPTER_SEARCH query=%s raw_results=%d filter_active=%s",
            query[:60],
            len(results),
            self._per_agent_filter is not None
            and getattr(self._per_agent_filter, "is_active", False),
        )

        # Apply per-agent domain filter (custom agent config)
        if self._per_agent_filter is not None and self._per_agent_filter.is_active:
            before = len(results)
            results = [
                r for r in results
                if self._per_agent_filter.is_allowed(r.url).allowed
            ]
            logger.info(
                "BRAVE_ADAPTER_FILTERED before=%d after=%d",
                before, len(results),
            )

        return list(results)


async def create_framework_tools(
    *,
    brave_client: Any | None = None,
    crawler: Any | None = None,
    domain_filter_config: Any | None = None,
    enterprise_tools: list[Any] | None = None,
    user_token: str | None = None,
    file_search_tool: Any | None = None,
    chat_id: str | None = None,
    user_id: str | None = None,
) -> list[ResearchTool]:
    """Create framework tools once per workflow execution.

    All dependencies are injected at construction time (constructor DI).
    The returned tools are ready to use throughout the workflow.

    Args:
        brave_client: Optional Brave search client for web search.
        crawler: Optional web crawler for content extraction.
        domain_filter_config: Optional DomainFilterConfig for per-agent domain filtering.
        enterprise_tools: Optional list of app enterprise tools to wrap.
        user_token: Optional OBO token for enterprise tool authentication.
        file_search_tool: Optional pre-built app file search tool instance.
        chat_id: Optional chat ID for enterprise tool context.
        user_id: Optional user ID for enterprise tool context.

    Returns:
        List of framework ResearchTool instances.
    """
    tools: list[ResearchTool] = []

    # Web search tool
    if brave_client is not None:
        try:
            from databricks_deep_research.tools.builtins.web_search import (
                WebSearchTool,
            )

            adapted_client = BraveSearchAdapter(
                client=brave_client,
                domain_filter_config=domain_filter_config,
            )
            tools.append(WebSearchTool(
                search_client=adapted_client,
            ))
        except ImportError:
            logger.warning("WEB_SEARCH_TOOL_UNAVAILABLE (missing httpx)")

    # Web crawl tool
    if crawler is not None:
        try:
            from databricks_deep_research.tools.builtins.web_crawl import (
                WebCrawlTool,
            )

            adapted_crawler = CrawlerAdapter(crawler)
            tools.append(WebCrawlTool(crawler=adapted_crawler))
        except ImportError:
            logger.warning("WEB_CRAWL_TOOL_UNAVAILABLE (missing trafilatura)")

    # File search tool (pre-built by framework_orchestrator)
    if file_search_tool is not None:
        tools.append(EnterpriseToolAdapter(
            app_tool=file_search_tool,
            user_token=user_token,
            chat_id=chat_id,
            user_id=user_id,
        ))

    # Enterprise tool adapters
    if enterprise_tools:
        for app_tool in enterprise_tools:
            tools.append(EnterpriseToolAdapter(
                app_tool=app_tool,
                user_token=user_token,
                chat_id=chat_id,
                user_id=user_id,
            ))

    logger.info(
        "FRAMEWORK_TOOLS_CREATED count=%d names=%s",
        len(tools),
        [t.definition.name for t in tools],
    )
    return tools


__all__ = ["BraveSearchAdapter", "CrawlerAdapter", "create_framework_tools", "EnterpriseToolAdapter"]
