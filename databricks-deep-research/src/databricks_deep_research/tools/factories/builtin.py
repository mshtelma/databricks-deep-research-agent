"""Builtin tool factory — creates web_search, web_crawl, file_search tools."""

from __future__ import annotations

from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

_SUPPORTED_KINDS = frozenset({"web_search", "web_crawl", "file_search"})


class BuiltinToolFactory:
    """Creates web_search, web_crawl, and file_search tools from declarations."""

    def supports(self, kind: str) -> bool:
        return kind in _SUPPORTED_KINDS

    async def create(
        self, decl: ToolDeclaration, ctx: ToolFactoryContext
    ) -> ResearchTool:
        if decl.kind == "web_search":
            if ctx.search_client is None:
                raise ValueError(
                    f"search_client required in ToolFactoryContext for "
                    f"web_search tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.web_search import WebSearchTool

            return WebSearchTool(
                search_client=ctx.search_client,
                domain_filter=decl.config.get("domain_filter"),
                max_results=decl.config.get("max_results", 5),
            )

        if decl.kind == "web_crawl":
            from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool

            return WebCrawlTool(
                crawler=ctx.crawler,
                timeout=decl.config.get("timeout", 30.0),
                max_content_length=decl.config.get("max_content_length", 50_000),
            )

        if decl.kind == "file_search":
            if ctx.file_index is None:
                raise ValueError(
                    f"file_index required in ToolFactoryContext for "
                    f"file_search tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.file_search import FileSearchTool

            return FileSearchTool(file_index=ctx.file_index)

        raise ValueError(f"Unsupported kind: {decl.kind}")
