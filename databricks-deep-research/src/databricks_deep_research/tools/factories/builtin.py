"""Builtin tool factory — creates web_search, web_crawl, file_search tools."""

from __future__ import annotations

import logging
import os
from typing import Any

from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)

_SUPPORTED_KINDS = frozenset({
    "web_search", "web_crawl", "file_search", "compute", "compute_namespace",
    "delta_read", "delta_grep", "delta_context", "delta_table_read",
})

_SEARCH_PROVIDERS = frozenset({"brave", "jina"})
_CRAWL_PROVIDERS = frozenset({"jina"})


def _resolve_search_provider(provider: str, ctx: ToolFactoryContext) -> Any:
    """Create a SearchClient for the named provider."""
    if provider == "brave":
        api_key = ctx.api_keys.get("brave") or os.environ.get("BRAVE_API_KEY")
        if not api_key:
            raise ValueError(
                "Brave search requires BRAVE_API_KEY env var or "
                "api_keys['brave'] in ToolFactoryContext"
            )
        from databricks_deep_research.tools.builtins.brave_search import (
            BraveSearchAdapter,
        )

        return BraveSearchAdapter(api_key=api_key)

    if provider == "jina":
        api_key = ctx.api_keys.get("jina") or os.environ.get("JINA_API_KEY")
        from databricks_deep_research.tools.builtins.jina_search import (
            JinaSearchAdapter,
        )

        return JinaSearchAdapter(api_key=api_key)

    raise ValueError(
        f"Unknown search provider: {provider!r}. "
        f"Supported: {sorted(_SEARCH_PROVIDERS)}"
    )


def _resolve_crawl_provider(provider: str, ctx: ToolFactoryContext) -> Any:
    """Create a ContentCrawler for the named provider."""
    if provider == "jina":
        api_key = ctx.api_keys.get("jina") or os.environ.get("JINA_API_KEY")
        from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter

        return JinaCrawlAdapter(api_key=api_key)

    raise ValueError(
        f"Unknown crawl provider: {provider!r}. "
        f"Supported: {sorted(_CRAWL_PROVIDERS)}"
    )


class BuiltinToolFactory:
    """Creates web_search, web_crawl, and file_search tools from declarations."""

    def supports(self, kind: str) -> bool:
        return kind in _SUPPORTED_KINDS

    async def create(
        self, decl: ToolDeclaration, ctx: ToolFactoryContext
    ) -> ResearchTool:
        if decl.kind == "web_search":
            provider = decl.config.get("provider")
            if provider is None:
                # Legacy path: use pre-built ctx.search_client.
                if ctx.search_client is None:
                    raise ValueError(
                        f"search_client required in ToolFactoryContext for "
                        f"web_search tool '{decl.name}'"
                    )
                search_client = ctx.search_client
            else:
                search_client = _resolve_search_provider(provider, ctx)

            from databricks_deep_research.tools.builtins.web_search import WebSearchTool

            return WebSearchTool(
                search_client=search_client,
                domain_filter=decl.config.get("domain_filter"),
                max_results=decl.config.get("max_results", 5),
                max_content_per_result=decl.config.get(
                    "max_content_per_result", 5000
                ),
            )

        if decl.kind == "web_crawl":
            provider = decl.config.get("provider")
            if provider is not None:
                crawler = _resolve_crawl_provider(provider, ctx)
            else:
                crawler = ctx.crawler
            from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool

            return WebCrawlTool(
                crawler=crawler,
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

        if decl.kind == "compute":
            from databricks_deep_research.tools.builtins.compute import PythonComputeTool

            return PythonComputeTool(
                name=decl.name,
                allowed_modules=decl.config.get("allowed_modules"),
                extra_modules=decl.config.get("extra_modules"),
                max_execution_seconds=decl.config.get("max_execution_seconds", 10.0),
                max_output_chars=decl.config.get("max_output_chars", 10_000),
                max_code_length=decl.config.get("max_code_length", 20_000),
                description=decl.description,
            )

        if decl.kind == "compute_namespace":
            from databricks_deep_research.tools.builtins.compute import PythonComputeTool
            from databricks_deep_research.tools.builtins.compute_namespace import (
                ComputeNamespaceListTool,
            )

            def _resolve_compute() -> PythonComputeTool | None:
                """Lazy resolution: look up sibling 'compute' tool from resolver cache."""
                compute_name = decl.config.get("compute_tool_name", "compute")
                cached = ctx.extras.get("_resolver_cache", {}).get(compute_name)
                if isinstance(cached, PythonComputeTool):
                    return cached
                return None

            return ComputeNamespaceListTool(
                compute_resolver=_resolve_compute,
                name=decl.name,
                description=decl.description,
            )

        if decl.kind == "delta_read":
            if not ctx.workspace_client:
                raise ValueError(
                    f"workspace_client required in ToolFactoryContext for "
                    f"delta_read tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.delta_read import DeltaReadTool

            return DeltaReadTool(
                name=decl.name,
                description=decl.description,
                table_name=decl.config["table_name"],
                columns=decl.config.get("columns", ["*"]),
                workspace_client=ctx.workspace_client,
                warehouse_id=decl.config["warehouse_id"],
                content_column=decl.config.get("content_column", "content"),
                order_by=decl.config.get("order_by", "chunk_id"),
                exclude_chunk_types=decl.config.get("exclude_chunk_types"),
            )

        if decl.kind == "delta_grep":
            if not ctx.workspace_client:
                raise ValueError(
                    f"workspace_client required in ToolFactoryContext for "
                    f"delta_grep tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.delta_read import DeltaGrepTool

            return DeltaGrepTool(
                name=decl.name,
                description=decl.description,
                table_name=decl.config["table_name"],
                columns=decl.config.get("columns", ["*"]),
                workspace_client=ctx.workspace_client,
                warehouse_id=decl.config["warehouse_id"],
                content_column=decl.config.get("content_column", "content"),
                order_by=decl.config.get("order_by", "chunk_id"),
                exclude_chunk_types=decl.config.get("exclude_chunk_types"),
            )

        if decl.kind == "delta_context":
            if not ctx.workspace_client:
                raise ValueError(
                    f"workspace_client required in ToolFactoryContext for "
                    f"delta_context tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.delta_read import DeltaContextTool

            return DeltaContextTool(
                name=decl.name,
                description=decl.description,
                table_name=decl.config["table_name"],
                columns=decl.config.get("columns", ["*"]),
                workspace_client=ctx.workspace_client,
                warehouse_id=decl.config["warehouse_id"],
                content_column=decl.config.get("content_column", "content"),
                order_by=decl.config.get("order_by", "chunk_id"),
            )

        if decl.kind == "delta_table_read":
            if not ctx.workspace_client:
                raise ValueError(
                    f"workspace_client required in ToolFactoryContext for "
                    f"delta_table_read tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.delta_read import (
                DeltaTableReadTool,
            )

            # Optional: auto-inject parsed JSON into sibling compute namespace
            _compute_resolver = None
            store_as = decl.config.get("store_in_compute")
            if store_as:
                _compute_name = decl.config.get("compute_tool_name", "compute")

                def _resolve_compute() -> Any:  # type: ignore[misc]
                    cached = ctx.extras.get("_resolver_cache", {}).get(_compute_name)
                    return cached if hasattr(cached, "inject_variable") else None

                _compute_resolver = _resolve_compute

            return DeltaTableReadTool(
                name=decl.name,
                description=decl.description,
                table_name=decl.config["table_name"],
                columns=decl.config.get("columns", ["*"]),
                workspace_client=ctx.workspace_client,
                warehouse_id=decl.config["warehouse_id"],
                content_column=decl.config.get("content_column", "content"),
                pk_column=decl.config.get("pk_column", "chunk_id"),
                store_in_compute=store_as,
                compute_resolver=_compute_resolver,
                structural_analysis=bool(decl.config.get("structural_analysis")),
            )

        raise ValueError(f"Unsupported kind: {decl.kind}")
