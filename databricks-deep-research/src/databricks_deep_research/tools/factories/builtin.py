"""Builtin tool factory — creates web_search, web_crawl, file_search tools."""

from __future__ import annotations

from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

_SUPPORTED_KINDS = frozenset({
    "web_search", "web_crawl", "file_search", "compute", "compute_namespace",
    "delta_read", "delta_grep", "delta_context",
})


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

        raise ValueError(f"Unsupported kind: {decl.kind}")
