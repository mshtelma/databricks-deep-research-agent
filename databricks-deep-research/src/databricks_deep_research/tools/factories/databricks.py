"""Databricks tool factory — creates vector_search, genie, knowledge_assistant tools."""

from __future__ import annotations

from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

_SUPPORTED_KINDS = frozenset({"vector_search", "genie", "knowledge_assistant"})


class DatabricksToolFactory:
    """Creates vector_search, genie, and knowledge_assistant tools from declarations."""

    def supports(self, kind: str) -> bool:
        return kind in _SUPPORTED_KINDS

    async def create(
        self, decl: ToolDeclaration, ctx: ToolFactoryContext
    ) -> ResearchTool:
        if ctx.workspace_client is None:
            raise ValueError(
                f"workspace_client required in ToolFactoryContext for "
                f"{decl.kind} tool '{decl.name}'"
            )

        if decl.kind == "vector_search":
            if "index_name" not in decl.config:
                raise ValueError(
                    f"'index_name' required in config for vector_search tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.vector_search import (
                DatabricksVectorSearchTool,
            )

            return DatabricksVectorSearchTool(
                workspace_client=ctx.workspace_client,
                name=decl.name,
                index_name=decl.config["index_name"],
                columns=decl.config.get("columns"),
                num_results=decl.config.get("num_results", 10),
                query_type=decl.config.get("query_type"),
                filters_json=decl.config.get("filters_json"),
                description=decl.description,
            )

        if decl.kind == "genie":
            if "space_id" not in decl.config:
                raise ValueError(
                    f"'space_id' required in config for genie tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.genie import DatabricksGenieTool

            return DatabricksGenieTool(
                workspace_client=ctx.workspace_client,
                name=decl.name,
                space_id=decl.config["space_id"],
                description=decl.description,
            )

        if decl.kind == "knowledge_assistant":
            if "endpoint_name" not in decl.config:
                raise ValueError(
                    f"'endpoint_name' required in config for "
                    f"knowledge_assistant tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.knowledge_assistant import (
                DatabricksKnowledgeAssistantTool,
            )

            return DatabricksKnowledgeAssistantTool(
                workspace_client=ctx.workspace_client,
                name=decl.name,
                endpoint_name=decl.config["endpoint_name"],
                description=decl.description,
            )

        raise ValueError(f"Unsupported kind: {decl.kind}")
