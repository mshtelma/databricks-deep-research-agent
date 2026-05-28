"""Databricks tool factory — creates vector_search, genie, knowledge_assistant tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from databricks_deep_research.tools.catalog_types import CatalogCard, SafeProbe
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

_SUPPORTED_KINDS = frozenset({"vector_search", "genie", "knowledge_assistant"})

_VS_OPTIMIZATION_KEYS = frozenset({
    "query_policy", "multi_query", "hyde", "rerank",
    "num_alternatives", "rerank_threshold", "rrf_k",
})


class DatabricksToolFactory:
    """Creates vector_search, genie, and knowledge_assistant tools from declarations."""

    catalog_cards: ClassVar[Mapping[str, CatalogCard]] = {
        "vector_search": CatalogCard(
            summary="Semantic-search an embedded text corpus and return ranked passages.",
            input_prose=(
                "Provide a natural-language query. The tool issues a vector-search "
                "lookup against the configured Databricks Vector Search index and "
                "returns the most semantically similar passages. Use rich, "
                "descriptive phrasing; query rewriting and multi-query strategies "
                "may be applied per the configured query_policy."
            ),
            output_prose=(
                "Returns a list of passages ranked by similarity. Each entry "
                "includes the passage content, its source identifier, and a "
                "similarity score. Citations register through the framework so the "
                "synthesizer can ground claims back to specific passages."
            ),
        ),
        "genie": CatalogCard(
            summary="Ask a natural-language question against a Databricks Genie data room.",
            input_prose=(
                "Provide a question in plain English about the data in the configured "
                "Genie space. Genie translates the question to SQL, runs it against "
                "the underlying tables, and returns a structured answer. Best for "
                "well-defined analytical questions over governed tables."
            ),
            output_prose=(
                "Returns Genie's structured response, typically including the "
                "generated SQL, a tabular result, and a natural-language summary. "
                "Result rows can be referenced as evidence."
            ),
        ),
        "knowledge_assistant": CatalogCard(
            summary="Query a Databricks Knowledge Assistant endpoint for cited answers.",
            input_prose=(
                "Provide a natural-language question. The tool calls the configured "
                "Knowledge Assistant serving endpoint, which performs retrieval "
                "against its bound corpus and synthesizes a cited answer."
            ),
            output_prose=(
                "Returns the assistant's answer with inline citations to specific "
                "documents. Source pointers carry through the framework's URL "
                "registry so the synthesizer can ground further claims."
            ),
        ),
    }

    safe_probes: ClassVar[Mapping[str, SafeProbe | None]] = {
        "vector_search": None,
        "genie": None,
        "knowledge_assistant": None,
    }

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

            vs_metadata = {k: v for k, v in decl.config.items() if k in _VS_OPTIMIZATION_KEYS}

            return DatabricksVectorSearchTool(
                workspace_client=ctx.workspace_client,
                name=decl.name,
                index_name=decl.config["index_name"],
                columns=decl.config.get("columns"),
                num_results=decl.config.get("num_results", 10),
                query_type=decl.config.get("query_type"),
                filters_json=decl.config.get("filters_json"),
                description=decl.description,
                metadata=vs_metadata,
                exclude_chunk_types=decl.config.get("exclude_chunk_types"),
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
