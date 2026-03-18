"""Tool factory for creating tools from user data sources.

Dynamically creates tool instances (VectorSearchTool, GenieTool,
KnowledgeAssistantTool) from UserDataSource configurations stored
in the database, or from auto-discovered sources in the discovery cache.

Part of 007-enterprise-data-sources feature (T015).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deep_research.agent.tools.base import ResearchTool
from deep_research.models.data_source import DataSourceType, UserDataSource
from deep_research.services.obo_client import OBODatabricksClient

if TYPE_CHECKING:
    from deep_research.schemas.discovery import DiscoveredSource

logger = logging.getLogger(__name__)


async def create_tools_from_user_sources(
    sources: list[UserDataSource],
    obo_client: OBODatabricksClient | None = None,
) -> list[ResearchTool]:
    """Create tool instances from user data source configurations.

    Dynamically creates appropriate tool instances based on source type:
    - VECTOR_SEARCH -> VectorSearchTool
    - GENIE -> GenieTool
    - KNOWLEDGE_ASSISTANT -> KnowledgeAssistantTool

    Args:
        sources: List of UserDataSource configurations.
        obo_client: OBO client for enterprise authentication.

    Returns:
        List of ResearchTool instances ready for use.
    """
    if obo_client is None:
        obo_client = OBODatabricksClient()

    tools: list[ResearchTool] = []

    for source in sources:
        try:
            source_type = DataSourceType(source.type)

            if source_type == DataSourceType.VECTOR_SEARCH:
                tool = _create_vector_search_tool(source, obo_client)
            elif source_type == DataSourceType.GENIE:
                tool = _create_genie_tool(source, obo_client)
            elif source_type == DataSourceType.KNOWLEDGE_ASSISTANT:
                tool = _create_knowledge_assistant_tool(source, obo_client)
            else:
                logger.debug(
                    "Skipping unsupported source type",
                    extra={"source_id": str(source.id), "type": source.type},
                )
                continue

            if tool:
                tools.append(tool)
                logger.info(
                    "Created tool from user source",
                    extra={
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "tool_name": tool.definition.name,
                    },
                )

        except Exception as e:
            logger.error(
                "TOOL_CREATION_FROM_SOURCE_FAILED",
                extra={
                    "source_id": str(source.id),
                    "source_name": source.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

    return tools


def _create_vector_search_tool(
    source: UserDataSource,
    obo_client: OBODatabricksClient,
) -> "ResearchTool | None":
    """Create VectorSearchTool from UserDataSource config."""
    from deep_research.agent.tools.user_vector_search import UserVectorSearchTool

    config = source.config

    return UserVectorSearchTool(
        obo_client=obo_client,
        data_source=source,
        endpoint_name=config.get("endpoint_name", ""),
        index_name=config.get("index_name", source.endpoint_identifier),
        columns=config.get("columns", []),
        columns_to_rerank=config.get("columns_to_rerank", []),
        enable_hybrid=config.get("enable_hybrid", True),
        enable_reranking=config.get("enable_reranking", True),
        num_results=config.get("num_results", 10),
        description=source.description,
    )


def _create_genie_tool(
    source: UserDataSource,
    obo_client: OBODatabricksClient,
) -> "ResearchTool | None":
    """Create GenieTool from UserDataSource config."""
    from deep_research.agent.tools.genie import GenieTool

    config = source.config

    return GenieTool(
        obo_client=obo_client,
        space_id=config.get("space_id", source.endpoint_identifier),
        name=source.name,
        description=source.description or f"Query {source.name} for data analysis",
        example_questions=config.get("example_questions", []),
    )


def _create_knowledge_assistant_tool(
    source: UserDataSource,
    obo_client: OBODatabricksClient,
) -> "ResearchTool | None":
    """Create KnowledgeAssistantTool from UserDataSource config."""
    from deep_research.agent.tools.knowledge_assistant import (
        create_knowledge_assistant_from_user_source,
    )

    return create_knowledge_assistant_from_user_source(source, obo_client)


async def get_enabled_tools_for_user(
    user_id: str,
    user_token: str | None,
    session: Any,
) -> list[ResearchTool]:
    """Get all enabled tools for a user, including user-configured sources.

    Combines:
    - System tools (web_search, web_crawl)
    - Config-based tools (from app.yaml vector_search endpoints)
    - User data sources (from database)

    Args:
        user_id: Databricks workspace user ID.
        user_token: User's OAuth token for OBO authentication.
        session: Async SQLAlchemy session.

    Returns:
        List of all enabled ResearchTool instances.
    """
    from deep_research.services.data_source_service import DataSourceService

    tools: list[ResearchTool] = []

    # Get user's configured data sources
    service = DataSourceService(session)
    sources, _ = await service.get_accessible_sources(user_id, only_valid=True)

    if sources:
        obo_client = OBODatabricksClient()
        user_tools = await create_tools_from_user_sources(sources, obo_client)
        tools.extend(user_tools)
        logger.info(
            "FACTORY_USER_TOOLS_LOADED",
            extra={
                "count": len(user_tools),
                "has_obo_token": user_token is not None,
            },
        )

    return tools


async def create_tools_from_discovered_sources(
    sources: list[DiscoveredSource],
) -> list[ResearchTool]:
    """Create tool instances from auto-discovered sources.

    Unlike create_tools_from_user_sources() which requires saved DB records,
    this creates tools from discovery cache entries. Used when the user selects
    sources from discovery UI without saving them.

    Args:
        sources: DiscoveredSource objects from the discovery cache.

    Returns:
        List of ResearchTool instances ready for use.
    """
    obo_client = OBODatabricksClient()
    tools: list[ResearchTool] = []

    for source in sources:
        try:
            # source_type is a string due to use_enum_values=True on DiscoveredSource
            source_type = DataSourceType(source.source_type)

            if source_type == DataSourceType.VECTOR_SEARCH:
                tool = _create_vs_from_discovery(source, obo_client)
            elif source_type == DataSourceType.GENIE:
                tool = _create_genie_from_discovery(source, obo_client)
            elif source_type == DataSourceType.KNOWLEDGE_ASSISTANT:
                tool = _create_ka_from_discovery(source, obo_client)
            else:
                logger.warning(
                    "DISCOVERY_TOOL_SKIP_UNSUPPORTED_TYPE",
                    extra={
                        "source_id": source.source_id,
                        "source_type": source.source_type,
                    },
                )
                continue

            if tool:
                tools.append(tool)
                logger.info(
                    "TOOL_CREATED_FROM_DISCOVERY",
                    extra={
                        "source_id": source.source_id,
                        "source_name": source.name,
                        "source_type": source.source_type,
                        "tool_name": tool.definition.name,
                    },
                )
        except Exception as e:
            logger.error(
                "TOOL_FROM_DISCOVERY_FAILED",
                extra={
                    "source_id": source.source_id,
                    "source_name": source.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

    return tools


def _create_vs_from_discovery(
    source: DiscoveredSource,
    obo_client: OBODatabricksClient,
) -> ResearchTool | None:
    """Create UserVectorSearchTool from a DiscoveredSource."""
    from deep_research.agent.tools.user_vector_search import UserVectorSearchTool
    from deep_research.services.vector_search_query import ColumnRoles

    metadata = source.metadata
    index_name = metadata.get("index_name", source.endpoint_name)
    endpoint_name = metadata.get("endpoint_name", "")

    # Use pre-discovered columns if available (from discovery enrichment).
    # If empty/missing, tool will discover at query time via get_index().
    columns = metadata.get("queryable_columns") or None
    content_column = metadata.get("content_column")
    primary_key = metadata.get("primary_key", "id")

    # Construct ColumnRoles from enrichment metadata so the query parser
    # maps the actual content column (e.g. "text_content"), not the
    # hardcoded legacy name "content".
    column_roles: ColumnRoles | None = None
    if columns and content_column:
        column_roles = ColumnRoles(
            id_column=primary_key,
            content_column=content_column,
            all_columns=columns,
        )

    # Reranking target: the content column is what should be reranked
    columns_to_rerank = [content_column] if content_column else []

    return UserVectorSearchTool(
        obo_client=obo_client,
        source_name=source.name,
        endpoint_name=endpoint_name,
        index_name=index_name,
        columns=columns,
        columns_to_rerank=columns_to_rerank,
        column_roles=column_roles,
        enable_hybrid="hybrid" in source.capabilities,
        enable_reranking="reranking" in source.capabilities,
        description=source.description,
    )


def _create_genie_from_discovery(
    source: DiscoveredSource,
    obo_client: OBODatabricksClient,
) -> ResearchTool | None:
    """Create GenieTool from a DiscoveredSource."""
    from deep_research.agent.tools.genie import GenieTool

    metadata = source.metadata
    space_id = metadata.get("space_id", source.endpoint_name)

    return GenieTool(
        obo_client=obo_client,
        space_id=space_id,
        name=source.name,
        description=source.description or f"Query {source.name} for data analysis",
    )


def _create_ka_from_discovery(
    source: DiscoveredSource,
    obo_client: OBODatabricksClient,
) -> ResearchTool | None:
    """Create KnowledgeAssistantTool from a DiscoveredSource."""
    from deep_research.agent.tools.knowledge_assistant import KnowledgeAssistantTool

    metadata = source.metadata
    endpoint_name = metadata.get("endpoint_name", source.endpoint_name)

    return KnowledgeAssistantTool(
        endpoint_name=endpoint_name,
        description=source.description or f"Ask {source.name} questions",
        obo_client=obo_client,
    )


def create_tools_from_source_ids(
    source_ids: list[str],
) -> list[ResearchTool]:
    """Create tools directly from source IDs — no discovery or DB needed.

    Last-resort fallback when both DB and discovery cache are unavailable.
    Parses source_id prefix to determine tool type and extracts the
    identifier needed for construction.

    Source ID format (from discovery_service.py):
    - "assistant:{endpoint_name}" → KnowledgeAssistantTool
    - "genie:{space_id}" → GenieTool
    - "vs:{index_name}" → UserVectorSearchTool
    """
    obo_client = OBODatabricksClient()
    tools: list[ResearchTool] = []

    for source_id in source_ids:
        try:
            tool = _create_tool_from_source_id(source_id, obo_client)
            if tool:
                tools.append(tool)
                logger.info(
                    "TOOL_CREATED_FROM_SOURCE_ID",
                    extra={
                        "source_id": source_id,
                        "tool_name": tool.definition.name,
                        "source_type": tool.definition.source_type,
                    },
                )
        except Exception as e:
            logger.warning(
                "TOOL_FROM_SOURCE_ID_FAILED",
                extra={
                    "source_id": source_id,
                    "error": str(e)[:200],
                    "error_type": type(e).__name__,
                },
            )

    return tools


def _create_tool_from_source_id(
    source_id: str,
    obo_client: OBODatabricksClient,
) -> ResearchTool | None:
    """Parse source_id and create the appropriate tool."""
    if source_id.startswith("assistant:"):
        from deep_research.agent.tools.knowledge_assistant import KnowledgeAssistantTool

        endpoint_name = source_id[len("assistant:"):]
        if not endpoint_name:
            logger.warning(
                "TOOL_SOURCE_ID_EMPTY_IDENTIFIER",
                extra={"source_id": source_id},
            )
            return None
        return KnowledgeAssistantTool(
            endpoint_name=endpoint_name,
            obo_client=obo_client,
        )

    elif source_id.startswith("genie:"):
        from deep_research.agent.tools.genie import GenieTool

        space_id = source_id[len("genie:"):]
        if not space_id:
            logger.warning(
                "TOOL_SOURCE_ID_EMPTY_IDENTIFIER",
                extra={"source_id": source_id},
            )
            return None
        truncated = f"{space_id[:12]}..." if len(space_id) > 12 else space_id
        return GenieTool(
            obo_client=obo_client,
            space_id=space_id,
            name=f"Genie ({truncated})",
        )

    elif source_id.startswith("vs:"):
        from deep_research.agent.tools.user_vector_search import UserVectorSearchTool

        index_name = source_id[len("vs:"):]
        if not index_name:
            logger.warning(
                "TOOL_SOURCE_ID_EMPTY_IDENTIFIER",
                extra={"source_id": source_id},
            )
            return None
        return UserVectorSearchTool(
            obo_client=obo_client,
            index_name=index_name,
        )

    else:
        logger.debug(
            "TOOL_SOURCE_ID_UNKNOWN_PREFIX",
            extra={"source_id": source_id[:50]},
        )
        return None
