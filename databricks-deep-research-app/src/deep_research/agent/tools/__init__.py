"""
Research Tools Module
====================

Provides the tool infrastructure for research agents:
- ResearchTool: Protocol for implementing tools
- ToolDefinition: Tool schema for LLM function calling
- ToolResult: Result from tool execution
- ResearchContext: Context passed to tool execution
- ToolRegistry: Central registry for tool management

ResearchTool Implementations:
- WebSearchTool: Web search via Brave API
- WebCrawlTool: Web page content extraction
- UserVectorSearchTool: Databricks Vector Search (with OBO, hybrid, reranking, multi-query RRF)
- KnowledgeAssistantTool: Databricks Knowledge Assistants
"""

# Tool infrastructure (new plugin architecture)
from deep_research.agent.tools.base import (
    ResearchContext,
    ResearchTool,
    ToolDefinition,
    ToolList,
    ToolMap,
    ToolResult,
)
from deep_research.agent.tools.knowledge_assistant import (
    KnowledgeAssistantTool,
    create_knowledge_assistant_tools_from_config,
)
from deep_research.agent.tools.registry import ToolRegistry, ToolRegistryError

# Legacy tool interface (for backward compatibility with OpenAI format)
from deep_research.agent.tools.research_tools import (
    RESEARCH_TOOLS,
    get_tool_by_name,
    get_tool_names,
)

# Source routing for enterprise data sources (007-enterprise-data-sources, T053)
from deep_research.agent.tools.source_routing import (
    filter_tool_definitions_by_constraint,
    filter_tools_by_constraint,
    get_prioritized_sources,
    get_tool_source_type,
    prompt_for_required_sources,
    should_force_required_source_query,
    validate_required_sources_consulted,
)
from deep_research.agent.tools.user_vector_search import (
    UserVectorSearchTool,
    reciprocal_rank_fusion,
)

# ResearchTool implementations
from deep_research.agent.tools.web_crawler import WebCrawler, WebCrawlTool, web_crawl
from deep_research.agent.tools.web_search import WebSearchTool, web_search

__all__ = [
    # Tool infrastructure (protocol and registry)
    "ResearchTool",
    "ToolDefinition",
    "ToolResult",
    "ResearchContext",
    "ToolList",
    "ToolMap",
    "ToolRegistry",
    "ToolRegistryError",
    # ResearchTool implementations
    "WebSearchTool",
    "WebCrawlTool",
    "UserVectorSearchTool",
    "reciprocal_rank_fusion",
    "KnowledgeAssistantTool",
    "create_knowledge_assistant_tools_from_config",
    # Legacy functional interface (still used by agents)
    "web_search",
    "web_crawl",
    "WebCrawler",
    # Legacy OpenAI format tool definitions
    "RESEARCH_TOOLS",
    "get_tool_names",
    "get_tool_by_name",
    # Source routing (007-enterprise-data-sources, T053)
    "filter_tools_by_constraint",
    "filter_tool_definitions_by_constraint",
    "validate_required_sources_consulted",
    "prompt_for_required_sources",
    "get_tool_source_type",
    "get_prioritized_sources",
    "should_force_required_source_query",
]
