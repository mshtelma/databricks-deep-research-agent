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
- VectorSearchTool: Databricks Vector Search
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
from deep_research.agent.tools.registry import ToolRegistry, ToolRegistryError

# ResearchTool implementations
from deep_research.agent.tools.web_crawler import WebCrawler, WebCrawlTool, web_crawl
from deep_research.agent.tools.web_search import WebSearchTool, web_search
from deep_research.agent.tools.vector_search import (
    VectorSearchTool,
    create_vector_search_tools_from_config,
)
from deep_research.agent.tools.knowledge_assistant import (
    KnowledgeAssistantTool,
    create_knowledge_assistant_tools_from_config,
)

# Legacy tool interface (for backward compatibility with OpenAI format)
from deep_research.agent.tools.research_tools import (
    RESEARCH_TOOLS,
    get_tool_by_name,
    get_tool_names,
)

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
    "VectorSearchTool",
    "create_vector_search_tools_from_config",
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
]
