"""Research tools for ReAct-based agentic researcher.

Provides OpenAI-format tool definitions for:
- web_search: Search the web for information (returns indexed results)
- web_crawl: Fetch full content using source index

These tools follow the OpenAI function calling schema and are executed
by Python functions in the ReAct researcher loop.

SECURITY: The LLM never sees actual URLs - only numeric indices.
This prevents URL hallucination and prompt injection via malicious URLs.
URL resolution happens internally via UrlRegistry.

The LLM naturally judges content quality after crawling - no dedicated
quality evaluation tool needed.
"""

from typing import Any

# OpenAI-format tool definitions for the research agent
# Only 2 tools: search and crawl. LLM judges quality naturally.
# URLs are hidden - LLM only sees indices.
RESEARCH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information. Returns numbered results with titles and snippets. "
                "Use the result INDEX numbers (0, 1, 2, etc.) to select sources for crawling."
            ),
            "parameters": {
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
                        "description": "Number of results to return (default: 5, max: 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_crawl",
            "description": (
                "Fetch full content from a source. Use the INDEX number from search results "
                "(0, 1, 2, etc.). Returns extracted page text for analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index number of the source from search results (0, 1, 2, etc.)",
                    },
                },
                "required": ["index"],
            },
        },
    },
]


def get_tool_names() -> list[str]:
    """Get the names of all available research tools."""
    return [tool["function"]["name"] for tool in RESEARCH_TOOLS]


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get a tool definition by name."""
    for tool in RESEARCH_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None
