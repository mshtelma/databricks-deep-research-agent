"""Synthesis tools for ReAct-based report generation.

Provides OpenAI-format tool definitions for grounded synthesis:
- search_evidence: Search pre-collected evidence pool for relevant snippets
- read_snippet: Get full text of an evidence span by index

These tools operate over the evidence pool from Stage 1 (Evidence Pre-Selection),
NOT the raw web. The LLM must use these tools before making factual claims
to ensure grounded generation.

SECURITY: The LLM never sees actual URLs - only numeric indices.
URL/source resolution happens internally via EvidenceRegistry.

The key difference from research tools:
- Research tools: Search/crawl the live web
- Synthesis tools: Search/read pre-collected, pre-ranked evidence
"""

from typing import Any

# OpenAI-format tool definitions for ReAct synthesis
# 2 tools: search_evidence and read_snippet
# URLs are hidden - LLM only sees indices
SYNTHESIS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": (
                "Search the collected evidence pool for snippets relevant to a claim you want to make. "
                "Returns top-5 matching evidence spans with their indices. "
                "Use this BEFORE writing any factual claim to find supporting evidence. "
                "After finding relevant evidence, use read_snippet to see the full quote."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language description of the fact or claim you want to support. "
                            "Be specific about what data, statistic, or fact you're looking for. "
                            "Example: 'GPT-4 accuracy on MMLU benchmark'"
                        ),
                    },
                    "claim_type": {
                        "type": "string",
                        "enum": ["factual", "numeric", "comparative", "definition"],
                        "description": (
                            "Type of claim to help prioritize numeric vs prose evidence. "
                            "Use 'numeric' when looking for statistics or numbers."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_snippet",
            "description": (
                "Get the full text of a specific evidence span by index. "
                "Call this after search_evidence returns relevant indices to read the actual quote. "
                "Returns the full quote text, source title, and section heading. "
                "Base your claim on what you read - use similar phrasing and exact numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index of the evidence from search_evidence results (0, 1, 2, etc.)",
                    },
                },
                "required": ["index"],
            },
        },
    },
]


def get_synthesis_tool_names() -> list[str]:
    """Get the names of all available synthesis tools."""
    return [tool["function"]["name"] for tool in SYNTHESIS_TOOLS]


def get_synthesis_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get a synthesis tool definition by name."""
    for tool in SYNTHESIS_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results for LLM consumption.

    Args:
        results: List of result dicts from EvidenceRegistry.search()

    Returns:
        Formatted string with numbered results.
    """
    if not results:
        return "No relevant evidence found. Try a different search query."

    lines = ["Found relevant evidence:"]
    for r in results:
        numeric_flag = " [NUMERIC]" if r.get("has_numeric") else ""
        lines.append(
            f"[{r['index']}] {r['title']}{numeric_flag}\n"
            f"    Preview: {r['snippet_preview']}"
        )
    lines.append("\nUse read_snippet with an index to see the full quote.")
    return "\n".join(lines)


def format_snippet(
    index: int,
    title: str | None,
    quote_text: str,
    section_heading: str | None,
    citation_key: str,
) -> str:
    """Format a snippet for LLM consumption.

    Args:
        index: Evidence index.
        title: Source title.
        quote_text: Full quote text.
        section_heading: Section where quote was found.
        citation_key: Citation key for referencing.

    Returns:
        Formatted string with evidence details.
    """
    section = f" (Section: {section_heading})" if section_heading else ""
    return (
        f"Evidence [{index}] from {title or 'Unknown Source'}{section}:\n"
        f'"{quote_text}"\n\n'
        f"Citation key: [{citation_key}]\n"
        f'Use: <cite key="{citation_key}">Your claim based on this evidence.</cite>'
    )
