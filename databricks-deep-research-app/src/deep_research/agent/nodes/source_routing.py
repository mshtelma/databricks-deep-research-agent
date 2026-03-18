"""Source-aware tool filtering for researcher steps.

This module implements intelligent tool filtering based on source hints
from the planner or manual step definitions. It ensures the researcher
only uses tools appropriate for each step.

Part of 007-enterprise-data-sources feature (T038).
"""

from copy import deepcopy
from typing import Any

from deep_research.agent.state import ResearchState
from deep_research.core.logging_utils import get_logger
from deep_research.schemas.manual_step import SourceConstraint
from deep_research.schemas.plan import StepSourceHint

logger = get_logger(__name__)


def filter_tools_for_step(
    step_id: str,
    available_tools: list[dict[str, Any]],
    state: ResearchState,
) -> list[dict[str, Any]]:
    """Filter tools based on source hints for a specific step.

    This function filters the available tools based on:
    1. Source hints from the planner (via phase_results)
    2. Source constraints from manual step definitions
    3. Source scope configuration

    Args:
        step_id: ID of the current research step.
        available_tools: List of OpenAI-format tool definitions.
        state: Current research state with source configuration.

    Returns:
        Filtered list of tools matching the step's source requirements.
        Tools are ordered by priority (priority 1 first).
    """
    # Get source hints from phase results (set by source-aware planner)
    source_hints = _get_source_hints_for_step(step_id, state)

    # Get source constraints (from manual step or planner)
    source_constraint = state.get_source_constraint(step_id)

    # If no hints or constraints, return all tools
    if not source_hints and not source_constraint:
        logger.debug(
            "SOURCE_ROUTING_NO_HINTS",
            step_id=step_id,
            tool_count=len(available_tools),
        )
        return available_tools

    # Build list of allowed tool names
    allowed_tools: list[tuple[dict[str, Any], int]] = []  # (tool, priority)

    for tool in available_tools:
        tool_name = _get_tool_name(tool)
        tool_source_type = _infer_source_type_from_tool(tool_name)

        # Check against source hints
        hint_priority = _get_priority_from_hints(tool_name, tool_source_type, source_hints)

        # Check against source constraints
        is_allowed_by_constraint = _is_tool_allowed_by_constraint(
            tool_name, tool_source_type, source_constraint
        )

        if hint_priority is not None:
            # Tool is explicitly hinted
            if is_allowed_by_constraint:
                allowed_tools.append((tool, hint_priority))
        elif not source_hints and is_allowed_by_constraint:
            # No hints, but tool is allowed by constraint
            allowed_tools.append((tool, 3))  # Default to optional priority

    # Sort by priority (1 = highest priority)
    allowed_tools.sort(key=lambda x: x[1])
    filtered_tools = [t[0] for t in allowed_tools]

    logger.info(
        "SOURCE_ROUTING_FILTERED",
        step_id=step_id,
        original_count=len(available_tools),
        filtered_count=len(filtered_tools),
        tool_names=[_get_tool_name(t) for t in filtered_tools],
    )

    return filtered_tools


def inject_query_hints(
    tool: dict[str, Any],
    query_hint: str | None,
) -> dict[str, Any]:
    """Inject a query hint into a tool definition.

    Modifies the tool's description to include the query hint,
    helping the LLM understand how to use the tool for this step.

    Args:
        tool: OpenAI-format tool definition.
        query_hint: Query hint to inject (e.g., "Search for revenue metrics").

    Returns:
        Modified tool definition with hint incorporated.
    """
    if not query_hint:
        return tool

    # Deep copy to avoid modifying the original
    modified_tool = deepcopy(tool)

    # Inject hint into description
    if "function" in modified_tool:
        original_desc = modified_tool["function"].get("description", "")
        modified_tool["function"]["description"] = (
            f"{original_desc}\n\n**Query Hint for this step**: {query_hint}"
        )

    return modified_tool


def execute_step_with_source_routing(
    step_id: str,
    available_tools: list[dict[str, Any]],
    state: ResearchState,
) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    """Filter tools and prepare query hints for a research step.

    Combines tool filtering with query hint injection for a complete
    source routing solution.

    Args:
        step_id: ID of the current research step.
        available_tools: List of OpenAI-format tool definitions.
        state: Current research state.

    Returns:
        Tuple of:
        - Filtered list of tools with query hints injected
        - Dict mapping tool names to their query hints (for logging/debugging)
    """
    # Filter tools
    filtered_tools = filter_tools_for_step(step_id, available_tools, state)

    # Get source hints for query hint injection
    source_hints = _get_source_hints_for_step(step_id, state)

    # Build query hints map
    query_hints_map: dict[str, str | None] = {}

    # Inject query hints into tools
    tools_with_hints: list[dict[str, Any]] = []
    for tool in filtered_tools:
        tool_name = _get_tool_name(tool)
        query_hint = _get_query_hint_for_tool(tool_name, source_hints)
        query_hints_map[tool_name] = query_hint

        if query_hint:
            tools_with_hints.append(inject_query_hints(tool, query_hint))
        else:
            tools_with_hints.append(tool)

    logger.debug(
        "SOURCE_ROUTING_PREPARED",
        step_id=step_id,
        tool_count=len(tools_with_hints),
        hints_count=sum(1 for h in query_hints_map.values() if h),
    )

    return tools_with_hints, query_hints_map


def check_source_budget(
    source_type: str,
    state: ResearchState,
) -> bool:
    """Check if there's remaining budget for a source type.

    Args:
        source_type: Type of source (web_search, vector_search, etc.).
        state: Current research state with budget tracking.

    Returns:
        True if budget allows more queries.
    """
    remaining = state.get_source_budget(source_type)
    return remaining > 0


def get_tools_within_budget(
    tools: list[dict[str, Any]],
    state: ResearchState,
) -> list[dict[str, Any]]:
    """Filter tools to only those with remaining budget.

    Args:
        tools: List of tool definitions.
        state: Current research state.

    Returns:
        Tools that still have query budget available.
    """
    budgeted_tools: list[dict[str, Any]] = []

    for tool in tools:
        tool_name = _get_tool_name(tool)
        source_type = _infer_source_type_from_tool(tool_name)

        if check_source_budget(source_type, state):
            budgeted_tools.append(tool)
        else:
            logger.debug(
                "SOURCE_ROUTING_BUDGET_EXCEEDED",
                tool_name=tool_name,
                source_type=source_type,
            )

    return budgeted_tools


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _get_source_hints_for_step(
    step_id: str,
    state: ResearchState,
) -> list[StepSourceHint]:
    """Get source hints for a step from phase results.

    The source-aware planner stores hints in phase_results under
    'source_hints_{step_id}' or in a consolidated 'step_source_hints' dict.

    Args:
        step_id: ID of the step.
        state: Research state.

    Returns:
        List of source hints for the step.
    """
    hints: list[StepSourceHint] = []

    # Try to get hints from phase results
    phase_results = state.get_all_phase_results()

    # Check for step-specific hints
    step_hints_key = f"source_hints_{step_id}"
    if step_hints_key in phase_results:
        raw_hints = phase_results[step_hints_key]
        hints.extend(_parse_raw_hints(raw_hints))

    # Check for consolidated hints dict
    if "step_source_hints" in phase_results:
        step_source_hints = phase_results["step_source_hints"]
        if isinstance(step_source_hints, dict) and step_id in step_source_hints:
            raw_hints = step_source_hints[step_id]
            hints.extend(_parse_raw_hints(raw_hints))

    return hints


def _parse_raw_hints(raw_hints: Any) -> list[StepSourceHint]:
    """Parse raw hints from phase results into StepSourceHint objects.

    Args:
        raw_hints: Raw hints data (list of dicts or StepSourceHint objects).

    Returns:
        List of StepSourceHint objects.
    """
    hints: list[StepSourceHint] = []

    if isinstance(raw_hints, list):
        for hint in raw_hints:
            if isinstance(hint, StepSourceHint):
                hints.append(hint)
            elif isinstance(hint, dict):
                try:
                    hints.append(StepSourceHint(**hint))
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "SOURCE_ROUTING_INVALID_HINT",
                        hint=str(hint)[:100],
                        error=str(e)[:100],
                    )

    return hints


def _get_tool_name(tool: dict[str, Any]) -> str:
    """Extract tool name from OpenAI-format tool definition.

    Args:
        tool: Tool definition dict.

    Returns:
        Tool name string.
    """
    if "function" in tool:
        name = tool["function"].get("name", "unknown")
        return str(name) if name else "unknown"
    name = tool.get("name", "unknown")
    return str(name) if name else "unknown"


def _infer_source_type_from_tool(tool_name: str) -> str:
    """Infer source type from tool name.

    Args:
        tool_name: Name of the tool.

    Returns:
        Source type string.
    """
    # Built-in tool mappings
    tool_to_source: dict[str, str] = {
        "web_search": "web_search",
        "web_crawl": "web_search",  # Part of web search workflow
        "vector_search": "vector_search",
        "genie_query": "genie",
        "query_genie": "genie",  # Alternative naming
        "knowledge_assistant": "knowledge_assistant",
        "file_search": "uploaded_file",
    }

    # Direct mapping
    if tool_name in tool_to_source:
        return tool_to_source[tool_name]

    # Pattern-based inference for dynamic tool names
    if tool_name.startswith("search_") or tool_name.startswith("vs_"):
        return "vector_search"
    if tool_name.startswith("query_genie_") or tool_name.startswith("genie_"):
        return "genie"
    if tool_name.startswith("ask_") or tool_name.endswith("_assistant"):
        return "knowledge_assistant"

    # Default to custom/unknown
    return "custom"


def _get_priority_from_hints(
    tool_name: str,
    tool_source_type: str,
    hints: list[StepSourceHint],
) -> int | None:
    """Get priority for a tool from source hints.

    Args:
        tool_name: Name of the tool.
        tool_source_type: Source type of the tool.
        hints: List of source hints.

    Returns:
        Priority (1-3) if tool is hinted, None otherwise.
    """
    for hint in hints:
        # Match by exact name or type
        if hint.source_name == tool_name or hint.source_type == tool_source_type:
            return hint.priority

        # Also match by source type contained in tool name
        if hint.source_type in tool_name:
            return hint.priority

    return None


def _get_query_hint_for_tool(
    tool_name: str,
    hints: list[StepSourceHint],
) -> str | None:
    """Get query hint for a tool from source hints.

    Args:
        tool_name: Name of the tool.
        hints: List of source hints.

    Returns:
        Query hint string if available, None otherwise.
    """
    tool_source_type = _infer_source_type_from_tool(tool_name)

    for hint in hints:
        # Match by exact name or type
        if hint.source_name == tool_name or hint.source_type == tool_source_type:
            return hint.query_hint

    return None


def _is_tool_allowed_by_constraint(
    tool_name: str,
    tool_source_type: str,
    constraint: SourceConstraint | None,
) -> bool:
    """Check if a tool is allowed by source constraints.

    Args:
        tool_name: Name of the tool.
        tool_source_type: Source type of the tool.
        constraint: Source constraint (or None for no constraint).

    Returns:
        True if tool is allowed.
    """
    if constraint is None:
        return True

    return constraint.is_source_allowed(tool_name, tool_source_type)
