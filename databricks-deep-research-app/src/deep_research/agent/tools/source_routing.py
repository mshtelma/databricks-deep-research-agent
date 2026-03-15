"""Source Routing Module for Enterprise Data Sources.

This module provides functions for filtering and routing tools based on
SourceConstraint configurations defined in manual_step.py.

Part of 007-enterprise-data-sources feature (T053).

Key Functions:
- filter_tools_by_constraint: Filter tools based on allowed/excluded sources
- validate_required_sources_consulted: Check if required sources were queried
- prompt_for_required_sources: Generate prompts for unconsulted required sources
- validate_source_constraints: Validate constraints for conflicts/errors
- handle_source_unavailable: Graceful degradation when sources fail
"""

from dataclasses import dataclass, field
from typing import Any

from deep_research.agent.tools.base import ResearchTool
from deep_research.core.logging_utils import get_logger
from deep_research.schemas.manual_step import SourceConstraint, SourceConstraintType

logger = get_logger(__name__)


# Mapping from tool name to source type
# Used to match tools against SourceConstraint.allowed_types
TOOL_SOURCE_TYPE_MAPPING: dict[str, str] = {
    "web_search": "web_search",
    "web_crawl": "web_search",  # Same source type as web_search
    "vector_search": "vector_search",
    "genie_query": "genie",
    "knowledge_assistant_query": "knowledge_assistant",
    "file_search": "uploaded_file",
}


def get_tool_source_type(tool_name: str) -> str:
    """Get the source type for a tool.

    Args:
        tool_name: Name of the tool.

    Returns:
        Source type string (e.g., 'web_search', 'vector_search').
    """
    # Check explicit mapping first
    if tool_name in TOOL_SOURCE_TYPE_MAPPING:
        return TOOL_SOURCE_TYPE_MAPPING[tool_name]

    # Check for partial matches in base name
    for known_tool, source_type in TOOL_SOURCE_TYPE_MAPPING.items():
        if known_tool in tool_name.lower():
            return source_type

    # Default to tool name as source type
    return tool_name


def get_tool_source_name(tool: ResearchTool) -> str:
    """Extract source name from a tool.

    For vector_search and knowledge_assistant tools, the source name
    is typically encoded in the tool name (e.g., 'search_product_docs').

    Args:
        tool: ResearchTool instance.

    Returns:
        Source name string.
    """
    return tool.definition.name


def filter_tools_by_constraint(
    constraint: SourceConstraint | None,
    tools: list[ResearchTool],
) -> list[ResearchTool]:
    """Filter tools based on source constraints.

    Applies constraint rules in this order:
    1. Exclude tools for excluded_sources
    2. Filter by allowed_types if specified
    3. Filter by allowed_sources if constraint is EXCLUSIVE

    Args:
        constraint: SourceConstraint from manual step, or None for no filtering.
        tools: List of ResearchTool instances to filter.

    Returns:
        Filtered list of tools that satisfy the constraint.
    """
    if constraint is None:
        return tools

    filtered: list[ResearchTool] = []

    for tool in tools:
        tool_name = tool.definition.name
        tool_source_type = get_tool_source_type(tool_name)
        tool_source_name = get_tool_source_name(tool)

        # Check using the constraint's built-in method
        if constraint.is_source_allowed(tool_source_name, tool_source_type):
            filtered.append(tool)
        else:
            logger.debug(
                "TOOL_FILTERED_BY_CONSTRAINT",
                tool_name=tool_name,
                source_type=tool_source_type,
                constraint_type=constraint.constraint_type,
            )

    # Log filter results
    if len(filtered) != len(tools):
        logger.info(
            "TOOLS_FILTERED_BY_CONSTRAINT",
            original_count=len(tools),
            filtered_count=len(filtered),
            constraint_type=constraint.constraint_type,
            allowed_types=constraint.allowed_types,
            excluded_sources=constraint.excluded_sources,
        )

    return filtered


def filter_tool_definitions_by_constraint(
    constraint: SourceConstraint | None,
    tool_definitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter OpenAI-format tool definitions based on source constraints.

    This is for use with the legacy RESEARCH_TOOLS format used by react_researcher.

    Args:
        constraint: SourceConstraint from manual step, or None for no filtering.
        tool_definitions: List of OpenAI-format tool definitions.

    Returns:
        Filtered list of tool definitions that satisfy the constraint.
    """
    if constraint is None:
        return tool_definitions

    filtered: list[dict[str, Any]] = []

    for tool_def in tool_definitions:
        # Extract tool name from OpenAI format
        func_def = tool_def.get("function", {})
        tool_name = func_def.get("name", "")

        if not tool_name:
            continue

        tool_source_type = get_tool_source_type(tool_name)

        # Check using the constraint's built-in method
        # For OpenAI format, source_name = tool_name
        if constraint.is_source_allowed(tool_name, tool_source_type):
            filtered.append(tool_def)
        else:
            logger.debug(
                "TOOL_DEF_FILTERED_BY_CONSTRAINT",
                tool_name=tool_name,
                source_type=tool_source_type,
            )

    return filtered


def validate_required_sources_consulted(
    constraint: SourceConstraint | None,
    queried_sources: set[str],
) -> list[str]:
    """Check if all required sources from constraint were queried.

    Args:
        constraint: SourceConstraint from manual step, or None.
        queried_sources: Set of source names that have been queried.

    Returns:
        List of required sources that haven't been queried yet.
        Empty list means all required sources have been consulted.
    """
    if constraint is None:
        return []

    # Use constraint's built-in method
    missing = constraint.get_missing_required(queried_sources)

    if missing:
        logger.info(
            "MISSING_REQUIRED_SOURCES",
            missing_sources=missing,
            queried_sources=list(queried_sources),
            required_sources=constraint.required_sources,
        )

    return missing


def prompt_for_required_sources(
    missing_sources: list[str],
    step_title: str,
    step_objective: str,
) -> str:
    """Generate prompts to ensure required sources are consulted.

    Called when required sources haven't been queried yet.

    Args:
        missing_sources: List of required source names not yet queried.
        step_title: Title of the current research step.
        step_objective: Objective of the current research step.

    Returns:
        Prompt text instructing the LLM to query required sources.
    """
    if not missing_sources:
        return ""

    sources_list = ", ".join(f"'{s}'" for s in missing_sources)

    prompt = f"""## Required Sources Not Yet Consulted

The following data sources are **required** for this research step but have not been queried yet:
{sources_list}

**Current Step:** {step_title}
**Objective:** {step_objective}

You MUST query these required sources before completing this step. Use the appropriate tool
for each source (e.g., vector_search for knowledge bases, genie_query for data rooms).

After querying all required sources, you may continue with additional sources if needed.
"""
    return prompt


def get_prioritized_sources(
    constraint: SourceConstraint | None,
    available_sources: list[str],
) -> tuple[list[str], list[str]]:
    """Get sources ordered by priority based on constraint.

    For REQUIRED constraint: required sources first, then others
    For PREFERRED constraint: allowed sources first, then others
    For EXCLUSIVE constraint: only allowed sources, in order

    Args:
        constraint: SourceConstraint from manual step, or None.
        available_sources: List of all available source names.

    Returns:
        Tuple of (primary_sources, secondary_sources).
    """
    if constraint is None:
        return available_sources, []

    primary: list[str] = []
    secondary: list[str] = []

    if constraint.constraint_type == SourceConstraintType.EXCLUSIVE:
        # Only allowed sources, in the order specified
        if constraint.allowed_sources:
            for source in constraint.allowed_sources:
                if source in available_sources and source not in constraint.excluded_sources:
                    primary.append(source)
        return primary, []

    if constraint.constraint_type == SourceConstraintType.REQUIRED:
        # Required sources first, then others
        for source in constraint.required_sources:
            if source in available_sources and source not in constraint.excluded_sources:
                primary.append(source)

        for source in available_sources:
            if source not in primary and source not in constraint.excluded_sources:
                secondary.append(source)

        return primary, secondary

    # PREFERRED: allowed sources preferred, others available
    if constraint.allowed_sources:
        for source in constraint.allowed_sources:
            if source in available_sources and source not in constraint.excluded_sources:
                primary.append(source)

    for source in available_sources:
        if source not in primary and source not in constraint.excluded_sources:
            secondary.append(source)

    return primary, secondary


def should_force_required_source_query(
    constraint: SourceConstraint | None,
    queried_sources: set[str],
    tool_call_count: int,
    max_tool_calls: int,
) -> bool:
    """Determine if we should force querying required sources.

    Called during ReAct loop to check if we're running out of budget
    without having consulted required sources.

    Args:
        constraint: SourceConstraint from manual step, or None.
        queried_sources: Set of source names already queried.
        tool_call_count: Current number of tool calls made.
        max_tool_calls: Maximum allowed tool calls.

    Returns:
        True if we should force required source queries.
    """
    if constraint is None:
        return False

    missing = constraint.get_missing_required(queried_sources)
    if not missing:
        return False

    # Force if we've used more than 70% of budget and still have required sources
    budget_threshold = 0.7
    budget_used = tool_call_count / max_tool_calls if max_tool_calls > 0 else 0

    if budget_used >= budget_threshold:
        logger.warning(
            "FORCING_REQUIRED_SOURCE_QUERY",
            missing_sources=missing,
            tool_call_count=tool_call_count,
            max_tool_calls=max_tool_calls,
            budget_used_pct=int(budget_used * 100),
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Source Constraint Validation (T105)
# ---------------------------------------------------------------------------


@dataclass
class ConstraintValidationResult:
    """Result of source constraint validation.

    Contains lists of errors (fatal issues) and warnings (non-fatal issues).
    """

    errors: list[str] = field(default_factory=list)
    """Fatal validation errors that prevent execution."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal warnings about potential issues."""

    @property
    def is_valid(self) -> bool:
        """Return True if no fatal errors."""
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


def validate_source_constraints(
    constraint: SourceConstraint | None,
    available_sources: list[str] | None = None,
    enabled_source_types: list[str] | None = None,
) -> ConstraintValidationResult:
    """Validate source constraints for conflicts and errors.

    Checks for:
    1. Conflicts between allowed_sources and excluded_sources
    2. Conflicts between required_sources and excluded_sources
    3. Conflicts between allowed_types and enabled source types
    4. Required sources that don't exist in available sources
    5. Empty allowed_sources with EXCLUSIVE constraint

    Args:
        constraint: SourceConstraint to validate.
        available_sources: List of available source names (for existence checks).
        enabled_source_types: List of enabled source types in the system.

    Returns:
        ConstraintValidationResult with errors and warnings.
    """
    result = ConstraintValidationResult()

    if constraint is None:
        return result

    # 1. Check for conflicts between allowed_sources and excluded_sources
    if constraint.allowed_sources and constraint.excluded_sources:
        overlap = set(constraint.allowed_sources) & set(constraint.excluded_sources)
        if overlap:
            result.add_error(
                f"Source conflict: sources are both allowed and excluded: {sorted(overlap)}"
            )

    # 2. Check for conflicts between required_sources and excluded_sources
    if constraint.required_sources and constraint.excluded_sources:
        overlap = set(constraint.required_sources) & set(constraint.excluded_sources)
        if overlap:
            result.add_error(
                f"Source conflict: required sources are excluded: {sorted(overlap)}"
            )

    # 3. Check required_sources vs allowed_sources (if EXCLUSIVE)
    if (
        constraint.constraint_type == SourceConstraintType.EXCLUSIVE
        and constraint.allowed_sources
        and constraint.required_sources
    ):
        not_in_allowed = set(constraint.required_sources) - set(constraint.allowed_sources)
        if not_in_allowed:
            result.add_error(
                f"Required sources not in allowed list (EXCLUSIVE mode): {sorted(not_in_allowed)}"
            )

    # 4. Check for empty allowed_sources with EXCLUSIVE constraint
    if (
        constraint.constraint_type == SourceConstraintType.EXCLUSIVE
        and constraint.allowed_sources is not None
        and len(constraint.allowed_sources) == 0
    ):
        result.add_error(
            "EXCLUSIVE constraint has empty allowed_sources - no sources can be used"
        )

    # 5. Check allowed_types against enabled source types
    if constraint.allowed_types and enabled_source_types:
        invalid_types = set(constraint.allowed_types) - set(enabled_source_types)
        if invalid_types:
            result.add_warning(
                f"Allowed types not available in system: {sorted(invalid_types)}"
            )

    # 6. Check required_sources against available sources
    if constraint.required_sources and available_sources:
        missing = set(constraint.required_sources) - set(available_sources)
        if missing:
            result.add_warning(
                f"Required sources not found in available sources: {sorted(missing)}"
            )

    # 7. Check min_sources vs max_sources
    if (
        constraint.max_sources is not None
        and constraint.min_sources > constraint.max_sources
    ):
        result.add_error(
            f"Invalid source limits: min_sources ({constraint.min_sources}) "
            f"> max_sources ({constraint.max_sources})"
        )

    # 8. Check if min_sources can be satisfied
    if available_sources and constraint.min_sources > len(available_sources):
        result.add_warning(
            f"min_sources ({constraint.min_sources}) exceeds available sources "
            f"({len(available_sources)})"
        )

    # Log validation results
    if result.errors or result.warnings:
        logger.info(
            "SOURCE_CONSTRAINT_VALIDATION",
            is_valid=result.is_valid,
            error_count=len(result.errors),
            warning_count=len(result.warnings),
            constraint_type=constraint.constraint_type,
        )

    return result


# ---------------------------------------------------------------------------
# Graceful Degradation (T106)
# ---------------------------------------------------------------------------


@dataclass
class SourceAvailability:
    """Tracks source availability and fallback status."""

    source_name: str
    source_type: str
    is_available: bool = True
    error_message: str | None = None
    fallback_used: bool = False
    fallback_source: str | None = None


@dataclass
class DegradationResult:
    """Result of graceful degradation handling."""

    can_continue: bool
    """Whether research can continue with remaining sources."""

    available_sources: list[str]
    """Sources that are still available."""

    unavailable_sources: list[SourceAvailability]
    """Sources that failed with their status."""

    warnings: list[str]
    """Warnings to surface to the user or logs."""


def handle_source_unavailable(
    failed_source: str,
    failed_source_type: str,
    error_message: str,
    all_sources: list[str],
    constraint: SourceConstraint | None = None,
    already_unavailable: list[str] | None = None,
) -> DegradationResult:
    """Handle graceful degradation when a source becomes unavailable.

    Determines whether research can continue and what sources remain available.
    For required sources, degradation may not be possible.

    Args:
        failed_source: Name of the source that failed.
        failed_source_type: Type of the failed source.
        error_message: Error message from the failure.
        all_sources: List of all configured source names.
        constraint: Optional SourceConstraint that may require the failed source.
        already_unavailable: Sources already known to be unavailable.

    Returns:
        DegradationResult with continuation status and available sources.
    """
    already_unavailable = already_unavailable or []
    warnings: list[str] = []

    # Track the failed source
    failed_availability = SourceAvailability(
        source_name=failed_source,
        source_type=failed_source_type,
        is_available=False,
        error_message=error_message[:500] if error_message else None,
    )

    # Build list of unavailable sources
    unavailable_set = set(already_unavailable) | {failed_source}
    unavailable_sources = [failed_availability]

    # Calculate remaining available sources
    available_sources = [s for s in all_sources if s not in unavailable_set]

    # Check if the failed source is required
    is_required = False
    if constraint and failed_source in constraint.required_sources:
        is_required = True
        warnings.append(
            f"Required source '{failed_source}' is unavailable: {error_message}"
        )

    # Check if we have enough sources to continue
    min_required = constraint.min_sources if constraint else 1
    can_continue = len(available_sources) >= min_required

    if not can_continue:
        if is_required:
            warnings.append(
                f"Cannot continue: required source '{failed_source}' unavailable "
                f"and no fallback available"
            )
        else:
            warnings.append(
                f"Cannot continue: only {len(available_sources)} sources available, "
                f"minimum required is {min_required}"
            )

    # Log the degradation event
    logger.warning(
        "SOURCE_UNAVAILABLE_DEGRADATION",
        failed_source=failed_source,
        failed_source_type=failed_source_type,
        is_required=is_required,
        can_continue=can_continue,
        available_count=len(available_sources),
        error_preview=error_message[:100] if error_message else None,
    )

    return DegradationResult(
        can_continue=can_continue,
        available_sources=available_sources,
        unavailable_sources=unavailable_sources,
        warnings=warnings,
    )


def suggest_fallback_sources(
    failed_source_type: str,
    available_tools: list[ResearchTool],
    constraint: SourceConstraint | None = None,
) -> list[str]:
    """Suggest fallback sources when a source type fails.

    Maps failed source types to potential alternatives:
    - vector_search -> web_search (for general knowledge)
    - genie -> web_search (for data questions)
    - knowledge_assistant -> vector_search, web_search

    Args:
        failed_source_type: Type of source that failed.
        available_tools: List of available tools.
        constraint: Optional constraint to check fallback eligibility.

    Returns:
        List of suggested fallback source names.
    """
    # Define fallback mappings (source_type -> alternative source_types)
    fallback_map: dict[str, list[str]] = {
        "vector_search": ["web_search"],
        "genie": ["web_search"],
        "knowledge_assistant": ["vector_search", "web_search"],
    }

    alternatives = fallback_map.get(failed_source_type, ["web_search"])
    suggestions: list[str] = []

    for tool in available_tools:
        tool_type = get_tool_source_type(tool.definition.name)

        if tool_type in alternatives:
            # Check if this fallback is allowed by constraint
            if constraint and not constraint.is_source_allowed(
                tool.definition.name, tool_type
            ):
                continue

            suggestions.append(tool.definition.name)

    if suggestions:
        logger.info(
            "FALLBACK_SOURCES_SUGGESTED",
            failed_source_type=failed_source_type,
            suggested_sources=suggestions,
        )

    return suggestions
