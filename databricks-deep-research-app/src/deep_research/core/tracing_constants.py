"""Centralized MLflow tracing constants for consistent span naming.

This module provides standardized span naming patterns across the Deep Research Agent
to ensure traces are descriptive and easy to navigate for debugging.

Naming Pattern:
    {domain}.{stage_or_phase}.{operation}[{index}]

Examples:
    - research.classify.coordinator
    - research.execute.researcher.step_2
    - citation.stage_7.claim[2].fact[0].verify
    - tool.web_search.step_1.query_1
"""

from __future__ import annotations

import json
from typing import Any

# =============================================================================
# Domain Prefixes
# =============================================================================

DOMAIN_RESEARCH = "research"
DOMAIN_CITATION = "citation"
DOMAIN_TOOL = "tool"

# =============================================================================
# Research Phases
# =============================================================================

PHASE_CLASSIFY = "classify"
PHASE_BACKGROUND = "background"
PHASE_PLAN = "plan"
PHASE_EXECUTE = "execute"
PHASE_REFLECT = "reflect"
PHASE_SYNTHESIS = "synthesis"

# =============================================================================
# Citation Stages
# =============================================================================

STAGE_1_EVIDENCE = "stage_1"
STAGE_2_GENERATION = "stage_2"
STAGE_3_CONFIDENCE = "stage_3"
STAGE_4_VERIFICATION = "stage_4"
STAGE_5_CORRECTION = "stage_5"
STAGE_6_NUMERIC = "stage_6"
STAGE_7_ARE = "stage_7"

# =============================================================================
# Citation Pipeline Operations (Stages 1-6)
# =============================================================================

OP_PRESELECT = "preselect"
OP_GENERATE = "generate"
OP_VERIFY = "verify"
OP_VERIFY_NUMERIC = "verify_numeric"
OP_CORRECT = "correct"

# =============================================================================
# Stage 7 Operations (ARE Pattern)
# =============================================================================

OP_RETRIEVE_AND_REVISE = "retrieve_and_revise"
OP_DECOMPOSE = "decompose"
OP_DECOMPOSE_BATCH = "decompose_batch"
OP_PROCESS = "process"
OP_INTERNAL_SEARCH = "internal_search"
OP_EXTERNAL_SEARCH = "external_search"
OP_ENTAILMENT_CHECK = "entailment_check"
OP_RECONSTRUCT = "reconstruct"
OP_SOFTEN = "soften"

# =============================================================================
# Common Attribute Keys
# =============================================================================

# Decision attributes
ATTR_DECISION = "decision"
ATTR_DECISION_VALUE = "decision.value"
ATTR_DECISION_REASONING = "decision.reasoning"
ATTR_DECISION_COMPLEXITY = "decision.complexity"

# Claim attributes
ATTR_CLAIM_INDEX = "claim.index"
ATTR_CLAIM_TEXT = "claim.text"
ATTR_CLAIM_VERDICT = "claim.verdict"

# Fact attributes
ATTR_FACT_INDEX = "fact.index"
ATTR_FACT_TEXT = "fact.text"

# Verification attributes
ATTR_VERIFIED = "verified"
ATTR_EVIDENCE_SOURCE = "evidence_source"
ATTR_ENTAILMENT_SCORE = "entailment.score"
ATTR_ENTAILMENT_REASONING = "entailment.reasoning"
ATTR_ENTAILS = "entailment.entails"

# Search attributes
ATTR_SEARCH_QUERY = "search.query"
ATTR_SEARCH_COUNT = "search.count"
ATTR_SEARCH_RESULTS_COUNT = "search.results_count"
ATTR_SEARCH_ATTEMPT = "search.attempt"
ATTR_SEARCH_TOP_URLS = "search.top_urls"

# Decomposition attributes
ATTR_DECOMPOSITION_FACT_COUNT = "decomposition.fact_count"
ATTR_DECOMPOSITION_REASONING = "decomposition.reasoning"
ATTR_DECOMPOSITION_SKIPPED = "decomposition.skipped"

# Revision attributes
ATTR_REVISION_TYPE = "revision.type"
ATTR_VERIFIED_COUNT = "verified_count"
ATTR_SOFTENED_COUNT = "softened_count"

# Plan attributes
ATTR_PLAN_ITERATION = "plan.iteration"
ATTR_PLAN_STEPS_COUNT = "plan.steps_count"
ATTR_PLAN_THOUGHT = "plan.thought"

# Step attributes
ATTR_STEP_INDEX = "step.index"
ATTR_STEP_TITLE = "step.title"
ATTR_STEP_TYPE = "step.type"

# Batch attributes
ATTR_BATCH_TOTAL = "batch.total"
ATTR_BATCH_FILTERED = "batch.filtered"

# Input/Output attributes
ATTR_INPUT_CLAIMS_COUNT = "input.claims_count"
ATTR_INPUT_EVIDENCE_POOL_SIZE = "input.evidence_pool_size"
ATTR_OUTPUT_FULLY_VERIFIED = "output.fully_verified"
ATTR_OUTPUT_PARTIALLY_SOFTENED = "output.partially_softened"
ATTR_OUTPUT_FULLY_SOFTENED = "output.fully_softened"

# Crawl attributes
ATTR_CRAWL_URLS_COUNT = "crawl.urls_count"
ATTR_CRAWL_SUCCESSFUL = "crawl.successful"
ATTR_CRAWL_FAILED = "crawl.failed"

# Softening attributes
ATTR_SOFTENING_STRATEGY = "softening.strategy"
ATTR_SOFTENING_INPUT = "softening.input"
ATTR_SOFTENING_OUTPUT = "softening.output"


# =============================================================================
# Span Name Builders
# =============================================================================


def research_span_name(
    phase: str,
    agent: str,
    step: int | None = None,
    iteration: int | None = None,
) -> str:
    """Build a research domain span name.

    Args:
        phase: Research phase (classify, plan, execute, reflect, synthesis)
        agent: Agent name (coordinator, planner, researcher, reflector)
        step: Optional step index for multi-step operations
        iteration: Optional iteration number for plan refinement

    Returns:
        Formatted span name like "research.execute.researcher.step_2"

    Examples:
        >>> research_span_name("classify", "coordinator")
        'research.classify.coordinator'
        >>> research_span_name("execute", "researcher", step=2)
        'research.execute.researcher.step_2'
        >>> research_span_name("plan", "planner", iteration=1)
        'research.plan.planner.iteration_1'
    """
    name = f"{DOMAIN_RESEARCH}.{phase}.{agent}"
    if step is not None:
        name = f"{name}.step_{step}"
    elif iteration is not None:
        name = f"{name}.iteration_{iteration}"
    return name


def citation_span_name(stage: str, operation: str, *indices: int) -> str:
    """Build a citation domain span name.

    Args:
        stage: Citation stage (stage_1 through stage_7)
        operation: Operation being performed (decompose, verify, etc.)
        indices: Optional indices for claim/fact identification

    Returns:
        Formatted span name like "citation.stage_7.claim[2].fact[0].verify"

    Examples:
        >>> citation_span_name("stage_7", "retrieve_and_revise")
        'citation.stage_7.retrieve_and_revise'
        >>> citation_span_name("stage_7", "decompose", 2)
        'citation.stage_7.claim[2].decompose'
        >>> citation_span_name("stage_7", "verify", 2, 0)
        'citation.stage_7.claim[2].fact[0].verify'
    """
    name = f"{DOMAIN_CITATION}.{stage}"

    # Add claim and fact indices if provided
    if len(indices) >= 1:
        name = f"{name}.claim[{indices[0]}]"
    if len(indices) >= 2:
        name = f"{name}.fact[{indices[1]}]"

    name = f"{name}.{operation}"
    return name


def tool_span_name(tool: str, context: str | None = None) -> str:
    """Build a tool domain span name.

    Args:
        tool: Tool name (web_search, web_crawl)
        context: Optional context (step_1, background, verification)

    Returns:
        Formatted span name like "tool.web_search.step_1"

    Examples:
        >>> tool_span_name("web_search")
        'tool.web_search'
        >>> tool_span_name("web_search", "step_1")
        'tool.web_search.step_1'
        >>> tool_span_name("web_crawl", "background")
        'tool.web_crawl.background'
    """
    name = f"{DOMAIN_TOOL}.{tool}"
    if context:
        name = f"{name}.{context}"
    return name


# =============================================================================
# Attribute Value Helpers
# =============================================================================


def truncate_for_attr(text: str | None, max_length: int = 200) -> str:
    """Truncate text for use in span attributes.

    Args:
        text: Text to truncate
        max_length: Maximum length (default 200)

    Returns:
        Truncated text with ellipsis if needed
    """
    if text is None:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def list_to_attr(items: list[Any], max_items: int = 5) -> str:
    """Convert a list to a JSON string for span attributes.

    Args:
        items: List of items to convert
        max_items: Maximum number of items to include

    Returns:
        JSON string representation

    Examples:
        >>> list_to_attr(["a", "b", "c"], max_items=2)
        '["a", "b"]'
    """
    truncated = items[:max_items]
    return json.dumps(truncated)


def safe_attr_value(value: Any) -> str | int | float | bool:
    """Convert a value to a safe attribute type.

    MLflow span attributes must be str, int, float, or bool.
    Complex types are converted to JSON strings.

    Args:
        value: Value to convert

    Returns:
        Safe attribute value
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value)[:10])  # Limit list size
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)
