"""Source-specific query rewriting for enterprise data sources.

Transforms naive queries (e.g., "Step title: step description") into
source-optimized queries for Vector Search, Genie, and Knowledge Assistants.

Key design decisions:
- Never blocks research: on any failure, falls back to original query
- Uses FAST model tier for rewriting (~150 tokens, minimal cost)
- Timeout protection per rewrite call
- Dispatches to source-type-specific rewriters

Part of enterprise query optimization feature.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from deep_research.agent.prompts.query_rewriter import (
    GENIE_REWRITE_PROMPT,
    KA_REWRITE_PROMPT,
    QUERY2DOC_PROMPT,
    VS_MULTI_QUERY_PROMPT,
    VS_REWRITE_PROMPT,
)
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.services.llm.client import LLMClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RewrittenQuery(BaseModel):
    """Result of source-specific query rewriting."""

    primary_query: str
    """Source-optimized main query."""

    alternate_queries: list[str] = Field(default_factory=list)
    """Additional queries for multi-query / RRF execution."""

    strategy_used: str = "direct"
    """Audit trail: which strategy produced this result."""


class QueryRewriteConfig(BaseModel):
    """Per-source-type rewriting configuration."""

    enabled: bool = True
    strategy: str = "direct"
    """Strategy: direct | multi_query | query2doc | schema_aware | step_back."""

    max_alternate_queries: int = Field(default=3, ge=1, le=5)
    enable_query2doc: bool = False
    """Generate pseudo-doc, concat as text expansion (works with ALL index types)."""

    model_tier: str = "fast"
    """ModelTier for rewriting LLM calls."""

    timeout_seconds: float = Field(default=10.0, gt=0)
    fallback_on_failure: bool = True
    """If rewrite fails, use original query (never block research)."""

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Structured output models for LLM parsing
# ---------------------------------------------------------------------------


class _VSRewriteOutput(BaseModel):
    query: str


class _VSMultiQueryOutput(BaseModel):
    queries: list[str]


class _Query2DocOutput(BaseModel):
    passage: str


class _GenieRewriteOutput(BaseModel):
    question: str


class _KARewriteOutput(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Per-source rewriters
# ---------------------------------------------------------------------------


def _build_naive_query(step_title: str, step_description: str, original_query: str) -> str:
    """Build the naive fallback query (same as old concatenation)."""
    return f"{step_title}: {step_description or original_query}"


async def _rewrite_for_vector_search(
    llm: LLMClient,
    step_title: str,
    step_description: str,
    original_query: str,
    source_description: str,  # noqa: ARG001
    config: QueryRewriteConfig,
    previous_observations: list[str] | None,  # noqa: ARG001
) -> RewrittenQuery:
    """Rewrite query for Vector Search sources."""
    from deep_research.services.llm.types import ModelTier

    tier = ModelTier(config.model_tier)

    if config.strategy == "multi_query":
        # Generate multiple reformulations
        messages = [
            {
                "role": "user",
                "content": VS_MULTI_QUERY_PROMPT.format(
                    step_title=step_title,
                    step_description=step_description or "",
                    original_query=original_query,
                ),
            }
        ]
        response = await llm.complete(
            messages=messages,
            tier=tier,
            max_tokens=500,
            structured_output=_VSMultiQueryOutput,
        )
        if response.structured:
            queries = response.structured.queries[: config.max_alternate_queries]
        else:
            parsed = _VSMultiQueryOutput.model_validate_json(response.content)
            queries = parsed.queries[: config.max_alternate_queries]

        primary = queries[0] if queries else _build_naive_query(step_title, step_description, original_query)
        alternates = queries[1:] if len(queries) > 1 else []

        # Optionally append query2doc expansion
        if config.enable_query2doc:
            doc_expansion = await _generate_query2doc(llm, step_title, original_query, tier)
            if doc_expansion:
                primary = f"{primary}\n\n{doc_expansion}"

        return RewrittenQuery(
            primary_query=primary,
            alternate_queries=alternates,
            strategy_used="multi_query" + ("_query2doc" if config.enable_query2doc else ""),
        )

    elif config.strategy == "query2doc":
        # Generate pseudo-document and concatenate with original
        doc_expansion = await _generate_query2doc(llm, step_title, original_query, tier)
        if doc_expansion:
            primary = f"{original_query}\n\n{doc_expansion}"
        else:
            primary = _build_naive_query(step_title, step_description, original_query)
        return RewrittenQuery(primary_query=primary, strategy_used="query2doc")

    else:
        # Direct: single natural language sentence
        messages = [
            {
                "role": "user",
                "content": VS_REWRITE_PROMPT.format(
                    step_title=step_title,
                    step_description=step_description or "",
                    original_query=original_query,
                ),
            }
        ]
        response = await llm.complete(
            messages=messages,
            tier=tier,
            max_tokens=300,
            structured_output=_VSRewriteOutput,
        )
        if response.structured:
            query = response.structured.query
        else:
            parsed = _VSRewriteOutput.model_validate_json(response.content)
            query = parsed.query

        return RewrittenQuery(primary_query=query, strategy_used="direct")


async def _generate_query2doc(
    llm: LLMClient,
    step_title: str,
    original_query: str,
    tier: Any,
) -> str:
    """Generate a pseudo-document for Query2Doc expansion."""
    messages = [
        {
            "role": "user",
            "content": QUERY2DOC_PROMPT.format(
                step_title=step_title,
                original_query=original_query,
            ),
        }
    ]
    response = await llm.complete(
        messages=messages,
        tier=tier,
        max_tokens=300,
        structured_output=_Query2DocOutput,
    )
    if response.structured:
        return response.structured.passage
    parsed = _Query2DocOutput.model_validate_json(response.content)
    return parsed.passage


async def _rewrite_for_genie(
    llm: LLMClient,
    step_title: str,
    step_description: str,
    original_query: str,
    source_description: str,
    config: QueryRewriteConfig,
    previous_observations: list[str] | None,  # noqa: ARG001
) -> RewrittenQuery:
    """Rewrite query for Genie (SQL analytics) sources."""
    from deep_research.services.llm.types import ModelTier

    tier = ModelTier(config.model_tier)

    messages = [
        {
            "role": "user",
            "content": GENIE_REWRITE_PROMPT.format(
                step_title=step_title,
                step_description=step_description or "",
                original_query=original_query,
                source_description=source_description or "Enterprise analytics database",
            ),
        }
    ]
    response = await llm.complete(
        messages=messages,
        tier=tier,
        max_tokens=300,
        structured_output=_GenieRewriteOutput,
    )
    if response.structured:
        question = response.structured.question
    else:
        parsed = _GenieRewriteOutput.model_validate_json(response.content)
        question = parsed.question

    return RewrittenQuery(primary_query=question, strategy_used="schema_aware")


async def _rewrite_for_knowledge_assistant(
    llm: LLMClient,
    step_title: str,
    step_description: str,
    original_query: str,
    source_description: str,  # noqa: ARG001
    config: QueryRewriteConfig,
    previous_observations: list[str] | None,
) -> RewrittenQuery:
    """Rewrite query for Knowledge Assistant sources."""
    from deep_research.services.llm.types import ModelTier

    tier = ModelTier(config.model_tier)

    # Format previous observations for context
    obs_text = "(No previous findings)"
    if previous_observations:
        obs_text = "\n".join(f"- {obs[:300]}" for obs in previous_observations[-3:])

    messages = [
        {
            "role": "user",
            "content": KA_REWRITE_PROMPT.format(
                step_title=step_title,
                step_description=step_description or "",
                original_query=original_query,
                previous_observations=obs_text,
            ),
        }
    ]
    response = await llm.complete(
        messages=messages,
        tier=tier,
        max_tokens=300,
        structured_output=_KARewriteOutput,
    )
    if response.structured:
        question = response.structured.question
    else:
        parsed = _KARewriteOutput.model_validate_json(response.content)
        question = parsed.question

    return RewrittenQuery(primary_query=question, strategy_used="step_back")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Type for rewriter functions
_RewriterFn = Callable[
    [Any, str, str, str, str, QueryRewriteConfig, list[str] | None],
    Coroutine[Any, Any, RewrittenQuery],
]

_REWRITERS: dict[str, _RewriterFn] = {
    "vector_search": _rewrite_for_vector_search,
    "genie": _rewrite_for_genie,
    "knowledge_assistant": _rewrite_for_knowledge_assistant,
}


async def rewrite_for_source_type(
    llm: LLMClient,
    source_type: str,
    step_title: str,
    step_description: str,
    original_query: str,
    source_description: str = "",
    config: QueryRewriteConfig | None = None,
    previous_observations: list[str] | None = None,
) -> RewrittenQuery:
    """Dispatch to source-specific rewriter. Returns original on failure if fallback enabled.

    Args:
        llm: LLM client for generating rewritten queries.
        source_type: Source type identifier (vector_search, genie, knowledge_assistant).
        step_title: Current research step title.
        step_description: Current research step description.
        original_query: Original user query.
        source_description: Description of the specific data source.
        config: Query rewrite configuration. Uses defaults if None.
        previous_observations: Recent research observations for context.

    Returns:
        RewrittenQuery with source-optimized query text.
    """
    config = config or QueryRewriteConfig()

    naive_query = _build_naive_query(step_title, step_description, original_query)

    if not config.enabled:
        return RewrittenQuery(primary_query=naive_query, strategy_used="disabled")

    rewriter = _REWRITERS.get(source_type)
    if not rewriter:
        return RewrittenQuery(primary_query=naive_query, strategy_used="unknown_source_type")

    try:
        return await asyncio.wait_for(
            rewriter(
                llm,
                step_title,
                step_description,
                original_query,
                source_description,
                config,
                previous_observations,
            ),
            timeout=config.timeout_seconds,
        )
    except Exception as e:
        logger.warning(
            "QUERY_REWRITE_FAILED",
            source_type=source_type,
            strategy=config.strategy,
            error=str(e)[:200],
            error_type=type(e).__name__,
        )
        if config.fallback_on_failure:
            return RewrittenQuery(
                primary_query=naive_query,
                strategy_used="fallback",
            )
        raise
