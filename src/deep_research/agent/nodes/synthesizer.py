"""Synthesizer agent - generates final research report."""

from collections.abc import AsyncGenerator
from typing import Any

import mlflow
from mlflow.entities import SpanType

from deep_research.agent.config import get_report_limits, get_synthesizer_config
from deep_research.agent.prompts.synthesizer import (
    STREAMING_SYNTHESIZER_SYSTEM_PROMPT,
    STRUCTURED_SYNTHESIZER_SYSTEM_PROMPT,
    STRUCTURED_SYNTHESIZER_USER_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_PROMPT,
)
from deep_research.agent.prompts.utils import build_system_prompt
from deep_research.agent.state import ResearchState
from deep_research.core.exceptions import StructuredSynthesisError
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing_constants import PHASE_SYNTHESIS, research_span_name
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

logger = get_logger(__name__)


async def run_synthesizer(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Synthesizer agent to create final report.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Returns:
        Updated state with final report.
    """
    span_name = research_span_name(PHASE_SYNTHESIS, "synthesizer")

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        logger.info(
            "SYNTHESIZER_START",
            observations=len(state.all_observations),
            sources=len(state.sources),
            query=truncate(state.query, 60),
        )

        # Format observations
        observations_str = ""
        if state.all_observations:
            observations_str = "\n\n---\n\n".join(
                f"**Observation {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
            )
        else:
            observations_str = "(No research observations available)"

        # Format sources
        sources_list = ""
        for i, source in enumerate(state.sources[:20]):  # Limit to 20 sources
            title = source.title or "Untitled"
            sources_list += f"[{i + 1}] {title}\n    URL: {source.url}\n"
            if source.snippet:
                sources_list += f"    Snippet: {source.snippet[:200]}...\n"

        # Calculate research stats
        steps_executed = sum(
            1 for s in (state.current_plan.steps if state.current_plan else [])
            if s.status.value in ("completed", "skipped")
        )

        # Get effective research depth from state
        depth_label = state.resolve_depth()

        # Get word count and token limits from centralized research_types config
        limits = get_report_limits(depth_label)
        min_words = limits.min_words
        max_words = limits.max_words
        max_tokens = limits.max_tokens

        # Build system prompt with user's custom instructions if available
        system_prompt = build_system_prompt(
            SYNTHESIZER_SYSTEM_PROMPT,
            state.system_instructions,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": SYNTHESIZER_USER_PROMPT.format(
                    query=state.query,
                    research_depth=depth_label,
                    plan_iterations=state.plan_iterations,
                    steps_executed=steps_executed,
                    sources_count=len(state.sources),
                    all_observations=observations_str,
                    sources_list=sources_list or "(No sources collected)",
                    min_words=min_words,
                    max_words=max_words,
                ),
            },
        ]

        try:
            response = await llm.complete(
                messages=messages,
                tier=ModelTier.COMPLEX,
                max_tokens=max_tokens,
            )

            state.complete(response.content)
            logger.info(
                "SYNTHESIZER_COMPLETE",
                report_len=len(response.content),
                report_preview=truncate(response.content, 150),
            )

            # Add span attributes
            span.set_attributes({
                "observations_count": len(state.all_observations),
                "sources_count": len(state.sources),
                "report_length": len(response.content),
            })

        except Exception as e:
            logger.error(
                "SYNTHESIZER_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            # Fallback: use collected observations
            fallback_report = f"## Research Summary\n\n{observations_str}"
            state.complete(fallback_report)

        return state


async def run_structured_synthesizer(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Synthesizer agent to create structured JSON output.

    Uses LLM's structured_output parameter for guaranteed schema compliance.
    Does NOT support streaming - returns complete structured response.

    Args:
        state: Current research state with output_schema set.
        llm: LLM client for completions.

    Returns:
        Updated state with final_report and final_report_structured populated.
    """
    span_name = research_span_name(PHASE_SYNTHESIS, "structured_synthesizer")

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        logger.info(
            "STRUCTURED_SYNTHESIZER_START",
            observations=len(state.all_observations),
            sources=len(state.sources),
            output_schema=state.output_schema.__name__ if state.output_schema else None,
            query=truncate(state.query, 60),
        )

        if not state.output_schema:
            logger.error("STRUCTURED_SYNTHESIZER_ERROR: No output_schema provided")
            return await run_synthesizer(state, llm)

        # Format observations with source references
        observations_str = ""
        if state.all_observations:
            observations_str = "\n\n---\n\n".join(
                f"**Observation {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
            )
        else:
            observations_str = "(No research observations available)"

        # Format sources with indices for citation
        sources_list = ""
        for i, source in enumerate(state.sources[:20]):
            title = source.title or "Untitled"
            sources_list += f"[{i + 1}] {title}\n    URL: {source.url}\n"
            if source.snippet:
                sources_list += f"    Snippet: {source.snippet[:200]}...\n"

        # Combine observations and sources for context
        context = f"""## Research Observations
{observations_str}

## Available Sources (reference by number in source_refs)
{sources_list or "(No sources collected)"}"""

        # Use custom prompts if provided, otherwise use framework defaults
        system_prompt = state.structured_system_prompt or STRUCTURED_SYNTHESIZER_SYSTEM_PROMPT
        user_prompt_template = state.structured_user_prompt or STRUCTURED_SYNTHESIZER_USER_PROMPT

        # Build with any additional system_instructions
        system_prompt = build_system_prompt(system_prompt, state.system_instructions)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt_template.format(
                    query=state.query,
                    context=context,
                ),
            },
        ]

        try:
            # Use structured output - the key difference from run_synthesizer!
            response = await llm.complete(
                messages=messages,
                tier=ModelTier.COMPLEX,
                max_tokens=8000,
                structured_output=state.output_schema,
            )

            # Get structured result
            if response.structured:
                state.final_report_structured = response.structured
                # Also store JSON string for compatibility
                state.final_report = response.structured.model_dump_json(indent=2)
                logger.info(
                    "STRUCTURED_SYNTHESIZER_COMPLETE",
                    report_len=len(state.final_report),
                    schema=state.output_schema.__name__,
                )
            else:
                # Fallback: try to parse response as JSON
                try:
                    state.final_report_structured = state.output_schema.model_validate_json(
                        response.content
                    )
                    state.final_report = response.content
                    logger.info(
                        "STRUCTURED_SYNTHESIZER_COMPLETE_FALLBACK",
                        report_len=len(state.final_report),
                    )
                except Exception as parse_error:
                    logger.error(
                        "STRUCTURED_SYNTHESIZER_PARSE_ERROR",
                        error=str(parse_error)[:200],
                    )
                    # Last resort: return raw content as markdown
                    state.final_report = response.content

            state.complete(state.final_report)

            # Add span attributes
            span.set_attributes({
                "observations_count": len(state.all_observations),
                "sources_count": len(state.sources),
                "report_length": len(state.final_report),
                "structured_output_success": state.final_report_structured is not None,
            })

            # CRITICAL: Inject actual sources into structured output
            # The LLM only uses source indices in source_refs - it doesn't populate the sources array
            if state.final_report_structured and hasattr(state.final_report_structured, "sources"):
                state.final_report_structured.sources = [
                    {
                        "id": str(i + 1),  # 1-based to match source_refs indices
                        "url": s.url,
                        "title": s.title or "Untitled",
                        "snippet": s.snippet,
                        "type": "web_page",
                        "confidence": s.relevance_score or 1.0,
                    }
                    for i, s in enumerate(state.sources[:20])  # Match 20-source limit in prompt
                ]
                # Update final_report to include injected sources
                state.final_report = state.final_report_structured.model_dump_json(indent=2)
                logger.info(
                    "SOURCES_INJECTED",
                    count=len(state.final_report_structured.sources),
                )

        except Exception as e:
            logger.error(
                "STRUCTURED_SYNTHESIZER_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            # Signal to orchestrator - let it choose streaming fallback
            raise StructuredSynthesisError(str(e), state) from e

        return state


async def stream_synthesis(state: ResearchState, llm: LLMClient) -> AsyncGenerator[str, None]:
    """Stream the synthesis output for real-time display.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Yields:
        Content chunks as they are generated.
    """
    logger.info(
        "SYNTHESIZER_STREAM_START",
        observations=len(state.all_observations),
        sources=len(state.sources),
    )

    # Format observations
    observations_str = ""
    if state.all_observations:
        observations_str = "\n\n---\n\n".join(
            f"**Observation {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
        )

    # Format sources
    sources_list = ""
    for source in state.sources[:15]:
        title = source.title or "Untitled"
        sources_list += f"- [{title}]({source.url})\n"

    # Get effective research depth from state
    depth_label = state.resolve_depth()

    # Get word count and token limits from centralized research_types config
    limits = get_report_limits(depth_label)
    min_words = limits.min_words
    max_words = limits.max_words
    max_tokens = limits.max_tokens

    # Build system prompt with user's custom instructions if available
    system_prompt = build_system_prompt(
        STREAMING_SYNTHESIZER_SYSTEM_PROMPT.format(
            min_words=min_words, max_words=max_words
        ),
        state.system_instructions,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Create a research report in {min_words}-{max_words} words.

## Query
{state.query}

## Research Findings
{observations_str}

## Available Sources
{sources_list}

Be concise. Cite inline as [Title](url).""",
        },
    ]

    full_content = ""
    try:
        async for chunk in llm.stream(
            messages=messages,
            tier=ModelTier.COMPLEX,
            max_tokens=max_tokens,
        ):
            full_content += chunk
            yield chunk

        # Update state with full content
        state.complete(full_content)

    except Exception as e:
        logger.error(
            "SYNTHESIZER_STREAM_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
            content_len=len(full_content),
        )
        error_msg = f"\n\n*Error during synthesis: {e}*"
        yield error_msg
        state.complete(full_content + error_msg)


async def post_verify_structured_output(
    state: ResearchState,
    llm: LLMClient,
) -> ResearchState:
    """Post-verify claims in structured output using stages 4-6.

    Extracts verifiable claims from the structured output, runs them through
    the verification pipeline, and applies corrections to source_refs.

    Args:
        state: Research state with final_report_structured populated.
        llm: LLM client for verification calls.

    Returns:
        Updated state with verified and corrected structured output.
    """
    from deep_research.services.citation.claim_extractor import StructuredClaimExtractor
    from deep_research.services.citation.evidence_selector import RankedEvidence
    from deep_research.services.citation.post_verifier import PostVerifier, VerifiedClaim

    span_name = research_span_name(PHASE_SYNTHESIS, "post_verification")

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        if not state.final_report_structured:
            logger.warning("POST_VERIFY_SKIP", reason="no_structured_output")
            return state

        schema_name = state.output_schema.__name__ if state.output_schema else "unknown"
        logger.info(
            "POST_VERIFY_START",
            schema=schema_name,
            sources_count=len(state.sources),
        )

        # Step 1: Build evidence pool from sources and observations
        evidence_pool = _build_evidence_pool(state)

        # Step 2: Extract claims from structured output
        extractor = StructuredClaimExtractor()
        claims = extractor.extract(state.final_report_structured)

        if not claims:
            logger.info("POST_VERIFY_SKIP", reason="no_claims_extracted")
            return state

        # Step 3: Run verification (convert SourceInfo to PostVerifier's SourceInfo format)
        from deep_research.services.citation.post_verifier import SourceInfo as PostSourceInfo

        sources_for_verifier = [
            PostSourceInfo(url=s.url, title=s.title, snippet=s.snippet)
            for s in state.sources
        ]

        verifier = PostVerifier(
            llm=llm,
            sources=sources_for_verifier,
            evidence_pool=evidence_pool,
        )
        result = await verifier.verify_claims(claims)

        # Step 4: Apply corrections if any
        if result.corrections_applied > 0 and state.final_report_structured:
            state.final_report_structured = _apply_corrections(
                output=state.final_report_structured,
                verified_claims=result.verified_claims,
            )
            state.final_report = state.final_report_structured.model_dump_json(indent=2)

        # Add span attributes
        span.set_attributes({
            "claims_extracted": len(claims),
            "claims_verified": len(result.verified_claims),
            "support_rate": result.support_rate,
            "corrections_applied": result.corrections_applied,
        })

        logger.info(
            "POST_VERIFY_COMPLETE",
            verified=len(result.verified_claims),
            supported=result.supported_count,
            partial=result.partial_count,
            unsupported=result.unsupported_count,
            corrections=result.corrections_applied,
            support_rate=f"{result.support_rate:.1%}",
        )

        return state


def _build_evidence_pool(state: ResearchState) -> list:
    """Build evidence pool from state sources and observations.

    Converts SourceInfo objects to RankedEvidence format required by verifiers.
    Uses observations as evidence snippets when available.
    """
    from deep_research.services.citation.evidence_selector import RankedEvidence

    evidence_pool: list[RankedEvidence] = []

    # If state already has evidence_pool populated (from prior citation synthesis)
    # convert EvidenceInfo to RankedEvidence
    if state.evidence_pool:
        for ev in state.evidence_pool:
            evidence_pool.append(
                RankedEvidence(
                    source_id=None,  # EvidenceInfo doesn't have source_id
                    source_url=ev.source_url,
                    source_title=None,  # Not stored in EvidenceInfo
                    quote_text=ev.quote_text,
                    start_offset=ev.start_offset,
                    end_offset=ev.end_offset,
                    section_heading=ev.section_heading or "",
                    relevance_score=ev.relevance_score or 0.5,
                    has_numeric_content=ev.has_numeric_content,
                )
            )
        return evidence_pool

    # Otherwise, build from sources + observations
    for i, source in enumerate(state.sources[:30]):  # Limit for performance
        # Use snippet as evidence if available
        quote = source.snippet or ""
        if not quote and state.all_observations:
            # Try to find observation mentioning this source
            for obs in state.all_observations:
                if source.url and source.url in obs:
                    quote = obs[:500]  # First 500 chars
                    break

        if not quote:
            continue

        evidence_pool.append(
            RankedEvidence(
                source_id=None,
                source_url=source.url or "",
                source_title=source.title or "",
                quote_text=quote,
                start_offset=0,
                end_offset=len(quote),
                section_heading="",
                relevance_score=0.5,  # Default score
                has_numeric_content=bool(any(c.isdigit() for c in quote)),
            )
        )

    return evidence_pool


def _apply_corrections(
    output: Any,
    verified_claims: list,
) -> Any:
    """Apply verification corrections back to structured output.

    Only applies source_refs corrections (not text changes) to preserve
    the original content while fixing citation references.
    """
    import re

    from deep_research.services.citation.post_verifier import VerifiedClaim

    data = output.model_dump()
    corrections_applied = 0

    for vc in verified_claims:
        if not isinstance(vc, VerifiedClaim) or not vc.corrected_source_refs:
            continue

        # Parse field path to find source_refs location
        field_path = vc.original.field_path

        # Determine source_refs path based on what exists in the data
        refs_path = _get_source_refs_path(field_path, data)
        if refs_path:
            try:
                _set_nested_value(data, refs_path, vc.corrected_source_refs)
                corrections_applied += 1
            except (KeyError, IndexError) as e:
                logger.warning(
                    "CORRECTION_FAILED",
                    field_path=field_path,
                    error=str(e),
                )

    if corrections_applied > 0:
        logger.info("CORRECTIONS_APPLIED", count=corrections_applied)

    return output.__class__.model_validate(data)


def _get_source_refs_path(field_path: str, data: dict) -> str | None:
    """Determine source_refs path for a given text field path.

    Auto-discovers by checking what source_refs fields exist at that location.
    No hardcoded schema knowledge.
    """
    import re

    # Extract the base path (parent object)
    if "." in field_path:
        parts = field_path.rsplit(".", 1)
        base_path, field_name = parts
    else:
        base_path = ""
        field_name = field_path

    # Get the parent object
    parent = _get_nested_value(data, base_path) if base_path else data
    if not isinstance(parent, dict):
        return None

    # Try pattern 1: field_source_refs
    specific_refs = f"{field_name}_source_refs"
    if specific_refs in parent:
        return f"{base_path}.{specific_refs}" if base_path else specific_refs

    # Try pattern 2: generic source_refs in same object
    if "source_refs" in parent:
        return f"{base_path}.source_refs" if base_path else "source_refs"

    return None


def _get_nested_value(data: dict, path: str) -> Any:
    """Get value from nested dict using dot notation with array support."""
    import re

    if not path:
        return data

    current = data

    for key in re.split(r"\.(?![^\[]*\])", path):
        match = re.match(r"(\w+)\[(\d+)\]", key)
        if match:
            array_key, index = match.groups()
            if not isinstance(current, dict) or array_key not in current:
                return None
            current = current[array_key]
            if not isinstance(current, list) or int(index) >= len(current):
                return None
            current = current[int(index)]
        else:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

    return current


def _set_nested_value(data: dict, path: str, value: Any) -> None:
    """Set value in nested dict using dot notation with array support.

    Args:
        data: Root dictionary to modify.
        path: Dot-separated path with optional array indices (e.g., "a.b[0].c").
        value: Value to set.
    """
    import re

    keys = re.split(r"\.(?![^\[]*\])", path)  # Split on dots not inside brackets
    current = data

    for i, key in enumerate(keys[:-1]):
        # Handle array notation
        match = re.match(r"(\w+)\[(\d+)\]", key)
        if match:
            array_key, index = match.groups()
            current = current[array_key][int(index)]
        else:
            current = current[key]

    # Set final value
    final_key = keys[-1]
    match = re.match(r"(\w+)\[(\d+)\]", final_key)
    if match:
        array_key, index = match.groups()
        current[array_key][int(index)] = value
    else:
        current[final_key] = value
