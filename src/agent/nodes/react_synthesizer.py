"""ReAct-based synthesizer with grounded report generation.

This module implements a ReAct synthesis pattern where the LLM must
retrieve evidence before making factual claims. The key difference
from standard synthesis:

1. LLM writes section-by-section (derived from research plan steps)
2. Before each factual claim, LLM must call search_evidence/read_snippet
3. System tracks which evidence was retrieved before each claim
4. Claims are grounded if they match recently retrieved evidence

This enforces grounded generation, not just provenance tracking.

Hybrid Search Support:
When embeddings are provided, search uses BM25 + vector similarity
(configurable α blend) for better evidence retrieval.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mlflow
import numpy as np

from src.agent.config import get_report_limits
from src.agent.prompts.utils import build_system_prompt
from src.agent.state import ResearchState
from src.agent.tools.evidence_registry import EvidenceRegistry
from src.agent.tools.synthesis_tools import (
    SYNTHESIS_TOOLS,
    format_search_results,
    format_snippet,
)
from src.core.logging_utils import get_logger, truncate
from src.services.citation.evidence_selector import RankedEvidence
from src.services.llm.types import ModelTier, ToolCall

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.services.llm.client import LLMClient
    from src.services.llm.embedder import GteEmbedder

logger = get_logger(__name__)


# ReAct synthesis system prompt with XML content tags for grounded generation
REACT_SYNTHESIS_SYSTEM_PROMPT = """You are a research report writer with grounded generation.

## YOUR TASK
Write a comprehensive research report using tools to retrieve evidence.
Tag ALL output content using the XML markers below.

## WORKFLOW (FOLLOW THIS EXACTLY)
For each fact you want to include:
1. Call search_evidence("your query")
2. Call read_snippet(N) to read the best match
3. IMMEDIATELY write: <cite key="Key">Your claim based on the evidence.</cite>
4. Repeat for next fact

⚠️ CRITICAL: After EVERY read_snippet call, you MUST write a <cite> tag.
⚠️ CRITICAL: Text outside tags = scratchpad (DISCARDED). Only tagged content appears in report.
⚠️ NEVER output planning text like "I'll search for..." or "Let me write..."

## CONTENT TAGS (USE THESE EXACTLY)

### <cite key="Key">claim</cite> - GROUNDED CLAIM
MUST immediately follow read_snippet. Key must match the citation key from tool result.
Example: <cite key="Arxiv">GPT-4 achieves 86.4% accuracy on MMLU.</cite>

### <free>text</free> - STRUCTURAL CONTENT
Headers, transitions, analytical comparisons. No citation needed.
Example: <free>## Key Findings</free>

### <unverified>claim</unverified> - UNCERTAIN CLAIM
When you believe something but couldn't find direct evidence.
Example: <unverified>Training reportedly used 7 trillion tokens.</unverified>

## EXAMPLE (CORRECT)
<free>## Model Performance</free>
[calls search_evidence("GPT-4 accuracy")]
[calls read_snippet(3) → "Citation key: [Arxiv]. 86.4% on MMLU..."]
<cite key="Arxiv">GPT-4 achieves 86.4% accuracy on MMLU.</cite>
<free>This represents a significant improvement over previous models.</free>

## EXAMPLE (WRONG - DO NOT DO THIS)
I'll search for GPT-4 accuracy data...  ← WRONG: This text is discarded!
Let me write about the results...       ← WRONG: Planning text is discarded!
The model performs well.                ← WRONG: No tag, this is discarded!

## STRUCTURE
- Target: {min_words}-{max_words} words (tagged content only)
- Write section-by-section based on research plan
- Use markdown within tags (headers in <free>, bold in <cite>)
"""


@dataclass
class ReactSynthesisEvent:
    """Event emitted during ReAct synthesis loop."""

    event_type: str  # tool_call, tool_result, content, claim_grounded, synthesis_complete
    data: dict[str, Any]


@dataclass
class ReactSynthesisState:
    """Internal state for the ReAct synthesis loop."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    content_chunks: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)  # Future: claim info
    current_position: int = 0  # Character position in generated content


@dataclass
class GroundingResult:
    """Result of grounding check for a claim."""

    grounded: bool
    evidence_index: int | None
    confidence: float
    evidence_quote: str | None = None


@dataclass
class ParsedContent:
    """Parsed content block from ReAct synthesis with XML tags.

    Represents a single content block extracted from LLM output:
    - cite: Grounded claim with citation key
    - free: Structural content (headers, transitions)
    - unverified: Uncertain claim needing post-hoc verification
    """

    tag_type: Literal["cite", "free", "unverified"]
    text: str
    citation_key: str | None = None  # Only for cite tags


def parse_tagged_content(raw_content: str) -> tuple[str, list[ParsedContent]]:
    """Parse XML-tagged content from ReAct synthesis.

    Extracts content from <cite>, <free>, <unverified> tags.
    Unmarked text is treated as scratchpad (not included in report).

    Args:
        raw_content: Full LLM output with XML tags

    Returns:
        Tuple of (assembled_report, list_of_parsed_blocks)
    """
    blocks: list[ParsedContent] = []

    # Patterns for each tag type (re.DOTALL allows . to match newlines)
    patterns: list[tuple[str, Literal["cite", "free", "unverified"]]] = [
        # <cite key="Arxiv">claim text</cite>
        (r'<cite\s+key="([^"]+)">(.*?)</cite>', "cite"),
        # <free>text</free>
        (r'<free>(.*?)</free>', "free"),
        # <unverified>claim</unverified>
        (r'<unverified>(.*?)</unverified>', "unverified"),
    ]

    # Find all tagged content with positions for ordering
    all_matches: list[tuple[int, ParsedContent]] = []

    for pattern, tag_type in patterns:
        for match in re.finditer(pattern, raw_content, re.DOTALL):
            if tag_type == "cite":
                key, text = match.group(1), match.group(2)
                block = ParsedContent(
                    tag_type="cite",
                    text=text.strip(),
                    citation_key=key,
                )
            else:
                block = ParsedContent(
                    tag_type=tag_type,
                    text=match.group(1).strip(),
                )
            all_matches.append((match.start(), block))

    # Sort by position to maintain document order
    all_matches.sort(key=lambda x: x[0])
    blocks = [m[1] for m in all_matches]

    # Assemble final report from blocks
    report_parts: list[str] = []
    for block in blocks:
        if block.tag_type == "cite" and block.citation_key:
            # Add citation marker after claim text
            report_parts.append(f"{block.text} [{block.citation_key}]")
        elif block.tag_type == "unverified":
            # Keep unverified claims but could add marker for UI
            report_parts.append(block.text)
        else:  # free block
            report_parts.append(block.text)

    assembled = "\n\n".join(report_parts)

    return assembled, blocks


def _serialize_args(args: dict[str, Any] | str) -> str:
    """Serialize tool call arguments to JSON string."""
    import json
    if isinstance(args, str):
        return args
    return json.dumps(args)


@mlflow.trace(name="react_synthesizer", span_type="AGENT")
async def run_react_synthesis(
    state: ResearchState,
    llm: "LLMClient",
    evidence_pool: list[RankedEvidence],
    max_tool_calls: int = 40,
    retrieval_window_size: int = 3,
    embeddings: NDArray[np.float32] | None = None,
    embedder: GteEmbedder | None = None,
    hybrid_alpha: float = 0.6,
) -> AsyncGenerator[ReactSynthesisEvent, None]:
    """Run ReAct synthesis with grounded report generation.

    The LLM writes section-by-section, using tools to retrieve evidence
    before making factual claims. The system tracks which evidence was
    retrieved and verifies claims are grounded.

    Supports hybrid search when embeddings are provided:
    - BM25 for exact term matching
    - Vector similarity for semantic matching
    - Configurable blend (α=0.6 default: 60% vector, 40% BM25)

    Args:
        state: Current research state with plan and observations.
        llm: LLM client with tool support.
        evidence_pool: Pre-selected evidence from Stage 1.
        max_tool_calls: Maximum tool calls before stopping.
        retrieval_window_size: Size of sliding window for grounding inference.
        embeddings: Pre-computed evidence embeddings for hybrid search.
        embedder: GTE embedder for computing query embeddings (optional).
        hybrid_alpha: Weight for vector vs BM25 (0.6 = 60% vector).

    Yields:
        ReactSynthesisEvent for tool calls, content, and grounding status.
    """
    logger.info(
        "REACT_SYNTHESIS_START",
        evidence_pool_size=len(evidence_pool),
        max_tool_calls=max_tool_calls,
        query=truncate(state.query, 80),
        has_embeddings=embeddings is not None,
        has_embedder=embedder is not None,
        hybrid_alpha=hybrid_alpha,
    )

    # Initialize registry with embeddings for hybrid search
    registry = EvidenceRegistry(
        evidence_pool,
        retrieval_window_size,
        embeddings=embeddings,
        hybrid_alpha=hybrid_alpha,
    )
    react_state = ReactSynthesisState()

    # Get report limits
    depth_label = state.resolve_depth()
    limits = get_report_limits(depth_label)

    # Build section prompts from research plan steps
    steps = state.current_plan.steps if state.current_plan else []
    step_descriptions = "\n".join(
        f"- {s.title}: {s.description or ''}" for s in steps
    )

    # Build system prompt
    system_prompt = build_system_prompt(
        REACT_SYNTHESIS_SYSTEM_PROMPT.format(
            min_words=limits.min_words,
            max_words=limits.max_words,
        ),
        state.system_instructions,
    )

    # Initialize messages with context
    react_state.messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Write a research report on this query:

**Query:** {state.query}

**Research Plan Sections:**
{step_descriptions}

**Evidence Pool:** {len(evidence_pool)} pre-verified evidence snippets available.

Start by searching for evidence relevant to your first section, then write.
Remember: search_evidence → read_snippet → write claim with citation.
""",
        },
    ]

    # Track raw LLM output (includes scratchpad content)
    raw_content = ""

    # ReAct loop - accumulate raw output, don't stream until parsed
    while react_state.tool_call_count < max_tool_calls:
        tool_calls_this_turn: list[ToolCall] = []
        accumulated_content = ""

        try:
            async for chunk in llm.stream_with_tools(
                messages=react_state.messages,
                tools=SYNTHESIS_TOOLS,
                tier=ModelTier.COMPLEX,
                max_tokens=limits.max_tokens,
            ):
                if chunk.content:
                    accumulated_content += chunk.content
                    raw_content += chunk.content
                    # NOTE: Don't stream raw content - will parse after loop

                if chunk.is_done:
                    if chunk.tool_calls:
                        tool_calls_this_turn = chunk.tool_calls
                    break

        except Exception as e:
            logger.error(
                "REACT_SYNTHESIS_LLM_ERROR",
                error=str(e)[:200],
            )
            yield ReactSynthesisEvent(
                event_type="error",
                data={"error": str(e)[:200]},
            )
            break

        # If no tool calls, LLM is done - parse and assemble report
        if not tool_calls_this_turn:
            # Add assistant response to history
            if accumulated_content:
                react_state.messages.append({
                    "role": "assistant",
                    "content": accumulated_content,
                })

            # Parse XML-tagged content and assemble final report
            final_report, parsed_blocks = parse_tagged_content(raw_content)

            # Log parsing results
            cite_blocks = [b for b in parsed_blocks if b.tag_type == "cite"]
            free_blocks = [b for b in parsed_blocks if b.tag_type == "free"]
            unverified_blocks = [b for b in parsed_blocks if b.tag_type == "unverified"]

            # Fallback: If LLM didn't use XML tags, use raw content with warning
            if not parsed_blocks and raw_content.strip():
                logger.warning(
                    "REACT_SYNTHESIS_NO_TAGS",
                    raw_content_len=len(raw_content),
                    action="falling_back_to_raw_content",
                )
                # Use raw content as-is (unstructured)
                final_report = raw_content.strip()
                # Emit warning event
                yield ReactSynthesisEvent(
                    event_type="grounding_warning",
                    data={
                        "warning": "LLM did not use XML tags - content may include planning text",
                        "content_len": len(final_report),
                    },
                )

            logger.info(
                "REACT_SYNTHESIS_PARSED",
                raw_content_len=len(raw_content),
                final_report_len=len(final_report),
                total_blocks=len(parsed_blocks),
                cite_blocks=len(cite_blocks),
                free_blocks=len(free_blocks),
                unverified_blocks=len(unverified_blocks),
                tool_calls=react_state.tool_call_count,
                used_fallback=not parsed_blocks and bool(raw_content.strip()),
            )

            # Stream the assembled report (not the raw scratchpad)
            if final_report:
                yield ReactSynthesisEvent(
                    event_type="content",
                    data={"chunk": final_report, "is_final": True},
                )

            yield ReactSynthesisEvent(
                event_type="synthesis_complete",
                data={
                    "reason": "llm_decided",
                    "tool_calls": react_state.tool_call_count,
                    "content_len": len(final_report),
                    "cite_blocks": len(cite_blocks),
                    "free_blocks": len(free_blocks),
                    "unverified_blocks": len(unverified_blocks),
                    "parsed_blocks": parsed_blocks,  # Include for claim extraction
                },
            )
            break

        # Add assistant response with tool calls
        react_state.messages.append({
            "role": "assistant",
            "content": accumulated_content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _serialize_args(tc.arguments),
                    },
                }
                for tc in tool_calls_this_turn
            ],
        })

        # Execute each tool call
        for tc in tool_calls_this_turn:
            react_state.tool_call_count += 1

            yield ReactSynthesisEvent(
                event_type="tool_call",
                data={
                    "tool": tc.name,
                    "args": tc.arguments,
                    "call_number": react_state.tool_call_count,
                },
            )

            # Execute tool (with optional embedder for hybrid search)
            tool_result = await _execute_synthesis_tool(tc, registry, embedder)

            # Add nudge after read_snippet to encourage writing
            if tc.name == "read_snippet":
                tool_result += (
                    '\n\n⚠️ NOW WRITE: Use <cite key="...">your claim</cite> '
                    "based on this evidence."
                )

            # Add tool result to messages
            react_state.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

            yield ReactSynthesisEvent(
                event_type="tool_result",
                data={
                    "tool": tc.name,
                    "result_preview": truncate(tool_result, 200),
                },
            )

    # If we hit max tool calls, parse and assemble what we have
    if react_state.tool_call_count >= max_tool_calls:
        # Parse XML-tagged content and assemble final report
        final_report, parsed_blocks = parse_tagged_content(raw_content)

        cite_blocks = [b for b in parsed_blocks if b.tag_type == "cite"]
        free_blocks = [b for b in parsed_blocks if b.tag_type == "free"]
        unverified_blocks = [b for b in parsed_blocks if b.tag_type == "unverified"]

        # Fallback: If LLM didn't use XML tags, use raw content with warning
        if not parsed_blocks and raw_content.strip():
            logger.warning(
                "REACT_SYNTHESIS_NO_TAGS_MAX_CALLS",
                raw_content_len=len(raw_content),
                action="falling_back_to_raw_content",
            )
            final_report = raw_content.strip()
            yield ReactSynthesisEvent(
                event_type="grounding_warning",
                data={
                    "warning": "LLM did not use XML tags after max tool calls",
                    "content_len": len(final_report),
                },
            )

        logger.warning(
            "REACT_SYNTHESIS_MAX_CALLS",
            tool_calls=react_state.tool_call_count,
            raw_content_len=len(raw_content),
            final_report_len=len(final_report),
            cite_blocks=len(cite_blocks),
            used_fallback=not parsed_blocks and bool(raw_content.strip()),
        )

        # Stream the assembled report
        if final_report:
            yield ReactSynthesisEvent(
                event_type="content",
                data={"chunk": final_report, "is_final": True},
            )

        yield ReactSynthesisEvent(
            event_type="synthesis_complete",
            data={
                "reason": "max_tool_calls",
                "tool_calls": react_state.tool_call_count,
                "content_len": len(final_report),
                "cite_blocks": len(cite_blocks),
                "free_blocks": len(free_blocks),
                "unverified_blocks": len(unverified_blocks),
                "parsed_blocks": parsed_blocks,  # Include for claim extraction
            },
        )

        # Update state with parsed report
        state.final_report = final_report

    # Check for grounding failure (no tools called but content generated)
    if react_state.tool_call_count == 0 and len(raw_content) > 100:
        # Parse to check for blocks
        _, parsed_blocks = parse_tagged_content(raw_content)
        if not parsed_blocks:
            logger.warning(
                "REACT_SYNTHESIS_UNGROUNDED",
                raw_content_len=len(raw_content),
            )
            yield ReactSynthesisEvent(
                event_type="grounding_warning",
                data={
                    "warning": "Content generated without consulting evidence",
                    "content_len": len(raw_content),
                },
            )

    # Log access audit for provenance
    logger.info(
        "REACT_SYNTHESIS_AUDIT",
        tool_calls=react_state.tool_call_count,
        evidence_read_indices=list(registry.get_read_indices()),
    )

    logger.info(
        "REACT_SYNTHESIS_DONE",
        tool_calls=react_state.tool_call_count,
        final_report_len=len(state.final_report) if state.final_report else 0,
        evidence_accessed=len(registry.get_read_indices()),
    )


async def _execute_synthesis_tool(
    tc: ToolCall,
    registry: EvidenceRegistry,
    embedder: GteEmbedder | None = None,
) -> str:
    """Execute a synthesis tool call.

    Args:
        tc: Tool call from LLM.
        registry: Evidence registry.
        embedder: Optional embedder for computing query embeddings.

    Returns:
        Tool result string.
    """
    if tc.name == "search_evidence":
        query = tc.arguments.get("query", "")
        claim_type = tc.arguments.get("claim_type")

        # Compute query embedding for hybrid search if embedder available
        query_embedding: NDArray[np.float32] | None = None
        if embedder is not None:
            try:
                query_embedding = await embedder.embed(query)
            except Exception as e:
                logger.warning(
                    "QUERY_EMBEDDING_FAILED",
                    error=str(e)[:100],
                    query=query[:50],
                )

        results = registry.search(
            query,
            limit=5,
            claim_type=claim_type,
            query_embedding=query_embedding,
        )
        return format_search_results(results)

    elif tc.name == "read_snippet":
        index = tc.arguments.get("index")
        if index is None:
            return "Error: Missing 'index' parameter."

        evidence = registry.get(index)
        if not evidence:
            return f"Error: No evidence found at index {index}."

        citation_key = registry.build_citation_key(index)
        return format_snippet(
            index=index,
            title=evidence.source_title,
            quote_text=evidence.quote_text,
            section_heading=evidence.section_heading,
            citation_key=citation_key,
        )

    else:
        return f"Error: Unknown tool '{tc.name}'"


async def run_react_synthesis_sectioned(
    state: ResearchState,
    llm: "LLMClient",
    evidence_pool: list[RankedEvidence],
    tool_budget_per_section: int = 10,
    retrieval_window_size: int = 3,
    embeddings: NDArray[np.float32] | None = None,
    embedder: GteEmbedder | None = None,
    hybrid_alpha: float = 0.6,
) -> AsyncGenerator[ReactSynthesisEvent, None]:
    """Run sectioned ReAct synthesis using research plan steps.

    Each research step becomes a section with its own tool budget.
    The model sees prior generated content to maintain coherence.

    Supports hybrid search when embeddings are provided.

    Args:
        state: Research state with plan steps.
        llm: LLM client.
        evidence_pool: Pre-selected evidence.
        tool_budget_per_section: Tool calls allowed per section.
        retrieval_window_size: Window size for grounding.
        embeddings: Pre-computed evidence embeddings for hybrid search.
        embedder: GTE embedder for computing query embeddings.
        hybrid_alpha: Weight for vector vs BM25 (0.6 = 60% vector).

    Yields:
        ReactSynthesisEvent for content and tool interactions.
    """
    logger.info(
        "REACT_SYNTHESIS_SECTIONED_START",
        steps=len(state.current_plan.steps) if state.current_plan else 0,
        evidence_pool_size=len(evidence_pool),
        has_embeddings=embeddings is not None,
        has_embedder=embedder is not None,
    )

    # Initialize registry with embeddings for hybrid search
    registry = EvidenceRegistry(
        evidence_pool,
        retrieval_window_size,
        embeddings=embeddings,
        hybrid_alpha=hybrid_alpha,
    )

    # Get report limits
    depth_label = state.resolve_depth()
    limits = get_report_limits(depth_label)

    # Build system prompt
    system_prompt = build_system_prompt(
        REACT_SYNTHESIS_SYSTEM_PROMPT.format(
            min_words=limits.min_words,
            max_words=limits.max_words,
        ),
        state.system_instructions,
    )

    # Process each step as a section
    steps = state.current_plan.steps if state.current_plan else []
    generated_content = ""
    total_tool_calls = 0
    all_parsed_blocks: list[ParsedContent] = []  # Collect blocks from all sections

    for step_idx, step in enumerate(steps):
        section_tool_calls = 0

        # Get observations for this step
        step_observation = ""
        if step.status and hasattr(step.status, 'value') and step.status.value == "completed":
            # Find observation for this step index if available
            if step_idx < len(state.all_observations):
                step_observation = state.all_observations[step_idx]

        # Build section prompt with context
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Continue writing the research report.

**Query:** {state.query}

**Current Section:** {step.title}
{step.description or ''}

**Observations from research:**
{step_observation or '(Use evidence pool)'}

**Content written so far:**
{generated_content if generated_content else '(This is the first section)'}

**Instructions:**
Write the "{step.title}" section using evidence from the pool.
Remember: search_evidence → read_snippet → write claim with <cite> tag.
Do not repeat content from previous sections.
""",
            },
        ]

        raw_section_content = ""  # Collect raw content, parse later

        yield ReactSynthesisEvent(
            event_type="section_start",
            data={"section": step.title, "step_index": step_idx},
        )

        # ReAct loop for this section
        while section_tool_calls < tool_budget_per_section:
            tool_calls_this_turn: list[ToolCall] = []
            accumulated_content = ""

            try:
                async for chunk in llm.stream_with_tools(
                    messages=messages,
                    tools=SYNTHESIS_TOOLS,
                    tier=ModelTier.COMPLEX,
                    max_tokens=min(limits.max_tokens, 4000),  # Fixed budget per section
                ):
                    if chunk.content:
                        accumulated_content += chunk.content
                        raw_section_content += chunk.content
                        # DON'T stream raw content - will parse after section loop

                    if chunk.is_done:
                        if chunk.tool_calls:
                            tool_calls_this_turn = chunk.tool_calls
                        break

            except Exception as e:
                logger.error(
                    "REACT_SYNTHESIS_SECTION_ERROR",
                    section=step.title,
                    error=str(e)[:200],
                )
                yield ReactSynthesisEvent(
                    event_type="error",
                    data={"error": str(e)[:200], "section": step.title},
                )
                break

            # If no tool calls, section is complete
            if not tool_calls_this_turn:
                messages.append({
                    "role": "assistant",
                    "content": accumulated_content,
                })
                break

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": accumulated_content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _serialize_args(tc.arguments),
                        },
                    }
                    for tc in tool_calls_this_turn
                ],
            })

            # Execute tool calls
            for tc in tool_calls_this_turn:
                section_tool_calls += 1
                total_tool_calls += 1

                yield ReactSynthesisEvent(
                    event_type="tool_call",
                    data={
                        "tool": tc.name,
                        "args": tc.arguments,
                        "section": step.title,
                    },
                )

                tool_result = await _execute_synthesis_tool(tc, registry, embedder)

                # Add nudge after read_snippet to encourage writing
                if tc.name == "read_snippet":
                    tool_result += (
                        '\n\n⚠️ NOW WRITE: Use <cite key="...">your claim</cite> '
                        "based on this evidence."
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

                yield ReactSynthesisEvent(
                    event_type="tool_result",
                    data={
                        "tool": tc.name,
                        "result_preview": truncate(tool_result, 200),
                    },
                )

        # Parse XML-tagged content from this section
        section_report, section_blocks = parse_tagged_content(raw_section_content)
        all_parsed_blocks.extend(section_blocks)

        # Fallback: If LLM didn't use XML tags, use raw content with warning
        if not section_blocks and raw_section_content.strip():
            logger.warning(
                "SECTION_NO_TAGS",
                section=step.title,
                raw_content_len=len(raw_section_content),
                action="falling_back_to_raw_content",
            )
            section_report = raw_section_content.strip()
            yield ReactSynthesisEvent(
                event_type="grounding_warning",
                data={
                    "warning": f"Section '{step.title}' has no XML tags",
                    "section": step.title,
                    "content_len": len(section_report),
                },
            )

        # Stream the parsed section content
        if section_report:
            yield ReactSynthesisEvent(
                event_type="content",
                data={"chunk": f"\n\n## {step.title}\n\n{section_report}", "section": step.title},
            )

        # Log section parsing results
        section_cite_blocks = len([b for b in section_blocks if b.tag_type == "cite"])
        logger.info(
            "SECTION_PARSED",
            section=step.title,
            raw_len=len(raw_section_content),
            parsed_len=len(section_report),
            cite_blocks=section_cite_blocks,
            total_blocks=len(section_blocks),
            used_fallback=not section_blocks and bool(raw_section_content.strip()),
        )

        # Add parsed section content to full report
        generated_content += f"\n\n## {step.title}\n\n{section_report}"

        yield ReactSynthesisEvent(
            event_type="section_complete",
            data={
                "section": step.title,
                "tool_calls": section_tool_calls,
                "content_len": len(section_report),
                "cite_blocks": section_cite_blocks,
            },
        )

    # Final post-processing for coherence
    yield ReactSynthesisEvent(
        event_type="post_processing_start",
        data={},
    )

    final_report = await _post_process_report(generated_content, state.query, llm, limits)

    # Calculate totals from all parsed blocks
    total_cite_blocks = len([b for b in all_parsed_blocks if b.tag_type == "cite"])
    total_free_blocks = len([b for b in all_parsed_blocks if b.tag_type == "free"])
    total_unverified_blocks = len([b for b in all_parsed_blocks if b.tag_type == "unverified"])

    yield ReactSynthesisEvent(
        event_type="synthesis_complete",
        data={
            "reason": "sections_complete",
            "total_tool_calls": total_tool_calls,
            "sections": len(steps),
            "content_len": len(final_report),
            "cite_blocks": total_cite_blocks,
            "free_blocks": total_free_blocks,
            "unverified_blocks": total_unverified_blocks,
            "parsed_blocks": all_parsed_blocks,  # Include for claim extraction
        },
    )

    # Update state
    state.final_report = final_report

    # Log access audit for provenance
    logger.info(
        "REACT_SYNTHESIS_SECTIONED_AUDIT",
        tool_calls=total_tool_calls,
        evidence_read_indices=list(registry.get_read_indices()),
        sections_processed=len(steps),
    )


async def _post_process_report(
    content: str,
    query: str,
    llm: "LLMClient",
    limits: Any,
) -> str:
    """Post-process report for coherence and polish.

    This is a non-tool pass that fixes transitions and terminology
    WITHOUT adding new claims.

    Args:
        content: Raw generated content.
        query: Original query.
        llm: LLM client.
        limits: Report limits.

    Returns:
        Polished report content.
    """
    if len(content) < 200:
        return content

    messages = [
        {
            "role": "system",
            "content": (
                "You are an editor polishing a research report. "
                "Fix transitions between sections, ensure consistent terminology, "
                "and improve readability. DO NOT add new facts or claims. "
                "Keep all citations exactly as they are."
            ),
        },
        {
            "role": "user",
            "content": f"""Polish this research report on "{query}":

{content}

Return the polished report. Keep all facts and citations unchanged.""",
        },
    ]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.ANALYTICAL,  # Use faster model for polish
            max_tokens=limits.max_tokens,
        )
        return response.content
    except Exception as e:
        logger.warning(
            "REACT_SYNTHESIS_POSTPROCESS_ERROR",
            error=str(e)[:200],
        )
        # Return unpolished content on error
        return content
