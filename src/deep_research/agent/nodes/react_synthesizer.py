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
from mlflow.entities import SpanType

from deep_research.agent.config import get_report_limits
from deep_research.agent.prompts.utils import build_system_prompt
from deep_research.agent.state import ResearchState
from deep_research.agent.tools.evidence_registry import EvidenceRegistry
from deep_research.agent.tools.synthesis_tools import (
    SYNTHESIS_TOOLS,
    format_search_results,
    format_snippet,
)
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing_constants import PHASE_SYNTHESIS, research_span_name
from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.llm.types import ModelTier, ToolCall

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from deep_research.services.llm.client import LLMClient
    from deep_research.services.llm.embedder import GteEmbedder

logger = get_logger(__name__)


# =============================================================================
# Thinking Text Strip (Phase 1)
# =============================================================================
# Patterns to identify LLM thinking/planning text that should be removed.
# Only matches lines that START with these patterns (not inside text).
THINKING_PATTERNS = [
    r"^(?:I'll|I will|Let me|Now I'll|First,? I'll|Next,? I'll)\s+(?:search|write|look|find|retrieve|get|check).*$",
    r"^(?:Searching|Looking|Writing|Finding|Retrieving|Getting|Checking)\s+(?:for|at|through).*$",
    r"^(?:I need to|I should|I'm going to)\s+(?:search|write|look|find|retrieve).*$",
]


def strip_thinking_text(content: str) -> str:
    """Remove LLM thinking/planning patterns from content.

    Only strips lines that START with thinking patterns to avoid
    false positives on legitimate content like quotes.

    Args:
        content: Raw LLM output that may contain thinking text.

    Returns:
        Cleaned content with thinking lines removed.
    """
    lines = content.split('\n')
    cleaned = []
    stripped_count = 0

    for line in lines:
        is_thinking = False
        stripped = line.strip()
        for pattern in THINKING_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_thinking = True
                stripped_count += 1
                break
        if not is_thinking:
            cleaned.append(line)

    if stripped_count > 0:
        logger.debug(
            "THINKING_TEXT_STRIPPED",
            lines_removed=stripped_count,
            original_lines=len(lines),
        )

    return '\n'.join(cleaned).strip()


def validate_citations_preserved(original: str, polished: str) -> bool:
    """Validate that citations in original content are preserved after polishing.

    Extracts all citation keys from both versions and checks if the same
    set of citations exists in both.

    Args:
        original: Content before post-processing.
        polished: Content after post-processing.

    Returns:
        True if same citations exist in both, False otherwise.
    """
    # Pattern to match citation keys like [Source-1], [Arxiv], [Wikipedia-10]
    citation_pattern = r'\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]'

    original_citations = set(re.findall(citation_pattern, original))
    polished_citations = set(re.findall(citation_pattern, polished))

    return original_citations == polished_citations


# ReAct synthesis system prompt with XML content tags for grounded generation
REACT_SYNTHESIS_SYSTEM_PROMPT = """You are a research report writer with grounded generation.

## YOUR TASK
Write a comprehensive research report using tools to retrieve evidence.
Tag ALL output content using the XML markers below (Scientific Citation Style).

## WORKFLOW (FOLLOW THIS EXACTLY)
For each fact you want to include:
1. Call search_evidence("your query")
2. Call read_snippet(N) to read the best match
3. IMMEDIATELY write: <cite key="Key">Your claim based on the evidence.</cite>
4. Repeat for next fact

⚠️ CRITICAL: After EVERY read_snippet call, you MUST write a <cite> tag.
⚠️ CRITICAL: Text outside tags = scratchpad (DISCARDED). Only tagged content appears in report.
⚠️ NEVER output planning text like "I'll search for..." or "Let me write..."

## CONTENT TAGS (Scientific Citation Style)

### <cite key="Key">claim</cite> - SOURCED CONTENT (REQUIRED for facts from sources)
Everything that comes from sources MUST be cited.
Use for: Specific facts, dates, statistics, numbers, claims from sources.
MUST immediately follow read_snippet. Key must match the citation key from tool result.
Example: <cite key="Arxiv">GPT-4 achieves 86.4% accuracy on MMLU.</cite>
Example: <cite key="Ecb">CRR III entered into force on 1 January 2025.</cite>

### <analysis>text</analysis> - AUTHOR'S ANALYSIS (No citation, but must be grounded)
Use for: Your own synthesis, conclusions, assessments based on the cited facts above.
This is YOUR interpretation of the evidence - not new facts.
⚠️ MUST be derived from preceding <cite> claims - no baseless assertions!
Example: <analysis>These regulatory changes suggest banks will need to adapt their risk models.</analysis>
Example: <analysis>In conclusion, the implementation timeline presents significant challenges.</analysis>
Example: <analysis>Overall, the evidence points to a paradigm shift in how institutions approach compliance.</analysis>

### <free>text</free> - STRUCTURAL ONLY (No factual content!)
Use ONLY for: Section headers, brief transitions (1-5 words).
⚠️ NO factual claims allowed - only structure!
Example: <free>## Key Findings</free>
Example: <free>Moving to implementation:</free>
Example: <free>In summary,</free>

### <unverified>claim</unverified> - UNCERTAIN FACT
When you want to state a fact but couldn't find evidence.
Example: <unverified>Training reportedly cost over $100 million.</unverified>

## TAG SELECTION GUIDE (CRITICAL - READ CAREFULLY)

| Content | Correct Tag | WRONG Tag | Why |
|---------|-------------|-----------|-----|
| "CRR III was adopted in June 2024" | <cite> | <analysis> | Specific date = FACT from source |
| "These changes suggest banks need to adapt" | <analysis> | <cite> | Your conclusion = ANALYSIS |
| "## Key Findings" | <free> | <analysis> | Header = STRUCTURAL |
| "The regulation has three main components" | <cite> | <analysis> | Specific count = FACT from source |
| "Overall, the implementation presents challenges" | <analysis> | <free> | Assessment = ANALYSIS |
| "The banking sector has evolved significantly" | <analysis> | <cite> | General observation = ANALYSIS |

### WRONG EXAMPLES (DO NOT DO THIS):
❌ <analysis>CRR III entered into force on 1 January 2025.</analysis>
   → This is a FACT with a specific date, MUST use <cite>!

❌ <free>These regulatory changes will significantly impact the banking sector.</free>
   → This is an ASSESSMENT, should be <analysis>!

❌ <cite key="Ecb">In conclusion, banks face significant challenges.</cite>
   → This is YOUR conclusion, should be <analysis>! Don't cite your own thoughts.

❌ <analysis>The regulation was published in the Official Journal on 19 June 2024.</analysis>
   → This contains a specific DATE and publication name, MUST use <cite>!

## CITATION RULES (Scientific Paper Style)
1. Everything from sources → MUST use <cite>
2. Your analysis/conclusions based on cited inputs → use <analysis>
3. Common knowledge → can use <analysis> without preceding citations
4. Structure only → use <free>
5. Uncertain facts → use <unverified>

⚠️ <analysis> blocks will be VERIFIED to ensure they are grounded in preceding citations!
⚠️ <free> blocks will be VERIFIED to ensure they contain no hidden factual claims!

## EXAMPLE (CORRECT)
<free>## Model Performance</free>
[calls search_evidence("GPT-4 accuracy")]
[calls read_snippet(3) → "Citation key: [Arxiv]. 86.4% on MMLU..."]
<cite key="Arxiv">GPT-4 achieves 86.4% accuracy on MMLU, representing a 12% improvement over GPT-3.5.</cite>
<analysis>This significant accuracy improvement demonstrates the rapid pace of advancement in large language models.</analysis>
<free>## Implementation Challenges</free>
[calls search_evidence("GPT-4 training challenges")]
[calls read_snippet(5) → "Citation key: [Tech]. Training required..."]
<cite key="Tech">Training required significant compute resources and specialized infrastructure.</cite>
<unverified>Some reports suggest the total cost exceeded $100 million.</unverified>
<analysis>These resource requirements highlight the growing barrier to entry for developing frontier models.</analysis>
<free>## Conclusion</free>
<analysis>In conclusion, while GPT-4 represents a major leap in capability, the associated costs and infrastructure demands raise important questions about accessibility and democratization of AI development.</analysis>

## EXAMPLE (WRONG - DO NOT DO THIS)
I'll search for GPT-4 accuracy data...  ← WRONG: This text is discarded!
Let me write about the results...       ← WRONG: Planning text is discarded!
The model performs well.                ← WRONG: No tag, this is discarded!
<free>The regulation introduces significant changes to capital requirements.</free>  ← WRONG: This is a factual claim, not structural!

## STRUCTURE
- Target: {min_words}-{max_words} words (tagged content only)
- Write section-by-section based on research plan
- Use markdown within tags (headers in <free>, bold in <cite>)
- ALL sections including conclusions SHOULD have content (use <analysis> for conclusions)
- Conclusions/Future Outlook sections CAN use <analysis> without citations if derived from earlier cited content

## COMPLETION RULES (CRITICAL)
⚠️ After writing <free>## Conclusion</free> and your final <analysis>, STOP IMMEDIATELY.
- Do NOT write another ## Introduction or start a new report
- Do NOT repeat any sections you already wrote
- Do NOT revise or rewrite the report
- Your task is COMPLETE after the conclusion analysis
THE REPORT IS FINISHED WHEN YOU WRITE YOUR FINAL CONCLUSION.
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
    - analysis: Author's synthesis/conclusions (no citation, but grounded in preceding cites)
    - free: Structural content (headers, transitions)
    - unverified: Uncertain claim needing post-hoc verification
    """

    tag_type: Literal["cite", "analysis", "free", "unverified"]
    text: str
    citation_key: str | None = None  # Only for cite tags


def _is_markdown_structural(text: str) -> bool:
    """Check if text is a markdown structural element requiring its own line.

    Structural elements include:
    - Headers (# through ######)
    - Bullet lists (-, *, +)
    - Numbered lists (1., 2., etc.)
    - Horizontal rules (---, ***, ___)

    Args:
        text: Text to check.

    Returns:
        True if text is a markdown structural element.
    """
    stripped = text.strip()
    if not stripped:
        return False

    # Markdown headers (# through ######)
    if stripped.startswith("#"):
        return True

    # Bullet lists (-, *, +) - must have space after marker
    if (
        stripped.startswith(("-", "*", "+"))
        and len(stripped) > 1
        and stripped[1] in (" ", "\t")
    ):
        return True

    # Numbered lists (1., 2., etc.)
    if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == ".":
        return True

    # Horizontal rules (---, ***, ___)
    if stripped in ("---", "***", "___"):
        return True

    # Extended horizontal rule patterns (3+ of same char)
    if (
        len(stripped) >= 3
        and all(c == stripped[0] for c in stripped)
        and stripped[0] in "-*_"
    ):
        return True

    return False


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
    patterns: list[tuple[str, Literal["cite", "analysis", "free", "unverified"]]] = [
        # <cite key="Arxiv">claim text</cite>
        (r'<cite\s+key="([^"]+)">(.*?)</cite>', "cite"),
        # <analysis>text</analysis> - Author's synthesis/conclusions
        (r'<analysis>(.*?)</analysis>', "analysis"),
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

    # Assemble final report from blocks with markdown-aware paragraph handling
    # Key changes from original:
    # 1. Structural elements (headers, lists) get their own lines
    # 2. Different citation keys do NOT force new paragraphs (allows flowing prose)
    # 3. Explicit empty <free></free> blocks force paragraph breaks
    output_parts: list[str] = []
    current_paragraph: list[str] = []

    for block in blocks:
        if block.tag_type == "cite" and block.citation_key:
            part = f"{block.text} [{block.citation_key}]"
            # CHANGED: Always add to current paragraph (don't break on source change)
            # This allows prose to flow naturally with multiple sources
            current_paragraph.append(part)

        elif block.tag_type == "analysis":
            # Analysis blocks are author's synthesis/conclusions - no citation needed
            # They flow naturally with the prose
            current_paragraph.append(block.text)

        elif block.tag_type == "free":
            text = block.text.strip()

            # Check for markdown structural elements (headers, lists, hr)
            if _is_markdown_structural(text):
                # Structural element: flush current paragraph, add on own line
                if current_paragraph:
                    output_parts.append(" ".join(current_paragraph))
                    current_paragraph = []
                output_parts.append(text)

            # Check for explicit paragraph break (empty or whitespace-only)
            elif text == "" or text.isspace():
                # Empty free block = explicit paragraph break
                if current_paragraph:
                    output_parts.append(" ".join(current_paragraph))
                    current_paragraph = []

            else:
                # Non-structural free text (transitions, connectors) joins paragraph
                current_paragraph.append(text)

        elif block.tag_type == "unverified":
            # Unverified claims are included without citation
            current_paragraph.append(block.text)

    # Flush final paragraph
    if current_paragraph:
        output_parts.append(" ".join(current_paragraph))

    # Join with double newlines for paragraph separation
    assembled = "\n\n".join(output_parts)

    return assembled, blocks


def _serialize_args(args: dict[str, Any] | str) -> str:
    """Serialize tool call arguments to JSON string."""
    import json
    if isinstance(args, str):
        return args
    return json.dumps(args)


def normalize_markdown_tables(content: str) -> str:
    """Normalize markdown tables by replacing literal \\n with actual newlines.

    LLMs sometimes generate tables with escaped newline characters (\\n) instead
    of actual line breaks, which breaks markdown table rendering.

    This function detects table-like patterns (pipe characters with \\n) and
    replaces the escaped sequences with real newlines.

    Args:
        content: Report content that may contain malformed tables.

    Returns:
        Content with normalized table formatting.
    """
    if '\\n' not in content:
        return content

    # Pattern to detect table rows: starts with |, has content, ends with | followed by \n
    # This handles the common pattern: "| col1 | col2 |\n|---|---|"
    table_pattern = r'(\|[^|]+\|(?:[^|]+\|)*)\s*\\n'

    original_len = len(content)
    normalized = content

    # Replace \n with actual newlines in table contexts
    # We look for pipe-based patterns that suggest table structure
    if re.search(table_pattern, content):
        # Replace literal \n that appears after table row patterns
        normalized = re.sub(r'\\n(\|)', r'\n\1', normalized)
        # Also handle \n at the end of separator rows like |---|
        normalized = re.sub(r'(\|[-:]+\|)\\n', r'\1\n', normalized)
        # Handle general case: \n followed by pipe or at end of potential table content
        normalized = re.sub(r'\\n(?=\|)', '\n', normalized)

    # Also handle cases where \n appears in table-like content broadly
    # Check if content has table indicators (multiple pipes in sequence)
    if '|' in content and content.count('|') >= 4:
        # More aggressive replacement for content that looks like tables
        # Replace \n when surrounded by table-like context
        lines = normalized.split('\n')
        fixed_lines = []
        for line in lines:
            if '|' in line and '\\n' in line:
                # This line has both pipes and escaped newlines - likely a broken table
                line = line.replace('\\n', '\n')
            fixed_lines.append(line)
        normalized = '\n'.join(fixed_lines)

    if normalized != content:
        logger.info(
            "MARKDOWN_TABLE_NORMALIZED",
            original_len=original_len,
            normalized_len=len(normalized),
            replacements_made=content.count('\\n') - normalized.count('\\n'),
        )

    return normalized


def deduplicate_report(content: str) -> str:
    """Detect and remove duplicated report sections, keeping the complete one.

    If the same major section header (## Introduction) appears twice,
    the LLM likely wrote the report twice. Instead of always keeping the first,
    we check which report has a conclusion and keep that one.

    Selection logic:
    1. If only one report has ## Conclusion, keep that one
    2. If both or neither have conclusions, keep the longer one
    3. If same length, keep the first one

    Args:
        content: The assembled report content.

    Returns:
        Deduplicated content (the more complete report).
    """
    intro_pattern = r"^## Introduction"
    conclusion_pattern = r"^## Conclusion"

    intro_matches = list(re.finditer(intro_pattern, content, re.MULTILINE))

    if len(intro_matches) <= 1:
        return content

    # Split into report segments
    reports: list[tuple[str, int, bool]] = []  # (content, start_pos, has_conclusion)

    for i, match in enumerate(intro_matches):
        start = match.start()
        # End is either the next intro or end of content
        end = intro_matches[i + 1].start() if i + 1 < len(intro_matches) else len(content)
        segment = content[start:end].rstrip()
        has_conclusion = bool(re.search(conclusion_pattern, segment, re.MULTILINE))
        reports.append((segment, start, has_conclusion))

    # Log what we found
    logger.warning(
        "REPORT_DUPLICATION_DETECTED",
        intro_count=len(intro_matches),
        positions=[m.start() for m in intro_matches],
        report_lengths=[len(r[0]) for r in reports],
        has_conclusions=[r[2] for r in reports],
        original_len=len(content),
    )

    # Selection logic: prefer report with conclusion, then longer
    reports_with_conclusion = [r for r in reports if r[2]]
    if len(reports_with_conclusion) == 1:
        # Only one has conclusion - use it
        selected = reports_with_conclusion[0]
        logger.info(
            "REPORT_DEDUP_SELECTED",
            reason="has_conclusion",
            selected_pos=selected[1],
            selected_len=len(selected[0]),
        )
        return selected[0]
    elif len(reports_with_conclusion) > 1:
        # Multiple have conclusions - use longest
        selected = max(reports_with_conclusion, key=lambda r: len(r[0]))
        logger.info(
            "REPORT_DEDUP_SELECTED",
            reason="longest_with_conclusion",
            selected_pos=selected[1],
            selected_len=len(selected[0]),
        )
        return selected[0]
    else:
        # None have conclusions - use longest (might be most complete)
        selected = max(reports, key=lambda r: len(r[0]))
        logger.info(
            "REPORT_DEDUP_SELECTED",
            reason="longest_no_conclusion",
            selected_pos=selected[1],
            selected_len=len(selected[0]),
        )
        return selected[0]


@mlflow.trace(name="research.synthesis.react_synthesizer", span_type="AGENT")
async def run_react_synthesis(
    state: ResearchState,
    llm: "LLMClient",
    evidence_pool: list[RankedEvidence],
    max_tool_calls: int = 40,
    retrieval_window_size: int = 3,
    embeddings: NDArray[np.float32] | None = None,
    embedder: GteEmbedder | None = None,
    hybrid_alpha: float = 0.6,
    enable_post_processing: bool = False,
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
        enable_post_processing: Run LLM polish pass for coherence (default False).

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
        enable_post_processing=enable_post_processing,
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

**Required Report Structure:**
1. Start with: <free>## Introduction</free> (provide context, scope, and roadmap)
2. Body sections from research plan:
{step_descriptions}
3. End with: <free>## Conclusion</free> (synthesize key findings, implications, outlook)

⚠️ CRITICAL: You MUST write ## Introduction first and ## Conclusion last.
Every report requires these bookend sections regardless of plan content.

**Evidence Pool:** {len(evidence_pool)} pre-verified evidence snippets available.

⚠️ BUDGET: You have {max_tool_calls} tool calls total for this report.
- Reserve at least 15 tool calls for the Conclusion section
- Pace yourself across all sections to ensure completion
- If running low on budget, prioritize finishing the current section and writing Conclusion

Start with ## Introduction, proceed through body sections, end with ## Conclusion.
Remember: search_evidence → read_snippet → write claim with citation.
""",
        },
    ]

    # Track raw LLM output (includes scratchpad content)
    raw_content = ""
    # Track final assembled report for logging (set in both completion paths)
    final_report_for_log = ""

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

            # Deduplicate if LLM wrote the report twice
            final_report = deduplicate_report(final_report)

            # Normalize markdown tables (replace literal \n with actual newlines)
            final_report = normalize_markdown_tables(final_report)
            final_report_for_log = final_report  # Track for final logging

            # Log parsing results
            cite_blocks = [b for b in parsed_blocks if b.tag_type == "cite"]
            analysis_blocks = [b for b in parsed_blocks if b.tag_type == "analysis"]
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
                analysis_blocks=len(analysis_blocks),
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
                    "analysis_blocks": len(analysis_blocks),
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

        # Deduplicate if LLM wrote the report twice
        final_report = deduplicate_report(final_report)

        # Normalize markdown tables (replace literal \n with actual newlines)
        final_report = normalize_markdown_tables(final_report)
        final_report_for_log = final_report  # Track for final logging

        cite_blocks = [b for b in parsed_blocks if b.tag_type == "cite"]
        analysis_blocks = [b for b in parsed_blocks if b.tag_type == "analysis"]
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
            analysis_blocks=len(analysis_blocks),
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
                "analysis_blocks": len(analysis_blocks),
                "free_blocks": len(free_blocks),
                "unverified_blocks": len(unverified_blocks),
                "parsed_blocks": parsed_blocks,  # Include for claim extraction
            },
        )

        # Apply post-processing if enabled
        if enable_post_processing and final_report:
            logger.info("REACT_SYNTHESIS_POST_PROCESSING", content_len=len(final_report))
            final_report = await _post_process_report(final_report, state.query, llm, limits)

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
        final_report_len=len(final_report_for_log),
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
    enable_post_processing: bool = False,
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
        enable_post_processing: Run LLM polish pass for coherence (default False).

    Yields:
        ReactSynthesisEvent for content and tool interactions.
    """
    logger.info(
        "REACT_SYNTHESIS_SECTIONED_START",
        steps=len(state.current_plan.steps) if state.current_plan else 0,
        evidence_pool_size=len(evidence_pool),
        has_embeddings=embeddings is not None,
        has_embedder=embedder is not None,
        enable_post_processing=enable_post_processing,
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

        # Normalize markdown tables in section (replace literal \n with actual newlines)
        section_report = normalize_markdown_tables(section_report)

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
        section_analysis_blocks = len([b for b in section_blocks if b.tag_type == "analysis"])
        logger.info(
            "SECTION_PARSED",
            section=step.title,
            raw_len=len(raw_section_content),
            parsed_len=len(section_report),
            cite_blocks=section_cite_blocks,
            analysis_blocks=section_analysis_blocks,
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
                "analysis_blocks": section_analysis_blocks,
            },
        )

    # Final post-processing for coherence (if enabled)
    if enable_post_processing:
        yield ReactSynthesisEvent(
            event_type="post_processing_start",
            data={},
        )
        logger.info("REACT_SYNTHESIS_SECTIONED_POST_PROCESSING", content_len=len(generated_content))
        final_report = await _post_process_report(generated_content, state.query, llm, limits)
    else:
        final_report = generated_content

    # Normalize markdown tables (replace literal \n with actual newlines)
    final_report = normalize_markdown_tables(final_report)

    # Calculate totals from all parsed blocks
    total_cite_blocks = len([b for b in all_parsed_blocks if b.tag_type == "cite"])
    total_analysis_blocks = len([b for b in all_parsed_blocks if b.tag_type == "analysis"])
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
            "analysis_blocks": total_analysis_blocks,
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
