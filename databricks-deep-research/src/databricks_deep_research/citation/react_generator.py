"""Stage 2 (REACT mode): Tool-based evidence retrieval for report generation.

Instead of dumping all evidence into the prompt (which overwhelms the context
window at 100+ spans), the LLM calls ``search_evidence`` + ``read_snippet``
tools to find and read evidence on demand, then writes ``<cite key="K">``
tags.  Keeps the prompt at ~5-10K tokens regardless of pool size.

Parallel to ``InterleavedGenerator`` in claim_generator.py which handles
STRICT/NATURAL modes.  This module handles REACT mode exclusively.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from databricks_deep_research.citation.citation_keys import build_citation_key_map
from databricks_deep_research.citation.synthesis_tools import (
    SYNTHESIS_TOOLS,
    EvidenceSearchIndex,
    SynthesisToolExecutor,
    build_assistant_message,
    build_evidence_source_index,
)
from databricks_deep_research.citation.types import (
    ClaimRole,
    InterleavedClaim,
    RankedEvidence,
)
from databricks_deep_research.citation.utils import has_numeric_content as _has_numeric_content
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_REACT_GENERATION_PROMPT = """You are a research report writer with grounded generation.

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

### <analysis>text</analysis> - AUTHOR'S ANALYSIS (No citation, but must be grounded)
Use for: Your own synthesis, conclusions, assessments based on the cited facts above.
⚠️ MUST be derived from preceding <cite> claims - no baseless assertions!
Example: <analysis>These results suggest a significant improvement over prior approaches.</analysis>

### <free>text</free> - STRUCTURAL ONLY (No factual content!)
Use ONLY for: Section headers, brief transitions (1-5 words), code blocks with markdown fences.
⚠️ NO factual claims allowed - only structure!
Example: <free>## Key Findings</free>

### <unverified>claim</unverified> - UNCERTAIN FACT
When you want to state a fact but couldn't find evidence.
Example: <unverified>Training reportedly cost over $100 million.</unverified>

## TAG SELECTION GUIDE (CRITICAL)

| Content | Correct Tag | WRONG Tag | Why |
|---------|-------------|-----------|-----|
| "CRR III was adopted in June 2024" | <cite> | <analysis> | Specific date = FACT from source |
| "These changes suggest banks need to adapt" | <analysis> | <cite> | Your conclusion = ANALYSIS |
| "## Key Findings" | <free> | <analysis> | Header = STRUCTURAL |
| "The regulation has three main components" | <cite> | <analysis> | Specific count = FACT from source |

### WRONG EXAMPLES (DO NOT DO THIS):
❌ <analysis>CRR III entered into force on 1 January 2025.</analysis>
   → This is a FACT with a specific date, MUST use <cite>!
❌ <free>These regulatory changes will significantly impact the banking sector.</free>
   → This is an ASSESSMENT, should be <analysis>!
❌ <cite key="Ecb">In conclusion, banks face significant challenges.</cite>
   → This is YOUR conclusion, should be <analysis>! Don't cite your own thoughts.

## STRUCTURE
- Target: {min_words}-{max_words} words (tagged content only)
- Write section-by-section based on research plan
- Use markdown within tags (headers in <free>, bold in <cite>)
- ALL sections including conclusions SHOULD have content

## COMPLETION RULES (CRITICAL)
⚠️ After writing <free>## Conclusion</free> and your final <analysis>, STOP IMMEDIATELY.
- Do NOT write another ## Introduction or start a new report
- Do NOT repeat any sections you already wrote
- Do NOT revise or rewrite the report
THE REPORT IS FINISHED WHEN YOU WRITE YOUR FINAL CONCLUSION.
"""


# ---------------------------------------------------------------------------
# XML tag parser
# ---------------------------------------------------------------------------

_REACT_TAG_PATTERN = re.compile(
    r'<cite\s+key="([^"]+)">(.*?)</cite>'
    r"|<analysis>(.*?)</analysis>"
    r"|<free>(.*?)</free>"
    r"|<unverified>(.*?)</unverified>",
    re.DOTALL,
)


def _parse_react_content(
    raw_content: str,
    evidence_pool: list[RankedEvidence],
    key_map: dict[int, str],
) -> tuple[str, list[InterleavedClaim]]:
    """Parse XML-tagged content into assembled report + claims.

    Returns (assembled_report, claims) where:
    - assembled_report: clean markdown with [Key] citations
    - claims: InterleavedClaim objects compatible with stages 3-7
    """
    reverse_key_map: dict[str, int] = {key: idx for idx, key in key_map.items()}

    claims: list[InterleavedClaim] = []
    output_parts: list[str] = []
    cursor = 0

    for match in _REACT_TAG_PATTERN.finditer(raw_content):
        cite_key, cite_text = match.group(1), match.group(2)
        analysis_text = match.group(3)
        free_text = match.group(4)
        unverified_text = match.group(5)

        if cite_key is not None and cite_text is not None:
            text = cite_text.strip()
            if not text:
                continue
            assembled = f"{text} [{cite_key}]"
            position_start = cursor
            output_parts.append(assembled)
            cursor += len(assembled) + 2  # +2 for \n\n join

            evidence_index = reverse_key_map.get(cite_key)
            evidence = (
                evidence_pool[evidence_index]
                if evidence_index is not None and 0 <= evidence_index < len(evidence_pool)
                else None
            )

            claims.append(InterleavedClaim(
                claim_text=text,
                claim_type="numeric" if _has_numeric_content(text) else "general",
                position_start=position_start,
                position_end=position_start + len(assembled),
                evidence=evidence,
                evidence_index=evidence_index,
                evidences=[evidence] if evidence else [],
                evidence_indices=[evidence_index] if evidence_index is not None else [],
                confidence_score=evidence.relevance_score if evidence else None,
                claim_role=ClaimRole.FACT.value,
                citation_key=cite_key,
                citation_keys=[cite_key],
            ))

        elif analysis_text is not None:
            text = analysis_text.strip()
            if not text:
                continue
            position_start = cursor
            output_parts.append(text)
            cursor += len(text) + 2

            claims.append(InterleavedClaim(
                claim_text=text,
                claim_type="general",
                position_start=position_start,
                position_end=position_start + len(text),
                evidence=None,
                evidence_index=None,
                claim_role=ClaimRole.ANALYSIS.value,
            ))

        elif free_text is not None:
            text = free_text.strip()
            if not text:
                output_parts.append("")
                cursor += 2
                continue
            position_start = cursor
            output_parts.append(text)
            cursor += len(text) + 2

            claims.append(InterleavedClaim(
                claim_text=text,
                claim_type="general",
                position_start=position_start,
                position_end=position_start + len(text),
                evidence=None,
                evidence_index=None,
                claim_role=ClaimRole.FREE.value,
                from_free_block=True,
            ))

        elif unverified_text is not None:
            text = unverified_text.strip()
            if not text:
                continue
            position_start = cursor
            output_parts.append(text)
            cursor += len(text) + 2

            claims.append(InterleavedClaim(
                claim_text=text,
                claim_type="numeric" if _has_numeric_content(text) else "general",
                position_start=position_start,
                position_end=position_start + len(text),
                evidence=None,
                evidence_index=None,
                confidence_score=0.0,
                claim_role=ClaimRole.FACT.value,
            ))

    assembled = "\n\n".join(part for part in output_parts)
    return assembled, claims


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _post_process_react_content(raw: str) -> str:
    """Deduplicate if LLM wrote the report twice."""
    if not raw.strip():
        return raw

    intro_positions = [m.start() for m in re.finditer(r"<free>\s*##?\s*Introduction", raw)]
    if len(intro_positions) > 1:
        for pos in reversed(intro_positions):
            candidate = raw[pos:]
            if re.search(r"<free>\s*##?\s*Conclusion", candidate):
                return candidate
        return raw[intro_positions[-1]:]

    return raw


# ---------------------------------------------------------------------------
# ReactGenerator
# ---------------------------------------------------------------------------

class ReactGenerator:
    """Generate reports using tool-based evidence retrieval (REACT mode).

    Parallel to ``InterleavedGenerator`` in claim_generator.py which
    handles STRICT/NATURAL modes.
    """

    def __init__(self, llm: FrameworkLLMClient) -> None:
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        *,
        target_word_count: int = 600,
        max_tokens: int = 8000,
        max_tool_calls: int = 40,
        section_descriptions: str = "",
    ) -> AsyncGenerator[tuple[str, InterleavedClaim | None], None]:
        """Run the ReAct synthesis loop.

        Yields (content, claim) tuples:
        - Final assembled content is yielded once as (content, None)
        - Individual claims are yielded as ("", claim)
        """
        key_map = build_citation_key_map(evidence_pool)

        search_index = EvidenceSearchIndex.create(evidence_pool, llm_client=self._llm)
        executor = SynthesisToolExecutor(evidence_pool, key_map, search_index)

        system_prompt = _REACT_GENERATION_PROMPT.format(
            min_words=target_word_count,
            max_words=target_word_count * 2,
        )

        source_index = build_evidence_source_index(evidence_pool, key_map)

        user_content = (
            f"Query: {query}\n\n"
            f"{source_index}\n\n"
            f"Budget: {max_tool_calls} tool calls total — pace across sections.\n"
        )
        if section_descriptions:
            user_content += f"\nReport Structure:\n{section_descriptions}\n"
        user_content += (
            "\nStart writing. Use search_evidence → read_snippet → "
            "<cite> for each fact."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_content = ""
        tool_calls_used = 0

        while tool_calls_used < max_tool_calls:
            response = await self._llm.complete(
                messages=messages,
                tier=ModelTier.analytical,
                max_tokens=max_tokens,
                tools=SYNTHESIS_TOOLS,
            )

            raw_content += response.content or ""

            if not response.tool_calls:
                break

            messages.append(build_assistant_message(response))

            for tc in response.tool_calls:
                tool_calls_used += 1
                result = await executor.execute(tc.function_name, tc.arguments)
                if tc.function_name == "read_snippet":
                    result += (
                        '\n\n⚠️ NOW WRITE: Use <cite key="...">claim</cite> '
                        "based on this evidence."
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        logger.info(
            "REACT_SYNTHESIS_COMPLETE tool_calls=%d raw_chars=%d read_indices=%d",
            tool_calls_used,
            len(raw_content),
            len(executor.read_indices),
        )

        cleaned = _post_process_react_content(raw_content)
        assembled, claims = _parse_react_content(cleaned, evidence_pool, key_map)

        if not claims and raw_content.strip():
            logger.warning(
                "REACT_SYNTHESIS_NO_TAGS raw_chars=%d "
                "falling_back_to_raw_content",
                len(raw_content),
            )
            assembled = cleaned

        yield assembled, None
        for claim in claims:
            yield "", claim
