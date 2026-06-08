"""Stage 2: Interleaved Generation -- ReClaim pattern.

Generates claims constrained by pre-selected evidence using the
reference-first claim generation pattern.  The LLM receives an
evidence pool with indexed spans and produces text with ``[N]``
citation markers that are then replaced with human-readable keys.

Two generation modes are supported:

- **strict**: Every sentence must cite evidence.  Maximum citation density.
- **natural**: Light-touch citations with better prose quality.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from databricks_deep_research.citation.citation_keys import (
    build_citation_key_map,
    replace_numeric_markers,
)
from databricks_deep_research.citation.types import (
    ClaimRole,
    InterleavedClaim,
    RankedEvidence,
)
from databricks_deep_research.citation.utils import has_numeric_content as _has_numeric_content
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)

_ABBREVIATION_SENTINEL = "\u2024"
_PROTECTED_ABBREVIATIONS = (
    "u.s.",
    "u.k.",
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "vs.",
    "etc.",
)
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|(?:\d+|[A-Za-z])[.)])\s+")
_LIST_MARKER_ONLY_PATTERN = re.compile(r"^\s*(?:[-*+]|(?:\d+|[A-Za-z])[.)])\s*$")
_MARKDOWN_LABEL_PREFIX = re.compile(r"^\s*\*\*[^*]+\*\*:\s*")
_NUMBER_ONLY_PATTERN = re.compile(r"^\s*\d+[.)]?\s*$")
_MARKDOWN_LABEL_ONLY_PATTERN = re.compile(r"^\s*\*\*[^*]+\*\*:?\s*$")
_CLAIM_VERB_PATTERN = re.compile(
    r"\b("
    r"is|are|was|were|has|have|had|reported|reports|increased|decreased|declined|"
    r"reached|totaled|generated|recorded|grew|rose|fell|used|uses|required|requires|"
    r"contains|contain|supports|support|achieved|mark|marks|indicates|suggests|"
    r"remained|remain|includes|included|stands|compared|represents|representing"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GenerationMode(StrEnum):
    """Generation mode for the interleaved generator.

    - ``STRICT``:  Heavy ``[N]`` constraints, one claim per sentence.
    - ``NATURAL``: Light-touch ``[N]`` citations, balanced quality.
    """

    STRICT = "strict"
    NATURAL = "natural"


@dataclass(frozen=True)
class InterleavedGenerationConfig:
    """Configuration knobs for the interleaved generator."""

    min_evidence_similarity: float = 0.5
    generation_mode: GenerationMode = GenerationMode.NATURAL
    # Per-evidence-quote cap applied when formatting the generation prompt.
    # Wired from the pipeline-wide CitationConfig.max_evidence_chars at
    # construction time so all 5 truncation sites stay aligned.
    max_evidence_chars: int = 3000


# ---------------------------------------------------------------------------
# Pydantic structured-output model
# ---------------------------------------------------------------------------


class ClaimEvidenceMatchOutput(BaseModel):
    """Output from the claim-evidence matching LLM call."""

    evidence_index: int | None = Field(
        default=None,
        description="Index of best matching evidence (null if no match)",
    )
    entailment: Literal["full", "partial", "none"] = Field(
        default="none",
        description="Level of entailment: full, partial, or none",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why evidence supports/doesn't support claim",
    )


# ---------------------------------------------------------------------------
# Prompt templates  (inlined from app -- no external dependency)
# ---------------------------------------------------------------------------

_STRICT_GENERATION_PROMPT = """\
You are a Research Synthesizer generating a comprehensive response with inline citations.

## LENGTH REQUIREMENT
- Target length: {target_word_count} words — write a thorough report that reaches this target
- Maximum: {max_word_count} words — do not exceed this limit
- Do NOT stop writing before reaching the target unless you have exhausted all relevant evidence
- Cover ALL aspects of the research query comprehensively
- Structure your response with clear sections and subsections

## CRITICAL RULE: Reference-First Generation
For EVERY claim you make:
1. FIRST select the supporting evidence from the pool below
2. THEN write the claim constrained by that evidence
3. IMMEDIATELY cite the evidence using [source_index] notation

## CITATION DIVERSITY
- Cite the most relevant sources for each claim — do NOT force-cite irrelevant sources
- DISTRIBUTE citations across sources — do NOT over-rely on any single source
- Each source should be cited at most 3-4 times maximum
- If multiple sources say the same thing, cite the BEST one, not all of them
- Aim to cite at least {min_sources_to_cite} different sources, but only if genuinely relevant

## Evidence Pool ({evidence_count} evidence spans from {source_count} sources)
{evidence_pool}

## Query
{query}
{generation_instructions_section}

## Generation Guidelines

### Citation Format
- Use [0], [1], [2], etc. to cite evidence spans by their index
- Place citation IMMEDIATELY after the claim it supports
- One claim per sentence for clear attribution

### Claim Types
- **Fact Claims**: Verifiable source-grounded statements with inline citations
- **Numeric Claims**: Statistics, values, metrics [1] - ensure exact match with source, and state the value's unit of measure or currency exactly as the source expresses it (e.g. `$3.2 billion`, `15%`, `1,200 units`) — never a bare number
- **Analysis Blocks**: Use `<analysis>...</analysis>` only for interpretation of already-established cited facts
- **Free Blocks**: Use `<free>...</free>` ONLY for markdown headings (## or ###) and single-sentence transitions between sections — NOTHING else

### Structure
- Use markdown headings (##, ###) to organize your response
- Keep introductions and conclusions structural, not analytical claims
- Cover multiple aspects of the topic using different evidence sources
- Aim for depth and comprehensiveness

### Tables
When comparative or tabular data is needed, use proper markdown table syntax:
| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

Do NOT use structured lists as a substitute for tables.
Only use a markdown table if the query explicitly calls for tabular comparison.

### What NOT to Do
- NEVER make claims without citing evidence
- NEVER synthesize numbers not in the evidence
- NEVER present a numeric value without its unit or currency, and NEVER use a source's internal field or column identifier (e.g. a raw column label) as the value's meaning — translate it into what it represents
- NEVER paraphrase in a way that changes meaning
- NEVER cite evidence that doesn't support your claim
- NEVER add editorial framing like "strong foundation", "healthy performance", or "resilience"
  unless those words are directly supported by the evidence
- NEVER introduce a new number, date, quarter, ranking, or causal claim inside `<analysis>`
- NEVER put factual payload inside `<free>` blocks
- ALWAYS place analysis AFTER the fact sentences it interprets
- KEEP analysis bounded: use language like "may indicate", "suggests", or "appears consistent with"
- NEVER combine a table block with narrative commentary in the same paragraph
- NEVER write a comparison across multiple quarters unless every referenced value is cited
- Avoid filler, but ensure thorough coverage — do not stop before reaching the target
- NEVER cite the same source more than 4 times
- NEVER use structured lists when a table is requested - use markdown table syntax
- NEVER add meta-commentary about the report
- NEVER include a "How to read this report" section or describe your citation process
- NEVER reference the <free> or <analysis> tags in your visible text — they are internal markup only
- NEVER wrap executive summaries, introductions, or any multi-sentence content in <free> tags
- NEVER offer follow-up work or additional analyses
- NEVER end with questions or invitations for feedback

## Example Output (note the diversity of citations)
"The company reported Q4 revenue of $3.2 billion [0], representing a 15% year-over-year \
increase [1]. The CEO attributed this growth to strong demand in Asia [2], particularly in \
the consumer electronics segment [3]. Analysts noted that market expansion in Europe also \
contributed significantly [4]."

## Response
Generate a well-structured response (target: {target_word_count} words, max: {max_word_count} words) with inline \
citations for every claim:"""

_NATURAL_GENERATION_PROMPT = """\
You are a Research Synthesizer writing an engaging, comprehensive report.

## Evidence Pool ({evidence_count} evidence spans from {source_count} sources)
{evidence_pool}

## Query
{query}
{generation_instructions_section}

## Writing Guidelines

### Writing Quality (MOST IMPORTANT)
- Write naturally and engagingly - this should read like quality journalism
- Use varied sentence structures and paragraph lengths
- Craft smooth transitions between ideas
- Make complex topics accessible
- Use concrete examples and clear explanations

### Citation Style
- Use [0], [1], [2], etc. to cite evidence by index
- Cite when stating specific facts, statistics, or claims from sources
- **Don't over-cite** - not every sentence needs a citation
- Prioritize natural flow over citation density
- Multiple related facts can share one citation when appropriate
- Aim to use evidence from multiple sources for credibility

### Structure
- Use markdown headings (##, ###) to organize your response
- Put interpretation inside `<analysis>...</analysis>` blocks
- Put structural transitions/openers inside `<free>...</free>` blocks
- Keep introductions and conclusions structural unless analysis is clearly grounded
- Target: {target_word_count} words — write a thorough report (max: {max_word_count} words)
- Cover the topic comprehensively — do not stop before reaching the word target

### Tables
For comparative data, use markdown tables:
| Header | Header |
|--------|--------|
| Data   | Data   |

### What TO Do
- Write for readability first, citations second
- Cite specific facts, numbers, and claims that need attribution
- Always state a numeric value's unit of measure or currency as the source expresses it (e.g. `$3.2 billion`, `15%`); never present a bare number or use a source's internal field/column identifier as its meaning
- Let prose flow naturally between cited and non-cited material
- Use all available evidence to build comprehensive coverage

### What NOT to Do
- Don't force "one citation per sentence" - that's mechanical
- Don't interrupt natural paragraph flow just to add citations
- Don't sacrifice readability for citation density
- Don't synthesize or invent numbers not in the evidence
- Don't over-rely on a single source - spread citations across sources
- Don't use structured lists when tables are requested
- Don't introduce new numeric/date/entity payload in `<analysis>` blocks
- Don't put factual claims inside `<free>` blocks
- Don't use `<free>` for anything other than headings and single-sentence transitions
- Don't describe the citation system or tagging approach in the output
- Don't add meta-commentary about the report itself
- Don't offer follow-up work or additional analyses
- Don't end with questions or engagement prompts

## Example of Natural Writing Style
"The renewable energy sector has experienced remarkable growth in recent years [0]. Solar \
panel costs, for instance, have dropped by over 80% since 2010, making residential \
installations increasingly accessible [1]. This price decline, combined with government \
incentives in many regions, has driven adoption rates to record highs. Industry analysts \
project continued expansion, with some estimates suggesting renewables could account for 50% \
of global electricity generation by 2030 [2]."

## Response
Write an engaging, well-researched report (target: {target_word_count} words, max: {max_word_count} words) using `<analysis>` and `<free>` blocks where appropriate:"""

_CLAIM_EVIDENCE_MATCHING_PROMPT = """\
Match this claim to the most relevant evidence span.

## Claim
{claim_text}

## Evidence Pool
{evidence_pool}

## Task
1. Find the evidence span that BEST supports this claim
2. Verify the claim is ENTAILED by (fully supported by) the evidence
3. If no evidence supports the claim, mark as "no_match"

## Response Format
```json
{{
  "evidence_index": 0,
  "entailment": "full" | "partial" | "none",
  "reasoning": "why this evidence supports (or doesn't support) the claim"
}}
```

Respond with the matching result:"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class InterleavedGenerator:
    """Stage 2: Interleaved Generation.

    Generates claims constrained by pre-selected evidence using the ReClaim
    pattern for high citation accuracy.

    Args:
        llm_client: Framework LLM client for model calls.
        config: Optional generation configuration.  Uses sensible defaults
            when not provided.
    """

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        config: InterleavedGenerationConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or InterleavedGenerationConfig()

    # ----- evidence selection (heuristic) ----------------------------------

    def select_best_evidence(
        self,
        _query: str,
        claim_context: str,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[RankedEvidence | None, int | None]:
        """Select the best evidence for a potential claim via keyword overlap.

        Returns:
            ``(best_evidence, index)`` or ``(None, None)``.
        """
        if not evidence_pool:
            return None, None

        best_score = 0.0
        best_evidence: RankedEvidence | None = None
        best_index: int | None = None
        claim_lower = claim_context.lower()

        for i, evidence in enumerate(evidence_pool):
            quote_lower = evidence.quote_text.lower()
            words = set(re.findall(r"\b\w{3,}\b", claim_lower))
            if not words:
                continue

            matches = sum(1 for w in words if w in quote_lower)
            score = matches / len(words)
            if evidence.relevance_score:
                score = (score + evidence.relevance_score) / 2

            if score > best_score:
                best_score = score
                best_evidence = evidence
                best_index = i

        if best_score >= self._config.min_evidence_similarity:
            return best_evidence, best_index
        return None, None

    # ----- single constrained claim ----------------------------------------

    async def generate_constrained_claim(
        self,
        query: str,
        evidence: RankedEvidence,
        context: str = "",
    ) -> str:
        """Generate a single claim constrained by *evidence*."""
        prompt = (
            f'Generate a factual claim based ONLY on this evidence.\n\n'
            f'Evidence from {evidence.source_title or "source"}:\n'
            f'"{evidence.quote_text}"\n\n'
            f'Query context: {query}\n'
            f'{f"Additional context: {context}" if context else ""}\n\n'
            f'Write ONE concise factual claim that is fully supported by the evidence:'
        )
        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.simple,
        )
        return response.content.strip()

    # ----- claim-evidence matching -----------------------------------------

    async def match_claim_to_evidence(
        self,
        claim_text: str,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[int | None, str, str]:
        """Match *claim_text* to the best evidence in the pool.

        Returns:
            ``(evidence_index, entailment_level, reasoning)``.
        """
        if not evidence_pool:
            return None, "none", "No evidence available"

        evidence_cap = self._config.max_evidence_chars
        evidence_text = "\n".join(
            f"[{i}] {e.quote_text[:evidence_cap]}..."
            if len(e.quote_text) > evidence_cap
            else f"[{i}] {e.quote_text}"
            for i, e in enumerate(evidence_pool)
        )
        prompt = _CLAIM_EVIDENCE_MATCHING_PROMPT.format(
            claim_text=claim_text,
            evidence_pool=evidence_text,
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.simple,
                structured_output=ClaimEvidenceMatchOutput,
            )
            if response.structured:
                output: ClaimEvidenceMatchOutput = response.structured
                return output.evidence_index, output.entailment, output.reasoning
        except Exception as exc:
            logger.warning("Failed to match claim to evidence: %s", exc)

        return None, "none", "Failed to match"

    # ----- interleaving (claim-only generator) -----------------------------

    async def synthesize_with_interleaving(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        previous_content: str = "",
        target_word_count: int = 600,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[InterleavedClaim, None]:
        """Yield claims extracted from interleaved generation.

        Wraps :meth:`synthesize_with_streaming` and yields only the
        ``InterleavedClaim`` objects (discarding the raw content chunks).
        """
        async for _content, claim in self.synthesize_with_streaming(
            query=query,
            evidence_pool=evidence_pool,
            previous_content=previous_content,
            target_word_count=target_word_count,
            max_tokens=max_tokens,
        ):
            if claim:
                yield claim

    # ----- streaming synthesis (main entry point) --------------------------

    async def synthesize_with_streaming(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        previous_content: str = "",
        target_word_count: int = 600,
        max_tokens: int = 2000,
        generation_instructions: str = "",
    ) -> AsyncGenerator[tuple[str, InterleavedClaim | None], None]:
        """Generate content with interleaved claims and streaming.

        Yields ``(content_chunk, claim_or_none)`` tuples:

        * The first yield contains the full generated content (with
          human-readable citation keys) and ``None`` for the claim.
        * Subsequent yields contain ``""`` for content and an
          ``InterleavedClaim`` each.
        """
        if not evidence_pool:
            logger.warning("No evidence pool provided for interleaved generation")
            return

        unique_sources = len({e.source_url for e in evidence_pool if e.source_url})
        logger.info(
            "INTERLEAVED_GENERATION_START query=%s evidence=%d unique_sources=%d "
            "target_words=%d max_tokens=%d previous_content_chars=%d",
            query[:120],
            len(evidence_pool),
            unique_sources,
            target_word_count,
            max_tokens,
            len(previous_content),
        )

        logger.info(
            "GENERATION_EVIDENCE_POOL evidence=%d unique_sources=%d",
            len(evidence_pool),
            unique_sources,
        )

        # Format evidence pool for the prompt
        evidence_cap = self._config.max_evidence_chars
        evidence_text = "\n".join(
            f"[{i}] Source: {e.source_title or 'Unknown'}\n"
            f'   Quote: "{e.quote_text[:evidence_cap]}{"..." if len(e.quote_text) > evidence_cap else ""}"'
            for i, e in enumerate(evidence_pool)
        )

        min_sources_to_cite = max(2, min(10, unique_sources // 3))
        max_word_count = int(target_word_count * 1.3)
        instructions = generation_instructions.strip()
        generation_instructions_section = (
            "\n## Workflow-Specific Report Contract\n"
            f"{instructions}\n\n"
            "Apply this contract for section names, required deliverables, "
            "tone, and quality gates as strictly as the evidence allows. If a "
            "required section lacks citeable evidence, keep the section heading "
            "but do not invent factual content."
            if instructions
            else ""
        )

        # Select prompt template
        mode = self._config.generation_mode
        if mode == GenerationMode.NATURAL:
            prompt = _NATURAL_GENERATION_PROMPT.format(
                query=query,
                generation_instructions_section=generation_instructions_section,
                evidence_pool=evidence_text,
                target_word_count=target_word_count,
                max_word_count=max_word_count,
                evidence_count=len(evidence_pool),
                source_count=unique_sources,
            )
            logger.info(
                "GENERATION_MODE mode=natural evidence=%d sources=%d",
                len(evidence_pool),
                unique_sources,
            )
        elif mode == GenerationMode.STRICT:
            prompt = _STRICT_GENERATION_PROMPT.format(
                query=query,
                generation_instructions_section=generation_instructions_section,
                evidence_pool=evidence_text,
                target_word_count=target_word_count,
                max_word_count=max_word_count,
                evidence_count=len(evidence_pool),
                source_count=unique_sources,
                min_sources_to_cite=min_sources_to_cite,
            )
            logger.info(
                "GENERATION_MODE mode=strict evidence=%d sources=%d",
                len(evidence_pool),
                unique_sources,
            )
        else:
            raise ValueError(
                f"Generation mode '{mode}' not supported in claim_generator. "
                "Classical mode should be handled at pipeline level."
            )

        if previous_content:
            # Reframe so the LLM does NOT autoregress over previous_content as
            # prose to continue. The prior framing ("Previous content: ... /
            # Continue from here:") caused the model to skip the [N] markers
            # entirely (root cause of the 45-of-45 grounding-warning banner
            # observed on shell-app deployments). Treat previous_content as
            # background notes only and re-instruct the model to write from
            # scratch with mandatory [N] citations.
            prompt += (
                "\n\n## Research Notes (background context only — NOT a draft to continue)\n"
                f"{previous_content}\n\n"
                "Now write the response from scratch. The instructions above are binding: "
                "every factual claim MUST be followed by one or more [N] citation markers "
                "from the evidence pool. Do NOT continue the research notes verbatim; "
                "do NOT produce a citation-free narrative."
            )

        try:
            key_map = build_citation_key_map(evidence_pool)

            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.analytical,
                max_tokens=max_tokens,
            )
            content = response.content
            logger.debug(
                "INTERLEAVED_GENERATION_RESPONSE chars=%d citation_markers=%d",
                len(content),
                len(re.findall(r"\[[A-Za-z0-9-]+\]|\[\d+\]", content)),
            )

            # Replace [0], [1] with [Arxiv], [Github], etc.
            content_with_keys = replace_numeric_markers(content, key_map)
            yield content_with_keys, None

            # Build reverse map for claim parsing
            reverse_key_map = {key: idx for idx, key in key_map.items()}

            claims = _parse_interleaved_content(
                content_with_keys, evidence_pool, reverse_key_map
            )
            logger.info(
                "INTERLEAVED_GENERATION_PARSED claims=%d cited=%d uncited=%d roles=%s types=%s",
                len(claims),
                sum(1 for claim in claims if claim.citation_keys or claim.citation_key),
                sum(1 for claim in claims if not (claim.citation_keys or claim.citation_key)),
                dict(Counter(claim.claim_role for claim in claims)),
                dict(Counter(claim.claim_type for claim in claims)),
            )
            for claim in claims:
                logger.debug(
                    "CLAIM_PARSED text=%.60s evidence_index=%s key=%s pos=(%d,%d)",
                    claim.claim_text,
                    claim.evidence_index,
                    claim.citation_key,
                    claim.position_start,
                    claim.position_end,
                )
                yield "", claim

        except Exception:
            logger.error("Interleaved generation failed", exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Parsing helpers (module-level, stateless)
# ---------------------------------------------------------------------------


def _parse_interleaved_content(
    content: str,
    evidence_pool: list[RankedEvidence],
    reverse_key_map: dict[str, int] | None = None,
) -> list[InterleavedClaim]:
    """Parse generated content into claims with citation linkage.

    The content may use either numeric markers (``[0]``, ``[1]``) or
    human-readable keys (``[Arxiv]``, ``[Github]``).  When
    *reverse_key_map* is provided the parser expects human-readable keys.
    """
    claims: list[InterleavedClaim] = []

    key_citation_pattern = r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]"
    key_clean_pattern = r"\s*\[[A-Za-z][A-Za-z0-9-]*(?:-\d+)?\]"
    numeric_citation_pattern = r"\[(\d+)\]"
    numeric_clean_pattern = r"\s*\[\d+\]"

    for sentence, sentence_start, block_role in _split_claim_segments(content):
        if not sentence.strip():
            continue

        # Strip leading markdown headers
        header_pattern = r"^(?:#+\s+[^\n]*\n*)+"
        header_match = re.match(header_pattern, sentence)
        header_len = len(header_match.group()) if header_match else 0
        claim_position_start = sentence_start + header_len

        clean_sentence = sentence[header_len:] if header_match else sentence

        citation_matches: list[str] = []
        matched_numeric_citations = False

        if reverse_key_map:
            citation_matches = re.findall(key_citation_pattern, clean_sentence)
            if not citation_matches:
                citation_matches = re.findall(numeric_citation_pattern, clean_sentence)
                matched_numeric_citations = bool(citation_matches)
        else:
            citation_matches = re.findall(numeric_citation_pattern, clean_sentence)

        claim_text = re.sub(key_clean_pattern, "", clean_sentence)
        claim_text = re.sub(numeric_clean_pattern, "", claim_text).strip()

        if not claim_text:
            continue

        claim_type = "numeric" if _has_numeric_content(claim_text) else "general"
        claim_role = block_role or ClaimRole.FACT.value

        evidence: RankedEvidence | None = None
        evidence_index: int | None = None
        evidences: list[RankedEvidence] = []
        evidence_indices: list[int] = []
        citation_key: str | None = None
        citation_keys: list[str] | None = None

        if citation_matches:
            first_match = citation_matches[0]
            citation_keys = list(citation_matches)

            if reverse_key_map and not matched_numeric_citations:
                citation_key = first_match
                evidence_index = reverse_key_map.get(citation_key)
                for key in citation_matches:
                    idx = reverse_key_map.get(key)
                    if idx is None or not 0 <= idx < len(evidence_pool):
                        continue
                    if idx not in evidence_indices:
                        evidence_indices.append(idx)
                        evidences.append(evidence_pool[idx])
                if evidence_index is not None and 0 <= evidence_index < len(evidence_pool):
                    evidence = evidence_pool[evidence_index]
            else:
                for numeric_key in citation_matches:
                    try:
                        idx = int(numeric_key)
                    except ValueError:
                        continue
                    if 0 <= idx < len(evidence_pool) and idx not in evidence_indices:
                        evidence_indices.append(idx)
                        evidences.append(evidence_pool[idx])
                try:
                    idx = int(first_match)
                    if 0 <= idx < len(evidence_pool):
                        evidence = evidence_pool[idx]
                        evidence_index = idx
                except ValueError:
                    pass

        sentence_end = sentence_start + len(sentence.rstrip())

        claims.append(
            InterleavedClaim(
                claim_text=claim_text,
                claim_type=claim_type,
                position_start=claim_position_start,
                position_end=sentence_end,
                evidence=evidence,
                evidence_index=evidence_index,
                evidences=evidences,
                evidence_indices=evidence_indices,
                confidence_score=evidence.relevance_score if evidence else None,
                claim_role=claim_role,
                citation_key=citation_key,
                citation_keys=citation_keys,
                from_free_block=claim_role == ClaimRole.FREE.value,
            )
        )

    return claims


def _split_claim_segments(content: str) -> list[tuple[str, int, str | None]]:
    """Split content into verification-ready segments while respecting block structure."""
    if not content:
        return []

    segments: list[tuple[str, int, str | None]] = []
    cursor = 0
    block_pattern = re.compile(
        r"<(?P<role>analysis|free)>(?P<body>.*?)</(?P=role)>",
        re.IGNORECASE | re.DOTALL,
    )

    for match in block_pattern.finditer(content):
        if match.start() > cursor:
            segments.extend(
                _split_block_segments(
                    content[cursor:match.start()],
                    cursor,
                    ClaimRole.FACT.value,
                )
            )
        role = match.group("role").lower()
        body = match.group("body")
        segments.extend(
            _split_block_segments(
                body,
                match.start("body"),
                role,
            )
        )
        cursor = match.end()

    if cursor < len(content):
        segments.extend(
            _split_block_segments(
                content[cursor:],
                cursor,
                ClaimRole.FACT.value,
            )
        )

    return segments


def _split_block_segments(
    block: str,
    block_start: int,
    role: str | None,
) -> list[tuple[str, int, str | None]]:
    """Split a block into claim-like segments while preserving a role hint."""
    segments: list[tuple[str, int, str | None]] = []
    cursor = 0
    for part in re.split(r"(\n\s*\n)", block):
        if not part:
            continue
        if re.fullmatch(r"\n\s*\n", part):
            cursor += len(part)
            continue

        part_start = block_start + cursor
        cursor += len(part)

        if _is_markdown_table_block(part):
            line_offset = 0
            table_row_index = 0
            for line in part.splitlines(keepends=True):
                stripped = line.strip()
                line_start = part_start + line_offset
                line_offset += len(line)
                if not stripped:
                    continue
                if stripped.startswith("|"):
                    if re.fullmatch(r"\|?[\s:\-|]+\|?", stripped):
                        continue
                    if table_row_index == 0:
                        table_row_index += 1
                        continue
                    table_row_index += 1
                    candidate_text, candidate_start = _normalize_claim_segment(
                        line.rstrip(),
                        line_start,
                    )
                    if _is_claim_candidate(candidate_text, role, table_row=True):
                        segments.append((candidate_text, candidate_start, role))
                    continue
                for sentence, sentence_start in _split_sentences_with_offsets(
                    line,
                    line_start,
                ):
                    candidate_text, candidate_start = _normalize_claim_segment(
                        sentence,
                        sentence_start,
                    )
                    if _is_claim_candidate(candidate_text, role):
                        segments.append((candidate_text, candidate_start, role))
            continue

        if _has_list_structure(part):
            line_offset = 0
            for line in part.splitlines(keepends=True):
                stripped = line.strip()
                line_start = part_start + line_offset
                line_offset += len(line)
                if not stripped:
                    continue
                for sentence, sentence_start in _split_sentences_with_offsets(line, line_start):
                    candidate_text, candidate_start = _normalize_claim_segment(
                        sentence,
                        sentence_start,
                    )
                    if _is_claim_candidate(candidate_text, role):
                        segments.append((candidate_text, candidate_start, role))
            continue

        for sentence, sentence_start in _split_sentences_with_offsets(part, part_start):
            candidate_text, candidate_start = _normalize_claim_segment(
                sentence,
                sentence_start,
            )
            if _is_claim_candidate(candidate_text, role):
                segments.append((candidate_text, candidate_start, role))

    return segments


def _split_sentences_with_offsets(text: str, start: int) -> list[tuple[str, int]]:
    """Split a block of prose into sentence-like spans with absolute offsets."""
    spans: list[tuple[str, int]] = []
    local_cursor = 0
    protected_text = _protect_abbreviations(text)
    pieces = re.split(r"(?<=[.!?])\s+", protected_text)
    if len(pieces) == 1 and pieces[0].strip():
        return [(_restore_abbreviations(pieces[0]), start)]

    for piece in pieces:
        if not piece.strip():
            local_cursor += len(piece)
            continue
        restored_piece = _restore_abbreviations(piece)
        offset = text.find(restored_piece, local_cursor)
        if offset < 0:
            offset = local_cursor
        spans.append((restored_piece, start + offset))
        local_cursor = offset + len(restored_piece)
    return spans


def _protect_abbreviations(text: str) -> str:
    protected = text
    for abbr in _PROTECTED_ABBREVIATIONS:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        protected = pattern.sub(
            lambda match: match.group(0).replace(".", _ABBREVIATION_SENTINEL),
            protected,
        )
    return protected


def _restore_abbreviations(text: str) -> str:
    return text.replace(_ABBREVIATION_SENTINEL, ".")


def _has_list_structure(part: str) -> bool:
    return any(
        _LIST_ITEM_PATTERN.match(line.strip()) or line.lstrip().startswith("#")
        for line in part.splitlines()
        if line.strip()
    )


def _normalize_claim_segment(text: str, start: int) -> tuple[str, int]:
    normalized = text
    normalized_start = start

    leading_ws = len(normalized) - len(normalized.lstrip())
    if leading_ws:
        normalized_start += leading_ws
        normalized = normalized.lstrip()

    list_match = _LIST_ITEM_PATTERN.match(normalized)
    if list_match:
        normalized_start += list_match.end()
        normalized = normalized[list_match.end():]

    label_match = _MARKDOWN_LABEL_PREFIX.match(normalized)
    if label_match:
        normalized_start += label_match.end()
        normalized = normalized[label_match.end():]

    return normalized.strip(), normalized_start


def _is_claim_candidate(text: str, role: str | None, *, table_row: bool = False) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    if _LIST_MARKER_ONLY_PATTERN.fullmatch(stripped):
        return False
    if _NUMBER_ONLY_PATTERN.fullmatch(stripped):
        return False
    if _MARKDOWN_LABEL_ONLY_PATTERN.fullmatch(stripped):
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False
    if table_row:
        return bool(_has_numeric_content(stripped) or re.search(r"\[[^\]]+\]", stripped))
    if role == ClaimRole.FREE.value:
        return len(stripped.split()) >= 2
    if re.search(r"\[[^\]]+\]", stripped):
        return len(stripped.split()) >= 2
    if _has_numeric_content(stripped):
        return True
    if _CLAIM_VERB_PATTERN.search(stripped):
        return True
    return len(stripped.split()) >= 4 and stripped.endswith((".", "!", "?"))


def _is_markdown_table_block(block: str) -> bool:
    """Return True when a block appears to be a markdown table."""
    table_lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(table_lines) < 2:
        return False
    if not all(line.startswith("|") for line in table_lines[:2]):
        return False
    return bool(re.fullmatch(r"\|?[\s:\-|]+\|?", table_lines[1]))
