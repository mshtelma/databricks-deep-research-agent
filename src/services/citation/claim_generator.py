"""Stage 2: Interleaved Generation service.

Generates claims constrained by pre-selected evidence using the
ReClaim pattern (reference-first claim generation).
"""

import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.prompts.citation.interleaved_generation import (
    CLAIM_EVIDENCE_MATCHING_PROMPT,
    INTERLEAVED_GENERATION_PROMPT,
    NATURAL_GENERATION_PROMPT,
)
from src.core.app_config import GenerationMode, get_app_config
from src.core.logging_utils import get_logger
from src.services.citation.citation_keys import (
    build_citation_key_map,
    replace_numeric_markers,
)
from src.services.citation.evidence_selector import RankedEvidence
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

logger = get_logger(__name__)


# Pydantic models for structured LLM output
class ClaimEvidenceMatchOutput(BaseModel):
    """Output from claim-evidence matching LLM call."""

    evidence_index: int | None = Field(
        default=None,
        description="Index of best matching evidence (null if no match)"
    )
    entailment: Literal["full", "partial", "none"] = Field(
        default="none",
        description="Level of entailment: full, partial, or none"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why evidence supports/doesn't support claim"
    )


@dataclass
class InterleavedClaim:
    """A claim generated with evidence constraint."""

    claim_text: str
    claim_type: str  # "general" or "numeric"
    position_start: int
    position_end: int
    evidence: RankedEvidence | None
    evidence_index: int | None
    confidence_score: float | None
    citation_key: str | None = None  # Primary key like "Arxiv", "Zhipu"
    citation_keys: list[str] | None = None  # All keys for multi-marker sentences


class InterleavedGenerator:
    """Stage 2: Interleaved Generation.

    Generates claims constrained by pre-selected evidence using
    the ReClaim pattern for 90% citation accuracy.

    Supports two generation modes (classical mode is handled at pipeline level):
    - "natural": Light-touch [N] citations, balanced quality + verification
    - "strict": Heavy [N] constraints (current behavior), maximum citations
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize interleaved generator.

        Args:
            llm_client: LLM client for generation.
        """
        self._llm = llm_client
        self._citation_config = get_app_config().citation_verification
        self._config = self._citation_config.interleaved_generation
        self._generation_mode = self._citation_config.generation_mode

    def select_best_evidence(
        self,
        _query: str,
        claim_context: str,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[RankedEvidence | None, int | None]:
        """Select the best evidence for a potential claim.

        Args:
            query: Research query context.
            claim_context: Context for the claim to be generated.
            evidence_pool: Available evidence spans.

        Returns:
            Tuple of (best evidence, index) or (None, None).
        """
        if not evidence_pool:
            return None, None

        # Find evidence with highest relevance to claim context
        best_score = 0.0
        best_evidence = None
        best_index = None

        claim_lower = claim_context.lower()

        for i, evidence in enumerate(evidence_pool):
            # Simple keyword overlap scoring
            quote_lower = evidence.quote_text.lower()
            words = set(re.findall(r"\b\w{3,}\b", claim_lower))

            if not words:
                continue

            matches = sum(1 for word in words if word in quote_lower)
            score = matches / len(words)

            # Boost by original relevance score
            if evidence.relevance_score:
                score = (score + evidence.relevance_score) / 2

            if score > best_score:
                best_score = score
                best_evidence = evidence
                best_index = i

        if best_score >= self._config.min_evidence_similarity:
            return best_evidence, best_index

        return None, None

    async def generate_constrained_claim(
        self,
        query: str,
        evidence: RankedEvidence,
        context: str = "",
    ) -> str:
        """Generate a single claim constrained by evidence.

        Args:
            query: Research query.
            evidence: Evidence to base claim on.
            context: Additional context for the claim.

        Returns:
            Generated claim text.
        """
        prompt = f"""Generate a factual claim based ONLY on this evidence.

Evidence from {evidence.source_title or 'source'}:
"{evidence.quote_text}"

Query context: {query}
{f'Additional context: {context}' if context else ''}

Write ONE concise factual claim that is fully supported by the evidence:"""

        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.SIMPLE,
        )

        return response.content.strip()

    async def match_claim_to_evidence(
        self,
        claim_text: str,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[int | None, str, str]:
        """Match a claim to the best evidence in the pool.

        Args:
            claim_text: Claim text to match.
            evidence_pool: Available evidence spans.

        Returns:
            Tuple of (evidence_index, entailment_level, reasoning).
        """
        if not evidence_pool:
            return None, "none", "No evidence available"

        # Format evidence pool for LLM
        evidence_text = "\n".join(
            f"[{i}] {e.quote_text[:300]}..."
            if len(e.quote_text) > 300
            else f"[{i}] {e.quote_text}"
            for i, e in enumerate(evidence_pool)
        )

        prompt = CLAIM_EVIDENCE_MATCHING_PROMPT.format(
            claim_text=claim_text,
            evidence_pool=evidence_text,
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.BULK_ANALYSIS,  # Use Gemini for matching/ranking
                structured_output=ClaimEvidenceMatchOutput,
            )

            if response.structured:
                output: ClaimEvidenceMatchOutput = response.structured
                return (output.evidence_index, output.entailment, output.reasoning)

        except Exception as e:
            logger.warning(f"Failed to match claim to evidence: {e}")

        return None, "none", "Failed to match"

    async def synthesize_with_interleaving(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        previous_content: str = "",
        target_word_count: int = 600,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[InterleavedClaim, None]:
        """Generate content with interleaved claims and citations.

        This is an async generator that yields claims as they are generated.

        Args:
            query: Research query.
            evidence_pool: Pre-selected evidence spans.
            previous_content: Previously generated content for context.
            target_word_count: Target word count for the report.
            max_tokens: Maximum tokens to generate.

        Yields:
            InterleavedClaim objects as they are generated.
        """
        # Use the new streaming method and just yield claims
        async for _content, claim in self.synthesize_with_streaming(
            query=query,
            evidence_pool=evidence_pool,
            previous_content=previous_content,
            target_word_count=target_word_count,
            max_tokens=max_tokens,
        ):
            if claim:
                yield claim

    async def synthesize_with_streaming(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        previous_content: str = "",
        target_word_count: int = 600,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[tuple[str, InterleavedClaim | None], None]:
        """Generate content with streaming, preserving markdown structure.

        This method streams the raw LLM output (with markdown formatting intact)
        while also extracting claims for verification.

        Args:
            query: Research query.
            evidence_pool: Pre-selected evidence spans.
            previous_content: Previously generated content for context.
            target_word_count: Target word count for the report.
            max_tokens: Maximum tokens to generate.

        Yields:
            Tuples of (content_chunk, claim_if_extracted).
            - First yields stream the raw content chunks
            - Then yields claims extracted from the full content
        """
        if not evidence_pool:
            logger.warning("No evidence pool provided for interleaved generation")
            return

        # Count unique sources
        unique_sources = len(set(e.source_url for e in evidence_pool if e.source_url))

        # Format evidence pool for prompt
        evidence_text = "\n".join(
            f"[{i}] Source: {e.source_title or 'Unknown'}\n"
            f"   Quote: \"{e.quote_text[:400]}{'...' if len(e.quote_text) > 400 else ''}\""
            for i, e in enumerate(evidence_pool)
        )

        # Calculate minimum sources to cite (at least half of available, min 2)
        min_sources_to_cite = max(2, unique_sources // 2)

        # Select prompt based on generation mode
        # Classical mode is handled at pipeline level, so should never reach here
        if self._generation_mode == GenerationMode.NATURAL:
            prompt = NATURAL_GENERATION_PROMPT.format(
                query=query,
                evidence_pool=evidence_text,
                target_word_count=target_word_count,
                evidence_count=len(evidence_pool),
                source_count=unique_sources,
            )
            logger.info(
                "GENERATION_MODE",
                mode="natural",
                evidence_count=len(evidence_pool),
                source_count=unique_sources,
            )
        elif self._generation_mode == GenerationMode.STRICT:
            prompt = INTERLEAVED_GENERATION_PROMPT.format(
                query=query,
                evidence_pool=evidence_text,
                target_word_count=target_word_count,
                evidence_count=len(evidence_pool),
                source_count=unique_sources,
                min_sources_to_cite=min_sources_to_cite,
            )
            logger.info(
                "GENERATION_MODE",
                mode="strict",
                evidence_count=len(evidence_pool),
                source_count=unique_sources,
            )
        else:
            # Classical mode should be handled at pipeline level
            raise ValueError(
                f"Generation mode '{self._generation_mode}' not supported in claim_generator. "
                "Classical mode should be handled at pipeline level."
            )

        if previous_content:
            prompt += f"\n\nPrevious content:\n{previous_content}\n\nContinue from here:"

        try:
            # Build citation key map from evidence pool
            # Maps: {0: "Arxiv", 1: "Zhipu", 2: "Github"}
            key_map = build_citation_key_map(evidence_pool)

            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.ANALYTICAL,  # Use balanced model for synthesis
                max_tokens=max_tokens,
            )

            content = response.content

            # Replace numeric markers [0], [1] with human-readable keys [Arxiv], [Zhipu]
            content_with_keys = replace_numeric_markers(content, key_map)

            # Yield the content with human-readable citation keys
            yield content_with_keys, None

            # Build reverse map: citation_key -> evidence_index
            reverse_key_map = {key: idx for idx, key in key_map.items()}

            # Parse claims from the REPLACED content (with human-readable keys)
            # This ensures positions match the actual text that will be stored/displayed
            claims = self._parse_interleaved_content(
                content_with_keys, evidence_pool, reverse_key_map
            )
            for claim in claims:
                logger.debug(
                    "CLAIM_PARSED",
                    claim_text=claim.claim_text[:60],
                    evidence_index=claim.evidence_index,
                    citation_key=claim.citation_key,
                    position=(claim.position_start, claim.position_end),
                )

                yield "", claim

        except Exception as e:
            logger.error(f"Interleaved generation failed: {e}")
            raise

    def _parse_interleaved_content(
        self,
        content: str,
        evidence_pool: list[RankedEvidence],
        reverse_key_map: dict[str, int] | None = None,
    ) -> list[InterleavedClaim]:
        """Parse generated content into claims with citations.

        Args:
            content: Generated content with citation markers.
                    Can be numeric [0], [1] or human-readable [Arxiv], [Zhipu].
            evidence_pool: Evidence pool for resolving citations.
            reverse_key_map: Optional mapping from citation key to evidence index.
                            If provided, expects human-readable keys in content.

        Returns:
            List of InterleavedClaim objects.
        """
        claims = []

        # Determine citation pattern based on whether we have human-readable keys
        # Human-readable: [Arxiv], [Zhipu-2], [Github]
        # Numeric: [0], [1], [2]
        if reverse_key_map:
            citation_pattern = r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]"
            clean_pattern = r"\s*\[[A-Za-z][A-Za-z0-9-]*(?:-\d+)?\]"
        else:
            citation_pattern = r"\[(\d+)\]"
            clean_pattern = r"\s*\[\d+\]"

        # Split content into sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)

        position = 0
        for sentence in sentences:
            if not sentence.strip():
                position += len(sentence) + 1
                continue

            # Strip leading markdown headers from sentence
            # Headers like "## Title\n" or "### Subtitle\n" at start
            header_pattern = r"^(?:#+\s+[^\n]*\n*)+"
            header_match = re.match(header_pattern, sentence)
            header_len = len(header_match.group()) if header_match else 0

            # Adjust position to start after headers
            claim_position_start = position + header_len

            # Remove headers from sentence before extracting claim text
            clean_sentence = sentence[header_len:] if header_match else sentence

            # Find citation markers (human-readable or numeric)
            citation_matches = re.findall(citation_pattern, clean_sentence)

            # Clean sentence (remove citation markers for claim text)
            claim_text = re.sub(clean_pattern, "", clean_sentence).strip()

            if not claim_text:
                position += len(sentence) + 1
                continue

            # Determine claim type
            claim_type = "numeric" if self._has_numeric_content(claim_text) else "general"

            # Get primary evidence from first citation and collect ALL citation keys
            evidence = None
            evidence_index = None
            citation_key = None
            citation_keys: list[str] | None = None

            if citation_matches:
                first_match = citation_matches[0]

                if reverse_key_map:
                    # Human-readable keys: store ALL keys for multi-marker resolution
                    citation_key = first_match  # Primary key (first match)
                    citation_keys = list(citation_matches)  # ALL keys in this sentence

                    # Get evidence from primary citation
                    evidence_index = reverse_key_map.get(citation_key)
                    if evidence_index is not None and 0 <= evidence_index < len(evidence_pool):
                        evidence = evidence_pool[evidence_index]
                else:
                    # Numeric index: parse directly
                    try:
                        idx = int(first_match)
                        if 0 <= idx < len(evidence_pool):
                            evidence = evidence_pool[idx]
                            evidence_index = idx
                    except ValueError:
                        pass

            # Calculate end position based on actual sentence length
            sentence_end = position + len(sentence.rstrip())

            claim = InterleavedClaim(
                claim_text=claim_text,
                claim_type=claim_type,
                position_start=claim_position_start,
                position_end=sentence_end,
                evidence=evidence,
                evidence_index=evidence_index,
                confidence_score=evidence.relevance_score if evidence else None,
                citation_key=citation_key,
                citation_keys=citation_keys,
            )
            claims.append(claim)

            position += len(sentence) + 1

        return claims

    def _has_numeric_content(self, text: str) -> bool:
        """Check if text contains numeric content.

        Args:
            text: Text to check.

        Returns:
            True if contains numbers/statistics.
        """
        patterns = [
            r"\$[\d,.]+[BMK]?",  # Currency
            r"\d+(?:\.\d+)?%",  # Percentages
            r"\d+(?:,\d{3})+",  # Large numbers
            r"\d+\s*(?:billion|million|thousand)",  # Written numbers
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
