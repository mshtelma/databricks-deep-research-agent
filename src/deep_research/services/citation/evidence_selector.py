"""Stage 1: Evidence Pre-Selection service.

Extracts minimal, relevant evidence spans from source documents
BEFORE generation to enable claim-level citations.

Supports optional embedding computation for hybrid search in ReAct synthesis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

from deep_research.agent.prompts.citation.evidence_selection import (
    EVIDENCE_PRESELECTION_PROMPT,
)
from deep_research.core.app_config import get_app_config
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from deep_research.services.llm.embedder import GteEmbedder

logger = logging.getLogger(__name__)


# Pydantic models for structured LLM output
class EvidenceSpanOutput(BaseModel):
    """A single evidence span from LLM extraction."""

    quote_text: str = Field(description="Exact quote from source (50-500 chars)")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance 0.0-1.0")
    has_numeric: bool = Field(description="True if contains numbers/statistics")
    section: str | None = Field(default=None, description="Section heading if identifiable")


class EvidenceSpansOutput(BaseModel):
    """Output from evidence extraction LLM call."""

    spans: list[EvidenceSpanOutput] = Field(
        default_factory=list,
        description="List of extracted evidence spans"
    )


@dataclass
class RankedEvidence:
    """An evidence span with relevance ranking."""

    source_id: UUID | None
    source_url: str
    source_title: str | None
    quote_text: str
    start_offset: int | None
    end_offset: int | None
    section_heading: str | None
    relevance_score: float
    has_numeric_content: bool
    is_snippet_based: bool = False  # True if from Brave Search snippet (lower confidence)


@dataclass
class ContentChunk:
    """A chunk of source content for processing long documents."""

    text: str
    start_offset: int
    end_offset: int
    chunk_index: int


def chunk_content(
    content: str,
    chunk_size: int = 8000,
    overlap: int = 1000,
) -> list[ContentChunk]:
    """Split content into overlapping chunks for processing long documents.

    Args:
        content: Full source content to chunk.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of ContentChunk objects.
    """
    if len(content) <= chunk_size:
        return [ContentChunk(content, 0, len(content), 0)]

    chunks = []
    start = 0
    idx = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        # Try to break at sentence boundary if not at end
        if end < len(content):
            for sep in [". ", ".\n", "! ", "? "]:
                pos = content[end - 200 : end].rfind(sep)
                if pos != -1:
                    end = end - 200 + pos + len(sep)
                    break

        chunks.append(ContentChunk(content[start:end], start, end, idx))
        idx += 1
        start = end - overlap

        if start >= len(content) - overlap:
            break

    return chunks


def merge_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate spans from multiple chunks.

    Removes duplicate or overlapping spans, keeping highest relevance.

    Args:
        spans: List of span dicts from multiple chunks.

    Returns:
        Deduplicated list of spans sorted by relevance.
    """
    if not spans:
        return []

    seen: set[str] = set()
    unique = []

    # Sort by relevance descending to keep best first
    for span in sorted(spans, key=lambda x: x.get("relevance_score", 0), reverse=True):
        quote = span.get("quote_text", "").strip().lower()

        # Skip if this is a duplicate or subset of an existing span
        if quote in seen:
            continue

        # Check for containment (either direction)
        is_contained = any(quote in s or s in quote for s in seen if s)

        if not is_contained and quote:
            unique.append(span)
            seen.add(quote)

    return unique


class EvidencePreSelector:
    """Stage 1: Evidence Pre-Selection.

    Extracts minimal, relevant evidence spans from source documents
    using a hybrid keyword + semantic scoring approach.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize evidence pre-selector.

        Args:
            llm_client: LLM client for evidence extraction.
        """
        self._llm = llm_client
        self._config = get_app_config().citation_verification.evidence_preselection

    def segment_into_spans(
        self,
        content: str,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> list[dict[str, Any]]:
        """Segment source content into candidate spans.

        Uses sentence boundaries and paragraph breaks to identify
        natural segmentation points.

        Args:
            content: Raw source content.
            min_length: Minimum span length (default from config).
            max_length: Maximum span length (default from config).

        Returns:
            List of span dicts with text and offsets.
        """
        min_len = min_length or self._config.min_span_length
        max_len = max_length or self._config.max_span_length

        spans = []

        # Split by paragraphs first
        paragraphs = re.split(r"\n\s*\n", content)

        offset = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                offset += 2  # Account for paragraph break
                continue

            # For long paragraphs, split by sentences
            if len(para) > max_len:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_span = ""
                span_start = offset

                for sentence in sentences:
                    if len(current_span) + len(sentence) <= max_len:
                        current_span += (" " if current_span else "") + sentence
                    else:
                        if len(current_span) >= min_len:
                            spans.append(
                                {
                                    "text": current_span.strip(),
                                    "start_offset": span_start,
                                    "end_offset": span_start + len(current_span),
                                }
                            )
                        span_start = offset + para.find(sentence)
                        current_span = sentence

                # Add remaining span
                if len(current_span) >= min_len:
                    spans.append(
                        {
                            "text": current_span.strip(),
                            "start_offset": span_start,
                            "end_offset": span_start + len(current_span),
                        }
                    )
            elif len(para) >= min_len:
                spans.append(
                    {
                        "text": para,
                        "start_offset": offset,
                        "end_offset": offset + len(para),
                    }
                )

            offset += len(para) + 2  # +2 for paragraph break

        return spans

    def compute_relevance(
        self,
        query: str,
        span_text: str,
    ) -> float:
        """Compute relevance score for a span.

        Uses hybrid keyword + semantic scoring based on config.

        Args:
            query: Research query.
            span_text: Evidence span text.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        method = self._config.relevance_computation_method

        keyword_score = self._keyword_relevance(query, span_text)
        has_numeric = self._detect_numeric_content(span_text)

        if method == "keyword":
            score = keyword_score
        elif method == "semantic":
            # For now, use keyword as fallback (semantic would use embeddings)
            score = keyword_score
        else:  # hybrid (default)
            # Combine keyword and semantic (semantic is stubbed for now)
            score = keyword_score

        # Apply numeric content boost
        if has_numeric:
            score = min(1.0, score + self._config.numeric_content_boost)

        return score

    def _keyword_relevance(self, query: str, span_text: str) -> float:
        """Compute keyword-based relevance score.

        Args:
            query: Research query.
            span_text: Evidence span text.

        Returns:
            Keyword relevance score between 0.0 and 1.0.
        """
        # Normalize text
        query_lower = query.lower()
        span_lower = span_text.lower()

        # Extract query terms (simple tokenization)
        query_terms = set(re.findall(r"\b\w{3,}\b", query_lower))

        if not query_terms:
            return 0.0

        # Count matching terms
        matches = sum(1 for term in query_terms if term in span_lower)

        # Base score from term overlap
        base_score = matches / len(query_terms)

        # Boost for exact phrase matches
        if query_lower in span_lower:
            base_score = min(1.0, base_score + 0.3)

        return base_score

    def _detect_numeric_content(self, text: str) -> bool:
        """Detect if text contains numeric content.

        Args:
            text: Text to check.

        Returns:
            True if text contains numbers/statistics.
        """
        # Match various numeric patterns
        patterns = [
            r"\$[\d,.]+[BMK]?",  # Currency
            r"\d+(?:\.\d+)?%",  # Percentages
            r"\d{4}",  # Years
            r"\d+(?:,\d{3})+",  # Large numbers with commas
            r"\d+\s*(?:billion|million|thousand)",  # Written numbers
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    async def select_evidence_spans(
        self,
        query: str,
        sources: list[dict[str, Any]],
        max_spans_per_source: int | None = None,
    ) -> list[RankedEvidence]:
        """Select evidence spans from sources.

        Main entry point for Stage 1 evidence pre-selection.
        For long sources, uses chunked processing to avoid truncation.

        Args:
            query: Research query.
            sources: List of source dicts with url, title, content.
            max_spans_per_source: Max spans per source (default from config).

        Returns:
            List of ranked evidence spans sorted by relevance.
        """
        max_spans = max_spans_per_source or self._config.max_spans_per_source

        # Get chunk config with defaults
        chunk_size = getattr(self._config, "chunk_size", 8000)
        chunk_overlap = getattr(self._config, "chunk_overlap", 1000)
        max_chunks = getattr(self._config, "max_chunks_per_source", 5)

        all_evidence: list[RankedEvidence] = []

        for source in sources:
            source_url = source.get("url", "")
            source_title = source.get("title")
            source_content = source.get("content", "")
            source_snippet = source.get("snippet", "")
            source_id = source.get("id")

            # Use snippet as fallback when content is not available
            # This handles cases where web fetch failed or was not performed
            if not source_content and source_snippet:
                # Create evidence directly from snippet (no LLM extraction needed)
                evidence = RankedEvidence(
                    source_id=source_id,
                    source_url=source_url,
                    source_title=source_title,
                    quote_text=source_snippet,
                    start_offset=None,
                    end_offset=None,
                    section_heading=None,
                    relevance_score=0.5,  # Lower confidence for snippet-based evidence
                    has_numeric_content=self._detect_numeric_content(source_snippet),
                    is_snippet_based=True,
                )
                all_evidence.append(evidence)
                continue

            if not source_content:
                continue

            try:
                all_spans: list[dict[str, Any]] = []

                # Use chunked processing for long sources
                if len(source_content) > chunk_size:
                    chunks = chunk_content(
                        source_content,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap,
                    )

                    logger.info(
                        f"Processing {min(len(chunks), max_chunks)} chunks "
                        f"for source: {source_url[:50]}"
                    )

                    # Process each chunk (up to max_chunks)
                    for chunk in chunks[:max_chunks]:
                        chunk_spans = await self._extract_spans_with_llm(
                            query=query,
                            source_url=source_url,
                            source_title=f"{source_title or 'Unknown'} "
                            f"(chunk {chunk.chunk_index + 1}/{min(len(chunks), max_chunks)})",
                            source_content=chunk.text,
                        )
                        all_spans.extend(chunk_spans)

                    # Merge and deduplicate spans from all chunks
                    all_spans = merge_spans(all_spans)
                else:
                    # Small source - process directly
                    all_spans = await self._extract_spans_with_llm(
                        query=query,
                        source_url=source_url,
                        source_title=source_title or "Unknown",
                        source_content=source_content,
                    )

                # Convert to RankedEvidence
                for span in all_spans[:max_spans]:
                    evidence = RankedEvidence(
                        source_id=source_id,
                        source_url=source_url,
                        source_title=source_title,
                        quote_text=span.get("quote_text", ""),
                        start_offset=None,  # LLM extraction doesn't provide offsets
                        end_offset=None,
                        section_heading=span.get("section"),
                        relevance_score=span.get("relevance_score", 0.5),
                        has_numeric_content=span.get("has_numeric", False),
                    )
                    all_evidence.append(evidence)

            except Exception as e:
                logger.warning(
                    f"Failed to extract evidence from {source_url}: {e}"
                )
                # Fallback to heuristic extraction
                fallback_spans = self._heuristic_extract(
                    query, source_content, source_url, source_title, source_id
                )
                all_evidence.extend(fallback_spans[:max_spans])

        # Sort by relevance and filter by threshold
        threshold = self._config.relevance_threshold
        filtered = [e for e in all_evidence if e.relevance_score >= threshold]
        filtered.sort(key=lambda e: e.relevance_score, reverse=True)

        logger.info(
            f"Selected {len(filtered)} evidence spans from {len(sources)} sources"
        )
        return filtered

    async def _extract_spans_with_llm(
        self,
        query: str,
        source_url: str,
        source_title: str,
        source_content: str,
    ) -> list[dict[str, Any]]:
        """Extract evidence spans using LLM with structured generation.

        Args:
            query: Research query.
            source_url: Source URL.
            source_title: Source title.
            source_content: Source content.

        Returns:
            List of span dicts from LLM.
        """
        prompt = EVIDENCE_PRESELECTION_PROMPT.format(
            query=query,
            source_url=source_url,
            source_title=source_title,
            source_content=source_content,
        )

        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.BULK_ANALYSIS,  # Use Gemini for large-context extraction
            structured_output=EvidenceSpansOutput,  # Use structured generation
        )

        # Extract structured output
        if response.structured:
            output: EvidenceSpansOutput = response.structured
            return [
                {
                    "quote_text": span.quote_text,
                    "relevance_score": span.relevance_score,
                    "has_numeric": span.has_numeric,
                    "section": span.section,
                }
                for span in output.spans
            ]

        logger.warning("Structured output not available in response")
        return []

    def _heuristic_extract(
        self,
        query: str,
        content: str,
        source_url: str,
        source_title: str | None,
        source_id: UUID | None,
    ) -> list[RankedEvidence]:
        """Fallback heuristic extraction.

        Args:
            query: Research query.
            content: Source content.
            source_url: Source URL.
            source_title: Source title.
            source_id: Source ID if available.

        Returns:
            List of extracted evidence spans.
        """
        spans = self.segment_into_spans(content)
        evidence = []

        for span in spans:
            text = span["text"]
            relevance = self.compute_relevance(query, text)
            has_numeric = self._detect_numeric_content(text)

            if relevance >= self._config.relevance_threshold:
                evidence.append(
                    RankedEvidence(
                        source_id=source_id,
                        source_url=source_url,
                        source_title=source_title,
                        quote_text=text,
                        start_offset=span["start_offset"],
                        end_offset=span["end_offset"],
                        section_heading=None,
                        relevance_score=relevance,
                        has_numeric_content=has_numeric,
                    )
                )

        return evidence


async def compute_evidence_embeddings(
    evidence_pool: list[RankedEvidence],
    embedder: GteEmbedder,
) -> NDArray[np.float32]:
    """Compute embeddings for all evidence spans.

    Used for hybrid search in ReAct synthesis. Computes embeddings
    in batches for efficiency.

    Args:
        evidence_pool: List of RankedEvidence from pre-selection.
        embedder: GTE embedder client.

    Returns:
        2D numpy array of shape (n_evidence, embed_dim).
    """
    if not evidence_pool:
        return np.array([], dtype=np.float32).reshape(0, 0)

    texts = [ev.quote_text for ev in evidence_pool]

    logger.info(f"Computing embeddings for {len(texts)} evidence spans")

    embeddings = await embedder.embed_batch(texts)

    logger.info(
        f"Computed embeddings: shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}"
    )

    return embeddings


@dataclass
class EvidenceWithEmbeddings:
    """Evidence pool with pre-computed embeddings for hybrid search."""

    evidence: list[RankedEvidence]
    embeddings: NDArray[np.float32]

    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are available."""
        return self.embeddings.size > 0


async def preselect_evidence_with_embeddings(
    query: str,
    sources: list[dict[str, Any]],
    llm_client: LLMClient,
    embedder: GteEmbedder | None = None,
    max_spans_per_source: int | None = None,
) -> EvidenceWithEmbeddings:
    """Pre-select evidence and optionally compute embeddings.

    Convenience function that combines evidence pre-selection and
    embedding computation for ReAct synthesis.

    Args:
        query: Research query.
        sources: List of source dicts with url, title, content.
        llm_client: LLM client for evidence extraction.
        embedder: Optional GTE embedder for hybrid search.
        max_spans_per_source: Max spans per source.

    Returns:
        EvidenceWithEmbeddings containing evidence and optional embeddings.
    """
    selector = EvidencePreSelector(llm_client)
    evidence = await selector.select_evidence_spans(
        query=query,
        sources=sources,
        max_spans_per_source=max_spans_per_source,
    )

    if embedder is not None and evidence:
        embeddings = await compute_evidence_embeddings(evidence, embedder)
    else:
        embeddings = np.array([], dtype=np.float32).reshape(0, 0)

    return EvidenceWithEmbeddings(evidence=evidence, embeddings=embeddings)
