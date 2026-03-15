"""Stage 1: Evidence Pre-Selection for the deep-research framework.

Extracts minimal, relevant evidence spans from source documents BEFORE
generation to enable claim-level citations.  Includes content-quality
filtering (merged from the app's content_evaluator) so low-quality
sources are rejected before LLM extraction runs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)

# -- Prompt ----------------------------------------------------------------

EVIDENCE_PRESELECTION_PROMPT = """\
You are an Evidence Selector for a research synthesis system.
Extract the most relevant, citable text spans from the source below.

## Output Requirements
Identify 5-20 minimal text passages containing:
- Direct facts, statistics, or claims relevant to the query
- Numeric data (prioritize these)
- Expert quotes or authoritative statements
- Key definitions or explanations

Each span must have: quote_text (50-1000 chars), relevance_score (0-1), \
has_numeric (bool), section (heading or null).

## Guidelines
- Extract MINIMAL spans - just the supporting fact
- Boost relevance for spans with numeric data
- Avoid navigation text, boilerplate, disclaimers, or redundancy
- Spans must make sense in isolation

## Query
{query}

## Source: {source_title}
URL: {source_url}

Content:
{source_content}

Respond as JSON: {{"spans": [{{"quote_text": "...", "relevance_score": 0.85, \
"has_numeric": true, "section": "..."}}]}}
Extract the most relevant evidence spans:\
"""

# -- Structured output models ----------------------------------------------

class _SpanOutput(BaseModel):
    quote_text: str = Field(description="Exact quote (50-1000 chars)")
    relevance_score: float = Field(ge=0.0, le=1.0)
    has_numeric: bool = Field(description="Contains numbers/statistics")
    section: str | None = Field(default=None)

class _SpansOutput(BaseModel):
    spans: list[_SpanOutput] = Field(default_factory=list)

# -- Public data models ----------------------------------------------------

@dataclass
class RankedEvidence:
    """An evidence span with relevance ranking."""
    source_url: str
    quote_text: str
    relevance_score: float
    canonical_source_url: str | None = None
    source_title: str | None = None
    source_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    has_numeric_content: bool = False
    source_pool_index: int | None = None
    evidence_pool_index: int | None = None
    is_snippet_based: bool = False

@dataclass
class ContentQuality:
    """Result of content quality evaluation."""
    score: float
    word_count: int
    is_paywall: bool = False
    is_abstract_only: bool = False
    is_navigation_heavy: bool = False
    has_numeric_data: bool = False
    has_specific_facts: bool = False
    reason: str = ""

@dataclass
class EvidenceSelectionConfig:
    """Configuration knobs for the evidence selector."""
    max_spans_per_source: int = 10
    min_span_length: int = 50
    max_span_length: int = 1000
    relevance_threshold: float = 0.3
    numeric_content_boost: float = 0.2
    chunk_size: int = 8_000
    chunk_overlap: int = 1_000
    max_chunks_per_source: int = 5
    quality_min_score: float = 0.5
    model_tier: str | ModelTier = ModelTier.analytical

@dataclass
class EvidenceResult:
    """Return value from evidence pre-selection."""
    evidence: list[RankedEvidence] = field(default_factory=list)
    sources_accepted: int = 0
    sources_rejected: int = 0

# -- Content quality (merged from content_evaluator.py) --------------------

_PAYWALL = [
    "subscribe to read", "sign in to continue", "create an account",
    "login required", "premium content", "members only",
    "subscription required", "please log in", "access denied",
    "register to view", "unlock this article", "start your free trial",
]
_ABSTRACT = [
    "abstract only", "full text not available", "access the full article",
    "view full text", "read the full paper", "download the full pdf",
    "to read the full-text", "available in pdf", "purchase this article",
]
_NAV = [
    "skip to main content", "javascript is required", "cookies policy",
    "privacy policy", "terms of service", "about us", "contact us",
    "advertisement", "sponsored content", "related articles",
    "you might also like", "share this article",
]
_NUM_RE = [
    r"\$[\d,]+(?:\.\d+)?", r"\d+(?:\.\d+)?%", r"\b\d{4}\b",
    r"\b\d+(?:,\d{3})+(?:\.\d+)?\b", r"\b\d+\.\d+\b",
    r"(?:million|billion|trillion|thousand)\s+(?:dollars|euros|pounds)?",
]
_FACT_RE = [
    r'"[^"]{20,200}"', r"according to\s+[\w\s]+,",
    r"(?:reported|announced|stated|said)\s+that",
    r"the\s+(?:study|research|report|survey)\s+(?:found|shows|indicates)",
]

def evaluate_content_quality(content: str, _query: str = "") -> ContentQuality:
    """Heuristic quality scoring: detects paywalls, abstracts, nav-heavy pages."""
    if not content or len(content.strip()) < 100:
        return ContentQuality(score=0.0, word_count=0, reason="Empty or minimal content")
    lower = content.lower()
    wc = len(content.split())
    is_pw = sum(1 for p in _PAYWALL if p in lower) >= 2
    is_ab = sum(1 for p in _ABSTRACT if p in lower) >= 2
    is_nv = sum(1 for p in _NAV if p in lower) >= 4 or (wc < 300 and sum(1 for p in _NAV if p in lower) >= 2)
    has_num = sum(len(re.findall(p, content, re.I)) for p in _NUM_RE) >= 3
    has_fact = sum(len(re.findall(p, content, re.I)) for p in _FACT_RE) >= 2 or wc > 500
    s = 0.5
    if is_pw:
        s -= 0.4
    if is_ab:
        s -= 0.3
    if is_nv:
        s -= 0.2
    if wc < 200:
        s -= 0.2
    if has_num:
        s += 0.2
    if has_fact:
        s += 0.15
    if wc > 500:
        s += 0.1
    if wc > 1000:
        s += 0.1
    s = max(0.0, min(1.0, s))
    reason = ("Paywall detected" if is_pw else "Abstract only" if is_ab
              else "Navigation heavy" if is_nv else f"Insufficient ({wc} words)" if wc < 200
              else "High quality" if s >= 0.7 else "Acceptable" if s >= 0.5 else "Low quality")
    return ContentQuality(score=s, word_count=wc, is_paywall=is_pw, is_abstract_only=is_ab,
                          is_navigation_heavy=is_nv, has_numeric_data=has_num,
                          has_specific_facts=has_fact, reason=reason)

def filter_high_quality_sources(
    sources: list[dict[str, Any]], query: str = "", min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Return only sources whose content passes quality checks."""
    out: list[dict[str, Any]] = []
    for src in sources:
        c = src.get("content", "")
        if not c:
            continue
        q = evaluate_content_quality(c, query)
        if q.score >= min_score and not q.is_abstract_only:
            out.append(src)
        else:
            logger.debug("SOURCE_REJECTED url=%s score=%.2f reason=%s",
                         str(src.get("url", ""))[:60], q.score, q.reason)
    logger.info("QUALITY_FILTER total=%d accepted=%d rejected=%d",
                len(sources), len(out), len(sources) - len(out))
    return out

# -- Chunking helpers ------------------------------------------------------

@dataclass
class _Chunk:
    text: str
    start: int
    end: int
    index: int

def _chunk_content(content: str, size: int = 8000, overlap: int = 1000) -> list[_Chunk]:
    if len(content) <= size:
        return [_Chunk(content, 0, len(content), 0)]
    chunks: list[_Chunk] = []
    pos, idx = 0, 0
    while pos < len(content):
        end = min(pos + size, len(content))
        if end < len(content):
            for sep in [". ", ".\n", "! ", "? "]:
                p = content[end - 200 : end].rfind(sep)
                if p != -1:
                    end = end - 200 + p + len(sep)
                    break
        chunks.append(_Chunk(content[pos:end], pos, end, idx))
        idx += 1
        pos = end - overlap
        if pos >= len(content) - overlap:
            break
    return chunks

def _merge_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for sp in sorted(spans, key=lambda x: x.get("relevance_score", 0), reverse=True):
        q = sp.get("quote_text", "").strip().lower()
        if not q or q in seen or any(q in s or s in q for s in seen):
            continue
        out.append(sp)
        seen.add(q)
    return out

# -- Keyword helpers -------------------------------------------------------

def _keyword_relevance(query: str, text: str) -> float:
    ql, tl = query.lower(), text.lower()
    terms = set(re.findall(r"\b\w{3,}\b", ql))
    if not terms:
        return 0.0
    score = sum(1 for t in terms if t in tl) / len(terms)
    if ql in tl:
        score = min(1.0, score + 0.3)
    return score

def _has_numeric(text: str) -> bool:
    pats = [r"\$[\d,.]+[BMK]?", r"\d+(?:\.\d+)?%", r"\d{4}",
            r"\d+(?:,\d{3})+", r"\d+\s*(?:billion|million|thousand)"]
    return any(re.search(p, text, re.I) for p in pats)

# -- EvidenceSelector ------------------------------------------------------

class EvidenceSelector:
    """Stage 1: extract evidence spans via LLM with heuristic fallback."""

    def __init__(self, llm_client: FrameworkLLMClient,
                 config: EvidenceSelectionConfig | None = None) -> None:
        self._llm = llm_client
        self._cfg = config or EvidenceSelectionConfig()

    async def select_evidence(
        self, query: str, sources: list[dict[str, Any]], *,
        max_spans_per_source: int | None = None, filter_quality: bool = True,
    ) -> EvidenceResult:
        """Select evidence spans from *sources* relevant to *query*.

        Args:
            query: Research question.
            sources: Dicts with keys url, title, content (optional snippet, id).
            max_spans_per_source: Override per-source span cap.
            filter_quality: Pre-filter low-quality sources.

        Returns:
            EvidenceResult with ranked evidence and filter statistics.
        """
        cap = max_spans_per_source or self._cfg.max_spans_per_source
        if filter_quality:
            good = filter_high_quality_sources(sources, query, self._cfg.quality_min_score)
            rej = len(sources) - len(good)
        else:
            good, rej = sources, 0

        pool: list[RankedEvidence] = []
        for src in good:
            url = src.get("url", "")
            canonical_url = src.get("canonical_url") or url
            title = src.get("title")
            content = src.get("content", "")
            snippet = src.get("snippet", "")
            sid = src.get("id")
            source_pool_index = src.get("source_pool_index")

            if not content and snippet:
                score = _keyword_relevance(query, snippet) if query else 0.5
                has_num = _has_numeric(snippet)
                if has_num:
                    score = min(1.0, score + self._cfg.numeric_content_boost)
                score = max(0.3, score)  # floor: snippet presence signals relevance
                pool.append(RankedEvidence(
                    source_url=url,
                    canonical_source_url=canonical_url,
                    source_title=title,
                    source_id=sid,
                    quote_text=snippet,
                    relevance_score=score,
                    has_numeric_content=has_num,
                    source_pool_index=source_pool_index,
                    is_snippet_based=True,
                ))
                continue
            if not content:
                continue

            try:
                spans = await self._extract_from_source(query, url, title or "Unknown", content)
                for sp in spans[:cap]:
                    pool.append(RankedEvidence(
                        source_url=url, canonical_source_url=canonical_url,
                        source_title=title, source_id=sid,
                        quote_text=sp.get("quote_text", ""),
                        relevance_score=sp.get("relevance_score", 0.5),
                        section_heading=sp.get("section"),
                        has_numeric_content=sp.get("has_numeric", False),
                        source_pool_index=source_pool_index,
                    ))
            except Exception:
                logger.warning("LLM extraction failed for %s, heuristic fallback",
                               url[:60], exc_info=True)
                pool.extend(
                    self._heuristic_extract(
                        query,
                        content,
                        url,
                        canonical_url,
                        title,
                        sid,
                        source_pool_index,
                    )[:cap]
                )

        thr = self._cfg.relevance_threshold
        result = sorted([e for e in pool if e.relevance_score >= thr],
                        key=lambda e: e.relevance_score, reverse=True)
        logger.info("EVIDENCE_SELECTED spans=%d sources=%d rejected=%d",
                     len(result), len(good), rej)
        return EvidenceResult(evidence=result, sources_accepted=len(good), sources_rejected=rej)

    # -- LLM extraction ----------------------------------------------------

    async def _extract_from_source(self, query: str, url: str,
                                   title: str, content: str) -> list[dict[str, Any]]:
        cfg = self._cfg
        if len(content) <= cfg.chunk_size:
            return await self._llm_extract(query, url, title, content)
        chunks = _chunk_content(content, cfg.chunk_size, cfg.chunk_overlap)
        mc = cfg.max_chunks_per_source
        logger.info("Processing %d/%d chunks for %s", min(len(chunks), mc), len(chunks), url[:60])
        all_sp: list[dict[str, Any]] = []
        for ch in chunks[:mc]:
            t = f"{title} (chunk {ch.index + 1}/{min(len(chunks), mc)})"
            all_sp.extend(await self._llm_extract(query, url, t, ch.text))
        return _merge_spans(all_sp)

    async def _llm_extract(self, query: str, url: str,
                           title: str, content: str) -> list[dict[str, Any]]:
        prompt = EVIDENCE_PRESELECTION_PROMPT.format(
            query=query, source_url=url, source_title=title, source_content=content)
        resp = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=self._cfg.model_tier, structured_output=_SpansOutput)
        if resp.structured is not None:
            out: _SpansOutput = resp.structured
            return [{"quote_text": s.quote_text, "relevance_score": s.relevance_score,
                     "has_numeric": s.has_numeric, "section": s.section} for s in out.spans]
        logger.warning("Structured output unavailable; returning empty spans")
        return []

    # -- Heuristic fallback ------------------------------------------------

    def _heuristic_extract(
        self,
        query: str,
        content: str,
        url: str,
        canonical_url: str,
        title: str | None,
        sid: str | None,
        source_pool_index: int | None,
    ) -> list[RankedEvidence]:
        ev: list[RankedEvidence] = []
        for sp in self._segment(content):
            txt = sp["text"]
            score = _keyword_relevance(query, txt)
            num = _has_numeric(txt)
            if num:
                score = min(1.0, score + self._cfg.numeric_content_boost)
            if score >= self._cfg.relevance_threshold:
                ev.append(RankedEvidence(
                    source_url=url,
                    canonical_source_url=canonical_url,
                    source_title=title,
                    source_id=sid,
                    quote_text=txt,
                    start_offset=sp["start"],
                    end_offset=sp["end"],
                    relevance_score=score,
                    has_numeric_content=num,
                    source_pool_index=source_pool_index,
                ))
        return ev

    def _segment(self, content: str) -> list[dict[str, Any]]:
        mn, mx = self._cfg.min_span_length, self._cfg.max_span_length
        spans: list[dict[str, Any]] = []
        off = 0
        for para in re.split(r"\n\s*\n", content):
            para = para.strip()
            if not para:
                off += 2
                continue
            if len(para) > mx:
                cur, cs = "", off
                for sent in re.split(r"(?<=[.!?])\s+", para):
                    if len(cur) + len(sent) <= mx:
                        cur += (" " if cur else "") + sent
                    else:
                        if len(cur) >= mn:
                            spans.append({"text": cur.strip(), "start": cs, "end": cs + len(cur)})
                        cs = off + para.find(sent)
                        cur = sent
                if len(cur) >= mn:
                    spans.append({"text": cur.strip(), "start": cs, "end": cs + len(cur)})
            elif len(para) >= mn:
                spans.append({"text": para, "start": off, "end": off + len(para)})
            off += len(para) + 2
        return spans

# -- Convenience entry point -----------------------------------------------

async def select_evidence(
    query: str, sources: list[dict[str, Any]], llm_client: FrameworkLLMClient, *,
    config: EvidenceSelectionConfig | None = None,
    max_spans_per_source: int | None = None, filter_quality: bool = True,
) -> EvidenceResult:
    """One-call wrapper: create an EvidenceSelector, run it, return results."""
    return await EvidenceSelector(llm_client, config).select_evidence(
        query, sources, max_spans_per_source=max_spans_per_source,
        filter_quality=filter_quality)
