"""LLM-powered query optimization for vector search tools.

Replaces regex-based VectorIndexQueryPolicy with a 3-stage pipeline:
  Stage 1: Multi-query generation (LLM)
  Stage 2: Parallel VS retrieval + Reciprocal Rank Fusion
  Stage 3: LLM reranking (optional)

Each stage has graceful fallback: if it fails, the pipeline continues
with the previous stage's output.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.json_parsing import parse_llm_json
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizedQueries:
    """Output of Stage 1: LLM query generation."""

    primary_query: str
    alternate_queries: list[str]


@dataclass(frozen=True)
class VSOptimizationConfig:
    """Validated config extracted from tool metadata."""

    multi_query: bool = True
    hyde: bool = True
    rerank: bool = True
    num_alternatives: int = 3
    rerank_threshold: int = 3
    rrf_k: int = 60

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> VSOptimizationConfig:
        """Validate and clamp config values."""

        def _int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(hi, int(metadata.get(key, default))))
            except (TypeError, ValueError):
                return default

        return cls(
            multi_query=bool(metadata.get("multi_query", True)),
            hyde=bool(metadata.get("hyde", True)),
            rerank=bool(metadata.get("rerank", True)),
            num_alternatives=_int("num_alternatives", 3, 1, 5),
            rerank_threshold=_int("rerank_threshold", 3, 0, 10),
            rrf_k=_int("rrf_k", 60, 1, 100),
        )


class VectorQueryOptimizer:
    """Stateless, reentrant optimizer. Safe for concurrent use."""

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def optimize_and_execute(
        self,
        tool: Any,  # ResearchTool protocol
        original_query: str,
        tool_context: ToolContext,
    ) -> tuple[ToolResult, dict[str, Any]]:
        """Run 3-stage pipeline. Returns (result, trace_metadata).

        trace_metadata contains all intermediate data for observability.
        """
        cfg = VSOptimizationConfig.from_metadata(tool.definition.metadata or {})
        trace: dict[str, Any] = {
            "original_query": original_query,
            "config": cfg.__dict__,
        }
        t0 = time.monotonic()

        # -- Stage 1: LLM query generation --
        try:
            optimized = await self._generate_queries(
                original_query, tool.definition, tool_context, cfg
            )
            trace["generated_queries"] = (
                [optimized.primary_query] + optimized.alternate_queries
            )
        except Exception:
            logger.exception(
                "VS_QUERY_GEN_FAILED tool=%s, falling back to original query",
                tool.definition.name,
            )
            optimized = OptimizedQueries(
                primary_query=original_query, alternate_queries=[]
            )
            trace["generated_queries"] = [original_query]
            trace["query_gen_fallback"] = True

        trace["stage1_ms"] = int((time.monotonic() - t0) * 1000)

        # -- Stage 2: Parallel retrieval + RRF merge --
        t1 = time.monotonic()
        all_queries = [optimized.primary_query]
        if cfg.multi_query:
            all_queries += optimized.alternate_queries

        # Deduplicate identical queries
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in all_queries:
            normalized = q.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(q)
        trace["unique_query_count"] = len(unique_queries)

        # Execute in parallel with per-call error handling
        async def _safe_execute(query: str) -> ToolResult | None:
            try:
                args = tool.validate_arguments({"query": query, "num_results": 10})
                result: ToolResult = await tool.execute(args, tool_context)
                return result
            except Exception:
                logger.exception(
                    "VS_CALL_FAILED tool=%s query=%s",
                    tool.definition.name,
                    query[:100],
                )
                return None

        raw_results = await asyncio.gather(
            *[_safe_execute(q) for q in unique_queries]
        )
        results = [r for r in raw_results if r is not None and r.success]
        trace["vs_calls_total"] = len(unique_queries)
        trace["vs_calls_success"] = len(results)

        if not results:
            trace["stage2_ms"] = int((time.monotonic() - t1) * 1000)
            trace["total_ms"] = int((time.monotonic() - t0) * 1000)
            return (
                ToolResult(content="No results found.", success=True, sources=[], data={}),
                trace,
            )

        merged = self._rrf_merge(results, k=cfg.rrf_k) if len(results) > 1 else results[0]

        trace["rrf_result_count"] = len(merged.sources)
        trace["stage2_ms"] = int((time.monotonic() - t1) * 1000)

        # -- Stage 3: LLM reranking (optional) --
        t2 = time.monotonic()
        if cfg.rerank and merged.sources:
            try:
                reranked_sources = await self._llm_rerank(
                    original_query, merged.sources[:15], tool.definition, cfg
                )
                trace["rerank_input"] = len(merged.sources[:15])
                trace["rerank_output"] = len(reranked_sources)
            except Exception:
                logger.exception(
                    "VS_RERANK_FAILED tool=%s, using un-reranked results",
                    tool.definition.name,
                )
                reranked_sources = merged.sources[:10]
                trace["rerank_fallback"] = True
        else:
            reranked_sources = merged.sources[:10]

        trace["stage3_ms"] = int((time.monotonic() - t2) * 1000)
        trace["total_ms"] = int((time.monotonic() - t0) * 1000)
        trace["final_result_count"] = len(reranked_sources)

        # Build strategy label from active stages
        parts = ["llm"]
        if cfg.multi_query and len(unique_queries) > 1:
            parts.append("multi_query")
            if cfg.hyde:
                parts.append("hyde")
            parts.append("rrf")
        if cfg.rerank:
            parts.append("rerank")
        trace["strategy"] = "_".join(parts)

        # Build final ToolResult
        content_lines = []
        for idx, src in enumerate(reranked_sources):
            text = (src.snippet or src.content or "")[:300]
            content_lines.append(
                f"- Vector search result {idx + 1}: "
                f"chunk_id: {src.title}; chunk_content: {text}"
            )
        content = "\n".join(content_lines) or "No results found."

        final = ToolResult(
            content=content,
            success=True,
            sources=reranked_sources,
            data={
                "result_count": len(reranked_sources),
                "source_kind": "vector_index",
                "strategy": trace["strategy"],
            },
        )
        return final, trace

    # -- Stage 1: Query Generation --

    async def _generate_queries(
        self,
        original_query: str,
        definition: ToolDefinition,
        context: ToolContext,
        cfg: VSOptimizationConfig,
    ) -> OptimizedQueries:
        """Generate optimized queries using LLM.

        Generic -- uses tool description as sole domain signal.
        """
        recent = (context.recent_observations or [])[-3:]
        recent_text = (
            "\n".join(str(o)[:200] for o in recent) if recent else "None yet."
        )

        num_alts = cfg.num_alternatives if cfg.multi_query else 0
        hyde_instruction = ""
        if cfg.hyde and num_alts > 0:
            hyde_instruction = (
                "\n- Make the LAST alternate query a hypothetical document excerpt: "
                "a 2-3 sentence passage written AS IF it were an actual chunk from "
                "this knowledge base that perfectly answers the query. This helps "
                "bridge the embedding gap between questions and stored documents."
            )

        alt_instruction = ""
        if num_alts > 0:
            alt_instruction = (
                f"Also generate {num_alts} alternate queries from different angles."
            )

        prompt = f"""Generate optimized vector search queries for a knowledge base.

## Knowledge Base Description
{definition.description}

## Original Query
{original_query}

## Research Context
Root topic: {context.query}
Recent findings: {recent_text}

## Task
Rewrite the query to maximize retrieval quality from the knowledge base above.
{alt_instruction}

Key principles:
- Use the VOCABULARY of the target knowledge base, not conversational language
- Infer document terminology from the knowledge base description above
- Each alternate query should cover a genuinely different retrieval angle{hyde_instruction}

Return ONLY valid JSON (no markdown, no commentary):
{{"primary_query": "...", "alternate_queries": [{', '.join(['"..."'] * num_alts)}]}}"""

        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier="simple",
            temperature=0.3,
            max_tokens=512,
        )

        parsed = self._parse_json_response(response)
        primary = str(parsed.get("primary_query", original_query)).strip()
        alternates = [
            str(q).strip()
            for q in parsed.get("alternate_queries", [])
            if str(q).strip()
        ]

        if not primary:
            primary = original_query

        return OptimizedQueries(
            primary_query=primary,
            alternate_queries=alternates[: cfg.num_alternatives],
        )

    # -- Stage 2: RRF Merge --

    def _rrf_merge(self, results: list[ToolResult], k: int = 60) -> ToolResult:
        """Reciprocal Rank Fusion. Deduplicates by content hash, not URL."""
        scores: dict[str, float] = {}
        source_map: dict[str, SourceInfo] = {}

        for result in results:
            for rank, source in enumerate(result.sources):
                key = self._source_dedup_key(source)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                if key not in source_map:
                    source_map[key] = source

        sorted_keys = sorted(scores, key=lambda sk: scores[sk], reverse=True)
        merged_sources = [source_map[sk] for sk in sorted_keys[:15]]

        content_lines = []
        for idx, src in enumerate(merged_sources):
            text = (src.snippet or src.content or "")[:300]
            content_lines.append(f"[{idx + 1}] {src.title}: {text}")

        return ToolResult(
            content="\n".join(content_lines) or "No results found.",
            success=True,
            sources=merged_sources,
            data={"result_count": len(merged_sources)},
        )

    @staticmethod
    def _source_dedup_key(source: SourceInfo) -> str:
        """Stable dedup key: hash of content (or fallback to title + snippet)."""
        text = source.content or source.snippet or ""
        if text:
            return hashlib.sha256(text.encode()).hexdigest()
        return hashlib.sha256(f"{source.title}|{source.snippet}".encode()).hexdigest()

    # -- Stage 3: LLM Reranking --

    async def _llm_rerank(
        self,
        original_query: str,
        candidates: list[SourceInfo],
        definition: ToolDefinition,
        cfg: VSOptimizationConfig,
    ) -> list[SourceInfo]:
        """Score each candidate's relevance via LLM, filter and reorder."""
        if not candidates:
            return []

        doc_entries = []
        for i, src in enumerate(candidates):
            content = (src.content or src.snippet or "")[:400]
            doc_entries.append(f"[{i}] {content}")

        prompt = f"""Rate each document's relevance to the query (0-10 scale).

Query: {original_query}
Knowledge base: {definition.description[:200]}

Documents:
{chr(10).join(doc_entries)}

Return ONLY valid JSON: {{"scores": [s0, s1, ...]}}"""

        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier="simple",
            temperature=0.0,
            max_tokens=256,
        )

        parsed = self._parse_json_response(response)
        raw_scores = parsed.get("scores", [])

        scores: list[float] = []
        for i, _candidate in enumerate(candidates):
            if i < len(raw_scores):
                try:
                    score = float(raw_scores[i])
                    scores.append(max(0.0, min(10.0, score)))
                except (TypeError, ValueError):
                    scores.append(5.0)
            else:
                scores.append(5.0)

        scored = sorted(zip(candidates, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [src for src, score in scored if score >= cfg.rerank_threshold][:10]

    # -- Utilities --

    @staticmethod
    def _parse_json_response(response: Any) -> dict[str, Any]:
        """Robust JSON extraction from LLM response."""
        text = ""
        if hasattr(response, "content"):
            text = str(response.content)
        elif hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            text = str(getattr(msg, "content", ""))
        else:
            text = str(response)

        parsed, _ = parse_llm_json(text, default={}, site="vector_query_optimizer")
        if not parsed:
            logger.warning("VS_JSON_PARSE_FAILED text=%s", text[:300])
            return {}
        result: dict[str, Any] = parsed
        return result
