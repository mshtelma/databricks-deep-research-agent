"""Generic ReAct execution loop.

Manages LLM conversation state with tool calling iteration.
Reused by any agent that needs tool calling (researcher, synthesizer, etc.).

The loop is stateless — messages are passed in, not stored internally.
This makes it easy to test and compose.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.source_aware import (
    PlannedToolArguments,
    admit_tool_result,
    plan_tool_arguments,
    select_step_tools,
    tool_source_kind,
)
from databricks_deep_research.agents.vector_query_optimizer import VectorQueryOptimizer
from databricks_deep_research.events.types import (
    AgentStreamChunkEvent,
    StreamEvent,
    ToolCacheHitEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient, LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import ResearchTool, ToolContext
from databricks_deep_research.tracing import trace_span

logger = logging.getLogger(__name__)


def _build_request_id(tool_name: str, arguments: dict[str, Any], scope: str) -> str:
    raw = json.dumps({"tool": tool_name, "args": arguments, "scope": scope}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tool call cache
# ---------------------------------------------------------------------------


class ToolCallCache:
    """Dedup cache for tool calls — stores results AND source metadata."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, list[Any]]] = {}

    def _make_key(self, tool_name: str, arguments: dict[str, Any], scope: str = "") -> str:
        raw = json.dumps({"tool": tool_name, "args": arguments, "scope": scope}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        scope: str = "",
    ) -> tuple[str, list[Any]] | None:
        return self._cache.get(self._make_key(tool_name, arguments, scope))

    def put(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        sources: list[Any] | None = None,
        *,
        scope: str = "",
    ) -> None:
        self._cache[self._make_key(tool_name, arguments, scope)] = (result, sources or [])


# ---------------------------------------------------------------------------
# ReAct loop result
# ---------------------------------------------------------------------------


@dataclass
class ReactResult:
    """Result of a ReAct loop execution."""

    content: str
    tool_calls_made: int = 0
    events: list[StreamEvent] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    sources: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compaction helpers
# ---------------------------------------------------------------------------


def _summarize_tool_result(content: str, max_chars: int = 800) -> str:
    """Preserve key data points when compacting a tool result.

    Keeps lines that contain pipe characters (markdown table rows),
    numeric digits (data values), or metadata markers (``[...]`` headers).
    Discards narrative text and whitespace to fit within *max_chars*.
    """
    lines = content.split("\n")
    kept: list[str] = []
    char_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        has_pipe = "|" in stripped
        has_number = any(c.isdigit() for c in stripped)
        is_metadata = stripped.startswith("[") and "]" in stripped

        if has_pipe or has_number or is_metadata:
            kept.append(stripped)
            char_count += len(stripped) + 1
            if char_count >= max_chars:
                kept.append("...[additional data truncated]")
                break

    if not kept:
        return f"[Prior results — {len(content)} chars, no tabular data]"

    return (
        f"[Compacted from {len(content)} chars — key data preserved:]\n"
        + "\n".join(kept)
    )


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


class ReactLoop:
    """Generic ReAct execution loop.

    Iterates: LLM call → parse tool calls → execute tools → add results → continue
    until no tool calls returned or max_tool_calls reached.
    """

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        tools: list[ResearchTool],
        *,
        tool_context: ToolContext | None = None,
        cache: ToolCallCache | None = None,
        node_id: str = "",
        max_tool_calls: int = 20,
        model_tier: str = "analytical",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        subtype: str = "",
        max_result_chars: int = 4000,
        compaction_strategy: str = "truncate",
    ) -> None:
        self._llm = llm_client
        self._ctx = tool_context or ToolContext()
        self._all_tools = {t.definition.name: t for t in tools}
        self._tools = dict(self._all_tools)
        self._tool_defs = [self._to_openai_tool(t) for t in tools]
        self._cache = cache or ToolCallCache()
        self._node_id = node_id
        self._max_tool_calls = max_tool_calls
        self._model_tier = model_tier
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stream = stream
        self._subtype = subtype
        self._max_result_chars = max_result_chars
        self._compaction_strategy = compaction_strategy
        self._fallback_tools: dict[str, ResearchTool] = {}
        self._fallback_enabled = False
        self._fallback_retry_used = False
        self._cache_scope = self._build_cache_scope()
        self._runtime_store = getattr(self._ctx, "runtime_store", None)
        self._step_query_signatures: set[str] = set()
        self._tool_outcome_history: dict[str, list[dict[str, Any]]] = {}
        self._vs_optimizer = VectorQueryOptimizer(llm_client)
        self._seen_source_urls: set[str] = set()
        self._consecutive_zero_novel_rounds: int = 0
        self._same_tool_consecutive_rounds: int = 0
        self._last_round_tool: str = ""
        self._budget_warned: bool = False
        self._compact_after_rounds: int = max(2, max_tool_calls * 2 // 5)

    # -- Budget-aware guidance -----------------------------------------------

    def _inject_budget_guidance(
        self,
        messages: list[dict[str, Any]],
        remaining: int,
    ) -> list[dict[str, Any]] | None:
        """Inject budget awareness; return restricted tool_defs or None.

        At ≤25% budget remaining (once): warn to start writing findings.
        At ≤2 calls remaining: critical message + restrict to compute-only.
        """
        if remaining <= 0:
            return None

        if remaining <= 2:
            messages.append({
                "role": "system",
                "content": (
                    f"CRITICAL: Only {remaining} tool call(s) remaining. "
                    "Include your COMPLETE FINAL OUTPUT text in this response. "
                    "Write your full findings/answer alongside any last tool call. "
                    "If you stored values via compute, reference them in your output."
                ),
            })
            logger.info(
                "REACT_BUDGET_CRITICAL node=%s remaining=%d",
                self._node_id, remaining,
            )
            compute_defs = [
                td for td in self._tool_defs
                if td["function"]["name"] == "compute"
            ]
            return compute_defs if compute_defs else None

        remaining_pct = remaining / self._max_tool_calls if self._max_tool_calls > 0 else 1.0
        if remaining_pct <= 0.25 and not self._budget_warned:
            self._budget_warned = True
            messages.append({
                "role": "system",
                "content": (
                    f"BUDGET: {remaining} tool calls remaining out of "
                    f"{self._max_tool_calls}. Start writing your findings. "
                    "Store any remaining values in compute."
                ),
            })
            logger.info(
                "REACT_BUDGET_WARNING node=%s remaining=%d",
                self._node_id, remaining,
            )

        return None

    def _normalize_query_signature(self, tool_name: str, rewritten_query: str) -> str:
        normalized = " ".join(str(rewritten_query).lower().split())
        return f"{tool_name}:{normalized}"

    def _record_tool_outcome(self, tool_name: str, meta: dict[str, Any], rewritten_query: str) -> None:
        history = self._tool_outcome_history.setdefault(tool_name, [])
        history.append({**meta, "rewritten_query": rewritten_query})
        if len(history) > 6:
            del history[0]

    _JACCARD_DEDUP_THRESHOLD = 0.8  # >80% word overlap = near-duplicate

    def _is_low_yield_duplicate(self, tool_name: str, rewritten_query: str) -> bool:
        """Return True if this query is a (near-)duplicate of a previous one for this tool."""
        signature = self._normalize_query_signature(tool_name, rewritten_query)

        # Exact match on normalized query — unconditionally skip
        if signature in self._step_query_signatures:
            logger.info(
                "REACT_EXACT_DEDUP_SKIP node=%s tool=%s query=%r",
                self._node_id, tool_name, rewritten_query[:200],
            )
            return True

        # Near-duplicate: Jaccard overlap on word sets for the same tool
        query_words = set(rewritten_query.lower().split())
        if len(query_words) < 2:
            return False

        for prev_sig in self._step_query_signatures:
            if not prev_sig.startswith(f"{tool_name}:"):
                continue
            prev_words = set(prev_sig.split(":", 1)[1].split())
            if not prev_words:
                continue
            intersection = len(query_words & prev_words)
            union = len(query_words | prev_words)
            if union > 0 and intersection / union > self._JACCARD_DEDUP_THRESHOLD:
                logger.info(
                    "REACT_JACCARD_DEDUP_SKIP node=%s tool=%s jaccard=%.2f query=%r prev=%r",
                    self._node_id, tool_name, intersection / union,
                    rewritten_query[:200], prev_sig.split(":", 1)[1][:200],
                )
                return True

        return False

    def _dedup_check_and_register(
        self, tool_name: str, query: str,
    ) -> dict[str, Any] | None:
        """Check if query is a near-duplicate; if not, register it for future checks.

        Returns a skip-meta dict if duplicate (caller should return early),
        or None if the query is novel (caller should proceed with execution).
        """
        if self._is_low_yield_duplicate(tool_name, query):
            logger.info(
                "REACT_DEDUP_SKIP node=%s tool=%s query=%r",
                self._node_id, tool_name, query[:200],
            )
            return {
                "tool_success": True, "tool_error": "",
                "raw_source_count": 0, "accepted_source_count": 0,
                "accepted_substantive_count": 0,
                "accepted_low_value_count": 0,
                "rejected_source_count": 0,
                "evidence_quality": "empty",
                "failure_mode": "duplicate_low_yield",
                "needs_adaptation": True,
            }
        self._step_query_signatures.add(
            self._normalize_query_signature(tool_name, query)
        )
        return None

    # -- Public interface ---------------------------------------------------

    async def execute(
        self,
        messages: list[dict[str, Any]],
    ) -> ReactResult:
        """Run the ReAct loop on the given messages.

        Returns the final ReactResult with accumulated content and events.
        """
        self._apply_step_tool_selection()

        logger.info(
            "REACT_START node=%s tools=%s max_calls=%d",
            self._node_id, list(self._tools.keys()), self._max_tool_calls,
        )

        async with trace_span(
            f"react_loop.{self._subtype or 'generic'}",
            span_type="CHAIN",
            attributes={
                "react.tools": str(list(self._tools.keys())),
                "react.max_calls": self._max_tool_calls,
            },
        ) as loop_span:
            events: list[StreamEvent] = []
            sources: list[Any] = []
            total_usage: dict[str, int] = {}
            call_count = 0
            first_turn_retried = False

            while True:
                # Compact old tool results to limit prompt growth
                if call_count > 0:
                    self._compact_old_tool_results(messages)

                # Budget-aware guidance + optional tool restriction
                remaining = self._max_tool_calls - call_count
                restricted_tools = self._inject_budget_guidance(messages, remaining)
                active_tool_defs = (
                    restricted_tools
                    if restricted_tools is not None
                    else (self._tool_defs if self._tools else None)
                )

                # LLM call
                if self._stream and call_count == 0 and not first_turn_retried:
                    response, stream_events = await self._stream_call(messages)
                    events.extend(stream_events)
                else:
                    response = await self._llm.complete(
                        messages,
                        self._model_tier,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        tools=active_tool_defs,
                    )

                # Track usage
                _merge_usage(total_usage, response.usage)
                logger.info(
                    "REACT_LLM_USAGE node=%s call=%d prompt_tokens=%d "
                    "completion_tokens=%d total_tokens=%d",
                    self._node_id, call_count,
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                    response.usage.get("total_tokens", 0),
                )

                # First-turn retry for evidence-gathering subtypes
                if (
                    not response.tool_calls
                    and call_count == 0
                    and not first_turn_retried
                    and self._subtype in ("background", "researcher")
                ):
                    first_turn_retried = True
                    messages.append(self._assistant_msg(response))
                    messages.append({
                        "role": "system",
                        "content": (
                            "You have search tools available. "
                            "Call at least one to gather evidence before responding."
                        ),
                    })
                    continue

                # No tool calls — done
                if not response.tool_calls or call_count >= self._max_tool_calls:
                    if (
                        not response.tool_calls
                        and self._fallback_tools
                        and not self._fallback_enabled
                        and not sources
                        and call_count > 0
                    ):
                        self._enable_fallback_tools(messages, reason="no_tool_calls_with_zero_accepted_sources")
                        continue
                    if (
                        not response.tool_calls
                        and self._fallback_enabled
                        and not sources
                        and not self._fallback_retry_used
                        and call_count > 0
                    ):
                        self._fallback_retry_used = True
                        messages.append({
                            "role": "system",
                            "content": (
                                "You still have no accepted evidence. "
                                "Call one of the available fallback tools before answering."
                            ),
                        })
                        continue
                    exit_reason = (
                        "no_tool_calls" if not response.tool_calls
                        else "max_calls_reached"
                    )

                    # ── Compute namespace fallback ─────────────────────
                    # When max_calls is hit with empty content, dump any
                    # values the agent stored in the compute tool's
                    # namespace.  Zero-cost (no extra LLM call).
                    if (
                        exit_reason == "max_calls_reached"
                        and not response.content.strip()
                    ):
                        compute_tool = self._all_tools.get("compute")
                        if compute_tool and hasattr(compute_tool, "_namespace"):
                            user_vars = {
                                k: repr(v)
                                for k, v in compute_tool._namespace.items()
                                if isinstance(v, (int, float, str, list, dict, tuple, bool))
                            }
                            if user_vars:
                                parts = [f"{k} = {v}" for k, v in user_vars.items()]
                                response = LLMResponse(
                                    content="Extracted data:\n" + "\n".join(parts),
                                    tool_calls=[],
                                    model=response.model,
                                    usage=response.usage,
                                )
                                exit_reason = "namespace_fallback"
                                logger.info(
                                    "REACT_NAMESPACE_FALLBACK node=%s vars=%d",
                                    self._node_id, len(user_vars),
                                )

                    logger.info(
                        "REACT_DONE node=%s calls=%d content_len=%d sources=%d "
                        "exit_reason=%s",
                        self._node_id, call_count, len(response.content),
                        len(sources), exit_reason,
                    )
                    if loop_span:
                        span_attrs: dict[str, Any] = {
                            "react.total_calls": call_count,
                            "react.total_sources": len(sources),
                        }
                        for k, v in total_usage.items():
                            span_attrs[f"react.{k}"] = v
                        loop_span.set_attributes(span_attrs)
                    return ReactResult(
                        content=response.content,
                        tool_calls_made=call_count,
                        events=events,
                        token_usage=total_usage,
                        sources=sources,
                    )

                # Add assistant message with tool calls
                messages.append(self._assistant_msg(response))

                # -- Phase 1: Parse, classify (cached vs uncached) --
                responded_tc_ids: set[str] = set()
                cached_results: dict[str, tuple[str, list[Any]]] = {}
                to_execute: list[tuple[ToolCall, dict[str, Any]]] = []
                sources_before_round = len(sources)

                for tc in response.tool_calls:
                    if call_count >= self._max_tool_calls:
                        break
                    call_count += 1
                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    # Step-scoped cache (same step, same query)
                    cached = self._cache.get(tc.function_name, args, scope=self._cache_scope)
                    if cached is None:
                        # Global cross-step cache (any step, same tool + args)
                        cached = self._cache.get(tc.function_name, args, scope="")
                    if cached is not None:
                        content, cached_sources = cached
                        cached_results[tc.id] = (content, cached_sources)
                        events.append(ToolCacheHitEvent(
                            node_id=self._node_id, timestamp=_now(),
                            tool_name=tc.function_name,
                            cache_key=f"{tc.function_name}:{hash(tc.arguments)}",
                        ))
                    else:
                        to_execute.append((tc, args))
                        events.append(ToolCallEvent(
                            node_id=self._node_id, timestamp=_now(),
                            tool_name=tc.function_name, arguments=args,
                        ))

                # -- Phase 2: Execute uncached in parallel --
                exec_results: dict[str, tuple[str, list[Any], dict[str, Any]]] = {}
                if to_execute:
                    tasks = [
                        asyncio.create_task(self._execute_single_tool(tc, args))
                        for tc, args in to_execute
                    ]
                    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in raw_results:
                        if isinstance(r, BaseException):
                            logger.error(
                                "REACT_TOOL_GATHER_ERROR node=%s error=%s",
                                self._node_id, r,
                            )
                            continue
                        tc_id, result_content, tool_sources, tool_result_meta = r
                        exec_results[tc_id] = (
                            result_content,
                            tool_sources,
                            tool_result_meta,
                        )

                # -- Phase 3: Reassemble in original order --
                for tc in response.tool_calls:
                    if tc.id in cached_results:
                        content, cached_srcs = cached_results[tc.id]
                        sources.extend(cached_srcs)
                        messages.append(self._tool_msg(tc.id, content))
                        responded_tc_ids.add(tc.id)
                    elif tc.id in exec_results:
                        result_content, tool_srcs, tool_result_meta = exec_results[tc.id]
                        sources.extend(tool_srcs)
                        try:
                            exec_args = json.loads(tc.arguments) if tc.arguments else {}
                        except json.JSONDecodeError:
                            exec_args = {}
                        self._cache.put(
                            tc.function_name,
                            exec_args,
                            result_content,
                            tool_srcs,
                            scope=self._cache_scope,
                        )
                        # Global cross-step cache
                        self._cache.put(
                            tc.function_name,
                            exec_args,
                            result_content,
                            tool_srcs,
                            scope="",
                        )
                        messages.append(self._tool_msg(tc.id, result_content))
                        responded_tc_ids.add(tc.id)
                        events.append(ToolResultEvent(
                            node_id=self._node_id, timestamp=_now(),
                            tool_name=tc.function_name,
                            result_summary=result_content[:200],
                            source_count=int(tool_result_meta.get("accepted_source_count", 0)),
                            raw_source_count=int(tool_result_meta.get("raw_source_count", 0)),
                            accepted_source_count=int(tool_result_meta.get("accepted_source_count", 0)),
                            rejected_source_count=int(tool_result_meta.get("rejected_source_count", 0)),
                            tool_success=bool(tool_result_meta.get("tool_success", True)),
                            tool_error=str(tool_result_meta.get("tool_error", "") or ""),
                        ))

                    if call_count >= self._max_tool_calls:
                        break

                # Ensure ALL tool_calls have tool_result messages (prevents
                # "tool_use without tool_result" errors from Anthropic API)
                for tc in response.tool_calls:
                    if tc.id not in responded_tc_ids:
                        messages.append(self._tool_msg(
                            tc.id, "Skipped: tool call budget exhausted"
                        ))

                # Track novel sources for diminishing-returns detection
                novel_urls_this_round: set[str] = set()
                for tc in response.tool_calls:
                    if tc.id in exec_results:
                        _, tool_srcs, _ = exec_results[tc.id]
                        for src in tool_srcs:
                            url = str(src.get("url", "") if isinstance(src, dict) else getattr(src, "url", ""))
                            url = url.rstrip("/").lower()
                            if url and url not in self._seen_source_urls:
                                novel_urls_this_round.add(url)
                self._seen_source_urls.update(novel_urls_this_round)

                if len(novel_urls_this_round) == 0 and responded_tc_ids:
                    self._consecutive_zero_novel_rounds += 1
                else:
                    self._consecutive_zero_novel_rounds = 0

                if self._consecutive_zero_novel_rounds >= 2:
                    logger.info(
                        "REACT_EARLY_STOP_NUDGE node=%s rounds=%d seen_urls=%d",
                        self._node_id, self._consecutive_zero_novel_rounds,
                        len(self._seen_source_urls),
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "The last 2 rounds of tool calls returned no new unique sources. "
                            "You likely have sufficient evidence. Synthesize your findings "
                            "and provide your observation now."
                        ),
                    })

                # Track tool diversity — nudge LLM when it hammers a single tool
                round_tool_names: set[str] = set()
                for tc in response.tool_calls:
                    if tc.id in responded_tc_ids:
                        round_tool_names.add(tc.function_name)

                if len(round_tool_names) == 1 and len(self._tools) > 1:
                    only_tool = next(iter(round_tool_names))
                    if only_tool == self._last_round_tool:
                        self._same_tool_consecutive_rounds += 1
                    else:
                        self._same_tool_consecutive_rounds = 1
                    self._last_round_tool = only_tool
                else:
                    self._same_tool_consecutive_rounds = 0
                    self._last_round_tool = ""

                if self._same_tool_consecutive_rounds >= 3 and len(self._tools) > 1:
                    other_tools = [n for n in self._tools if n != self._last_round_tool]
                    logger.info(
                        "REACT_TOOL_DIVERSITY_NUDGE node=%s repeated_tool=%s "
                        "rounds=%d other_tools=%s",
                        self._node_id, self._last_round_tool,
                        self._same_tool_consecutive_rounds, other_tools,
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            f"You have used only '{self._last_round_tool}' for the last "
                            f"{self._same_tool_consecutive_rounds} rounds. "
                            f"Other tools are available: {', '.join(other_tools)}. "
                            "For cross-validation and broader coverage, try querying "
                            "a different tool before concluding this step."
                        ),
                    })
                    self._same_tool_consecutive_rounds = 0

                if (
                    self._fallback_tools
                    and not self._fallback_enabled
                    and len(sources) == sources_before_round
                    and responded_tc_ids
                ):
                    self._enable_fallback_tools(messages, reason="zero_accepted_sources")

    # -- Parallel tool execution --------------------------------------------

    async def _execute_single_tool(
        self, tc: ToolCall, args: dict[str, Any],
    ) -> tuple[str, str, list[Any], dict[str, Any]]:
        """Execute one tool call. Returns (tc_id, content, sources, diagnostics)."""
        tool_name = tc.function_name
        tool = self._tools.get(tool_name)
        if tool is None:
            return tc.id, f"Error: Unknown tool '{tool_name}'", [], {
                "tool_success": False,
                "tool_error": f"Unknown tool '{tool_name}'",
                "raw_source_count": 0,
                "accepted_source_count": 0,
                "rejected_source_count": 0,
            }

        # Build rich log of tool arguments
        log_args = dict(args)
        if tool_name == "web_crawl" and "url_index" in args and self._ctx.url_registry:
            resolved_url = self._ctx.url_registry.resolve(args["url_index"])
            log_args["_resolved_url"] = resolved_url or "UNKNOWN"

        logger.info(
            "REACT_TOOL_CALL node=%s tool=%s args=%s",
            self._node_id, tool_name,
            {k: str(v)[:200] for k, v in log_args.items()},
        )

        try:
            # Check for VS query optimization (LLM or passthrough mode)
            source_kind = tool_source_kind(tool.definition)
            query_policy = (tool.definition.metadata or {}).get("query_policy", "")

            if source_kind == "vector_index" and query_policy in ("llm", "passthrough"):
                # Dedup: exact + Jaccard check against step query signatures
                original_query = str(args.get("query", "")).strip()
                skip_meta = self._dedup_check_and_register(tool_name, original_query)
                if skip_meta is not None:
                    return tc.id, f"Skipped duplicate {tool_name} query", [], skip_meta

                # Global cache check with original (pre-optimizer) args
                vs_cached = self._cache.get(tc.function_name, args, scope="")
                if vs_cached is not None:
                    cached_content, cached_sources = vs_cached
                    logger.info(
                        "VS_GLOBAL_CACHE_HIT node=%s tool=%s query=%r",
                        self._node_id, tool_name, str(args.get("query", ""))[:200],
                    )
                    return tc.id, cached_content, cached_sources, {
                        "tool_success": True,
                        "tool_error": "",
                        "raw_source_count": len(cached_sources),
                        "accepted_source_count": len(cached_sources),
                        "accepted_substantive_count": len(cached_sources),
                        "accepted_low_value_count": 0,
                        "rejected_source_count": 0,
                        "evidence_quality": "cached",
                        "failure_mode": "",
                        "needs_adaptation": False,
                    }
                result = await self._execute_vs_optimized(tc, tool, args, query_policy)
                _tc_id, vs_content, vs_sources, vs_meta = result
                if vs_meta.get("tool_success", False) and vs_sources:
                    self._cache.put(tc.function_name, args, vs_content, vs_sources, scope="")
                return result

            # ── Builtin deterministic tools (compute) ────────────────────
            # Compute results are mathematical outputs, not retrieval sources.
            # Routing through admission creates a synthetic source, scores it
            # on keyword overlap (always ≈0 for numbers), and rejects it —
            # replacing the actual computation with "No relevant results
            # accepted."  Bypass the entire admission pipeline.
            if tool_source_kind(tool.definition) == "builtin":
                tool_result = await tool.execute(
                    tool.validate_arguments(args), self._ctx
                )
                logger.info(
                    "BUILTIN_TOOL_RESULT node=%s tool=%s success=%s content_len=%d",
                    self._node_id,
                    tool_name,
                    tool_result.success,
                    len(tool_result.content),
                )
                return tc.id, tool_result.content, [], {
                    "tool_success": tool_result.success,
                    "tool_error": tool_result.error or "",
                    "raw_source_count": 0,
                    "accepted_source_count": 0,
                    "accepted_substantive_count": 0,
                    "accepted_low_value_count": 0,
                    "rejected_source_count": 0,
                    "evidence_quality": "builtin",
                    "failure_mode": "" if tool_result.success else "tool_error",
                    "needs_adaptation": not tool_result.success,
                }

            planned = plan_tool_arguments(
                tool.definition,
                args,
                current_step=self._ctx.current_step,
                root_query=self._ctx.query,
                background_summary=self._ctx.background_summary,
                recent_observations=self._ctx.recent_observations,
            )
            logger.info(
                "QUERY_PLAN_APPLIED node=%s tool=%s source_type=%s strategy=%s "
                "original_query=%r rewritten_query=%r step_title=%r",
                self._node_id,
                tool_name,
                tool_source_kind(tool.definition),
                planned.strategy,
                planned.original_query[:300],
                planned.rewritten_query[:300],
                self._current_step_title()[:200],
            )
            if planned.alternate_queries:
                logger.info(
                    "VECTOR_QUERY_ALTERNATES node=%s tool=%s count=%d alternates=%s",
                    self._node_id,
                    tool_name,
                    len(planned.alternate_queries),
                    [query[:200] for query in planned.alternate_queries],
                )

            skip_meta = self._dedup_check_and_register(tool_name, planned.rewritten_query)
            if skip_meta is not None:
                return tc.id, f"Skipped duplicate {tool_name} query", [], skip_meta

            # Post-rewrite cache: catches different raw queries → same rewrite
            rewritten_cache_args = {k: v for k, v in planned.arguments.items() if k != "_alternate_queries"}
            post_cached = self._cache.get(tc.function_name, rewritten_cache_args, scope=self._cache_scope)
            if post_cached is None:
                post_cached = self._cache.get(tc.function_name, rewritten_cache_args, scope="")
            if post_cached is not None:
                logger.info(
                    "POST_REWRITE_CACHE_HIT node=%s tool=%s query=%r",
                    self._node_id, tool_name, planned.rewritten_query[:200],
                )
                content, cached_sources = post_cached
                return tc.id, content, cached_sources, {
                    "tool_success": True, "tool_error": "",
                    "raw_source_count": len(cached_sources),
                    "accepted_source_count": len(cached_sources),
                    "accepted_substantive_count": len(cached_sources),
                    "accepted_low_value_count": 0,
                    "rejected_source_count": 0,
                    "evidence_quality": "cached", "failure_mode": "",
                    "needs_adaptation": False,
                }

            validated_args = tool.validate_arguments(planned.arguments)
            async with trace_span(
                f"tool.{tool_name}", span_type="TOOL",
                attributes={
                    "tool.name": tool_name,
                    "tool.args": str({k: str(v)[:100] for k, v in planned.arguments.items()}),
                    "tool.query": str(planned.rewritten_query)[:500],
                },
            ) as tool_span:
                tool_result = await tool.execute(validated_args, self._ctx)
                admitted = admit_tool_result(
                    tool.definition,
                    tool_result,
                    current_step=self._ctx.current_step,
                    root_query=self._ctx.query,
                )
                logger.info(
                    "ENTERPRISE_RESULTS_RETRIEVED node=%s tool=%s source_type=%s raw=%d "
                    "accepted=%d rejected=%d top_titles=%s",
                    self._node_id,
                    tool_name,
                    tool_source_kind(tool.definition),
                    len(admitted.raw_sources),
                    admitted.accepted_count,
                    admitted.rejected_count,
                    [src.get("title", "")[:120] for src in admitted.raw_sources[:5]],
                )
                if admitted.accepted_sources:
                    logger.info(
                        "ENTERPRISE_RESULTS_ACCEPTED node=%s tool=%s reasons=%s",
                        self._node_id,
                        tool_name,
                        [src.get("admission_reason", "")[:160] for src in admitted.accepted_sources[:5]],
                    )
                if admitted.rejected_sources:
                    logger.info(
                        "ENTERPRISE_RESULTS_REJECTED node=%s tool=%s reasons=%s",
                        self._node_id,
                        tool_name,
                        [src.get("admission_reason", "")[:160] for src in admitted.rejected_sources[:5]],
                    )
                    logger.info(
                        "ADMISSION_REJECTION_CONTEXT node=%s tool=%s source_kind=%s "
                        "step_title=%r root_query=%r "
                        "rejected_titles=%s rejected_relevance_scores=%s",
                        self._node_id,
                        tool_name,
                        tool_source_kind(tool.definition),
                        self._current_step_title()[:200],
                        self._ctx.query[:200],
                        [s.get("title", "")[:120] for s in admitted.rejected_sources[:5]],
                        [s.get("relevance_score") for s in admitted.rejected_sources[:5]],
                    )
                if tool_span:
                    tool_span.set_attributes({
                        "tool.result_len": len(admitted.content),
                        "tool.success": tool_result.success,
                        "tool.error": tool_result.error or "",
                        "tool.source_count": admitted.accepted_count,
                        "tool.raw_source_count": len(admitted.raw_sources),
                        "tool.accepted_source_count": admitted.accepted_count,
                        "tool.rejected_source_count": admitted.rejected_count,
                        "tool.original_query": planned.original_query[:500],
                        "tool.rewritten_query": planned.rewritten_query[:500],
                        "tool.query_strategy": planned.strategy,
                        "tool.accepted_substantive_count": admitted.accepted_substantive_count,
                        "tool.accepted_low_value_count": admitted.accepted_low_value_count,
                        "tool.evidence_quality": admitted.evidence_quality,
                        "tool.failure_mode": admitted.failure_mode,
                        "tool.needs_adaptation": admitted.needs_adaptation,
                        "tool.failure_class": str(tool_result.data.get("failure_class", "")),
                        "tool.suppressed_by_failure_cache": bool(
                            tool_result.data.get("suppressed_by_failure_cache", False)
                        ),
                        "tool.suppression_scope": str(tool_result.data.get("suppression_scope", "")),
                    })
                meta = {
                    "tool_success": tool_result.success,
                    "tool_error": tool_result.error or "",
                    "raw_source_count": len(admitted.raw_sources),
                    "accepted_source_count": admitted.accepted_count,
                    "accepted_substantive_count": admitted.accepted_substantive_count,
                    "accepted_low_value_count": admitted.accepted_low_value_count,
                    "rejected_source_count": admitted.rejected_count,
                    "evidence_quality": admitted.evidence_quality,
                    "failure_mode": admitted.failure_mode,
                    "needs_adaptation": admitted.needs_adaptation,
                    "failure_class": str(tool_result.data.get("failure_class", "")),
                    "suppressed_by_failure_cache": bool(
                        tool_result.data.get("suppressed_by_failure_cache", False)
                    ),
                    "suppression_scope": str(tool_result.data.get("suppression_scope", "")),
                }
                # Cache with rewritten args too (so post-rewrite lookup hits next time)
                rewritten_put_args = {k: v for k, v in planned.arguments.items() if k != "_alternate_queries"}
                self._cache.put(tc.function_name, rewritten_put_args, admitted.content, admitted.accepted_sources, scope="")

                self._record_tool_outcome(tool_name, meta, planned.rewritten_query)
                return tc.id, admitted.content, admitted.accepted_sources, meta
        except Exception as exc:
            logger.warning(
                "REACT_TOOL_ERROR node=%s tool=%s error=%s",
                self._node_id, tool_name, exc,
            )
            return tc.id, f"Error executing {tool_name}: {exc}", [], {
                "tool_success": False,
                "tool_error": str(exc),
                "raw_source_count": 0,
                "accepted_source_count": 0,
                "rejected_source_count": 0,
            }

    async def _execute_vs_optimized(
        self,
        tc: ToolCall,
        tool: ResearchTool,
        args: dict[str, Any],
        query_policy: str,
    ) -> tuple[str, str, list[Any], dict[str, Any]]:
        """Handle VS tools with LLM optimization or passthrough.

        Returns the same (tc_id, content, sources, meta) tuple as
        _execute_single_tool for seamless integration.
        """
        tool_name = tool.definition.name
        original_query = str(args.get("query", "")).strip()

        async with trace_span(
            f"tool.{tool_name}", span_type="TOOL",
            attributes={
                "tool.name": tool_name,
                "tool.query_policy": query_policy,
                "tool.original_query": original_query[:500],
                "tool.query": original_query[:500],
            },
        ) as tool_span:
            if query_policy == "passthrough":
                # Passthrough: use agent's query as-is
                validated = tool.validate_arguments(args)
                result = await tool.execute(validated, self._ctx)
                planned = PlannedToolArguments(
                    arguments=args,
                    original_query=original_query,
                    rewritten_query=original_query,
                    alternate_queries=[],
                    strategy="passthrough",
                    source_hint=None,
                )
                trace_meta: dict[str, Any] = {
                    "strategy": "passthrough",
                    "generated_queries": [original_query],
                }
            else:
                # LLM optimization pipeline
                result, trace_meta = await self._vs_optimizer.optimize_and_execute(
                    tool, original_query, self._ctx
                )
                generated = trace_meta.get("generated_queries", [original_query])
                planned = PlannedToolArguments(
                    arguments={"query": generated[0] if generated else original_query},
                    original_query=original_query,
                    rewritten_query=generated[0] if generated else original_query,
                    alternate_queries=generated[1:] if len(generated) > 1 else [],
                    strategy=trace_meta.get("strategy", "llm"),
                    source_hint=None,
                )
                logger.info(
                    "VS_OPTIMIZATION_COMPLETE node=%s tool=%s strategy=%s "
                    "queries=%d vs_calls=%d/%d rerank=%d->%d total_ms=%d",
                    self._node_id, tool_name,
                    trace_meta.get("strategy"),
                    trace_meta.get("unique_query_count", 1),
                    trace_meta.get("vs_calls_success", 0),
                    trace_meta.get("vs_calls_total", 0),
                    trace_meta.get("rerank_input", 0),
                    trace_meta.get("rerank_output", 0),
                    trace_meta.get("total_ms", 0),
                )

            # Call admit_tool_result() for framework source tracking
            admitted = admit_tool_result(
                tool.definition,
                result,
                current_step=self._ctx.current_step,
                root_query=self._ctx.query,
            )

            logger.info(
                "VS_OPTIMIZED_RESULTS node=%s tool=%s strategy=%s raw=%d "
                "accepted=%d rejected=%d",
                self._node_id, tool_name,
                planned.strategy,
                len(admitted.raw_sources),
                admitted.accepted_count,
                admitted.rejected_count,
            )

            if tool_span:
                tool_span.set_attributes({
                    "tool.result_len": len(admitted.content),
                    "tool.success": result.success,
                    "tool.error": result.error or "",
                    "tool.source_count": admitted.accepted_count,
                    "tool.raw_source_count": len(admitted.raw_sources),
                    "tool.accepted_source_count": admitted.accepted_count,
                    "tool.rejected_source_count": admitted.rejected_count,
                    "tool.original_query": planned.original_query[:500],
                    "tool.rewritten_query": planned.rewritten_query[:500],
                    "tool.query_strategy": planned.strategy,
                    "tool.accepted_substantive_count": admitted.accepted_substantive_count,
                    "tool.accepted_low_value_count": admitted.accepted_low_value_count,
                    "tool.evidence_quality": admitted.evidence_quality,
                    "tool.failure_mode": admitted.failure_mode,
                    "tool.needs_adaptation": admitted.needs_adaptation,
                    "tool.vs_stage1_ms": trace_meta.get("stage1_ms", 0),
                    "tool.vs_stage2_ms": trace_meta.get("stage2_ms", 0),
                    "tool.vs_stage3_ms": trace_meta.get("stage3_ms", 0),
                    "tool.vs_total_ms": trace_meta.get("total_ms", 0),
                })

            meta = {
                "tool_success": result.success,
                "tool_error": result.error or "",
                "raw_source_count": len(admitted.raw_sources),
                "accepted_source_count": admitted.accepted_count,
                "accepted_substantive_count": admitted.accepted_substantive_count,
                "accepted_low_value_count": admitted.accepted_low_value_count,
                "rejected_source_count": admitted.rejected_count,
                "evidence_quality": admitted.evidence_quality,
                "failure_mode": admitted.failure_mode,
                "needs_adaptation": admitted.needs_adaptation,
            }
            self._record_tool_outcome(tool_name, meta, planned.rewritten_query)
            return tc.id, admitted.content, admitted.accepted_sources, meta

    def _apply_step_tool_selection(self) -> None:
        """Expose preferred tools first and keep the rest as fallback."""
        selection = select_step_tools(
            list(self._all_tools.values()),
            self._ctx.current_step,
        )
        self._tools = {tool.definition.name: tool for tool in selection.active_tools}
        self._tool_defs = [self._to_openai_tool(tool) for tool in selection.active_tools]
        self._fallback_tools = {
            tool.definition.name: tool
            for tool in selection.fallback_tools
        }
        logger.info(
            "STEP_TOOL_FILTERED node=%s step_title=%r active=%s fallback=%s reasons=%s",
            self._node_id,
            self._current_step_title()[:200],
            list(self._tools.keys()),
            list(self._fallback_tools.keys()),
            selection.reasons,
        )

    def _enable_fallback_tools(self, messages: list[dict[str, Any]], *, reason: str) -> None:
        """Expose deferred fallback tools after preferred sources miss."""
        if self._fallback_enabled or not self._fallback_tools:
            return
        self._fallback_enabled = True
        self._consecutive_zero_novel_rounds = 0
        self._same_tool_consecutive_rounds = 0
        self._last_round_tool = ""
        self._tools.update(self._fallback_tools)
        self._tool_defs = [self._to_openai_tool(tool) for tool in self._tools.values()]
        logger.info(
            "FALLBACK_SOURCE_WIDENED node=%s reason=%s newly_enabled=%s",
            self._node_id,
            reason,
            list(self._fallback_tools.keys()),
        )
        messages.append({
            "role": "system",
            "content": (
                "Preferred sources returned no accepted evidence for the current step. "
                "Fallback sources are now available. Use them only to fill the gap."
            ),
        })

    def _build_cache_scope(self) -> str:
        current_step = self._ctx.current_step
        if current_step is None:
            return self._ctx.query[:120]
        step_id = ""
        if isinstance(current_step, dict):
            step_id = str(current_step.get("id", "") or current_step.get("title", ""))
        else:
            step_id = str(getattr(current_step, "id", "") or getattr(current_step, "title", ""))
        return step_id[:120] or self._ctx.query[:120]

    def _current_step_title(self) -> str:
        current_step = self._ctx.current_step
        if current_step is None:
            return ""
        if isinstance(current_step, dict):
            return str(current_step.get("title", "") or current_step.get("description", ""))
        return str(getattr(current_step, "title", "") or getattr(current_step, "description", ""))

    # -- Message compaction -------------------------------------------------

    def _compact_old_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Compact tool results from prior iterations to limit prompt growth.

        Supports two strategies:
        - ``truncate`` (default): hard-truncate old tool results to ``max_result_chars``.
        - ``mask``: replace old tool results with one-line placeholders, keeping
          the last 2 tool-calling iterations fully intact.
        """
        if self._max_result_chars <= 0:
            return

        # Indices of assistant messages that triggered tool calls
        tc_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]

        if not tc_indices:
            return

        if self._compaction_strategy == "mask":
            # Delay compaction until ~40% of budget is used
            if len(tc_indices) < self._compact_after_rounds:
                return
            # Keep last 3 iterations intact (was 2) for better data retention
            keep_from = tc_indices[-3] if len(tc_indices) >= 3 else 0
            for i in range(keep_from):
                msg = messages[i]
                if msg.get("role") == "tool":
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > self._max_result_chars:
                        msg["content"] = _summarize_tool_result(
                            content, max_chars=self._max_result_chars,
                        )
        else:
            # Original truncation behavior (backward compat)
            last_tc_idx = tc_indices[-1]
            if last_tc_idx <= 0:
                return
            for i in range(last_tc_idx):
                msg = messages[i]
                content = msg.get("content", "")
                if (msg.get("role") == "tool" and isinstance(content, str)
                        and len(content) > self._max_result_chars):
                    msg["content"] = content[:self._max_result_chars] + (
                        f"\n...[truncated from {len(content)} chars]"
                    )

    # -- Streaming -----------------------------------------------------------

    async def _stream_call(
        self, messages: list[dict[str, Any]]
    ) -> tuple[LLMResponse, list[StreamEvent]]:
        """Stream an LLM call, emitting AgentStreamChunkEvents."""
        events: list[StreamEvent] = []
        chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        async for item in self._llm.stream(
            messages,
            self._model_tier,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tools=self._tool_defs if self._tools else None,
        ):
            if isinstance(item, str):
                chunks.append(item)
                events.append(AgentStreamChunkEvent(
                    node_id=self._node_id, timestamp=_now(),
                    chunk=item, subtype=self._subtype,
                ))
            elif isinstance(item, ToolCall):
                tool_calls.append(item)

        content = "".join(chunks)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=self._model_tier,
        ), events

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _to_openai_tool(tool: ResearchTool) -> dict[str, Any]:
        """Convert a ResearchTool to OpenAI function-calling format."""
        defn = tool.definition
        return {
            "type": "function",
            "function": {
                "name": defn.name,
                "description": defn.description,
                "parameters": defn.parameters,
            },
        }

    @staticmethod
    def _assistant_msg(response: LLMResponse) -> dict[str, Any]:
        """Build an assistant message from LLM response (with tool calls)."""
        msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    @staticmethod
    def _tool_msg(tool_call_id: str, content: str) -> dict[str, Any]:
        """Build a tool result message."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _merge_usage(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] = total.get(k, 0) + v
