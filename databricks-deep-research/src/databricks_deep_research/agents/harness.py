"""Agent execution harness.

Constructs AgentInput from state + config, calls LLM (optionally via ReAct
loop), parses AgentOutput, writes to state + pools, and emits events.

This is the single point of interaction between the workflow executor and
an agent node.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import get_builtin
from databricks_deep_research.agents.config import AgentNodeConfig, PoolWriteConfig
from databricks_deep_research.agents.execution.output_normalizer import (
    NormalizedResearchOutput,
    build_observation_records,
    build_source_records,
    merge_and_dedup_sources,
    source_is_low_value,
    source_is_substantive,
)
from databricks_deep_research.agents.execution.output_normalizer import (
    build_observation_from_sources as _build_observation_from_sources,
)
from databricks_deep_research.agents.execution.output_normalizer import (
    has_substantive_text as _has_substantive_text,
)
from databricks_deep_research.agents.execution.output_normalizer import (
    is_semantically_empty as _is_semantically_empty,
)
from databricks_deep_research.agents.execution.output_normalizer import (
    normalize_research_output as _normalize_research_output_impl,
)
from databricks_deep_research.agents.execution.pool_projection import (
    PoolWriteBatch,
    build_research_pool_batch,
)
from databricks_deep_research.agents.execution.pool_projection import (
    extract_pool_items as _extract_pool_items,
)
from databricks_deep_research.agents.execution.state_projection import (
    project_research_state as _project_research_state_impl,
)
from databricks_deep_research.agents.isolation import AgentInput, AgentOutput
from databricks_deep_research.agents.prompt_context import (
    CompiledSynthesisContext,
    compile_pool_section,
    compile_synthesis_context,
    compile_typed_synthesis_context,
    default_synthesis_context,
    merge_token_usage,
)
from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.agents.source_aware import summarize_recent_observations
from databricks_deep_research.errors import WorkflowError
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    StreamEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.memory import (
    CHAT_MEMORY_APPENDIX_STATE_KEY,
    inject_attached_context_block,
)
from databricks_deep_research.pools.pool_state import PoolState
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    TableRegistry,
    ToolContext,
    UrlRegistry,
)
from databricks_deep_research.tracing import trace_span
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.runtime_core.selectors import (
    resolve_input_key,
    select_background_summary,
)
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


class _UnparsedJSONOutput(str):
    """Marker for JSON-configured agent output that could not be parsed."""


# ---------------------------------------------------------------------------
# json_repair sanity checks
# ---------------------------------------------------------------------------

# When json_repair produces output that is drastically smaller than the
# input and the input carried substantive text, treat the parse as a false
# success and fall through to raw preservation. Observed prod failure
# mode: 19 858 chars of researcher reasoning collapsed to a 59-char list.
_JSON_REPAIR_MIN_SIZE_RATIO = 0.1

# Kill-switch. When set to a falsy value the resilient repair check is
# skipped and the previous (narrower) behavior is restored exactly.
_RESILIENT_JSON_REPAIR_ENV = "HARNESS_RESILIENT_JSON_REPAIR"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_suspicious_repair(
    parsed: Any,
    content: str,
    *,
    subtype: str | None = None,
) -> str | None:
    """Return a short reason string when a json_repair result looks like a
    false success. Return ``None`` when the parse is considered healthy.

    The caller raises on non-None so the catch-all ``except`` falls
    through to :class:`_UnparsedJSONOutput` — preserving the raw content
    instead of shipping a tiny hallucination.

    Policy:

    * Empty scalars (``""``/``None``) from any non-whitespace content
      are always suspicious — matches the pre-existing behavior so
      "malformed but short" payloads still fall through to
      :class:`_UnparsedJSONOutput` as callers expect.
    * Empty containers (``[]``/``{}``) are suspicious only for
      ``researcher``-subtype agents with substantive input — other
      subtypes legitimately emit empty structured results.
    * Size collapse (parsed stringified length < 10% of input length)
      is suspicious when the input is non-trivial (≥500 chars).
    """
    # 1. Empty scalar from any non-whitespace content — preserve the
    # pre-existing aggressive behavior for compat with tests and
    # callers that rely on WorkflowError being raised for "empty" parses.
    if parsed in ("", None) and content.strip():
        return "empty scalar from non-empty content"

    # 2. Remaining checks require substantive text to avoid false
    # positives on intentionally-short agent outputs.
    if not _has_substantive_text(content, min_length=20):
        return None

    content_len = len(content)

    if isinstance(parsed, (list, dict)) and not parsed:
        if subtype == "researcher":
            return f"empty {type(parsed).__name__} from substantive text"
        return None

    parsed_len = len(str(parsed))
    if (
        content_len >= 500
        and parsed_len / max(content_len, 1) < _JSON_REPAIR_MIN_SIZE_RATIO
    ):
        ratio = parsed_len / content_len
        return (
            f"size collapse: content_len={content_len} "
            f"parsed_len={parsed_len} ratio={ratio:.3f}"
        )

    return None


# ---------------------------------------------------------------------------
# Context serialization
# ---------------------------------------------------------------------------


def _serialize_for_context(value: Any) -> str:
    """Serialize a state value for prompt injection.

    Handles Pydantic models (JSON), dicts (JSON), and primitives (str).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return str(value.model_dump_json(indent=2))
    if isinstance(value, dict):
        return json.dumps(value, indent=2, default=str)
    if isinstance(value, list | tuple):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _serialize_source_for_pool(source: Any) -> Any:
    """Normalize a tool source into a pool-friendly dict when possible."""
    if isinstance(source, dict):
        return source

    if not hasattr(source, "url"):
        return source

    item = {
        "url": getattr(source, "url", str(source)),
        "title": getattr(source, "title", "") or "",
        "snippet": getattr(source, "snippet", "") or "",
    }

    source_type = getattr(source, "source_type", None)
    if source_type:
        item["source_type"] = source_type

    source_kind = getattr(source, "source_kind", None)
    if source_kind:
        item["source_kind"] = source_kind

    content = getattr(source, "content", None)
    if content:
        item["content"] = content

    relevance_score = getattr(source, "relevance_score", None)
    if relevance_score is not None:
        item["relevance_score"] = relevance_score

    evidence_quality = getattr(source, "evidence_quality", None)
    if evidence_quality:
        item["evidence_quality"] = evidence_quality

    admission_status = getattr(source, "admission_status", None)
    if admission_status:
        item["admission_status"] = admission_status

    return item


# ---------------------------------------------------------------------------
# Main harness function
# ---------------------------------------------------------------------------


def _normalize_research_output(
    parsed: Any,
    config: AgentNodeConfig,
    sources: list[Any] | None = None,
) -> NormalizedResearchOutput | None:
    serialized_sources = [_serialize_source_for_pool(source) for source in (sources or [])]
    result = _normalize_research_output_impl(parsed, config, serialized_sources)
    # DR_LEAK_TRACE output_normalize_out: capture what the normalizer emits
    # to state_text — this is what downstream consumers (synthesizer prompt,
    # pool writes) actually read.
    if result is not None:
        try:
            logger.info(
                "DR_LEAK_TRACE phase=output_normalize_out "
                "output_key=%s status=%s repair=%s sources=%d "
                "state_text_len=%d state_text_head=%r",
                config.output_key,
                result.research_status,
                result.repair_mode,
                len(result.sources),
                len(result.state_text),
                (result.state_text or "")[:300].replace("\n", "\\n"),
            )
        except Exception as _exc:  # pragma: no cover — diagnostic only
            logger.debug("DR_LEAK_TRACE output_normalize_out skipped: %s", _exc)
    return result


def _build_pool_batches(
    normalized_research_output: NormalizedResearchOutput | None,
    pool_writes: list[PoolWriteConfig],
    output_key: str,
) -> dict[tuple[str, str], PoolWriteBatch]:
    if normalized_research_output is None:
        return {}
    return {
        (pw.pool, pw.extract): build_research_pool_batch(normalized_research_output, pw, output_key)
        for pw in pool_writes
    }


def _project_research_state(
    node_id: str,
    config: AgentNodeConfig,
    state: WorkflowState,
    normalized_research_output: NormalizedResearchOutput,
) -> tuple[str, dict[str, Any]]:
    return _project_research_state_impl(node_id, config, state, normalized_research_output)


async def execute_agent(
    node_id: str,
    config: AgentNodeConfig,
    state: WorkflowState,
    llm_client: FrameworkLLMClient,
    tools: list[ResearchTool],
    pools: dict[str, PoolState],
    *,
    url_registry: UrlRegistry | None = None,
    table_registry: TableRegistry | None = None,
    stream: bool = False,
    tool_call_cache: Any | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    execution_context: ExecutionContext | None = None,
) -> AgentOutput:
    """Execute a single agent node.

    Steps:
    1. Build AgentInput from state + config (resolve input_keys, render prompts)
    2. Call LLM (simple or ReAct loop if tools present)
    3. Parse output (text or structured)
    4. Write output to state
    5. Execute pool writes
    6. Return AgentOutput with events

    Args:
        node_id: Unique identifier for this node execution.
        config: Agent configuration from the workflow definition.
        state: Current workflow state.
        llm_client: LLM client for making completions.
        tools: Resolved tool instances for this agent.
        pools: Pool registry (name -> PoolState).
        url_registry: Shared URL registry for tool calls.
        stream: Whether to stream the final LLM response.

    Returns:
        AgentOutput with content, events, pool writes, token usage.
    """
    logger.info(
        "AGENT_START node=%s subtype=%s model_tier=%s output_key=%s",
        node_id, config.subtype, config.model_tier, config.output_key,
    )

    async with trace_span(
        f"agent.{config.subtype}",
        span_type="AGENT",
        attributes={
            "agent.node_id": node_id,
            "agent.subtype": config.subtype,
            "agent.model_tier": config.model_tier,
            "agent.output_key": config.output_key,
        },
    ) as agent_span:
        # -- 0a. Builtin enrichment ----------------------------------------------
        builtin = get_builtin(config.subtype)
        if builtin and builtin.enrich_config:
            config = builtin.enrich_config(config, state, dict(runtime_context or {}))

        # -- 0b. Keep prompt variables and input_keys in sync ------------------
        if config.system_prompt or config.user_prompt_template:
            renderer = SafeTemplateRenderer()
            detected_keys = (
                renderer.extract_variables(config.system_prompt)
                | renderer.extract_variables(config.user_prompt_template)
            )
            existing_keys = list(config.input_keys)
            merged_keys = sorted(set(existing_keys) | detected_keys)
            added_keys = sorted(set(merged_keys) - set(existing_keys))
            if merged_keys != existing_keys:
                config = config.model_copy(update={"input_keys": merged_keys})
                logger.info(
                    "AGENT_INPUT_KEYS_AUGMENTED node=%s existing=%s detected=%s added=%s merged=%s",
                    node_id,
                    existing_keys,
                    sorted(detected_keys),
                    added_keys,
                    merged_keys,
                )

        # -- 1. Build AgentInput ------------------------------------------------
        synthesis_context_usage: dict[str, int] = {}
        synthesis_context_stats: dict[str, Any] = {}
        synthesis_source_records: list[dict[str, Any]] = []
        if config.subtype == "synthesizer":
            typed_context = compile_typed_synthesis_context(
                state.runtime_state(),
                config=config.synthesis_context,
            )
            if typed_context is not None:
                synthesis_context = typed_context
            else:
                synthesis_context = await compile_synthesis_context(
                    query=state.query,
                    pools=pools,
                    llm_client=llm_client,
                    config=config.synthesis_context,
                )
            _write_synthesis_context_state(node_id, state, synthesis_context)
            synthesis_context_usage = synthesis_context.token_usage
            synthesis_context_stats = {
                "synth.compaction_applied": synthesis_context.stats.compaction_applied,
                "synth.observation_items_in": synthesis_context.stats.observation_items_in,
                "synth.observation_items_out": synthesis_context.stats.observation_items_out,
                "synth.source_items_in": synthesis_context.stats.source_items_in,
                "synth.source_clusters_out": synthesis_context.stats.source_clusters_out,
                "synth.context_chars_before": synthesis_context.stats.context_chars_before,
                "synth.context_chars_after": synthesis_context.stats.context_chars_after,
            }
            if state.runtime_store is not None:
                pack = state.runtime_store.build_synthesis_input_pack()
                if pack.observation_count == 0 and pack.source_count == 0:
                    state.runtime_store.set_synthesis_mode("insufficient")
                elif pack.blocked_reason_count > 0:
                    state.runtime_store.set_synthesis_mode("partial")
                else:
                    state.runtime_store.set_synthesis_mode("full")

            # -- Inject compiled synthesis context into template vars ---------
            if synthesis_context is not None:
                runtime_context = {} if runtime_context is None else dict(runtime_context)
                runtime_context["all_observations"] = synthesis_context.all_observations

                runtime_context["fallback_discovery_sources"] = synthesis_context.fallback_discovery_sources

                # Build numbered sources list AND structured records from the
                # SAME data source (pool) to guarantee index alignment.
                # The prompt source #N and synthesis_source_records[N-1] will
                # always refer to the same URL.
                #
                # Each bullet carries the URL-derived title, the URL, any
                # snippet, and a bounded slice of page content (when the tool
                # actually captured it). Previous behaviour was title+URL only
                # which starved the synthesiser of the page text it was told
                # to cite — a documented cause of fabricated tool/feature
                # names (see plan at come-with-a-very-rustling-sundae.md).
                numbered_lines: list[str] = []
                sources_pool = pools.get("sources")
                resolved_src_cfg = (
                    (config.synthesis_context.sources
                     if config.synthesis_context and config.synthesis_context.sources
                     else None)
                    or default_synthesis_context().sources
                )
                assert resolved_src_cfg is not None
                top_k = max(0, resolved_src_cfg.keep_full_top_k)
                cap_top = resolved_src_cfg.max_content_chars_top_k
                cap_tail = resolved_src_cfg.max_content_chars_other
                if sources_pool is not None:
                    seen_urls: set[str] = set()
                    for item in sources_pool.snapshot():
                        if not isinstance(item, dict) or not source_is_substantive(item):
                            continue
                        url = str(item.get("url", "") or "")
                        title = str(item.get("title", "") or "")
                        # Replace generic VS fallback titles with tool-level metadata
                        if re.match(r'^Vector search result \d+$', title):
                            source_desc = str(item.get("source_description", "") or "")
                            source_name = str(item.get("source_name", "") or "")
                            if source_desc:
                                title = source_desc[:120]
                            elif source_name:
                                title = source_name
                        if url and url in seen_urls:
                            continue
                        if url:
                            seen_urls.add(url)
                        idx = len(synthesis_source_records) + 1
                        label = title or url or f"Source {idx}"
                        if url:
                            # Truncate URL to prevent prompt bloat from tracking params
                            display_url = url[:150]
                            head_line = f"{idx}. {label} — {display_url}"
                        else:
                            head_line = f"{idx}. {label}"

                        entry_lines = [head_line]
                        snippet_raw = str(item.get("snippet", "") or "").strip()
                        content_raw = str(item.get("content", "") or "").strip()
                        if resolved_src_cfg.include_snippet and snippet_raw:
                            entry_lines.append(f"    Snippet: {snippet_raw}")
                        if resolved_src_cfg.include_content and content_raw:
                            cap = cap_top if (idx - 1) < top_k else cap_tail
                            body = content_raw
                            if cap > 0 and len(body) > cap:
                                body = body[:cap].rstrip() + "…"
                            entry_lines.append(f"    Content: {body}")
                        numbered_lines.append("\n".join(entry_lines))
                        synthesis_source_records.append({
                            "index": str(idx),
                            "url": url,
                            "title": title,
                            "snippet": str(
                                item.get("snippet", "") or item.get("content", "")
                            )[:200],
                        })
                runtime_context["sources_list"] = "\n".join(numbered_lines)

        agent_input, input_token_usage = await _build_input(
            node_id,
            config,
            state,
            llm_client,
            tools,
            pools,
            runtime_context=runtime_context,
        )

        # -- 2. Build messages ---------------------------------------------------
        messages = _build_messages(agent_input)

        # -- 3. Execute (simple or ReAct) ----------------------------------------
        events: list[StreamEvent] = []
        sources: list[Any] = []
        token_usage: dict[str, int] = merge_token_usage(
            synthesis_context_usage,
            input_token_usage,
        )
        delegated_to_react = False

        # Populate ToolContext.extras with reserved-prefix keys.
        # Precedence (later wins): Agent API path (state-stashed
        # ``_framework_extras``) -> orchestrator path
        # (``execution_context.user_id`` / ``approval_broker``). The
        # orchestrator-supplied values take priority on conflict so the
        # request-scoped identity wins over Agent dataclass defaults.
        tool_extras: dict[str, Any] = {}
        stashed_extras = state.get("_framework_extras")
        if isinstance(stashed_extras, dict):
            tool_extras.update(stashed_extras)
        if execution_context is not None:
            if execution_context.user_id is not None:
                tool_extras["_framework_user_id"] = execution_context.user_id
            if execution_context.approval_broker is not None:
                tool_extras["_framework_approval_broker"] = execution_context.approval_broker
        logger.info(
            "HARNESS_EXTRAS_POPULATED node=%s user_id_set=%s broker_set=%s",
            node_id,
            "_framework_user_id" in tool_extras,
            "_framework_approval_broker" in tool_extras,
        )
        tool_ctx = ToolContext(
            query=state.query,
            url_registry=url_registry,
            table_registry=table_registry,
            current_step=state.get("current_step"),
            background_summary=_resolve_background_summary(state),
            recent_observations=summarize_recent_observations(state.get_all("findings")),
            discovered_sources=_get_pool_items(pools, "discovery_sources", 10),
            extras=tool_extras,
        )

        builtin_result: AgentOutput | None = None
        if builtin and builtin.execute:
            builtin_result = await builtin.execute(
                node_id,
                config,
                state,
                llm_client,
                tools,
                pools,
                agent_input,
                messages,
                tool_ctx,
            )

        if builtin_result is not None:
            content = builtin_result.content
            events.extend(builtin_result.events)
            sources.extend(builtin_result.sources)
            token_usage = merge_token_usage(token_usage, builtin_result.token_usage)
        elif tools and (config.max_tool_calls or 0) > 0:
            # ReAct loop
            delegated_to_react = True
            loop = ReactLoop(
                llm_client,
                tools,
                tool_context=tool_ctx,
                cache=tool_call_cache,
                node_id=node_id,
                max_tool_calls=config.max_tool_calls or 20,
                model_tier=config.model_tier,
                stream=stream,
                subtype=config.subtype,
                max_result_chars=config.max_result_chars,
                compaction_strategy=config.compaction_strategy,
                keep_intact_iterations=config.keep_intact_iterations,
                dedup_jaccard_threshold=config.dedup_jaccard_threshold,
                force_convergence=config.force_convergence,
                convergence_rounds=config.convergence_rounds,
                per_tool_limits=config.per_tool_limits,
                hint_queries=config.hint_queries,
                suppress_planning_final_output=bool(
                    config.extras.get("suppress_planning_final_output", False)
                ),
            )
            result = await loop.execute(messages)
            content = result.content
            events.extend(result.events)
            sources.extend(result.sources)
            token_usage = merge_token_usage(token_usage, result.token_usage)
            # DR_LEAK_TRACE harness_return: capture what the harness pulled
            # out of the ReactResult. Compare to react_exit content_head — if
            # they differ, something between ReactLoop.execute and here
            # mutated the content.
            _content_str = content if isinstance(content, str) else (
                json.dumps(content, default=str) if content is not None else ""
            )
            logger.info(
                "DR_LEAK_TRACE phase=harness_return "
                "node=%s subtype=%s sources=%d content_len=%d content_head=%r",
                node_id,
                getattr(config, "subtype", None),
                len(sources),
                len(_content_str),
                _content_str[:300].replace("\n", "\\n"),
            )
        else:
            # Simple single LLM call
            response = await llm_client.complete(
                messages,
                config.model_tier,
                max_tokens=config.conversation_budget,
                structured_output=config.output_model,
                event_sink=events.append,
                node_id=node_id,
            )
            content = response.content
            token_usage = merge_token_usage(token_usage, response.usage)
            if response.structured is not None:
                content = response.structured

        # -- 4. Parse output -----------------------------------------------------
        parsed = _parse_output(content, config)
        parsed = _enrich_parsed_output(parsed, config, sources)
        normalized_research_output = _normalize_research_output(parsed, config, sources)
        skip_pool_writes = isinstance(parsed, dict) and bool(parsed.get("_skip_pool_writes"))
        if isinstance(parsed, dict) and "_skip_pool_writes" in parsed:
            parsed = {k: v for k, v in parsed.items() if k != "_skip_pool_writes"}

        state_output = parsed
        pool_output = parsed
        if config.subtype == "researcher":
            if normalized_research_output is not None:
                state_output, structured_findings = _project_research_state(
                    node_id, config, state, normalized_research_output
                )
                pool_output = structured_findings
                if state.runtime_store is not None:
                    source_records = build_source_records(
                        normalized_research_output.sources
                    )
                    step_id = None
                    current_step = state.get(config.item_state_key) if hasattr(config, "item_state_key") else None
                    if isinstance(current_step, dict):
                        step_id = str(current_step.get("id", "") or "") or None
                    observation_records = build_observation_records(normalized_research_output, step_id=step_id)
                    delta = state.runtime_store.ingest_evidence(
                        producer_node_id=node_id,
                        sources=source_records,
                        observations=observation_records,
                    )
                    state.runtime_store.publish_artifact(
                        artifact_id=f"research_outcome_{node_id}_{len(state.runtime_store.runtime().artifacts)}",
                        artifact_type="research_outcome",
                        producer_node_id=node_id,
                        payload={
                            "status": normalized_research_output.research_status,
                            "blocking_reason": normalized_research_output.blocking_reason,
                            "queries": normalized_research_output.search_queries,
                            "evidence_delta": delta.model_dump(mode="json"),
                        },
                        status=(
                            "blocked" if normalized_research_output.research_status == "blocked" else
                            "degraded" if normalized_research_output.research_status == "insufficient_data" else
                            "success"
                        ),
                        substantive=bool(observation_records or source_records),
                    )
                logger.info(
                    "RESEARCH_OUTPUT_NORMALIZED node=%s status=%s repair_mode=%s queries=%d",
                    node_id,
                    normalized_research_output.research_status,
                    normalized_research_output.repair_mode or "",
                    len(normalized_research_output.search_queries),
                )
            elif isinstance(parsed, dict):
                structured_findings = dict(parsed)
                state_output = str(
                    structured_findings.get("observation")
                    or structured_findings.get("findings")
                    or ""
                )
                state.append(node_id, f"{config.output_key}_structured", structured_findings)
                for key in ("research_status", "blocking_reason", "search_queries", "key_points", "sources_used"):
                    if key in structured_findings:
                        state.append(node_id, key, structured_findings.get(key))
                logger.info(
                    "RESEARCHER_STRUCTURED_OUTPUT node=%s status=%s blocking_reason=%r queries=%d",
                    node_id,
                    structured_findings.get("research_status", "ok"),
                    structured_findings.get("blocking_reason"),
                    len(structured_findings.get("search_queries", []) or []),
                )

        # -- 5. Write to state ---------------------------------------------------
        # DR_LEAK_TRACE state_write: capture the value that lands in state
        # under output_key. Downstream agents read this — if it differs from
        # the react_exit content, something between react_loop and here
        # mutated the content (output normalization, projection, parsing).
        try:
            _state_value_str = (
                state_output if isinstance(state_output, str)
                else json.dumps(state_output, default=str, ensure_ascii=False)
            )
            logger.info(
                "DR_LEAK_TRACE phase=state_write origin=agent "
                "node=%s subtype=%s output_key=%s value_len=%d value_head=%r",
                node_id,
                getattr(config, "subtype", None),
                config.output_key,
                len(_state_value_str),
                _state_value_str[:300].replace("\n", "\\n"),
            )
        except Exception as _exc:  # pragma: no cover — diagnostic only
            logger.debug("DR_LEAK_TRACE state_write skipped: %s", _exc)
        state.append(node_id, config.output_key, state_output)

        # -- 6. Pool writes ------------------------------------------------------
        pool_writes: dict[str, list[Any]] = {}
        normalized_batches = _build_pool_batches(normalized_research_output, config.pool_writes, config.output_key)
        for pw in config.pool_writes:
            if skip_pool_writes:
                allow_sources = pw.extract == "sources" and bool(sources)
                allow_observation_fallback = (
                    config.subtype == "researcher"
                    and pw.extract in {config.output_key, "findings", "observation"}
                    and isinstance(state_output, str)
                    and bool(state_output.strip())
                )
                if not (allow_sources or allow_observation_fallback):
                    logger.info(
                        "POOL_WRITE_SKIP_BLOCKED_OUTPUT pool=%s node=%s extract=%s",
                        pw.pool, node_id, pw.extract,
                    )
                    continue
            batch = normalized_batches.get((pw.pool, pw.extract))
            items = list(batch.items) if batch is not None else _extract_pool_items(pool_output, pw, config.output_key)
            # Fallback: if extraction found nothing and the ReAct loop collected
            # sources separately, use those for any pool_write targeting "sources".
            if not items and pw.extract == "sources":
                normalized_sources = []
                if isinstance(pool_output, dict):
                    raw_normalized_sources = pool_output.get("sources", [])
                    if isinstance(raw_normalized_sources, list):
                        normalized_sources = [
                            item
                            for item in raw_normalized_sources
                            if not _is_semantically_empty(item)
                            and source_is_substantive(item)
                        ]
                if normalized_sources:
                    items = normalized_sources
                    logger.info(
                        "POOL_WRITE_NORMALIZED_SOURCES pool=%s count=%d",
                        pw.pool, len(items),
                    )
                elif sources:
                    items = [
                        serialized
                        for serialized in (_serialize_source_for_pool(s) for s in sources)
                        if source_is_substantive(serialized)
                    ]
                    logger.info(
                        "POOL_WRITE_REACT_SOURCES pool=%s count=%d",
                        pw.pool, len(items),
                    )
            if (
                not items
                and config.subtype == "researcher"
                and pw.extract in {config.output_key, "findings", "observation"}
                and isinstance(state_output, str)
                and state_output.strip()
            ):
                items = [state_output]
                logger.info(
                    "POOL_WRITE_OBSERVATION_FALLBACK pool=%s output_key=%s len=%d",
                    pw.pool, config.output_key, len(state_output),
                )
            if items and pw.pool in pools:
                async with trace_span(
                    f"pool.write.{pw.pool}",
                    span_type="TOOL",
                    attributes={
                        "pool.name": pw.pool,
                        "pool.extract": pw.extract,
                        "pool.items_written": len(items),
                        "pool.total_after": pools[pw.pool].count() + len(items),
                    },
                ):
                    pool_writes[pw.pool] = items
                    for item in items:
                        pools[pw.pool].add(item)
                logger.info(
                    "POOL_WRITE_SUCCESS pool=%s items=%d total=%d",
                    pw.pool, len(items), pools[pw.pool].count(),
                )
            elif pw.pool in pools:
                async with trace_span(
                    f"pool.write_empty.{pw.pool}",
                    span_type="TOOL",
                    attributes={
                        "pool.name": pw.pool,
                        "pool.extract": pw.extract,
                        "pool.output_type": type(parsed).__name__,
                        "pool.state_output_type": type(state_output).__name__,
                        "pool.output_key": config.output_key,
                    },
                ):
                    pass
                logger.warning(
                    "POOL_WRITE_EMPTY pool=%s extract=%s output_type=%s output_preview=%s",
                    pw.pool, pw.extract, type(pool_output).__name__, str(pool_output)[:100],
                )
                if batch is not None and batch.skip_reason:
                    logger.info(
                        "POOL_WRITE_SKIPPED pool=%s reason=%s",
                        pw.pool,
                        batch.skip_reason,
                    )

        logger.info(
            "AGENT_OUTPUT node=%s output_key=%s output_len=%d type=%s",
            node_id, config.output_key, len(str(state_output)), type(state_output).__name__,
        )

        if agent_span:
            span_attrs: dict[str, Any] = {
                "agent.output_len": len(str(state_output)),
                "agent.output_type": type(state_output).__name__,
                "agent.delegated_to_react": delegated_to_react,
            }
            span_attrs.update(synthesis_context_stats)
            # Only report tokens directly when NOT using ReAct loop
            # (ReAct loop reports via child react_loop.* span)
            if not delegated_to_react:
                for k, v in token_usage.items():
                    span_attrs[f"agent.{k}"] = v
            agent_span.set_attributes(span_attrs)

        # -- 7. Emit output event ------------------------------------------------
        preview = str(state_output)[:200] if state_output else ""
        events.append(AgentOutputEvent(
            node_id=node_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            output_key=config.output_key,
            output_preview=preview,
        ))
        if state.runtime_store is not None:
            runtime = state.runtime_store.runtime()
            if config.subtype == "coordinator" and hasattr(state_output, "complexity"):
                with contextlib.suppress(Exception):
                    state.runtime_store.set_coordination(state_output)
            if node_id in runtime.nodes:
                runtime.nodes[node_id].output_key = config.output_key
                runtime.nodes[node_id].output_preview = preview[:400]

        # -- Fallback: inject pool sources if LLM didn't populate them ----
        if config.subtype == "synthesizer" and synthesis_source_records:
            _inject_sources_into_output(state_output, synthesis_source_records)

        # -- Post-process: compute citation_stats from actual source_refs ---
        if config.subtype == "synthesizer" and state_output is not None:
            _compute_citation_stats(state_output, len(synthesis_source_records))

        # -- 8. Builtin post-processing ------------------------------------------
        if state.runtime_store is not None and config.subtype == "synthesizer":
            mode = "full"
            runtime = state.runtime_store.runtime()
            if runtime.capabilities.synthesis is not None:
                mode = runtime.capabilities.synthesis.mode
            state.runtime_store.publish_report_artifact(
                producer_node_id=node_id,
                report=state_output,
                mode=mode,
            )

        if builtin and builtin.post_process:
            domain_events = builtin.post_process(node_id, state_output, config, state)
            events.extend(domain_events)

        return AgentOutput(
            content=state_output,
            output_key=config.output_key,
            pool_writes=pool_writes,
            sources=sources,
            token_usage=token_usage,
            events=events,
        )


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


async def _build_input(
    _node_id: str,
    config: AgentNodeConfig,
    state: WorkflowState,
    llm_client: FrameworkLLMClient,
    tools: list[ResearchTool],
    pools: dict[str, PoolState],
    *,
    runtime_context: Mapping[str, Any] | None = None,
) -> tuple[AgentInput, dict[str, int]]:
    """Construct AgentInput from state and config."""
    # Resolve input_keys from state
    context: dict[str, Any] = {}
    for key in config.input_keys:
        resolved = None
        if "." in key:
            resolved = state.get_nested(key)
        else:
            resolved = resolve_input_key(state, key)
            if resolved is None:
                resolved = state.get(key)
        context[key] = resolved

    for key, value in dict(runtime_context or {}).items():
        if key == "query":
            continue
        context[key] = value

    # Extract the reserved chat-memory appendix before template rendering.
    # The double-underscore prefix signals a reserved state key that must NOT
    # become a template variable — it is appended to the rendered system
    # prompt as an untrusted <attached_context> block via
    # inject_attached_context_block. Seeded by the orchestrator at turn start
    # (see deep_research.agent.framework_orchestrator). When memory is empty
    # or disabled the appendix is an empty string and injection is a no-op,
    # preserving byte-identical backward compatibility.
    chat_memory_appendix: str = ""
    state_appendix = state.get(CHAT_MEMORY_APPENDIX_STATE_KEY)
    if isinstance(state_appendix, str):
        chat_memory_appendix = state_appendix
    context_appendix = context.pop(CHAT_MEMORY_APPENDIX_STATE_KEY, None)
    if isinstance(context_appendix, str) and context_appendix:
        chat_memory_appendix = context_appendix

    # Auto-inject compute namespace summary for downstream agents.
    # Enables prompts to reference {compute_namespace} without discovery calls.
    # We only write the key when at least one tool actually exposes a
    # namespace_snapshot() — agents with no compute tool bound (e.g. the
    # designer architect) get no compute_namespace key at all, which keeps
    # the variable absent from their prompt context rather than leaking a
    # misleading "(compute tool not available)" placeholder.
    ns_summary: str | None = None
    for tool in tools:
        if hasattr(tool, "namespace_snapshot") and callable(tool.namespace_snapshot):
            try:
                ns_summary = tool.namespace_snapshot()
            except Exception:
                logger.warning("NAMESPACE_SNAPSHOT_FAILED node=%s", _node_id, exc_info=True)
                ns_summary = "(error reading compute namespace)"
            break
    if ns_summary is not None and context.get("compute_namespace") is None:
        context["compute_namespace"] = ns_summary

    # Auto-inject temporal context (current_date, current_iso_datetime,
    # current_timezone) unless the caller has already provided one. Caller
    # overrides are honoured so regression tests can fix the clock (set any
    # of the three keys to anchor the run deterministically).
    if context.get("current_date") is None:
        from databricks_deep_research.agents.temporal import PromptTemporalContext
        context.update(PromptTemporalContext.now().as_context_keys())
        logger.info(
            "TEMPORAL_CONTEXT_INJECTED node=%s date=%s tz=%s",
            _node_id, context.get("current_date"), context.get("current_timezone"),
        )

    # Auto-inject the synthesizer revision block. Computed best-effort from
    # state — returns empty when the workflow has no reflector pass or the
    # reflector's decision was not 'adjust'. Cost is two state lookups +
    # one Pydantic validation when applicable; near-zero otherwise.
    if context.get("revision_block_md") is None:
        try:
            from databricks_deep_research.agents.revision import build_revision_block_md
            # WorkflowState exposes per-key access via .get(), not a values
            # dict; we build the minimal three-key snapshot the revision
            # builder needs to keep its API dict-shaped (testable in isolation).
            state_snapshot: dict[str, Any] = {}
            for _rk in ("draft_report", "coverage_review", "revision_passes_remaining"):
                _v = state.get(_rk) if hasattr(state, "get") else None
                if _v is not None:
                    state_snapshot[_rk] = _v
            block_md = build_revision_block_md(state_snapshot)
        except Exception:  # noqa: BLE001 — never fail the run because of a hook
            block_md = ""
            logger.exception("REVISION_BLOCK_BUILD_FAILED node=%s", _node_id)
        if block_md:
            context["revision_block_md"] = block_md
            logger.info(
                "REVISION_BLOCK_INJECTED node=%s chars=%d",
                _node_id, len(block_md),
            )

    sources_pool = pools.get("sources")
    if sources_pool is not None and sources_pool.count() > 0:
        pooled_sources = sources_pool.snapshot()
        substantive = sum(1 for item in pooled_sources if source_is_substantive(item))
        low_value = sum(1 for item in pooled_sources if source_is_low_value(item))
        quality_counts: dict[str, int] = {}
        for item in pooled_sources:
            if isinstance(item, dict):
                quality = str(item.get("evidence_quality", "unknown"))
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
        if context.get("source_quality") is None:
            context["source_quality"] = {
                "substantive_sources": substantive,
                "low_value_sources": low_value,
                "quality_counts": quality_counts,
            }

    # Render prompts with variable substitution.
    # ``state.query`` is always available; context keys only override when
    # they resolve to a non-None value so that ``query`` from the state
    # constructor is never clobbered by an empty log lookup.
    renderer = SafeTemplateRenderer()
    context_vars = {k: _serialize_for_context(v) for k, v in context.items()}

    resolved_preview = {k: (v[:80] if v else "<empty>") for k, v in context_vars.items()}
    logger.info(
        "AGENT_CONTEXT node=%s input_keys=%s resolved=%s",
        _node_id, config.input_keys, resolved_preview,
    )

    # Materialized per-agent tool catalog block. Runtime resolution prefers
    # save-time prose when the declaration hash + registry version match and
    # falls back to the same pure renderer when the persisted block is absent
    # or stale.
    from databricks_deep_research.tools.catalog_service import CatalogService

    tool_catalog_text = CatalogService.from_default_factories().resolve_for_runtime(
        config,
        tools,
        node_id=_node_id,
    )
    template_vars = {
        "query": state.query,
        "tool_catalog": tool_catalog_text,
        **{k: v for k, v in context_vars.items() if v},
    }

    template_var_preview = {
        k: (str(v)[:60] if v else "<empty>") for k, v in template_vars.items()
    }
    logger.info(
        "AGENT_TEMPLATE_VARS node=%s keys=%s preview=%s",
        _node_id, list(template_vars.keys()), template_var_preview,
    )

    has_tool_catalog_placeholder = "{tool_catalog}" in config.system_prompt
    system_prompt = renderer.render(config.system_prompt, template_vars)
    if tool_catalog_text and not has_tool_catalog_placeholder:
        system_prompt = f"{system_prompt}\n\n{tool_catalog_text}"
    user_prompt = renderer.render(config.user_prompt_template, template_vars)

    # Append the chat-memory <attached_context> block to the system prompt.
    # No-op when chat_memory_appendix is empty; preserves byte-identical
    # prompts for workflows without memory. See injection.py for the helper.
    system_prompt = inject_attached_context_block(system_prompt, chat_memory_appendix)

    logger.info(
        "AGENT_PROMPTS node=%s system_len=%d user_len=%d "
        "system_preview=%s user_preview=%s chat_memory_appendix_len=%d",
        _node_id, len(system_prompt), len(user_prompt),
        system_prompt[:150], user_prompt[:300],
        len(chat_memory_appendix),
    )

    # Pool injection (small pools injected directly into prompt)
    pool_sections: dict[str, Any] = {}
    pool_token_usage: dict[str, int] = {}
    for pi in config.pool_inject:
        pool = pools.get(pi.pool)
        if pool:
            section = await compile_pool_section(
                pool_name=pi.pool,
                pool=pool,
                query=state.query,
                config=pi,
                llm_client=llm_client,
            )
            pool_sections[pi.pool] = section
            pool_token_usage = merge_token_usage(pool_token_usage, section.token_usage)
            logger.info(
                "AGENT_POOL_INJECT node=%s pool=%s items_injected=%d "
                "pool_total=%d compacted=%s preview=%s",
                _node_id, pi.pool, len(section.raw_items), pool.count(),
                section.compacted,
                section.rendered_text[:100] if section.rendered_text else "<empty>",
            )
        else:
            logger.warning("POOL_INJECT_MISSING pool=%s node=%s", pi.pool, _node_id)

    return (
        AgentInput(
            query=state.query,
            context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=[t.definition for t in tools],
            pool_sections=pool_sections,
            conversation_history=state.conversation_history,  # wire multi-turn history into agent
        ),
        pool_token_usage,
    )


def _resolve_background_summary(state: WorkflowState) -> str:
    return select_background_summary(state)


def _write_synthesis_context_state(
    node_id: str,
    state: WorkflowState,
    context: CompiledSynthesisContext,
) -> None:
    """Persist compiled synthesizer context under legacy prompt variable keys only when needed."""
    if state.runtime_store is not None:
        return
    state.append(node_id, "all_observations", context.all_observations)
    state.append(node_id, "sources_list", context.sources_list)
    state.append(
        node_id,
        "fallback_discovery_sources",
        context.fallback_discovery_sources,
    )


def _inject_sources_into_output(
    output: Any, source_records: list[dict[str, Any]]
) -> None:
    """Inject or enrich source records in synthesizer output.

    Three behaviors:
    1. If output.sources is empty → replace with source_records (existing)
    2. If output.sources has items but some lack URLs → enrich via index match (NEW)
    3. If output.sources is fully populated → no-op
    """
    # -- Phase 1: Replace empty sources (existing behavior) ---
    if isinstance(output, dict):
        if not output.get("sources"):
            output["sources"] = source_records
    elif hasattr(output, "sources") and not output.sources:
        try:
            output.sources = source_records
        except Exception:
            logger.warning(
                "SYNTH_SOURCE_INJECT_FAILED output_type=%s",
                type(output).__name__,
            )

    # -- Phase 2: Enrich existing sources that lack URLs ---
    existing: list[Any] | None = None
    if isinstance(output, dict):
        existing = output.get("sources")
    elif hasattr(output, "sources"):
        existing = output.sources

    if not existing or not source_records:
        return

    record_by_idx = {r["index"]: r for r in source_records}
    enriched_count = 0

    for src in existing:
        try:
            if isinstance(src, dict):
                idx = str(src.get("index", "") or src.get("id", ""))
                record = record_by_idx.get(idx)
                if record and not src.get("url"):
                    src["url"] = record["url"]
                    enriched_count += 1
                if record and not src.get("snippet"):
                    src["snippet"] = record.get("snippet", "")
            elif hasattr(src, "index"):
                idx = str(src.index)
                record = record_by_idx.get(idx)
                if record and not getattr(src, "url", ""):
                    src.url = record["url"]
                    enriched_count += 1
                if record and not getattr(src, "snippet", ""):
                    src.snippet = record.get("snippet", "")
        except Exception:
            continue

    if enriched_count:
        logger.info(
            "SYNTH_SOURCE_ENRICH enriched=%d/%d sources",
            enriched_count, len(existing),
        )


def _compute_citation_stats(output: Any, total_sources: int) -> None:
    """Count actual citations across output and update citation_stats in-place.

    Structured synthesizer outputs expose citation fields as ``source_refs``.
    Reclaim synthesis returns markdown with numeric ``[N]`` markers instead,
    so count those markers too; otherwise traces misleadingly report zero
    citations even when the citation pipeline verified and numbered the report.
    """
    all_refs: list[str] = []
    fields_with_refs = 0
    fields_total = 0

    def _collect(obj: Any) -> None:
        nonlocal fields_with_refs, fields_total
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key.endswith("source_refs"):
                    fields_total += 1
                    if isinstance(val, list) and val:
                        fields_with_refs += 1
                        all_refs.extend(str(v) for v in val)
                elif isinstance(val, (dict, list)):
                    _collect(val)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _collect(item)

    try:
        data = output.model_dump() if hasattr(output, "model_dump") else output
        if isinstance(data, dict):
            _collect(data)
        elif isinstance(data, str):
            refs = re.findall(r"\[(\d+)\]", data)
            fields_total = 1 if data else 0
            if refs:
                fields_with_refs = 1
                all_refs.extend(refs)
    except Exception:
        return

    valid = sum(
        1 for r in all_refs
        if r.isdigit() and 1 <= int(r) <= max(total_sources, 1)
    )

    # Mutate existing CitationStats in-place (avoids type replacement issues)
    cs = getattr(output, "citation_stats", None)
    if cs is not None and hasattr(cs, "total_citations"):
        try:
            cs.total_citations = len(all_refs)
            cs.valid_citations = valid
            cs.invalid_citations = len(all_refs) - valid
            cs.coverage_percentage = (
                round(fields_with_refs / fields_total * 100, 1)
                if fields_total > 0 else 0.0
            )
        except Exception:
            pass
    elif isinstance(output, dict):
        output["citation_stats"] = {
            "total_citations": len(all_refs),
            "valid_citations": valid,
            "invalid_citations": len(all_refs) - valid,
            "coverage_percentage": (
                round(fields_with_refs / fields_total * 100, 1)
                if fields_total > 0 else 0.0
            ),
        }

    logger.info(
        "CITATION_STATS total=%d valid=%d invalid=%d coverage=%.1f%% fields=%d/%d",
        len(all_refs), valid, len(all_refs) - valid,
        round(fields_with_refs / fields_total * 100, 1) if fields_total else 0,
        fields_with_refs, fields_total,
    )


def _get_pool_items(
    pools: dict[str, PoolState],
    pool_name: str,
    max_items: int,
) -> list[Any]:
    pool = pools.get(pool_name)
    if pool is None or pool.count() == 0:
        return []
    return pool.get_recent(max_items)


def _build_messages(agent_input: AgentInput) -> list[dict[str, Any]]:
    """Build OpenAI-format messages from AgentInput."""
    messages: list[dict[str, Any]] = []

    if agent_input.system_prompt:
        messages.append({"role": "system", "content": agent_input.system_prompt})

    # Build user message with context and pool content
    parts: list[str] = []
    if agent_input.user_prompt:
        parts.append(agent_input.user_prompt)
    elif agent_input.query:
        parts.append(agent_input.query)

    # Inject pool content into user message
    for pool_name, section in agent_input.pool_sections.items():
        if section.rendered_text:
            parts.append(f"\n## {pool_name}\n{section.rendered_text}")

    # Add conversation history if present
    for msg in agent_input.conversation_history:
        messages.append(msg)

    if parts:
        messages.append({"role": "user", "content": "\n\n".join(parts)})

    # DR_LEAK_TRACE synth_prompt_in: capture what each agent's user message
    # contains (especially synthesizers/reflectors that read lane outputs).
    # Log per-pool snippets so we can see which lane's content surfaces.
    try:
        _pool_dump = {
            pool_name: (section.rendered_text or "")[:300].replace("\n", "\\n")
            for pool_name, section in agent_input.pool_sections.items()
        }
        _sys_head = (agent_input.system_prompt or "")[:150].replace("\n", "\\n")
        _user_head = ("\n\n".join(parts))[:300].replace("\n", "\\n")
        logger.info(
            "DR_LEAK_TRACE phase=synth_prompt_in "
            "system_head=%r user_head=%r pools=%r",
            _sys_head,
            _user_head,
            _pool_dump,
        )
    except Exception as _exc:  # pragma: no cover — diagnostic only
        logger.debug("DR_LEAK_TRACE synth_prompt_in skipped: %s", _exc)

    return messages


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_output(content: Any, config: AgentNodeConfig) -> Any:
    """Parse agent output based on config.output_format."""
    # Already a Pydantic model (structured output)
    if not isinstance(content, str):
        logger.info(
            "AGENT_OUTPUT_PARSE format=%s input_type=%s output_type=%s output_len=%d",
            config.output_format, type(content).__name__,
            type(content).__name__, len(str(content)),
        )
        return content

    if config.output_format == "json":
        try:
            parsed = json.loads(content)
            logger.info(
                "AGENT_OUTPUT_PARSE format=json input_type=str "
                "output_type=%s output_len=%d",
                type(parsed).__name__, len(str(parsed)),
            )
            return parsed
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    try:
                        parsed = json.loads(content[start:end].strip())
                        logger.info(
                            "AGENT_OUTPUT_PARSE format=json_codeblock input_type=str "
                            "output_type=%s output_len=%d",
                            type(parsed).__name__, len(str(parsed)),
                        )
                        return parsed
                    except json.JSONDecodeError:
                        pass
            # Try json_repair if available
            try:
                import json_repair
                parsed = json_repair.loads(content)
                if _env_flag(_RESILIENT_JSON_REPAIR_ENV, default=True):
                    suspicion = _is_suspicious_repair(
                        parsed, content, subtype=config.subtype
                    )
                    if suspicion is not None:
                        raise ValueError(f"json_repair suspicious: {suspicion}")
                else:
                    # Legacy behavior — narrower checks, kept for rollback.
                    if parsed in ("", None) and content.strip():
                        raise ValueError("json_repair produced empty output")
                    if (
                        config.subtype == "researcher"
                        and isinstance(parsed, dict)
                        and not parsed
                        and _has_substantive_text(content, min_length=20)
                    ):
                        raise ValueError(
                            "json_repair produced empty dict from substantive text"
                        )
                logger.info(
                    "AGENT_OUTPUT_PARSE format=json_repaired input_type=str "
                    "output_type=%s output_len=%d content_len=%d",
                    type(parsed).__name__, len(str(parsed)), len(content),
                )
                return parsed
            except (ImportError, Exception) as exc:
                logger.warning(
                    "JSON_PARSE_FAILURE reason=%s content_preview=%s",
                    str(exc)[:100], content[:200],
                )
                return _UnparsedJSONOutput(content)

    # text or markdown — return as-is
    logger.info(
        "AGENT_OUTPUT_PARSE format=%s input_type=str "
        "output_type=str output_len=%d",
        config.output_format, len(content),
    )
    return content


def _build_pool_write_batch(
    normalized: NormalizedResearchOutput,
    pool_write: PoolWriteConfig,
    output_key: str,
) -> list[Any]:
    if pool_write.extract == "sources":
        return [] if normalized.skip_source_writes else list(normalized.sources)
    if pool_write.extract in {output_key, "findings", "observation"}:
        text_value = normalized.observation_text or normalized.findings_text or normalized.state_text
        return [] if normalized.skip_observation_writes or not text_value.strip() else [text_value]
    return []


def _enrich_parsed_output(
    parsed: Any,
    config: AgentNodeConfig,
    sources: list[Any],
) -> Any:
    """Attach ReAct-collected context to structured outputs when needed."""
    serialized_sources = [_serialize_source_for_pool(source) for source in sources]

    if isinstance(parsed, _UnparsedJSONOutput):
        if config.subtype == "planner":
            raise WorkflowError("Malformed structured output for planner")
        if config.subtype == "researcher":
            raw_text = str(parsed).strip()
            has_substantive_text = _has_substantive_text(raw_text)
            return {
                config.output_key: raw_text if has_substantive_text else "",
                "findings": raw_text if has_substantive_text else "",
                "observation": raw_text if has_substantive_text else "",
                "search_queries": [],
                "key_points": [],
                "sources_used": [],
                "research_status": (
                    "ok"
                    if has_substantive_text
                    else ("insufficient_data" if serialized_sources else "blocked")
                ),
                "blocking_reason": None if has_substantive_text else "malformed_structured_output",
                "sources_found": len(serialized_sources),
                "sources": serialized_sources,
                "_skip_pool_writes": not has_substantive_text and not serialized_sources,
            }
        if config.subtype in {"reflector", "evaluator"}:
            return {
                "decision": "replan",
                "reasoning": (
                    "Structured output parsing failed for the evaluator/reflector "
                    "response. Replanning to recover from malformed output."
                ),
                "suggested_changes": [],
                "_skip_pool_writes": True,
            }
        return str(parsed)

    if config.subtype not in {"background", "researcher"}:
        return parsed

    if config.subtype == "researcher":
        if isinstance(parsed, str):
            return {
                config.output_key: parsed,
                "findings": parsed,
                "observation": parsed,
                "search_queries": [],
                "key_points": [],
                "sources_used": [],
                "research_status": "ok" if parsed.strip() else "insufficient_data",
                "blocking_reason": None,
                "sources_found": len(serialized_sources),
                "sources": serialized_sources,
            }

        if hasattr(parsed, "model_dump"):
            parsed_dict = parsed.model_dump()
        elif isinstance(parsed, dict):
            parsed_dict = dict(parsed)
        else:
            return parsed

        if not parsed_dict and serialized_sources:
            parsed_dict = {"sources": serialized_sources}

        observation = str(
            parsed_dict.get("observation")
            or parsed_dict.get("findings")
            or parsed_dict.get(config.output_key)
            or ""
        )
        if not observation.strip() and serialized_sources:
            synthesized_observation = _build_observation_from_sources(serialized_sources)
            if synthesized_observation:
                observation = synthesized_observation
        existing_sources = parsed_dict.get("sources", [])
        if not isinstance(existing_sources, list):
            existing_sources = []
        merged_sources = merge_and_dedup_sources(existing_sources, serialized_sources)
        parsed_dict[config.output_key] = observation
        parsed_dict["findings"] = observation
        parsed_dict.setdefault("observation", observation)
        parsed_dict.setdefault("search_queries", [])
        parsed_dict.setdefault("key_points", [])
        parsed_dict.setdefault("sources_used", [])
        parsed_dict.setdefault(
            "research_status",
            "ok" if observation.strip() else "insufficient_data",
        )
        parsed_dict.setdefault("blocking_reason", None)
        parsed_dict["sources_found"] = len(merged_sources)
        parsed_dict["sources"] = merged_sources
        if (
            _is_semantically_empty(observation)
            and _is_semantically_empty(parsed_dict.get("search_queries", []))
            and _is_semantically_empty(parsed_dict.get("key_points", []))
            and _is_semantically_empty(parsed_dict.get("sources", []))
        ):
            parsed_dict["research_status"] = "blocked"
            parsed_dict["blocking_reason"] = parsed_dict.get("blocking_reason") or "empty_research_output"
            parsed_dict["_skip_pool_writes"] = True
        elif _is_semantically_empty(observation) and serialized_sources:
            parsed_dict["research_status"] = "insufficient_data"
            parsed_dict["_skip_pool_writes"] = False
        return parsed_dict

    if isinstance(parsed, str):
        return {
            "summary": parsed,
            "query_decomposition": [],
            "data_landscape": _build_data_landscape(serialized_sources),
            "discovered_sources": serialized_sources,
        }

    if hasattr(parsed, "model_dump"):
        parsed_dict = parsed.model_dump()
    elif isinstance(parsed, dict):
        parsed_dict = dict(parsed)
    else:
        return parsed

    if not parsed_dict.get("discovered_sources"):
        parsed_dict["discovered_sources"] = serialized_sources
    if not parsed_dict.get("data_landscape"):
        parsed_dict["data_landscape"] = _build_data_landscape(serialized_sources)
    return parsed_dict


def _build_data_landscape(sources: list[Any]) -> dict[str, Any]:
    """Group discovered sources into a planner-friendly landscape summary."""
    grouped: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_name = str(
            source.get("source_name")
            or source.get("index_name")
            or source.get("source_type")
            or "unknown_source"
        )
        group = grouped.setdefault(source_name, {
            "source_name": source_name,
            "source_type": source.get("source_type", "unknown"),
            "document_count": 0,
            "sample_titles": [],
        })
        group["document_count"] += 1
        title = str(source.get("title", "") or "")
        if title and title not in group["sample_titles"] and len(group["sample_titles"]) < 5:
            group["sample_titles"].append(title)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda item: int(item.get("document_count", 0)),
        reverse=True,
    )
    return {
        "sources": ordered_groups,
        "top_sources": [item["source_name"] for item in ordered_groups[:5]],
    }
