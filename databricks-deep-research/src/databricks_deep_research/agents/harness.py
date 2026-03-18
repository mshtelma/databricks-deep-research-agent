"""Agent execution harness.

Constructs AgentInput from state + config, calls LLM (optionally via ReAct
loop), parses AgentOutput, writes to state + pools, and emits events.

This is the single point of interaction between the workflow executor and
an agent node.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import get_builtin
from databricks_deep_research.agents.config import AgentNodeConfig, PoolWriteConfig
from databricks_deep_research.agents.execution.output_normalizer import (
    build_observation_records,
    build_source_records,
    NormalizedResearchOutput,
    build_observation_from_sources as _build_observation_from_sources,
    has_substantive_text as _has_substantive_text,
    is_semantically_empty as _is_semantically_empty,
    merge_and_dedup_sources,
    normalize_research_output as _normalize_research_output_impl,
)
from databricks_deep_research.agents.execution.pool_projection import (
    PoolWriteBatch,
    build_research_pool_batch,
    extract_pool_items as _extract_pool_items,
)
from databricks_deep_research.agents.execution.state_projection import (
    project_research_state as _project_research_state_impl,
)
from databricks_deep_research.agents.isolation import AgentInput, AgentOutput
from databricks_deep_research.agents.prompt_context import (
    compile_typed_synthesis_context,
    CompiledSynthesisContext,
    compile_pool_section,
    compile_synthesis_context,
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
from databricks_deep_research.pools.pool_state import PoolState
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.tools.protocol import ResearchTool, ToolContext, UrlRegistry
from databricks_deep_research.tracing import trace_span
from databricks_deep_research.workflow.state import WorkflowState
from databricks_deep_research.workflow.runtime_core.selectors import resolve_input_key, select_background_summary

logger = logging.getLogger(__name__)


class _UnparsedJSONOutput(str):
    """Marker for JSON-configured agent output that could not be parsed."""


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
        return value.model_dump_json(indent=2)
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
    return _normalize_research_output_impl(parsed, config, serialized_sources)


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
    stream: bool = False,
    tool_call_cache: Any | None = None,
    runtime_context: Mapping[str, Any] | None = None,
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
            typed_context = compile_typed_synthesis_context(state.runtime_state())
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
                if runtime_context is None:
                    runtime_context = {}
                else:
                    runtime_context = dict(runtime_context)
                runtime_context["all_observations"] = synthesis_context.all_observations

                runtime_context["fallback_discovery_sources"] = synthesis_context.fallback_discovery_sources

                # Build numbered sources list AND structured records from the
                # SAME data source (pool) to guarantee index alignment.
                # The prompt source #N and synthesis_source_records[N-1] will
                # always refer to the same URL.
                numbered_lines: list[str] = []
                sources_pool = pools.get("sources")
                if sources_pool is not None:
                    seen_urls: set[str] = set()
                    for item in sources_pool.snapshot():
                        if not isinstance(item, dict):
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
                            numbered_lines.append(f"{idx}. {label} — {display_url}")
                        else:
                            numbered_lines.append(f"{idx}. {label}")
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

        tool_ctx = ToolContext(
            query=state.query,
            url_registry=url_registry,
            current_step=state.get("current_step"),
            background_summary=_resolve_background_summary(state),
            recent_observations=summarize_recent_observations(state.get_all("findings")),
            discovered_sources=_get_pool_items(pools, "discovery_sources", 10),
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
                max_tool_calls=config.max_tool_calls,
                model_tier=config.model_tier,
                stream=stream,
                subtype=config.subtype,
                max_result_chars=config.max_result_chars,
            )
            result = await loop.execute(messages)
            content = result.content
            events.extend(result.events)
            sources.extend(result.sources)
            token_usage = merge_token_usage(token_usage, result.token_usage)
        else:
            # Simple single LLM call
            response = await llm_client.complete(
                messages,
                config.model_tier,
                max_tokens=config.conversation_budget,
                structured_output=config.output_model,
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
                    source_records = build_source_records(normalized_research_output.sources)
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
                        normalized_sources = [item for item in raw_normalized_sources if not _is_semantically_empty(item)]
                if normalized_sources:
                    items = normalized_sources
                    logger.info(
                        "POOL_WRITE_NORMALIZED_SOURCES pool=%s count=%d",
                        pw.pool, len(items),
                    )
                elif sources:
                    items = [_serialize_source_for_pool(s) for s in sources]
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
                try:
                    state.runtime_store.set_coordination(state_output)
                except Exception:
                    pass
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

    sources_pool = pools.get("sources")
    if sources_pool is not None and sources_pool.count() > 0:
        pooled_sources = sources_pool.snapshot()
        substantive = sum(1 for item in pooled_sources if isinstance(item, dict) and item.get("admission_status") == "accepted")
        low_value = sum(1 for item in pooled_sources if isinstance(item, dict) and item.get("admission_status") == "accepted_low_value")
        quality_counts: dict[str, int] = {}
        for item in pooled_sources:
            if isinstance(item, dict):
                quality = str(item.get("evidence_quality", "unknown"))
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
        context.setdefault("source_quality", {
            "substantive_sources": substantive,
            "low_value_sources": low_value,
            "quality_counts": quality_counts,
        })

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

    template_vars = {
        "query": state.query,
        **{k: v for k, v in context_vars.items() if v},
    }

    template_var_preview = {
        k: (str(v)[:60] if v else "<empty>") for k, v in template_vars.items()
    }
    logger.info(
        "AGENT_TEMPLATE_VARS node=%s keys=%s preview=%s",
        _node_id, list(template_vars.keys()), template_var_preview,
    )

    system_prompt = renderer.render(config.system_prompt, template_vars)
    user_prompt = renderer.render(config.user_prompt_template, template_vars)

    logger.info(
        "AGENT_PROMPTS node=%s system_len=%d user_len=%d "
        "system_preview=%s user_preview=%s",
        _node_id, len(system_prompt), len(user_prompt),
        system_prompt[:150], user_prompt[:300],
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
    elif hasattr(output, "sources"):
        if not output.sources:
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
    """Count actual source_refs across output and update citation_stats in-place."""
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
                end = content.index("```", start)
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
                if parsed in ("", None) and content.strip():
                    raise ValueError("json_repair produced empty output")
                if (
                    config.subtype == "researcher"
                    and isinstance(parsed, dict)
                    and not parsed
                    and _has_substantive_text(content, min_length=20)
                ):
                    raise ValueError("json_repair produced empty dict from substantive text")
                logger.info(
                    "AGENT_OUTPUT_PARSE format=json_repaired input_type=str "
                    "output_type=%s output_len=%d",
                    type(parsed).__name__, len(str(parsed)),
                )
                return parsed
            except (ImportError, Exception):
                logger.warning("JSON_PARSE_FAILURE content=%s", content[:200])
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


# ---------------------------------------------------------------------------
# Pool extraction
# ---------------------------------------------------------------------------


def _extract_pool_items(
    output: Any, pw: PoolWriteConfig, output_key: str = ""
) -> list[Any]:
    """Extract items from agent output for pool writes.

    Handles three cases:
    1. Structured output (dict/model): navigate dot-path normally.
    2. Self-referential: ``extract`` matches ``output_key`` and output is a
       non-navigable type (e.g. plain text).  The entire output IS the value.
    3. Mismatch: ``extract`` path doesn't exist on a plain string → empty list.
    """
    # Case 2: self-referential extraction — the output itself is the value.
    # e.g. output_key="findings", extract="findings", output is a markdown string.
    if (
        pw.extract == output_key
        and isinstance(output, str)
        and output.strip()
    ):
        return [output]

    # Case 1: navigate dot-path on structured output
    current = output
    for part in pw.extract.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return []
        if current is None:
            return []

    if isinstance(current, list):
        return [item for item in current if not _is_semantically_empty(item)]
    if _is_semantically_empty(current):
        return []
    return [current]
