"""Designer-side AST auto-repair (Layer 2 of the hardening plan).

The architect (Opus) occasionally emits identifiers the framework does not
recognize — e.g. ``subtype: lane_researcher`` (not a builtin) or
``model_tier: standard`` (no such tier). When the defect is deterministic
and unambiguously fixable, this module rewrites it in-place AND emits a
``NormalizationFix`` record so the structural gate, the SSE stream, and
ultimately the UI can show the user what was repaired. Nothing silent.

See ``.omc/plans/designer-hardening.md`` for the full layered defense
architecture; this module implements Layer 2.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from deep_research.core.app_config import (
    fill_databricks_search_defaults,
    get_app_config,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationFix:
    """One deterministic auto-repair applied to an architect-emitted AST.

    - ``kind`` is a stable machine label for the repair. The frontend renders
      known labels with specific icons and falls back gracefully for new labels.
    - ``path`` uses the dot-notation address scheme established by
      ``mutations._split_path`` (e.g. ``root.children.1.config.subtype``).
    - ``before``/``after`` are the literal values pre- and post-repair.
      Captured as ``Any`` because they may be strings, lists, ints, etc.
    - ``rationale`` is a single user-facing sentence explaining *why*.
    """

    kind: str
    path: str
    before: Any
    after: Any
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "rationale": self.rationale,
        }


@dataclass
class _NormalizerContext:
    """Mutable carrier so each rewriter can append fixes without globals."""

    fixes: list[NormalizationFix] = field(default_factory=list)

    def record(
        self,
        *,
        kind: str,
        path: str,
        before: Any,
        after: Any,
        rationale: str,
    ) -> None:
        self.fixes.append(
            NormalizationFix(
                kind=kind,
                path=path,
                before=before,
                after=after,
                rationale=rationale,
            )
        )


# ---------------------------------------------------------------------------
# Canonical identifier sets (kept narrow; fallback when registry unreachable)
# ---------------------------------------------------------------------------


_KNOWN_SUBTYPES: frozenset[str] = frozenset(
    {
        "coordinator",
        "planner",
        "researcher",
        "reflector",
        "synthesizer",
        "background",
        "custom",
        # Router topology classifier: a builtin subtype whose enrich synthesizes a
        # typed route discriminator. Must be preserved by the normalizer (never
        # remapped to coordinator) so the conditional router resolves at runtime.
        "router_classifier",
    }
)


# Heuristic table — keys are unknown subtypes the architect has emitted in the
# wild; values are the closest framework-recognized subtype. Order does not
# matter (each unknown subtype maps to exactly one canonical target).
_SUBTYPE_ALIASES: dict[str, str] = {
    # Class-A1 examples observed in scaffold-and-run artifacts:
    "lane_researcher": "researcher",
    "research_lane": "researcher",
    "investigator": "researcher",
    "analyst": "researcher",
    "specialist": "researcher",
    "fundamentals_researcher": "researcher",
    "risk_researcher": "researcher",
    "market_researcher": "researcher",
    "sentiment_researcher": "researcher",
    "competitor_researcher": "researcher",
    "lane": "researcher",
    # Synthesizer aliases
    "summarizer": "synthesizer",
    "reporter": "synthesizer",
    "writer": "synthesizer",
    "compositor": "synthesizer",
    # Reflector / evaluator aliases
    "evaluator": "reflector",
    "reviewer": "reflector",
    "critic_agent": "reflector",
    "judge": "reflector",
    "qa": "reflector",
    # Planner aliases
    "decomposer": "planner",
    "task_planner": "planner",
    # Coordinator aliases
    "router": "coordinator",
    "dispatcher": "coordinator",
}


_KNOWN_TIERS: frozenset[str] = frozenset(
    {
        "simple",
        "analytical",
        "complex",
        "bulk_analysis",
        "fast",
    }
)


# Architect-emitted tier aliases observed in artifacts.
_TIER_ALIASES: dict[str, str] = {
    "standard": "analytical",
    "default": "analytical",
    "balanced": "analytical",
    "reasoning": "complex",
    "deep": "complex",
    "advanced": "complex",
    "powerful": "complex",
    "lite": "simple",
    "light": "simple",
    "quick": "simple",
    "bulk": "bulk_analysis",
}


# Per-subtype tier when the alias cannot be resolved by the table.
_TIER_FALLBACK_BY_SUBTYPE: dict[str, str] = {
    "researcher": "analytical",
    "reflector": "analytical",
    "planner": "analytical",
    "synthesizer": "complex",
    "coordinator": "simple",
    "background": "simple",
    "custom": "analytical",
}


_TOOL_KIND_ALIASES: dict[str, str] = {
    "search": "web_search",
    "web": "web_search",
    "websearch": "web_search",
    "browse": "web_search",
    "google": "web_search",
    "crawl": "web_crawl",
    "fetch": "web_crawl",
    "page_fetch": "web_crawl",
    "scrape": "web_crawl",
    "vector": "vector_search",
    "vectordb": "vector_search",
    "rag": "vector_search",
    "kb": "knowledge_assistant",
    "knowledge_base": "knowledge_assistant",
    "files": "file_search",
    "file": "file_search",
    "brave": "web_search",
}


_RETRIEVAL_TOOL_KINDS: frozenset[str] = frozenset(
    {
        "web_search",
        "web_crawl",
        "web_research",
        "vector_search",
        "genie",
        "knowledge_assistant",
        "file_search",
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    }
)
_WEB_TOOL_KINDS: frozenset[str] = frozenset(
    {"web_search", "web_crawl", "web_research"}
)
_CORPUS_TOOL_KINDS: frozenset[str] = frozenset(
    {
        "vector_search",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    }
)

_GROUNDING_MODES: frozenset[str] = frozenset(
    {"none", "classical_lite", "reclaim"}
)
_EVIDENCE_POOL_NAMES: frozenset[str] = frozenset({"observations", "sources"})
_MIN_RESEARCHER_USER_PROMPT_CHARS = 250
_GENERIC_RESEARCHER_USER_PROMPT_MARKERS: tuple[str, ...] = (
    "Execute the following research step:",
    "{step_title}",
)
_STATIC_PARALLEL_FORBIDDEN_TEMPLATE_VARS: frozenset[str] = frozenset(
    {
        "current_step",
        "page_contents",
        "previous_observations",
        "research_plan",
        "search_results",
        "step_description",
        "step_title",
        "step_type",
    }
)
_STATIC_PARALLEL_FORBIDDEN_INPUT_KEYS: frozenset[str] = (
    _STATIC_PARALLEL_FORBIDDEN_TEMPLATE_VARS
)
_STATIC_PARALLEL_ALLOWED_TEMPLATE_VARS: frozenset[str] = frozenset(
    {"query", "coordination", "conversation_history"}
)
_GENERAL_RESEARCHER_ALLOWED_TEMPLATE_VARS: frozenset[str] = (
    _STATIC_PARALLEL_ALLOWED_TEMPLATE_VARS
    | _STATIC_PARALLEL_FORBIDDEN_TEMPLATE_VARS
    | frozenset({"findings", "reflection", "evaluation"})
)
_SUBQUESTION_HEADING_MARKERS: tuple[str, ...] = (
    "sub-questions",
    "subquestions",
    "sub questions",
)
_OUTPUT_SECTION_HEADING_MARKERS: tuple[str, ...] = (
    "required output",
    "output structure",
    "output sections",
)
_SEARCH_STRATEGY_HEADING_MARKERS: tuple[str, ...] = ("search strategy",)
_UNKNOWNS_HANDLING_MARKERS: tuple[str, ...] = (
    "data unavailable",
    "definition of done",
    "do not improvise",
)


# ---------------------------------------------------------------------------
# Walker — yields (node_dict, dot_path) tuples for every node in the tree
# ---------------------------------------------------------------------------


def _walk_nodes(ast: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Pre-order walk of every node in the AST.

    Handles the four composite shapes the framework supports:
    - ``children`` list (sequence, parallel, loop, conditional)
    - ``config.body`` single node (plan_and_execute body)
    - ``config.evaluator`` single node (plan_and_execute evaluator)
    - ``config.children`` list (nested composites that re-use the same shape)
    """
    out: list[tuple[dict[str, Any], str]] = []

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        out.append((node, path))
        # children list
        children = node.get("children")
        if isinstance(children, list):
            for idx, child in enumerate(children):
                visit(child, f"{path}.children.{idx}")
        # config.body — plan_and_execute
        config = node.get("config")
        if isinstance(config, dict):
            body = config.get("body")
            if isinstance(body, dict):
                visit(body, f"{path}.config.body")
            evaluator = config.get("evaluator")
            if isinstance(evaluator, dict):
                visit(evaluator, f"{path}.config.evaluator")
            # Nested children inside config (defensive — some node types nest)
            nested_children = config.get("children")
            if isinstance(nested_children, list):
                for idx, child in enumerate(nested_children):
                    visit(child, f"{path}.config.children.{idx}")

    root = ast.get("root")
    if isinstance(root, dict):
        visit(root, "root")
    return out


def _walk_nodes_with_context(
    ast: dict[str, Any],
) -> list[tuple[dict[str, Any], str, bool, bool]]:
    """Pre-order walk carrying topology context.

    Returns ``(node, path, in_plan_and_execute, in_static_parallel)``.
    ``in_static_parallel`` means a node is beneath a ``parallel`` composite
    that is not itself beneath a ``plan_and_execute`` body. Those researchers
    are one-shot static lanes and must not depend on planner step fields.
    """

    out: list[tuple[dict[str, Any], str, bool, bool]] = []

    def visit(
        node: Any,
        path: str,
        *,
        in_plan_and_execute: bool,
        in_static_parallel: bool,
    ) -> None:
        if not isinstance(node, dict):
            return
        node_type = str(node.get("type") or "")
        current_in_plan = in_plan_and_execute or node_type == "plan_and_execute"
        current_static_parallel = in_static_parallel or (
            node_type == "parallel" and not current_in_plan
        )
        out.append((node, path, current_in_plan, current_static_parallel))

        children = node.get("children")
        if isinstance(children, list):
            for idx, child in enumerate(children):
                visit(
                    child,
                    f"{path}.children.{idx}",
                    in_plan_and_execute=current_in_plan,
                    in_static_parallel=current_static_parallel,
                )

        config = node.get("config")
        if isinstance(config, dict):
            body = config.get("body")
            if isinstance(body, dict):
                visit(
                    body,
                    f"{path}.config.body",
                    in_plan_and_execute=current_in_plan,
                    in_static_parallel=current_static_parallel,
                )
            evaluator = config.get("evaluator")
            if isinstance(evaluator, dict):
                visit(
                    evaluator,
                    f"{path}.config.evaluator",
                    in_plan_and_execute=current_in_plan,
                    in_static_parallel=current_static_parallel,
                )
            nested_children = config.get("children")
            if isinstance(nested_children, list):
                for idx, child in enumerate(nested_children):
                    visit(
                        child,
                        f"{path}.config.children.{idx}",
                        in_plan_and_execute=current_in_plan,
                        in_static_parallel=current_static_parallel,
                    )

    root = ast.get("root")
    if isinstance(root, dict):
        visit(
            root,
            "root",
            in_plan_and_execute=False,
            in_static_parallel=False,
        )
    return out


# ---------------------------------------------------------------------------
# Individual normalizers
# ---------------------------------------------------------------------------


def _infer_subtype_from_id(node_id: str | None) -> str | None:
    """Infer the agent subtype from its node id (e.g. ``synthesizer`` →
    ``synthesizer``). Used as a backstop when an architect's update_block
    accidentally drops the ``subtype`` field via a config replacement."""
    if not isinstance(node_id, str) or not node_id:
        return None
    lid = node_id.lower()
    # Match known subtypes by substring (e.g. lane_1-researcher → researcher).
    for known in _KNOWN_SUBTYPES:
        if known in lid:
            return known
    return None


def _normalize_subtypes(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Rewrite unknown subtypes to the closest known one.

    Also handles the case where ``subtype`` is missing entirely — falls back
    to inferring from the node id (so a node with ``id='synthesizer'`` but
    no subtype field gets ``subtype='synthesizer'``)."""
    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        subtype = config.get("subtype")
        if not isinstance(subtype, str) or not subtype:
            # Missing subtype — try to infer from the node id. Falls through
            # to the standard alias/fallback path if inference fails.
            inferred = _infer_subtype_from_id(node.get("id"))
            if inferred is not None:
                config["subtype"] = inferred
                ctx.record(
                    kind="subtype_rewrite",
                    path=f"{path}.config.subtype",
                    before=subtype,  # None or empty
                    after=inferred,
                    rationale=(
                        "Agent node had no subtype; inferred from id "
                        f"'{node.get('id')!r}'. The architect's update_block "
                        "likely overwrote config with patches that omitted "
                        "subtype — Layer 2 backstop restores it."
                    ),
                )
                continue
            # No inference possible — fall through to alias table with the
            # original (None/empty) value; ends up at the 'researcher' fallback.
            subtype = ""
        if subtype in _KNOWN_SUBTYPES:
            continue
        # Try alias table; fall back to "researcher" (most useful default —
        # gets retrieval auto-bound downstream).
        canonical = _SUBTYPE_ALIASES.get(subtype.lower(), "researcher")
        config["subtype"] = canonical
        ctx.record(
            kind="subtype_rewrite",
            path=f"{path}.config.subtype",
            before=subtype,
            after=canonical,
            rationale=(
                f"The framework recognizes only {sorted(_KNOWN_SUBTYPES)}; "
                f"'{subtype}' was rewritten to '{canonical}' (closest match)."
            ),
        )


def _normalize_model_tiers(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Rewrite unknown model_tiers to a valid one."""
    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        tier = config.get("model_tier")
        if not isinstance(tier, str):
            continue
        if tier in _KNOWN_TIERS:
            continue
        canonical: str | None = _TIER_ALIASES.get(tier.lower())
        if canonical is None:
            raw_subtype = config.get("subtype")
            subtype = raw_subtype if isinstance(raw_subtype, str) else ""
            canonical = _TIER_FALLBACK_BY_SUBTYPE.get(subtype, "analytical")
        config["model_tier"] = canonical
        ctx.record(
            kind="tier_rewrite",
            path=f"{path}.config.model_tier",
            before=tier,
            after=canonical,
            rationale=(
                f"'{tier}' is not a configured model tier; "
                f"'{canonical}' is the default for this agent's subtype."
            ),
        )


def _lift_mcp_servers(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Lift ``kind == 'mcp'`` tool declarations into the workflow ``mcp_servers``.

    An MCP server is authored as an ``mcp`` card in the tool picker, but the
    framework keeps remote MCP servers in the top-level ``mcp_servers`` list (not
    the per-tool ``tools`` section): each is built into an ``MCPToolset``
    per-request and its discovered tools are injected via the resolver override
    (B2). This step moves each ``mcp`` tool decl's ``config`` into ``mcp_servers``
    (dedup by name) and removes it from ``tools``, so the persisted definition
    round-trips through the framework loader's ``mcp_servers`` parsing (P0a).
    """
    tools = ast.get("tools")
    if not isinstance(tools, list):
        return
    existing = ast.get("mcp_servers")
    if not isinstance(existing, list):
        existing = []
    seen = {s.get("name") for s in existing if isinstance(s, dict)}

    lifted: list[dict[str, Any]] = []
    remaining: list[Any] = []
    for idx, tool in enumerate(tools):
        if not (isinstance(tool, dict) and tool.get("kind") == "mcp"):
            remaining.append(tool)
            continue
        config = tool.get("config")
        server = dict(config) if isinstance(config, dict) else {}
        name = server.get("name") or tool.get("name")
        if not name:
            # A nameless MCP card can't bind; drop it with a recorded fix rather
            # than persisting an unusable server.
            ctx.record(
                kind="mcp_server_dropped",
                path=f"tools.{idx}",
                before="mcp tool (no name)",
                after="(removed)",
                rationale="an MCP server declaration requires a 'name'.",
            )
            continue
        server["name"] = name
        if name in seen:
            ctx.record(
                kind="mcp_server_dedup",
                path=f"tools.{idx}",
                before=f"duplicate mcp server '{name}'",
                after="(removed)",
                rationale=f"an MCP server named '{name}' is already declared.",
            )
            continue
        seen.add(name)
        lifted.append(server)
        ctx.record(
            kind="mcp_server_lift",
            path=f"tools.{idx}",
            before=f"tool(kind=mcp, name={name})",
            after=f"mcp_servers[{name}]",
            rationale=(
                "MCP servers live in the workflow 'mcp_servers' list, not 'tools'; "
                "lifted so the server is built per-request under OBO."
            ),
        )

    if lifted or len(remaining) != len(tools):
        ast["tools"] = remaining
        ast["mcp_servers"] = list(existing) + lifted


def _normalize_tool_kinds(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Rewrite tool kind aliases in the top-level tools declarations."""
    tools = ast.get("tools")
    if not isinstance(tools, list):
        return
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        kind = tool.get("kind")
        if not isinstance(kind, str):
            continue
        if kind in _RETRIEVAL_TOOL_KINDS or kind in {"genie"}:
            continue
        canonical = _TOOL_KIND_ALIASES.get(kind.lower())
        if canonical is None:
            continue
        tool["kind"] = canonical
        ctx.record(
            kind="tool_kind_rewrite",
            path=f"tools.{idx}.kind",
            before=kind,
            after=canonical,
            rationale=(
                f"'{kind}' is an alias; rewriting to canonical "
                f"framework kind '{canonical}'."
            ),
        )


# Web-search tool kinds whose backend is provider-selectable. ``web_crawl`` is
# excluded — it fetches a given URL and has no search provider.
_WEB_PROVIDER_TOOL_KINDS: frozenset[str] = frozenset({"web_search", "web_research"})


def apply_web_search_provider_defaults(
    definition: dict[str, Any],
    *,
    search_cfg: Any | None = None,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Make web tools that EXPLICITLY pin ``provider: databricks`` self-describing,
    in place, on a raw definition dict.

    Persistence stores only EXPLICIT per-tool intent. A tool that leaves
    ``config.provider`` blank is left **byte-for-byte untouched** so it keeps
    INHERITING the workspace ``search.provider`` (and the ``search.databricks``
    endpoint) at runtime — i.e. ``app.yaml`` stays a live global lever and the
    designer never silently bakes the current default into a saved definition.
    For a tool that explicitly selects ``databricks`` but omits the serving
    endpoint / tuning, those keys are filled from the ``search.databricks`` block
    when absent — so the explicit choice is self-describing and the adapter's
    ``max_results`` does not silently truncate a ``web_research`` lane's
    ``total_results``. Explicit ``brave``/``jina`` tools have nothing to fill and
    are left untouched.

    Shared by the designer normalizer (records the diffs as fixes) and the
    agent-save path (discards them) so a ``provider: databricks`` web tool
    persisted via the UI — which bypasses the full normalizer — is still
    self-describing. Returns ``(tool_index, before_config, after_config)`` for each
    tool actually mutated. ``search_cfg`` defaults to the app config's ``search``
    block; a config-read failure yields no changes.
    """
    tools = definition.get("tools")
    if not isinstance(tools, list):
        return []
    if search_cfg is None:
        try:
            search_cfg = get_app_config().search
        except Exception:  # pragma: no cover - defensive: config unreadable
            return []

    changes: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("kind") not in _WEB_PROVIDER_TOOL_KINDS:
            continue
        config = tool.get("config")
        # Only an EXPLICIT per-tool databricks selection is filled. Blank/absent
        # provider (inherit) and explicit brave/jina are left untouched so the
        # global default + endpoint stay a live runtime lever.
        if not isinstance(config, dict) or config.get("provider") != "databricks":
            continue
        before = copy.deepcopy(config)
        # web_research passes total_results as the search count; floor the
        # adapter's max_results so it is not capped below the requested count.
        min_results = 0
        for count_key in ("total_results", "max_results"):
            raw = config.get(count_key)
            if isinstance(raw, int) and raw > min_results:
                min_results = raw
        fill_databricks_search_defaults(
            config, search_cfg.databricks, min_results=min_results
        )
        if config != before:
            changes.append((idx, before, copy.deepcopy(config)))
    return changes


def _normalize_web_search_provider(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Fill the Databricks endpoint/tuning onto web tools that explicitly pin
    ``provider: databricks`` and record each as a normalization fix.

    Thin recording wrapper over :func:`apply_web_search_provider_defaults` (see it
    for semantics): inherited (blank-provider) tools are intentionally left
    untouched so they keep following the workspace default + endpoint at runtime.
    """
    for idx, before, after in apply_web_search_provider_defaults(ast):
        tool = ast["tools"][idx]
        ctx.record(
            kind="web_search_provider",
            path=f"tools.{idx}.config",
            before=before,
            after=after,
            rationale=(
                f"Filled Databricks search endpoint/tuning for explicit provider "
                f"'databricks' on '{tool.get('name') or tool.get('kind')}'."
            ),
        )


def _collect_pool_references(ast: dict[str, Any]) -> set[str]:
    """Find every pool name referenced by ``pool_inject`` or ``pool_writes``."""
    refs: set[str] = set()
    for node, _ in _walk_nodes(ast):
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        for key in ("pool_inject", "pool_writes"):
            value = config.get(key)
            if isinstance(value, str) and value:
                refs.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item:
                        refs.add(item)
                    elif isinstance(item, dict):
                        name = item.get("pool") or item.get("name")
                        if isinstance(name, str) and name:
                            refs.add(name)
    return refs


def _declared_pool_names(ast: dict[str, Any]) -> set[str]:
    pools = ast.get("pools")
    if not isinstance(pools, list):
        return set()
    return {
        str(p.get("name"))
        for p in pools
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    }


def _config_pool_names(config: dict[str, Any], field_name: str) -> set[str]:
    value = config.get(field_name)
    names: set[str] = set()
    if isinstance(value, str) and value.strip():
        names.add(value.strip())
        return names
    if not isinstance(value, list):
        return names
    for item in value:
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
        elif isinstance(item, dict):
            name = item.get("pool") or item.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _workflow_has_evidence_pool_contract(ast: dict[str, Any]) -> bool:
    """Whether the AST uses the generic observations/sources evidence pools."""
    declared_or_referenced = _declared_pool_names(ast) | _collect_pool_references(ast)
    if not declared_or_referenced >= _EVIDENCE_POOL_NAMES:
        return False

    observed_writers: set[str] = set()
    for node, _ in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        if config.get("subtype") == "synthesizer":
            continue
        observed_writers.update(_config_pool_names(config, "pool_writes"))
    return observed_writers >= _EVIDENCE_POOL_NAMES


_POOL_WRITE_ALLOWED_KEYS: frozenset[str] = frozenset(
    {"pool", "extract", "transform"}
)
_POOL_INJECT_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "pool",
        "threshold",
        "format",
        "max_items",
        "max_item_chars",
        "compaction",
    }
)


def _default_pool_write_extract(pool_name: str, output_key: Any) -> str:
    """Infer the canonical extract path for an underspecified pool write."""
    if pool_name == "sources":
        return "sources"
    if isinstance(output_key, str) and output_key.strip():
        return output_key
    return "output"


def _normalize_pool_write_item(
    item: Any,
    *,
    output_key: Any,
) -> dict[str, Any] | Any:
    if isinstance(item, str):
        pool_name = item.strip()
        if not pool_name:
            return item
        return {
            "pool": pool_name,
            "extract": _default_pool_write_extract(pool_name, output_key),
        }
    if not isinstance(item, dict):
        return item

    pool_value = item.get("pool") or item.get("name")
    if not isinstance(pool_value, str) or not pool_value.strip():
        return item
    pool_name = pool_value.strip()

    extract = item.get("extract") or item.get("path") or item.get("field")
    if not isinstance(extract, str) or not extract.strip():
        extract = _default_pool_write_extract(pool_name, output_key)

    normalized: dict[str, Any] = {
        "pool": pool_name,
        "extract": extract.strip(),
    }
    transform = item.get("transform")
    if isinstance(transform, str) and transform.strip():
        normalized["transform"] = transform
    return normalized


def _normalize_pool_inject_item(item: Any) -> dict[str, Any] | Any:
    if isinstance(item, str):
        pool_name = item.strip()
        return {"pool": pool_name} if pool_name else item
    if not isinstance(item, dict):
        return item

    pool_value = item.get("pool") or item.get("name")
    if not isinstance(pool_value, str) or not pool_value.strip():
        return item

    normalized = {
        key: value
        for key, value in item.items()
        if key in _POOL_INJECT_ALLOWED_KEYS
    }
    normalized["pool"] = pool_value.strip()
    return normalized


def _normalize_pool_specs(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Rewrite common pool spec drift to the AgentNodeConfig schema.

    The designer LLM sometimes emits legacy/user-facing shapes such as
    ``{"pool": "sources", "fields": ["url", "title"]}``. The framework
    runtime needs explicit extract paths, so this repair keeps the intended
    pool target and chooses the only deterministic extract for that pool.
    """
    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue

        pool_writes = config.get("pool_writes")
        if isinstance(pool_writes, (str, list)):
            raw_items = [pool_writes] if isinstance(pool_writes, str) else pool_writes
            normalized_writes = [
                _normalize_pool_write_item(item, output_key=config.get("output_key"))
                for item in raw_items
            ]
            if normalized_writes != pool_writes:
                config["pool_writes"] = normalized_writes
                ctx.record(
                    kind="pool_spec_rewrite",
                    path=f"{path}.config.pool_writes",
                    before=pool_writes,
                    after=normalized_writes,
                    rationale=(
                        "Rewrote pool_writes to the framework schema. "
                        "Pool writes must declare both 'pool' and 'extract'; "
                        "metadata-only fields are not executable extract paths."
                    ),
                )

        pool_inject = config.get("pool_inject")
        if isinstance(pool_inject, (str, list)):
            raw_items = [pool_inject] if isinstance(pool_inject, str) else pool_inject
            normalized_inject = [
                _normalize_pool_inject_item(item)
                for item in raw_items
            ]
            if normalized_inject != pool_inject:
                config["pool_inject"] = normalized_inject
                ctx.record(
                    kind="pool_spec_rewrite",
                    path=f"{path}.config.pool_inject",
                    before=pool_inject,
                    after=normalized_inject,
                    rationale=(
                        "Rewrote pool_inject to the framework schema. "
                        "Pool injections must be objects keyed by 'pool'."
                    ),
                )


def _auto_declare_pools(ast: dict[str, Any], ctx: _NormalizerContext) -> None:
    """Every referenced pool gets a default declaration if missing."""
    referenced = _collect_pool_references(ast)
    if not referenced:
        return
    pools = ast.setdefault("pools", [])
    if not isinstance(pools, list):
        return
    declared = {
        p.get("name")
        for p in pools
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    }
    for name in sorted(referenced - declared):
        default_pool = {
            "name": name,
            "dedup_key": "content_hash",
            "max_items": 100,
        }
        pools.append(default_pool)
        ctx.record(
            kind="auto_declare_pool",
            path=f"pools.{len(pools) - 1}",
            before=None,
            after=default_pool,
            rationale=(
                f"Pool '{name}' was referenced by an agent but not declared; "
                "auto-declared with default dedup_key + max_items=100."
            ),
        )


# Match (in priority order) the patterns that should NOT be touched, vs
# the lone literal braces that need escaping. Order matters — patterns are
# alternated greedily, so an already-escaped {{ wins over a lone {.
_BRACE_TOKEN_RE = re.compile(
    r"""
    \{\{               # already-escaped open brace
    | \}\}             # already-escaped close brace
    | \{[A-Za-z_][A-Za-z0-9_]*\}   # legit template var: {ident}
    | \{               # lone literal open brace
    | \}               # lone literal close brace
    """,
    re.VERBOSE,
)
_SIMPLE_TEMPLATE_VAR_RE = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})"
)


def _escape_literal_braces(template: str) -> tuple[str, int]:
    """Escape literal ``{`` / ``}`` in *template* that aren't part of a valid
    Jinja-style ``{var}`` or already-escaped ``{{`` / ``}}``.

    Used to fix prompts that include literal JSON examples (output schemas
    of the form ``{"complexity": "simple"}``) — those braces trip the
    framework's ``SafeTemplateRenderer`` because they look like template
    variables with non-identifier contents.

    Returns ``(escaped_template, count_of_braces_doubled)``. When *count*
    is zero, the template is unchanged.
    """
    count = 0

    def _replace(m: re.Match[str]) -> str:
        nonlocal count
        match = m.group(0)
        if match == "{":
            count += 1
            return "{{"
        if match == "}":
            count += 1
            return "}}"
        # {{, }}, or {var} — leave alone.
        return match

    new = _BRACE_TOKEN_RE.sub(_replace, template)
    return new, count


def _coerce_unknown_template_variables_to_query(
    template: str,
    *,
    allowed_vars: frozenset[str],
) -> tuple[str, list[str]]:
    """Bind unknown Designer placeholder aliases to the user query.

    Unknown single-brace identifiers are not executable state keys. In
    Designer-authored lane prompts they normally mean "the target thing the
    user supplied" (company, city, case id, product, etc.), so the neutral
    runtime binding is ``{query}``.
    """

    replaced: list[str] = []

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable in allowed_vars:
            return match.group(0)
        replaced.append(variable)
        return "{query}"

    new_template = _SIMPLE_TEMPLATE_VAR_RE.sub(replace, template)
    return new_template, sorted(set(replaced))


def _escape_literal_braces_in_prompts(
    ast: dict[str, Any], ctx: _NormalizerContext
) -> None:
    """Walk every agent node's ``system_prompt`` and ``user_prompt_template``
    and escape literal ``{`` / ``}`` that the architect emitted as part of
    a JSON example or output-schema directive.

    The framework's ``SafeTemplateRenderer`` rejects any ``{`` not followed
    by a valid template identifier — so when Opus writes a prompt like
    ``"Return JSON: {complexity: 'simple'}"``, the runner crashes with
    ``TemplateSecurityError: Forbidden pattern in template``. This
    normalizer rewrites those braces to ``{{`` / ``}}`` so the renderer
    accepts the prompt verbatim.
    """
    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        for field_name in ("system_prompt", "user_prompt_template"):
            value = config.get(field_name)
            if not isinstance(value, str) or not value:
                continue
            new_value, count = _escape_literal_braces(value)
            if count == 0:
                continue
            config[field_name] = new_value
            ctx.record(
                kind="brace_escape",
                path=f"{path}.config.{field_name}",
                before=f"<{count} literal braces, e.g. {value[:60]!r}>",
                after=f"<escaped to {{{{ }}}}, length {len(new_value)}>",
                rationale=(
                    f"Escaped {count} literal '{{' / '}}' in "
                    f"{field_name} so SafeTemplateRenderer accepts it. "
                    "The architect included a JSON-shaped example whose "
                    "braces would otherwise be parsed as malformed "
                    "template variables."
                ),
            )


def _has_marker(template: str, markers: tuple[str, ...]) -> bool:
    lower = template.lower()
    return any(marker in lower for marker in markers)


def _has_query_binding(template: str) -> bool:
    return "{query}" in template or "query" in template.lower()[:300]


def _bounded_inline(value: str, *, max_length: int) -> str:
    cleaned = " ".join(str(value).strip().split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 15].rstrip() + " ...(truncated)"


def _bounded_multiline(value: str, *, max_length: int) -> str:
    cleaned = str(value).strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 15].rstrip() + " ...(truncated)"


def _wrap_researcher_user_prompt_contract(
    *,
    lane_focus: str,
    designer_template: str,
    search_strategy_block: str | None = None,
) -> str:
    """Wrap substantive Designer text with the runtime lane prompt contract.

    This is domain-neutral repair for AST mutations. The Designer still owns
    the use-case-specific brief in ``designer_template``; the wrapper only
    restores the execution scaffolding required by the lane runner.
    """
    focus = _bounded_inline(lane_focus or "the assigned lane focus", max_length=420)
    normalized_template, _ = _coerce_unknown_template_variables_to_query(
        designer_template,
        allowed_vars=_GENERAL_RESEARCHER_ALLOWED_TEMPLATE_VARS,
    )
    template = _bounded_multiline(normalized_template, max_length=2600)
    strategy_block = search_strategy_block or (
        "### Search strategy\n"
        "- Start with queries that combine {query} with the lane focus terms above.\n"
        "- Prefer primary or high-authority sources, then refine searches around gaps."
    )
    wrapped = (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        f"Lane focus: {focus}\n\n"
        "### Sub-questions you MUST address (in this order)\n"
        f"1. What are the most decision-relevant facts for this lane focus: {focus}?\n"
        "2. Which current evidence supports or contradicts those facts?\n"
        "3. What metrics, events, entities, or comparisons materially change the interpretation?\n"
        "4. What uncertainties, data gaps, or conflicting signals remain?\n"
        "5. What bottom-line implications should the final report carry forward?\n\n"
        "### Required output structure\n"
        "- **Evidence summary**: cite the strongest findings and source context.\n"
        "- **Analysis and implications**: explain why the evidence matters for the user goal.\n"
        "- **Unknowns and caveats**: mark missing, stale, or conflicting evidence explicitly.\n\n"
        f"{strategy_block}\n\n"
        "### Definition of done\n"
        "Each sub-question has a concise answer with citeable source text, OR "
        "is marked \"Data unavailable\" -- DO NOT improvise.\n\n"
        "### Designer-authored lane brief\n"
        f"{template}"
    )
    return _bounded_multiline(wrapped, max_length=4000)


def _count_numbered_items_under_heading(
    template: str,
    heading_markers: tuple[str, ...],
) -> int:
    if not template:
        return 0
    lines = template.splitlines()
    heading_idx: int | None = None
    for idx, raw_line in enumerate(lines):
        if _has_marker(raw_line, heading_markers):
            heading_idx = idx
            break
    if heading_idx is None:
        return 0

    count = 0
    blank_streak = 0
    for raw_line in lines[heading_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            blank_streak += 1
            if blank_streak >= 2 and count > 0:
                break
            continue
        blank_streak = 0
        if stripped.startswith("#"):
            break
        if (
            len(stripped) > 2
            and stripped[0].isdigit()
            and (
                stripped[1] == "."
                or (len(stripped) > 2 and stripped[1].isdigit() and stripped[2] == ".")
            )
        ):
            count += 1
    return count


def _count_output_section_bullets(template: str) -> int:
    if not template:
        return 0
    lines = template.splitlines()
    heading_idx: int | None = None
    for idx, raw_line in enumerate(lines):
        if _has_marker(raw_line, _OUTPUT_SECTION_HEADING_MARKERS):
            heading_idx = idx
            break
    if heading_idx is None:
        return 0

    count = 0
    blank_streak = 0
    for raw_line in lines[heading_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            blank_streak += 1
            if blank_streak >= 2 and count > 0:
                break
            continue
        blank_streak = 0
        if stripped.startswith("#"):
            break
        if stripped.startswith(("-", "*", "+")):
            count += 1
    return count


def _declared_tool_kinds(ast: dict[str, Any]) -> set[str]:
    kinds: set[str] = set()
    for tool in ast.get("tools", []) or []:
        if isinstance(tool, dict) and isinstance(tool.get("kind"), str):
            kinds.add(tool["kind"])
    return kinds


def _researcher_search_strategy_block(tool_kinds: set[str]) -> str:
    if tool_kinds & _CORPUS_TOOL_KINDS and not tool_kinds & _WEB_TOOL_KINDS:
        corpus_tools = ", ".join(sorted(tool_kinds & _CORPUS_TOOL_KINDS))
        return (
            "### Search strategy\n"
            f"- Start with the available corpus retrieval tools ({corpus_tools}) "
            "against the named resources, combining {query} with the lane focus terms.\n"
            "- Use exact read/load/neighbor/aggregate tools for supporting records; "
            "retrieval metadata alone is not sufficient evidence."
        )
    return (
        "### Search strategy\n"
        "- Run focused searches for each sub-question; refine with source names, "
        "official documents, or exact phrases found in promising results.\n"
        "- Crawl or retrieve source text before relying on a result; titles and "
        "metadata alone are not citeable evidence."
    )


def _normalize_researcher_user_prompts(
    ast: dict[str, Any], ctx: _NormalizerContext
) -> None:
    """Ensure researcher user prompts carry the runtime investigation contract.

    The designer LLM owns use-case-specific lane semantics. This normalizer
    only restores generic execution structure around substantive Designer
    text. Empty, generic, or too-short prompts stay visible to the Designer
    validation loop instead of being replaced with canned domain content.
    """
    search_strategy_block = _researcher_search_strategy_block(_declared_tool_kinds(ast))
    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict) or config.get("subtype") != "researcher":
            continue

        raw_template = config.get("user_prompt_template")
        template = raw_template if isinstance(raw_template, str) else str(raw_template or "")
        stripped = template.strip()

        generic = _has_marker(stripped, _GENERIC_RESEARCHER_USER_PROMPT_MARKERS)
        subquestions = _count_numbered_items_under_heading(
            stripped,
            _SUBQUESTION_HEADING_MARKERS,
        )
        output_bullets = _count_output_section_bullets(stripped)
        has_output_heading = _has_marker(stripped, _OUTPUT_SECTION_HEADING_MARKERS)

        if (
            not stripped
            or generic
            or len(stripped) < _MIN_RESEARCHER_USER_PROMPT_CHARS
        ):
            continue

        structurally_incomplete = (
            subquestions < 5
            or not has_output_heading
            or output_bullets < 3
        )
        if structurally_incomplete:
            lane_focus = str(node.get("label") or node.get("id") or "").strip()
            new_template = _wrap_researcher_user_prompt_contract(
                lane_focus=lane_focus,
                designer_template=stripped,
                search_strategy_block=search_strategy_block,
            )
            config["user_prompt_template"] = new_template
            missing: list[str] = []
            if subquestions < 5:
                missing.append("subquestions")
            if not has_output_heading:
                missing.append("output_heading")
            if has_output_heading and output_bullets < 3:
                missing.append("output_bullets")
            ctx.record(
                kind="researcher_prompt_contract",
                path=f"{path}.config.user_prompt_template",
                before=f"<{len(template)} chars; missing={','.join(missing)}>",
                after=f"<{len(new_template)} chars; wrapped designer brief>",
                rationale=(
                    "Substantive Designer-authored researcher prompt was "
                    "missing required runtime structure; wrapped it with a "
                    "domain-neutral investigation contract while preserving "
                    "the original lane brief."
                ),
            )
            continue

        additions: list[str] = []
        new_template = stripped
        if not _has_query_binding(new_template):
            new_template = (
                "## Investigation Brief\n\n"
                "You are investigating: **{query}**\n\n"
                f"{new_template}"
            )
            additions.append("query_binding")
        if not _has_marker(new_template, _SEARCH_STRATEGY_HEADING_MARKERS):
            new_template += f"\n\n{search_strategy_block}"
            additions.append("search_strategy")
        if not _has_marker(new_template, _UNKNOWNS_HANDLING_MARKERS):
            new_template += (
                "\n\n### Definition of done\n"
                "Each sub-question has a concise answer with citeable source "
                "text, OR is marked \"Data unavailable\" -- DO NOT improvise."
            )
            additions.append("unknowns_handling")
        reasons = additions

        if new_template == template:
            continue
        config["user_prompt_template"] = new_template
        ctx.record(
            kind="researcher_prompt_contract",
            path=f"{path}.config.user_prompt_template",
            before=f"<{len(template)} chars>",
            after=f"<{len(new_template)} chars; repairs={','.join(reasons)}>",
            rationale=(
                "Researcher prompts must bind the runtime query, ask concrete "
                "sub-questions, define output structure, specify search "
                "strategy, and mark unavailable data instead of improvising."
            ),
        )


def _strip_static_parallel_forbidden_lines(template: str) -> tuple[str, list[str]]:
    """Remove plan/execute scratchpad lines from a static lane prompt."""
    kept: list[str] = []
    removed_vars: list[str] = []
    for raw_line in template.splitlines():
        variables = set(_SIMPLE_TEMPLATE_VAR_RE.findall(raw_line))
        forbidden = variables & _STATIC_PARALLEL_FORBIDDEN_TEMPLATE_VARS
        if forbidden:
            removed_vars.extend(sorted(forbidden))
            continue
        kept.append(raw_line)
    return "\n".join(kept).strip(), sorted(set(removed_vars))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_static_parallel_researchers(
    ast: dict[str, Any], ctx: _NormalizerContext
) -> None:
    """Repair mutation-authored static parallel lanes.

    Static lanes run once under a ``parallel`` node and do not receive planner
    step state. When an LLM mutates those lanes with plan_and_execute prompt
    fragments (``{step_title}``, ``{search_results}``, etc.), the workflow may
    pass schema validation but the lane starts from empty planning/scratchpad
    text instead of the real static lane brief. This pass strips those fields
    while preserving Designer-authored lane substance.
    """
    for node, path, in_plan, in_static_parallel in _walk_nodes_with_context(ast):
        if in_plan or not in_static_parallel or node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict) or config.get("subtype") != "researcher":
            continue

        raw_template = config.get("user_prompt_template")
        template = raw_template if isinstance(raw_template, str) else ""
        stripped_template, removed_vars = _strip_static_parallel_forbidden_lines(
            template
        )
        sanitized_template, replaced_vars = _coerce_unknown_template_variables_to_query(
            stripped_template,
            allowed_vars=_STATIC_PARALLEL_ALLOWED_TEMPLATE_VARS,
        )
        if sanitized_template != template:
            config["user_prompt_template"] = sanitized_template
            ctx.record(
                kind="static_parallel_lane_prompt",
                path=f"{path}.config.user_prompt_template",
                before=(
                    f"<{len(template)} chars; removed_vars={removed_vars}; "
                    f"query_aliases={replaced_vars}>"
                ),
                after=f"<{len(sanitized_template)} chars; static lane prompt>",
                rationale=(
                    "Static parallel lane prompts cannot depend on planner "
                    "step or tool-scratchpad variables. Removed those lines "
                    "and mapped unknown target placeholders to {query}."
                ),
            )

        raw_input_keys = config.get("input_keys")
        current_input_keys = (
            list(raw_input_keys) if isinstance(raw_input_keys, list) else []
        )
        cleaned_input_keys = [
            key
            for key in current_input_keys
            if isinstance(key, str)
            and key not in _STATIC_PARALLEL_FORBIDDEN_INPUT_KEYS
        ]
        if "query" not in cleaned_input_keys:
            cleaned_input_keys.insert(0, "query")
        if (
            "{coordination}" in sanitized_template
            and "coordination" not in cleaned_input_keys
        ):
            cleaned_input_keys.append("coordination")
        cleaned_input_keys = _dedupe_preserve_order(cleaned_input_keys)
        if cleaned_input_keys != raw_input_keys:
            config["input_keys"] = cleaned_input_keys
            ctx.record(
                kind="static_parallel_lane_inputs",
                path=f"{path}.config.input_keys",
                before=raw_input_keys,
                after=cleaned_input_keys,
                rationale=(
                    "Static parallel lane researchers consume the user query "
                    "and optional coordinator scope, not plan_and_execute "
                    "step state or tool scratchpad fields."
                ),
            )


def _normalize_synthesizer_grounding(
    ast: dict[str, Any], ctx: _NormalizerContext
) -> None:
    """Default evidence-pool synthesizers to grounded synthesis.

    This is intentionally domain-agnostic: if a workflow writes citeable
    ``sources`` and factual ``observations`` pools, a downstream synthesizer
    should consume them through the grounded pipeline unless the workflow
    explicitly opts out with ``grounding_mode="none"``.
    """
    if not _workflow_has_evidence_pool_contract(ast):
        return

    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        if config.get("subtype") != "synthesizer":
            continue

        grounding_mode = config.get("grounding_mode")
        if grounding_mode == "none":
            continue

        if grounding_mode not in {"classical_lite", "reclaim"}:
            config["grounding_mode"] = "reclaim"
            ctx.record(
                kind="synthesizer_grounding_default",
                path=f"{path}.config.grounding_mode",
                before=grounding_mode,
                after="reclaim",
                rationale=(
                    "Synthesizer consumes workflow evidence pools but had no "
                    "grounding mode; defaulted to reclaim so final answers are "
                    "generated from collected sources instead of plain LLM text."
                ),
            )

        pool_inject = config.get("pool_inject")
        before_inject = copy.deepcopy(pool_inject)
        normalized_inject = list(pool_inject) if isinstance(pool_inject, list) else []
        injected_names = _config_pool_names({"pool_inject": normalized_inject}, "pool_inject")
        for pool_name in ("observations", "sources"):
            if pool_name not in injected_names:
                normalized_inject.append({"pool": pool_name, "threshold": 0})
        if normalized_inject != before_inject:
            config["pool_inject"] = normalized_inject
            ctx.record(
                kind="synthesizer_pool_inject_default",
                path=f"{path}.config.pool_inject",
                before=before_inject,
                after=normalized_inject,
                rationale=(
                    "Grounded synthesizer must read the generic observations "
                    "and sources pools collected by upstream research agents."
                ),
            )

        output_schema = config.get("output_schema")
        before_schema = copy.deepcopy(output_schema)
        schema = copy.deepcopy(output_schema) if isinstance(output_schema, dict) else {}
        claim_disposition = schema.get("claim_disposition")
        if not isinstance(claim_disposition, dict):
            claim_disposition = {}
        if claim_disposition.get("abstained") != "remove":
            claim_disposition["abstained"] = "remove"
            schema["claim_disposition"] = claim_disposition
        if schema != before_schema:
            config["output_schema"] = schema
            ctx.record(
                kind="synthesizer_output_schema_default",
                path=f"{path}.config.output_schema",
                before=before_schema,
                after=schema,
                rationale=(
                    "Grounded designer synthesizers remove abstained claims by "
                    "default so unverified or unsupported statements do not "
                    "leak into final reports."
                ),
            )


def _set_minimum_max_tool_calls(
    ast: dict[str, Any], ctx: _NormalizerContext
) -> None:
    """Agents with tools but missing/zero max_tool_calls get a sensible floor."""
    for node, path in _walk_nodes(ast):
        if node.get("type") != "agent":
            continue
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        tools = config.get("tools")
        if not isinstance(tools, list) or not tools:
            continue
        max_calls = config.get("max_tool_calls")
        if isinstance(max_calls, int) and max_calls > 0:
            continue
        subtype = config.get("subtype") if isinstance(config.get("subtype"), str) else ""
        floor = 6 if subtype == "researcher" else 3
        config["max_tool_calls"] = floor
        ctx.record(
            kind="set_minimum_max_tool_calls",
            path=f"{path}.config.max_tool_calls",
            before=max_calls,
            after=floor,
            rationale=(
                f"Agent has {len(tools)} tool(s) bound but no max_tool_calls "
                f"budget; set to {floor} so the ReAct loop can actually run."
            ),
        )


def _lift_config_error_handling(
    ast: dict[str, Any], ctx: _NormalizerContext
) -> None:
    """Move accidentally nested ``config.error_handling`` to node level."""
    for node, path in _walk_nodes(ast):
        config = node.get("config")
        if not isinstance(config, dict):
            continue
        nested = config.get("error_handling")
        if not isinstance(nested, dict):
            continue
        before_node_value = copy.deepcopy(node.get("error_handling"))
        config.pop("error_handling", None)
        if "error_handling" not in node:
            node["error_handling"] = nested
        ctx.record(
            kind="error_handling_lift",
            path=f"{path}.error_handling",
            before={
                "config.error_handling": nested,
                "node.error_handling": before_node_value,
            },
            after=node.get("error_handling"),
            rationale=(
                "WorkflowNode.error_handling is a node-level field. Removed "
                "the invalid config.error_handling nesting emitted by the "
                "mutation tool call."
            ),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def normalize_ast(
    ast: dict[str, Any],
) -> tuple[dict[str, Any], list[NormalizationFix]]:
    """Apply deterministic auto-repairs to ``ast``.

    Returns ``(new_ast, fixes)``. The input is never mutated — a deep copy is
    rewritten in place. The fix order in the returned list matches the order
    in which repairs were applied (subtypes → tiers → tool-kind aliases →
    pool declarations → grounding defaults → max_tool_calls).

    Pure function. Zero LLM calls. Safe to invoke from a synchronous code
    path (gate, validator, test).
    """
    if not isinstance(ast, dict):
        return ast, []
    new_ast = copy.deepcopy(ast)
    ctx = _NormalizerContext()
    _normalize_subtypes(new_ast, ctx)
    _normalize_model_tiers(new_ast, ctx)
    _lift_mcp_servers(new_ast, ctx)
    _normalize_tool_kinds(new_ast, ctx)
    _normalize_web_search_provider(new_ast, ctx)
    _normalize_pool_specs(new_ast, ctx)
    _auto_declare_pools(new_ast, ctx)
    _lift_config_error_handling(new_ast, ctx)
    _normalize_synthesizer_grounding(new_ast, ctx)
    _set_minimum_max_tool_calls(new_ast, ctx)
    _normalize_static_parallel_researchers(new_ast, ctx)
    _normalize_researcher_user_prompts(new_ast, ctx)
    _escape_literal_braces_in_prompts(new_ast, ctx)
    return new_ast, list(ctx.fixes)


__all__ = [
    "NormalizationFix",
    "normalize_ast",
]
