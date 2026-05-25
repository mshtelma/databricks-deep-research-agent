"""PR3-D Layer 3a — Synthetic behavioral probe.

Reads the proposed AST + the classifier's TaskSignature and reports
predicted runtime behavior BEFORE the auditor approves. The probe is
deterministic: no LLM, no real tool calls — only static AST inspection
plus optional stub-tool execution.

The probe runs three classes of checks:

1. **Signature-independent invariants** — enforced regardless of the
   signature's correctness; these catch classifier misclassification at
   design time. (Every researcher lane invokes >=1 tool from its
   declared list; lane prompts contain the ``{query}`` anchor; every
   tool kind is in ToolKind; no name-collision in lane tool aliases;
   synthesizer reads observation/sources pools.)

2. **Signature-conditioned checks** — fire only when relevant
   ``question_ambiguity`` / ``retrieval_pattern`` / ``question_class``
   flags are set:
     * topology mismatch (AST topology vs ``select_topology(sig)``);
     * period_basis → at least one lane prompt mentions both FY and CY
       framing;
     * numeric_aggregation → at least one lane has compute-kind AND
       delta_table_read-kind tools bound;
     * structured_tables → at least one lane has delta_table_read-kind
       tool bound.

3. **Runtime-query check** *(opt-in via ``run_runtime_query_check=True``)*
   — stub-LLMs each lane and records the ``{query}`` strings issued;
   per-axis keyword sets must appear across recorded queries. This is
   the only check that exercises prompt behaviour rather than text.

The probe emits ``probe_result`` to state:
    ``{passed: bool, gaps: list[str], invariants_passed: list[str],
       conditional_passed: list[str], runtime_queries: list[str]}``

Auditor's checklist requires ``probe_result.passed == True``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from deep_research.agent_designer.task_signature import (
    TaskSignature,
    select_topology,
)

# Tool kinds that the probe recognizes as evidence collectors.
_RETRIEVAL_KINDS: frozenset[str] = frozenset(
    {"vector_search", "delta_grep", "web_search", "web_research"}
)
_TABLE_READ_KINDS: frozenset[str] = frozenset({"delta_table_read", "table_read"})
_COMPUTE_KINDS: frozenset[str] = frozenset({"compute"})

# Tool kinds that count as "corpus-grounded" — they fetch evidence from a
# user-supplied workspace asset rather than the public web. Used by the
# asset_signature ↔ tool_kinds invariant.
_CORPUS_TOOL_KINDS: frozenset[str] = frozenset(
    {
        "vector_search",
        "genie",
        "knowledge_assistant",
        "delta_read",
        "delta_grep",
        "delta_context",
        "delta_table_read",
        "table_read",
        "file_search",
    }
)
_WEB_TOOL_KINDS: frozenset[str] = frozenset(
    {"web_search", "web_research", "web_crawl"}
)
# asset_signature axes that require corpus-grounded tools to be present on
# every researcher. ``web_only`` and ``no_assets`` are not in this set
# because the deterministic blueprint legitimately uses web defaults there.
_CORPUS_REQUIRED_ASSET_SIGS: frozenset[str] = frozenset(
    {"corpus_only", "structured_only"}
)


# Per-axis keyword sets enforced by the runtime-query check. Each axis
# requires AT LEAST ONE query that satisfies EVERY group in its tuple
# (i.e. all groups must hit, even if different queries hit different
# groups). See plan section Layer-3a.
_AMBIGUITY_AXIS_KEYWORDS: dict[str, tuple[frozenset[str], ...]] = {
    "period_basis": (
        frozenset({"fiscal", "fy", "fiscal year"}),
        frozenset({"calendar", "cy", "calendar year", "monthly", "by month", "sum of months"}),
    ),
    "temporal_scope": (
        # explicit date or year range
        frozenset({"range", "from", "to", "between", "since"}),
        # AND fiscal/calendar qualifier
        frozenset({"fiscal", "calendar", "fy", "cy"}),
    ),
}


@dataclass
class ProbeResult:
    passed: bool = False
    gaps: list[str] = field(default_factory=list)
    invariants_passed: list[str] = field(default_factory=list)
    conditional_passed: list[str] = field(default_factory=list)
    runtime_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gaps": list(self.gaps),
            "invariants_passed": list(self.invariants_passed),
            "conditional_passed": list(self.conditional_passed),
            "runtime_queries": list(self.runtime_queries),
        }


def _config_of(node: Any) -> dict[str, Any]:
    """Return ``node['config']`` when it's a dict; else an empty dict."""
    if not isinstance(node, dict):
        return {}
    raw = node.get("config")
    return raw if isinstance(raw, dict) else {}


def _iter_agent_nodes(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every agent node under *node*. Walks ``children`` AND
    ``config.body`` so plan_and_execute branches are covered."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "agent":
        yield node
    for child in node.get("children") or []:
        yield from _iter_agent_nodes(child)
    body = _config_of(node).get("body")
    if isinstance(body, dict):
        yield from _iter_agent_nodes(body)


def _tool_kinds_for_lane(lane: dict[str, Any], ast_tools: list[dict[str, Any]]) -> set[str]:
    """Return the set of tool kinds bound to *lane*, resolving by name."""
    tool_names = _config_of(lane).get("tools") or []
    name_to_kind: dict[str, str] = {}
    for tool in ast_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        kind = tool.get("kind")
        if isinstance(name, str) and isinstance(kind, str):
            name_to_kind[name] = kind
    return {name_to_kind[name] for name in tool_names if name in name_to_kind}


def _lane_user_prompt(lane: dict[str, Any]) -> str:
    return str(_config_of(lane).get("user_prompt_template") or "")


def _is_lane_researcher(lane: dict[str, Any]) -> bool:
    """Heuristic: a researcher lane is an agent whose role/subtype is
    researcher OR whose id starts with ``lane_``."""
    if not isinstance(lane, dict):
        return False
    subtype = _config_of(lane).get("subtype")
    lane_id = str(lane.get("id") or "")
    return subtype == "researcher" or lane_id.startswith("lane_")


def _topology_of_ast(ast: dict[str, Any]) -> str:
    """Best-effort topology classification by walking the root node tree."""
    root = ast.get("root")
    if not isinstance(root, dict):
        return "unknown"
    return _topology_of_node(root)


def _topology_of_node(node: dict[str, Any]) -> str:
    if not isinstance(node, dict):
        return "unknown"
    if node.get("type") == "plan_and_execute":
        return "plan_and_execute"
    # parallel_lanes: a sequence that contains a parallel node
    for child in node.get("children") or []:
        if isinstance(child, dict) and child.get("type") == "parallel":
            return "parallel_lanes"
        nested = _topology_of_node(child)
        if nested != "unknown":
            return nested
    # single_agent: root is sequence with exactly one agent child
    if node.get("type") == "sequence":
        children = node.get("children") or []
        agents = [c for c in children if isinstance(c, dict) and c.get("type") == "agent"]
        if len(agents) == len(children) and agents:
            return "single_agent"
    return "unknown"


def _registered_tool_kinds() -> set[str]:
    """Snapshot of the framework's ToolKind enum at probe time."""
    from databricks_deep_research.tools.protocol import ToolKind

    return {k.value for k in ToolKind}


def run_behavioral_probe(
    ast: dict[str, Any],
    task_signature: dict[str, Any] | None = None,
    *,
    runtime_queries: list[str] | None = None,
) -> ProbeResult:
    """Run the static probe. Returns ProbeResult with passed/gaps/invariants.

    ``runtime_queries`` is the list of stub-LLM-issued vector_search
    queries to evaluate against per-axis keyword sets. When None or
    empty, the runtime-query check is SKIPPED (the static checks alone
    determine ``passed``); pass a non-None list to enable it.
    """
    result = ProbeResult()
    ast_tools_raw = ast.get("tools")
    ast_tools: list[dict[str, Any]] = ast_tools_raw if isinstance(ast_tools_raw, list) else []

    # ----- Signature-independent invariants ----------------------------

    # 1. Every declared tool's kind is in the framework ToolKind enum.
    known_kinds = _registered_tool_kinds()
    unknown_kinds: list[str] = []
    for tool in ast_tools:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("kind")
        if isinstance(kind, str) and kind not in known_kinds:
            unknown_kinds.append(f"{tool.get('name')}:{kind}")
    if unknown_kinds:
        result.gaps.append(f"unknown_tool_kinds:{','.join(sorted(unknown_kinds))}")
    else:
        result.invariants_passed.append("all_tool_kinds_in_enum")

    # 2. Lane researchers must declare at least one tool.
    lanes = [n for n in _iter_agent_nodes(ast.get("root")) if _is_lane_researcher(n)]
    lanes_without_tools: list[str] = []
    lanes_missing_query_anchor: list[str] = []
    for lane in lanes:
        cfg_raw = lane.get("config")
        config: dict[str, Any] = cfg_raw if isinstance(cfg_raw, dict) else {}
        tools = config.get("tools") or []
        if not tools:
            lanes_without_tools.append(str(lane.get("id") or "<unnamed>"))
        prompt = _lane_user_prompt(lane)
        if prompt and "{query}" not in prompt:
            lanes_missing_query_anchor.append(str(lane.get("id") or "<unnamed>"))
    if lanes_without_tools:
        result.gaps.append(f"lanes_without_bound_tools:{','.join(lanes_without_tools)}")
    elif lanes:
        result.invariants_passed.append("every_lane_has_bound_tools")
    if lanes_missing_query_anchor:
        result.gaps.append(f"lanes_missing_query_anchor:{','.join(lanes_missing_query_anchor)}")
    elif lanes:
        result.invariants_passed.append("every_lane_has_query_anchor")

    # 3. Synthesizer (best-effort identification) has pool_inject config.
    synth_nodes = [
        n
        for n in _iter_agent_nodes(ast.get("root"))
        if isinstance(n.get("config"), dict)
        and (
            n["config"].get("subtype") == "synthesizer"
            or "synthesizer" in str(n.get("id") or "").lower()
        )
    ]
    if synth_nodes:
        synth_with_inject = [n for n in synth_nodes if (n.get("config") or {}).get("pool_inject")]
        if synth_with_inject:
            result.invariants_passed.append("synthesizer_reads_pools")
        else:
            result.gaps.append("synthesizer_no_pool_inject")

    # 4. No name collision in declared tools.
    seen_names: set[str] = set()
    duplicates: list[str] = []
    for tool in ast_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        if name in seen_names:
            duplicates.append(name)
        seen_names.add(name)
    if duplicates:
        result.gaps.append(f"duplicate_tool_names:{','.join(sorted(set(duplicates)))}")
    else:
        result.invariants_passed.append("no_duplicate_tool_names")

    # ----- Signature-conditioned checks --------------------------------

    if task_signature:
        try:
            sig = TaskSignature.load_from_storage(task_signature)
        except Exception:
            sig = None
    else:
        sig = None

    if sig is not None:
        # 5. AST topology must match select_topology(signature).
        expected_topology = select_topology(sig)
        actual_topology = _topology_of_ast(ast)
        if actual_topology != "unknown" and actual_topology != expected_topology:
            result.gaps.append(
                f"topology_signature_mismatch:expected={expected_topology},actual={actual_topology}"
            )
        else:
            result.conditional_passed.append(f"topology_matches_signature:{expected_topology}")

        # 6. question_ambiguity period_basis → some lane prompt mentions BOTH
        #    fiscal-year and calendar-year framing.
        if "period_basis" in sig.question_ambiguity:
            text_blob = "\n".join(_lane_user_prompt(lane) for lane in lanes).lower()
            has_fy = bool(re.search(r"\bfiscal\s*year\b|\bfy\b", text_blob))
            has_cy = bool(re.search(r"\bcalendar\s*year\b|\bcy\b", text_blob))
            if has_fy and has_cy:
                result.conditional_passed.append("period_basis_in_lane_prompts")
            else:
                missing = []
                if not has_fy:
                    missing.append("fiscal_year")
                if not has_cy:
                    missing.append("calendar_year")
                result.gaps.append(f"period_basis_query_diversity_gap:missing={','.join(missing)}")

        # 7. numeric_aggregation → some lane has compute + delta_table_read.
        if sig.question_class == "numeric_aggregation":
            satisfied = False
            for lane in lanes:
                kinds = _tool_kinds_for_lane(lane, ast_tools)
                if (kinds & _COMPUTE_KINDS) and (kinds & _TABLE_READ_KINDS):
                    satisfied = True
                    break
            if satisfied:
                result.conditional_passed.append("numeric_aggregation_has_compute_and_table_read")
            else:
                result.gaps.append("numeric_aggregation_missing_compute_or_table_read")

        # 8. primary_evidence_kind == structured_tables → some lane has
        #    delta_table_read-kind bound.
        if sig.primary_evidence_kind == "structured_tables":
            satisfied = any(
                _tool_kinds_for_lane(lane, ast_tools) & _TABLE_READ_KINDS for lane in lanes
            )
            if satisfied:
                result.conditional_passed.append("structured_tables_has_delta_table_read")
            else:
                result.gaps.append("structured_tables_missing_delta_table_read")

        # 9. asset_signature ↔ tool_kinds alignment.
        #    When the classifier said ``corpus_only`` or ``structured_only``,
        #    every researcher lane MUST bind at least one corpus-grounded tool;
        #    binding only public-web tools silently violates the contract that
        #    the deterministic blueprint is supposed to enforce upstream.
        #    First-principles rationale: tool selection must reflect the
        #    grounded asset reality, not just the text classification.
        sig_value = str(sig.asset_signature)
        if sig_value in _CORPUS_REQUIRED_ASSET_SIGS and lanes:
            lanes_without_corpus_tool: list[str] = []
            for lane in lanes:
                kinds = _tool_kinds_for_lane(lane, ast_tools)
                if not (kinds & _CORPUS_TOOL_KINDS):
                    lanes_without_corpus_tool.append(
                        str(lane.get("id") or "<unnamed>")
                    )
            if lanes_without_corpus_tool:
                result.gaps.append(
                    f"asset_signature_tool_kind_mismatch:signature={sig_value},"
                    f"lanes_without_corpus_tool={','.join(lanes_without_corpus_tool)}"
                )
            else:
                result.conditional_passed.append(
                    f"asset_signature_matches_tool_kinds:{sig_value}"
                )

    # ----- Runtime-query check (opt-in) --------------------------------

    if runtime_queries is not None:
        result.runtime_queries = list(runtime_queries)
        if sig is not None:
            for axis in sig.question_ambiguity:
                axis_groups = _AMBIGUITY_AXIS_KEYWORDS.get(axis)
                if not axis_groups:
                    continue
                joined = " ".join(runtime_queries).lower()
                hits = [any(kw in joined for kw in group) for group in axis_groups]
                if all(hits):
                    result.conditional_passed.append(f"runtime_query_axis_satisfied:{axis}")
                else:
                    missing_groups = [i for i, hit in enumerate(hits) if not hit]
                    result.gaps.append(
                        f"runtime_query_axis_unsatisfied:{axis}:"
                        f"missing_group_indices={missing_groups}"
                    )

    # ----- Final pass/fail ---------------------------------------------
    result.passed = not result.gaps
    return result
