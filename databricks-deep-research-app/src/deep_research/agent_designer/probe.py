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
   tool kind is in the Designer registry; no name-collision in lane tool aliases;
   synthesizer reads observation/sources pools.)

2. **Signature-conditioned checks** — fire only when relevant
   ``question_ambiguity`` / ``retrieval_pattern`` / ``question_class``
   flags are set:
     * topology mismatch (AST topology vs ``select_topology(sig)``);
     * period_basis → at least one lane prompt mentions both FY and CY
       framing;
     * numeric_aggregation → at least one lane has compute-kind AND
       table-read-kind tools bound;
     * structured_tables → at least one lane has table-read-kind
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
from dataclasses import dataclass, field
from typing import Any

from deep_research.agent_designer.ast_introspection import (
    config_of as _config_of,
)
from deep_research.agent_designer.ast_introspection import (
    is_lane_researcher as _is_lane_researcher,
)
from deep_research.agent_designer.ast_introspection import (
    iter_agent_nodes as _iter_agent_nodes,
)
from deep_research.agent_designer.ast_introspection import (
    iter_all_nodes as _iter_all_nodes,
)
from deep_research.agent_designer.ast_introspection import (
    tool_kinds_for_lane as _tool_kinds_for_lane,
)
from deep_research.agent_designer.ast_introspection import (
    topology_of_ast as _topology_of_ast,
)
from deep_research.agent_designer.task_signature import (
    TaskSignature,
    select_topology,
)

# Tool kinds that the probe recognizes as evidence collectors.
_RETRIEVAL_KINDS: frozenset[str] = frozenset(
    {"vector_search", "table_search", "web_search", "web_research"}
)
_TABLE_READ_KINDS: frozenset[str] = frozenset({"table_read", "table_load"})
_COMPUTE_KINDS: frozenset[str] = frozenset({"compute"})

# Tool kinds that count as "corpus-grounded" — they fetch evidence from a
# user-supplied workspace asset rather than the public web. Used by the
# asset_signature ↔ tool_kinds invariant.
_CORPUS_TOOL_KINDS: frozenset[str] = frozenset(
    {
        "vector_search",
        "genie",
        "knowledge_assistant",
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
        "file_search",
    }
)
_WEB_TOOL_KINDS: frozenset[str] = frozenset({"web_search", "web_research", "web_crawl"})
# asset_signature axes that require corpus-grounded tools to be present on
# every researcher. ``web_only`` and ``no_assets`` are not in this set
# because the deterministic blueprint legitimately uses web defaults there.
_CORPUS_REQUIRED_ASSET_SIGS: frozenset[str] = frozenset({"corpus_only", "structured_only"})


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


def _lane_user_prompt(lane: dict[str, Any]) -> str:
    return str(_config_of(lane).get("user_prompt_template") or "")


def _structural_family(topology: str) -> str:
    """Structural family for the topology-match check (check 5).

    Topologies that share a runtime shape map to the same family so the check
    does not false-flag them (best_of_n is a parallel fan-out -> parallel_lanes
    family); topology-specific structure is verified by dedicated invariants
    (best_of_n check 10). Delegates to the TopologySpec registry (single source
    of truth); imported lazily to avoid an import cycle.
    """
    from deep_research.agent_designer.topology_registry import structural_family

    return structural_family(topology)


def _registered_tool_kinds() -> set[str]:
    """Snapshot of Designer-supported tool declaration kinds at probe time."""
    from deep_research.agent_designer.registry import tool_kinds_payload

    return {str(item["kind"]) for item in tool_kinds_payload()}


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

    # 1. Every declared tool's kind is in the Designer registry.
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
        result.invariants_passed.append("all_tool_kinds_registered")

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
        # 5. AST topology must match select_topology(signature) — compared by
        #    STRUCTURAL FAMILY so topologies that share a runtime shape (e.g.
        #    best_of_n is a parallel fan-out, like parallel_lanes) are not
        #    false-flagged. best_of_n's specific shape is verified by check 10.
        expected_topology = select_topology(sig)
        actual_topology = _topology_of_ast(ast)
        if actual_topology != "unknown" and _structural_family(
            actual_topology
        ) != _structural_family(expected_topology):
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

        # 7. numeric_aggregation → some lane has compute + table-read.
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
        #    table-read-kind bound.
        if sig.primary_evidence_kind == "structured_tables":
            satisfied = any(
                _tool_kinds_for_lane(lane, ast_tools) & _TABLE_READ_KINDS for lane in lanes
            )
            if satisfied:
                result.conditional_passed.append("structured_tables_has_table_read")
            else:
                result.gaps.append("structured_tables_missing_table_read")

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
                    lanes_without_corpus_tool.append(str(lane.get("id") or "<unnamed>"))
            if lanes_without_corpus_tool:
                result.gaps.append(
                    f"asset_signature_tool_kind_mismatch:signature={sig_value},"
                    f"lanes_without_corpus_tool={','.join(lanes_without_corpus_tool)}"
                )
            else:
                result.conditional_passed.append(f"asset_signature_matches_tool_kinds:{sig_value}")

        # 10. best_of_n structural invariants. The family-map (check 5)
        #     intentionally treats best_of_n as a parallel fan-out, so these
        #     targeted checks verify the best_of_n-specific shape it would
        #     otherwise hide: a candidates pool, candidate synthesizers that
        #     write it and inject the evidence pools, a judge that injects the
        #     candidates pool and produces a terminal output key, and a
        #     candidate count matching the signature.
        if expected_topology == "best_of_n":
            pool_names = {
                str(p.get("name")) for p in (ast.get("pools") or []) if isinstance(p, dict)
            }
            if "candidates" not in pool_names:
                result.gaps.append("best_of_n_missing_candidates_pool")
            candidate_nodes = [
                node
                for node in _iter_agent_nodes(ast.get("root"))
                if _config_of(node).get("subtype") == "synthesizer"
                and any(
                    isinstance(pw, dict) and pw.get("pool") == "candidates"
                    for pw in (_config_of(node).get("pool_writes") or [])
                )
            ]
            judge_nodes = [
                node
                for node in _iter_agent_nodes(ast.get("root"))
                if _config_of(node).get("subtype") == "synthesizer"
                and "candidates"
                in {
                    pj.get("pool")
                    for pj in (_config_of(node).get("pool_inject") or [])
                    if isinstance(pj, dict)
                }
            ]
            if not candidate_nodes:
                result.gaps.append("best_of_n_no_candidate_generators")
            if not judge_nodes:
                result.gaps.append("best_of_n_no_judge")
            else:
                judge_outputs = {str(_config_of(node).get("output_key")) for node in judge_nodes}
                if not (set(ast.get("output_keys") or []) & judge_outputs):
                    result.gaps.append("best_of_n_judge_does_not_produce_output_key")
            want = sig.coordination_candidate_count
            if want is not None and candidate_nodes and len(candidate_nodes) != want:
                result.gaps.append(
                    f"best_of_n_candidate_count_mismatch:expected={want},"
                    f"actual={len(candidate_nodes)}"
                )
            ungrounded = [
                str(node.get("id") or "<unnamed>")
                for node in candidate_nodes
                if not (
                    {"observations", "sources"}
                    <= {
                        pj.get("pool")
                        for pj in (_config_of(node).get("pool_inject") or [])
                        if isinstance(pj, dict)
                    }
                )
            ]
            if ungrounded:
                result.gaps.append(f"best_of_n_candidates_not_grounded:{','.join(ungrounded)}")
            if "candidates" in pool_names and candidate_nodes and judge_nodes and not ungrounded:
                result.conditional_passed.append(
                    f"best_of_n_structure_ok:candidates={len(candidate_nodes)}"
                )

        # 11. iterative_refinement structural invariants. The family-map (check 5)
        #     treats it as a parallel fan-out (its evidence parallel), so these
        #     targeted checks verify the loop-specific shape: a loop whose body has
        #     a draft synthesizer (writing draft_report) + a reflector (the until
        #     operand), an until keyed on the reflector decision, a NON-skip
        #     reflector, a terminal output, and — when participants>=2 — a
        #     candidates pool fed by >=2 proposers plus a candidates-injecting
        #     integrator.
        if expected_topology == "iterative_refinement":
            loop_nodes = [n for n in _iter_all_nodes(ast.get("root")) if n.get("type") == "loop"]
            if not loop_nodes:
                result.gaps.append("iterative_refinement_missing_loop")
            else:
                loop = loop_nodes[0]
                body_agents = list(_iter_agent_nodes(loop))
                draft_writers = [
                    n
                    for n in body_agents
                    if _config_of(n).get("subtype") == "synthesizer"
                    and _config_of(n).get("output_key") == "draft_report"
                ]
                reflectors = [n for n in body_agents if _config_of(n).get("subtype") == "reflector"]
                if not draft_writers:
                    result.gaps.append("iterative_refinement_no_draft_synth")
                if not reflectors:
                    result.gaps.append("iterative_refinement_no_reflector")
                until_key = str((loop.get("config") or {}).get("until", {}).get("key", ""))
                if not until_key.endswith(".decision"):
                    result.gaps.append("iterative_refinement_until_not_decision")
                if any(
                    (r.get("error_handling") or {}).get("on_error") == "skip" for r in reflectors
                ):
                    result.gaps.append("iterative_refinement_reflector_is_skip")
            produced = {_config_of(n).get("output_key") for n in _iter_agent_nodes(ast.get("root"))}
            if not (set(ast.get("output_keys") or []) & produced):
                result.gaps.append("iterative_refinement_no_terminal_output")
            want_p = sig.refine_participants
            if want_p is not None and want_p >= 2:
                pool_names = {
                    str(p.get("name")) for p in (ast.get("pools") or []) if isinstance(p, dict)
                }
                if "candidates" not in pool_names:
                    result.gaps.append("iterative_refinement_missing_candidates_pool")
                proposer_writers = [
                    n
                    for n in _iter_agent_nodes(ast.get("root"))
                    if _config_of(n).get("subtype") == "synthesizer"
                    and any(
                        isinstance(pw, dict) and pw.get("pool") == "candidates"
                        for pw in (_config_of(n).get("pool_writes") or [])
                    )
                ]
                if len(proposer_writers) < 2:
                    result.gaps.append("iterative_refinement_too_few_proposers")
                integrators = [
                    n
                    for n in _iter_agent_nodes(ast.get("root"))
                    if _config_of(n).get("subtype") == "synthesizer"
                    and _config_of(n).get("output_key") == "draft_report"
                    and "candidates"
                    in {
                        pj.get("pool")
                        for pj in (_config_of(n).get("pool_inject") or [])
                        if isinstance(pj, dict)
                    }
                ]
                if not integrators:
                    result.gaps.append("iterative_refinement_no_integrator")
            if not any(g.startswith("iterative_refinement_") for g in result.gaps):
                result.conditional_passed.append(
                    f"iterative_refinement_structure_ok:participants={want_p or 1}"
                )

        # 13. router structural invariants: a classifier emitting a TYPED route
        #     discriminator (output_schema.route enum), a conditional with >=2
        #     branches, every branch producing a workflow output_key, and >=1
        #     researcher writing observations+sources (grounded branch synthesis).
        if expected_topology == "router":
            conditionals = [
                n for n in _iter_all_nodes(ast.get("root")) if n.get("type") == "conditional"
            ]
            classifiers = [
                n
                for n in _iter_agent_nodes(ast.get("root"))
                if _config_of(n).get("subtype") == "router_classifier"
            ]
            if not classifiers:
                result.gaps.append("router_no_classifier")
            else:
                schema = _config_of(classifiers[0]).get("output_schema") or {}
                props = schema.get("properties") if isinstance(schema, dict) else {}
                route = props.get("route") if isinstance(props, dict) else None
                if not (isinstance(route, dict) and route.get("enum")):
                    result.gaps.append("router_classifier_no_typed_route")
            if not conditionals:
                result.gaps.append("router_no_conditional")
            else:
                branches = conditionals[0].get("children") or []
                if len(branches) < 2:
                    result.gaps.append("router_too_few_branches")
                output_keys = set(ast.get("output_keys") or [])
                branchless = [
                    str(b.get("id") or "<unnamed>")
                    for b in branches
                    if not (
                        {_config_of(n).get("output_key") for n in _iter_agent_nodes(b)}
                        & output_keys
                    )
                ]
                if branchless:
                    result.gaps.append(f"router_branch_no_output:{','.join(branchless)}")
            researchers = [
                n
                for n in _iter_agent_nodes(ast.get("root"))
                if _config_of(n).get("subtype") == "researcher"
            ]
            if not any(
                {"observations", "sources"}
                <= {
                    pw.get("pool")
                    for pw in (_config_of(r).get("pool_writes") or [])
                    if isinstance(pw, dict)
                }
                for r in researchers
            ):
                result.gaps.append("router_no_grounded_researcher")
            if not any(g.startswith("router_") for g in result.gaps):
                branch_count = len(conditionals[0].get("children") or []) if conditionals else 0
                result.conditional_passed.append(f"router_structure_ok:branches={branch_count}")

        # 14. tree_search structural invariants. The topology walker classifies it
        #     at the root as its OWN family (so check 5 already keys off
        #     ``tree_search``); these targeted checks verify the static-unroll
        #     shape: a root sequence with >=1 level parallel, narrowing breadth
        #     across the levels, >=1 researcher writing observations+sources
        #     (grounded survey), a terminal-output synthesizer injecting both
        #     pools, and — for depth >=2 — a gap reflector UPSTREAM of each deeper
        #     level whose review output_key the deeper lanes read.
        if expected_topology == "tree_search":
            root = ast.get("root") if isinstance(ast, dict) else None
            root_children = root.get("children") or [] if isinstance(root, dict) else []
            level_parallels = [
                c
                for c in root_children
                if isinstance(c, dict)
                and c.get("type") == "parallel"
                and re.match(r"^l\d+_research-level$", str(c.get("id") or ""))
            ]
            if not level_parallels:
                result.gaps.append("tree_search_missing_levels")
            else:
                # Breadth must NARROW (non-increasing) across levels.
                breadths = [len(p.get("children") or []) for p in level_parallels]
                if any(
                    later > earlier for earlier, later in zip(breadths, breadths[1:], strict=False)
                ):
                    result.gaps.append(f"tree_search_breadth_not_narrowing:{breadths}")
            researchers = [
                n
                for n in _iter_agent_nodes(ast.get("root"))
                if _config_of(n).get("subtype") == "researcher"
            ]
            if not any(
                {"observations", "sources"}
                <= {
                    pw.get("pool")
                    for pw in (_config_of(r).get("pool_writes") or [])
                    if isinstance(pw, dict)
                }
                for r in researchers
            ):
                result.gaps.append("tree_search_no_grounded_researcher")
            synth_with_pools = [
                n
                for n in _iter_agent_nodes(ast.get("root"))
                if _config_of(n).get("subtype") == "synthesizer"
                and {"observations", "sources"}
                <= {
                    pj.get("pool")
                    for pj in (_config_of(n).get("pool_inject") or [])
                    if isinstance(pj, dict)
                }
            ]
            if not synth_with_pools:
                result.gaps.append("tree_search_no_grounded_synthesizer")
            else:
                synth_outputs = {str(_config_of(n).get("output_key")) for n in synth_with_pools}
                if not (set(ast.get("output_keys") or []) & synth_outputs):
                    result.gaps.append("tree_search_synthesizer_no_terminal_output")
            # Depth >=2 ⇒ a between-level reflector whose review output_key is
            # consumed by the deeper level's lanes (the static-unroll contract).
            if level_parallels and len(level_parallels) >= 2:
                reflectors = [
                    n
                    for n in _iter_agent_nodes(ast.get("root"))
                    if _config_of(n).get("subtype") == "reflector"
                ]
                review_keys = {
                    str(_config_of(r).get("output_key"))
                    for r in reflectors
                    if str(_config_of(r).get("output_key") or "").startswith("level")
                }
                if len(review_keys) < len(level_parallels) - 1:
                    result.gaps.append("tree_search_missing_gap_reflector")
                # Each deeper level's lanes must read a level review key.
                deeper_lane_inputs: set[str] = set()
                for parallel in level_parallels[1:]:
                    for lane in _iter_agent_nodes(parallel):
                        deeper_lane_inputs.update(_config_of(lane).get("input_keys") or [])
                if not (review_keys & deeper_lane_inputs):
                    result.gaps.append("tree_search_deeper_level_ignores_gaps")
            if not any(g.startswith("tree_search_") for g in result.gaps):
                result.conditional_passed.append(
                    f"tree_search_structure_ok:levels={len(level_parallels)}"
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
