"""Build-time dataflow reachability checks for workflow definitions.

Complements ``condition_contracts.py`` (which checks condition *type* correctness)
by checking dataflow *reachability*:

* **Pass A — dangling reads:** every effective read (prompt-template variable ∪
  declared ``input_keys`` ∪ tool ``input_mapping`` ∪ condition keys) must resolve
  to a producer visible in lexical scope. A dangling *control* read (a loop
  ``until`` / conditional branch / plan_and_execute evaluator source with no
  producer) is error-severity.
* **Pass B — dead stores:** a produced value consumed by nobody (across STATE,
  POOL, and the RUNTIME-RETURN control channel) is a warning, except terminal
  workflow outputs and pool round-trips.

This is a focused, existence-only walk over a plain ``set[str]`` of visible keys.
It deliberately mirrors ``condition_contracts.py``'s per-node scoping rules with
the same verified field access, but does NOT modify or share that validator's
traversal (Pass A needs only existence, so the schema/availability lattice is
irrelevant). It ships lint-first: diagnostics are warnings unless
``DATAFLOW_CHECK_STRICT`` is set, at which point error-severity diagnostics
become validation errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    SubworkflowNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.conditions import (
    CompositeCondition,
    LLMCondition,
    StateCondition,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.runtime_keys import runtime_seed_keys

_renderer = SafeTemplateRenderer()

# Must match SafeTemplateRenderer's {% for x in y %} syntax: group 1 = loop var.
_FOR_LOOP_VAR = re.compile(r"\{%\s*for\s+(\w+)\s+in\s+\w+\s*%\}")


@dataclass(frozen=True)
class Diagnostic:
    """A single dataflow diagnostic. ``severity`` is intrinsic; lint/strict mode
    only governs whether error-severity diagnostics block validation."""

    message: str
    severity: Literal["error", "warning"]


@dataclass
class DataflowReport:
    """Accumulated diagnostics. ``errors`` block validation (strict mode);
    ``warnings`` are logged only (lint mode)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _template_reads(template: str) -> set[str]:
    """STATE keys a template reads, EXCLUDING loop-local variables.

    ``extract_variables`` returns BOTH the ``{% for x in items %}`` iterable AND
    the loop var ``x`` (because ``{x}`` in the body is matched as a plain
    ``{var}``). The loop var is a local binding, not a state read, so subtract it
    or it false-flags as dangling.
    """
    if not template:
        return set()
    return _renderer.extract_variables(template) - set(_FOR_LOOP_VAR.findall(template))


def effective_reads(cfg: AgentNodeConfig) -> set[str]:
    """STATE keys an agent actually consumes: declared ``input_keys`` ∪ the
    variables referenced by its system/user prompt templates (the authoritative
    read signal — ``input_keys`` are documentation-only), minus loop-local
    variables. Runtime-injected keys are excluded at the Pass A *seed*, not here.
    """
    reads = set(cfg.input_keys)
    reads |= _template_reads(cfg.system_prompt or "")
    reads |= _template_reads(cfg.user_prompt_template or "")
    return reads


# ---------------------------------------------------------------------------
# Pass A — dangling reads (existence-only lexical walk over set[str])
# ---------------------------------------------------------------------------


def _condition_keys(
    cond: StateCondition | LLMCondition | CompositeCondition,
) -> set[str]:
    """State ROOT keys a condition reads.

    Dot-path keys (e.g. ``gate_result.status``) resolve to their root binding
    (``gate_result``); the trailing segments are field access on that binding's
    schema, validated by ``condition_contracts._resolve_condition_path`` — they
    are NOT separate state keys. Pass A only needs root existence, so reduce every
    condition key to its first dot-segment.
    """
    if isinstance(cond, StateCondition):
        return {cond.key.split(".", 1)[0]}
    if isinstance(cond, LLMCondition):
        return {v.split(".", 1)[0] for v in _renderer.extract_variables(cond.prompt_template)}
    keys: set[str] = set()
    for child in cond.conditions:
        keys |= _condition_keys(child)
    return keys


def dangling_reads(definition: WorkflowDefinition) -> list[str]:
    """Pass A (existence-only): every effective read must resolve to a producer
    visible in lexical scope. Seeds the workflow's declared ``required_inputs``
    (NOT just 'query') ∪ ``RUNTIME_INJECTED_KEYS``. ``current_step`` is bound only
    inside plan_and_execute (``_resolve_pae``), never a global seed. POOL reads are
    global (Pass B), not lexically scoped here.
    """
    dangling: list[str] = []
    seed = set(definition.required_inputs) | runtime_seed_keys(definition.runtime_injected_keys)
    _resolve(definition.root, seed, dangling)
    return dangling


def _resolve(node: WorkflowNode, visible: set[str], dangling: list[str]) -> set[str]:
    """Returns the set of STATE keys this node EXPORTS to its parent scope."""
    if node.type == NodeType.agent:
        cfg = AgentNodeConfig(**node.config)
        for key in effective_reads(cfg):
            if key not in visible:
                dangling.append(
                    f"node '{node.id}': read '{key}' has no producer in scope"
                )
        return {cfg.output_key} if cfg.output_key else set()
    if node.type == NodeType.tool:
        cfg_t = ToolNodeConfig(**node.config)
        # input_mapping maps tool-arg -> outer state key; the VALUES must resolve.
        # input_literals are constants and bind_namespace is runtime-mediated —
        # neither needs a producer.
        for state_key in cfg_t.input_mapping.values():
            if state_key not in visible:
                dangling.append(
                    f"tool '{node.id}': input_mapping reads '{state_key}' with no producer"
                )
        return {key for key in (cfg_t.output_key, cfg_t.output_data_key) if key}
    if node.type == NodeType.sequence:
        local = set(visible)
        exported: set[str] = set()
        for child in node.children:
            child_exports = _resolve(child, local, dangling)
            local |= child_exports
            exported |= child_exports
        return exported
    if node.type == NodeType.parallel:
        # Siblings are hidden from each other; each sees only the inherited scope.
        par_exports: set[str] = set()
        for child in node.children:
            par_exports |= _resolve(child, set(visible), dangling)
        return par_exports
    if node.type == NodeType.conditional:
        cfg_c = ConditionalNodeConfig(**node.config)
        for branch in cfg_c.conditions:
            for key in _condition_keys(branch.condition):
                if key not in visible:
                    dangling.append(
                        f"conditional '{node.id}': condition reads '{key}' with no producer"
                    )
        cond_exports: set[str] = set()
        for child in node.children:
            cond_exports |= _resolve(child, set(visible), dangling)
        return cond_exports
    if node.type == NodeType.loop:
        return _resolve_loop(node, visible, dangling)
    if node.type == NodeType.plan_and_execute:
        return _resolve_pae(node, visible, dangling)
    if node.type == NodeType.subworkflow:
        cfg_s = SubworkflowNodeConfig(**node.config)
        # input_mapping maps inner-param -> OUTER state key (the values).
        for outer_key in cfg_s.input_mapping.values():
            if outer_key not in visible:
                dangling.append(
                    f"subworkflow '{node.id}': input_mapping reads '{outer_key}' with no producer"
                )
        return {cfg_s.output_key} if cfg_s.output_key else set()
    return set()


def _resolve_pae(node: WorkflowNode, visible: set[str], dangling: list[str]) -> set[str]:
    """plan_and_execute scope: the planner output and the per-iteration
    ``item_state_key`` (current_step) are visible to the body/evaluator ONLY (NOT
    globally). Exports planner + evaluator outputs to the outer scope (mirrors
    condition_contracts exporting them at the PAE boundary).
    """
    cfg = PlanAndExecuteNodeConfig(**node.config)
    planner = AgentNodeConfig(**cfg.planner)
    for key in effective_reads(planner):
        if key not in visible:
            dangling.append(
                f"node '{node.id}.planner': read '{key}' has no producer in scope"
            )
    inner = set(visible) | {planner.output_key, cfg.item_state_key}
    body_produced: set[str] = set()
    if cfg.body is not None:
        # 2-pass loop-carry: the body re-runs per item, so its outputs carry across
        # iterations. Pass 1 collects body-produced keys (throwaway sink); pass 2
        # resolves the body's reads against the fixpoint.
        body_sink: list[str] = []
        body_produced = _resolve(cfg.body, inner, body_sink)
        inner |= body_produced
        _resolve(cfg.body, inner, dangling)
    # Export planner + body + evaluator outputs: every agent writes its output_key
    # to state, so the body's outputs (e.g. 'findings') remain available to nodes
    # AFTER the plan_and_execute (e.g. the synthesizer). Mirrors the builder's
    # availability walk.
    exported = {planner.output_key} | body_produced
    if cfg.evaluator is not None:
        evaluator = AgentNodeConfig(**cfg.evaluator)
        for key in effective_reads(evaluator):
            if key not in inner:
                dangling.append(
                    f"node '{node.id}.evaluator': read '{key}' has no producer in scope"
                )
        exported |= {evaluator.output_key}
    return exported


def _resolve_loop(node: WorkflowNode, visible: set[str], dangling: list[str]) -> set[str]:
    """2-pass loop-carry fixpoint: a loop body may read a key a *later* node (or a
    later iteration) produces. Pass 1 collects all body-produced keys into a
    throwaway sink; pass 2 resolves reads against ``visible`` ∪ that fixpoint.
    """
    sink: list[str] = []
    produced: set[str] = set()
    scratch = set(visible)
    for child in node.children:
        child_exports = _resolve(child, scratch, sink)
        produced |= child_exports
        scratch |= child_exports
    carry_visible = set(visible) | produced
    exported: set[str] = set()
    for child in node.children:
        exported |= _resolve(child, carry_visible, dangling)
    for key in _condition_keys(LoopNodeConfig(**node.config).until):
        if key not in carry_visible:
            dangling.append(
                f"loop '{node.id}': until-condition reads '{key}' with no producer"
            )
    return exported


# ---------------------------------------------------------------------------
# Pass B (dead stores) + public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Producer:
    """A value produced by a node on a channel (used only by Pass B)."""

    key: str
    channel: Literal["state", "pool"]
    node_id: str


def control_consumed_keys(definition: WorkflowDefinition) -> set[str]:
    """Keys consumed as CONTROL (RUNTIME-RETURN + condition channels): a
    plan_and_execute evaluator's output_key (the loop branches on the returned
    decision), loop ``until`` keys, and conditional branch keys.
    """
    keys: set[str] = set()
    _walk_control(definition.root, keys)
    return keys


def _walk_control(node: WorkflowNode, keys: set[str]) -> None:
    if node.type == NodeType.plan_and_execute:
        cfg = PlanAndExecuteNodeConfig(**node.config)
        if cfg.evaluator is not None:
            keys.add(AgentNodeConfig(**cfg.evaluator).output_key)
        if cfg.body is not None:
            _walk_control(cfg.body, keys)
    elif node.type == NodeType.loop:
        keys |= _condition_keys(LoopNodeConfig(**node.config).until)
    elif node.type == NodeType.conditional:
        for branch in ConditionalNodeConfig(**node.config).conditions:
            keys |= _condition_keys(branch.condition)
    for child in node.children:
        _walk_control(child, keys)


def _collect_producers_reads(
    definition: WorkflowDefinition,
) -> tuple[list[Producer], set[str], set[str]]:
    """One walk gathering producers + reads across STATE and POOL (reads are flat
    key-sets here). STATE producers = agent/tool/PAE-slot output_key; POOL
    producers = pool_writes pools. STATE reads = effective_reads ∪ tool
    input_mapping values ∪ condition keys; POOL reads = pool_inject pools.
    """
    producers: list[Producer] = []
    state_reads: set[str] = set()
    pool_reads: set[str] = set()

    def walk(node: WorkflowNode) -> None:
        if node.type == NodeType.agent:
            cfg = AgentNodeConfig(**node.config)
            if cfg.output_key:
                producers.append(Producer(cfg.output_key, "state", node.id))
            state_reads.update(effective_reads(cfg))
            producers.extend(Producer(pw.pool, "pool", node.id) for pw in cfg.pool_writes)
            pool_reads.update(pi.pool for pi in cfg.pool_inject)
        elif node.type == NodeType.tool:
            cfg_t = ToolNodeConfig(**node.config)
            if cfg_t.output_key:
                producers.append(Producer(cfg_t.output_key, "state", node.id))
            if cfg_t.output_data_key:
                producers.append(Producer(cfg_t.output_data_key, "state", node.id))
            state_reads.update(cfg_t.input_mapping.values())
        elif node.type == NodeType.plan_and_execute:
            cfg_p = PlanAndExecuteNodeConfig(**node.config)
            planner = AgentNodeConfig(**cfg_p.planner)
            producers.append(Producer(planner.output_key, "state", node.id))
            state_reads.update(effective_reads(planner))
            if cfg_p.evaluator is not None:
                evaluator = AgentNodeConfig(**cfg_p.evaluator)
                producers.append(Producer(evaluator.output_key, "state", node.id))
                state_reads.update(effective_reads(evaluator))
            if cfg_p.body is not None:
                walk(cfg_p.body)
        elif node.type == NodeType.conditional:
            for branch in ConditionalNodeConfig(**node.config).conditions:
                state_reads.update(_condition_keys(branch.condition))
        elif node.type == NodeType.loop:
            state_reads.update(_condition_keys(LoopNodeConfig(**node.config).until))
        for child in node.children:
            walk(child)

    walk(definition.root)
    return producers, state_reads, pool_reads


def detect_dead_stores(definition: WorkflowDefinition) -> list[Diagnostic]:
    """Pass B — a producer consumed by NO read on any channel is a dead store
    (warning). Terminal workflow outputs and pool round-trips are exempt; control
    consumption (loop/conditional/PAE evaluator) counts as consumption. A dangling
    *control read* (the error tier) is surfaced by Pass A, not here.
    """
    producers, state_reads, _pool_reads = _collect_producers_reads(definition)
    consumed_state = set(state_reads) | control_consumed_keys(definition)
    terminal = set(definition.output_keys)
    # A node that writes to a pool routes its substantive output through that pool
    # (consumed downstream by pool_inject AND by the runtime citation pipeline), so
    # its state output_key being unread is NOT a dead store. POOL producers are not
    # flagged at all: pool consumption is runtime-mediated (e.g. the citation
    # pipeline reads the sources/observations pools) and not statically
    # determinable from the AST.
    pool_writing_nodes = {p.node_id for p in producers if p.channel == "pool"}
    runtime_keys = set(definition.runtime_injected_keys)
    diagnostics: list[Diagnostic] = []
    for producer in producers:
        if producer.channel == "pool":
            continue
        if producer.key in terminal or producer.key in consumed_state:
            continue
        if producer.key in runtime_keys:
            continue  # runtime-mediated producer (consumed outside the static graph)
        if producer.node_id in pool_writing_nodes:
            continue
        diagnostics.append(
            Diagnostic(
                message=(
                    f"state '{producer.key}' (produced by '{producer.node_id}') "
                    "is read by nobody"
                ),
                severity="warning",
            )
        )
    return diagnostics


def validate_dataflow_contracts(
    definition: WorkflowDefinition, *, strict: bool
) -> DataflowReport:
    """Run Pass A (dangling reads — error-severity) and Pass B (dead stores).

    Lint-first: in lint mode (``strict=False``) every diagnostic is a warning; in
    strict mode error-severity diagnostics become validation errors and
    warning-severity diagnostics stay warnings.
    """
    diagnostics: list[Diagnostic] = [
        Diagnostic(message=message, severity="error")
        for message in dangling_reads(definition)
    ]
    diagnostics.extend(detect_dead_stores(definition))
    report = DataflowReport()
    for diag in diagnostics:
        if strict and diag.severity == "error":
            report.errors.append(diag.message)
        else:
            report.warnings.append(diag.message)
    return report
