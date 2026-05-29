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
from databricks_deep_research.workflow.runtime_keys import RUNTIME_INJECTED_KEYS

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


def effective_reads(cfg: AgentNodeConfig, *, exclude_runtime: bool = False) -> set[str]:
    """STATE keys an agent actually consumes: declared ``input_keys`` ∪ the
    variables referenced by its system/user prompt templates (the authoritative
    read signal — ``input_keys`` are documentation-only), minus loop-local
    variables. Optionally drop runtime-injected keys.
    """
    reads = set(cfg.input_keys)
    reads |= _template_reads(cfg.system_prompt or "")
    reads |= _template_reads(cfg.user_prompt_template or "")
    if exclude_runtime:
        reads -= RUNTIME_INJECTED_KEYS
    return reads


# ---------------------------------------------------------------------------
# Pass A — dangling reads (existence-only lexical walk over set[str])
# ---------------------------------------------------------------------------


def _condition_keys(
    cond: StateCondition | LLMCondition | CompositeCondition,
) -> set[str]:
    """State keys a condition reads (mirrors condition_contracts._validate_condition)."""
    if isinstance(cond, StateCondition):
        return {cond.key}
    if isinstance(cond, LLMCondition):
        return set(_renderer.extract_variables(cond.prompt_template))
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
    seed = set(definition.required_inputs) | set(RUNTIME_INJECTED_KEYS)
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
        for state_key in cfg_t.input_mapping.values():
            if state_key not in visible:
                dangling.append(
                    f"tool '{node.id}': input_mapping reads '{state_key}' with no producer"
                )
        return {cfg_t.output_key} if cfg_t.output_key else set()
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
    if cfg.body is not None:
        inner |= _resolve(cfg.body, inner, dangling)
    exported = {planner.output_key}
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
    """Single forward pass (Phase 2 upgrades this to a 2-pass loop-carry fixpoint).
    A loop ``until`` condition reads state as control; a dangling control read here
    is surfaced like any other dangling read.
    """
    local = set(visible)
    exported: set[str] = set()
    for child in node.children:
        child_exports = _resolve(child, local, dangling)
        local |= child_exports
        exported |= child_exports
    for key in _condition_keys(LoopNodeConfig(**node.config).until):
        if key not in local:
            dangling.append(
                f"loop '{node.id}': until-condition reads '{key}' with no producer"
            )
    return exported


# ---------------------------------------------------------------------------
# Pass B (dead stores) + public entry point
# ---------------------------------------------------------------------------


def detect_dead_stores(definition: WorkflowDefinition) -> list[Diagnostic]:
    """Pass B — produced values consumed by nobody. Implemented in Phase 2
    (US-DF5); returns no diagnostics until then."""
    _ = definition
    return []


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
