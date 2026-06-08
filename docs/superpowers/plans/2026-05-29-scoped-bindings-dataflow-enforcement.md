# Scoped Bindings & Dataflow Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a build-time dataflow checker for generated workflows that flags reads with no producer ("read from nowhere") and produced values that nothing consumes ("nothing wasted") across the STATE, POOL, and RUNTIME-RETURN channels, and fix the body-reflector defect that motivated it.

**Architecture:** A new `dataflow_contracts.py` in the framework implements a **focused, existence-only tree-walk** (`set[str]` visibility) that mirrors `condition_contracts.py`'s per-node scoping rules with verified field access and is tested against the same topology corpus — it does **not** modify or share that validator's traversal (Pass A needs only existence, so the schema/availability lattice is irrelevant). It checks each node's *effective reads* — prompt-template variables (the authoritative read signal; `input_keys` are documentation-only per `agents/config.py:259-262`) ∪ declared `input_keys` ∪ tool `input_mapping` values ∪ condition keys — minus a framework-level **runtime-injected key registry**. Pass A = dangling reads (data + control). Pass B = dead stores (zero consumption edges across all channels), severity-tiered. Ships **lint-first** (warnings, non-blocking), flips to **strict** (errors) per `DATAFLOW_CHECK_STRICT` once the generated-workflow corpus measures clean.

**Tech Stack:** Python 3.11, Pydantic v2, pytest (`unit` marker), mypy strict, ruff. Framework pkg `databricks-deep-research`; app builder in `databricks-deep-research-app`.

**Spec:** `docs/superpowers/specs/2026-05-29-scoped-bindings-dataflow-enforcement-design.md`

---

## Consensus revisions incorporated (v2)

This plan was hardened through two review passes (Architect + Codex gpt-5.5 xhigh). v1 → v2 changes, with evidence:

1. **Reuse condition_contracts' traversal; do not hand-roll a parallel walk.** v1's parallel walk got core field names wrong (`node.node_type`→ real `node.type` at `definition.py:100`; `ConditionalNodeConfig.branches`→ real `.conditions` at `config.py:207`; subworkflow `input_keys`→ real `input_mapping`/`output_mapping` at `config.py:230-231`). Reusing the existing, tested traversal eliminates this class of bug by construction and fixes the "reuse vocabulary" principle (v1 only *claimed* reuse).
2. **Reads are prompt-template variables, not static `input_keys`.** `config.py:259-262` states `input_keys` are documentation-only; the harness auto-detects reads from templates via `SafeTemplateRenderer.extract_variables()` (`harness.py:353-364`). Raw `input_keys` only render if the template references them (`harness.py:1042-1060`). The checker must extract template variables as the primary read set.
3. **Runtime-injected key registry, not a 6-key hand-list.** A 40+-key inventory already exists app-side (`_RUNTIME_TEMPLATE_KEYS`, `workflow_builder.py:28-80`). Promote it to a framework-level registry (the framework can't import the app); seed Pass A from it.
4. **Drop the subtype-based "redundant decision-maker" check — it false-positives a real node.** The live PAE evaluator is `subtype="reflector"` (`workflow_builder.py:1968`), identical to the dead body reflector, and `parallel_lanes` has a legitimate `coverage-reflector` (`subtype="reflector"`, `output_key="coverage_review"`) read as *data* by the finalizer (`workflow_builder.py:2478-2513`). Decision-maker status is **positional** (consumed by a control edge), never subtype. See "What the checker does NOT catch."
5. **Tool-node reads.** `ToolNodeConfig.input_mapping.values()` are STATE reads the runtime resolves (`executor.py:1010-1014`); the ledger must record them.
6. **Phase 0: `{evaluation}` must be an actual prompt slot.** Adding `evaluation` to `input_keys` alone is a no-op (it only renders if the template references it). To honor "synthesizer reads evaluation," add an explicit `{evaluation}` slot to the synthesizer directive.
7. **Severity is intrinsic to the diagnostic; lint/strict only gates blocking.** The error tier is a *dangling control read* (a loop/conditional/PAE control source with no producer); dead data stores are warnings. Spec §3.2 was updated to match — a "dead control *producer*" read only as data is intentionally not flagged (it would false-positive the legitimate `coverage-reflector`).
8. **Adversarial broken-workflow tests** (false-negative measurement), real loader-based fixtures (not `WorkflowDefinition(**dict)`), parametrized domains.

### Iteration-2 fixes (v2 → final)
A second Architect + Codex (gpt-5.5 xhigh) pass found, and this plan applies:
9. **[BLOCKER] Pass A seeding:** seed `definition.required_inputs` (not just `query`) and **scope `current_step` to plan_and_execute only** (it was wrongly a global seed) — `condition_contracts.py:142-152`, `plan_execute_runner.py:378`.
10. **[MAJOR] Loop-local over-collection:** `extract_variables` returns the `{%for x in …%}` loop var (as `{x}` in the body); `effective_reads` now subtracts loop-locals (`test_renderer.py:82-84`).
11. **[MAJOR] Severity/spec reconciliation:** dropped the spec's "dead control producer = error" tier (it would false-positive the legitimate `coverage-reflector`); the error tier is now a **dangling control read**. Spec §3.2 updated to match.
12. **[MAJOR] Coercion decoupling:** the app's `_RUNTIME_TEMPLATE_KEYS` feeds lane-prompt coercion (`workflow_builder.py:84-91`), so it is **left unchanged**; the framework registry is separate.
13. **[MAJOR] Test paths:** corrected to `tests/unit/workflow/` and `tests/unit/agent_designer/`.
14. **[MINOR] `set[str]` existence walk** (no private `ValidationScope`/`_UNKNOWN_SCHEMA` coupling); Task 1.0 refactor **skipped** (`condition_contracts.py` unchanged); subworkflow dead code removed.

**Consensus status:** 3 iterations (Planner→Architect→Codex gpt-5.5 xhigh each). **The BLOCKER is RESOLVED** — both the iter-3 Architect (`APPROVE`) and the iter-3 Codex (`BLOCKER status: RESOLVED`) confirm Pass A now seeds `definition.required_inputs` and scopes `current_step` to plan_and_execute. Iter-3 Codex returned `ITERATE` solely for **internal-consistency drift** — stale prose left by the iter-2 finalize edits (a `Modify app workflow_builder` Files line, "shared scope helpers" / Task 1.0 remnants, a "seed `current_step`" phrase, and the spec §1.3/§2.4 `DEAD STORE` glyph) contradicting the corrected snippets. **All four were fixed** in iteration 3 (verified clean). No correctness/logic defects remain; the Architect validated all post-iter-2 code edits against source. Remaining items are judgment calls in the ADR (keep `{evaluation}` vs. drop it; full traversal unification), not blockers.

---

## Planning Rationale (RALPLAN-DR)

### Principles
1. **Less permissive, by resolution only.** Tighten which producer a read resolves to; never change state storage/mutability (state is SSA-like: frozen `StateEntry`, serialized reads — `state.py:45`, `harness.py:720`).
2. **Structural checks, never domain/topology checks** ([[feedback-no-hardcoded-domains-or-topologies]]).
3. **Lint before strict.** A checker that over-blocks is worse than none; every diagnostic ships non-blocking, is measured against the corpus, then gated.
4. **Reuse the existing traversal and vocabulary** — `condition_contracts.py` already walks the tree correctly with the right field access; build on it, do not fork it.
5. **Prevention over cleanup** ([[feedback-prevention-over-cleanup]]).

### Decision Drivers
1. **Phase 0 correctness.** Verified: `evaluation` lands in STATE (`harness.py:720`); the synthesizer prompt does not reference `{reflection}` as a template var (so deleting the reflector leaves no dangling var); but raw `input_keys` are a no-op without a template slot (`harness.py:1042-1060`) — so `{evaluation}` must be added to the prompt.
2. **Zero false positives on current output.** The 3 builder topologies must pass with zero warnings — including all runtime-injected and template-derived reads.
3. **The checker must earn strictness.** Phase 5 (free-form synthesis) is gated on measured zero *false negatives*; this plan delivers the checker **and** the adversarial measurement, not the unlock.

### Viable Options Considered
- **Option A-refined (chosen after iter-2): focused new module with its own existence-only walk (`set[str]` visibility), verified field access, tested against the same topology corpus as `condition_contracts`.** Pros: leaves the working condition validator untouched; no private-type coupling; Pass A needs only existence, so the `FieldSchema`/availability lattice is irrelevant. Cons: a second (small) walk — mitigated by shared corpus tests (Task 1.5); full unification is a follow-up.
- **Option B (rejected after iter-2): refactor `condition_contracts` to expose shared scope helpers and reuse them.** Why rejected: iter-2 review showed Pass A is existence-only, so reusing the schema-heavy `ValidationScope`/`_merge_branch_outputs` lattice is unnecessary AND risky — a naive reuse drops the `maybe`/`always` availability the validator computes, reintroducing the very drift "reuse" was meant to prevent. The refactor's blast radius on a working validator isn't justified for an existence check.
- **Option C (rejected): full lexical scoping + free-form synthesis in one plan.** Cons: synthesis is gated on a measurement that doesn't exist yet; violates lint-before-strict.

**Why Option A-refined:** the v1 failure was wrong *field names*, not the parallel walk itself. With every field name now verified against source and `set[str]` existence semantics, a focused walk is correct, simpler, and leaves `condition_contracts.py` unchanged. Honesty note: this is a deliberate second walk — the plan does **not** claim to reuse `condition_contracts`' traversal; it mirrors the scoping rules and is tested against the same topologies (Task 1.5), with unification deferred to a follow-up.

### Pre-Mortem (3 scenarios)
1. **Strict flip blocks deploys on a missed runtime-injected key.** Mitigation: lint-first; the registry is the full 40+-key inventory (Task 1.1), not a guess; corpus gate (Task 1.5/2.5) asserts zero warnings before any strict flip; new seed keys must cite the producing context-builder.
2. **The shared-helper refactor breaks the working condition validator.** Mitigation: Task 1.0 is a pure extraction guarded by the existing `tests/unit/workflow/test_condition_contracts.py` — it must stay green with zero diffs to expected errors; no behavior change, only factoring.
3. **The checker passes a genuinely broken workflow (false negative) → Phase 5 ships silently-wrong topologies.** Mitigation: Task 2.4 builds adversarial broken workflows (dangling data read, dangling control read, dead store, dangling tool input) and asserts each is flagged — this is the false-negative measurement Phase 5's gate requires.

### Test Strategy
- **Unit:** `tests/unit/workflow/test_dataflow_contracts.py` — read extraction, registry seeding, dangling (data+control), dead-store tiers, fixpoint, lexical resolution.
- **Integration (corpus):** `tests/unit/workflow/test_dataflow_corpus.py` — build each topology via the **real loader** across a parametrized domain set; assert zero warnings (the strict-flip gate).
- **Adversarial (false-negative):** `tests/unit/workflow/test_dataflow_adversarial.py` — inject each defect class; assert detection.
- **E2e (smoke):** `make e2e-medium` plan_and_execute path still yields a grounded report after Phase 0.
- **Observability:** diagnostics logged with prefix `DATAFLOW_LINT` (lint) / raised as `WorkflowValidationError` (strict).

---

## Scope of THIS plan

| Phase | In plan? | Deliverable |
|---|---|---|
| **0** | ✅ Full TDD | Delete body reflector; synthesizer gains a real `{evaluation}` slot. |
| **1** | ✅ Full TDD | Runtime-key registry; effective-read model; Pass A (dangling reads, `set[str]` walk), lint. |
| **2** | ✅ Full TDD | Control edges; Pass B (dead stores) + fixpoint; adversarial tests; strict-flip gate. |
| **3** | ⛔ Deferred | Visualization (own plan). |
| **4** | ⛔ Deferred | Spike: is `plan_and_execute` a macro? |
| **5** | ⛔ Deferred | Structured-freedom synthesis — gated on Task 2.4/2.5 measurement. |

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `databricks-deep-research-app/.../agent_designer/workflow_builder.py` | Remove body reflector; add `{evaluation}` slot to synthesizer. | Modify (`:1925-1950`, `:2008-2032`) |
| `databricks-deep-research/.../workflow/runtime_keys.py` | Framework-level `RUNTIME_INJECTED_KEYS` registry. | **Create** |
| `databricks-deep-research/.../workflow/condition_contracts.py` | **Unchanged** (Task 1.0 skipped; Pass A is self-contained). | — |
| `databricks-deep-research/.../workflow/dataflow_contracts.py` | Read extraction, ledger, Pass A, Pass B, `validate_dataflow_contracts`. | **Create** |
| `databricks-deep-research/.../workflow/validation.py` | Call dataflow check after condition contracts; gate strict. | Modify (`:241-242`) |
| `databricks-deep-research-app/.../agent_designer/workflow_builder.py` | **Unchanged for the registry** (Task 1.1 Step 4 — aliasing would change lane coercion); DRY reconciliation is a follow-up. | — |
| `databricks-deep-research/tests/unit/workflow/test_dataflow_contracts.py` | Unit tests. | **Create** |
| `databricks-deep-research/tests/unit/workflow/test_dataflow_adversarial.py` | False-negative measurement. | **Create** |
| `databricks-deep-research/tests/unit/workflow/test_dataflow_corpus.py` | False-positive gate (loader-based). | **Create** |
| `databricks-deep-research-app/tests/unit/agent_designer/test_workflow_builder_plan_execute.py` | Phase 0 assertions. | **Create/extend** |

---

## Phase 0 — Fix the triggering agent's builder

### Task 0.1: Remove the dead body reflector; give the synthesizer a real `{evaluation}` slot

**Files:**
- Modify: `databricks-deep-research-app/src/deep_research/agent_designer/workflow_builder.py:1925-1950`, `:2008-2032`, `_plan_execute_synthesizer_directive` (`:2210`)
- Test: `databricks-deep-research-app/tests/unit/agent_designer/test_workflow_builder_plan_execute.py`

- [ ] **Step 1: Read the canonical brief fixture.** Open `tests/unit/agent_designer/` and find how existing tests construct a `WorkflowDesignBrief` / call `_build_plan_and_execute_workflow`. Reuse that exact fixture (do NOT invent `model_construct(...)` with `[...]`). If none exists, build the brief via the real compile path used elsewhere in the builder tests.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/agent_designer/test_workflow_builder_plan_execute.py
from deep_research.agent_designer.workflow_builder import _build_plan_and_execute_workflow


def _find(node: dict, node_id: str) -> dict:
    if node.get("id") == node_id:
        return node
    for child in node.get("children", []) or []:
        hit = _find(child, node_id)
        if hit:
            return hit
    cfg = node.get("config", {})
    for key in ("body", "planner", "evaluator"):
        sub = cfg.get(key)
        if isinstance(sub, dict):
            hit = _find(sub, node_id)
            if hit:
                return hit
    return {}


def test_plan_execute_body_is_direct_researcher(plan_execute_brief):  # fixture from Step 1
    wf = _build_plan_and_execute_workflow(intent="t", name="t", compiled_brief=plan_execute_brief)
    pae = _find(wf["root"], "plan-and-execute")
    body = pae["config"]["body"]
    assert body["type"] == "agent" and body["config"]["subtype"] == "researcher"
    assert body["config"]["output_key"] != "reflection"


def test_synthesizer_has_evaluation_slot_not_reflection(plan_execute_brief):
    wf = _build_plan_and_execute_workflow(intent="t", name="t", compiled_brief=plan_execute_brief)
    synth = _find(wf["root"], "synthesizer")
    assert "evaluation" in synth["config"]["input_keys"]
    assert "reflection" not in synth["config"]["input_keys"]
    # The slot must actually be referenced so the read is not a no-op.
    rendered = synth["config"]["system_prompt"] + synth["config"].get("user_prompt_template", "")
    assert "{evaluation}" in rendered
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd databricks-deep-research-app && uv run pytest tests/unit/agent_designer/test_workflow_builder_plan_execute.py -v`
Expected: FAIL — body is a `sequence`; `reflection` in synthesizer input_keys; no `{evaluation}` slot.

- [ ] **Step 4: Delete the body reflector, collapse body**

Delete the `reflector = make_agent_node(...)` block (`:1925-1945`) and replace `body = make_sequence(node_id="research-body", ..., children=[researcher, reflector])` (`:1946-1950`) with:

```python
    # Body is the direct researcher. The loop's continue/replan/complete decision
    # is driven by the evaluator (``evaluator=`` below); a body-level reflector
    # would emit a control decision nothing reads — a dead control output.
    # See docs/superpowers/specs/2026-05-29-scoped-bindings-dataflow-enforcement-design.md §1.1.
    body = researcher
```

Leave `reflector_system` / `_reflector_workflow_directive(...)` — still used by `evaluator=` (`:1976-1983`).

- [ ] **Step 5: Repoint synthesizer input_keys + add the `{evaluation}` slot**

At `:2012` change `input_keys=["query", "research_plan", "findings", "reflection"]` to `["query", "research_plan", "findings", "evaluation"]`.

In `_plan_execute_synthesizer_directive` (`:2210`), append to the returned directive string a labelled control-context slot (consistent with the existing "control signals only, not evidence" framing at `:2216-2225`):

```python
        "\n\n## Final coverage assessment (control context — NOT evidence)\n"
        "The evaluator's final assessment of coverage and gaps is below. Use it "
        "ONLY to decide which gaps to flag; do not cite it as evidence.\n"
        "{evaluation}\n"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd databricks-deep-research-app && uv run pytest tests/unit/agent_designer/test_workflow_builder_plan_execute.py -v`
Expected: PASS.

- [ ] **Step 7: Typecheck + lint + commit**

Run: `make typecheck-framework && cd databricks-deep-research-app && uv run ruff check src/deep_research/agent_designer/workflow_builder.py`
```bash
git add databricks-deep-research-app/src/deep_research/agent_designer/workflow_builder.py \
        databricks-deep-research-app/tests/unit/agent_designer/test_workflow_builder_plan_execute.py
git commit -m "fix(designer): remove dead body reflector; synthesizer reads evaluation

Body reflector emitted a control decision the runtime never reads (the loop is
driven by the evaluator's return value). Body is now the direct researcher;
synthesizer gains an explicit {evaluation} control-context slot.

Co-authored-by: Isaac"
```

> **ADR alternative (for final review):** the sibling `parallel_lanes` finalizer drops plan_and_execute artifacts and reads only from pools (`workflow_builder.py:2438-2441`). A stricter, more consistent Phase 0 would drop `evaluation` entirely. We implement "synthesizer reads evaluation" (the user's decision) with a real slot; the drop-alternative is recorded in the ADR.

### Task 0.2: Smoke — grounded report still produced

- [ ] **Step 1:** Run `make e2e-medium`. Expected: PASS — plan_and_execute run still produces a grounded report (evidence flows from pools via `pool_inject`).

---

## Phase 1 — Shared scope walk, runtime-key registry, read model, Pass A

### Task 1.0 (OPTIONAL — skipped): shared scope-helper extraction

Earlier drafts extracted scope helpers from `condition_contracts.py` so the dataflow walk could reuse them. **This is no longer required and is intentionally skipped:** Pass A is existence-only (`set[str]`), implements its own focused walk with verified field access (Task 1.3), and does **not** modify `condition_contracts.py`. Revisit ONLY if the two walks demonstrably drift on a real workflow (a scoping bug found in one but not the other) — at which point extract a single shared traversal as its own test-guarded PR. Net effect of this plan on `condition_contracts.py`: **unchanged.**

### Task 1.1: Framework-level runtime-injected key registry

**Files:**
- Create: `databricks-deep-research/src/databricks_deep_research/workflow/runtime_keys.py`
- Test: `databricks-deep-research/tests/unit/workflow/test_runtime_keys.py`
- (App `workflow_builder.py` is intentionally **NOT** modified — see Step 4: aliasing its `_RUNTIME_TEMPLATE_KEYS` would change lane-prompt coercion.)

- [ ] **Step 1: Write the failing test**

```python
from databricks_deep_research.workflow.runtime_keys import RUNTIME_INJECTED_KEYS


def test_registry_includes_known_runtime_keys():
    for k in ("all_observations", "plan_summary", "step_title", "source_quality",
              "reflector_feedback", "current_date", "tool_catalog",
              "claims", "verification_summary"):
        # NOTE: "current_step" is intentionally NOT here — it is PAE-scoped, not global.
        assert k in RUNTIME_INJECTED_KEYS
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError`.

- [ ] **Step 3: Create the registry** (seeded from the verified inventory: app `_RUNTIME_TEMPLATE_KEYS` `workflow_builder.py:28-80` + `state.py:_RUNTIME_DERIVED_KEYS` + runner bookkeeping. **Not** `current_step` — it is PAE-scoped and bound in `_resolve_pae`, not a global seed.)

```python
# src/databricks_deep_research/workflow/runtime_keys.py
"""Keys present in workflow STATE / template context at runtime WITHOUT any node
``output_key`` producing them. Maintained next to the runtime that injects them
so the dataflow checker's seed set cannot rot. Each group cites its producer.
"""
from __future__ import annotations

from databricks_deep_research.workflow.state import _RUNTIME_DERIVED_KEYS

# Harness-injected template variables (see agent_designer _RUNTIME_TEMPLATE_KEYS
# and harness._build_input). Auto-rendered into prompts; never a node output_key.
_HARNESS_TEMPLATE_KEYS = frozenset({
    "all_observations", "background", "completed_steps", "compute_namespace",
    "conversation_history", "current_date", "current_iso_datetime", "current_timezone",
    "fallback_discovery_sources", "file_context", "iteration", "max_steps", "max_words",
    "min_steps", "min_words", "observation", "page_contents", "plan_iterations",
    "plan_summary", "previous_observations", "reflector_feedback", "remaining_steps",
    "replan_budget", "revision_block_md", "research_depth", "search_results",
    "source_quality", "source_topics", "sources_count", "sources_list",
    "step_description", "step_prompt_guidance", "step_title", "step_type",
    "steps_completed", "steps_executed", "total_steps", "tool_catalog",
})
# plan_execute_runner bookkeeping appends (runner :209-212, :419-428).
_RUNNER_BOOKKEEPING_KEYS = frozenset({
    "observed_tool_kinds", "missing_required_tool_kind_groups", "last_blocked_step",
})
# Root state key. NOTE: the per-iteration item_state_key (current_step) is NOT a
# global runtime key — plan_and_execute injects it into the loop-body scope ONLY
# (plan_execute_runner.py:378). It is bound inside _resolve_pae, not seeded here,
# so a `current_step` read OUTSIDE a plan_and_execute is correctly flagged dangling.
# (The checker additionally seeds definition.required_inputs at the root — see
# dangling_reads — mirroring condition_contracts.py:142-152.)
_ROOT_KEYS = frozenset({"query"})

RUNTIME_INJECTED_KEYS: frozenset[str] = (
    _ROOT_KEYS | _RUNTIME_DERIVED_KEYS | _HARNESS_TEMPLATE_KEYS | _RUNNER_BOOKKEEPING_KEYS
)
```

- [ ] **Step 4: Do NOT modify the app in this task (decoupling guard).** The app's `_RUNTIME_TEMPLATE_KEYS` has **two** consumers: semantic validation (`workflow_builder.py:2825-2835`, where a superset is safe/additive) **and** the lane-prompt coercion allow-set `_DESIGNER_LANE_TEMPLATE_ALLOWED_VARS` (`workflow_builder.py:84-91`), which rewrites any template var NOT in the set to `{query}` via `_coerce_unknown_template_variables_to_query`. Aliasing `_RUNTIME_TEMPLATE_KEYS` to the framework **superset** (which adds `claims`, `verification_summary`, `analysis_summary`, `verification_details`, `observed_tool_kinds`, `missing_required_tool_kind_groups`, `last_blocked_step`) would silently STOP coercing those words — e.g. a designer lane prompt with `{claims}` would resolve to the citation pipeline's claims list instead of the user's query. That is an un-analyzed behavior change. **Therefore: leave the app's `_RUNTIME_TEMPLATE_KEYS` exactly as-is.** The framework `RUNTIME_INJECTED_KEYS` is seeded from the same inventory but maintained independently for now; full DRY reconciliation (deriving the app set from the framework registry, guarded by a coercion-behavior test) is a follow-up (see ADR follow-ups).

- [ ] **Step 5: Run the framework registry test + commit**

Run: `cd databricks-deep-research && uv run pytest tests/unit/workflow/test_runtime_keys.py -v`
Expected: PASS. (App is untouched, so no app test run needed here.)
```bash
git add databricks-deep-research/src/databricks_deep_research/workflow/runtime_keys.py \
        databricks-deep-research/tests/unit/workflow/test_runtime_keys.py
git commit -m "feat(workflow): framework runtime-injected key registry

Seeded from the same inventory as the app's _RUNTIME_TEMPLATE_KEYS but kept
separate: aliasing the app set to this superset would change lane-prompt
coercion behavior. DRY reconciliation is a guarded follow-up.

Co-authored-by: Isaac"
```

### Task 1.2: Effective-read extraction

**Files:**
- Create: `databricks-deep-research/src/databricks_deep_research/workflow/dataflow_contracts.py`
- Test: `databricks-deep-research/tests/unit/workflow/test_dataflow_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
from databricks_deep_research.workflow.dataflow_contracts import effective_reads
from databricks_deep_research.agents.config import AgentNodeConfig


def test_effective_reads_union_template_vars_input_keys():
    cfg = AgentNodeConfig(
        subtype="researcher", output_key="findings",
        input_keys=["query", "current_step"],
        user_prompt_template="Investigate {query} for {focus_area}.",
        system_prompt="",
    )
    reads = effective_reads(cfg)
    assert {"query", "current_step", "focus_area"} <= reads     # union of input_keys + template vars


def test_effective_reads_excludes_runtime_injected():
    cfg = AgentNodeConfig(subtype="reflector", output_key="reflection",
                          input_keys=["query"], system_prompt="{plan_summary} {all_observations}",
                          user_prompt_template="")
    # plan_summary/all_observations are runtime-injected → not dangling reads
    reads = effective_reads(cfg, exclude_runtime=True)
    assert "plan_summary" not in reads and "all_observations" not in reads
    assert "query" in reads


def test_effective_reads_excludes_loop_local_variable():
    # {%for s in sources_list%}{s}{%endfor%}: the iterable 'sources_list' is a read,
    # but the loop var 's' is a LOCAL binding, not a state read — must not be flagged
    # dangling. (extract_variables returns both; _template_reads subtracts the loop var.)
    cfg = AgentNodeConfig(subtype="synthesizer", output_key="report", input_keys=["query"],
                          system_prompt="",
                          user_prompt_template="{%for s in sources_list%}{s}{%endfor%}")
    reads = effective_reads(cfg)
    assert "sources_list" in reads
    assert "s" not in reads
```

- [ ] **Step 2: Run to verify it fails** — `effective_reads` undefined.

- [ ] **Step 3: Implement**

```python
# dataflow_contracts.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from databricks_deep_research.agents.config import (
    AgentNodeConfig, ConditionalNodeConfig, LoopNodeConfig,
    PlanAndExecuteNodeConfig, SubworkflowNodeConfig, ToolNodeConfig,
)
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.conditions import (
    CompositeCondition, LLMCondition, StateCondition,
)
from databricks_deep_research.workflow.definition import (
    NodeType, WorkflowDefinition, WorkflowNode,
)
from databricks_deep_research.workflow.runtime_keys import RUNTIME_INJECTED_KEYS

Channel = Literal["state", "pool"]
EdgeKind = Literal["data", "control"]
_renderer = SafeTemplateRenderer()


# Must match SafeTemplateRenderer's _FOR_OPEN syntax (renderer.py): group 1 = loop var.
_FOR_LOOP_VAR = re.compile(r"\{%\s*for\s+(\w+)\s+in\s+\w+\s*%\}")


def _template_reads(template: str) -> set[str]:
    """STATE keys a template reads, EXCLUDING loop-local variables. ``extract_variables``
    returns BOTH the ``{%for x in items%}`` iterable AND the loop var ``x`` (because ``{x}``
    in the body is matched as a plain ``{var}`` — see tests/test_renderer.py:82-84). The
    loop var is a local binding, not a state read, so subtract it or it false-flags as
    dangling."""
    if not template:
        return set()
    return _renderer.extract_variables(template) - set(_FOR_LOOP_VAR.findall(template))


def effective_reads(cfg: AgentNodeConfig, *, exclude_runtime: bool = False) -> set[str]:
    """STATE keys an agent actually consumes: declared input_keys ∪ template
    variables (the authoritative read signal — config.py:259-262), minus loop-local
    variables. Optionally drop runtime-injected keys."""
    reads = set(cfg.input_keys)
    reads |= _template_reads(cfg.system_prompt or "")
    reads |= _template_reads(cfg.user_prompt_template or "")
    if exclude_runtime:
        reads -= RUNTIME_INJECTED_KEYS
    return reads
```

- [ ] **Step 4: Run to verify it passes; commit**

Run: `cd databricks-deep-research && uv run pytest tests/unit/workflow/test_dataflow_contracts.py -k effective_reads -v`
```bash
git add -A && git commit -m "feat(workflow): effective-read extraction (template vars + input_keys)

Co-authored-by: Isaac"
```

### Task 1.3: Pass A — dangling reads via a focused existence-only walk

**Files:** Modify `dataflow_contracts.py`; test same file.

- [ ] **Step 1: Write the failing tests** (build nodes the way `tests/unit/workflow/test_condition_contracts.py` does — read it first for the canonical `WorkflowNode` construction)

```python
from databricks_deep_research.workflow.dataflow_contracts import dangling_reads


def test_sequence_read_resolves_to_prior_sibling(seq, agent):
    defn = _defn(seq("root", [
        agent("coordinator", subtype="coordinator", input_keys=["query"], output_key="coordination"),
        agent("researcher", subtype="researcher", input_keys=["query", "coordination"], output_key="findings"),
    ]), outputs=["findings"])
    assert dangling_reads(defn) == []


def test_unproduced_key_is_dangling(seq, agent):
    defn = _defn(seq("root", [
        agent("researcher", subtype="researcher", input_keys=["query", "nope"], output_key="findings"),
    ]), outputs=["findings"])
    assert any("nope" in d for d in dangling_reads(defn))


def test_runtime_injected_key_is_not_dangling(seq, agent):
    defn = _defn(seq("root", [
        agent("reflector", subtype="reflector",
              system_prompt="{plan_summary}", input_keys=["query"], output_key="reflection"),
    ]), outputs=["reflection"])
    assert dangling_reads(defn) == []   # plan_summary is runtime-injected
```

- [ ] **Step 2: Run to verify they fail** — `dangling_reads` undefined.

- [ ] **Step 3: Implement Pass A as a focused existence-only walk (`set[str]` visibility)**

Pass A only needs *existence* ("does this read resolve to ANY producer reachable in scope?"), so visibility is a plain `set[str]` of keys — no `FieldSchema`/availability lattice (that is `condition_contracts`' concern; coupling to its private `_UNKNOWN_SCHEMA` is unnecessary). This walk deliberately **mirrors** `condition_contracts`' per-node scoping rules with the same verified field access and is tested against the same topology corpus (Task 1.5); it does not claim to reuse that traversal. Full traversal unification is a follow-up (ADR).

```python
def dangling_reads(definition: WorkflowDefinition) -> list[str]:
    """Pass A (existence-only): every effective read must resolve to a producer
    visible in lexical scope. Seeds the workflow's declared ``required_inputs``
    (NOT just 'query' — mirrors condition_contracts.py:142-152) ∪ RUNTIME_INJECTED_KEYS.
    ``current_step`` is NOT a global seed — it is bound only inside plan_and_execute
    (_resolve_pae). POOL reads are global (Task 2.2), not lexically scoped."""
    dangling: list[str] = []
    seed = set(definition.required_inputs) | set(RUNTIME_INJECTED_KEYS)
    _resolve(definition.root, seed, dangling)
    return dangling


def _resolve(node: WorkflowNode, visible: set[str], dangling: list[str]) -> set[str]:
    """Returns the set of STATE keys this node EXPORTS to its parent scope."""
    if node.type == NodeType.agent:
        cfg = AgentNodeConfig(**node.config)
        for k in effective_reads(cfg):
            if k not in visible:
                dangling.append(f"node '{node.id}': read '{k}' has no producer in scope")
        return {cfg.output_key} if cfg.output_key else set()
    if node.type == NodeType.tool:
        cfg_t = ToolNodeConfig(**node.config)
        for state_key in cfg_t.input_mapping.values():   # values are outer state keys (executor.py:1010-1013)
            if state_key not in visible:
                dangling.append(f"tool '{node.id}': input_mapping reads '{state_key}' with no producer")
        return {cfg_t.output_key} if cfg_t.output_key else set()
    if node.type == NodeType.sequence:
        local = set(visible); exported: set[str] = set()
        for child in node.children:
            ex = _resolve(child, local, dangling); local |= ex; exported |= ex
        return exported
    if node.type == NodeType.parallel:
        exported = set()
        for child in node.children:                      # siblings hidden from each other
            exported |= _resolve(child, set(visible), dangling)
        return exported
    if node.type == NodeType.conditional:
        cfg_c = ConditionalNodeConfig(**node.config)
        for branch in cfg_c.conditions:                  # real field: conditions (config.py:207)
            for k in _condition_keys(branch.condition):
                if k not in visible:
                    dangling.append(f"conditional '{node.id}': condition reads '{k}' with no producer")
        exported = set()
        for child in node.children:
            exported |= _resolve(child, set(visible), dangling)
        return exported
    if node.type == NodeType.loop:
        return _resolve_loop(node, visible, dangling)     # 2-pass fixpoint in Task 2.3
    if node.type == NodeType.plan_and_execute:
        return _resolve_pae(node, visible, dangling)
    if node.type == NodeType.subworkflow:
        cfg_s = SubworkflowNodeConfig(**node.config)
        for outer_key in cfg_s.input_mapping.values():   # values are OUTER state keys (condition_contracts.py:226-233)
            if outer_key not in visible:
                dangling.append(f"subworkflow '{node.id}': input_mapping reads '{outer_key}' with no producer")
        return {cfg_s.output_key} if cfg_s.output_key else set()
    return set()


def _resolve_pae(node: WorkflowNode, visible: set[str], dangling: list[str]) -> set[str]:
    """plan_and_execute scope: planner output + the per-iteration item_state_key
    (current_step) are visible to the body/evaluator ONLY (plan_execute_runner.py:378;
    mirrors condition_contracts.py:323-328), NOT globally. Exports planner+evaluator
    outputs to the outer scope (condition_contracts.py:344-354)."""
    cfg = PlanAndExecuteNodeConfig(**node.config)
    planner = AgentNodeConfig(**cfg.planner)
    for k in effective_reads(planner):
        if k not in visible:
            dangling.append(f"node '{node.id}.planner': read '{k}' has no producer in scope")
    inner = set(visible) | {planner.output_key, cfg.item_state_key}
    if cfg.body is not None:                              # body is already a WorkflowNode (config.py:243)
        inner |= _resolve(cfg.body, inner, dangling)
    if cfg.evaluator is not None:
        ev = AgentNodeConfig(**cfg.evaluator)
        for k in effective_reads(ev):
            if k not in inner:
                dangling.append(f"node '{node.id}.evaluator': read '{k}' has no producer in scope")
        inner |= {ev.output_key}
    exported = {planner.output_key}
    if cfg.evaluator is not None:
        exported |= {AgentNodeConfig(**cfg.evaluator).output_key}
    return exported
```

Add the two small helpers below. All field names are verified against `agents/config.py` / `workflow/definition.py` / `workflow/conditions.py`.

```python
def _condition_keys(cond: StateCondition | LLMCondition | CompositeCondition) -> set[str]:
    """State keys a condition reads (mirrors condition_contracts._validate_condition)."""
    if isinstance(cond, StateCondition):
        return {cond.key}
    if isinstance(cond, LLMCondition):
        return set(_renderer.extract_variables(cond.prompt_template))
    if isinstance(cond, CompositeCondition):
        keys: set[str] = set()
        for c in cond.conditions:
            keys |= _condition_keys(c)
        return keys
    return set()


def _resolve_loop(node: WorkflowNode, visible: set[str], dangling: list[str]) -> set[str]:
    """Single forward pass now; Task 2.3 upgrades this to a 2-pass loop-carry fixpoint."""
    local = set(visible); exported: set[str] = set()
    for child in node.children:
        ex = _resolve(child, local, dangling); local |= ex; exported |= ex
    for k in _condition_keys(LoopNodeConfig(**node.config).until):   # 'until' reads as control
        if k not in local:
            dangling.append(f"loop '{node.id}': until-condition reads '{k}' with no producer")
    return exported
```

- [ ] **Step 4: Run to verify they pass; typecheck**

Run: `cd databricks-deep-research && uv run pytest tests/unit/workflow/test_dataflow_contracts.py -k "dangling or resolve" -v && make typecheck-framework`

- [ ] **Step 5: Commit** — `feat(workflow): Pass A dangling-read detection (shared scope walk)`

### Task 1.4: Public entry + wire into validation.py (lint mode, intrinsic severity)

**Files:** Modify `dataflow_contracts.py`, `validation.py`; test `tests/unit/workflow/test_validation_dataflow_integration.py`.

- [ ] **Step 1: Write the failing tests**

```python
from databricks_deep_research.workflow.dataflow_contracts import validate_dataflow_contracts, Diagnostic


def test_lint_mode_warnings_not_errors():
    rep = validate_dataflow_contracts(_dangling_workflow(), strict=False)
    assert rep.errors == [] and any("nope" in w for w in rep.warnings)


def test_strict_promotes_error_severity_only():
    # a dangling read is error-severity; a dead data store is warning-severity
    rep = validate_dataflow_contracts(_dangling_workflow(), strict=True)
    assert any("nope" in e for e in rep.errors)
```

```python
# test_validation_dataflow_integration.py
import logging
from databricks_deep_research.workflow.validation import validate_workflow


def test_lint_logs_not_raises(caplog, monkeypatch):
    monkeypatch.setenv("DATAFLOW_CHECK_STRICT", "false")
    with caplog.at_level(logging.WARNING):
        validate_workflow(_dangling_workflow())     # must not raise
    assert any("DATAFLOW_LINT" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement the entry with intrinsic severity**

```python
@dataclass(frozen=True)
class Diagnostic:
    message: str
    severity: Literal["error", "warning"]


@dataclass
class DataflowReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_dataflow_contracts(definition: WorkflowDefinition, *, strict: bool) -> DataflowReport:
    diags: list[Diagnostic] = []
    diags += [Diagnostic(m, "error") for m in dangling_reads(definition)]   # dangling reads = error-severity
    diags += detect_dead_stores(definition)                                  # mixed severities (Task 2.2)
    report = DataflowReport()
    for d in diags:
        if strict and d.severity == "error":
            report.errors.append(d.message)
        else:
            report.warnings.append(d.message)
    return report
```

- [ ] **Step 4: Wire into validation.py** (after `:242`, inside `if not errors:`)

```python
        from databricks_deep_research.workflow.dataflow_contracts import validate_dataflow_contracts
        report = validate_dataflow_contracts(definition, strict=_dataflow_strict_enabled())
        for w in report.warnings:
            logger.warning("DATAFLOW_LINT %s", w)
        errors.extend(report.errors)
```

Add near top of `validation.py`:
```python
import os
def _dataflow_strict_enabled() -> bool:
    return os.getenv("DATAFLOW_CHECK_STRICT", "false").lower() in {"1", "true", "yes"}
```

(`detect_dead_stores` returns `[]` until Task 2.2 — define a stub returning `[]` now so this compiles.)

- [ ] **Step 5: Run tests; typecheck; commit** — `feat(workflow): wire dataflow Pass A into validation (lint)`

### Task 1.5: Corpus false-positive gate (loader-based, parametrized domains)

**Files:** Create `databricks-deep-research/tests/unit/workflow/test_dataflow_corpus.py` (or app-side if cross-package import is cleaner — match existing patterns).

- [ ] **Step 1: Write the corpus test using the REAL loader**

```python
import pytest
from databricks_deep_research.workflow.loader import load_workflow_from_dict
from databricks_deep_research.workflow.dataflow_contracts import validate_dataflow_contracts

TOPOLOGIES = ["single_agent", "parallel_lanes", "plan_and_execute"]   # framework-defined arrangements


@pytest.fixture(params=TOPOLOGIES)
def topology(request): return request.param


@pytest.mark.parametrize("domain", _sample_domains())   # derive from brief schema, not a hardcoded list
def test_generated_workflow_has_no_dataflow_warnings(topology, domain):
    wf_dict = _build_designer_workflow(domain=domain, topology=topology)
    defn = load_workflow_from_dict(wf_dict)          # canonical validation path
    report = validate_dataflow_contracts(defn, strict=False)
    assert report.warnings == [], f"{topology}/{domain}: {report.warnings}"
```

`_sample_domains()` returns a small neutral set parametrized from the brief schema / a generic fixture — NOT a hardcoded `finance/medicine/law` list (constitution).

- [ ] **Step 2: Run; iterate to zero warnings**

Run: `cd databricks-deep-research && uv run pytest tests/unit/workflow/test_dataflow_corpus.py -v`
For each warning: real dangle (fix builder) or missed runtime key (add to `runtime_keys.py` WITH a citation to the producing context-builder)? Iterate to zero. Never broaden seeds blindly.

- [ ] **Step 3: Commit** — `test(workflow): dataflow corpus false-positive gate (loader-based)`

---

## Phase 2 — Control edges, Pass B, fixpoint, adversarial measurement

### Task 2.1: Control-edge derivation

**Files:** Modify `dataflow_contracts.py`; test same file.

- [ ] **Step 1: Write failing tests** — PAE evaluator output is a control edge; loop `until` keys are control edges; conditional branch keys are control edges.

```python
from databricks_deep_research.workflow.dataflow_contracts import control_consumed_keys


def test_pae_evaluator_output_is_control(plan_execute_defn):
    assert "evaluation" in control_consumed_keys(plan_execute_defn)

def test_loop_until_keys_are_control(loop_until_done_defn):
    assert "done" in control_consumed_keys(loop_until_done_defn)
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** (uses verified field names: `cfg.conditions`, `cfg_l.until`, PAE `evaluator` dict)

```python
def control_consumed_keys(definition: WorkflowDefinition) -> set[str]:
    """Keys consumed as CONTROL: PAE evaluator output_key (loop branches on the
    returned decision, plan_execute_runner.py:475); loop 'until' keys; conditional
    branch keys."""
    keys: set[str] = set()
    _walk_control(definition.root, keys)
    return keys


def _walk_control(node: WorkflowNode, keys: set[str]) -> None:
    if node.type == NodeType.plan_and_execute:
        cfg = PlanAndExecuteNodeConfig(**node.config)
        if cfg.evaluator is not None:
            keys.add(AgentNodeConfig(**cfg.evaluator).output_key)
        if cfg.body is not None:
            _walk_control(cfg.body, keys)            # body is already a WorkflowNode
        return
    if node.type == NodeType.loop:
        keys |= set(_condition_keys(LoopNodeConfig(**node.config).until))
    if node.type == NodeType.conditional:
        for branch in ConditionalNodeConfig(**node.config).conditions:
            keys |= set(_condition_keys(branch.condition))
    for child in node.children:
        _walk_control(child, keys)
```

- [ ] **Step 4: Run; commit** — `feat(workflow): control-edge derivation`

### Task 2.2: Pass B — dead stores, severity-tiered

**Files:** Modify `dataflow_contracts.py`; test same file.

- [ ] **Step 1: Write failing tests**

```python
from databricks_deep_research.workflow.dataflow_contracts import detect_dead_stores


def test_unread_data_output_is_warning(unread_findings_defn):
    diags = detect_dead_stores(unread_findings_defn)
    assert any("findings" in d.message and d.severity == "warning" for d in diags)

def test_terminal_output_exempt(terminal_report_defn):
    assert not any("report" in d.message for d in detect_dead_stores(terminal_report_defn))

def test_pool_roundtrip_not_dead(pool_roundtrip_defn):
    assert not any("observations" in d.message for d in detect_dead_stores(pool_roundtrip_defn))

def test_evaluation_consumed_as_control_not_dead(plan_execute_defn):
    assert not any("evaluation" in d.message for d in detect_dead_stores(plan_execute_defn))
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** — collect all producers/reads (STATE output_keys + POOL pool_writes; reads = effective_reads + pool_inject + control + tool input_mapping), then flag zero-edge producers.

```python
@dataclass(frozen=True)
class Producer:
    key: str
    channel: Literal["state", "pool"]
    node_id: str


def _collect_producers_reads(
    definition: WorkflowDefinition,
) -> tuple[list[Producer], set[str], set[str]]:
    """One walk gathering producers + reads across STATE and POOL (used only by
    Pass B — reads here are flat key-sets, not Read objects). STATE producers =
    agent/tool/PAE-slot output_key; POOL producers = pool_writes pools. STATE reads
    = effective_reads ∪ tool input_mapping values ∪ condition keys; POOL reads =
    pool_inject pools. All field names verified (config.py)."""
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
            state_reads.update(cfg_t.input_mapping.values())
        elif node.type == NodeType.plan_and_execute:
            cfg_p = PlanAndExecuteNodeConfig(**node.config)
            for slot in (cfg_p.planner, cfg_p.evaluator):
                if slot is not None:
                    ac = AgentNodeConfig(**slot)
                    producers.append(Producer(ac.output_key, "state", node.id))
                    state_reads.update(effective_reads(ac))
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
    """A producer is dead iff no read (any channel) consumes its key. Dead data/pool
    producers are WARNINGS; the error tier is the dangling *control read* surfaced by
    Pass A (a loop/conditional/PAE control source with no producer). Terminal outputs
    and pools injected later are exempt. Note ``current_step`` is a PAE-injected
    producer here too, so the body's read of it is consumed."""
    producers, state_reads, pool_reads = _collect_producers_reads(definition)
    consumed_state = set(state_reads) | control_consumed_keys(definition)
    consumed_pool = set(pool_reads)
    terminal = set(definition.output_keys)
    diags: list[Diagnostic] = []
    for p in producers:
        if p.channel == "pool":
            if p.key not in consumed_pool:
                diags.append(Diagnostic(f"pool '{p.key}' (by '{p.node_id}') is never injected", "warning"))
        else:
            if p.key in terminal or p.key in consumed_state:
                continue
            diags.append(Diagnostic(f"state '{p.key}' (by '{p.node_id}') is read by nobody", "warning"))
    return diags
```

- [ ] **Step 4: Run; commit** — `feat(workflow): Pass B dead-store detection`

### Task 2.3: Loop-carry 2-pass fixpoint

**Files:** Modify `dataflow_contracts.py` (`_resolve_loop`, PAE body); test same file.

- [ ] **Step 1: Write failing test** — a loop body read of a key produced later in the same body must not be dangling.

- [ ] **Step 2: Run to verify fail** (single forward pass flags it).

- [ ] **Step 3: Implement the 2-pass fixpoint** — pass 1 resolves the body into a throwaway sink to collect all body-produced keys; pass 2 resolves reads against `scope ∪ body_produced`. Apply to both `_resolve_loop` and the body inside `_resolve_pae`.

- [ ] **Step 4: Run; full suite; typecheck; commit** — `feat(workflow): loop-carry 2-pass fixpoint`

### Task 2.4: Adversarial broken-workflow tests (false-negative measurement)

**Files:** Create `databricks-deep-research/tests/unit/workflow/test_dataflow_adversarial.py`.

- [ ] **Step 1: Build one broken workflow per defect class and assert detection**

```python
import pytest
from databricks_deep_research.workflow.dataflow_contracts import validate_dataflow_contracts

def test_injected_dangling_data_read_is_flagged(seq, agent):
    defn = _defn(seq("root", [agent("r", subtype="researcher",
                  input_keys=["query", "ghost"], output_key="findings")]), outputs=["findings"])
    assert any("ghost" in e for e in validate_dataflow_contracts(defn, strict=True).errors)

def test_injected_dead_store_is_flagged(...):
    # producer 'orphan' read by nobody, non-terminal -> warning
    ...

def test_injected_dangling_control_read_is_flagged(...):
    # loop until references a key no node produces -> error
    ...

def test_injected_dangling_tool_input_is_flagged(...):
    # tool input_mapping references an unproduced state key -> error
    ...
```

These four (dangling data, dead store, dangling control, dangling tool input) are the false-negative coverage the Phase 5 gate requires.

- [ ] **Step 2: Run; all must detect; commit** — `test(workflow): adversarial false-negative coverage`

### Task 2.5: Corpus gate covers Pass B; record the measurement

- [ ] **Step 1:** Re-run the corpus gate (now with Pass B + control edges active): `uv run pytest tests/unit/workflow/test_dataflow_corpus.py -v`. Expected: **zero** warnings across all topologies/domains (post-Phase-0, the plan_and_execute evaluator's `evaluation` is consumed as a control edge and the coverage-reflector's `coverage_review` is consumed as data by the finalizer — neither is flagged).
- [ ] **Step 2:** Write `docs/superpowers/plans/2026-05-29-dataflow-MEASUREMENT.md`: topologies × domains, false-positive count (must be 0), adversarial defect classes detected (must be all), date. This is the evidence Phase 5's gate consumes.
- [ ] **Step 3: Commit** — `test(workflow): corpus+adversarial measurement for strict-flip gate`

### Task 2.6: Strict-flip decision (gated, not automatic)

- [ ] **Step 1:** Only after Tasks 2.4–2.5 are green: change `_dataflow_strict_enabled()` default to `True` (or set `DATAFLOW_CHECK_STRICT=true` in the deploy env). Re-run the full suite + `make e2e-medium`. If anything breaks, revert to lint and fix the seed/builder — do not weaken the checker. Commit separately so it's easy to revert.

---

## What the checker does NOT catch (honesty boundary)

- **Redundant-but-connected nodes.** A reflector whose output is read as *data* by a downstream node (e.g. the body reflector → synthesizer, or the legitimate `coverage-reflector` → finalizer) is **not** a dataflow-correctness violation — it has a live data edge. The framework cannot structurally distinguish "useful critique" from "wasted reflection"; that is a semantic judgment. Phase 0 removes the *specific* body reflector by that judgment, not via the checker. (This is why the v1 subtype-based "redundant decision-maker = error" check was dropped: it would false-positive the `coverage-reflector`.)
- **Semantic correctness** of prompts/outputs, value types beyond existence, or whether a produced value is *good*.
- **Runtime-only dataflow** that bypasses STATE/POOL/conditions (anything injected purely via `runtime_context` that the static AST can't see — mitigated by the registry, but the registry is a maintained approximation).

---

## Future Phases (deferred — entry gates only)

- **Phase 3 — Visualization:** entry gate = Pass A+B strict & stable. Human DAG (STATE solid / POOL dashed / CONTROL bold; dead/dangling flagged) in the designer UI + a text renderer fed to a repair LLM. Both derive from the Task 2.2 ledger.
- **Phase 4 — `plan_and_execute` macro spike:** entry gate = control edges first-class (Task 2.1). Timeboxed (≤1 day) doc; no code commitment.
- **Phase 5 — Structured-freedom synthesis:** **hard gate** = Task 2.4/2.5 show zero false negatives AND zero corpus false positives. Frontier model synthesizes free-form typed-node ASTs; every output must pass the strict checker; failures route to the Phase 3 repair LLM. Own spec→plan cycle.

---

## ADR

- **Decision:** Build a build-time dataflow checker (`dataflow_contracts.py`) as a focused existence-only walk that mirrors `condition_contracts.py`'s scoping rules (leaving that validator unchanged), drives reads off prompt-template variables (minus loop-locals), models three channels (STATE/POOL/RUNTIME-RETURN), ships lint→strict, and fixes the body reflector (Phase 0) by deleting it and giving the synthesizer a real `{evaluation}` slot.
- **Drivers:** dataflow correctness on generated workflows; zero false positives on current output; an earned path to free-form synthesis (Phase 5).
- **Alternatives considered:** (A-naive, v1) parallel walk with unverified field names — rejected (compile-time field-name bugs). (B) shared-helper refactor of `condition_contracts` — rejected (unnecessary for an existence check; risks dropping its availability lattice → drift). (C) full lexical scoping + synthesis in one plan — rejected (synthesis gated on a not-yet-existing measurement). **Chosen: A-refined** — a parallel walk with every field name verified and `set[str]` existence semantics, leaving `condition_contracts.py` unchanged. **Phase 0 sub-decision:** add a real `{evaluation}` slot vs. drop reflection/evaluation entirely (parallel_lanes style) — chose the slot to honor "synthesizer reads evaluation"; drop-alternative recorded for the user's call.
- **Why chosen (A-refined):** verifying every field name against source eliminates the v1 bug class without touching the working condition validator; Pass A is existence-only so the availability lattice is irrelevant; a deliberate second walk tested against the same corpus is simpler and lower-risk than a shared-helper refactor. Full traversal unification is a follow-up, not a precondition.
- **Consequences:** `condition_contracts.py` is unchanged (Task 1.0 skipped); a new framework runtime-key registry maintained separately from the app's `_RUNTIME_TEMPLATE_KEYS` (the app is NOT modified — aliasing would change lane coercion); the checker is a *maintained approximation* of runtime reads (registry can drift) — mitigated by lint-first + corpus gate.
- **Follow-ups:** Phases 3–5; consolidate the two traversals fully if they drift; derive the runtime-key registry programmatically from context builders rather than a curated list.

---

## Definition of Done (Phases 0–2)

- [ ] Phase 0: body reflector gone; synthesizer has a real `{evaluation}` slot; builder tests pass; `make e2e-medium` green.
- [ ] `condition_contracts.py` unchanged (Task 1.0 skipped; Pass A is a self-contained `set[str]` walk).
- [ ] Framework `RUNTIME_INJECTED_KEYS` registry created; app `_RUNTIME_TEMPLATE_KEYS` left unchanged (coercion-safe; DRY reconciliation is a follow-up).
- [ ] `dataflow_contracts.py`: effective-read extraction (template vars minus loop-locals); Pass A (dangling data+control+tool reads, `set[str]` existence walk seeding `required_inputs`); Pass B (dead stores, severity-tiered, terminal+pool exemptions); loop-carry fixpoint.
- [ ] Wired into `validation.py` behind `DATAFLOW_CHECK_STRICT` (default lint); intrinsic diagnostic severity.
- [ ] Corpus gate: zero warnings across 3 topologies × parametrized domains (loader-based).
- [ ] Adversarial gate: all four defect classes detected (the false-negative measurement).
- [ ] `make typecheck-framework` + `uv run ruff check` clean; measurement doc written.
