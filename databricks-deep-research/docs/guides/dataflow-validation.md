# Scoped Bindings & Dataflow Validation

> Catch "read from nowhere" and "nothing wasted" dataflow bugs at load time, before any agent runs.

## Overview

A workflow passes data between nodes through named **state keys**: a node writes
its result to an `output_key`, and a later node reads it back through a prompt
template variable, an `input_keys` entry, a tool `input_mapping`, or a condition.
Nothing in the YAML forces those reads and writes to line up. An author can:

- **Read a key nobody writes** — a *dangling read*. The template variable renders
  as an empty string (missing variables are silently blank, see
  [YAML Workflow Authoring](yaml-workflow-authoring.md)), so the agent quietly
  works with no data instead of failing loudly.
- **Write a key nobody reads** — a *dead store*. The producing node still costs an
  LLM call, but its output goes nowhere — wasted compute, usually a wiring bug.

Dataflow validation walks the workflow tree at load time and flags both. It runs
automatically inside `validate_workflow` (`workflow/validation.py`), after
structural and condition-contract checks pass. The implementation lives in
`workflow/dataflow_contracts.py`; it complements `condition_contracts.py` (which
checks condition *type* correctness) by checking dataflow *reachability*.

## Concepts

### State keys & scoped bindings

Each node declares what it produces and consumes through its config:

- **Produces** — an `agent`/`tool` node's `output_key`; a `plan_and_execute`
  planner/evaluator `output_key`; a `pool_writes` entry (POOL channel).
- **Consumes** — what `effective_reads` computes for an agent (below), a tool's
  `input_mapping` *values*, a `subworkflow`'s `input_mapping` *values*, condition
  keys, and `pool_inject` pools.

Reads resolve **lexically**: a node sees the keys produced by earlier nodes in its
own block plus those inherited from enclosing blocks — not the flat union of
everything produced anywhere. `parallel` siblings are hidden from each other, and
`conditional` branches likewise (each sees only the inherited scope). This block
scoping makes "read from nowhere" a local, decidable question. Scoping changes only
*name resolution*, never storage (state is append-only, readers get serialized
copies), so tightening resolution cannot introduce data races.

### `effective_reads` and `dataflow_contracts`

`dataflow_contracts` is the validation **module**, not a YAML field — you do not
author "contracts" by hand. The per-node read contract is computed by
`effective_reads(cfg)`:

```
effective_reads = input_keys
                ∪ template variables in system_prompt
                ∪ template variables in user_prompt_template
                − loop-local variables (e.g. {% for x in items %})
```

The authoritative read signal is the **prompt templates**: at runtime the harness
auto-detects inputs from template variables via
`SafeTemplateRenderer.extract_variables()`, so `input_keys` is documentation-only.
If your template reads `{{ background }}`, that counts as a read whether or not
`background` is listed in `input_keys`.

### Runtime-injected keys

Some keys appear in state or prompt context at runtime **without** any node's
`output_key` producing them — the harness injects template variables, the
plan-and-execute runner appends bookkeeping, and the runtime store derives others.
The registry of these lives in `workflow/runtime_keys.py` as `RUNTIME_INJECTED_KEYS`
(the union of `_HARNESS_TEMPLATE_KEYS`, `_RUNNER_BOOKKEEPING_KEYS`, `_ROOT_KEYS`,
and `state.py`'s `_RUNTIME_DERIVED_KEYS`). Examples: `query`, `current_date`,
`conversation_history`, `sources_list`, `observation`, `completed_steps`,
`max_steps`, and the citation pipeline's `claims` / `verification_summary`.

Pass A **seeds** these as already-available, so a read of one is never flagged as
dangling. (The per-iteration `current_step` — a `plan_and_execute`
`item_state_key` — is *not* a global runtime key: it is bound only inside the
loop-body scope.)

If your workflow injects extra keys at runtime (e.g. a heavily
runtime-context-driven workflow such as the agent designer), declare them in the
top-level `runtime_injected_keys` field on the `WorkflowDefinition`. The checker
treats them as both available (a read is not dangling) **and** consumed (a producer
is not a dead store).

## What the validator checks

### Pass A — dangling reads (`dangling_reads`)

Every effective read must resolve to a producer visible in lexical scope. The walk
seeds the root scope with `definition.required_inputs` ∪ the runtime-injected
registry, then descends, exporting each node's `output_key` to its parent scope. A
read with no producer in scope is reported, e.g.:

```
node 'synthesizer': read 'reflection' has no producer in scope
```

Pass A covers agent reads, tool/subworkflow `input_mapping` values, conditional
branch keys, and loop `until` keys. A dangling **control** read (a loop `until`, a
conditional branch key, or a `plan_and_execute` evaluator source with no producer)
is error-severity — the container literally cannot make its decision. POOL reads
are not checked here (pools are global, handled in Pass B).

### Pass B — dead stores, control edges, loop-carry (`detect_dead_stores`)

A produced state value consumed by **nobody** — across STATE reads *and* the
control channel — is a dead store (warning-severity), e.g.:

```
state 'enrichment' (produced by 'enrichment_node') is read by nobody
```

Pass B handles three subtleties so it does not false-flag legitimate workflows:

- **Control edges count as consumption.** A `plan_and_execute` evaluator's
  `output_key` is consumed via the RUNTIME-RETURN channel (the loop branches on the
  returned decision, never reading it from state). `control_consumed_keys` adds
  evaluator outputs, loop `until` keys, and conditional branch keys to the consumed
  set, so they are not mislabeled dead.
- **Exemptions.** Terminal workflow outputs (`definition.output_keys`), keys in
  `runtime_injected_keys`, and the `output_key` of any node that also writes to a
  pool are exempt. POOL producers are never flagged — pool consumption is
  runtime-mediated (e.g. the citation pipeline reads the sources/observations
  pools) and not statically determinable.
- **Loop-carry fixpoint.** A loop (or `plan_and_execute`) body may read a key a
  *later* iteration produces. Both `_resolve_loop` and `_resolve_pae` run two
  passes: pass 1 collects all body-produced keys into a throwaway sink, pass 2
  resolves the body's reads against `visible` ∪ that fixpoint — so a loop-carried
  read is not a false dangle.

### Gates

Two **measurement** gates govern the design (they are roll-out/test gates, not
runtime code paths):

- **Corpus false-positive gate** — run the checker over the corpus of
  real-world / generated workflows and confirm it flags *zero* of them
  spuriously before tightening to strict. This is why the checker ships
  lint-first.
- **Adversarial false-negative gate** — feed the checker workflows with
  deliberately injected defects (a dangling read, a dead store) and confirm it
  *catches* every one. A false negative — passing a broken topology — is treated
  as strictly worse than over-conservatism.

## Lint vs strict mode

`validate_workflow` always calls
`validate_dataflow_contracts(definition, strict=_dataflow_strict_enabled())`. The
mode is selected by the `DATAFLOW_CHECK_STRICT` environment variable (default
`false` → **lint**):

| Mode | When | Pass A dangling read (error severity) | Pass B dead store (warning severity) |
|------|------|----------------------------------------|--------------------------------------|
| **Lint** (default) | `DATAFLOW_CHECK_STRICT` unset / `false` | Logged as `DATAFLOW_LINT` warning; **does not block** | Logged as `DATAFLOW_LINT` warning |
| **Strict** | `DATAFLOW_CHECK_STRICT` in `{1,true,yes}` | Added to `WorkflowValidationError.errors` → **load fails** | Still logged as a warning only |

Severity is intrinsic to the diagnostic; the mode only governs whether
**error-severity** diagnostics block the load. Dead stores stay warnings in both
modes. To find lint warnings, watch the logger for `DATAFLOW_LINT` lines, or call
`validate_dataflow_contracts(definition, strict=True)` directly and inspect
`report.errors` / `report.warnings`.

## Authoring guidance — fixing a finding

**Dangling read** (`read 'K' has no producer in scope`): the reader sees no
producer for `K`. Either —

1. **Add the producer.** Ensure an upstream node in the same (or an enclosing)
   block writes `output_key: K` before the reader runs. Remember scope rules:
   a `parallel`/`conditional` sibling cannot supply it.
2. **Fix the name.** Most often the read is a typo or a stale key — correct the
   template variable / `input_keys` entry to match the real producer.
3. **Mark it runtime-injected.** If the framework supplies `K` at runtime, add it
   to the top-level `runtime_injected_keys` list (only for genuinely
   runtime-mediated keys — do not use this to silence a real wiring bug).

**Dead store** (`state 'K' ... is read by nobody`): the producer's output goes
nowhere. Either —

1. **Consume it** — have a downstream node read `K` (template/`input_keys`),
   route it through a pool (`pool_writes`), use it as a condition/loop control, or
   list it in the workflow's `output_keys` if it is a terminal result.
2. **Remove the producer** if the value is genuinely unnecessary — it is one fewer
   LLM call.

## YAML example

`runtime_injected_keys` is the one author-settable field this feature adds. It sits
at the **top level** of the workflow, alongside `required_inputs` / `output_keys`:

```yaml
id: designer_style_workflow
name: Runtime-Context Workflow
version: 1
required_inputs: [query]
output_keys: [report]

# Keys this workflow injects at runtime (not produced by any node output_key).
# The dataflow checker treats them as both available and consumed.
runtime_injected_keys:
  - discovery_context
  - tool_catalog_override

root:
  id: main
  type: sequence
  label: Main
  children:
    - id: researcher
      type: agent
      label: Researcher
      config:
        subtype: researcher
        output_key: findings
        # Reads {{ query }} (required input) + {{ discovery_context }}
        # (runtime-injected) — neither is dangling.
        user_prompt_template: "Research {{ query }} using {{ discovery_context }}"
    - id: synthesizer
      type: agent
      label: Synthesizer
      config:
        subtype: synthesizer
        output_key: report                 # terminal output -> not a dead store
        user_prompt_template: "Write up {{ findings }}"   # consumes 'findings'
```

Here `findings` is live (read by the synthesizer), `report` is exempt (a terminal
`output_keys` value), and `discovery_context` / `tool_catalog_override` resolve via
`runtime_injected_keys` instead of dangling.

Most authors never set `runtime_injected_keys` — the read/write contract is derived
automatically from existing `output_key`, `input_keys`, and prompt templates, so a
correctly wired workflow passes with no extra fields.

## See also

- [YAML Workflow Authoring](yaml-workflow-authoring.md) — write workflows from scratch
- [Conditions and Branching](conditions-and-branching.md) — control flow and condition contracts
- [Workflow Definition Schema](../reference/workflow-definition-schema.md) — complete top-level field reference
- [Node Types Reference](../reference/node-types-reference.md) — all 8 node types and their config schemas
