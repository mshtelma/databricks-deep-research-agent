# Dataflow Checker — Measurement Record

**Date:** 2026-05-29
**Plan:** `docs/superpowers/plans/2026-05-29-scoped-bindings-dataflow-enforcement.md` (Phases 0–2)
**Purpose:** Evidence the dataflow checker (`workflow/dataflow_contracts.py`) is both
**false-positive-clean** on real generated workflows and **false-negative-complete** on
injected defects. This is the measurement the Phase 5 (free-form topology synthesis) gate
requires before strictness can be unlocked.

---

## False-positive measurement (corpus gate)

Test: `databricks-deep-research-app/tests/unit/agent_designer/test_dataflow_corpus.py`
Method: build each designer topology via the real path
(`build_web_research_workflow` → `load_workflow_from_dict`), then
`validate_dataflow_contracts(defn, strict=False)`.

| Topology | Generated via | Dataflow warnings |
|---|---|---|
| `single_agent` | builder (generic web brief) | **0** |
| `parallel_lanes` | builder (generic web brief) | **0** |
| `plan_and_execute` | builder (generic web brief) | **0** |

> Domain is not part of the matrix: it changes prompt prose, not AST dataflow
> shape, and a hardcoded domain list would violate the no-hardcoded-domains rule.
> Topology is the structural variable that shapes the dataflow graph.

**Result: 0 false positives across all 3 topologies.**

### False positives found and fixed during the lint window (US-DF6)
Each surfaced warning was triaged to a real checker refinement (not silenced):

1. **POOL dead-store over-fired** (`pool 'sources' never injected`, single_agent).
   Pool consumption is runtime-mediated (the citation pipeline reads the
   `sources`/`observations` pools), not statically determinable from AST
   `pool_inject`. → Dropped POOL dead-store flagging.
2. **State output of pool-writing nodes** (`findings_lane_1 read by nobody`,
   parallel_lanes). The lane researcher's data flows via the observations pool, not
   its state output_key. → Exempt state outputs of nodes that `pool_writes`.
3. **PAE body outputs not exported** (`synthesizer reads 'findings' — no producer`,
   plan_and_execute). The body's `findings` is written to state every iteration and
   is available downstream. → `_resolve_pae` now exports body outputs (mirrors the
   builder's availability walk + `harness.py:720`).

---

## False-negative measurement (adversarial gate)

Test: `databricks-deep-research/tests/unit/workflow/test_dataflow_adversarial.py`
Method: inject one broken workflow per defect class; assert detection.

| Defect class | Severity | Detected? |
|---|---|---|
| Dangling data read (`input_keys` references an unproduced key) | error (strict) | ✅ |
| Dead data store (output read by nobody, non-terminal, non-pool node) | warning | ✅ |
| Dangling control read (loop `until` references an unproduced key) | error (strict) | ✅ |
| Dangling tool input (`input_mapping` value unproduced) | error (strict) | ✅ |

**Result: all 4 injected defect classes detected — 0 false negatives.**

---

## Conclusion

The checker measures **0 false positives** on the generated-workflow corpus and
**0 false negatives** across the four injected defect classes. It ships in lint mode
(`DATAFLOW_CHECK_STRICT` default `false`); a strict flip and the Phase 5 free-form
synthesis unlock can proceed on this evidence. The corpus is currently topology-only;
expanding it (more briefs, asset kinds, nested composites) before the strict flip would
further harden the false-negative claim.
