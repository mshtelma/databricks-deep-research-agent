# OfficeQA Workflow v8 Hybrid

`workflow-v8-hybrid.yaml` is a schema-driven benchmark workflow that combines:
- structured question decomposition,
- operand-level evidence grounding with risk metadata,
- **in-loop cross-source verification** (independent agent checks highest-risk operand each iteration),
- risk-aware reflexion that sees both findings AND verification results,
- single-pass computation with reverse checks, and
- lightweight arithmetic auditing.

It is designed to reduce failures where an answer is internally consistent with a chosen source, but the source itself is not the best authority for the claim. The key architectural innovation is putting cross-verification INSIDE the reflexion loop, enabling iterative correction when discrepancies are found.

## Architecture

```
question_analyzer
  → research_loop (max 3 iterations):
      researcher → cross_verifier → reflector
  → assembler
  → arithmetic_auditor
  → final_formatter
```

## Stage Overview

1. `question_analyzer`
   - Converts the question into a minimal `operation_schema`.
   - Does not suggest specific source files.
   - Encodes task structure, verification sensitivity, and hierarchical-table risk.

2. `research_loop` (3-agent reflexion loop, max 3 iterations)
   - `researcher`: grounds operands and records provenance/risk metadata.
   - `cross_verifier`: independently verifies the highest-risk operand through a different source path.
   - `reflector`: evaluates evidence sufficiency AND cross-verification results; drives correction.
   - Loop exits when reflector decides "complete" (no unresolved conflicts or high-risk operands).

3. `assembler`
   - Selects the best grounded operands (preferring independently confirmed ones).
   - Computes the final value once with reverse checks.

4. `arithmetic_auditor`
   - Lightweight (0 tool calls) reverse check on the assembled answer.
   - Catches formula-application errors (e.g., operand order swaps).

5. `final_formatter`
   - Emits the final benchmark tag only.

## Core Principle

The workflow addresses three distinct failure modes:
- **Did we read the value correctly?** (transcription) — researcher's column header verification
- **Did we choose the right source for that value?** (source selection) — cross-verifier's independent path
- **Would an independent path support the same claim?** (confirmation) — reflexion loop iterates until verified

The critical insight: verification must happen INSIDE the loop so discrepancies can trigger re-research, not just flag the answer as uncertain.

## Failure Patterns Addressed

| Pattern | Root Cause | How Addressed |
|---|---|---|
| Subcategory vs Total | Hierarchical table columns, agent picks sub-column | Cross-verifier finds parent total independently; researcher prompt includes HIERARCHICAL TABLE HEADERS section |
| Split-Layout Tables | Side-by-side tables lose structure in Markdown | Cross-verifier uses different file; researcher prompt includes SPLIT-LAYOUT TABLES section |
| Row/Period Selection | Wrong year, fiscal vs calendar confusion | Cross-verifier catches FY/CY mismatch; enhanced period verification in prompts |
| External Data | CPI, exchange rates not in source docs | Cannot fix architecturally; cross-verifier can confirm "not in any source" more reliably |

## Inter-Stage Contracts

All major intermediate outputs are JSON-only prompt contracts.

### `operation_schema`

Produced by `question_analyzer`.

Fields:
- `operation_type`
- `output_shape`
- `rounding`
- `operands[]`
- `formula`
- `constraints[]`
- `verification_priority`

### `findings`

Produced by `researcher`.

Each operand contains:
- `id`, `status`, `value`, `unit`, `metric_label`, `time_period`
- `source_ref` (`file_id`, `table_id`, `row_label`, `column_label`)
- `match_quality`, `source_role`, `source_temporality`
- `independent_confirmation`, `confidence`, `disambiguation_note`

`summary` contains:
- `grounded_count`, `missing_count`, `ambiguity_count`
- `distinct_source_count`, `high_risk_operands`, `search_trace`

### `cross_verification` (NEW)

Produced by `cross_verifier`.

Fields:
- `status`: `confirmed` | `conflict` | `no_independent_source` | `all_clear`
- `target_operand_id`
- `researcher_value`
- `independent_value`
- `discrepancy_pct`
- `independent_source_ref` (`file_id`, `table_id`, `row_label`, `column_label`)
- `independence_strength`: `strong` | `medium` | `weak` | `none`
- `analysis`

### `reflection`

Produced by `reflector`.

Fields:
- `decision`: `complete` | `continue` | `adjust`
- `reasoning`, `evidence_sufficiency`
- `high_risk_operands`, `risk_reasons`, `suggested_changes`

Allowed risk reasons:
- `single_source_risk`, `retrospective_source_risk`, `definition_drift_risk`
- `convenience_source_risk`, `header_uncertainty`
- `cross_source_disagreement` (cross-verifier found different value)
- `subcategory_confusion` (cross-verifier suggests parent vs child mismatch)

### `assembled_answer`

Produced by `assembler`.

Fields:
- `status`: `computed` | `computed_with_risk` | `data_not_available`
- `final_value`, `rounding_applied`, `selected_operands[]`, `residual_risks`, `notes`

### `audited_answer` (NEW)

Produced by `arithmetic_auditor`.

Fields:
- `status`: `passed` | `flagged` | `data_not_available`
- `final_value`
- `audit_note`

## Role Boundaries

### Researcher
- Grounds operands with provenance and risk metadata.
- Stores `op1`, `op2`, ... in compute namespace.
- Does NOT compute the final derived answer.
- Verifies column headers and row disambiguation before storing values.
- Handles hierarchical table headers and split-layout tables.
- On iteration 2+: re-examines operands flagged by cross-verifier conflicts.

### Cross-Verifier
- Independently verifies ONE operand per iteration through a different source path.
- Smart skip: outputs `all_clear` with 0 tool calls when all operands are high-confidence.
- Uses `xv_` prefix for compute variables to avoid overwriting researcher's operands.
- Prefers strong independence (different file) over weak (same table, different route).

### Reflector
- Evaluates evidence sufficiency, epistemic risk, AND cross-verification results.
- NEVER marks "complete" when cross-verifier found a conflict.
- Relays cross-verifier's findings to the researcher with specific correction instructions.
- Uses `verification_priority` from operation_schema to calibrate rigor.

### Assembler
- Does not retrieve.
- Prefers operands independently confirmed by cross-verifier.
- Uses compute for arithmetic only (with auto-injected namespace preview).
- Runs scale validation and reverse checks before outputting.

### Arithmetic Auditor
- 0 tool calls. Runs one reverse check to catch formula-application errors.
- Passes through unchanged for lookups and data_not_available.

## Selection and Adjudication Rules

When multiple candidate values exist, stages should consistently prefer:
1. operands independently confirmed by cross-verification,
2. better definition/category match,
3. more direct source role,
4. closer temporality to the claimed period,
5. clearer row/column evidence,
6. higher confidence.

If materially different candidates remain plausibly valid after applying these rules,
the workflow should prefer `data_not_available` over guessing.

## Budget Analysis

| Scenario | Tool Calls |
|---|---|
| Best case (1 iteration, cross-verifier skips) | ~18 |
| Typical (2 iterations, 1 cross-verification) | ~51 |
| Worst case (3 full iterations) | ~119 |

## Notes on Maintainability

This workflow relies on compact, repeated vocabularies across prompts:
- `match_quality`, `source_role`, `source_temporality`
- `status` (per contract)
- `risk_reasons`
- `independence_strength`

If these contracts change, update both the YAML prompts and this document together.
