# Benchmarks

Claims about AI quality mean nothing without measurement. Deep Research Agent is
evaluated on **OfficeQA**, a challenging financial question-answering benchmark.

## What OfficeQA tests

OfficeQA is built on **US Treasury Bulletin** documents spanning 1939–2025 — 697
bulletins with over 58,000 structured tables. It is deliberately adversarial,
targeting capabilities that simple RAG cannot handle:

- Exact table-cell extraction
- Multi-step numerical computation (sums, averages, percent changes, regressions)
- Correct period and edition selection across document revisions
- Currency conversions requiring external data

Every question is graded **hard**: answers hinge on subtle distinctions between
fiscal and calendar years, document revisions published months apart, and table
hierarchies where the wrong row is one cell from the right one.

## How we ran it

We evaluate against the **Databricks-parsed edition** of the bulletins. The source
documents are ingested into Unity Catalog as Delta tables and a Databricks Vector
Search index, and every answer is grounded **entirely through Databricks
retrieval** — Vector Search over the parsed text, structured table/SQL tools over
the extracted tables, and a calculator. **No web search is used**: this is a
closed-corpus test over governed data, exactly the pattern for enterprise
deployments on Databricks.

| | |
|--|--|
| **Benchmark** | OfficeQA — full 133-question set ([databricks/officeqa](https://github.com/databricks/officeqa)) |
| **Model** | Claude Opus 4.6, via the Databricks Foundation Model API |
| **Retrieval** | Databricks Vector Search + structured table tools — no web search |
| **Scoring** | OfficeQA's official reward function: exact match, plus ±1% / ±5% numeric tolerance |

## Results

Our best workflow configuration on the full 133-question set:

<div class="dr-stats">
  <div class="dr-stat">
    <div class="dr-stat__value">55.6%</div>
    <div class="dr-stat__label">exact match</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">69.2%</div>
    <div class="dr-stat__label">within 1% tolerance</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">73.7%</div>
    <div class="dr-stat__label">within 5% tolerance</div>
  </div>
</div>

| Metric | Accuracy |
|--------|---------:|
| Exact match | 55.6% (74 / 133) |
| Within ±1% tolerance | 69.2% (92 / 133) |
| Within ±5% tolerance | 73.7% (98 / 133) |

Accuracy is measured over the **entire 133-question set** — any question the agent
cannot answer counts as wrong, so the numbers are not inflated by dropping hard
cases. The workflow (`workflow-v83-hybrid.yaml`) and scorer live under
[`benchmarks/officeqa/`](https://github.com/mshtelma/databricks-deep-research-agent/tree/main/benchmarks/officeqa),
which documents these results in full.

## Why it matters

OfficeQA is scored on **exact numerical extraction** — a single wrong row, edition,
or unit counts as a miss, so fluent-but-wrong answers earn no credit. That makes the
full-set exact-match rate a measure of analytical precision rather than surface
plausibility. The workflow behind these numbers was refined over many iterations,
each informed by systematic failure analysis and root-cause categorization.

[How citations are verified :octicons-arrow-right-24:](concepts/citation-pipeline.md){ .md-button }
