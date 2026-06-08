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

Questions often hinge on subtle distinctions between fiscal and calendar years,
document revisions, and hierarchical table structures.

## Results

Across **three full runs** of the complete 133-question benchmark (April 2026, Claude
Opus 4.6 via the Databricks Foundation Model API) — with **zero errors and zero
timeouts**:

<div class="dr-stats">
  <div class="dr-stat">
    <div class="dr-stat__value">54.9%</div>
    <div class="dr-stat__label">avg exact match &middot; peak 57.9%</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">69.2%</div>
    <div class="dr-stat__label">within 1% tolerance &middot; peak 72.2%</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">73.2%</div>
    <div class="dr-stat__label">within 5% tolerance &middot; peak 75.2%</div>
  </div>
</div>

| Metric | Average | Best run |
|--------|---------|----------|
| Exact match | 54.9% | 57.9% |
| Accuracy (±1% fuzzy tolerance) | 69.2% | 72.2% |
| Accuracy (±5% fuzzy tolerance) | 73.2% | 75.2% |

These are not cherry-picked — they are the average across three independent runs on the
full benchmark. The variance between runs (52.6%–57.9% exact match) reflects the
inherent non-determinism of multi-step agentic reasoning, and even the lowest run
exceeds 50% exact match.

## Why it matters

Achieving majority accuracy on **exact numerical extraction** — not just "close enough"
text generation — demonstrates analytical depth well beyond traditional RAG. The
workflow behind these numbers was refined over 20 iterations, each informed by
systematic failure analysis and root-cause categorization.

[How citations are verified :octicons-arrow-right-24:](concepts/citation-pipeline.md){ .md-button }
