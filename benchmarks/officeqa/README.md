# OfficeQA Benchmark

[OfficeQA](https://github.com/databricks/officeqa) is an adversarial financial
question-answering benchmark over US Treasury Bulletins — 697 bulletins spanning
1939–2025 with 58,000+ structured tables. The 133-question set tests exact
table-cell extraction, multi-step numerical computation, edition/revision
selection, and currency conversion; every question is graded *hard*.

## Results

Run with `workflow-v83-hybrid.yaml`, Claude Opus 4.6 via the Databricks Foundation
Model API, on the full 133-question set:

| Metric | Accuracy |
|--------|---------:|
| Exact match | 55.6% (74 / 133) |
| Within ±1% numeric tolerance | 69.2% (92 / 133) |
| Within ±5% numeric tolerance | 73.7% (98 / 133) |

Accuracy is over the entire 133-question set; any question the agent cannot answer
counts as wrong. Scored with the official OfficeQA reward function (`reward.py`).

Reference run: MLflow `14eaadda00d9`, 2026-04-01.

## Setup

The agent answers **entirely from the Databricks-parsed edition** of the bulletins —
ingested into Unity Catalog as Delta tables and a Databricks Vector Search index,
then queried through Vector Search and structured table/SQL tools. **No web search
is used.**

## Run it

```bash
uv run benchmarks/run.py officeqa \
  --workflow workflow-v83-hybrid.yaml \
  --model databricks-claude-opus-4-6
```

Corpus ingestion (Unity Catalog + Vector Search) is configured in `config.yaml`
(see `ingest.py`).
