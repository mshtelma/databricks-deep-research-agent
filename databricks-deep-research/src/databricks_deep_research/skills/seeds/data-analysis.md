---
name: data-analysis
description: A methodology for analyzing structured data (CSV/Excel/tabular sources) with pandas in the compute scratchpad — inspect, aggregate, summarize, then report grounded numbers.
---

# Data Analysis Skill

## Overview

Analyze structured/tabular data with **pandas in the compute scratchpad**.
Operate on DataFrames programmatically rather than eyeballing rows, so every
reported number is computed and reproducible.

## Workflow

### Step 1: Understand the requirement

Identify before touching data:

- **Inputs** — which table(s)/file(s) hold the data.
- **Goal** — the insight wanted (summary, filter, aggregation, comparison,
  trend, ranking).
- **Output** — how the result should be presented (a number, a small table, a
  chart — see the `chart` skill).

### Step 2: Inspect structure first

Never compute before you know the schema. For each DataFrame check:

- Column names and dtypes (`df.dtypes`).
- Row count (`len(df)`) and null counts (`df.isna().sum()`).
- A small sample (`df.head()`).
- Distinct values for the key categorical columns
  (`df["category"].value_counts()`).

Coerce types deliberately — parse dates with `pd.to_datetime(...)` and numeric
columns with `pd.to_numeric(..., errors="coerce")` before aggregating.

### Step 3: Analyze

Compute the answer with explicit pandas operations:

- **Aggregation / grouping** — `df.groupby("category")["amount"].agg(["sum",
  "mean", "count"])`.
- **Filtering** — boolean masks (`df[df["amount"] > 1000]`).
- **Joins** — `pd.merge(left, right, on="key", how="inner")`.
- **Time series** — set a datetime index and `resample("M").sum()` for monthly
  rollups, `.rolling(window).mean()` for moving averages.
- **Ranking / top-N** — `.sort_values(..., ascending=False).head(10)`.
- **Pivots** — `df.pivot_table(index=..., columns=..., values=...,
  aggfunc="sum")`.

### Step 4: Report grounded results

- State the **units** for every number (currency, %, count) — never emit a bare
  number or a raw column id.
- Round for readability but keep precision honest.
- Present small results inline as a markdown table; describe larger ones.
- Explain findings in plain language with the key takeaway, then offer a useful
  follow-up analysis.

## Methodology rules

- Show the computation, not just the conclusion — a number with no derivation is
  not trustworthy.
- Handle missing data explicitly; decide whether to drop, fill, or report nulls.
- Validate surprising results (re-check the filter/join that produced them)
  before reporting.
- Prefer vectorized pandas over per-row Python loops.
