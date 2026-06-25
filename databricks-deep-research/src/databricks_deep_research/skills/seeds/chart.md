---
name: chart
description: A methodology for turning a pandas DataFrame into a clear figure — pick the right chart type for the data, build it in the compute scratchpad, and label it well.
---

# Chart Skill

## Overview

Turn a prepared DataFrame into a single, clear figure. Build charts
programmatically in the compute scratchpad (matplotlib over a DataFrame) so the
figure is reproducible from the data. Prepare the data first with the
`data-analysis` skill.

## Step 1: Choose the chart type from the data shape

Match the chart to what the data represents:

| Data relationship | Chart type | DataFrame shape |
|---|---|---|
| Trend over time | Line (or area for cumulative) | datetime x, numeric y |
| Category comparison | Bar / column | category x, numeric y |
| Part-to-whole | Pie (few parts) / stacked bar | category + share |
| Distribution | Histogram / box plot | one numeric column |
| Correlation | Scatter | two numeric columns |
| Two different scales | Dual-axis line | shared x, two y |

Prefer the simplest chart that answers the question; avoid pie charts with many
slices.

## Step 2: Build the figure

In the compute scratchpad, plot from the DataFrame:

```python
import matplotlib

matplotlib.use("Agg")  # headless: render without a display
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(kind="bar", x="category", y="revenue", ax=ax, legend=False)
ax.set_title("Revenue by category")
ax.set_xlabel("Category")
ax.set_ylabel("Revenue (USD)")
fig.tight_layout()
```

## Step 3: Label everything

A figure is only useful if it is self-explanatory:

- **Title** — what the chart shows.
- **Axis labels with units** — e.g. "Revenue (USD)", "Date", "Count".
- **Legend** — only when more than one series is plotted.
- Sort categorical bars by value for readability; format large numbers
  (thousands separators / units).

## Step 4: Return the result

- Persist the figure (e.g. `fig.savefig(path)`) and reference it; do not read raw
  image bytes back into the conversation.
- Briefly state what the chart shows and the single most important takeaway.

## Methodology rules

- One message per chart — one clear idea per figure.
- Never invent data points; plot only what is in the DataFrame.
- Keep the styling minimal; clarity beats decoration.
