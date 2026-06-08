# Using the app

Once the app is [deployed to your workspace](deploy.md), open it and ask a question.
Here's what to expect and how to steer it.

## Choose a query mode

Each question is routed through one of three modes for progressive disclosure:

| Mode | Latency target | What happens |
|------|----------------|--------------|
| **Simple** | < 3 s | Direct LLM response, no research |
| **Web Search** | < 15 s | Lightweight pipeline: 2–5 sources; falls back to Simple on timeout |
| **Deep Research** | < 2 min | Full multi-step pipeline with planning, reflection, and citation verification |

## Pick a research depth

Within Deep Research, three depth profiles trade speed for thoroughness:

| Depth | Steps | Researcher mode | Output |
|-------|-------|-----------------|--------|
| **Light** | 1–3 | `classic` (single-pass) | 800–2,000 words |
| **Medium** | 3–6 | `react` (LLM controls tool calls) | 1,200–3,000 words |
| **Extended** | 5–10 | `react`, up to 20 tool calls/step | 1,500–4,000 words |

Depth can be selected automatically from query complexity, chosen by the user, or locked
by a custom agent.

## Watch research happen live

Research streams to the UI in real time over Server-Sent Events: you see the planner form
a strategy, each step execute its tool calls, the reflector evaluate coverage, and the
report materialize sentence by sentence — with citation verdicts appearing as claims are
checked. Stream state survives a browser refresh.

## Build a custom agent

Use the **Agent Designer** to turn a one-line prompt into a reusable, multi-agent
research workflow — choosing data sources, models, depth, and prompts entirely through the
UI. No code required.

[:octicons-arrow-right-24: Agent Designer](../concepts/agent-designer.md)

## Next steps

<div class="grid cards" markdown>

-   :material-tune-variant:{ .lg .middle } __Configure it__

    ---

    Model tiers, research profiles, endpoints, and search providers in `app.yaml`.

    [:octicons-arrow-right-24: Configuration](../guides/configuration.md)

-   :material-web:{ .lg .middle } __Web search providers__

    ---

    The default Databricks built-in search, plus optional Brave and Jina.

    [:octicons-arrow-right-24: Web Search Providers](../guides/web-search-providers.md)

-   :material-code-braces:{ .lg .middle } __Contribute__

    ---

    Run it locally, make changes, and open a pull request.

    [:octicons-arrow-right-24: Contributing](../contributing.md)

</div>
