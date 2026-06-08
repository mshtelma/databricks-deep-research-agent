# Quickstart

Once you've [installed](installation.md) the project and configured `.env`, you can
run the full app locally with hot reload.

## Start the dev servers

```bash
make dev
```

This starts both servers with hot reload:

- **Frontend** — <http://localhost:5173>
- **Backend API** — <http://localhost:8000>

Open the UI at <http://localhost:5173> and ask a question. To run the backend or
frontend alone, use `make dev-backend` or `make dev-frontend`. To use a different
port (e.g. alongside another worktree): `PORT=8001 make dev`.

## Choose a query mode

The app routes each question through one of three modes for progressive disclosure:

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

!!! info "Classic vs. ReAct researcher"
    - **`classic`** runs a fixed set of searches/crawls per step — faster, predictable.
    - **`react`** lets the LLM decide which tools to call within a budget — more
      intelligent, better for open-ended questions.

## Watch it work

Research streams to the UI in real time over Server-Sent Events: you see the planner
form a strategy, each research step execute its tool calls, the reflector evaluate
coverage, and the report materialize sentence by sentence — with citation verdicts
appearing as claims are checked. Stream state survives a browser refresh.

## Production build

To serve the built UI from the backend (single server on `:8000`):

```bash
make build      # build frontend into static/
make prod       # run the unified production server
```

## Next steps

<div class="grid cards" markdown>

-   :material-tune-variant:{ .lg .middle } __Configure it__

    ---

    Model tiers, research profiles, endpoints, and search providers in `app.yaml`.

    [:octicons-arrow-right-24: Configuration](../guides/configuration.md)

-   :material-auto-fix:{ .lg .middle } __Build a custom agent__

    ---

    Compose a multi-agent workflow from a prompt in the Agent Designer.

    [:octicons-arrow-right-24: Agent Designer](../concepts/agent-designer.md)

-   :material-rocket-launch-outline:{ .lg .middle } __Deploy to Databricks__

    ---

    Ship it as a native Databricks App in one command.

    [:octicons-arrow-right-24: Deploy](../deploy.md)

</div>
