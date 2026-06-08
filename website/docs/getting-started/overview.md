# Overview

**Databricks Deep Research Agent** turns the data assets you already have on
Databricks — Vector Search indexes, Genie spaces, Knowledge Assistant endpoints,
Delta tables, and frontier foundation models — into a single, governed research
experience that answers complex questions with verified citations.

It ships as two things:

- a **production application** that deploys as a native Databricks App, and
- a standalone **orchestration framework** (`databricks-deep-research`) you can
  import into notebooks, jobs, and your own apps.

## Deep research, not simple RAG

Traditional RAG retrieves a few documents and generates one response. That breaks
down on questions that need multiple angles, numerical reasoning, or synthesis
across sources.

Deep Research Agent runs an **iterative, self-correcting loop**: it plans a
multi-step investigation, executes each step with targeted tool calls, and
critically evaluates its own progress before deciding what to do next. Simple
questions get fast, direct answers; complex ones receive 1–10 research cycles,
each building on the last.

[:octicons-arrow-right-24: Meet the five agents](../concepts/agents.md)

## Research across every Databricks source

Every Databricks service is a first-class research tool, and an agent can use any
combination of them in one session:

| Source | What it does |
|--------|--------------|
| **Vector Search** | Semantic + hybrid retrieval with metadata filters and reranking |
| **Genie Spaces** | Natural-language questions over governed tables, translated to SQL |
| **Knowledge Assistants** | Query existing domain-expert serving endpoints |
| **Delta Tables** | Structured reads governed by Unity Catalog |
| **Unity Catalog Functions** | Register custom logic as agent tools |
| **Uploaded Files** | Indexed on the fly for full-text search in a conversation |
| **Web Search & Crawl** | Open-web search with domain filtering, plus page extraction |

The app **auto-discovers** the Vector Search indexes, Genie spaces, and Knowledge
Assistants available to each user — no manual wiring. An intelligent source-routing
layer decides which sources to query and how often.

## Answers you can trust

Rather than appending citation numbers after the fact, the system generates claims
**interleaved with evidence** and runs them through a 7-stage verification pipeline.
Every citation in the final report carries a **confidence badge** and a
**verification verdict** (supported / partially supported / insufficient), so the
reasoning is transparent and auditable.

[:octicons-arrow-right-24: How citations are verified](../concepts/citation-pipeline.md)

## Three depths for every question

| Depth | Steps | Output | Best for |
|-------|-------|--------|----------|
| **Light** | 1–3 | 800–2,000 words | Quick factual lookups |
| **Medium** | 3–6 | 1,200–3,000 words | The everyday default |
| **Extended** | 5–10 (up to 20 tool calls/step) | 1,500–4,000 words | Deep analytical work |

Depth can be selected automatically from query complexity, chosen by the user, or
locked by a Custom Agent.

## Governed by default

Every query to a Databricks data source flows through **On-Behalf-Of (OBO)
authentication** — the user's own OAuth token, not a shared service account. Unity
Catalog permissions, row-level security, and column masking all apply exactly as if
the user queried the data directly. A junior analyst and a VP can use the same agent
against the same Genie space and each sees only what they're authorized to.

Add comprehensive audit logging, CSRF protection, security headers, and the fact
that the LLM never sees raw URLs (only opaque integer references), and the platform
is built for environments where data security is non-negotiable.

## Where to next

<div class="grid cards" markdown>

-   :material-download-outline:{ .lg .middle } __Install it__

    ---

    Prerequisites, dependencies, and environment setup for local development.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-flash-outline:{ .lg .middle } __Run your first query__

    ---

    Start the dev servers and understand query modes and research depths.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-sitemap-outline:{ .lg .middle } __See the architecture__

    ---

    How the frontend, API, orchestrator, and model tiers fit together.

    [:octicons-arrow-right-24: Architecture](../concepts/architecture.md)

-   :material-rocket-launch-outline:{ .lg .middle } __Deploy to Databricks__

    ---

    A single command provisions Lakebase and starts the app.

    [:octicons-arrow-right-24: Deploy](../deploy.md)

</div>
