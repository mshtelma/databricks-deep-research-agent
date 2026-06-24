# Databricks Deep Research Framework

## A Multi-Agent Orchestration Engine for Building Research-Grade AI Applications on Databricks

There are dozens of multi-agent frameworks. Most of them solve the same problem: chain LLM calls together with some tool calling. They work well for demos. They struggle in production — especially when you need iterative reasoning over enterprise data, verifiable citations, and the operational controls that real deployments demand.

Databricks Deep Research is a different kind of framework. It was built specifically for **deep research workflows** — the kind where an AI system needs to plan an investigation, execute it across multiple data sources, critically evaluate its own progress, and produce a trustworthy report with verified citations. And it was built to do all of this natively on the Databricks Data Intelligence Platform, using the models, data assets, and governance you already have.

The result is a standalone Python library that you can install, configure with YAML, and embed in any application — from notebooks to production services.

---

## Define Complex Workflows in YAML, Not Code

Most multi-agent frameworks require you to wire agents together in Python — constructing graph edges, defining state schemas, writing routing logic. The result is orchestration code that's hard to read, harder to modify, and impossible for non-engineers to understand.

Deep Research takes a different approach: **workflows are YAML files.** A complete multi-agent research pipeline — with planning, parallel execution, conditional branching, reflection loops, and synthesis — is a declarative document that anyone can read, modify, and version-control.

The workflow engine supports 8 node types that compose into arbitrarily complex DAGs:

**Leaf nodes** execute work: `agent` (LLM call with optional tools), `tool` (direct tool execution without LLM), and `subworkflow` (delegate to another workflow definition).

**Composite nodes** control flow: `sequence` (run children in order), `parallel` (run children concurrently), `loop` (repeat until a condition is met), `conditional` (branch based on state), and `plan_and_execute` (a specialized meta-node that plans research steps, executes them, and evaluates progress in a reflective loop).

A production research pipeline — coordinator classifies the query, background agent gathers initial context, a plan-and-execute cycle runs iterative research with reflection, and a synthesizer produces a cited report — fits in roughly 60 lines of YAML. Modifying the research strategy means editing a config file, not refactoring Python code.

This makes the framework accessible to a much wider audience than typical agent frameworks. ML engineers can design workflows. Data scientists can tune research strategies. Product managers can review and understand what the system does. And software engineers can extend it with custom agents and tools when needed.

---

## Built for Iterative Research, Not Single-Pass Retrieval

The fundamental limitation of RAG is that it's a single pass: retrieve some documents, generate a response. If the retrieval misses key information, the response is incomplete. If the question requires multiple angles of investigation, the system can't adapt.

Deep Research implements an **iterative, self-correcting research loop** as a first-class workflow pattern. The `plan_and_execute` node type orchestrates a cycle where:

A **planner** agent analyzes the question and designs a multi-step research strategy — which aspects to investigate, in what order, using which data sources. It receives a catalog of available tools and data sources, so plans are grounded in what the system can actually query.

A **researcher** agent executes each step using a ReAct tool-calling loop. It can search Vector Search indexes, query Genie spaces, crawl web pages, or call any custom tool — collecting evidence into shared pools that persist across steps.

A **reflector** agent evaluates progress after each step. It has access to the research plan, collected evidence, and source quality metrics. Based on this, it decides: **continue** (more research needed), **adjust** (shift focus to a different angle), or **complete** (sufficient evidence gathered).

If the reflector finds gaps, the system can **replan** — feeding feedback from completed steps and blocked sources back into the planner for a new strategy. This replanning loop can run multiple cycles, with each iteration building on everything learned in previous cycles.

The practical impact is that the system adapts its effort to the difficulty of the question. A straightforward factual lookup might complete in one step. A complex analytical question might run 5–10 steps across multiple data sources, adjusting its strategy as it learns what information is available. This is the difference between keyword search and actual research.

---

## Six Agent Types That Work Together

The framework ships with six builtin agent subtypes, each designed for a specific role in the research process. Together they form a complete research team:

The **coordinator** classifies incoming queries by complexity and decides whether a question needs full research or can be answered directly. It's the traffic cop that ensures simple questions get fast answers while complex ones receive thorough investigation.

The **background** agent runs quick, broad discovery before planning begins — decomposing the query, assessing the data landscape, and identifying which sources are likely to be relevant. This gives the planner a head start so it can design informed research strategies rather than guessing.

The **planner** generates structured research plans with ordered steps, each annotated with source hints and expected evidence types. It's source-aware — it knows which Vector Search indexes, Genie spaces, and web search tools are available, and incorporates that knowledge into its plans.

The **researcher** executes individual research steps through a ReAct tool-calling loop. It calls tools, evaluates results, decides whether to search deeper or move on, and collects findings into shared evidence pools. Each researcher invocation handles one step of the plan, with configurable limits on tool calls and token budget.

The **reflector** provides critical evaluation after each research step. It examines the evidence collected so far, compares it against the research plan, identifies gaps, and decides whether to continue, adjust direction, or declare the research complete. This is the mechanism that prevents both premature stopping and unnecessary over-research.

The **synthesizer** produces the final output — a comprehensive report grounded in the collected evidence. It supports three grounding modes: direct generation (fastest), post-synthesis verification (balanced), and interleaved claim-evidence generation (most rigorous). In verification modes, the synthesizer feeds into the 7-stage citation pipeline.

Each agent type is registered through a simple pattern — provide a config enrichment function and an event post-processing function — making it straightforward to add custom agent subtypes for domain-specific roles without modifying the framework.

---

## Agents Build on Each Other's Work — And Can Never Destroy It

The most insidious problem in multi-agent systems isn't getting agents to cooperate — it's preventing them from sabotaging each other. In frameworks built on mutable shared state, any agent can overwrite, corrupt, or erase what a previous agent produced. A planner writes a strategy; a researcher accidentally overwrites it with its own output. A reflector's careful evaluation gets clobbered by a parallel agent writing to the same key. These bugs are silent, hard to reproduce, and devastating in production.

Deep Research solves this at the architectural level with an **append-only state model**. The workflow state is an immutable log: every agent write appends a new entry — it never modifies or deletes an existing one. When an agent reads a value, it gets the latest entry for that key via an O(1) index lookup. But every previous value is preserved in the log, creating a complete, tamper-proof history of how the research evolved.

This design has profound implications for multi-agent collaboration:

**Knowledge accumulates, never regresses.** The background agent's data landscape assessment, the planner's research strategy, each researcher's findings, each reflector's evaluation — all coexist in the state log. Later agents build on earlier agents' work by reading their outputs, but can never accidentally erase them. The synthesizer at the end of the pipeline has access to everything every agent produced, in order.

**Concurrent agents are safe by default.** When parallel researchers execute simultaneously, their writes don't conflict. Each appends to the log independently. There are no race conditions, no lost updates, no need for distributed locks. The async lock guards append ordering, not data integrity — the data model itself is conflict-free.

**The full history is always available.** Need to debug why the reflector decided to replan? Read the log entries for that step. Need to audit which sources were collected at each stage? The state log has timestamped entries for every write. Need to understand how the research direction shifted over time? Walk the log entries for the planner across replan cycles. This is built-in observability that mutable state frameworks simply cannot provide.

**Agent isolation reinforces state safety.** Each agent operates within a strict input/output boundary. An agent receives an `AgentInput` envelope with rendered prompts, pool sections, and tool definitions — all immutable. It produces an `AgentOutput` envelope with content, pool writes, and sources. The harness mediates all state interaction. Agents cannot reach into global state, other agents' outputs, or shared resources directly. This means a buggy or adversarial agent is contained — it can produce bad output, but it cannot corrupt the evidence that other agents collected.

This combination of append-only state and agent isolation is what makes the framework suitable for production multi-agent systems. It's not just a design pattern — it's a structural guarantee that the system's knowledge base can only grow, never shrink.

---

## Every Databricks Data Source as a Research Tool

The framework includes builtin tools for every major Databricks data service, all accessible through a unified protocol:

**Vector Search** enables semantic and hybrid search over your existing Unity Catalog indexes. The tool auto-discovers available columns, supports metadata filtering, and returns ranked results with relevance scores. Because it uses On-Behalf-Of authentication, every query respects the current user's Unity Catalog permissions.

**Genie** enables natural language questions over structured data. The agent can ask analytical questions — "What were Q3 revenue numbers by region?" — and Genie translates them to SQL against your governed tables. The tool maintains conversation state for follow-up questions and handles the asynchronous polling pattern transparently.

**Knowledge Assistants** connect to any serving endpoint that provides Q&A capabilities — fine-tuned models, RAG chains, or custom inference pipelines deployed on Databricks Model Serving.

**Delta Table** tools provide direct structured data access, with reads governed by Unity Catalog ACLs.

**Web Search** (via Brave API) and **Web Crawl** (via httpx + trafilatura) provide open web access with domain filtering, freshness controls, and content extraction.

**File Search** provides BM25 full-text search over uploaded or indexed documents.

All tools implement the same `ResearchTool` protocol — three methods and a structured result type — so custom tools integrate identically to builtins. A tool factory pattern with dependency injection means your custom tools receive workspace clients, user tokens, and API keys at construction time without manual wiring.

The framework's **source-aware planning** goes a step further: when the planner generates research steps, it receives a catalog of all available tools and their capabilities. Plans include source hints that tell the researcher which tools to prefer for each step. The researcher's admission filter then evaluates whether tool results are actually relevant to the current step before adding them to the evidence pool — preventing noise from drowning out signal.

---

## Verified Citations, Not Decorative Footnotes

Citation quality is one of the hardest unsolved problems in RAG applications. Most systems add reference numbers after generation — a cosmetic gesture that doesn't verify whether the cited source actually supports the claim.

Deep Research includes a **7-stage citation verification pipeline** that produces evidence-backed, confidence-rated citations:

**Evidence pre-selection** ranks the most relevant spans from all collected sources, chunking long documents and scoring by relevance to the query.

**Interleaved generation** produces claims alongside their supporting evidence, rather than generating text first and adding citations afterward. Each claim carries explicit citation markers tied to specific source material.

**Confidence classification** routes claims by certainty level — high-confidence claims pass through quickly, while uncertain claims receive deeper verification.

**Isolated verification** evaluates each flagged claim independently, producing a verdict: *supported*, *partially supported*, *unsupported*, or *contradicted* — with reasoning.

**Citation correction** automatically rebinds weak citations to stronger supporting sources from the evidence pool, or removes citations that can't be substantiated.

**Numeric QA verification** applies specialized checks to numeric claims — exact match, range overlap, and percentage-level accuracy — catching the subtle numerical errors that plague LLM-generated content.

**ARE retrieval** (Atomic Retrieval and Evaluation) decomposes claims with insufficient evidence into atomic facts and searches for additional supporting material.

The pipeline is fully configurable — enable or disable individual stages, choose between synthesis modes (classical, natural, strict), and control verification thresholds. In the Deep Research application, these verification results surface as confidence badges and verdict indicators in the UI, making the AI's reasoning transparent and auditable.

---

## Intelligent Model Routing Across Foundation Model API

The framework doesn't assume a single model for all tasks. Every LLM call is routed through a **tiered model system** that matches model capability to task complexity:

**Simple** tier handles lightweight tasks — query classification, quick lookups — using fast, cost-efficient models.

**Analytical** tier handles research steps, tool calling, and reflection — tasks that need good reasoning but run frequently.

**Complex** tier handles final synthesis, deep analysis, and citation verification — where output quality justifies the cost of a frontier model.

Each tier can be configured with multiple endpoints in a fallback chain. If the primary model hits a rate limit, the framework automatically tries the next endpoint. Health tracking monitors each endpoint's error rate, rate limit status, and tokens-per-minute consumption. Endpoints that fail repeatedly are temporarily removed from rotation and automatically restored after a cooldown period.

The routing supports two selection strategies: **priority** (always try the best model first) and **round-robin** (distribute load evenly). Both respect per-endpoint token budgets that prevent any single tier from monopolizing capacity.

For Databricks users, this means you can route through any model available on Foundation Model API — Claude, GPT, Gemini, Llama, or any custom model — with automatic failover and cost optimization. Configure it once in YAML; the framework handles the operational complexity.

---

## Shared Research Pools: A Growing Knowledge Base

The append-only state model guarantees that agents can't destroy each other's work. Research pools take this further — they provide a **structured, searchable knowledge base that grows richer as the investigation progresses**, enabling each agent to leverage everything discovered by every agent before it.

This is modeled after how effective human research teams work. A team investigating a complex question doesn't start from scratch at each step. They maintain shared notes — a running collection of sources found, observations made, and claims identified. Each team member adds to this shared base, and everyone can search through it to inform their next move. Pools bring this same pattern to multi-agent workflows.

A typical research workflow configures two pools: a **sources** pool (deduplicated by URL, capped at 200 items) and an **observations** pool (deduplicated by content hash, capped at 100 items). As researchers execute their steps, findings flow into these pools automatically. The evidence base grows with each step:

After step 1, the researcher has collected 5 sources and 3 observations. The reflector evaluates this limited evidence and decides to continue. After step 3, the pools contain 15 sources and 10 observations — the reflector now has a much richer picture and can make better coverage assessments. By step 6, the synthesizer receives a curated, deduplicated collection of 40+ sources and 25+ observations, organized and searchable — far better input than the raw output of any single tool call.

**Deduplication prevents noise accumulation.** Two strategies work independently or together: key-based dedup (e.g., reject a source if its URL already exists in the pool) and content-hash dedup (reject an observation if its SHA-256 hash matches an existing entry). This means that even when multiple research steps query overlapping sources, the pool stays clean. The synthesizer works from unique evidence, not a pile of duplicates.

**Capacity limits with FIFO eviction keep pools focused.** When a pool reaches its configured maximum, the oldest items are evicted to make room. This naturally favors recent, more-targeted findings over early broad strokes — the same way a human researcher refines their notes as understanding deepens.

**Hybrid search makes pools queryable, not just appendable.** Agents can search pools using a combination of BM25 keyword matching and vector similarity. This degrades gracefully — full hybrid when embeddings are available, BM25 when they're not, keyword overlap as a universal fallback. Each pool also exposes auto-generated tools (search, get recent, count, list topics) that agents can invoke during their ReAct loops, enabling mid-step evidence retrieval.

**Pool injection and pool writes create the feedback loop.** Pool injection reads from a pool and includes the content in an agent's prompt — the reflector sees all collected evidence, the planner sees discovered sources, the synthesizer receives the full observation history. Pool writes extract findings from agent output and append them to the appropriate pool. This creates a continuous feedback loop where each agent's contribution enriches the context for every subsequent agent.

The net effect is **compounding intelligence**: the quality of the system's output isn't determined by any single agent's capability, but by the accumulated evidence and reasoning of the entire team. A researcher that finds a critical source in step 4 improves the reflector's assessment in step 5, which leads to a better-targeted plan in cycle 2, which produces more relevant evidence for the synthesizer. This cumulative dynamic is what separates deep research from simple retrieval — and it's only possible because the underlying data model guarantees that knowledge accumulates safely.

---

## 50+ Typed Streaming Events for Real-Time Visibility

Every action in a workflow emits a typed streaming event. The framework defines over 50 event types organized into clear categories:

**Workflow lifecycle** events mark the start and completion of the overall workflow, with aggregate metrics (total tokens, sources, execution time).

**Node lifecycle** events track each node's execution — started, completed, errored, skipped, or budget-exceeded.

**Agent events** capture output production, including token-by-token streaming chunks for real-time synthesis display.

**Tool events** report every tool call, its arguments, results, source counts, and cache hits.

**Domain events** from builtin agents provide semantic meaning: the coordinator's classification, the planner's research strategy, the reflector's continue/adjust/complete decision, the synthesizer's grounding mode.

**Citation events** stream the verification pipeline's progress: claims generated, claims verified (with verdicts), citations corrected, numeric claims detected, and a final verification summary.

All events are Pydantic models with a discriminated union on `event_type`, so they deserialize reliably in any language. Consuming them is a simple async for-loop with pattern matching — no callback registration, no event bus, no framework-specific abstractions.

This streaming architecture means you can build real-time UIs that show research progress as it happens, log every decision for compliance, or feed events into monitoring systems — all from the same event stream.

---

## Extend with Custom Tools and Agents

The framework is designed to be extended without forking.

**Custom tools** implement a three-method protocol: a property that returns the tool's definition (name, description, JSON Schema parameters), an argument validation method, and an async execute method that returns a structured result with content, sources, and optional structured data. Dependencies like workspace clients, API keys, and user tokens are injected at construction time through a factory context — your tool logic stays clean and testable.

For tools that should be declarable in YAML, implement a **tool factory** that maps a tool "kind" string to a constructor. Register the factory with the tool resolver, and your custom tool becomes available to any workflow that declares it — no code changes to existing workflows required.

**Custom agent subtypes** follow a registration pattern: provide a function that enriches the agent's configuration with default prompts and settings, and a function that post-processes output into domain-specific events. Register the subtype by name, and it's immediately available in YAML workflows via `subtype: your_custom_type`. No inheritance hierarchies, no abstract base classes — just function registration.

The **URL registry** ensures that LLMs never see raw URLs. Every URL surfaced by a tool is registered and replaced with an integer index. When the LLM references "[3]" in its output, the registry resolves it back to the actual URL. This prevents an entire class of hallucinated-URL attacks that plague other frameworks, and it works automatically for every tool without opt-in.

---

## Cost and Resource Control Built In

Production AI applications need cost guardrails. The framework provides multiple layers of control that operate automatically:

**Token budgets** cap total LLM consumption at the workflow level. When the budget is exhausted, the system completes the current step gracefully and stops — no runaway costs from circular reasoning or infinite tool loops.

**Time budgets** enforce wall-clock limits on individual nodes. A researcher that takes too long triggers a budget-exceeded event and the workflow moves on, rather than blocking the entire pipeline.

**Tool call limits** cap how many times an agent can invoke tools in a single ReAct loop (default: 15). This prevents the LLM from entering search spirals while still allowing thorough investigation.

**Per-tool limits** let you constrain specific tools independently — allow 20 web searches but only 5 Genie queries per session.

**Rate limit resilience** with configurable backoff (linear or exponential), jitter, and multi-endpoint failover prevents failed requests from cascading into expensive retry storms.

Together, these controls let you deploy research workflows with predictable cost profiles. Set budgets in YAML; the framework enforces them.

---

## Plain Async Python — No Framework Lock-In

Deep Research is built on **plain async Python**, Pydantic, and the OpenAI client interface. There is no dependency on LangGraph, LangChain, CrewAI, AutoGen, DSPy, or any other agent framework.

This is a deliberate architectural choice. Framework lock-in is the hidden cost of most multi-agent libraries — you adopt their state management, their execution model, their abstractions, and when you hit a limitation, you're stuck. Deep Research uses standard Python patterns (async generators, dataclasses, protocols) that any Python developer already knows.

The LLM client wraps the standard OpenAI `AsyncOpenAI` interface, which means it works with any OpenAI-compatible API — including Databricks Foundation Model API, Azure OpenAI, and any other provider that implements the chat completions spec. Switching providers means changing a configuration value, not rewriting orchestration code.

The framework is installable as a standard Python package with optional extras — pick the capabilities you need (web search, crawling, pool search, tracing) and skip what you don't.

---

## MLflow Tracing for End-to-End Observability

Every workflow execution can be traced through MLflow with span-level granularity. Each agent call, tool execution, and LLM request creates a trace span with structured attributes — query text, model used, token counts, source counts, and timing.

Tracing is opt-in and adds zero overhead when disabled. Enable it with a single function call, and every subsequent workflow execution produces traces that are viewable in the MLflow UI, searchable by experiment, and exportable for analysis.

This gives you end-to-end visibility into research pipelines: which tools were called, what they returned, how much each step cost, where time was spent, and how the system arrived at its final answer.

---

## Embed Anywhere — Notebooks, Services, Pipelines

The framework is a Python library, not a standalone application. This means you can use it wherever Python runs:

**In notebooks** — import the runner, load a workflow, and execute research queries interactively. Stream events to see the research process unfold in real time.

**In web applications** — embed the framework behind a FastAPI or Flask endpoint. The async generator interface maps naturally to Server-Sent Events for real-time streaming UIs.

**In batch pipelines** — run research workflows at scale over lists of questions, collecting structured results for downstream analysis.

**In Databricks Jobs** — schedule automated research runs that produce reports, populate dashboards, or feed into other ML pipelines.

The framework exposes three API levels: a high-level `WorkflowRunner` for common use cases (load YAML, run query, get result), a mid-level `WorkflowExecutor` for custom execution control, and low-level primitives (agents, tools, pools) for building entirely custom orchestration logic.

---

## Comprehensive Documentation

The framework ships with **41 documentation files** organized into progressive learning tracks:

**Getting Started** (15 minutes) covers installation, a quickstart with three execution patterns, and Databricks authentication.

**Workflow Builder** (1–2 hours) teaches YAML workflow authoring, the builtin agent subtypes and tools, pool configuration, and event consumption — everything you need to build and customize research pipelines.

**Deep Dive** (half day) covers the full architecture, custom tool and agent development, the citation verification pipeline, conditions and branching, error handling, and the complete API reference.

Every concept has a dedicated document. Every builtin agent and tool is documented with parameters, examples, and tips. The reference section includes JSON Schema for workflow validation, a complete event type catalog, and the full configuration specification.

13 example workflows demonstrate every pattern — from a single-agent tool loop to a full production pipeline with parallel execution, conditional branching, reflection, and citation verification.

---

## OfficeQA Benchmark Results

Architecture claims are easy to make. Benchmark results are not. The Deep Research framework has been rigorously evaluated on **OfficeQA**, a challenging financial question-answering benchmark that exposes exactly the kinds of failures that simple RAG systems can't overcome.

OfficeQA consists of 133 questions over US Treasury Bulletin documents — 697 bulletins spanning 1939 to 2025, containing over 58,000 structured tables. The questions aren't simple lookups. They require exact table cell extraction from hierarchically structured documents, multi-step numerical computation (sums, averages, percent changes, linear regressions), correct selection among multiple editions and revisions of the same bulletin, and currency conversions requiring external data. This is the kind of adversarial, real-world analytical task that separates research-grade systems from demo-quality prototypes.

We evaluate against the **Databricks-parsed edition** of the bulletins: the documents are ingested into Unity Catalog as Delta tables and a Databricks Vector Search index, and every answer is grounded **entirely through Databricks retrieval** — Vector Search over the parsed text plus structured table/SQL tools — with **no web search**. This is a closed-corpus test over governed data, exactly the pattern for enterprise deployments.

On the complete 133-question benchmark (Claude Opus 4.6 via the Databricks Foundation Model API), our best workflow configuration achieves:

- **55.6% exact match accuracy** (74 / 133)
- **69.2% accuracy within 1% numeric tolerance** (92 / 133)
- **73.7% accuracy within 5% numeric tolerance** (98 / 133)

Accuracy is measured over the entire 133-question set — any question the agent cannot answer counts as wrong. The benchmark is adversarial: questions hinge on subtle distinctions between fiscal and calendar years, document revisions published months apart, and table hierarchies where the wrong row is one cell away from the right one.

What makes these results meaningful for the framework is that they validate the architectural choices described in this document. Achieving majority accuracy on exact numerical extraction requires every layer of the system working together:

**The iterative research loop** enables the system to discover that an initial table doesn't contain the right edition, replan to search for the correct revision, and try again — something a single-pass RAG system cannot do.

**The append-only state model** ensures that evidence collected in step 2 is still available in step 7, and that replanning in cycle 2 can build on everything learned in cycle 1 — without any risk of earlier findings being overwritten.

**Shared research pools** give the synthesizer access to the full, deduplicated evidence base — not just the output of the final research step, but the accumulated findings of the entire investigation.

**Source-aware planning** means the planner knows which tools can query structured tables versus unstructured documents, and designs research steps accordingly.

**Citation verification** catches the subtle numerical errors (wrong row, wrong edition, wrong unit) that would otherwise be invisible in a fluent-sounding response.

The benchmark development spanned 20 workflow iterations, each informed by systematic failure analysis with per-question root cause categorization. This iterative refinement process — test, analyze failures, adjust workflow, retest — is itself enabled by the framework's streaming events and append-only state, which make it possible to trace exactly why the system got a specific question wrong and what to change.

---

## Why This Framework Over Alternatives

If you're evaluating multi-agent frameworks for building AI applications on Databricks, here's what makes Deep Research different:

**It's purpose-built for research, not generic agent chaining.** The plan-and-execute loop with reflection, shared evidence pools, source-aware planning, and citation verification are first-class features — not patterns you have to build yourself on top of a generic graph engine.

**Its data model is designed for multi-agent safety.** Append-only state with agent isolation means agents build on each other's work without risk of corruption. Shared pools accumulate knowledge across steps. No mutable shared dictionaries, no race conditions, no silent overwrites. This isn't a best practice — it's a structural guarantee enforced by the framework.

**It's Databricks-native.** Vector Search, Genie, Knowledge Assistants, Foundation Model API, Unity Catalog, MLflow — these aren't plugins bolted on after the fact. They're builtin tools with OBO authentication, source-aware query optimization, and governance that works the way your platform expects.

**It's measured on a hard benchmark.** On OfficeQA — an adversarial numerical QA benchmark — the framework scores 55.6% exact match on the full 133-question set, answered entirely from a Databricks-governed corpus with no web search.

**It's YAML-first.** Workflows are configuration, not code. This lowers the barrier to entry for non-engineers, makes workflows auditable and version-controllable, and separates research strategy from implementation details.

**It verifies its own output.** The 7-stage citation pipeline doesn't just attach footnotes — it evaluates whether citations actually support claims, corrects weak references, and flags numeric inaccuracies. No other multi-agent framework includes this capability.

**It's operationally mature.** Tiered model routing with health tracking and failover, token and time budgets, rate limit resilience, and 50+ typed streaming events give you the production controls that demos don't need but deployments require.

**It has no framework dependencies.** Plain async Python means no lock-in, no hidden abstractions, and no framework-specific debugging. Standard patterns, standard tools, standard Python.
