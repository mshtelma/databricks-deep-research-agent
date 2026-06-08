---
title: Home
template: home.html
hide:
  - navigation
  - toc
---

Most enterprise AI stops at single-shot RAG. **Deep Research Agent** plans a
multi-step investigation, runs it across every data source in your workspace,
and verifies every claim before it reaches the user — with full Unity Catalog
governance and the whole process streamed live to the UI.

## What you get

<div class="grid cards" markdown>

-   :material-robot-outline:{ .lg .middle } __5-agent research loop__

    ---

    Coordinator, Planner, Researcher, Reflector, and Synthesizer collaborate in
    an iterative, self-correcting loop that adapts its effort to the question.

    [:octicons-arrow-right-24: How the agents work](concepts/agents.md)

-   :material-shield-check-outline:{ .lg .middle } __Verified citations__

    ---

    A 7-stage pipeline ranks evidence, generates claims interleaved with sources,
    and gives every citation a confidence badge and verification verdict.

    [:octicons-arrow-right-24: Citation pipeline](concepts/citation-pipeline.md)

-   :material-auto-fix:{ .lg .middle } __Build agents in the UI__

    ---

    Turn a one-line prompt into a runnable multi-agent workflow with the Agent
    Designer's synchronized chat + canvas — no code required.

    [:octicons-arrow-right-24: Agent Designer](concepts/agent-designer.md)

-   :material-database-search-outline:{ .lg .middle } __Every Databricks source__

    ---

    Vector Search, Genie spaces, Knowledge Assistants, Delta tables, UC functions,
    uploaded files, and the open web — all first-class research tools.

    [:octicons-arrow-right-24: Architecture](concepts/architecture.md)

-   :material-shield-lock-outline:{ .lg .middle } __Governed by default__

    ---

    On-Behalf-Of auth means Unity Catalog permissions, row-level security, and
    column masking apply to every query, exactly as they would for the user.

    [:octicons-arrow-right-24: Why it matters](getting-started/overview.md)

-   :material-rocket-launch-outline:{ .lg .middle } __Deploy as a Databricks App__

    ---

    One command provisions Lakebase, grants permissions, and starts the app —
    or consume the framework as a standalone Python package.

    [:octicons-arrow-right-24: Deploy guide](getting-started/deploy.md)

</div>

## Proven on a hard benchmark

Evaluated on **OfficeQA**, an adversarial financial QA benchmark built on US
Treasury Bulletins (697 bulletins, 58,000+ tables). Across three full runs of the
133-question set with Claude Opus 4.6 — zero errors, zero timeouts:

<div class="dr-stats">
  <div class="dr-stat">
    <div class="dr-stat__value">54.9%</div>
    <div class="dr-stat__label">avg exact match (peak 57.9%)</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">69.2%</div>
    <div class="dr-stat__label">within 1% tolerance</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">73.2%</div>
    <div class="dr-stat__label">within 5% tolerance</div>
  </div>
  <div class="dr-stat">
    <div class="dr-stat__value">3&times;</div>
    <div class="dr-stat__label">independent full runs</div>
  </div>
</div>

[See the benchmark details :octicons-arrow-right-24:](benchmarks.md){ .md-button }

## Start here

[Get started](getting-started/overview.md){ .md-button .md-button--primary }
[Deploy to Databricks](getting-started/deploy.md){ .md-button }
[Using the app](getting-started/quickstart.md){ .md-button }
