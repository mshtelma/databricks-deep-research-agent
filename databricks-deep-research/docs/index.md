# databricks-deep-research

A composable multi-agent orchestration framework for building research workflows on Databricks. Plain async Python, YAML-defined DAG workflows, 8 node types, 6 builtin agent subtypes, streaming events, 7-stage citation verification pipeline.

## Package Info

| | |
|---|---|
| **Version** | 0.2.0 |
| **Python** | 3.11+ |
| **Core deps** | `openai>=1.10`, `pydantic>=2.0`, `pyyaml>=6.0` |
| **Optional extras** | `web`, `crawl`, `search`, `tracing`, `all` |

```bash
pip install databricks-deep-research              # core only
pip install databricks-deep-research[all]          # everything
pip install databricks-deep-research[web,tracing]  # pick extras
```

## Key Capabilities

- **YAML-defined workflow DAGs** with 8 node types: `sequence`, `parallel`, `loop`, `conditional`, `agent`, `tool`, `subworkflow`, `plan_and_execute`
- **6 builtin agent subtypes**: coordinator, planner, researcher, reflector, synthesizer, background
- **ReAct tool-calling loop** with parallel execution and source-aware prompts
- **Multi-provider web search** -- `web_search`/`web_research` are backend-agnostic over a `SearchClient` (Databricks built-in web search, Brave, or Jina)
- **Delta/SQL table tools** -- the `table_*` family (discover, search, read, neighbors, load, aggregate) researches over bound Delta tables via `SourceKind.text_table`
- **Dataflow validation** -- dangling-read / dead-store checks at load time, lint by default
- **Shared research pools** with dedup, capacity limits, BM25+vector hybrid search
- **50+ typed streaming events** via a discriminated union (`FrameworkEvent`)
- **Model tier routing** (simple / analytical / complex) with health tracking and fallback
- **7-stage citation verification pipeline**: evidence selection, interleaved generation, confidence routing, NLI verification, citation correction, numeric QA, ARE retrieval
- **URL registry** for secure LLM-to-source reference mapping
- **Safe Jinja2 template rendering** for prompt authoring
- **MLflow `trace_span` integration** for observability

## Reading Tracks

### Quick Start (~15 min)

1. [Installation](getting-started/installation.md)
2. [Quickstart](getting-started/quickstart.md)
3. [Architecture](concepts/architecture.md)

### Workflow Builder (~1-2 hours)

1. Everything in Quick Start, plus:
2. [YAML Workflow Authoring](guides/yaml-workflow-authoring.md)
3. [Builtin Agents](guides/builtin-agents.md)
4. [Builtin Tools](guides/builtin-tools.md)
5. [Search Providers](guides/search-providers.md)
6. [SQL / Table Tools](guides/sql-table-tools.md)
7. [Pool Configuration](guides/pool-configuration.md)
8. [Model Configuration](guides/model-configuration.md)
9. [Streaming and Events](guides/streaming-and-events.md)

### Deep Dive (~half day)

1. Everything in Workflow Builder, plus:
2. All [Concepts](concepts/) docs (including [Runtime State](concepts/runtime-state.md))
3. [Custom Tools](guides/custom-tools.md)
4. [Custom Agents](guides/custom-agents.md)
5. [Citation Verification](guides/citation-verification.md)
6. [Dataflow Validation](guides/dataflow-validation.md)
7. All [Reference](reference/) docs

## Documentation Structure

| Section | Path | Contents |
|---|---|---|
| **Getting Started** | [`getting-started/`](getting-started/) | Installation, quickstart, architecture overview |
| **Concepts** | [`concepts/`](concepts/) | Workflow engine, agent harness, runtime state, pools, events, citation pipeline |
| **Guides** | [`guides/`](guides/) | YAML authoring, custom tools/agents, streaming, configuration |
| **Reference** | [`reference/`](reference/) | API docs for builtins, event catalog, config schema |
| **Security** | [`security/`](security/) | Threat-model notes (e.g. MCP/SSRF) |
| **Examples** | [`examples/`](examples/) | Sample workflows, tool implementations, integration patterns |

## See Also

- [GitHub repository](../)
- [`pyproject.toml`](../pyproject.toml)
- [Example workflows](../examples/)
