# databricks-deep-research

A composable multi-agent orchestration framework for building research workflows on Databricks. Plain async Python with YAML-defined DAG workflows, 8 node types, 6 builtin agent subtypes, streaming events, and a 7-stage citation verification pipeline.

## Key Capabilities

- **YAML-defined workflow DAGs** with 8 node types: `sequence`, `parallel`, `loop`, `conditional`, `agent`, `tool`, `subworkflow`, `plan_and_execute`
- **6 builtin agent subtypes**: coordinator, planner, researcher, reflector, synthesizer, background
- **ReAct tool-calling loop** with parallel execution and source-aware prompts
- **Shared research pools** with dedup, capacity limits, BM25+vector hybrid search
- **50+ typed streaming events** via a discriminated union (`FrameworkEvent`)
- **Model tier routing** (simple / analytical / complex) with health tracking and fallback
- **7-stage citation verification pipeline**: evidence selection, interleaved generation, confidence routing, NLI verification, citation correction, numeric QA, ARE retrieval
- **URL registry** for secure LLM-to-source reference mapping
- **Safe Jinja2 template rendering** for prompt authoring
- **MLflow `trace_span` integration** for observability

## Installation

```bash
# Install from source (not yet on PyPI)
uv pip install -e ".[all]"

# Or with pip
pip install -e ".[all]"
```

### Optional Extras

| Extra | Packages | Use Case |
|-------|----------|----------|
| `web` | `httpx>=0.24` | Web search tool (Brave API) |
| `crawl` | `trafilatura>=1.6` | Web page crawling and text extraction |
| `search` | `bm25s>=0.1`, `numpy>=1.24` | Pool hybrid BM25+vector search |
| `tracing` | `mlflow>=2.10` | MLflow `trace_span` observability |
| `all` | `web` + `crawl` + `search` + `tracing` | Full feature set |
| `dev` | `pytest>=8.0`, `pytest-asyncio>=0.23`, `mypy>=1.8`, `ruff>=0.2` | Development/testing |
| `integration` | `all` + `dev` + `databricks-sdk>=0.20` | Integration testing with Databricks |

## Quick Example

```python
import asyncio
from openai import AsyncOpenAI
from databricks_deep_research import WorkflowRunner, load_workflow

async def main():
    client = AsyncOpenAI()
    workflow = load_workflow("examples/simple_research.yaml")
    runner = WorkflowRunner(workflow, client, model_mapping={
        "simple": "gpt-4o-mini",
        "analytical": "gpt-4o",
        "complex": "gpt-4o",
    })

    result = await runner.run(query="What are the latest advances in quantum computing?")
    print(result.output)

asyncio.run(main())
```

### Stream Events

```python
async def main_streaming():
    client = AsyncOpenAI()
    workflow = load_workflow("examples/simple_research.yaml")
    runner = WorkflowRunner(workflow, client, model_mapping={
        "simple": "gpt-4o-mini",
        "analytical": "gpt-4o",
        "complex": "gpt-4o",
    })

    async for event in runner.stream(query="What is the current state of fusion energy?"):
        match event.event_type:
            case "node_started":
                print(f"Starting: {event.node_id}")
            case "plan_created":
                print(f"Research plan: {event.steps}")
            case "reflection_decision":
                print(f"Reflector says: {event.decision}")
            case "agent_stream_chunk":
                print(event.chunk, end="", flush=True)
            case "workflow_completed":
                print(f"\nDone! Tokens: {event.total_tokens}")

asyncio.run(main_streaming())
```

## Example Workflows

| File | Description |
|------|-------------|
| [`simple_research.yaml`](examples/simple_research.yaml) | Minimal research pipeline with planner, researcher, synthesizer |
| [`search_and_summarize.yaml`](examples/search_and_summarize.yaml) | Quick web search with summary |
| [`single_agent.yaml`](examples/single_agent.yaml) | Single agent with ReAct tool loop |
| [`parallel_research.yaml`](examples/parallel_research.yaml) | Parallel researcher execution |
| [`conditional_research.yaml`](examples/conditional_research.yaml) | Conditional branching based on query complexity |
| [`research_pipeline.yaml`](examples/research_pipeline.yaml) | Full pipeline with reflection loop |
| [`citation_pipeline.yaml`](examples/citation_pipeline.yaml) | Research with 7-stage citation verification |
| [`enterprise_research.yaml`](examples/enterprise_research.yaml) | Enterprise data sources (Vector Search, Genie) |
| [`classical_enterprise_vector_search.yaml`](examples/classical_enterprise_vector_search.yaml) | Classic researcher with Vector Search |
| [`genie_enterprise_research.yaml`](examples/genie_enterprise_research.yaml) | Genie-powered enterprise research |
| [`mixed_sources.yaml`](examples/mixed_sources.yaml) | Web + enterprise source combination |
| [`multi_source_research.yaml`](examples/multi_source_research.yaml) | Multiple source types in parallel |
| [`verified_enterprise_research.yaml`](examples/verified_enterprise_research.yaml) | Enterprise research with citation verification |

## Documentation

Full documentation at **[docs/index.md](docs/index.md)** (41 files).

| Section | Path | Contents |
|---------|------|----------|
| **Getting Started** | [`getting-started/`](docs/getting-started/) | Installation, quickstart, authentication |
| **Concepts** | [`concepts/`](docs/concepts/) | Workflow engine, agent harness, pools, events, citation pipeline |
| **Guides** | [`guides/`](docs/guides/) | YAML authoring, custom tools/agents, streaming, configuration |
| **Reference** | [`reference/`](docs/reference/) | API docs for builtins, event catalog, config schema |
| **Examples** | [`examples/`](docs/examples/) | Sample workflows, tool implementations, integration patterns |

### Reading Tracks

| Track | Time | Path |
|-------|------|------|
| **Quick Start** | ~15 min | Installation -> Quickstart -> Architecture |
| **Workflow Builder** | ~1-2 hours | Quick Start + YAML Authoring, Builtin Agents/Tools, Pools, Events |
| **Deep Dive** | ~half day | All concepts, custom tools/agents, citation pipeline, full reference |

## Part of the Deep Research Monorepo

This package is one of two in the [databricks-deep-research-agent](../) uv workspace:

| Package | Description |
|---------|-------------|
| **databricks-deep-research** (this) | Standalone orchestration framework |
| [**databricks-deep-research-app**](../databricks-deep-research-app/) | Production FastAPI + React app built on the framework |

See the [root README](../README.md) for monorepo setup and the [app documentation](../docs/README.md) for application-specific guides.

## Development

```bash
# Install with dev extras
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Type check
uv run mypy src/databricks_deep_research --strict

# Lint
uv run ruff check src/
```

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright &copy; 2026 the Databricks Deep Research Agent contributors.
