# Installation

Install the `databricks-deep-research` framework for multi-agent workflow orchestration.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install from Source (current method - not yet on PyPI)

```bash
# Clone the monorepo
git clone <repo-url>
cd databricks-deep-research-agent/databricks-deep-research

# Install with uv (recommended)
uv pip install -e ".[all]"

# Or with pip
pip install -e ".[all]"
```

## Optional Extras

| Extra | Packages | Use Case |
|-------|----------|----------|
| `web` | `httpx>=0.24` | Web search tool (Brave API) |
| `crawl` | `trafilatura>=1.6` | Web page crawling and text extraction |
| `search` | `bm25s>=0.1`, `numpy>=1.24` | Pool hybrid BM25+vector search |
| `tracing` | `mlflow>=2.10` | MLflow `trace_span` observability |
| `all` | `web` + `crawl` + `search` + `tracing` | Full feature set |
| `dev` | `pytest>=8.0`, `pytest-asyncio>=0.23`, `mypy>=1.8`, `ruff>=0.2` | Development/testing |
| `integration` | `all` + `dev` + `databricks-sdk>=0.20` | Integration testing with Databricks |

## Verify Installation

```python
from databricks_deep_research import (
    WorkflowDefinition,
    WorkflowRunner,
    FrameworkLLMClient,
    load_workflow,
)
print("databricks-deep-research installed successfully!")
```

## Development Setup

```bash
# Install with dev extras
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/unit/ -v

# Type check
uv run mypy src/databricks_deep_research --strict

# Lint
uv run ruff check src/
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABRICKS_HOST` | For Databricks tools | Workspace URL |
| `DATABRICKS_TOKEN` | For Databricks tools | API token |
| `BRAVE_API_KEY` | For `web_search` | Brave Search API key |
| `OPENAI_API_KEY` | For standalone use | OpenAI API key (if not using Databricks) |

## See Also

- [Quick Start](quickstart.md)
- [Authentication](authentication.md)
- [pyproject.toml](../../pyproject.toml)
