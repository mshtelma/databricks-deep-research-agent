# Databricks Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()

A production-grade multi-agent research system with claim-level citation verification, built on Databricks infrastructure. Combines a 5-agent orchestration architecture with a 7-stage verification pipeline grounded in peer-reviewed research.

## Key Features

- **5-Agent Architecture** - Coordinator, Planner, Researcher, Reflector, and Synthesizer with step-by-step reflection
- **7-Stage Citation Pipeline** - Evidence pre-selection, interleaved generation, confidence classification, isolated verification, citation correction, numeric QA, and ARE-style revision
- **Tiered Query Modes** - Simple (<3s), Web Search (<15s), and Deep Research (<2min) for progressive disclosure
- **Scientific Grounding** - Every factual claim traced to evidence with verification verdicts based on ARE, FActScore, SAFE, CoVe, and ReClaim patterns
- **Real-time Streaming** - Server-Sent Events (SSE) for live research progress updates
- **Enterprise Ready** - OAuth token refresh, automatic failover, atomic persistence on Databricks Lakebase

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DEEP RESEARCH AGENT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Frontend  │    │                 5-Agent Orchestrator                │ │
│  │  React 18   │◀──▶│  ┌───────────┐ ┌─────────┐ ┌────────────┐          │ │
│  │  TanStack   │SSE │  │Coordinator│→│ Planner │→│ Researcher │          │ │
│  │  Tailwind   │    │  └───────────┘ └─────────┘ └────────────┘          │ │
│  └─────────────┘    │        ↓                         ↓                  │ │
│                     │  ┌───────────┐           ┌────────────┐             │ │
│  ┌─────────────┐    │  │ Reflector │◀──────────│ Synthesizer│             │ │
│  │  FastAPI    │    │  └───────────┘           └────────────┘             │ │
│  │  REST API   │◀──▶│        │                       │                    │ │
│  │  /v1/...    │    │        └───────────────────────┘                    │ │
│  └─────────────┘    │              7-Stage Citation Pipeline               │ │
│                     └─────────────────────────────────────────────────────┘ │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │ Lakebase    │◀──▶│     Databricks Foundation Model Endpoints           │ │
│  │ PostgreSQL  │    │  ┌─────────┐ ┌────────────┐ ┌───────────┐          │ │
│  │ Persistence │    │  │ Simple  │ │ Analytical │ │  Complex  │          │ │
│  └─────────────┘    │  │ (Gemini)│ │  (Claude)  │ │(Claude ER)│          │ │
│                     │  └─────────┘ └────────────┘ └───────────┘          │ │
│  ┌─────────────┐    └─────────────────────────────────────────────────────┘ │
│  │ Brave Search│                                                            │
│  │     API     │    ┌─────────────────────────────────────────────────────┐ │
│  └─────────────┘    │              MLflow Tracing (3.8+)                  │ │
│                     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend development)
- Databricks workspace with Foundation Model Endpoints
- Brave Search API key

### Local Development

```bash
# Install dependencies
make install

# Start development servers (backend + frontend)
make dev

# Access UI at http://localhost:5173
# API at http://localhost:8000
```

### Production Build

```bash
# Build frontend to static/
make build

# Run production server
make prod

# Access at http://localhost:8000
```

### Deploy to Databricks

```bash
# Deploy to dev workspace
make deploy TARGET=dev BRAVE_SCOPE=msh

# Deploy to production
make deploy TARGET=ais
```

## Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) folder:

| Document | Description |
|----------|-------------|
| [Architecture](./docs/architecture.md) | System design, technology stack, key decisions |
| [5-Agent System](./docs/agents.md) | Agent responsibilities, orchestration flow, state management |
| [Citation Pipeline](./docs/citation-pipeline.md) | 7-stage verification pipeline with scientific foundations |
| [LLM Interaction](./docs/llm-interaction.md) | Model tier routing, structured output, ReAct pattern |
| [Scientific Foundations](./docs/scientific-foundations.md) | Research papers and how they're applied |
| [Configuration](./docs/configuration.md) | YAML config system, environment variables |
| [Data Models](./docs/data-models.md) | Entity definitions and relationships |
| [API Reference](./docs/api.md) | REST endpoints and SSE event types |
| [Deployment](./docs/deployment.md) | Databricks Apps deployment guide |

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Backend | Python | 3.11+ |
| Frontend | TypeScript/React | 18.x |
| Framework | FastAPI | 0.109+ |
| Database | Databricks Lakebase | PostgreSQL |
| LLM Client | AsyncOpenAI | 1.10+ |
| Observability | MLflow | 3.8+ |
| Web Scraping | Trafilatura | 2.0+ |
| Search | Brave Search API | - |

## Configuration

All settings are centralized in `config/app.yaml`:

```yaml
# Model tier routing
models:
  simple:
    endpoints: [databricks-gemini-flash]
  analytical:
    endpoints: [databricks-claude-sonnet]
  complex:
    endpoints: [databricks-claude-sonnet-er]

# Research depth profiles
research_types:
  light:   { steps: {min: 1, max: 3}, researcher: {mode: classic} }
  medium:  { steps: {min: 3, max: 6}, researcher: {mode: react} }
  extended: { steps: {min: 5, max: 10}, researcher: {mode: react} }
```

See [Configuration Guide](./docs/configuration.md) for full details.

## Testing

```bash
# Unit tests (fast, no credentials)
make test

# Integration tests (real LLM/Brave)
make test-integration

# Complex long-running tests
make test-complex

# E2E Playwright tests
make e2e

# All tests
make test-all
```

## Scientific Foundations

The citation verification pipeline implements patterns from peer-reviewed research:

| Pattern | Paper | Application |
|---------|-------|-------------|
| **ReClaim** | [arXiv:2407.01796](https://arxiv.org/abs/2407.01796) | Interleaved generation with evidence constraints |
| **FActScore** | [arXiv:2305.14251](https://arxiv.org/abs/2305.14251) | Atomic fact decomposition |
| **SAFE** | [arXiv:2403.18802](https://arxiv.org/abs/2403.18802) | Multi-step reasoning with search |
| **ARE** | [arXiv:2410.16708](https://arxiv.org/abs/2410.16708) | Atomic facts for retrieval |
| **CoVe** | [arXiv:2309.11495](https://arxiv.org/abs/2309.11495) | Isolated verification |
| **CiteFix** | [arXiv:2504.15629](https://arxiv.org/abs/2504.15629) | Hybrid citation correction |
| **QAFactEval** | [arXiv:2112.08542](https://arxiv.org/abs/2112.08542) | QA-based numeric verification |

See [Scientific Foundations](./docs/scientific-foundations.md) for detailed explanations.

## Project Structure

```
src/                            # Python backend
├── agent/                      # 5-agent orchestration
│   ├── orchestrator.py         # Main async pipeline
│   ├── nodes/                  # Agent implementations
│   └── prompts/                # Agent prompts
├── api/                        # FastAPI routes
├── services/
│   ├── llm/                    # LLM client with tier routing
│   ├── citation/               # 7-stage verification pipeline
│   └── search/                 # Brave Search API
└── db/                         # Lakebase persistence

frontend/                       # React frontend
├── src/
│   ├── components/             # UI components
│   ├── hooks/                  # React hooks (streaming, etc.)
│   └── pages/                  # Page components
└── tests/                      # Vitest tests

e2e/                            # Playwright E2E tests
tests/                          # Python backend tests
config/                         # YAML configuration
docs/                           # Documentation
```

## Contributing

1. Follow the guidelines in [CLAUDE.md](./CLAUDE.md)
2. Ensure all tests pass: `make test-all`
3. Type check: `make typecheck`
4. Lint: `make lint`

## License

Proprietary
