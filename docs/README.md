# Deep Research Agent Documentation

Production-grade multi-agent system for intelligent web research with claim-level citation verification, built on Databricks infrastructure.

## Overview

The Deep Research Agent combines a **5-agent orchestration architecture** with a **7-stage verification pipeline** grounded in peer-reviewed research (ARE, FActScore, SAFE, CoVe, ReClaim, QAFactEval).

### Core Innovations

- **5-Agent Architecture**: Coordinator → Planner → Researcher → Reflector → Synthesizer with step-by-step reflection
- **7-Stage Citation Pipeline**: Evidence pre-selection, interleaved generation, confidence classification, isolated verification, citation correction, numeric QA, and ARE-style revision
- **Tiered Query Modes**: Simple (direct LLM), Web Search (<15s), Deep Research (full pipeline)
- **Scientific Grounding**: Every factual claim traced to evidence with verification verdicts

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System architecture, technology stack, key design decisions |
| [Agent Orchestration](./agents.md) | 5-agent design, orchestration flow, state management |
| [Citation Pipeline](./citation-pipeline.md) | 7-stage verification pipeline with scientific foundations |
| [LLM Interaction](./llm-interaction.md) | Model tier routing, structured output, ReAct pattern |
| [Scientific Foundations](./scientific-foundations.md) | Research papers and implemented patterns |
| [Configuration](./configuration.md) | YAML configuration system and accessors |
| [Data Models](./data-models.md) | Entity definitions and relationships |
| [API Reference](./api.md) | REST endpoints and SSE event types |
| [Deployment](./deployment.md) | Databricks Apps deployment guide |

## Quick Start

```bash
# Development (runs backend + frontend)
make dev

# Production build
make build
make prod

# Run tests
make test              # Unit tests
make test-integration  # Integration tests (requires credentials)
make e2e               # End-to-end Playwright tests

# Deploy to Databricks
make deploy TARGET=dev BRAVE_SCOPE=msh
```

## Architecture Diagram

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

## Query Modes

| Mode | Latency | Description |
|------|---------|-------------|
| **Simple** | <3s | Direct LLM response without web research |
| **Web Search** | <15s | Quick web search with 2-5 citations |
| **Deep Research** | <2min | Full multi-agent pipeline with verification |

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Backend | Python | 3.11+ |
| Frontend | TypeScript/React | 18.x |
| Framework | FastAPI | 0.109+ |
| Database | Databricks Lakebase | Preview |
| LLM Client | AsyncOpenAI | 1.10+ |
| Observability | MLflow | 3.8+ |
| Search | Brave Search API | - |

## Key Files

| Path | Purpose |
|------|---------|
| `src/agent/orchestrator.py` | Main orchestration pipeline |
| `src/agent/nodes/` | 5 agent implementations |
| `src/services/citation/pipeline.py` | 7-stage citation verification |
| `config/app.yaml` | Central configuration |
| `specs/` | Feature specifications |

## Related Specifications

- [001-deep-research-agent](../specs/001-deep-research-agent/) - Core agent architecture
- [003-claim-level-citations](../specs/003-claim-level-citations/) - Citation verification
- [004-tiered-query-modes](../specs/004-tiered-query-modes/) - Query mode selection
