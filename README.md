# Databricks Deep Research Agent

Deep research agent with 5-agent architecture (Coordinator, Planner, Researcher, Reflector, Synthesizer) featuring step-by-step reflection, web search via Brave API, and streaming chat UI with persistence on Databricks Lakebase.

## Architecture

- **Backend**: Python 3.11+, FastAPI, async orchestration
- **Frontend**: TypeScript, React 18, TanStack Query
- **Database**: Databricks Lakebase (PostgreSQL)
- **Observability**: MLflow

## Development

```bash
# Two terminals for development
make dev-backend    # Backend with hot reload (:8000)
make dev-frontend   # Frontend with hot reload (:5173)

# Production build
make build          # Build frontend to backend/static/
make prod           # Run unified server on :8000

# Quality checks
make typecheck      # Type check backend + frontend
make lint           # Lint backend + frontend
make test           # Run all tests
```

## License

Proprietary
