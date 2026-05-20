# Repository Guidelines

## Project Structure & Module Organization
Core backend code lives in `src/deep_research/`:
- `agent/` orchestration, nodes, prompts, and tools
- `api/v1/` FastAPI endpoints
- `services/`, `models/`, `schemas/`, `db/`, `plugins/`, `deployment/`

Frontend lives in `frontend/src/` (React + TypeScript), with domain components under `components/` and shared logic in `hooks/`, `services/`, and `types/`.  
Python tests are in `tests/unit`, `tests/integration`, and `tests/complex`.  
Browser E2E tests are in `e2e/tests` with Playwright page objects in `e2e/pages`.
Specifications live in `specs/<NNN-feature-name>/` and typically include `spec.md`, `plan.md`, `tasks.md`, `data-model.md`, `contracts/`, and `quickstart.md`.

## Build, Test, and Development Commands
- `make install`: install backend, frontend, and E2E dependencies.
- `make dev`: run backend (`:8000`) and frontend (`:5173`) with hot reload.
- `make build` / `make prod`: build frontend into `static/` and run unified server.
- `make test`: fast unit tests only.
- `make test-integration` / `make test-complex`: real-service tests (credentials required).
- `make test-all`: Python + frontend tests.
- `make e2e` (or `make e2e-fast`, `make e2e-medium`, `make e2e-slow`): Playwright suites.
- `make lint`, `make typecheck`, `make format`: quality gates.

## Coding Style & Naming Conventions
Python: 3.11+, async I/O patterns, full type annotations, Pydantic models for validation. Use `ruff` (line length 100) and `mypy --strict`.  
TypeScript: strict mode, ESLint enforced; React components use PascalCase filenames (for example `ChatSidebar.tsx`).  
General naming:
- Python modules/functions: `snake_case`
- Classes: `PascalCase`
- Tests: `test_*.py` and `*.spec.ts`

## Testing Guidelines
Use pytest markers and folders by scope (`unit`, `integration`, `complex`). Keep unit tests isolated and mock external services; reserve integration/complex for real Databricks/LLM/Brave flows.  
Coverage is collected (`pytest-cov`), but `fail_under` is currently `0`; contributors should still add regression tests for behavior changes.

## Commit & Pull Request Guidelines
Git history shows short, descriptive commit subjects (often feature-focused) and PR-based merges. Prefer clear, scoped messages (for example `api: enforce chat access control`) instead of generic `Bugfixing`.  
PRs should include:
- What changed and why
- Linked issue/spec (if available)
- Validation commands run (for example `make test`, `make lint`)
- UI screenshots/GIFs for frontend changes
- Spec updates for feature changes (`specs/.../tasks.md`, relevant contracts, and quickstart notes)

## Deployment & Configuration Tips
Primary deployment uses Databricks Asset Bundles (`databricks.yml`) and `make deploy TARGET=dev|ais [BRAVE_SCOPE=...]` with a two-phase Lakebase bootstrap/migrate flow.  
Use `.env` for local secrets (`DATABRICKS_*`, `LAKEBASE_*`, `BRAVE_API_KEY`); never commit secrets. Remote Lakebase is the default for local dev; `make db-local` is fallback only.
