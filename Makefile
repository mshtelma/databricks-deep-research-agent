# Deep Research Agent - Build and Development Commands
#
# Development:
#   make dev            - Run backend with hot reload
#   make dev-frontend   - Run frontend with hot reload (Vite)
#
# Production:
#   make build          - Build frontend to static/
#   make prod           - Build and run unified server
#
# Testing:
#   make test           - Run unit tests only (fast, no credentials)
#   make test-integration - Run integration tests (requires credentials)
#   make test-complex   - Run complex long-running tests (requires credentials)
#   make test-all       - Run all tests (Python + Frontend)
#
# E2E Testing:
#   make e2e            - Build + run E2E tests (auto-starts server)
#   make e2e-ui         - Run E2E tests with Playwright UI
#
# Utilities:
#   make clean          - Remove build artifacts
#   make clean_db       - Delete all chats/messages from database
#   make db-reset       - Reset database schema (downgrade + upgrade)
#   make typecheck      - Run type checking (backend + frontend)
#   make lint           - Run linting (backend + frontend)

.PHONY: dev dev-backend dev-frontend build prod clean clean_db db-reset typecheck lint install e2e e2e-ui e2e-debug test test-unit test-integration test-complex test-all-python test-frontend test-all

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Starting backend (:8000) and frontend (:5173)..."
	@echo "Backend logs: /tmp/deep-research-dev.log"
	@echo "Access UI at: http://localhost:5173"
	@echo "Press Ctrl+C to stop both servers"
	@bash -c '\
		cleanup() { \
			echo ""; \
			echo "Stopping servers..."; \
			lsof -ti:8000 | xargs kill -9 2>/dev/null || true; \
			lsof -ti:5173 | xargs kill -9 2>/dev/null || true; \
			echo "Servers stopped."; \
		}; \
		trap cleanup EXIT INT TERM; \
		(uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-dev.log) & \
		(cd frontend && npm run dev) & \
		wait'

dev-backend:
	@echo "Starting backend only with hot reload on :8000..."
	@echo "Logs: /tmp/deep-research-dev.log"
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-dev.log

dev-frontend:
	@echo "Starting frontend with hot reload on :5173..."
	cd frontend && npm run dev

dev-help:
	@echo "Development commands:"
	@echo "  make dev           - Run both backend and frontend (recommended)"
	@echo "  make dev-backend   - Run backend only (:8000)"
	@echo "  make dev-frontend  - Run frontend only (:5173)"
	@echo ""
	@echo "Access UI at: http://localhost:5173"

# =============================================================================
# Production Build
# =============================================================================

build:
	@echo "Building frontend..."
	cd frontend && npm ci && npm run build
	@echo ""
	@echo "Frontend built to static/"
	@ls -la static/

prod: build
	@echo "Starting production server on :8000..."
	@echo "Logs: /tmp/deep-research-prod.log"
	SERVE_STATIC=true uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-prod.log

# =============================================================================
# Installation
# =============================================================================

install:
	@echo "Installing backend dependencies..."
	uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && npm ci
	@echo "Installing E2E dependencies..."
	cd e2e && npm ci && npx playwright install
	@echo "Done!"

install-dev:
	@echo "Installing backend dependencies (dev)..."
	uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Installing E2E dependencies..."
	cd e2e && npm install && npx playwright install
	@echo "Done!"

# =============================================================================
# Quality Checks
# =============================================================================

typecheck:
	@echo "Type checking backend..."
	uv run mypy src --strict
	@echo "Type checking frontend..."
	cd frontend && npm run typecheck

lint:
	@echo "Linting backend..."
	uv run ruff check src
	@echo "Linting frontend..."
	cd frontend && npm run lint

format:
	@echo "Formatting backend..."
	uv run ruff format src
	@echo "Formatting frontend..."
	cd frontend && npm run format 2>/dev/null || true

# =============================================================================
# Testing
# =============================================================================

# Unit tests (fast, mocked, no credentials needed)
test:
	uv run pytest tests/unit -v

test-unit: test

# Integration tests (real LLM/Brave, test config, requires credentials)
test-integration:
	uv run pytest tests/integration -v --tb=short

# Complex tests (long-running, production config, requires credentials)
test-complex:
	uv run pytest tests/complex -v --tb=short --timeout=600

# All Python tests
test-all-python:
	uv run pytest tests -v --tb=short

# Frontend tests
test-frontend:
	cd frontend && npm run test

# All tests (Python + Frontend)
test-all: test-all-python test-frontend

# =============================================================================
# Database (local development via Docker)
# =============================================================================

db:
	@echo "Starting PostgreSQL via docker compose..."
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@echo "Running migrations..."
	DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/deep_research \
		LAKEBASE_INSTANCE_NAME= \
		uv run alembic upgrade head || echo "Note: Run migrations manually if needed"

db-stop:
	docker compose down

clean_db:
	@echo "Cleaning all chats, messages, and research data from database..."
	uv run ./scripts/clean-db.sh
	@echo "Done!"

db-reset:
	@echo "Resetting database schema (drops all tables and recreates)..."
	@echo "This will delete ALL data. Use clean_db to preserve schema."
	uv run alembic downgrade base
	uv run alembic upgrade head
	@echo "Database schema reset complete!"

# =============================================================================
# E2E Testing with Playwright
# Auto-starts server if not running (webServer in playwright.config.ts)
# Requires: docker compose up -d postgres (or run 'make db' first)
# =============================================================================

e2e:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend (ensures fresh static files)..."
	@make build
	@echo "Running E2E tests with Playwright..."
	@echo "Note: Requires PostgreSQL running (make db)"
	cd e2e && npm test

e2e-ui:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend (ensures fresh static files)..."
	@make build
	@echo "Opening Playwright UI..."
	cd e2e && npm run test:ui

e2e-debug:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend (ensures fresh static files)..."
	@make build
	@echo "Running E2E tests in debug mode..."
	cd e2e && npm run test:debug

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf static/
	rm -rf frontend/dist/
	rm -rf frontend/node_modules/.vite/
	rm -rf e2e/test-results/
	rm -rf e2e/playwright-report/
	@echo "Done!"

clean-all: clean
	rm -rf frontend/node_modules/
	rm -rf e2e/node_modules/
	rm -rf .venv/

# =============================================================================
# Databricks Deployment
# =============================================================================

deploy: build
	@echo "Deploying to Databricks Apps..."
	databricks apps deploy .
