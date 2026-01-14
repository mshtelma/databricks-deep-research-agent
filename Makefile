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
# Databricks Apps Deployment:
#   make deploy TARGET=dev BRAVE_SCOPE=msh  - Full deployment (recommended)
#   make quickstart     - Set up local development environment
#   make db-migrate-remote  - Run migrations manually (usually not needed)
#   make requirements   - Generate requirements.txt from pyproject.toml
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
#   make logs TARGET=dev - Download app logs (add FOLLOW=-f to follow)

.PHONY: dev dev-backend dev-frontend build prod clean clean_db db-reset db-migrate-remote typecheck lint install e2e e2e-ui e2e-debug test test-unit test-integration test-complex test-all-python test-frontend test-all quickstart deploy requirements bundle-validate bundle-deploy bundle-deploy-full bundle-deploy-prod bundle-summary logs

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Stopping any existing servers..."
	@./scripts/kill-server.sh 8000 || true
	@./scripts/kill-server.sh 5173 || true
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
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Starting backend only with hot reload on :8000..."
	@echo "Logs: /tmp/deep-research-dev.log"
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-dev.log

dev-frontend:
	@echo "Stopping any existing server on port 5173..."
	@./scripts/kill-server.sh 5173 || true
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
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
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
	DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres \
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

# Run migrations on deployed Lakebase instance
# Usage: make db-migrate-remote TARGET=dev
#        make db-migrate-remote TARGET=ais
# Note: Requires databricks CLI configured and app deployed
# The Lakebase instance name follows the pattern: deep-research-lakebase-dre-<suffix>
# Profile mapping: dev -> e2-demo-west, ais -> ais (from databricks.yml targets)
db-migrate-remote:
	@echo "Running migrations on deployed Lakebase instance..."
	@echo "Target: $(TARGET)"
	@echo ""
	@INSTANCE_NAME="deep-research-lakebase"; \
	case "$(TARGET)" in \
		dev) PROFILE="e2-demo-west" ;; \
		ais) PROFILE="ais" ;; \
		*) echo "ERROR: Unknown target $(TARGET). Use 'dev' or 'ais'."; exit 1 ;; \
	esac; \
	echo "Lakebase instance: $$INSTANCE_NAME"; \
	echo "Databricks profile: $$PROFILE"; \
	echo ""; \
	echo "Bootstrapping database and running migrations..."; \
	DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
	LAKEBASE_INSTANCE_NAME="$$INSTANCE_NAME" \
	uv run alembic upgrade head
	@echo ""
	@echo "Migrations complete!"

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
# Databricks Apps Deployment
# =============================================================================

# Quickstart: Set up local development environment
quickstart:
	@./scripts/quickstart.sh

# Generate requirements.txt for Databricks Apps
requirements:
	@echo "Generating requirements.txt from pyproject.toml..."
	uv pip compile pyproject.toml -o requirements.txt
	@echo "requirements.txt updated"

# =============================================================================
# Databricks Asset Bundles (DAB) Deployment
# Use DAB for infrastructure-as-code with configurable secret scope/key
# =============================================================================

# Validate bundle configuration
bundle-validate:
	databricks bundle validate

# Full deployment - ONE COMMAND DOES EVERYTHING
# Builds frontend, deploys bundle, waits for DB, runs migrations, starts app
# Usage: make deploy TARGET=dev BRAVE_SCOPE=msh
#        make deploy TARGET=ais
#
# Deployment sequence:
#   1. Build frontend (npm run build)
#   2. Generate requirements.txt
#   3. Deploy bundle (creates Lakebase + app)
#   4. Wait for Lakebase to be ready (retry with backoff)
#   5. Run migrations with developer credentials
#   6. Start/restart app
#   7. Show deployment summary
#
# NOTE: Migrations are run with developer credentials (not app service principal)
# because the app's service principal has CAN_CONNECT_AND_CREATE permission
# but cannot create tables in the public schema.
TARGET ?= ais
BRAVE_SCOPE ?=
deploy: build requirements
	@echo "=============================================="
	@echo "Full Deployment Pipeline (Two-Phase)"
	@echo "Target: $(TARGET)"
	@echo "=============================================="
	@# Determine profile and instance name based on target
	@INSTANCE_NAME="deep-research-lakebase"; \
	case "$(TARGET)" in \
		dev) PROFILE="e2-demo-west" ;; \
		ais) PROFILE="ais" ;; \
		*) echo "ERROR: Unknown target $(TARGET). Use 'dev' or 'ais'."; exit 1 ;; \
	esac; \
	DEPLOY_ARGS="-t $(TARGET)"; \
	if [ -n "$(BRAVE_SCOPE)" ]; then \
		DEPLOY_ARGS="$$DEPLOY_ARGS --var brave_secret_scope=$(BRAVE_SCOPE)"; \
	fi; \
	\
	echo ""; \
	echo "Phase 1: Bootstrap Infrastructure"; \
	echo "=================================="; \
	echo ""; \
	echo "Step 1/8: Deploying bundle with postgres (bootstrap)..."; \
	echo "  Instance: $$INSTANCE_NAME"; \
	echo "  Profile: $$PROFILE"; \
	echo "  Note: Using postgres database for initial deploy"; \
	databricks bundle deploy $$DEPLOY_ARGS --var lakebase_database=postgres || { echo "ERROR: Bundle deploy failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 2/8: Waiting for Lakebase to be ready..."; \
	echo "  (New instances may take 30-60 seconds to be connectable)"; \
	./scripts/wait-for-lakebase.sh "$$INSTANCE_NAME" "$$PROFILE" 10 || { echo "ERROR: Lakebase not ready"; exit 1; }; \
	\
	echo ""; \
	echo "Step 3/8: Creating deep_research database..."; \
	./scripts/create-database.sh "$$INSTANCE_NAME" "$$PROFILE" deep_research || { echo "ERROR: Database creation failed"; exit 1; }; \
	\
	echo ""; \
	echo "Phase 2: Complete Deployment"; \
	echo "============================"; \
	echo ""; \
	echo "Step 4/8: Re-deploying bundle with deep_research..."; \
	databricks bundle deploy $$DEPLOY_ARGS --var lakebase_database=deep_research || { echo "ERROR: Bundle re-deploy failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 5/8: Running database migrations..."; \
	DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
	LAKEBASE_INSTANCE_NAME="$$INSTANCE_NAME" \
	LAKEBASE_DATABASE="deep_research" \
	uv run alembic upgrade head || { echo "ERROR: Migrations failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 6/8: Granting permissions to app service principal..."; \
	./scripts/grant-app-permissions.sh "$$INSTANCE_NAME" "$$PROFILE" "deep_research" "deep-research-agent-dre-$(TARGET)" || { echo "ERROR: Permission grant failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 7/8: Starting app..."; \
	databricks bundle run -t $(TARGET) deep_research_agent || { echo "ERROR: App start failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 8/8: Deployment summary..."; \
	databricks bundle summary -t $(TARGET)
	@echo ""
	@echo "=============================================="
	@echo "Deployment Complete!"
	@echo "=============================================="
	@echo ""
	@echo "Verify deployment:"
	@echo "  1. Visit the app URL shown above"
	@echo "  2. Check /health endpoint returns {\"status\": \"healthy\"}"
	@echo "  3. Create a new chat to verify database works"
	@echo ""
	@echo "If app fails to start, check logs:"
	@echo "  make logs TARGET=$(TARGET)"
	@echo "  make logs TARGET=$(TARGET) FOLLOW=-f  # Follow logs in real-time"

# Alias for backwards compatibility
bundle-deploy: deploy

# Alias for backwards compatibility (migrations now run automatically)
bundle-deploy-full: deploy

# Deploy to production
bundle-deploy-prod: build requirements
	@echo "Deploying to production..."
	databricks bundle deploy -t prod
	@echo "Starting app..."
	databricks bundle run -t prod deep_research_agent

# Show bundle deployment summary
bundle-summary:
	databricks bundle summary -t $(TARGET)

# Download logs from deployed Databricks App via /logz/batch REST API
# Note: Logs are not persisted when app compute shuts down
# Usage: make logs TARGET=dev                         # Fetch logs once
#        make logs TARGET=dev FOLLOW=-f               # Follow logs (poll every 5s)
#        make logs TARGET=dev SEARCH="--search ERROR" # Filter logs
#        make logs TARGET=dev FOLLOW=-f SEARCH="--search ERROR"  # Combine options
FOLLOW ?=
SEARCH ?=
logs:
	@case "$(TARGET)" in \
		dev) PROFILE="e2-demo-west" ;; \
		ais) PROFILE="ais" ;; \
		*) echo "ERROR: Unknown target $(TARGET). Use 'dev' or 'ais'."; exit 1 ;; \
	esac; \
	./scripts/download-app-logs.sh "deep-research-agent-dre-$(TARGET)" "$$PROFILE" $(FOLLOW) $(SEARCH)
