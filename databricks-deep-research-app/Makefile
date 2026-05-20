# Deep Research Agent - Build and Development Commands
# (App-level Makefile — lives inside databricks-deep-research-app/)
#
# Development:
#   make dev            - Run backend + frontend (sources ../.env + ../.env.local-dev)
#   make dev-frontend   - Run frontend with hot reload (Vite)
#   make dev DEV_TARGET=dev  - Run locally against deployed dev project
#
# Production:
#   make build          - Build frontend to static/
#   make prod           - Build and run unified server
#
# Databricks Apps Deployment:
#   make deploy TARGET=dev BRAVE_SCOPE=msh  - Full deployment (recommended)
#   make app-deploy TARGET=dev              - Fast app-only redeploy (Python/yaml/vars,
#                                             no DB migrate, no provisioning, no grants)
#   make app-deploy TARGET=dev BUILD=1      - Also rebuild frontend + requirements.txt
#   make quickstart     - Set up local development environment
#   make db-migrate     - Run migrations on configured Lakebase (local dev)
#   make db-migrate-remote TARGET=dev  - Run migrations on deployed instance
#   make requirements   - Generate requirements.txt from pyproject.toml
#
# Testing:
#   make test           - Run unit tests only (fast, no credentials)
#   make test-integration - Run integration tests (requires credentials)
#   make test-complex   - Run complex long-running tests (requires credentials)
#   make test-all       - Run all tests (Python + Frontend)
#
# E2E Testing:
#   make e2e            - Build + run E2E tests (sources ../.env + ../.env.e2e)
#   make e2e-ui         - Run E2E tests with Playwright UI
#
# Database:
#   make db-provision TARGET=local-dev  -> full Autoscaling setup, writes ../.env.local-dev
#   make db-provision TARGET=e2e        -> full Autoscaling setup, writes ../.env.e2e
#   make db-cleanup TARGET=dev          -> remove orphaned resources from interrupted deploys
#   make db-migrate     -> uses ../.env default (deep_research)
#   make db-migrate DB_SUFFIX=dev  -> uses deep_research_dev
#
# Env file layout (all env files live in the parent/root directory):
#   ../.env              <- shared secrets (BRAVE_API_KEY, etc.) -- NOT touched by provisioning
#   ../.env.local-dev    <- written by: make db-provision TARGET=local-dev
#   ../.env.e2e          <- written by: make db-provision TARGET=e2e
#   ../.env.dev          <- written by: make deploy TARGET=dev
#   ../.env.ais          <- written by: make deploy TARGET=ais
#
# Utilities:
#   make clean          - Remove build artifacts
#   make clean_db       - Delete all chats/messages from database
#   make db-reset       - Reset database schema (downgrade + upgrade)
#   make typecheck      - Run type checking (backend + frontend)
#   make lint           - Run linting (backend + frontend)
#   make logs TARGET=dev - Download app logs (add FOLLOW=-f to follow)

.PHONY: dev dev-backend dev-frontend build build-framework prod clean clean_db db-provision db-cleanup db-migrate db-status db-reset db-migrate-remote db-local db-local-stop clean-e2e typecheck lint install e2e e2e-ui e2e-debug e2e-custom-agents test test-integration test-complex test-scaffold-and-run test-all-python test-frontend test-all quickstart deploy app-deploy requirements bundle-validate bundle-summary logs

# =============================================================================
# Target-bound env loading
# =============================================================================
# Each workflow binds to a target whose ../.env.{target} provides Lakebase vars.
# Override to run locally against a different project's database.
DEV_TARGET ?= local-dev
E2E_TARGET ?= e2e

# Root directory where .env files live (parent of this Makefile's directory)
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

# Helper: source ../.env then ../.env.{target} (target vars override shared ones)
# Usage in recipes: $(call load-env,$(DEV_TARGET)) && <command>
define load-env
set -a && [ -f $(ROOT_DIR)/.env ] && . $(ROOT_DIR)/.env || true && [ -f $(ROOT_DIR)/.env.$(1) ] && . $(ROOT_DIR)/.env.$(1) || true && set +a
endef

# =============================================================================
# Database Isolation Variables
# =============================================================================
DB_SUFFIX ?=
DB_BASE ?= deep_research
DB_NAME = $(if $(DB_SUFFIX),$(DB_BASE)_$(DB_SUFFIX),$(DB_BASE))
NPM_CI_FLAGS ?= --prefer-offline --no-audit --no-fund

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Stopping any existing servers..."
	@./scripts/kill-server.sh 8000 || true
	@./scripts/kill-server.sh 5173 || true
	@echo "Starting backend (:8000) and frontend (:5173)..."
	@echo "Target: $(DEV_TARGET) (env: ../.env + ../.env.$(DEV_TARGET))"
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
		($(call load-env,$(DEV_TARGET)) && uv run uvicorn deep_research.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-dev.log) & \
		(cd frontend && npm run dev) & \
		wait'

dev-backend:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Starting backend only with hot reload on :8000..."
	@echo "Target: $(DEV_TARGET) (env: ../.env + ../.env.$(DEV_TARGET))"
	@echo "Logs: /tmp/deep-research-dev.log"
	@bash -c '$(call load-env,$(DEV_TARGET)) && uv run uvicorn deep_research.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-dev.log'

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

build-framework:
	@echo "Building framework wheel..."
	@rm -f wheels/databricks_deep_research-*.whl
	cd ../databricks-deep-research && uv build --wheel --out-dir ../databricks-deep-research-app/wheels/
	@echo "Framework wheel built to wheels/"

build: build-framework
	@echo "Building frontend..."
	cd frontend && npm ci $(NPM_CI_FLAGS) && npm run build
	@echo ""
	@echo "Frontend built to static/"
	@ls -la static/
	@echo ""
	@echo "Creating symlink for local development..."
	@rm -f src/deep_research/static
	@ln -s ../../static src/deep_research/static
	@echo "Symlink created: src/deep_research/static -> ../../static"

# Build a wheel and verify it contains every package-data file the deployment
# translators (shell-app + spark-batch) expect at runtime. Catches the
# packaging regression class that broke AIS on 2026-05-12.
verify-wheel:
	@echo "Building wheel..."
	@rm -rf dist/
	uv build --wheel --out-dir dist
	@echo "Verifying wheel content..."
	uv run python scripts/verify_wheel.py

prod: build
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Starting production server on :8000..."
	@echo "Target: $(DEV_TARGET) (env: ../.env + ../.env.$(DEV_TARGET))"
	@echo "Logs: /tmp/deep-research-prod.log"
	@bash -c '$(call load-env,$(DEV_TARGET)) && SERVE_STATIC=true uv run uvicorn deep_research.main:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/deep-research-prod.log'

# =============================================================================
# Installation
# =============================================================================

install:
	@echo "Installing backend dependencies..."
	uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && npm ci $(NPM_CI_FLAGS)
	@echo "Installing E2E dependencies..."
	cd e2e && npm ci $(NPM_CI_FLAGS) && npx playwright install
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
	uv run mypy src/deep_research --strict
	@echo "Type checking frontend..."
	cd frontend && npm run typecheck

lint:
	@echo "Linting backend..."
	uv run ruff check src/deep_research
	@echo "Linting frontend..."
	cd frontend && npm run lint

format:
	@echo "Formatting backend..."
	uv run ruff format src/deep_research
	@echo "Formatting frontend..."
	cd frontend && npm run format 2>/dev/null || true

# =============================================================================
# Testing
# =============================================================================

# Unit tests (fast, mocked, no credentials needed)
test:
	uv run pytest tests/unit -v

# Integration tests (real LLM/Brave, test config, requires credentials)
test-integration:
	uv run pytest tests/integration -v --tb=short

# Complex tests (long-running, production config, requires credentials)
test-complex:
	uv run pytest tests/complex -v --tb=short --timeout=600

# Scaffold-and-Run live integration test — designer + WorkflowRunner end-to-end
# Writes per-case artifact dirs under tests/_runs/<ts>-<case>/ for analysis.
# Sources ../.env + ../.env.$(DEV_TARGET) automatically (same creds as `make dev`),
# so BRAVE_API_KEY and DATABRICKS_CONFIG_PROFILE / DATABRICKS_TOKEN are picked up
# without the caller having to export anything.
#
# Uses the DEFAULT config/app.yaml (not app.test.yaml) so the designer's
# analytical tier gets enough max_tokens for Claude extended thinking
# (CLAUDE_THINKING_BUDGETS[medium]=4096 requires max_tokens > 4096; app.test.yaml
# caps at 4000 and the API rejects).  Output artifacts are also more
# comparable to production this way.
#
# Usage: make test-scaffold-and-run                              (both cases)
#        make test-scaffold-and-run CASE=investment_research     (one case)
#        make test-scaffold-and-run DEV_TARGET=e2e               (different env file)
CASE ?=
test-scaffold-and-run:
	@echo "Target: $(DEV_TARGET) (env: ../.env + ../.env.$(DEV_TARGET))"
	@bash -c '$(call load-env,$(DEV_TARGET)) && \
		MLFLOW_ENABLED=false \
		uv run pytest tests/complex/test_scaffold_and_run.py -v -s --tb=short --timeout=1800 \
			$(if $(CASE),-k $(CASE),)'

# All Python tests
test-all-python:
	uv run pytest tests -v --tb=short

# Storage substrate unit + contract tests (fast, no credentials required)
test-storage:
	uv run pytest tests/unit/storage tests/contract/storage tests/services \
		tests/integration/storage/test_migration_cache_race.py \
		tests/unit/test_enricher_ordering.py tests/unit/test_lifespan_wiring.py \
		-v --tb=short

# AIS Lakebase integration test. Requires `../.env.ais` with DATABRICKS_CONFIG_PROFILE,
# PGHOST, LAKEBASE_DATABASE. Provisions a disposable temp schema per run.
#
# Run with:  make test-storage-ais-lakebase
test-storage-ais-lakebase:
	@test -f ../.env.ais || (echo "ERROR: ../.env.ais not found — see docs/storage/ais_setup.md"; exit 1)
	set -a && . ../.env.ais && set +a && \
		STORAGE_TEST_LAKEBASE=1 STORAGE_SERVICE_IMPL=cached STORAGE_BACKEND=lakebase \
		uv run pytest tests/integration/storage/test_lakebase_roundtrip.py -v --tb=short

# AIS SQL Warehouse integration test. Requires DATABRICKS_CONFIG_PROFILE +
# STORAGE_WAREHOUSE_ID + STORAGE_CATALOG in the loaded env.
test-storage-ais-warehouse:
	@test -f ../.env.ais || (echo "ERROR: ../.env.ais not found — see docs/storage/ais_setup.md"; exit 1)
	set -a && . ../.env.ais && set +a && \
		STORAGE_TEST_WAREHOUSE=1 STORAGE_SERVICE_IMPL=cached STORAGE_BACKEND=sql_warehouse \
		uv run pytest tests/integration/storage/test_warehouse_roundtrip.py -v --tb=short

# Frontend tests
test-frontend:
	cd frontend && npx vitest run --passWithNoTests

# All tests (Python + Frontend)
test-all: test-all-python test-frontend

# =============================================================================
# Database Infrastructure (one-time)
# =============================================================================

# Clean up orphaned Lakebase resources for a target
# Use when db-provision fails due to stale cloud resources from interrupted deploys
# (e.g., "workspace limit" or "branch already exists" errors)
# Usage: make db-cleanup TARGET=local-dev
#        make db-cleanup TARGET=dev
db-cleanup:
	@echo "=============================================="
	@echo "Cleaning orphaned Lakebase resources"
	@echo "Target: $(TARGET)"
	@echo "=============================================="
	@PROFILE=$$(uv run python3 -c "import yaml,sys;cfg=yaml.safe_load(open('databricks.yml'));t=cfg.get('targets',{}).get('$(TARGET)');sys.exit(1) if not t else print(t['workspace']['profile'])" 2>/dev/null) || \
		{ echo "ERROR: Unknown target '$(TARGET)'. Valid targets:"; uv run python3 -c "import yaml;[print(f'  {k}') for k in yaml.safe_load(open('databricks.yml'))['targets']]"; exit 1; }; \
	SUFFIX=$$(uv run python3 -c "import yaml;cfg=yaml.safe_load(open('databricks.yml'));print(cfg['targets']['$(TARGET)']['variables']['resource_suffix'])" 2>/dev/null) || \
		{ echo "ERROR: Could not extract resource_suffix for target '$(TARGET)'"; exit 1; }; \
	PROJECT_ID="deep-research-$$SUFFIX"; \
	echo "  Profile: $$PROFILE"; \
	echo "  Autoscaling project: $$PROJECT_ID"; \
	echo ""; \
	uv run python3 scripts/db-cleanup.py "$$PROFILE" "$$PROJECT_ID"; \
	echo ""; \
	echo "Removing stale Terraform state..."; \
	rm -f .databricks/bundle/$(TARGET)/terraform/terraform.tfstate \
	      .databricks/bundle/$(TARGET)/terraform/terraform.tfstate.backup 2>/dev/null; \
	echo "  State files removed."; \
	echo ""; \
	echo "Cleaning stale bundle state..."; \
	databricks bundle destroy -t $(TARGET) --auto-approve 2>&1 || true; \
	echo ""; \
	echo "Cleanup complete. You can now re-run: make db-provision TARGET=$(TARGET)"

# Provision Lakebase infrastructure + bootstrap dev/e2e databases
# Two-phase bootstrap: deploy with postgres -> create deep_research DB -> re-deploy
# Then: wait for Autoscaling -> discover PGHOST -> bootstrap dev/e2e databases
# Usage: make db-provision TARGET=dev
#        make db-provision TARGET=local-dev
db-provision:
	@uv run python -m deep_research.cli.provision --target $(TARGET)

# =============================================================================
# Database Management (ongoing)
# =============================================================================

# Run migrations using ../.env configuration (remote Lakebase)
# Use DB_SUFFIX to target a specific database:
#   make db-migrate              -> targets deep_research (from ../.env)
#   make db-migrate DB_SUFFIX=dev  -> targets deep_research_dev
#   make db-migrate DB_SUFFIX=e2e  -> targets deep_research_e2e
db-migrate:
	@echo "Running migrations on configured Lakebase instance..."
	@echo "Database: $(DB_NAME)"
	@if [ -n "$(TARGET)" ]; then echo "Target: $(TARGET) (env: ../.env + ../.env.$(TARGET))"; fi
	@bash -c '$(call load-env,$(or $(TARGET),$(DEV_TARGET))) && LAKEBASE_DATABASE=$(DB_NAME) uv run alembic upgrade head'
	@echo "Migrations complete!"

# Check current migration status
db-status:
	@echo "Checking migration status..."
	uv run alembic current


# =============================================================================
# Database (Local PostgreSQL - Fallback Only)
# =============================================================================
# NOTE: Local PostgreSQL is for offline/fallback scenarios only.
# Standard local development uses remote Lakebase configured via ../.env

db-local:
	@echo "Starting local PostgreSQL via docker compose (fallback mode)..."
	@echo "NOTE: Standard local dev uses remote Lakebase. This is for offline use only."
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@echo "Running migrations against local PostgreSQL..."
	DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres \
		LAKEBASE_INSTANCE_NAME= \
		uv run alembic upgrade head || echo "Note: Run migrations manually if needed"

db-local-stop:
	docker compose down

clean_db:
	@echo "Cleaning all chats, messages, and research data from database..."
	@echo "Database: $(DB_NAME)"
	LAKEBASE_DATABASE=$(DB_NAME) uv run ./scripts/clean-db.sh
	@echo "Done!"

# Clean E2E database specifically
clean-e2e:
	@echo "Cleaning E2E database..."
	bash -c '$(call load-env,$(E2E_TARGET)) && uv run ./scripts/clean-db.sh'
	@echo "Done!"

# Reset database schema (local or remote based on TARGET)
# Usage: make db-reset                    # Uses local ../.env config
#        make db-reset TARGET=dev         # Resets deployed Lakebase (auto-detects backend)
#        make db-reset TARGET=ais         # Resets ais Lakebase
db-reset:
	@echo "Resetting database schema (drops all tables and recreates)..."
	@echo "This will delete ALL data. Use clean_db to preserve schema."
	@if [ -z "$(TARGET)" ]; then \
		echo "Using local environment configuration..."; \
		uv run python -c "import asyncio; from deep_research.db.bootstrap import drop_all_tables; asyncio.run(drop_all_tables())" && \
		uv run alembic upgrade head; \
	else \
		PROFILE=$$(uv run python3 -c "import yaml,sys;cfg=yaml.safe_load(open('databricks.yml'));t=cfg.get('targets',{}).get('$(TARGET)');sys.exit(1) if not t else print(t['workspace']['profile'])" 2>/dev/null) || \
			{ echo "ERROR: Unknown target '$(TARGET)'. Valid targets:"; uv run python3 -c "import yaml;[print(f'  {k}') for k in yaml.safe_load(open('databricks.yml'))['targets']]"; exit 1; }; \
		echo "Target: $(TARGET)"; \
		echo "Databricks profile: $$PROFILE"; \
		echo ""; \
		echo "Detecting backend..."; \
		RESOURCE_SUFFIX=$$(uv run python3 -c "import yaml;cfg=yaml.safe_load(open('databricks.yml'));print(cfg['targets']['$(TARGET)']['variables']['resource_suffix'])" 2>/dev/null); \
		PROJECT_ID="deep-research-$$RESOURCE_SUFFIX"; \
		ENDPOINT_NAME=$$(DATABRICKS_CONFIG_PROFILE="$$PROFILE" uv run python -c \
			"from deep_research.deployment import discover_autoscaling_endpoint; print(discover_autoscaling_endpoint('$$PROJECT_ID'))" 2>/dev/null); \
		if [ -z "$$ENDPOINT_NAME" ] && [ -f $(ROOT_DIR)/.env.$(TARGET) ]; then \
			echo "  (falling back to $(ROOT_DIR)/.env.$(TARGET) for ENDPOINT_NAME + PGHOST)"; \
			ENDPOINT_NAME=$$(grep -E '^ENDPOINT_NAME=' $(ROOT_DIR)/.env.$(TARGET) | cut -d= -f2- | tr -d '"'); \
			ENV_PGHOST=$$(grep -E '^PGHOST=' $(ROOT_DIR)/.env.$(TARGET) | cut -d= -f2- | tr -d '"'); \
		fi; \
		if [ -n "$$ENDPOINT_NAME" ]; then \
			echo "Backend: Autoscaling (ENDPOINT_NAME=$$ENDPOINT_NAME)"; \
			echo "  Resolving PGHOST..."; \
			PGHOST=$$(DATABRICKS_CONFIG_PROFILE="$$PROFILE" uv run python -c \
				"from databricks.sdk import WorkspaceClient; w=WorkspaceClient(); ep=w.postgres.get_endpoint(name='$$ENDPOINT_NAME'); print(ep.status.hosts.host if ep.status and ep.status.hosts else '')" 2>/dev/null || echo ""); \
			if [ -z "$$PGHOST" ] && [ -n "$$ENV_PGHOST" ]; then \
				PGHOST="$$ENV_PGHOST"; \
				echo "  (PGHOST from .env.$(TARGET))"; \
			fi; \
			echo "  PGHOST: $$PGHOST"; \
			DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
			ENDPOINT_NAME="$$ENDPOINT_NAME" \
			PGHOST="$$PGHOST" \
			LAKEBASE_INSTANCE_NAME="" \
			LAKEBASE_DATABASE="deep_research" \
			uv run python -c "import asyncio; from deep_research.db.bootstrap import drop_all_tables; asyncio.run(drop_all_tables())" && \
			DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
			ENDPOINT_NAME="$$ENDPOINT_NAME" \
			PGHOST="$$PGHOST" \
			LAKEBASE_INSTANCE_NAME="" \
			LAKEBASE_DATABASE="deep_research" \
			uv run alembic upgrade head; \
		else \
			INSTANCE_NAME=$$(databricks bundle summary -t $(TARGET) --output json 2>/dev/null | jq -r '.resources.database_instances.deep_research_lakebase.name // empty'); \
			echo "Backend: Provisioned (LAKEBASE_INSTANCE_NAME=$$INSTANCE_NAME)"; \
			DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
			LAKEBASE_INSTANCE_NAME="$$INSTANCE_NAME" \
			LAKEBASE_DATABASE="deep_research" \
			uv run python -c "import asyncio; from deep_research.db.bootstrap import drop_all_tables; asyncio.run(drop_all_tables())" && \
			DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
			LAKEBASE_INSTANCE_NAME="$$INSTANCE_NAME" \
			LAKEBASE_DATABASE="deep_research" \
			uv run alembic upgrade head; \
		fi; \
	fi
	@echo "Database schema reset complete!"
	@echo "Note: storage schema (deep_research_state) is created by the app"
	@echo "      service principal at lifespan via LakebaseBackend.migrate()."
	@echo "      Do NOT pre-create it from here — the developer cannot transfer"
	@echo "      ownership cross-principal on Lakebase, and a developer-owned"
	@echo "      schema causes 'must be owner of table' at app startup."

# Run migrations on deployed Lakebase instance (auto-detects Provisioned vs Autoscaling)
# Usage: make db-migrate-remote TARGET=dev
#        make db-migrate-remote TARGET=ais
# Note: Requires databricks CLI configured and app deployed
# Profile is auto-detected from databricks.yml targets
db-migrate-remote:
	@echo "Running migrations on deployed Lakebase instance..."
	@echo "Target: $(TARGET)"
	@echo ""
	@PROFILE=$$(uv run python3 -c "import yaml,sys;cfg=yaml.safe_load(open('databricks.yml'));t=cfg.get('targets',{}).get('$(TARGET)');sys.exit(1) if not t else print(t['workspace']['profile'])" 2>/dev/null) || \
		{ echo "ERROR: Unknown target '$(TARGET)'. Valid targets:"; uv run python3 -c "import yaml;[print(f'  {k}') for k in yaml.safe_load(open('databricks.yml'))['targets']]"; exit 1; }; \
	echo "Databricks profile: $$PROFILE"; \
	echo ""; \
	echo "Detecting backend..."; \
	RESOURCE_SUFFIX=$$(uv run python3 -c "import yaml;cfg=yaml.safe_load(open('databricks.yml'));print(cfg['targets']['$(TARGET)']['variables']['resource_suffix'])" 2>/dev/null); \
	PROJECT_ID="deep-research-$$RESOURCE_SUFFIX"; \
	ENDPOINT_NAME=$$(DATABRICKS_CONFIG_PROFILE="$$PROFILE" uv run python -c \
		"from deep_research.deployment import discover_autoscaling_endpoint; print(discover_autoscaling_endpoint('$$PROJECT_ID'))" 2>/dev/null); \
	if [ -n "$$ENDPOINT_NAME" ]; then \
		echo "Backend: Autoscaling (ENDPOINT_NAME=$$ENDPOINT_NAME)"; \
		echo "Running migrations..."; \
		DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
		ENDPOINT_NAME="$$ENDPOINT_NAME" \
		LAKEBASE_INSTANCE_NAME="" \
		uv run alembic upgrade head; \
	else \
		INSTANCE_NAME=$$(databricks bundle summary -t $(TARGET) --output json 2>/dev/null | jq -r '.resources.database_instances.deep_research_lakebase.name // empty'); \
		echo "Backend: Provisioned (LAKEBASE_INSTANCE_NAME=$$INSTANCE_NAME)"; \
		echo "Running migrations..."; \
		DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
		LAKEBASE_INSTANCE_NAME="$$INSTANCE_NAME" \
		uv run alembic upgrade head; \
	fi
	@echo ""
	@echo "Migrations complete!"
	@echo "Note: storage schema (deep_research_state) is created by the app"
	@echo "      service principal at lifespan, not by this target."

# =============================================================================
# E2E Testing with Playwright
# Auto-starts server if not running (webServer in playwright.config.ts)
# Sources ../.env + ../.env.{E2E_TARGET} for database isolation from dev data
# =============================================================================

# E2E Test Categories:
#   fast       - Basic UI tests, no research (~1 min): smoke, chat-management
#   medium     - Light research operations (2-5 min): edit-message, follow-up, regenerate, stop-cancel, error-handling, provenance-export
#   slow       - Full research with verification (10 min): citations, grey-references, numeric-claims, research-flow, verification-summary
#   super-slow - Multiple parallel research sessions (15 min): parallel-research
#   custom-agents - Custom agent CRUD, UI, and research (CRUD ~30s, UI ~3 min, research ~30 min)

# File patterns for each category
E2E_FAST := smoke.spec.ts chat-management.spec.ts
E2E_MEDIUM := edit-message.spec.ts follow-up.spec.ts regenerate.spec.ts stop-cancel.spec.ts error-handling.spec.ts provenance-export.spec.ts
E2E_SLOW := citations.spec.ts grey-references.spec.ts numeric-claims.spec.ts research-flow.spec.ts verification-summary.spec.ts
E2E_SUPER_SLOW := parallel-research.spec.ts
E2E_CUSTOM_AGENTS := custom-agent-crud.spec.ts custom-agent-ui.spec.ts custom-agent-research.spec.ts

e2e:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend (ensures fresh static files)..."
	@make build
	@echo "Target: $(E2E_TARGET) (env: ../.env + ../.env.$(E2E_TARGET))"
	@echo "Running database migrations..."
	@bash -c '$(call load-env,$(E2E_TARGET)) && uv run alembic upgrade head'
	@echo "Running E2E tests with Playwright..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && npm test'

e2e-fast:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend..."
	@make build
	@echo "Running FAST E2E tests (~1 min)..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && npx playwright test $(E2E_FAST)'

e2e-medium:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend..."
	@make build
	@echo "Running MEDIUM E2E tests (2-5 min each)..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && RUN_SLOW_TESTS=1 RUN_INTEGRATION_TESTS=true npx playwright test $(E2E_MEDIUM)'

e2e-slow:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend..."
	@make build
	@echo "Running SLOW E2E tests (~10 min each)..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && RUN_SLOW_TESTS=1 npx playwright test $(E2E_SLOW)'

e2e-super-slow:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend..."
	@make build
	@echo "Running SUPER-SLOW E2E tests (parallel research, ~15 min)..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && RUN_SLOW_TESTS=1 npx playwright test $(E2E_SUPER_SLOW)'

e2e-all:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend..."
	@make build
	@echo "Target: $(E2E_TARGET) (env: ../.env + ../.env.$(E2E_TARGET))"
	@echo "Running database migrations..."
	@bash -c '$(call load-env,$(E2E_TARGET)) && uv run alembic upgrade head'
	@echo "Running ALL E2E tests (fast + medium + slow + super-slow)..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && RUN_SLOW_TESTS=1 RUN_INTEGRATION_TESTS=true npx playwright test'

e2e-ui:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend (ensures fresh static files)..."
	@make build
	@echo "Opening Playwright UI..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && RUN_SLOW_TESTS=1 RUN_INTEGRATION_TESTS=true npm run test:ui'

e2e-debug:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend (ensures fresh static files)..."
	@make build
	@echo "Running E2E tests in debug mode..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && npm run test:debug'

e2e-custom-agents:
	@echo "Stopping any existing server on port 8000..."
	@./scripts/kill-server.sh 8000 || true
	@echo "Building frontend..."
	@make build
	@echo "Running CUSTOM AGENT E2E tests (CRUD + UI + research, ~30 min)..."
	bash -c '$(call load-env,$(E2E_TARGET)) && cd e2e && RUN_SLOW_TESTS=1 RUN_INTEGRATION_TESTS=1 npx playwright test $(E2E_CUSTOM_AGENTS) --reporter=list'

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
requirements: build-framework
	@echo "Generating requirements.txt from pyproject.toml..."
	uv pip compile pyproject.toml --python-platform x86_64-unknown-linux-gnu -o requirements.txt
	@# Remove local editable installs (workspace deps installed via wheel below)
	@sed -i '' '/^-e file:/d' requirements.txt
	@# Tell pip to look in wheels/ for local packages (e.g. databricks-deep-research)
	@sed -i '' '1s/^/--find-links wheels\/\n/' requirements.txt
	@echo "" >> requirements.txt
	@echo "# Install framework wheel" >> requirements.txt
	@WHEEL=$$(ls wheels/databricks_deep_research-*.whl 2>/dev/null | head -1); \
	if [ -n "$$WHEEL" ]; then echo "$$WHEEL" >> requirements.txt; fi
	@echo "" >> requirements.txt
	@echo "# Install the app wheel (pre-built — no hatchling/build-isolation on platform)" >> requirements.txt
	@echo "# Build timestamp: $$(date -u +%Y%m%d%H%M%S)" >> requirements.txt
	@if [ ! -f "static/index.html" ]; then echo "ERROR: static/index.html not found — run 'make build' first"; exit 1; fi
	@rm -f wheels/databricks_deep_research_app-*.whl
	uv build --wheel --out-dir wheels/
	@APP_WHEEL=$$(ls wheels/databricks_deep_research_app-*.whl 2>/dev/null | head -1); \
	if [ -n "$$APP_WHEEL" ]; then \
		echo "$$APP_WHEEL" >> requirements.txt; \
		echo "  App wheel: $$APP_WHEEL ($$(du -h "$$APP_WHEEL" | cut -f1))"; \
	else \
		echo "ERROR: App wheel build failed — no .whl in wheels/"; exit 1; \
	fi
	@echo "requirements.txt updated (framework + app wheels, no build-on-platform)"

# =============================================================================
# Databricks Asset Bundles (DAB) Deployment
# Use DAB for infrastructure-as-code with configurable secret scope/key
# =============================================================================

# Validate bundle configuration
bundle-validate:
	databricks bundle validate

# MLflow experiment that the app traces into. Mirrors MLFLOW_EXPERIMENT_NAME
# in databricks.yml; granted to the app SP during deploy so set_experiment()
# does not fail silently inside the app process.
MLFLOW_EXPERIMENT_PATH ?= /Shared/deep-research-agent-experiments

# Full deployment - ONE COMMAND DOES EVERYTHING
# Builds frontend, deploys bundle, waits for DB, runs migrations, starts app
# Usage: make deploy TARGET=dev BRAVE_SCOPE=msh
#        make deploy TARGET=ais
#
# Deployment sequence:
#   1. Build frontend (npm run build)
#   2. Generate requirements.txt
#   3. Deploy bundle (creates Lakebase + app)
#   4. Wait for Lakebase to be ready + resolve PGHOST
#   5. Create database + re-deploy with endpoint info
#   6. Run migrations with developer credentials
#   7. Write ../.env.{target} with connection info
#   8. Grant permissions + start app
#   9. Show deployment summary
#
# NOTE: Migrations are run with developer credentials (not app service principal)
# because the app's service principal has CAN_CONNECT_AND_CREATE permission
# but cannot create tables in the public schema.
TARGET ?= ais
BRAVE_SCOPE ?=
deploy: build requirements
	@echo "=============================================="
	@echo "Full Deployment Pipeline (Autoscaling)"
	@echo "Target: $(TARGET)"
	@echo "=============================================="
	@PROFILE=$$(uv run python3 -c "import yaml,sys;cfg=yaml.safe_load(open('databricks.yml'));t=cfg.get('targets',{}).get('$(TARGET)');sys.exit(1) if not t else print(t['workspace']['profile'])" 2>/dev/null) || \
		{ echo "ERROR: Unknown target '$(TARGET)'. Valid targets:"; uv run python3 -c "import yaml;[print(f'  {k}') for k in yaml.safe_load(open('databricks.yml'))['targets']]"; exit 1; }; \
	RESOURCE_SUFFIX=$$(uv run python3 -c "import yaml;cfg=yaml.safe_load(open('databricks.yml'));print(cfg['targets']['$(TARGET)']['variables']['resource_suffix'])" 2>/dev/null); \
	PROJECT_ID="deep-research-$$RESOURCE_SUFFIX"; \
	APP_NAME=$$(databricks bundle summary -t $(TARGET) --output json 2>/dev/null | jq -r '.resources.apps.deep_research_agent.name // empty'); \
	DEPLOY_ARGS="-t $(TARGET)"; \
	if [ -n "$(BRAVE_SCOPE)" ]; then \
		DEPLOY_ARGS="$$DEPLOY_ARGS --var brave_secret_scope=$(BRAVE_SCOPE)"; \
	fi; \
	\
	echo ""; \
	echo "Step 1/9: Deploying bundle with postgres (bootstrap)..."; \
	echo "  Profile: $$PROFILE"; \
	echo "  Project: $$PROJECT_ID"; \
	echo "  Note: Using postgres database for initial deploy"; \
	databricks bundle deploy $$DEPLOY_ARGS --var lakebase_database=postgres || { echo "ERROR: Bundle deploy failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 2/9: Discovering Autoscaling endpoint..."; \
	ENDPOINT_NAME=$$(DATABRICKS_CONFIG_PROFILE="$$PROFILE" uv run python -c \
		"from deep_research.deployment import discover_autoscaling_endpoint; print(discover_autoscaling_endpoint('$$PROJECT_ID'))"); \
	if [ -z "$$ENDPOINT_NAME" ]; then \
		echo "ERROR: Could not discover Autoscaling endpoint for project $$PROJECT_ID"; exit 1; \
	fi; \
	echo "  ENDPOINT_NAME: $$ENDPOINT_NAME"; \
	\
	echo ""; \
	echo "Step 3/9: Waiting for Autoscaling endpoint to be ready..."; \
	DATABRICKS_CONFIG_PROFILE="$$PROFILE" ENDPOINT_NAME="$$ENDPOINT_NAME" LAKEBASE_DATABASE=postgres \
		uv run python -c "from deep_research.deployment.lakebase import wait_for_lakebase_sync; exit(0 if wait_for_lakebase_sync(endpoint_name='$$ENDPOINT_NAME', timeout_seconds=300, poll_interval_seconds=10) else 1)" \
		|| { echo "ERROR: Autoscaling endpoint not ready"; exit 1; }; \
	\
	echo "  Resolving PGHOST..."; \
	PGHOST=$$(DATABRICKS_CONFIG_PROFILE="$$PROFILE" uv run python -c \
		"from databricks.sdk import WorkspaceClient; w=WorkspaceClient(); ep=w.postgres.get_endpoint(name='$$ENDPOINT_NAME'); print(ep.status.hosts.host if ep.status and ep.status.hosts else '')" 2>/dev/null || echo ""); \
	echo "  PGHOST: $$PGHOST"; \
	export PGHOST; \
	\
	echo ""; \
	echo "Step 4/9: Creating deep_research database..."; \
	DATABRICKS_CONFIG_PROFILE="$$PROFILE" ENDPOINT_NAME="$$ENDPOINT_NAME" PGHOST="$$PGHOST" LAKEBASE_INSTANCE_NAME="" LAKEBASE_DATABASE=deep_research \
		uv run python -c "import asyncio; from deep_research.db.bootstrap import ensure_database_exists; asyncio.run(ensure_database_exists())" \
		|| { echo "ERROR: Database creation failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 5/9: Re-deploying bundle with deep_research + endpoint..."; \
	databricks bundle deploy $$DEPLOY_ARGS --var lakebase_database=deep_research --var endpoint_name=$$ENDPOINT_NAME --var pghost=$$PGHOST || { echo "ERROR: Bundle re-deploy failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 6/9: Running database migrations..."; \
	DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
	ENDPOINT_NAME="$$ENDPOINT_NAME" \
	PGHOST="$$PGHOST" \
	LAKEBASE_INSTANCE_NAME="" \
	LAKEBASE_DATABASE="deep_research" \
	uv run alembic upgrade head || { echo "ERROR: Migrations failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 7/9: Writing ../.env.$(TARGET) with connection info..."; \
	printf "# Lakebase connection for target: $(TARGET)\n# Generated by: make deploy TARGET=$(TARGET)\nDATABRICKS_CONFIG_PROFILE=%s\nENDPOINT_NAME=%s\nPGHOST=%s\nLAKEBASE_DATABASE=deep_research\n" "$$PROFILE" "$$ENDPOINT_NAME" "$$PGHOST" > $(ROOT_DIR)/.env.$(TARGET); \
	cat $(ROOT_DIR)/.env.$(TARGET); \
	\
	echo ""; \
	echo "Step 8/9: Granting permissions + starting app..."; \
	if [ -z "$$APP_NAME" ]; then \
		APP_NAME=$$(databricks bundle summary -t $(TARGET) --output json 2>/dev/null | jq -r '.resources.apps.deep_research_agent.name // empty'); \
	fi; \
	if [ -z "$$APP_NAME" ]; then \
		echo "ERROR: Could not resolve app name from bundle summary"; \
		echo "  Check that 'databricks bundle summary -t $(TARGET) --output json' returns the app config."; \
		exit 1; \
	fi; \
	echo "  Resolved APP_NAME: $$APP_NAME"; \
	echo "Configuring Databricks App Postgres resource..."; \
	DATABRICKS_CONFIG_PROFILE="$$PROFILE" \
	ENDPOINT_NAME="$$ENDPOINT_NAME" \
	LAKEBASE_DATABASE="deep_research" \
	uv run python -m deep_research.deployment.app_postgres_resource \
		--app-name "$$APP_NAME" \
		--profile "$$PROFILE" \
		--endpoint-name "$$ENDPOINT_NAME" \
		--database-name "deep_research" \
		|| { echo "ERROR: App Postgres resource configuration failed"; exit 1; }; \
	ENDPOINT_NAME="$$ENDPOINT_NAME" ./scripts/grant-app-permissions.sh "" "$$PROFILE" "deep_research" "$$APP_NAME" || { echo "ERROR: Permission grant failed"; exit 1; }; \
	echo "Granting MLflow experiment access to app SP..."; \
	./scripts/grant-app-mlflow-experiment.sh "$(MLFLOW_EXPERIMENT_PATH)" "$$APP_NAME" "$$PROFILE" CAN_EDIT \
		|| { echo "ERROR: MLflow experiment grant failed"; exit 1; }; \
	echo "Re-deploying and starting app with bundle run..."; \
	databricks bundle run -t $(TARGET) --var endpoint_name=$$ENDPOINT_NAME --var pghost=$$PGHOST deep_research_agent || { echo "ERROR: App start failed"; exit 1; }; \
	\
	echo ""; \
	echo "Step 9/9: Deployment summary..."; \
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

# Fast app-only redeploy: bundle deploy + bundle run with vars from ../.env.$(TARGET).
# Skips: DB migrations, Lakebase provisioning, permission grants, frontend build.
# Use for Python/yaml/databricks.yml-only changes after a successful `make deploy`.
# Usage: make app-deploy TARGET=dev
#        make app-deploy TARGET=dev BUILD=1          # Also rebuild frontend + requirements.txt
#        make app-deploy TARGET=dev BRAVE_SCOPE=msh  # Override Brave secret scope
BUILD ?=
app-deploy: $(if $(BUILD),build requirements,)
	@if [ ! -f $(ROOT_DIR)/.env.$(TARGET) ]; then \
		echo "ERROR: $(ROOT_DIR)/.env.$(TARGET) not found. Run 'make deploy TARGET=$(TARGET)' first."; \
		exit 1; \
	fi
	@echo "=============================================="
	@echo "App-only redeploy (no DB migrate, no grants)"
	@echo "Target: $(TARGET)"
	@echo "=============================================="
	@$(call load-env,$(TARGET)) && \
	if [ -z "$$ENDPOINT_NAME" ] || [ -z "$$PGHOST" ]; then \
		echo "ERROR: ENDPOINT_NAME or PGHOST missing in $(ROOT_DIR)/.env.$(TARGET)."; \
		echo "Run 'make deploy TARGET=$(TARGET)' to regenerate it."; \
		exit 1; \
	fi; \
	LAKEBASE_DB="$${LAKEBASE_DATABASE:-deep_research}"; \
	DEPLOY_ARGS="-t $(TARGET) --var lakebase_database=$$LAKEBASE_DB --var endpoint_name=$$ENDPOINT_NAME --var pghost=$$PGHOST"; \
	if [ -n "$(BRAVE_SCOPE)" ]; then \
		DEPLOY_ARGS="$$DEPLOY_ARGS --var brave_secret_scope=$(BRAVE_SCOPE)"; \
	fi; \
	echo ""; \
	echo "Step 1/3: databricks bundle deploy $$DEPLOY_ARGS"; \
	echo "  Profile:  $$DATABRICKS_CONFIG_PROFILE"; \
	echo "  Endpoint: $$ENDPOINT_NAME"; \
	echo "  PGHOST:   $$PGHOST"; \
	echo "  Database: $$LAKEBASE_DB"; \
	databricks bundle deploy $$DEPLOY_ARGS || { echo "ERROR: Bundle deploy failed"; exit 1; }; \
	echo ""; \
	echo "Step 2/3: Granting MLflow experiment access to app SP..."; \
	APP_NAME=$$(databricks bundle summary -t $(TARGET) --output json 2>/dev/null | jq -r '.resources.apps.deep_research_agent.name // empty'); \
	if [ -z "$$APP_NAME" ]; then echo "ERROR: Could not resolve app name from bundle summary"; exit 1; fi; \
	./scripts/grant-app-mlflow-experiment.sh "$(MLFLOW_EXPERIMENT_PATH)" "$$APP_NAME" "$$DATABRICKS_CONFIG_PROFILE" CAN_EDIT \
		|| { echo "ERROR: MLflow experiment grant failed"; exit 1; }; \
	echo ""; \
	echo "Step 3/3: databricks bundle run -t $(TARGET) deep_research_agent"; \
	databricks bundle run -t $(TARGET) --var endpoint_name=$$ENDPOINT_NAME --var pghost=$$PGHOST deep_research_agent \
		|| { echo "ERROR: App start failed"; exit 1; }
	@echo ""
	@echo "App redeploy complete. Logs: make logs TARGET=$(TARGET) FOLLOW=-f"

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
	@PROFILE=$$(uv run python3 -c "import yaml,sys;cfg=yaml.safe_load(open('databricks.yml'));t=cfg.get('targets',{}).get('$(TARGET)');sys.exit(1) if not t else print(t['workspace']['profile'])" 2>/dev/null) || \
		{ echo "ERROR: Unknown target '$(TARGET)'. Valid targets:"; uv run python3 -c "import yaml;[print(f'  {k}') for k in yaml.safe_load(open('databricks.yml'))['targets']]"; exit 1; }; \
	APP_NAME=$$(databricks bundle summary -t $(TARGET) --output json 2>/dev/null | jq -r '.resources.apps.deep_research_agent.name // empty'); \
	./scripts/download-app-logs.sh "$$APP_NAME" "$$PROFILE" $(FOLLOW) $(SEARCH)
