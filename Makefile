# =============================================================================
# Deep Research Agent - Monorepo Root Makefile
# =============================================================================
#
# This monorepo contains two sub-projects:
#
#   databricks-deep-research/       Framework library (agents, tools, pipeline)
#   databricks-deep-research-app/   Web application (FastAPI + React frontend)
#
# This Makefile is a thin delegator. All detailed build logic lives in the
# sub-project Makefiles. Targets here either:
#   - Forward to the app Makefile (dev, build, deploy, e2e, db-*, ...)
#   - Forward to the framework Makefile (test-framework, lint-framework, ...)
#   - Aggregate both (test, typecheck, lint, format, clean)
#
# Quick reference:
#   make dev                  Start app backend + frontend (hot reload)
#   make test                 Run tests for both framework and app
#   make test-framework       Framework tests only
#   make test-app             App tests only
#   make typecheck            Type-check both projects
#   make lint                 Lint both projects
#   make format               Format both projects
#   make build                Build frontend (app)
#   make deploy TARGET=dev    Deploy app to Databricks
#   make install              Install all workspace dependencies
#   make clean                Clean build artifacts in both projects
# =============================================================================

.PHONY: dev dev-backend dev-frontend dev-help build prod install install-dev \
        typecheck typecheck-framework typecheck-app lint lint-framework lint-app \
        format format-framework format-app \
        test test-framework test-app test-integration test-complex test-scaffold-and-run test-all-python test-frontend test-all \
        db-provision db-cleanup db-migrate db-status db-reset db-migrate-remote db-local db-local-stop clean_db clean-e2e \
        e2e e2e-fast e2e-medium e2e-slow e2e-super-slow e2e-all e2e-ui e2e-debug e2e-custom-agents \
        clean clean-all quickstart deploy deploy-unchecked app-deploy app-deploy-unchecked requirements bundle-validate bundle-summary logs \
        run-example \
        worktree worktree-list worktree-remove worktree-link

# Sub-project directories (relative to repo root)
APP_DIR     := databricks-deep-research-app
FRAMEWORK_DIR := databricks-deep-research

# =============================================================================
# Aggregated targets (both projects)
# =============================================================================

## Run all tests (framework + app)
test: test-framework test-app

## Type-check both projects
typecheck: typecheck-framework typecheck-app

## Lint both projects
lint: lint-framework lint-app

## Format both projects
format: format-framework format-app

## Clean build artifacts in both projects
clean: clean-framework clean-app

## Install all workspace dependencies (both projects)
install: install-framework install-app

# =============================================================================
# Framework targets
# =============================================================================

test-framework:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) test; \
	else \
		echo "Skipping framework tests ($(FRAMEWORK_DIR)/ not yet available)"; \
	fi

typecheck-framework:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) typecheck; \
	else \
		echo "Skipping framework typecheck ($(FRAMEWORK_DIR)/ not yet available)"; \
	fi

lint-framework:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) lint; \
	else \
		echo "Skipping framework lint ($(FRAMEWORK_DIR)/ not yet available)"; \
	fi

format-framework:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) format; \
	else \
		echo "Skipping framework format ($(FRAMEWORK_DIR)/ not yet available)"; \
	fi

clean-framework:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) clean; \
	else \
		echo "Skipping framework clean ($(FRAMEWORK_DIR)/ not yet available)"; \
	fi

install-framework:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) install; \
	else \
		echo "Skipping framework install ($(FRAMEWORK_DIR)/ not yet available)"; \
	fi

# =============================================================================
# App targets - delegate to app Makefile
# =============================================================================

test-app:
	$(MAKE) -C $(APP_DIR) test

typecheck-app:
	$(MAKE) -C $(APP_DIR) typecheck

lint-app:
	$(MAKE) -C $(APP_DIR) lint

format-app:
	$(MAKE) -C $(APP_DIR) format

clean-app:
	$(MAKE) -C $(APP_DIR) clean

clean-all:
	@if [ -d "$(FRAMEWORK_DIR)" ] && [ -f "$(FRAMEWORK_DIR)/Makefile" ]; then \
		$(MAKE) -C $(FRAMEWORK_DIR) clean-all 2>/dev/null || true; \
	fi
	$(MAKE) -C $(APP_DIR) clean-all

install-app:
	$(MAKE) -C $(APP_DIR) install

install-dev:
	$(MAKE) -C $(APP_DIR) install-dev

# --- Development ---

dev:
	$(MAKE) -C $(APP_DIR) dev

dev-backend:
	$(MAKE) -C $(APP_DIR) dev-backend

dev-frontend:
	$(MAKE) -C $(APP_DIR) dev-frontend

dev-help:
	$(MAKE) -C $(APP_DIR) dev-help

# --- Production Build ---

build:
	$(MAKE) -C $(APP_DIR) build

prod:
	$(MAKE) -C $(APP_DIR) prod

# --- Testing (app-specific) ---

test-integration:
	$(MAKE) -C $(APP_DIR) test-integration

test-complex:
	$(MAKE) -C $(APP_DIR) test-complex

# Scaffold + Run live integration test. CASE=<id> to scope to one case.
CASE ?=
SCAFFOLD_RUN_LOG ?=
test-scaffold-and-run:
	$(MAKE) -C $(APP_DIR) test-scaffold-and-run CASE="$(CASE)" SCAFFOLD_RUN_LOG="$(SCAFFOLD_RUN_LOG)"

test-all-python:
	$(MAKE) -C $(APP_DIR) test-all-python

test-frontend:
	$(MAKE) -C $(APP_DIR) test-frontend

test-all: test-framework
	$(MAKE) -C $(APP_DIR) test-all

# --- Database ---

db-provision:
	$(MAKE) -C $(APP_DIR) db-provision TARGET=$(TARGET)

db-cleanup:
	$(MAKE) -C $(APP_DIR) db-cleanup TARGET=$(TARGET)

db-migrate:
	$(MAKE) -C $(APP_DIR) db-migrate DB_SUFFIX=$(DB_SUFFIX) TARGET=$(TARGET)

db-status:
	$(MAKE) -C $(APP_DIR) db-status

db-reset:
	$(MAKE) -C $(APP_DIR) db-reset TARGET=$(TARGET)

db-migrate-remote:
	$(MAKE) -C $(APP_DIR) db-migrate-remote TARGET=$(TARGET)

db-local:
	$(MAKE) -C $(APP_DIR) db-local

db-local-stop:
	$(MAKE) -C $(APP_DIR) db-local-stop

clean_db:
	$(MAKE) -C $(APP_DIR) clean_db DB_SUFFIX=$(DB_SUFFIX)

clean-e2e:
	$(MAKE) -C $(APP_DIR) clean-e2e

# --- E2E Testing ---

e2e:
	$(MAKE) -C $(APP_DIR) e2e

e2e-fast:
	$(MAKE) -C $(APP_DIR) e2e-fast

e2e-medium:
	$(MAKE) -C $(APP_DIR) e2e-medium

e2e-slow:
	$(MAKE) -C $(APP_DIR) e2e-slow

e2e-super-slow:
	$(MAKE) -C $(APP_DIR) e2e-super-slow

e2e-all:
	$(MAKE) -C $(APP_DIR) e2e-all

e2e-ui:
	$(MAKE) -C $(APP_DIR) e2e-ui

e2e-debug:
	$(MAKE) -C $(APP_DIR) e2e-debug

e2e-custom-agents:
	$(MAKE) -C $(APP_DIR) e2e-custom-agents

# --- Deployment ---

quickstart:
	$(MAKE) -C $(APP_DIR) quickstart

requirements:
	$(MAKE) -C $(APP_DIR) requirements

bundle-validate:
	$(MAKE) -C $(APP_DIR) bundle-validate

bundle-summary:
	$(MAKE) -C $(APP_DIR) bundle-summary

TARGET ?= ais
BRAVE_SCOPE ?=
# `deploy` and `app-deploy` gate on `typecheck-framework` so attribute-name
# typos, missing kwargs, and signature drift cannot reach production. Strict
# mypy is configured in `databricks-deep-research/pyproject.toml`. For
# emergency reverts where typecheck cannot pass (e.g., a baseline-cleanup
# follow-up is still pending), use the `*-unchecked` variants.
deploy: typecheck-framework
	$(MAKE) -C $(APP_DIR) deploy TARGET=$(TARGET) BRAVE_SCOPE=$(BRAVE_SCOPE)

deploy-unchecked:
	$(MAKE) -C $(APP_DIR) deploy TARGET=$(TARGET) BRAVE_SCOPE=$(BRAVE_SCOPE)

# Fast app-only redeploy (Python/yaml/vars, no DB migrate, no grants).
# Set BUILD=1 to also rebuild frontend + requirements.txt.
BUILD ?=
app-deploy: typecheck-framework
	$(MAKE) -C $(APP_DIR) app-deploy TARGET=$(TARGET) BRAVE_SCOPE=$(BRAVE_SCOPE) BUILD=$(BUILD)

app-deploy-unchecked:
	$(MAKE) -C $(APP_DIR) app-deploy TARGET=$(TARGET) BRAVE_SCOPE=$(BRAVE_SCOPE) BUILD=$(BUILD)

FOLLOW ?=
SEARCH ?=
logs:
	$(MAKE) -C $(APP_DIR) logs TARGET=$(TARGET) FOLLOW=$(FOLLOW) SEARCH=$(SEARCH)

# --- Framework examples ---

WORKFLOW ?=
QUERY ?=
## Run a framework example workflow (e.g. make run-example WORKFLOW=simple_research QUERY="What is AI?")
run-example:
ifdef QUERY
	cd $(FRAMEWORK_DIR) && uv run examples/run_workflow.py $(WORKFLOW) "$(QUERY)"
else
	cd $(FRAMEWORK_DIR) && uv run examples/run_workflow.py $(WORKFLOW)
endif

# =============================================================================
# Git Worktrees
# =============================================================================
# Create isolated worktrees with shared .env files for parallel development.
# Worktrees live in ../.worktrees/<branch>/. All gitignored .env* files are
# auto-symlinked from the main worktree. BASE defaults to current branch.

## Create a worktree: make worktree BRANCH=feature-xyz [BASE=current] [INSTALL=1]
worktree:
	@if [ -z "$(BRANCH)" ]; then echo "Usage: make worktree BRANCH=<name> [BASE=<ref>] [INSTALL=1]" && exit 1; fi
	@bash scripts/worktree.sh create "$(BRANCH)" $(if $(BASE),"$(BASE)") $(if $(INSTALL),--install)

## List all worktrees
worktree-list:
	@bash scripts/worktree.sh list

## Remove a worktree: make worktree-remove BRANCH=feature-xyz [DELETE_BRANCH=1]
worktree-remove:
	@if [ -z "$(BRANCH)" ]; then echo "Usage: make worktree-remove BRANCH=<name> [DELETE_BRANCH=1]" && exit 1; fi
	@bash scripts/worktree.sh remove "$(BRANCH)" $(if $(DELETE_BRANCH),--delete-branch)

## Re-link env files: make worktree-link BRANCH=feature-xyz
worktree-link:
	@if [ -z "$(BRANCH)" ]; then echo "Usage: make worktree-link BRANCH=<name>" && exit 1; fi
	@bash scripts/worktree.sh link-env "$(BRANCH)"
