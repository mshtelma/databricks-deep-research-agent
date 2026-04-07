.PHONY: test test-unit test-integration test-complex typecheck lint format clean
.PHONY: test-complex-traces test-integration-traces test-all-traces

# Unit tests (fast, mocked, no credentials)
test:
	uv run --extra dev pytest tests/ -v --ignore=tests/integration --ignore=tests/complex

test-unit: test

# Integration tests (real LLM + Brave, requires creds)
test-integration:
	uv run --extra dev pytest tests/integration -v -s --timeout=300

# Complex tests (long-running research, requires creds)
test-complex:
	uv run --extra dev pytest tests/complex -v -s --timeout=600

# All tests
test-all:
	uv run --extra dev pytest tests/ -v -s --timeout=600

# Tests with MLflow trace collection
test-complex-traces:
	uv run --extra dev pytest tests/complex -v -s --timeout=600 --collect-traces

test-integration-traces:
	uv run --extra dev pytest tests/integration -v -s --timeout=300 --collect-traces

test-all-traces:
	uv run --extra dev pytest tests/ -v -s --timeout=600 --collect-traces --ignore=tests/unit

typecheck:
	uv run --extra dev mypy src/databricks_deep_research --strict

lint:
	uv run --extra dev ruff check src/databricks_deep_research tests/

format:
	uv run --extra dev ruff format src/databricks_deep_research tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
