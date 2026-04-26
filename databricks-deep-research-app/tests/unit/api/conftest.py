"""Conftest for unit API tests.

Sets minimal environment variables required by Settings validation before
the FastAPI app is imported. Unit tests use the 'fake' storage backend
so no real database or Lakebase instance is needed.
"""

import os

# Must be set before `from deep_research.main import app` executes.
# Settings.model_validator requires one of: DATABASE_URL, LAKEBASE_INSTANCE_NAME,
# or STORAGE_BACKEND=fake/sql_warehouse (non-lakebase). Using 'fake' satisfies
# the validator without any real infrastructure.
os.environ.setdefault("STORAGE_BACKEND", "fake")
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

# Provide a dummy secret key so JWTMiddleware doesn't fail at startup.
os.environ.setdefault("SECRET_KEY", "test-secret-key-unit-tests-only")

# Suppress LLM / workspace calls during unit tests.
os.environ.setdefault("DATABRICKS_HOST", "https://test.azuredatabricks.net")
os.environ.setdefault("DATABRICKS_TOKEN", "test-token")
os.environ.setdefault("LLM_ENDPOINT_NAME", "test-endpoint")
