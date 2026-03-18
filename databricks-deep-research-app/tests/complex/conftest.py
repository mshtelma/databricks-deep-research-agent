"""Shared fixtures for complex, long-running tests.

Complex tests use PRODUCTION configuration (config/app.yaml) with full settings:
- Full iteration counts (3+ plan iterations)
- Full step limits (10+ steps per plan)
- Standard token limits
- All features enabled

These tests are designed for:
- Multi-entity comparative research
- Deep dive investigations
- Comprehensive citation verification

Requirements:
- .env file with DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
- .env file with BRAVE_API_KEY
- Significant time (10+ minutes per test)

Run with:
    make test-complex
    uv run pytest tests/complex -v -s --timeout=600
"""

import os

import mlflow
import pytest

from tests.shared import (
    brave_client,  # noqa: F401
    cleanup_mlflow_run,  # noqa: F401
    llm_client,  # noqa: F401
    requires_all_credentials,  # noqa: F401
    requires_brave,  # noqa: F401
    requires_databricks,  # noqa: F401
    web_crawler,  # noqa: F401
)


# ---------------------------------------------------------------------------
# MLflow Configuration
# ---------------------------------------------------------------------------

# Default experiment path for complex tests (matches app.yaml deployment config)
DEFAULT_MLFLOW_EXPERIMENT = "/Shared/deep-research-agent"


@pytest.fixture(scope="session", autouse=True)
def setup_mlflow_tracking() -> None:
    """Configure MLflow to log to remote Databricks workspace.

    This fixture sets up MLflow tracking BEFORE any tests run:
    1. Sets tracking URI to 'databricks' (uses DATABRICKS_HOST + auth)
    2. Sets experiment from MLFLOW_EXPERIMENT_NAME env var or default

    Required environment:
    - DATABRICKS_HOST or DATABRICKS_CONFIG_PROFILE for authentication
    - Optional: MLFLOW_EXPERIMENT_NAME to override default experiment

    If Databricks credentials are not available, falls back to local tracking.
    """
    # Get experiment name from env or use default
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT)

    # Check if we have Databricks credentials
    has_databricks = bool(
        os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_CONFIG_PROFILE")
    )

    if has_databricks:
        # Set tracking URI to Databricks
        mlflow.set_tracking_uri("databricks")
        print(f"\n📊 MLflow: Tracking to Databricks workspace")
    else:
        # Fall back to local tracking if no credentials
        print("\n⚠️  MLflow: No Databricks credentials, using local tracking")

    # Set or create the experiment
    try:
        mlflow.set_experiment(experiment_name)
        print(f"📊 MLflow: Experiment = {experiment_name}")
    except Exception as e:
        # If experiment creation fails (e.g., permissions), log warning
        print(f"⚠️  MLflow: Could not set experiment '{experiment_name}': {e}")

    yield

    # No cleanup needed - MLflow handles connection lifecycle


# ---------------------------------------------------------------------------
# Production Configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def use_production_config() -> None:
    """Complex tests use production config (NOT test config).

    This fixture explicitly does NOT set APP_CONFIG_PATH, allowing
    the default config/app.yaml to be used with full production settings.

    Unlike integration tests which use minimal settings for speed,
    complex tests run with:
    - max_plan_iterations: 3 (full)
    - max_steps_per_plan: 10+ (full)
    - All models at full token limits
    - All citation verification stages enabled
    """
    # Ensure APP_CONFIG_PATH is NOT set (use production config)
    if "APP_CONFIG_PATH" in os.environ:
        del os.environ["APP_CONFIG_PATH"]

    # Clear any cached config to ensure fresh load
    from deep_research.core.app_config import clear_config_cache

    clear_config_cache()
    yield
    # No cleanup needed - production config is the default
