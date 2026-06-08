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

import pytest
from databricks_deep_research.tracing import (
    setup_mlflow_tracing,
    shutdown_mlflow_tracing,
)
from tests.shared import (
    brave_client,  # noqa: F401
    cleanup_mlflow_run,  # noqa: F401
    llm_client,  # noqa: F401
    requires_all_credentials,  # noqa: F401
    requires_brave,  # noqa: F401
    requires_databricks,  # noqa: F401
    resolve_trace_backend,
    web_crawler,  # noqa: F401
)

# ---------------------------------------------------------------------------
# MLflow Tracing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def setup_mlflow_tracking() -> None:
    """Enable MLflow tracing for complex / scaffold live runs (always on).

    Delegates to the framework's ``setup_mlflow_tracing``, which sets the
    tracking URI + experiment AND calls ``mlflow.tracing.enable()`` +
    ``mlflow.openai.autolog()`` — the latter two are what actually make the
    framework's existing ``trace_span`` instrumentation (agent harness, ReAct
    loop, workflow executor, plan-execute runner, citation pipeline) record
    spans. The legacy fixture set a URI + experiment but never enabled tracing,
    so nothing was captured.

    Backend is chosen by ``resolve_trace_backend()``: the Databricks workspace
    experiment when creds are configured, otherwise a local MLflow OSS sqlite
    store under ``tests/_runs/mlflow.db``. No test-body changes are required —
    spans flow automatically once tracing is enabled.
    """
    tracking_uri, experiment_name = resolve_trace_backend()
    # Local file/sqlite backends use MLflow's V3 span exporter, which writes
    # synchronously and does NOT support flush_trace_async_logging() — enabling
    # async logging there only emits a noisy "no attribute '_async_queue'" error
    # at teardown (traces still land). Async logging matters only for the
    # Databricks backend, where it avoids blocking on the network and the flush
    # in shutdown_mlflow_tracing() works.
    is_local_backend = tracking_uri.startswith(("sqlite:", "file:"))
    ok = setup_mlflow_tracing(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        async_logging=not is_local_backend,
    )
    if ok:
        print(f"\n📊 MLflow tracing ON — uri={tracking_uri} experiment={experiment_name}")
        if tracking_uri.startswith("sqlite:"):
            print(
                f"📊 Analyze later: mlflow ui --backend-store-uri {tracking_uri}"
                "  (open the Traces tab)"
            )
    else:
        print(f"\n⚠️  MLflow tracing setup failed (uri={tracking_uri}); spans will not be recorded")

    yield

    # Flush async-buffered spans before the process exits (mirrors the
    # benchmarks/run.py setup/shutdown pattern). Without this, async traces
    # can be dropped at interpreter teardown.
    shutdown_mlflow_tracing()


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
