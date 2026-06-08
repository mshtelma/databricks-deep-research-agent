"""Shared test helpers and fixtures for integration-like app suites."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest

from deep_research.agent.tools.web_crawler import WebCrawler
from deep_research.services.llm.client import LLMClient
from deep_research.services.search.brave import BraveSearchClient


def _has_databricks_creds() -> bool:
    """Check if Databricks credentials are available."""
    return bool(os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_CONFIG_PROFILE"))


def _has_brave_key() -> bool:
    """Check if Brave API key is available."""
    return bool(os.getenv("BRAVE_API_KEY"))


# Default Databricks workspace experiment for traced live suites (matches the
# app.yaml deployment config). Used for the Databricks and explicit-URI paths.
DEFAULT_DATABRICKS_EXPERIMENT = "/Shared/deep-research-agent"
# Experiment name used in the local MLflow OSS fallback (no Databricks creds).
LOCAL_TRACE_EXPERIMENT = "deep-research-scaffold"


def _local_trace_sqlite_uri() -> str:
    """Absolute sqlite tracking URI under ``tests/_runs`` (gitignored).

    Creates the directory if missing — sqlite needs the parent to exist and
    ``make test-complex`` does not pre-create ``tests/_runs``. The path sits
    next to the scaffold's per-case artifact dirs (``_RUNS_ROOT``).
    """
    runs_dir = Path(__file__).parent / "_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{runs_dir / 'mlflow.db'}"


def resolve_trace_backend() -> tuple[str, str]:
    """Pick the MLflow ``(tracking_uri, experiment_name)`` for live test suites.

    Precedence:

    1. Explicit ``MLFLOW_TRACKING_URI`` env override (paired with
       ``MLFLOW_EXPERIMENT_NAME`` or the Databricks default experiment).
    2. Databricks workspace experiment when creds are configured
       (``DATABRICKS_TOKEN`` / ``DATABRICKS_CONFIG_PROFILE``).
    3. Local MLflow OSS sqlite store under ``tests/_runs/mlflow.db`` otherwise —
       browse with ``mlflow ui --backend-store-uri <uri>`` (Traces tab) or query
       via ``mlflow.search_traces()``.
    """
    explicit_uri = os.getenv("MLFLOW_TRACKING_URI")
    if explicit_uri:
        return explicit_uri, os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_DATABRICKS_EXPERIMENT)
    if _has_databricks_creds():
        return "databricks", os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_DATABRICKS_EXPERIMENT)
    return _local_trace_sqlite_uri(), LOCAL_TRACE_EXPERIMENT


requires_databricks = pytest.mark.skipif(
    not _has_databricks_creds(),
    reason=(
        "Databricks credentials not configured "
        "(check .env for DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE)"
    ),
)
requires_brave = pytest.mark.skipif(
    not _has_brave_key(),
    reason="Brave API key not configured (check .env for BRAVE_API_KEY)",
)
requires_all_credentials = pytest.mark.skipif(
    not (_has_databricks_creds() and _has_brave_key()),
    reason="Both Databricks and Brave credentials required (check .env)",
)


@pytest.fixture(autouse=True)
def cleanup_mlflow_run() -> None:
    """Ensure MLflow runs are properly ended after each test."""
    yield
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
async def llm_client() -> LLMClient:
    """Create a real LLMClient with Databricks endpoints."""
    try:
        client = LLMClient()
    except (ValueError, OSError, RuntimeError) as exc:
        pytest.skip(f"Databricks auth unavailable: {exc}")
    yield client
    await client.close()


@pytest.fixture
async def brave_client() -> BraveSearchClient:
    """Create a real BraveSearchClient."""
    return BraveSearchClient()


@pytest.fixture
async def web_crawler() -> WebCrawler:
    """Create a real WebCrawler for fetching pages."""
    crawler = WebCrawler()
    yield crawler
    await crawler.close()
