"""Shared test helpers and fixtures for integration-like app suites."""

from __future__ import annotations

import os

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
