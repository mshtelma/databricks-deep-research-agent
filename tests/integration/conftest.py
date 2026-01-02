"""Shared fixtures and configuration for integration tests.

Integration tests use REAL API calls to:
- Databricks LLM endpoints
- Brave Search API

They also use a test-specific configuration (config/app.test.yaml) with
minimal iterations and smaller token limits for faster execution.
"""

import os
from collections.abc import AsyncGenerator

import mlflow
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.tools.web_crawler import WebCrawler
from src.core.app_config import clear_config_cache
from src.db.session import close_db, get_session_maker
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient


# ---------------------------------------------------------------------------
# Credential Checks
# ---------------------------------------------------------------------------


def _has_databricks_creds() -> bool:
    """Check if Databricks credentials are available."""
    return bool(os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_CONFIG_PROFILE"))


def _has_brave_key() -> bool:
    """Check if Brave API key is available."""
    return bool(os.getenv("BRAVE_API_KEY"))


# Skip markers for tests that require real credentials
requires_databricks = pytest.mark.skipif(
    not _has_databricks_creds(),
    reason="Databricks credentials not configured (check .env for DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE)",
)
requires_brave = pytest.mark.skipif(
    not _has_brave_key(),
    reason="Brave API key not configured (check .env for BRAVE_API_KEY)",
)

# Combined marker for tests that need both
requires_all_credentials = pytest.mark.skipif(
    not (_has_databricks_creds() and _has_brave_key()),
    reason="Both Databricks and Brave credentials required (check .env)",
)


# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def use_test_config() -> None:
    """Use test-specific config for all integration tests.

    This sets APP_CONFIG_PATH to use config/app.test.yaml which has:
    - Minimal iterations (1-2 max)
    - Smaller token limits
    - Faster model tier defaults
    - Disabled clarification
    """
    os.environ["APP_CONFIG_PATH"] = "config/app.test.yaml"
    # Clear any cached config to ensure fresh load
    clear_config_cache()
    yield
    # Cleanup after tests
    if "APP_CONFIG_PATH" in os.environ:
        del os.environ["APP_CONFIG_PATH"]
    clear_config_cache()


@pytest.fixture(autouse=True)
def cleanup_mlflow_run() -> None:
    """Ensure MLflow runs are properly ended after each test.

    This fixture prevents stale runs from leaking between tests.
    """
    yield
    # End any active runs after each test to prevent leakage
    while mlflow.active_run():
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def llm_client() -> LLMClient:
    """Create a real LLMClient with Databricks endpoints."""
    client = LLMClient()
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


# ---------------------------------------------------------------------------
# Database Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide async database session for tests.

    Ensures engine is fresh for each test to avoid event loop mismatch.
    Rolls back all changes after test completes for isolation.

    Note: Requires database configuration (LAKEBASE_* or DATABASE_URL).
    """
    # Clear any cached engine from previous tests (different event loop)
    await close_db()

    session_maker = get_session_maker()

    async with session_maker() as session:
        yield session
        # Rollback to clean up any test data
        await session.rollback()
