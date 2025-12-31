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

from src.agent.tools.web_crawler import WebCrawler
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient


# ---------------------------------------------------------------------------
# Credential Checks (same as integration tests)
# ---------------------------------------------------------------------------


def _has_databricks_creds() -> bool:
    """Check if Databricks credentials are available."""
    return bool(os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_CONFIG_PROFILE"))


def _has_brave_key() -> bool:
    """Check if Brave API key is available."""
    return bool(os.getenv("BRAVE_API_KEY"))


# Skip markers
requires_databricks = pytest.mark.skipif(
    not _has_databricks_creds(),
    reason="Databricks credentials not configured (check .env)",
)
requires_brave = pytest.mark.skipif(
    not _has_brave_key(),
    reason="Brave API key not configured (check .env)",
)
requires_all_credentials = pytest.mark.skipif(
    not (_has_databricks_creds() and _has_brave_key()),
    reason="Both Databricks and Brave credentials required (check .env)",
)


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
    from src.core.app_config import clear_config_cache

    clear_config_cache()
    yield
    # No cleanup needed - production config is the default


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
