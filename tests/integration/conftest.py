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
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from deep_research.agent.tools.web_crawler import WebCrawler
from deep_research.core.app_config import clear_config_cache
from deep_research.core.config import get_settings
import deep_research.db.session as _db_mod
from deep_research.db.session import get_database_url
from deep_research.services.llm.client import LLMClient
from deep_research.services.search.brave import BraveSearchClient

# Keep stale engines alive so Python's GC doesn't finalize their pooled
# asyncpg connections (Connection.__del__ → terminate() → transport.abort()
# → loop.call_soon() on the dead loop → RuntimeError).
_stale_engines: list[object] = []


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
    try:
        client = LLMClient()
    except (ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Databricks auth unavailable: {e}")
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

    Creates a **test-local** engine on the current event loop so that
    creation and disposal happen on the same loop — no cross-loop errors.

    Why not reuse the module-level engine via ``get_session_maker()``?
    -----------------------------------------------------------------
    pytest-asyncio creates a *new* event loop for every test function.
    The module-level engine from the previous test holds asyncpg
    connections bound to that test's (now-dead) loop.  Calling
    ``engine.dispose()`` on those connections triggers::

        RuntimeError: Event loop is closed

    SQLAlchemy's pool catches the error internally and **logs** it at
    ERROR level (with full traceback) — a try/except around ``dispose()``
    in *our* code cannot suppress it because the exception never
    propagates to us.

    The fix: never call ``dispose()`` cross-loop.  Instead we:

    1. Detach the stale engine from module state (no dispose).
    2. Stash it in ``_stale_engines`` so GC won't run asyncpg
       ``Connection.__del__`` (which hits the same dead-loop path).
    3. Build a fresh, test-local engine on the *current* loop.
    4. Dispose it ourselves at teardown (safe — same loop).

    Note: Requires database configuration (LAKEBASE_* or DATABASE_URL).
    """
    # 1. Detach stale engine — do NOT dispose (connections are on a dead loop).
    if _db_mod._engine is not None:
        _stale_engines.append(_db_mod._engine)
        _db_mod._engine = None
        _db_mod._async_session_maker = None

    # 2. Build a fresh engine scoped to THIS test's event loop.
    settings = get_settings()
    url = get_database_url(settings)
    connect_args = {"ssl": True} if settings.use_lakebase else {}

    engine = create_async_engine(
        url,
        echo=settings.debug,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with maker() as session:
        yield session
        await session.rollback()

    # 3. Dispose OUR engine (safe — created and destroyed on the same loop).
    await engine.dispose()
