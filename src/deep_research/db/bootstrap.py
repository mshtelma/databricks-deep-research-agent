"""Database bootstrap - creates database if not exist."""

import asyncio
import logging
import ssl
from urllib.parse import urlparse

import asyncpg  # type: ignore[import-untyped]

from deep_research.core.config import Settings, get_settings
from deep_research.db.session import get_credential_provider

logger = logging.getLogger(__name__)


async def ensure_database_exists(settings: Settings | None = None) -> None:
    """Create application database if it doesn't exist (idempotent).

    Connects to 'postgres' system database and creates the target database.
    Safe to call multiple times - does nothing if already exists.

    Args:
        settings: Application settings. Uses get_settings() if not provided.
    """
    if settings is None:
        settings = get_settings()

    target_database = settings.lakebase_database
    logger.info(f"Ensuring database '{target_database}' exists...")

    if settings.use_lakebase:
        await _ensure_lakebase_database(settings, target_database)
    else:
        await _ensure_local_database(settings, target_database)


async def _ensure_lakebase_database(settings: Settings, target_database: str) -> None:
    """Create database on Lakebase using OAuth authentication."""
    provider = get_credential_provider(settings)
    if not provider:
        raise RuntimeError("Lakebase configured but credential provider unavailable")

    cred = provider.get_credential()
    host = provider._get_instance_host()
    port = settings.lakebase_port

    # SSL context for asyncpg (Lakebase requires SSL)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    logger.info(f"Connecting to postgres system database at {host}...")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=cred.username,
        password=cred.token,
        database="postgres",  # Always connect to system database first
        ssl=ssl_context,
    )

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_database,
        )

        if exists:
            logger.info(f"Database '{target_database}' already exists")
        else:
            logger.info(f"Creating database '{target_database}'...")
            await conn.execute(f'CREATE DATABASE "{target_database}"')
            logger.info(f"Database '{target_database}' created successfully")
    finally:
        await conn.close()


async def _ensure_local_database(settings: Settings, target_database: str) -> None:
    """Create database on local PostgreSQL (fallback for DATABASE_URL)."""
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL not configured for local PostgreSQL")

    parsed = urlparse(str(settings.database_url))

    logger.info("Connecting to local postgres system database...")

    conn = await asyncpg.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or "postgres",
        password=parsed.password or "postgres",
        database="postgres",  # Always connect to system database first
    )

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_database,
        )

        if exists:
            logger.info(f"Database '{target_database}' already exists")
        else:
            logger.info(f"Creating database '{target_database}'...")
            await conn.execute(f'CREATE DATABASE "{target_database}"')
            logger.info(f"Database '{target_database}' created successfully")
    finally:
        await conn.close()


if __name__ == "__main__":
    # Allow running directly: uv run python -m src.db.bootstrap
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ensure_database_exists())
