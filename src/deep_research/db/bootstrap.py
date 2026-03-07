"""Database bootstrap - creates database if not exist."""

import asyncio
import logging
from urllib.parse import urlparse

import asyncpg  # type: ignore[import-untyped]

from deep_research.core.config import Settings, get_settings
from deep_research.db.grant_permissions import _validate_sql_identifier
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
    _validate_sql_identifier(target_database, "database name")
    provider = get_credential_provider(settings)
    if not provider:
        raise RuntimeError("Lakebase configured but credential provider unavailable")

    cred = provider.get_credential()
    host = provider.get_host()
    port = provider.get_port()

    # Determine bootstrap database based on backend type
    # Autoscaling: 'databricks_postgres' is auto-created; 'postgres' may not be accessible
    # Provisioned: 'postgres' system database is the standard bootstrap target
    if provider.get_backend_type() == "autoscaling":
        bootstrap_db = "databricks_postgres"
    else:
        bootstrap_db = "postgres"

    logger.info(f"Connecting to {bootstrap_db} database at {host}...")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=cred.username,
        password=cred.token,
        database=bootstrap_db,
        ssl="require",
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
    _validate_sql_identifier(target_database, "database name")
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


async def drop_all_tables(settings: Settings | None = None) -> None:
    """Drop all tables and enum types in the public schema.

    This is the inverse of migrations — used by db-reset when alembic downgrade
    fails due to unknown revisions from consolidated feature branches.

    Args:
        settings: Application settings. Uses get_settings() if not provided.
    """
    if settings is None:
        settings = get_settings()

    # Import here to avoid circular imports with session.py
    from sqlalchemy import pool, text
    from sqlalchemy.ext.asyncio import create_async_engine

    from deep_research.db.session import get_database_url

    connect_args = {"ssl": True} if settings.use_lakebase else {}
    engine = create_async_engine(
        get_database_url(settings),
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with engine.begin() as conn:
        # Drop all tables in public schema
        result = await conn.execute(
            text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        )
        tables = [row[0] for row in result.fetchall()]
        for table in tables:
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))

        # Drop all enum types in public schema
        result = await conn.execute(
            text(
                "SELECT t.typname FROM pg_type t "
                "JOIN pg_namespace n ON t.typnamespace = n.oid "
                "WHERE n.nspname = 'public' AND t.typtype = 'e'"
            )
        )
        types = [row[0] for row in result.fetchall()]
        for type_name in types:
            await conn.execute(
                text(f'DROP TYPE IF EXISTS "{type_name}" CASCADE')
            )

    await engine.dispose()
    logger.info(
        "DROP_ALL_TABLES dropped_tables=%d dropped_types=%d",
        len(tables),
        len(types),
    )


if __name__ == "__main__":
    # Allow running directly: uv run python -m deep_research.db.bootstrap
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ensure_database_exists())
