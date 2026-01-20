"""Alembic environment configuration for async migrations."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from deep_research.core.config import get_settings
from deep_research.db.base import Base
from deep_research.db.session import get_database_url

# Import all models to ensure they're registered with Base.metadata
from deep_research.models import (  # noqa: F401  # noqa: F401
    audit_log,
    chat,
    message,
    message_feedback,
    research_session,
    source,
    user_preferences,
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get the database URL from settings (supports Lakebase OAuth)
settings = get_settings()
database_url = get_database_url(settings)

# Note: We don't use config.set_main_option for the URL because the OAuth
# token contains % characters that ConfigParser interprets as interpolation.
# Instead, we use the database_url directly in run_async_migrations.

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    # Ensure database exists first (idempotent)
    from deep_research.db.bootstrap import ensure_database_exists

    await ensure_database_exists(settings)

    # For Lakebase: use SSL (asyncpg doesn't accept sslmode URL param)
    connect_args = {"ssl": True} if settings.use_lakebase else {}

    connectable = create_async_engine(
        database_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
