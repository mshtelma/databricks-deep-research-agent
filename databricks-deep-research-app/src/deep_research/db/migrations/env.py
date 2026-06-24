"""Alembic environment configuration for async migrations."""

import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from deep_research.core.config import get_settings
from deep_research.db.asyncpg_config import (
    lakebase_asyncpg_connect_args,
    lakebase_engine_kwargs,
)
from deep_research.db.base import Base
from deep_research.db.session import get_database_url

logger = logging.getLogger(__name__)

# Import all models to ensure they're registered with Base.metadata
from deep_research.models import (  # noqa: E402, F401
    audit_log,
    chat,
    message,
    message_feedback,
    research_session,
    skill,
    source,
    user,
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


async def _fix_unknown_revision(connection: AsyncConnection) -> None:
    """Detect and fix unknown alembic revisions from consolidated feature branches.

    When feature branch migrations (e.g. 016, 017, 019) get deployed to shared
    databases and then squashed into a single migration on main (e.g. 014),
    alembic's revision graph breaks. This function detects that situation and
    stamps the database to head so migrations can proceed.
    """
    # Check if alembic_version table exists (fresh DB → nothing to fix)
    result = await connection.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'alembic_version'"
        )
    )
    if not result.scalar():
        return

    # Get all stamped revisions
    result = await connection.execute(text("SELECT version_num FROM alembic_version"))
    rows = result.fetchall()
    if not rows:
        return

    # Check each revision against the script directory
    script = ScriptDirectory.from_config(config)
    unknown_revisions: list[str] = []
    for (version_num,) in rows:
        try:
            script.get_revision(version_num)
        except CommandError:
            unknown_revisions.append(version_num)

    if not unknown_revisions:
        return

    # Stamp to head — replaces all rows with the current head revision
    head = script.get_current_head()
    logger.warning(
        "ALEMBIC_UNKNOWN_REVISION detected=%s stamping_to='%s'",
        unknown_revisions,
        head,
    )
    await connection.execute(text("DELETE FROM alembic_version"))
    await connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:head)"),
        {"head": head},
    )
    await connection.commit()


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

    # PgBouncer-safe connect args (see db/asyncpg_config.py).
    connect_args = lakebase_asyncpg_connect_args(settings)
    engine_kwargs = lakebase_engine_kwargs(settings)

    connectable = create_async_engine(
        database_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
        **engine_kwargs,
    )

    async with connectable.connect() as connection:
        await _fix_unknown_revision(connection)
        await connection.run_sync(do_run_migrations)
        # Explicit commit required for async engines — without this, SQLAlchemy
        # issues ROLLBACK when the connection context exits, silently discarding
        # all DDL that Alembic executed via run_sync.
        await connection.commit()

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
