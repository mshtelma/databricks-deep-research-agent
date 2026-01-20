"""
Migration Utilities
===================

Provides utilities for running database migrations with support
for multiple migration paths (parent + child projects).
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_migration_paths(
    include_parent: bool = True,
    child_paths: list[str] | None = None,
) -> list[str]:
    """Get list of migration paths to use.

    Combines parent package migrations with child project migrations.

    Args:
        include_parent: Whether to include parent package migrations
        child_paths: Additional migration paths from child projects

    Returns:
        List of migration directory paths
    """
    paths = []

    if include_parent:
        # Find parent package migrations
        try:
            import deep_research

            parent_path = Path(deep_research.__file__).parent
            migrations_path = parent_path / "db" / "migrations"
            if migrations_path.exists():
                paths.append(str(migrations_path))
                logger.debug(
                    "Found parent migrations at: %s",
                    migrations_path,
                )
            else:
                # Try alternative locations
                alt_path = parent_path.parent / "db" / "migrations"
                if alt_path.exists():
                    paths.append(str(alt_path))
                    logger.debug(
                        "Found parent migrations at: %s",
                        alt_path,
                    )
        except ImportError:
            logger.debug("deep_research package not found")

    if child_paths:
        for path in child_paths:
            if os.path.exists(path):
                paths.append(path)
                logger.debug("Added child migration path: %s", path)
            else:
                logger.warning("Migration path not found: %s", path)

    return paths


async def run_migrations(
    migration_paths: list[str] | None = None,
    database_url: str | None = None,
    alembic_config_path: str | None = None,
    revision: str = "head",
) -> bool:
    """Run database migrations.

    Supports multiple migration paths for parent + child project scenarios.

    Args:
        migration_paths: List of migration directories
        database_url: Database connection URL (or from env)
        alembic_config_path: Path to alembic.ini (optional)
        revision: Target revision (default: "head")

    Returns:
        True if migrations succeeded, False otherwise
    """
    try:
        from alembic import command
        from alembic.config import Config

        # Get database URL
        if database_url is None:
            database_url = os.environ.get("DATABASE_URL")
            if not database_url:
                logger.error("DATABASE_URL not set")
                return False

        # Create Alembic config
        if alembic_config_path and os.path.exists(alembic_config_path):
            config = Config(alembic_config_path)
        else:
            config = Config()

        # Set database URL
        config.set_main_option("sqlalchemy.url", database_url)

        # Get migration paths
        if migration_paths is None:
            migration_paths = get_migration_paths()

        if not migration_paths:
            logger.error("No migration paths found")
            return False

        # Set version_locations for multi-path support
        version_locations = " ".join(migration_paths)
        config.set_main_option("version_locations", version_locations)

        logger.info(
            "Running migrations with paths: %s",
            migration_paths,
        )

        # Run migrations
        command.upgrade(config, revision)

        logger.info("Migrations completed successfully")
        return True

    except Exception as e:
        logger.error("Migration failed: %s", str(e))
        return False


def run_migrations_sync(
    migration_paths: list[str] | None = None,
    database_url: str | None = None,
    alembic_config_path: str | None = None,
    revision: str = "head",
) -> bool:
    """Synchronous version of run_migrations.

    Args:
        migration_paths: List of migration directories
        database_url: Database connection URL
        alembic_config_path: Path to alembic.ini
        revision: Target revision

    Returns:
        True if migrations succeeded
    """
    import asyncio

    return asyncio.run(
        run_migrations(
            migration_paths=migration_paths,
            database_url=database_url,
            alembic_config_path=alembic_config_path,
            revision=revision,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for running migrations.

    Usage:
        python -m deep_research.deployment.migrations run [--paths PATH1 PATH2]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "command",
        choices=["run", "paths"],
        help="Command to execute",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Additional migration paths",
    )
    parser.add_argument(
        "--database-url",
        help="Database URL (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--revision",
        default="head",
        help="Target revision (default: head)",
    )
    parser.add_argument(
        "--no-parent",
        action="store_true",
        help="Don't include parent package migrations",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "paths":
        paths = get_migration_paths(
            include_parent=not args.no_parent,
            child_paths=args.paths,
        )
        for path in paths:
            print(path)
    elif args.command == "run":
        paths = get_migration_paths(
            include_parent=not args.no_parent,
            child_paths=args.paths,
        )
        success = run_migrations_sync(
            migration_paths=paths,
            database_url=args.database_url,
            revision=args.revision,
        )
        if not success:
            exit(1)


if __name__ == "__main__":
    main()
