"""
Database Creation Utilities
===========================

Provides utilities for creating and managing databases on Lakebase.
Supports both Provisioned and Autoscaling backends.
"""

import asyncio
import logging
from typing import Any

from deep_research.deployment.lakebase_connection import (
    get_lakebase_connection_info,
)

logger = logging.getLogger(__name__)


async def database_exists(
    instance_name: str | None = None,
    database_name: str = "deep_research",
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Check if a database exists on a Lakebase instance.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        database_name: Name of the database to check
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if database exists, False otherwise
    """
    try:
        import asyncpg  # type: ignore[import-untyped]

        info = get_lakebase_connection_info(
            instance_name=instance_name,
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
        )

        # Determine bootstrap database (Autoscaling uses databricks_postgres)
        bootstrap_db = "databricks_postgres" if endpoint_name else "postgres"

        conn = await asyncpg.connect(
            host=info.host,
            port=info.port,
            user=info.username,
            password=info.token,
            database=bootstrap_db,
            ssl="require",
        )
        try:
            result = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                database_name,
            )
            return bool(result == 1)
        finally:
            await conn.close()

    except Exception as e:
        logger.debug("Error checking database existence: %s", str(e))
        return False


async def create_database(
    instance_name: str | None = None,
    database_name: str = "deep_research",
    if_not_exists: bool = True,
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Create a database on a Lakebase instance.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        database_name: Name of the database to create
        if_not_exists: If True, don't error if database exists (default True)
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if database was created or already exists, False on error
    """
    try:
        import asyncpg

        info = get_lakebase_connection_info(
            instance_name=instance_name,
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
        )

        # Determine bootstrap database (Autoscaling uses databricks_postgres)
        bootstrap_db = "databricks_postgres" if endpoint_name else "postgres"

        conn = await asyncpg.connect(
            host=info.host,
            port=info.port,
            user=info.username,
            password=info.token,
            database=bootstrap_db,
            ssl="require",
        )
        try:
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                database_name,
            )

            if exists:
                if if_not_exists:
                    logger.info(
                        "Database '%s' already exists on '%s'",
                        database_name,
                        instance_name or endpoint_name,
                    )
                    return True
                else:
                    logger.error(
                        "Database '%s' already exists on '%s'",
                        database_name,
                        instance_name or endpoint_name,
                    )
                    return False

            # Create database
            # Note: CREATE DATABASE cannot be run in a transaction
            await conn.execute(
                f'CREATE DATABASE "{database_name}"'  # Safe because we control the name
            )
            logger.info(
                "Created database '%s' on '%s'",
                database_name,
                instance_name or endpoint_name,
            )
            return True

        finally:
            await conn.close()

    except Exception as e:
        logger.error("Error creating database: %s", str(e))
        return False


async def ensure_database_exists(
    instance_name: str | None = None,
    database_name: str = "deep_research",
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Ensure a database exists, creating it if necessary.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        database_name: Name of the database
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if database exists or was created, False on error
    """
    return await create_database(
        instance_name=instance_name,
        database_name=database_name,
        if_not_exists=True,
        workspace_client=workspace_client,
        endpoint_name=endpoint_name,
    )


def create_database_sync(
    instance_name: str | None = None,
    database_name: str = "deep_research",
    if_not_exists: bool = True,
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Synchronous version of create_database.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        database_name: Name of the database to create
        if_not_exists: If True, don't error if database exists
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if database was created or already exists
    """
    return asyncio.run(
        create_database(
            instance_name=instance_name,
            database_name=database_name,
            if_not_exists=if_not_exists,
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for database operations.

    Usage:
        python -m deep_research.deployment.database create <instance> <database>
        python -m deep_research.deployment.database create --endpoint-name <endpoint> <database>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Lakebase database operations")
    parser.add_argument(
        "command",
        choices=["create", "exists"],
        help="Command to execute",
    )
    parser.add_argument("instance_name", nargs="?", help="Lakebase instance name (Provisioned)")
    parser.add_argument("database_name", help="Database name")
    parser.add_argument(
        "--endpoint-name",
        help="Autoscaling endpoint name (alternative to instance_name)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "create":
        success = create_database_sync(
            instance_name=args.instance_name,
            database_name=args.database_name,
            endpoint_name=args.endpoint_name,
        )
        if not success:
            exit(1)
    elif args.command == "exists":
        exists = asyncio.run(
            database_exists(
                instance_name=args.instance_name,
                database_name=args.database_name,
                endpoint_name=args.endpoint_name,
            )
        )
        print(f"Database exists: {exists}")
        if not exists:
            exit(1)


if __name__ == "__main__":
    main()
