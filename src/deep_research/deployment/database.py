"""
Database Creation Utilities
===========================

Provides utilities for creating and managing databases on Lakebase.
"""

import asyncio
import logging
from typing import Any

from deep_research.deployment.lakebase_connection import (
    extract_username_from_token,
    get_lakebase_host,
)

logger = logging.getLogger(__name__)


async def database_exists(
    instance_name: str,
    database_name: str,
    workspace_client: Any | None = None,
) -> bool:
    """Check if a database exists on a Lakebase instance.

    Args:
        instance_name: Name of the Lakebase instance
        database_name: Name of the database to check
        workspace_client: Optional WorkspaceClient

    Returns:
        True if database exists, False otherwise
    """
    try:
        import asyncpg

        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        # Get credentials
        cred = workspace_client.database.generate_database_credential(
            instance_names=[instance_name]
        )

        # Get correct hostname (from PGHOST or API lookup, not derived from instance name)
        host = get_lakebase_host(instance_name, workspace_client)

        # Get username from PGUSER or JWT token (not hardcoded "token")
        username = extract_username_from_token(cred.token)

        # Connect to postgres database to check
        conn = await asyncpg.connect(
            host=host,
            port=5432,
            user=username,
            password=cred.token,
            database="postgres",
            ssl="require",
        )
        try:
            result = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                database_name,
            )
            return result == 1
        finally:
            await conn.close()

    except Exception as e:
        logger.debug("Error checking database existence: %s", str(e))
        return False


async def create_database(
    instance_name: str,
    database_name: str,
    if_not_exists: bool = True,
    workspace_client: Any | None = None,
) -> bool:
    """Create a database on a Lakebase instance.

    Args:
        instance_name: Name of the Lakebase instance
        database_name: Name of the database to create
        if_not_exists: If True, don't error if database exists (default True)
        workspace_client: Optional WorkspaceClient

    Returns:
        True if database was created or already exists, False on error
    """
    try:
        import asyncpg

        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        # Get credentials
        cred = workspace_client.database.generate_database_credential(
            instance_names=[instance_name]
        )

        # Get correct hostname (from PGHOST or API lookup, not derived from instance name)
        host = get_lakebase_host(instance_name, workspace_client)

        # Get username from PGUSER or JWT token (not hardcoded "token")
        username = extract_username_from_token(cred.token)

        # Connect to postgres database (default database for admin operations)
        conn = await asyncpg.connect(
            host=host,
            port=5432,
            user=username,
            password=cred.token,
            database="postgres",
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
                        instance_name,
                    )
                    return True
                else:
                    logger.error(
                        "Database '%s' already exists on '%s'",
                        database_name,
                        instance_name,
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
                instance_name,
            )
            return True

        finally:
            await conn.close()

    except Exception as e:
        logger.error("Error creating database: %s", str(e))
        return False


async def ensure_database_exists(
    instance_name: str,
    database_name: str,
    workspace_client: Any | None = None,
) -> bool:
    """Ensure a database exists, creating it if necessary.

    Convenience function that combines existence check and creation.

    Args:
        instance_name: Name of the Lakebase instance
        database_name: Name of the database
        workspace_client: Optional WorkspaceClient

    Returns:
        True if database exists or was created, False on error
    """
    return await create_database(
        instance_name=instance_name,
        database_name=database_name,
        if_not_exists=True,
        workspace_client=workspace_client,
    )


def create_database_sync(
    instance_name: str,
    database_name: str,
    if_not_exists: bool = True,
    workspace_client: Any | None = None,
) -> bool:
    """Synchronous version of create_database.

    Args:
        instance_name: Name of the Lakebase instance
        database_name: Name of the database to create
        if_not_exists: If True, don't error if database exists
        workspace_client: Optional WorkspaceClient

    Returns:
        True if database was created or already exists
    """
    return asyncio.run(
        create_database(
            instance_name=instance_name,
            database_name=database_name,
            if_not_exists=if_not_exists,
            workspace_client=workspace_client,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for database operations.

    Usage:
        python -m deep_research.deployment.database create <instance> <database>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Lakebase database operations")
    parser.add_argument(
        "command",
        choices=["create", "exists"],
        help="Command to execute",
    )
    parser.add_argument("instance_name", help="Lakebase instance name")
    parser.add_argument("database_name", help="Database name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "create":
        success = create_database_sync(
            instance_name=args.instance_name,
            database_name=args.database_name,
        )
        if not success:
            exit(1)
    elif args.command == "exists":
        exists = asyncio.run(
            database_exists(
                instance_name=args.instance_name,
                database_name=args.database_name,
            )
        )
        print(f"Database exists: {exists}")
        if not exists:
            exit(1)


if __name__ == "__main__":
    main()
