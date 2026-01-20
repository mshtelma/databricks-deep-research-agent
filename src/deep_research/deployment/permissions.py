"""
Permission Utilities
====================

Provides utilities for granting database permissions to
app service principals.
"""

import asyncio
import logging
from typing import Any

from deep_research.deployment.lakebase_connection import (
    extract_username_from_token,
    get_lakebase_host,
)

logger = logging.getLogger(__name__)


async def get_app_service_principal(
    app_name: str,
    workspace_client: Any | None = None,
) -> str | None:
    """Get the service principal name for a Databricks App.

    Args:
        app_name: Name of the Databricks App
        workspace_client: Optional WorkspaceClient

    Returns:
        Service principal name, or None if not found
    """
    try:
        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        # Get app details
        app = workspace_client.apps.get(name=app_name)
        if app and app.service_principal_name:
            return app.service_principal_name

        logger.warning("App '%s' has no service principal", app_name)
        return None

    except Exception as e:
        logger.error(
            "Error getting service principal for app '%s': %s",
            app_name,
            str(e),
        )
        return None


async def grant_to_app(
    instance_name: str,
    database_name: str,
    app_name: str,
    workspace_client: Any | None = None,
) -> bool:
    """Grant database permissions to an app's service principal.

    Grants ALL privileges on all tables and sequences to the app,
    plus sets default privileges for future objects.

    Args:
        instance_name: Name of the Lakebase instance
        database_name: Name of the database
        app_name: Name of the Databricks App
        workspace_client: Optional WorkspaceClient

    Returns:
        True if permissions were granted, False on error
    """
    try:
        import asyncpg

        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        # Get service principal for app
        sp_name = await get_app_service_principal(app_name, workspace_client)
        if not sp_name:
            logger.error("Could not get service principal for app '%s'", app_name)
            return False

        logger.info(
            "Granting permissions to service principal '%s' on '%s.%s'",
            sp_name,
            instance_name,
            database_name,
        )

        # Get credentials
        cred = workspace_client.database.generate_database_credential(
            instance_names=[instance_name]
        )

        # Get correct hostname (from PGHOST or API lookup, not derived from instance name)
        host = get_lakebase_host(instance_name, workspace_client)

        # Get username from PGUSER or JWT token (not hardcoded "token")
        username = extract_username_from_token(cred.token)

        # Connect to the target database
        conn = await asyncpg.connect(
            host=host,
            port=5432,
            user=username,
            password=cred.token,
            database=database_name,
            ssl="require",
        )
        try:
            # Quote the service principal name for SQL safety
            # Service principal names are typically like "user@domain.com"
            quoted_sp = f'"{sp_name}"'

            # Grant on existing tables
            await conn.execute(
                f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {quoted_sp}"
            )
            logger.debug("Granted ALL on tables")

            # Grant on existing sequences
            await conn.execute(
                f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {quoted_sp}"
            )
            logger.debug("Granted ALL on sequences")

            # Set default privileges for future tables
            await conn.execute(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT ALL ON TABLES TO {quoted_sp}"
            )
            logger.debug("Set default privileges for tables")

            # Set default privileges for future sequences
            await conn.execute(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT ALL ON SEQUENCES TO {quoted_sp}"
            )
            logger.debug("Set default privileges for sequences")

            logger.info(
                "Successfully granted permissions to '%s'",
                sp_name,
            )
            return True

        finally:
            await conn.close()

    except Exception as e:
        logger.error("Error granting permissions: %s", str(e))
        return False


def grant_to_app_sync(
    instance_name: str,
    database_name: str,
    app_name: str,
    workspace_client: Any | None = None,
) -> bool:
    """Synchronous version of grant_to_app.

    Args:
        instance_name: Name of the Lakebase instance
        database_name: Name of the database
        app_name: Name of the Databricks App
        workspace_client: Optional WorkspaceClient

    Returns:
        True if permissions were granted
    """
    return asyncio.run(
        grant_to_app(
            instance_name=instance_name,
            database_name=database_name,
            app_name=app_name,
            workspace_client=workspace_client,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for permission operations.

    Usage:
        python -m deep_research.deployment.permissions grant <instance> <database> <app>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Grant database permissions")
    parser.add_argument(
        "command",
        choices=["grant", "sp-name"],
        help="Command to execute",
    )
    parser.add_argument("instance_name", help="Lakebase instance name")
    parser.add_argument("database_name", help="Database name")
    parser.add_argument("app_name", help="Databricks App name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "grant":
        success = grant_to_app_sync(
            instance_name=args.instance_name,
            database_name=args.database_name,
            app_name=args.app_name,
        )
        if not success:
            exit(1)
    elif args.command == "sp-name":
        sp_name = asyncio.run(get_app_service_principal(args.app_name))
        if sp_name:
            print(sp_name)
        else:
            exit(1)


if __name__ == "__main__":
    main()
