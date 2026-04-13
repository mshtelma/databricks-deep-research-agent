"""
Permission Utilities
====================

Provides utilities for granting database permissions to
app service principals. Supports both Provisioned and Autoscaling backends.
"""

import asyncio
import logging
from typing import Any

from deep_research.db.grant_permissions import _validate_sql_identifier
from deep_research.deployment.lakebase_connection import (
    get_lakebase_connection_info,
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
    instance_name: str | None = None,
    database_name: str = "deep_research",
    app_name: str = "deep-research-agent",
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Grant database permissions to an app's service principal.

    Grants ALL privileges on all tables and sequences to the app,
    plus sets default privileges for future objects.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        database_name: Name of the database
        app_name: Name of the Databricks App
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if permissions were granted, False on error
    """
    try:
        import asyncpg  # type: ignore[import-untyped]

        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        # Get service principal for app
        sp_name = await get_app_service_principal(app_name, workspace_client)
        if not sp_name:
            logger.error("Could not get service principal for app '%s'", app_name)
            return False

        # Validate SP name to prevent SQL injection in DDL statements
        _validate_sql_identifier(sp_name, "service principal name")

        logger.info(
            "Granting permissions to service principal '%s' on '%s.%s'",
            sp_name,
            instance_name or endpoint_name,
            database_name,
        )

        # Get connection info via shared helper (handles both backends)
        info = get_lakebase_connection_info(
            instance_name=instance_name,
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
        )

        # Connect to the target database
        conn = await asyncpg.connect(
            host=info.host,
            port=info.port,
            user=info.username,
            password=info.token,
            database=database_name,
            ssl="require",
        )
        try:
            # Quote the service principal name for SQL safety
            # Service principal names are typically like "user@domain.com"
            quoted_sp = f'"{sp_name}"'

            # For Autoscaling, create role via databricks_auth extension
            if endpoint_name:
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
                    await conn.execute(
                        "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')",
                        sp_name,
                    )
                    logger.debug("Created Autoscaling role for %s", sp_name)
                except Exception as e:
                    logger.warning(
                        "Autoscaling role creation failed for %s: %s. "
                        "This may require project-owner privileges.",
                        sp_name,
                        e,
                    )
            else:
                # Provisioned: Create role if it doesn't exist
                # PostgreSQL requires roles to exist before GRANT can target them
                # Lakebase creates app roles on first connection, but deployment
                # runs before the app starts, so we need to create the role ourselves
                # SECURITY INVARIANT: DDL statements below use f-string interpolation with
                # double-quoted identifiers. Safe ONLY because _validate_sql_identifier()
                # restricts input to [a-zA-Z0-9_\-\.]. If regex is widened, re-audit for injection.
                try:
                    await conn.execute(f"CREATE ROLE {quoted_sp} WITH LOGIN")
                    logger.debug("Created role %s", sp_name)
                except asyncpg.exceptions.DuplicateObjectError:
                    logger.debug("Role %s already exists", sp_name)

            # SECURITY INVARIANT: DDL statements below use f-string interpolation with
            # double-quoted identifiers. Safe ONLY because _validate_sql_identifier()
            # restricts input to [a-zA-Z0-9_\-\.]. If regex is widened, re-audit for injection.

            # Grant on existing tables (least privilege: no TRUNCATE/TRIGGER)
            await conn.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {quoted_sp}"
            )
            logger.debug("Granted SELECT/INSERT/UPDATE/DELETE on tables")

            # Grant on existing sequences (least privilege: USAGE/SELECT/UPDATE only)
            await conn.execute(
                f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO {quoted_sp}"
            )
            logger.debug("Granted USAGE, SELECT, UPDATE on sequences")

            # Set default privileges for future tables (least privilege)
            await conn.execute(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {quoted_sp}"
            )
            logger.debug("Set default privileges for tables")

            # Set default privileges for future sequences (least privilege)
            await conn.execute(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {quoted_sp}"
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
    instance_name: str | None = None,
    database_name: str = "deep_research",
    app_name: str = "deep-research-agent",
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Synchronous version of grant_to_app.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        database_name: Name of the database
        app_name: Name of the Databricks App
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if permissions were granted
    """
    return asyncio.run(
        grant_to_app(
            instance_name=instance_name,
            database_name=database_name,
            app_name=app_name,
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for permission operations.

    Usage:
        python -m deep_research.deployment.permissions grant <instance> <database> <app>
        python -m deep_research.deployment.permissions grant --endpoint-name <ep> <database> <app>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Grant database permissions")
    parser.add_argument(
        "command",
        choices=["grant", "sp-name"],
        help="Command to execute",
    )
    parser.add_argument("instance_name", nargs="?", help="Lakebase instance name (Provisioned)")
    parser.add_argument("database_name", help="Database name")
    parser.add_argument("app_name", help="Databricks App name")
    parser.add_argument(
        "--endpoint-name",
        help="Autoscaling endpoint name (alternative to instance_name)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "grant":
        success = grant_to_app_sync(
            instance_name=args.instance_name,
            database_name=args.database_name,
            app_name=args.app_name,
            endpoint_name=args.endpoint_name,
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
