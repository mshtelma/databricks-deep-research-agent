"""Grant table permissions to app service principal.

This module grants database table permissions to an app's service principal
after migrations have created tables. This is necessary because:

1. Developer runs migrations → tables are owned by developer
2. App's service principal has CAN_CONNECT_AND_CREATE on database
3. CAN_CONNECT_AND_CREATE doesn't grant SELECT/INSERT/UPDATE/DELETE on tables
4. Explicit GRANT statements are needed for the app to access tables
"""

import asyncio
import logging
import re

import asyncpg  # type: ignore[import-untyped]
from databricks.sdk import WorkspaceClient

from deep_research.core.config import Settings, get_settings
from deep_research.db.session import get_credential_provider

logger = logging.getLogger(__name__)

_SQL_IDENTIFIER_RE = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def _validate_sql_identifier(name: str, label: str = "identifier") -> str:
    """Validate a string is safe for use as a SQL identifier.

    Only allows alphanumeric, underscore, hyphen, and dot characters.
    Raises ValueError if the name contains dangerous characters.
    """
    if not name or not _SQL_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe SQL {label}: {name!r}. "
            f"Only alphanumeric, underscore, hyphen, and dot are allowed."
        )
    return name


async def grant_permissions_to_app(
    app_name: str,
    settings: Settings | None = None,
) -> None:
    """Grant all table permissions to the app's service principal.

    This must be run after migrations (which create tables owned by developer)
    to allow the app's service principal to access those tables.

    Args:
        app_name: Name of the Databricks App (or substring to match).
        settings: Application settings. Uses get_settings() if not provided.

    Raises:
        RuntimeError: If app or service principal not found, or grants fail.
    """
    if settings is None:
        settings = get_settings()

    if not settings.use_lakebase:
        logger.info("Not using Lakebase, skipping permission grants")
        return

    # Get app's service principal from Databricks
    ws = WorkspaceClient(profile=settings.databricks_config_profile)

    # List apps and find the matching one
    logger.info(f"Looking for app matching: {app_name}")
    apps = list(ws.apps.list())
    app_sp_id = None
    matched_app_name = None

    for app in apps:
        if app.name and app_name in app.name and app.service_principal_id:
            app_sp_id = app.service_principal_id
            matched_app_name = app.name
            logger.info(
                f"Found app '{app.name}' with service principal ID: {app_sp_id}"
            )
            break

    if not app_sp_id:
        available_apps = [a.name for a in apps if a.name]
        raise RuntimeError(
            f"Could not find app matching '{app_name}' or its service principal. "
            f"Available apps: {available_apps}"
        )

    # Get the service principal's application_id (used as username for Lakebase)
    # The service_principal_id from apps API is a numeric Databricks internal ID
    # We need to look up the service principal to get its application_id (client ID)
    sp_username = None
    try:
        # First try to get by ID (convert to string as API expects string)
        sp = ws.service_principals.get(str(app_sp_id))
        if sp.application_id:
            sp_username = sp.application_id
            logger.info(f"Found service principal application_id: {sp_username}")
    except Exception as e:
        logger.debug(
            f"Could not get service principal by ID {app_sp_id}: {e}. "
            "Trying to find by listing..."
        )
        # Fallback: list all service principals and find by ID
        for sp in ws.service_principals.list():
            if sp.id and str(sp.id) == str(app_sp_id) and sp.application_id:
                sp_username = sp.application_id
                logger.info(f"Found service principal via list: {sp_username}")
                break

    if not sp_username:
        # Last resort: use the numeric ID directly as username
        # This might work if Lakebase accepts numeric IDs
        logger.warning(
            f"Could not find application_id for service principal {app_sp_id}. "
            f"Using numeric ID as username."
        )
        sp_username = str(app_sp_id)

    _validate_sql_identifier(sp_username, "service principal username")
    logger.info(f"Service principal username for grants: {sp_username}")

    # Connect to database with developer credentials
    provider = get_credential_provider(settings)
    if not provider:
        raise RuntimeError("Lakebase credential provider not available")

    cred = provider.get_credential()
    host = provider.get_host()
    port = provider.get_port()

    logger.info(f"Connecting to {settings.lakebase_database} at {host}...")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=cred.username,
        password=cred.token,
        database=settings.lakebase_database,
        ssl="require",
    )

    try:
        # Autoscaling requires explicit role creation via databricks_auth extension
        if provider.get_backend_type() == "autoscaling":
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
                # Check existence first — pg_roles is readable by all Postgres users
                try:
                    role_exists = await conn.fetchval(
                        "SELECT 1 FROM pg_roles WHERE rolname = $1", sp_username
                    )
                except Exception as check_err:
                    logger.debug(
                        "pg_roles existence check failed for %s: %s — will attempt creation",
                        sp_username,
                        check_err,
                    )
                    role_exists = False

                if not role_exists:
                    await conn.execute(
                        "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')",
                        sp_username,
                    )
                    logger.info(f"Created Autoscaling role for {sp_username}")
                else:
                    logger.info(f"Autoscaling role already exists for {sp_username}, skipping")
            except Exception as e:
                logger.warning(
                    "Autoscaling role creation failed for %s: %s. "
                    "This may require project-owner privileges. "
                    "Try creating the role manually in the Lakebase SQL Editor.",
                    sp_username,
                    e,
                )

        # Grant permissions on existing tables
        logger.info(f"Granting ALL on all tables to {sp_username}...")
        await conn.execute(
            f'GRANT ALL ON ALL TABLES IN SCHEMA public TO "{sp_username}"'
        )
        logger.info("Granted ALL on all tables")

        # Grant permissions on existing sequences
        logger.info(f"Granting ALL on all sequences to {sp_username}...")
        await conn.execute(
            f'GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO "{sp_username}"'
        )
        logger.info("Granted ALL on all sequences")

        # Set default privileges for future tables created by current user
        logger.info("Setting default privileges for future tables...")
        await conn.execute(
            f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL ON TABLES TO "{sp_username}"
            '''
        )
        logger.info("Set default privileges for tables")

        # Set default privileges for future sequences
        logger.info("Setting default privileges for future sequences...")
        await conn.execute(
            f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL ON SEQUENCES TO "{sp_username}"
            '''
        )
        logger.info("Set default privileges for sequences")

    finally:
        await conn.close()

    logger.info(
        f"All permissions granted to service principal for app '{matched_app_name}'"
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.db.grant_permissions <app_name>")
        print("  app_name: Name of the Databricks App (or substring to match)")
        sys.exit(1)

    asyncio.run(grant_permissions_to_app(sys.argv[1]))
