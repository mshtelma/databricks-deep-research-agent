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
import ssl

import asyncpg  # type: ignore[import-untyped]
from databricks.sdk import WorkspaceClient

from deep_research.core.config import Settings, get_settings
from deep_research.db.session import get_credential_provider

logger = logging.getLogger(__name__)


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
        if app.name and app_name in app.name:
            # Get the service principal associated with this app
            if app.service_principal_id:
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
        logger.warning(
            f"Could not get service principal by ID {app_sp_id}: {e}. "
            "Trying to find by listing..."
        )
        # Fallback: list all service principals and find by ID
        for sp in ws.service_principals.list():
            if sp.id and str(sp.id) == str(app_sp_id):
                if sp.application_id:
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

    logger.info(f"Service principal username for grants: {sp_username}")

    # Connect to database with developer credentials
    provider = get_credential_provider(settings)
    if not provider:
        raise RuntimeError("Lakebase credential provider not available")

    cred = provider.get_credential()
    host = provider._get_instance_host()

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    logger.info(f"Connecting to {settings.lakebase_database} at {host}...")

    conn = await asyncpg.connect(
        host=host,
        port=settings.lakebase_port,
        user=cred.username,
        password=cred.token,
        database=settings.lakebase_database,
        ssl=ssl_context,
    )

    try:
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
        logger.info(f"Setting default privileges for future tables...")
        await conn.execute(
            f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT ALL ON TABLES TO "{sp_username}"
            '''
        )
        logger.info("Set default privileges for tables")

        # Set default privileges for future sequences
        logger.info(f"Setting default privileges for future sequences...")
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
