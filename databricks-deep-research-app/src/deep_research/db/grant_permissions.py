"""Grant table permissions to app service principal.

This module grants database table permissions to an app's service principal
after migrations have created tables. This is necessary because:

1. Developer runs migrations → tables are owned by developer
2. App's service principal has CAN_CONNECT_AND_CREATE on database
3. CAN_CONNECT_AND_CREATE doesn't grant SELECT/INSERT/UPDATE/DELETE on tables
4. Explicit GRANT statements are needed for the app to access tables

Scope notes:

* This module intentionally touches ONLY the ``public`` schema (legacy ORM
  tables created by alembic at deploy-time as the developer).
* The chat-document storage schema (``settings.storage_schema``, default
  ``deep_research_state``) is created and owned by the app SP itself at
  lifespan via ``LakebaseBackend.migrate()``. The developer cannot transfer
  ownership cross-principal on Lakebase — Postgres requires ``SET ROLE``
  on the new owner, which Lakebase does not grant between the developer
  and the app SP. Doing any CREATE SCHEMA / ALTER OWNER / ALTER TABLE
  OWNER on ``deep_research_state`` from here would leave the schema
  developer-owned and the app would hit "must be owner of table …" at
  startup. ``drop_storage_schema()`` below is the one-time recovery for
  a historically poisoned state.
"""

import asyncio
import logging
import os
import re

import asyncpg
from databricks.sdk import WorkspaceClient

from deep_research.core.config import Settings, get_settings
from deep_research.db.session import get_credential_provider

logger = logging.getLogger(__name__)

_SQL_IDENTIFIER_RE = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

# Opt-out env var for the strict role-existence check. When unset (default),
# a failed ``pg_roles`` query raises so the deploy fails loudly instead of
# silently shipping an app whose SP role does not exist in Lakebase. Set to
# any truthy value (``1``, ``true``, etc.) to fall back to the legacy
# "skip create on check failure" behaviour — useful only when an operator
# has already verified the role state out-of-band.
_TOLERATE_ROLE_CHECK_FAILURE_ENV = "GRANT_PERMISSIONS_TOLERATE_ROLE_CHECK_FAILURE"


def _tolerate_role_check_failure() -> bool:
    """Return True when the strict-mode opt-out env var is set to a truthy value."""
    raw = os.environ.get(_TOLERATE_ROLE_CHECK_FAILURE_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
        # Autoscaling requires explicit role creation via databricks_auth extension.
        if provider.get_backend_type() == "autoscaling":
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
            except Exception as e:
                logger.warning(
                    "Could not ensure databricks_auth extension: %s", e,
                )

            # Three-state role-existence guard. The naive path defaults to
            # role_exists=False on a pg_roles check failure and then calls
            # databricks_create_role. On Lakebase autoscaling that can
            # drop-and-recreate an already-existing role with a fresh OID,
            # orphaning every schema/table the prior OID owned and later
            # producing "permission denied for schema …" at app startup.
            # Three-state (True/False/None):
            #   True  → role exists; skip create (idempotent).
            #   False → role missing; create it.
            #   None  → verification failed. STRICT mode (default): raise so
            #           the deploy fails loudly instead of silently shipping
            #           an app whose role does not exist. Operators with
            #           verified out-of-band role state can opt back into
            #           the legacy skip-on-check-failure behaviour by setting
            #           ``GRANT_PERMISSIONS_TOLERATE_ROLE_CHECK_FAILURE=1``.
            role_exists: bool | None
            try:
                role_exists = bool(
                    await conn.fetchval(
                        "SELECT 1 FROM pg_roles WHERE rolname = $1", sp_username
                    )
                )
            except Exception as check_err:
                if _tolerate_role_check_failure():
                    logger.warning(
                        "pg_roles check failed for %s: %s — SKIPPING "
                        "databricks_create_role because "
                        "%s is set. If this is a fresh deploy, create the "
                        "role manually via the Lakebase SQL Editor.",
                        sp_username,
                        check_err,
                        _TOLERATE_ROLE_CHECK_FAILURE_ENV,
                    )
                    role_exists = None
                else:
                    raise RuntimeError(
                        f"pg_roles existence check failed for SP "
                        f"{sp_username!r}: {check_err!r}. Refusing to proceed "
                        f"because skipping role creation here would silently "
                        f"ship an app whose Lakebase role does not exist "
                        f"(symptom: 'password authentication failed for user "
                        f"{sp_username}' at runtime). Either: (a) verify the "
                        f"role exists in the Lakebase SQL Editor and re-run "
                        f"this command with {_TOLERATE_ROLE_CHECK_FAILURE_ENV}=1, "
                        f"or (b) create the role manually with "
                        f"`SELECT databricks_create_role('{sp_username}', "
                        f"'SERVICE_PRINCIPAL');`."
                    ) from check_err

            if role_exists is False:
                try:
                    await conn.execute(
                        "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')",
                        sp_username,
                    )
                    logger.info(f"Created Autoscaling role for {sp_username}")
                except Exception as e:
                    logger.warning(
                        "Autoscaling role creation failed for %s: %s. "
                        "This may require project-owner privileges. "
                        "Try creating the role manually in the Lakebase SQL Editor.",
                        sp_username,
                        e,
                    )
            elif role_exists is True:
                logger.info(
                    f"Autoscaling role already exists for {sp_username}, skipping"
                )
            # role_exists is None → opt-out branch taken above; skip create.

        # SECURITY INVARIANT: DDL statements below use f-string interpolation with
        # double-quoted identifiers. Safe ONLY because _validate_sql_identifier()
        # restricts input to [a-zA-Z0-9_\-\.]. If regex is widened, re-audit for injection.

        # Grant database-level CREATE so the app SP can create its own storage
        # schema (``settings.storage_schema``, default ``deep_research_state``)
        # at lifespan via ``LakebaseBackend.migrate()``. Without this grant,
        # ``CREATE SCHEMA IF NOT EXISTS`` fails at app startup with
        # ``InsufficientPrivilegeError: permission denied for database <db>`` —
        # even though the SP has Databricks ``CAN_CONNECT_AND_CREATE`` (which
        # maps only to Postgres ``CONNECT`` on Lakebase Autoscaling, not
        # ``CREATE``). CONNECT is included for idempotent self-documentation.
        # GRANT is idempotent in Postgres; safe to re-run.
        _validate_sql_identifier(settings.lakebase_database, "lakebase database")
        logger.info(
            "Granting CREATE, CONNECT on database %r to %s...",
            settings.lakebase_database, sp_username,
        )
        await conn.execute(
            f'GRANT CREATE, CONNECT ON DATABASE "{settings.lakebase_database}" '
            f'TO "{sp_username}"'
        )
        logger.info("Granted CREATE, CONNECT on database")

        # Grant permissions on existing tables (least privilege: no TRUNCATE/TRIGGER)
        logger.info(f"Granting SELECT/INSERT/UPDATE/DELETE on all tables to {sp_username}...")
        await conn.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{sp_username}"'
        )
        logger.info("Granted SELECT/INSERT/UPDATE/DELETE on all tables")

        # Grant permissions on existing sequences (least privilege: USAGE/SELECT/UPDATE only)
        logger.info(f"Granting USAGE, SELECT, UPDATE on all sequences to {sp_username}...")
        await conn.execute(
            f'GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO "{sp_username}"'
        )
        logger.info("Granted USAGE, SELECT, UPDATE on all sequences")

        # Set default privileges for future tables created by current user
        logger.info("Setting default privileges for future tables...")
        await conn.execute(
            f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{sp_username}"
            '''
        )
        logger.info("Set default privileges for tables")

        # Set default privileges for future sequences (least privilege)
        logger.info("Setting default privileges for future sequences...")
        await conn.execute(
            f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO "{sp_username}"
            '''
        )
        logger.info("Set default privileges for sequences")

        # Post-flight verification. Postgres GRANT statements succeed silently
        # against a nonexistent grantee in some PG versions/configurations, so
        # earlier success logs do NOT prove the role is usable. Re-query
        # pg_roles here to assert the SP can actually log in; raise a loud
        # error otherwise so the deploy fails BEFORE the app is started.
        # Mirrors strict-mode behaviour above: opt-out env var skips this too.
        if _tolerate_role_check_failure():
            logger.info(
                "Skipping post-flight role verification because %s is set",
                _TOLERATE_ROLE_CHECK_FAILURE_ENV,
            )
        else:
            try:
                rolcanlogin = await conn.fetchval(
                    "SELECT rolcanlogin FROM pg_roles WHERE rolname = $1",
                    sp_username,
                )
            except Exception as verify_err:
                raise RuntimeError(
                    f"Post-flight role verification failed for SP "
                    f"{sp_username!r}: {verify_err!r}. Grants may have been "
                    f"applied against a missing role; the app will fail with "
                    f"'password authentication failed' at runtime. Re-run "
                    f"this command after the Lakebase endpoint is healthy, or "
                    f"set {_TOLERATE_ROLE_CHECK_FAILURE_ENV}=1 if you have "
                    f"verified the role state out-of-band."
                ) from verify_err
            if rolcanlogin is None:
                raise RuntimeError(
                    f"Post-flight check: Lakebase role {sp_username!r} does "
                    f"NOT exist in database {settings.lakebase_database!r}. "
                    f"GRANT statements above ran but had no effective grantee; "
                    f"the app will hit 'password authentication failed for "
                    f"user {sp_username}' at first request. Create the role "
                    f"with `SELECT databricks_create_role('{sp_username}', "
                    f"'SERVICE_PRINCIPAL');` in the Lakebase SQL Editor and "
                    f"re-run this command."
                )
            if not rolcanlogin:
                raise RuntimeError(
                    f"Post-flight check: Lakebase role {sp_username!r} exists "
                    f"but rolcanlogin=False. The role cannot authenticate. "
                    f"Run `ALTER ROLE \"{sp_username}\" LOGIN;` in the "
                    f"Lakebase SQL Editor and re-run this command."
                )
            logger.info(
                "Post-flight verification OK for SP %s (rolcanlogin=true)",
                sp_username,
            )

    finally:
        await conn.close()

    logger.info(
        f"All permissions granted to service principal for app '{matched_app_name}'"
    )


async def drop_storage_schema(settings: Settings | None = None) -> None:
    """DESTRUCTIVE: drop ``settings.storage_schema`` as the current user.

    One-time recovery for the historical poisoned-schema state: when a
    previous deploy (or a local dev run of the app with developer creds)
    created ``deep_research_state`` owned by the developer, Postgres on
    Lakebase gives no way to transfer ownership to the app SP (``SET ROLE``
    cross-principal is denied). The only path forward is to DROP the
    schema as the current owner (developer) so the next app lifespan can
    recreate it under the SP.

    After this function runs successfully, run ``make deploy`` — the app's
    ``LakebaseBackend.migrate()`` at lifespan will recreate the schema and
    all tables owned by the SP.

    WARNING: Destroys every table in the storage schema
    (chat / user / prep_job / custom_agent / prompt_template / feedback /
    audit_log / uploaded_files / etc.). This is irreversible. Only run
    when you have confirmed the data is expendable, or when the current
    deploy is already broken and unrecoverable without it.

    Raises:
        RuntimeError: If Lakebase is not configured, the credential provider
            is unavailable, or the DROP SCHEMA is refused (e.g. the current
            user is not the owner of the schema).
    """
    if settings is None:
        settings = get_settings()
    if not settings.use_lakebase:
        raise RuntimeError("Not using Lakebase — nothing to reset.")

    provider = get_credential_provider(settings)
    if not provider:
        raise RuntimeError("Lakebase credential provider not available")

    cred = provider.get_credential()
    host = provider.get_host()
    port = provider.get_port()
    storage_schema = settings.storage_schema or "deep_research_state"
    _validate_sql_identifier(storage_schema, "storage schema")

    logger.warning(
        "DESTRUCTIVE: about to DROP SCHEMA %r CASCADE on database %r. "
        "All data in the schema will be permanently lost.",
        storage_schema, settings.lakebase_database,
    )

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=cred.username,
        password=cred.token,
        database=settings.lakebase_database,
        ssl="require",
    )
    try:
        await conn.execute(
            f'DROP SCHEMA IF EXISTS "{storage_schema}" CASCADE'
        )
        logger.info(
            "Dropped schema %r. Run `make deploy` to let the app SP "
            "recreate it owned by the SP.",
            storage_schema,
        )
    finally:
        await conn.close()


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
