"""Configure Databricks App access to a Lakebase Autoscaling database.

Read side (``--print-bundle-vars`` and the underlying helpers) is the
canonical path: it resolves the DAB ``postgres_branch`` and
``postgres_database_resource`` variables from the live workspace so the
Makefile can pass them as ``--var`` arguments to ``databricks bundle
deploy``.  The DAB ``lakebase-postgres`` app resource block in
``databricks.yml`` is now the single source of truth for the app↔Lakebase
binding.

Write side (:func:`configure_app_postgres_resource`) is **deprecated**: it
upserts the binding directly through the Apps API, which Terraform then
discovers on refresh and tries to remove on the next ``bundle deploy``,
producing the opaque ``failed to update app`` error that originally
motivated this refactor.  The function is retained for one release as an
explicit, operator-invoked repair tool only; it is no longer part of the
``make deploy`` pipeline and will be removed in a follow-up.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import warnings
from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App,
    AppResource,
    AppResourcePostgres,
    AppResourcePostgresPostgresPermission,
)

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_NAME = "lakebase-postgres"
BUNDLE_VAR_ORDER = ("postgres_branch", "postgres_database_resource")

_WRITE_SIDE_DEPRECATION_MSG = (
    "configure_app_postgres_resource() is deprecated. The lakebase-postgres "
    "binding is now managed by DAB via the lakebase-postgres app resource "
    "in databricks.yml. Use --print-bundle-vars (or the helper "
    "resolve_postgres_bundle_vars) to resolve postgres_branch and "
    "postgres_database_resource, then pass them as --var to "
    "'databricks bundle deploy'. This function will be removed in a "
    "follow-up release."
)


def branch_name_from_endpoint(endpoint_name: str) -> str:
    """Return ``projects/.../branches/...`` from a Postgres endpoint resource."""
    marker = "/endpoints/"
    if marker not in endpoint_name:
        raise ValueError(
            "ENDPOINT_NAME must be a Postgres endpoint resource path containing "
            f"{marker!r}; got {endpoint_name!r}"
        )
    return endpoint_name.split(marker, 1)[0]


def find_database_resource_name(databases: Iterable[Any], database_name: str) -> str:
    """Find the Databricks resource name for a Postgres database."""
    for database in databases:
        status = getattr(database, "status", None)
        postgres_database = getattr(status, "postgres_database", None)
        if postgres_database == database_name:
            name = getattr(database, "name", None)
            if not name:
                raise ValueError(
                    f"Database {database_name!r} matched but had no resource name"
                )
            return str(name)

    raise ValueError(
        f"Could not find Lakebase database {database_name!r}. "
        "Run database bootstrap before configuring app resources."
    )


def build_postgres_resource(
    *,
    resource_name: str,
    branch_name: str,
    database_resource_name: str,
) -> AppResource:
    """Build the App resource that grants the app access to Postgres."""
    return AppResource(
        name=resource_name,
        description="Lakebase Postgres database used by the app",
        postgres=AppResourcePostgres(
            branch=branch_name,
            database=database_resource_name,
            permission=AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE,
        ),
    )


def resolve_postgres_bundle_vars(
    *,
    client: WorkspaceClient,
    endpoint_name: str,
    database_name: str,
) -> dict[str, str]:
    """Resolve the DAB variables required for the app Postgres binding."""
    branch_name = branch_name_from_endpoint(endpoint_name)
    database_resource_name = find_database_resource_name(
        client.postgres.list_databases(parent=branch_name),
        database_name,
    )
    return {
        "postgres_branch": branch_name,
        "postgres_database_resource": database_resource_name,
    }


def format_bundle_var_args(bundle_vars: dict[str, str]) -> str:
    """Return shell-safe ``databricks bundle`` var arguments."""
    args: list[str] = []
    for key in BUNDLE_VAR_ORDER:
        value = bundle_vars[key]
        args.extend(["--var", f"{key}={value}"])
    return shlex.join(args)


def upsert_postgres_resource(
    resources: Iterable[AppResource] | None,
    postgres_resource: AppResource,
) -> list[AppResource]:
    """Return app resources with ``postgres_resource`` inserted or replaced."""
    updated: list[AppResource] = []
    replaced = False
    for resource in resources or []:
        if resource.name == postgres_resource.name:
            updated.append(postgres_resource)
            replaced = True
        else:
            updated.append(resource)

    if not replaced:
        updated.append(postgres_resource)

    return updated


def configure_app_postgres_resource(
    *,
    app_name: str,
    profile: str,
    endpoint_name: str,
    database_name: str,
    resource_name: str = DEFAULT_RESOURCE_NAME,
    timeout_seconds: int = 1200,
) -> str:
    """Attach an app-level Postgres resource for a Lakebase Autoscaling DB.

    .. deprecated::
        The lakebase-postgres binding is now managed by DAB. Mutating the
        app through the Apps API creates terraform drift on the next
        ``bundle deploy``. Use ``--print-bundle-vars`` to resolve binding
        vars and let DAB own the resource. This function is retained as an
        explicit operator repair tool and will be removed in a follow-up.
    """
    warnings.warn(
        _WRITE_SIDE_DEPRECATION_MSG, DeprecationWarning, stacklevel=2
    )
    logger.warning(_WRITE_SIDE_DEPRECATION_MSG)
    client = WorkspaceClient(profile=profile)
    branch_name = branch_name_from_endpoint(endpoint_name)
    database_resource_name = find_database_resource_name(
        client.postgres.list_databases(parent=branch_name),
        database_name,
    )

    app = client.apps.get(app_name)
    postgres_resource = build_postgres_resource(
        resource_name=resource_name,
        branch_name=branch_name,
        database_resource_name=database_resource_name,
    )
    resources = upsert_postgres_resource(app.resources, postgres_resource)

    waiter = client.apps.create_update(
        app_name,
        "resources",
        app=App(name=app_name, resources=resources),
    )
    waiter.result(timeout=timedelta(seconds=timeout_seconds))
    return database_resource_name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Grant a Databricks App access to a Lakebase Postgres database."
    )
    parser.add_argument("--app-name")
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        required=os.environ.get("DATABRICKS_CONFIG_PROFILE") is None,
    )
    parser.add_argument(
        "--endpoint-name",
        default=os.environ.get("ENDPOINT_NAME"),
        required=os.environ.get("ENDPOINT_NAME") is None,
    )
    parser.add_argument(
        "--database-name",
        default=os.environ.get("LAKEBASE_DATABASE", "deep_research"),
    )
    parser.add_argument("--resource-name", default=DEFAULT_RESOURCE_NAME)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--print-bundle-vars",
        action="store_true",
        help=(
            "Print DAB --var arguments for a DAB-managed app Postgres resource "
            "instead of mutating the app through the Apps API."
        ),
    )
    args = parser.parse_args(argv)
    if not args.print_bundle_vars and not args.app_name:
        parser.error("--app-name is required unless --print-bundle-vars is set")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    if args.print_bundle_vars:
        client = WorkspaceClient(profile=args.profile)
        bundle_vars = resolve_postgres_bundle_vars(
            client=client,
            endpoint_name=args.endpoint_name,
            database_name=args.database_name,
        )
        print(format_bundle_var_args(bundle_vars))
        return

    database_resource_name = configure_app_postgres_resource(
        app_name=args.app_name,
        profile=args.profile,
        endpoint_name=args.endpoint_name,
        database_name=args.database_name,
        resource_name=args.resource_name,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        "Configured Databricks App Postgres resource "
        f"{args.resource_name!r} for {database_resource_name}"
    )


if __name__ == "__main__":
    main()
