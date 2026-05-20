"""Configure Databricks App access to a Lakebase Autoscaling database."""

from __future__ import annotations

import argparse
import os
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

DEFAULT_RESOURCE_NAME = "lakebase-postgres"


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
    """Attach an app-level Postgres resource for a Lakebase Autoscaling DB."""
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
    parser.add_argument("--app-name", required=True)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
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
