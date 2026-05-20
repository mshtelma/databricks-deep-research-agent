"""Tests for Databricks App Lakebase resource configuration."""

from types import SimpleNamespace

import pytest
from databricks.sdk.service.apps import (
    AppResource,
    AppResourceSecret,
    AppResourceSecretSecretPermission,
)

from deep_research.deployment.app_postgres_resource import (
    branch_name_from_endpoint,
    build_postgres_resource,
    find_database_resource_name,
    upsert_postgres_resource,
)


def test_branch_name_from_endpoint() -> None:
    endpoint = "projects/deep-research-ais/branches/production/endpoints/primary"

    assert branch_name_from_endpoint(endpoint) == (
        "projects/deep-research-ais/branches/production"
    )


def test_branch_name_from_endpoint_rejects_invalid_resource() -> None:
    with pytest.raises(ValueError, match="ENDPOINT_NAME"):
        branch_name_from_endpoint("projects/deep-research-ais/branches/production")


def test_find_database_resource_name_matches_postgres_database() -> None:
    databases = [
        SimpleNamespace(
            name="projects/p/branches/b/databases/db-system",
            status=SimpleNamespace(postgres_database="databricks_postgres"),
        ),
        SimpleNamespace(
            name="projects/p/branches/b/databases/db-app",
            status=SimpleNamespace(postgres_database="deep_research"),
        ),
    ]

    assert find_database_resource_name(databases, "deep_research") == (
        "projects/p/branches/b/databases/db-app"
    )


def test_find_database_resource_name_errors_when_missing() -> None:
    databases = [
        SimpleNamespace(
            name="projects/p/branches/b/databases/db-system",
            status=SimpleNamespace(postgres_database="databricks_postgres"),
        )
    ]

    with pytest.raises(ValueError, match="deep_research"):
        find_database_resource_name(databases, "deep_research")


def test_upsert_postgres_resource_replaces_existing_resource() -> None:
    secret_resource = AppResource(
        name="brave-api-key",
        secret=AppResourceSecret(
            scope="deep-research-secrets",
            key="BRAVE_API_KEY",
            permission=AppResourceSecretSecretPermission.READ,
        ),
    )
    old_postgres_resource = build_postgres_resource(
        resource_name="lakebase-postgres",
        branch_name="projects/old/branches/production",
        database_resource_name="projects/old/branches/production/databases/db-old",
    )
    new_postgres_resource = build_postgres_resource(
        resource_name="lakebase-postgres",
        branch_name="projects/new/branches/production",
        database_resource_name="projects/new/branches/production/databases/db-new",
    )

    resources = upsert_postgres_resource(
        [secret_resource, old_postgres_resource],
        new_postgres_resource,
    )

    assert [resource.name for resource in resources] == [
        "brave-api-key",
        "lakebase-postgres",
    ]
    assert resources[1].postgres is not None
    assert resources[1].postgres.database == (
        "projects/new/branches/production/databases/db-new"
    )


def test_upsert_postgres_resource_appends_when_missing() -> None:
    postgres_resource = build_postgres_resource(
        resource_name="lakebase-postgres",
        branch_name="projects/p/branches/production",
        database_resource_name="projects/p/branches/production/databases/db-app",
    )

    resources = upsert_postgres_resource([], postgres_resource)

    assert resources == [postgres_resource]
