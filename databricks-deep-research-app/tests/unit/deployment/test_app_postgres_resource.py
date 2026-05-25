"""Tests for Databricks App Lakebase resource configuration."""

import warnings
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.service.apps import (
    AppResource,
    AppResourceSecret,
    AppResourceSecretSecretPermission,
)

from deep_research.deployment.app_postgres_resource import (
    _WRITE_SIDE_DEPRECATION_MSG,
    branch_name_from_endpoint,
    build_postgres_resource,
    configure_app_postgres_resource,
    find_database_resource_name,
    format_bundle_var_args,
    resolve_postgres_bundle_vars,
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


def test_resolve_postgres_bundle_vars() -> None:
    databases = [
        SimpleNamespace(
            name="projects/p/branches/production/databases/db-app",
            status=SimpleNamespace(postgres_database="deep_research"),
        )
    ]
    client = SimpleNamespace(
        postgres=SimpleNamespace(list_databases=lambda parent: databases)
    )

    bundle_vars = resolve_postgres_bundle_vars(
        client=client,
        endpoint_name="projects/p/branches/production/endpoints/primary",
        database_name="deep_research",
    )

    assert bundle_vars == {
        "postgres_branch": "projects/p/branches/production",
        "postgres_database_resource": (
            "projects/p/branches/production/databases/db-app"
        ),
    }


def test_format_bundle_var_args() -> None:
    args = format_bundle_var_args(
        {
            "postgres_branch": "projects/p/branches/production",
            "postgres_database_resource": (
                "projects/p/branches/production/databases/db-app"
            ),
        }
    )

    assert args == (
        "--var postgres_branch=projects/p/branches/production "
        "--var postgres_database_resource=projects/p/branches/production/databases/db-app"
    )


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


def test_configure_app_postgres_resource_emits_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Apps-API write side is deprecated — calling it must fire a
    ``DeprecationWarning`` so any surviving caller is loud in CI.
    """
    fake_app = SimpleNamespace(resources=[])

    class _FakeWaiter:
        def result(self, timeout: timedelta) -> None:
            return None

    fake_client = MagicMock()
    fake_client.postgres.list_databases.return_value = [
        SimpleNamespace(
            name="projects/p/branches/production/databases/db-app",
            status=SimpleNamespace(postgres_database="deep_research"),
        )
    ]
    fake_client.apps.get.return_value = fake_app
    fake_client.apps.create_update.return_value = _FakeWaiter()

    with patch(
        "deep_research.deployment.app_postgres_resource.WorkspaceClient",
        return_value=fake_client,
    ), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        configure_app_postgres_resource(
            app_name="deep-research-agent-ais",
            profile="ais",
            endpoint_name=(
                "projects/p/branches/production/endpoints/primary"
            ),
            database_name="deep_research",
        )

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings, (
        "configure_app_postgres_resource must fire a DeprecationWarning. "
        "If you intentionally removed the warning, also remove this test."
    )
    assert any(
        "lakebase-postgres" in str(w.message)
        for w in deprecation_warnings
    ), (
        "Deprecation message should explain that the lakebase-postgres "
        "binding is now DAB-managed. Current messages: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )


def test_deprecation_message_mentions_alternative() -> None:
    """The deprecation message must point at the supported replacement
    so an operator hitting it knows what to do."""
    assert "--print-bundle-vars" in _WRITE_SIDE_DEPRECATION_MSG
    assert "DAB" in _WRITE_SIDE_DEPRECATION_MSG
