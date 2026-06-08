"""Unit tests for WorkspaceClientResolver (services/deployment/auth.py).

Covers the OBO-vs-SP precedence the resolver enforces. The resolver is
the single source of truth for which Databricks identity runs a deployment-
lifecycle SDK call — getting this wrong leaks resources owned by other
identities or fails to clean up user-created Apps.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from deep_research.services.deployment.auth import WorkspaceClientResolver


def test_prefers_obo_when_present() -> None:
    obo_client = MagicMock(name="obo_workspace_client")
    resolver = WorkspaceClientResolver(obo_client=obo_client)

    chosen = resolver.resolve(
        purpose="shell_app.deactivate", deployment_id=uuid4()
    )

    assert chosen is obo_client


def test_falls_back_to_sp_when_obo_none() -> None:
    """No OBO → reach into ``get_databricks_auth().get_client()``."""
    sp_client = MagicMock(name="sp_workspace_client")
    resolver = WorkspaceClientResolver(obo_client=None)

    with patch(
        "deep_research.core.databricks_auth.get_databricks_auth"
    ) as mock_get_auth:
        mock_get_auth.return_value.get_client.return_value = sp_client
        chosen = resolver.resolve(
            purpose="mlflow_deploy.deactivate", deployment_id=uuid4()
        )

    assert chosen is sp_client
    mock_get_auth.assert_called_once_with()


def test_logs_source_per_resolution(caplog) -> None:  # noqa: ANN001
    """Every resolution must emit a structured log line so on-call can
    attribute a delete to the identity that ran it.
    """
    import logging

    obo_client = MagicMock(name="obo")
    resolver = WorkspaceClientResolver(obo_client=obo_client)

    deployment_id = uuid4()
    with caplog.at_level(logging.INFO, logger="deep_research.services.deployment.auth"):
        resolver.resolve(
            purpose="shell_app.deactivate", deployment_id=deployment_id
        )

    assert any(
        "WORKSPACE_CLIENT_RESOLVED" in r.getMessage()
        and "source=obo" in r.getMessage()
        and str(deployment_id) in r.getMessage()
        for r in caplog.records
    )


def test_logs_sp_fallback(caplog) -> None:  # noqa: ANN001
    import logging

    resolver = WorkspaceClientResolver(obo_client=None)
    deployment_id = uuid4()

    with (
        patch(
            "deep_research.core.databricks_auth.get_databricks_auth"
        ) as mock_get_auth,
        caplog.at_level(
            logging.INFO, logger="deep_research.services.deployment.auth"
        ),
    ):
        mock_get_auth.return_value.get_client.return_value = MagicMock()
        resolver.resolve(
            purpose="shell_app.deactivate", deployment_id=deployment_id
        )

    assert any(
        "WORKSPACE_CLIENT_RESOLVED" in r.getMessage()
        and "source=sp_fallback" in r.getMessage()
        for r in caplog.records
    )
