"""WorkspaceClientResolver — single source of truth for which Databricks
identity runs a deployment-lifecycle SDK call.

Background. Shell-App / MLflow-Agent deployments are created on behalf of
the requesting user (OBO-scoped ``WorkspaceClient`` passed into
``deploy_inline``). The parent-app service principal does not own those
resources, so a delete using the SP identity returns 403 from
``apps.delete`` / ``agents.delete_deployment``. The translator then raises
``DeploymentCleanupError``, the user sees a 409, and only the 4th retry
(once the row crosses ``MAX_CLEANUP_ATTEMPTS`` into ``CLEANUP_FAILED``)
succeeds.

The resolver lets user-initiated paths thread the request's OBO client
into ``translator.deactivate`` while keeping the janitor's call site
unchanged — it has no user context and must continue to use the SP.

Precedence
----------
1. ``obo_client`` supplied to the constructor — preferred when present.
2. Parent-app SP via ``get_databricks_auth().get_client()`` — fallback.

Every resolution emits a structured log entry (``WORKSPACE_CLIENT_RESOLVED``)
with the source so on-call can attribute a delete to the identity that ran
it.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


class WorkspaceClientResolver:
    """Returns the most-appropriate ``WorkspaceClient`` for a lifecycle call.

    Construct one per request: pass the user's OBO client when available,
    ``None`` otherwise (e.g., janitor / orphan-detection cron).
    """

    def __init__(self, obo_client: WorkspaceClient | None) -> None:
        self._obo_client = obo_client

    def resolve(self, *, purpose: str, deployment_id: UUID) -> WorkspaceClient:
        """Return the client to use for ``purpose`` on ``deployment_id``.

        Emits a structured log line so the chosen identity is traceable.
        """
        if self._obo_client is not None:
            logger.info(
                "WORKSPACE_CLIENT_RESOLVED source=obo purpose=%s deployment_id=%s",
                purpose,
                deployment_id,
            )
            return self._obo_client

        # Lazy import to keep this module decoupled from the Databricks
        # auth wiring at import time (avoids cycles with translators that
        # import this protocol).
        from deep_research.core.databricks_auth import (  # noqa: PLC0415
            get_databricks_auth,
        )

        logger.info(
            "WORKSPACE_CLIENT_RESOLVED source=sp_fallback purpose=%s deployment_id=%s",
            purpose,
            deployment_id,
        )
        return get_databricks_auth().get_client()
