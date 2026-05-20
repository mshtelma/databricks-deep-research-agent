"""InAppTranslator: in-app picker (Mode 1).

In-app deployment has no external Databricks resources -- the deployment row
is the deployment. The chat composer's agent picker reads from the same
``agent_deployments`` table to decide which agents to surface.

Per plan Section M's per-mode feature handling table, every AST feature
(sequence, parallel, loop, conditional, plan_and_execute, custom tool,
vector_search, genie) passes through unchanged because Mode 1 runs inside
the parent app and inherits its tool registry.

Implements the ``DeploymentTranslator`` protocol (validated at import time
by the runtime_checkable conformance check in ``services/deployment/__init__.py``).
"""
# Method args (agent, revision, config, deployment) are required by the
# DeploymentTranslator Protocol but unused for the in-app no-op path.
# Mode 1 has no external resources to validate or deactivate.
# ruff: noqa: ARG002
from __future__ import annotations

from typing import Any, ClassVar

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentResult,
    ValidationResult,
)


class InAppTranslator:
    """Translator for ``DeploymentMode.IN_APP``.

    Mode 1 has no external resources. ``validate`` only checks AST shape
    (handled at the API layer via ``AgentV2Service``); ``translate`` produces
    an inline reference; ``deploy`` and ``deactivate`` are effectively no-ops
    because the lifecycle lives entirely in our database row.
    """

    mode: ClassVar[DeploymentMode] = DeploymentMode.IN_APP

    async def validate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> ValidationResult:
        """In-app deployments are always valid: visibility is enforced by
        ``AgentV2Service`` on read; the agent picker filters via ``/can-run``.
        """
        return ValidationResult(valid=True)

    async def translate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> Artifact:
        """Produce a tiny in-memory artifact pointing at the revision.

        No external bytes are emitted; the framework loads the workflow
        definition directly from ``AgentRevision.definition`` JSONB at run
        time.
        """
        return Artifact(
            mode=DeploymentMode.IN_APP,
            payload={"revision_id": str(revision.rev_id)},
            metadata={"agent_id": str(agent.id)},
        )

    async def deploy(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
    ) -> DeploymentResult:
        """No-op deploy. The row's existence + ACTIVE status IS the deploy.

        ``DeploymentService.update_status(ACTIVE)`` is called by the API layer
        immediately after ``deploy()`` returns success.
        """
        return DeploymentResult(success=True)

    async def deactivate(self, deployment: AgentDeployment) -> None:
        """No-op deactivate. There are no external resources to release.

        Idempotent by definition.
        """
        return None
