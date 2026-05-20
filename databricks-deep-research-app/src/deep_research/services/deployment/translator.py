"""DeploymentTranslator Protocol + result dataclasses.

The translator protocol bridges plan Principle 1 (AST + revision_id is the
single source of truth) with the reality that each deployment mode needs a
mode-specific artifact (a YAML config for in-app, a zip for shell-app, an
MLflow model artifact for the agent path, an .ipynb for batch).

Each per-mode translator owns its mode's lifecycle (validate -> translate ->
deploy -> deactivate). ``deactivate()`` MUST be idempotent and treat
404/NotFound as success -- this is required by the cleanup lifecycle in
plan Section N.2 (the orphan-detection cron also relies on it).

Optional inline-deploy extension
---------------------------------
``InlineDeploymentTranslator`` is a sub-protocol of ``DeploymentTranslator``
that adds ``deploy_inline()``.  Only translators that support OBO-scoped
synchronous deployment implement it (e.g., ``ShellAppExporter`` after US-403).

Callers that need to detect support should use::

    translator = ...
    if isinstance(translator, InlineDeploymentTranslator):
        result = await translator.deploy_inline(artifact, config, deployment, workspace_client)
    else:
        # fallback or 400 "mode does not support inline deploy"
        ...

The ``getattr(translator, "deploy_inline", None)`` pattern also works for
callers that prefer duck-typing over Protocol isinstance checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode


class DeploymentCleanupError(Exception):
    """Raised by a translator's ``deactivate()`` when external-resource
    teardown actually failed (i.e., the resource still exists upstream).

    A 404 / NotFound from the upstream SDK MUST NOT raise this — that's the
    idempotency contract: deactivate is best-effort and treats "already
    gone" as success. Only genuine upstream failures (permission denied,
    network errors, 5xx, etc.) escalate to this exception.

    The API DELETE handler catches this, increments ``cleanup_attempts``,
    and (after ``MAX_CLEANUP_ATTEMPTS``) transitions the row to
    ``cleanup_failed`` so on-call can see the leaked resource.

    Added in W4 of the fix plan to stop the previous "swallow everything
    as success" pattern that masked real cleanup failures.
    """

    def __init__(
        self,
        message: str,
        *,
        resource: str | None = None,
        upstream_error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.resource = resource
        self.upstream_error_type = upstream_error_type


@dataclass(frozen=True)
class ValidationError:
    """One issue surfaced by ``DeploymentTranslator.validate()``."""

    message: str
    path: str | None = None
    severity: str = "error"  # "error" | "warning"


@dataclass(frozen=True)
class ValidationResult:
    """Result of pre-deploy validation.

    Errors block the deploy; warnings surface to the UI but do not block
    (e.g., Mode 4 ``ai_query`` warns about loop/conditional nodes that run
    server-side via the serving endpoint).
    """

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


@dataclass(frozen=True)
class Artifact:
    """Deployable artifact produced by ``translate()``.

    ``payload`` is bytes for zips/notebooks and a dict for inline JSON specs.
    ``metadata`` is opaque, mode-specific (e.g. SHA256 of the zip, MLflow
    run_id, etc.).
    """

    mode: DeploymentMode
    payload: bytes | dict[str, Any]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentResult:
    """Outcome of ``deploy()`` -- handed back to the service layer.

    On failure, ``error_message`` carries the human-readable reason.
    ``external_resource_ids`` is the **authoritative** record used by the
    orphan-detection cron (plan Section N.3).
    """

    success: bool
    endpoint_name: str | None = None
    model_name: str | None = None
    error_message: str | None = None
    external_resource_ids: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DeploymentTranslator(Protocol):
    """Per-mode deployment translator.

    Implementations live under ``services/deployment/`` and are registered
    with the service factory at app startup. The ``mode`` ClassVar drives
    dispatch from ``CreateDeploymentRequest.config.mode``.
    """

    mode: ClassVar[DeploymentMode]

    async def validate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> ValidationResult:
        """Pre-flight validation. Returns errors for unsupported features.

        ``agent`` is an ``AgentV2``; ``revision`` is an ``AgentRevision`` --
        kept loose-typed here to avoid an import cycle on ``models.agent_v2``.
        """
        ...

    async def translate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> Artifact:
        """Convert agent definition + config into a deployable artifact."""
        ...

    async def deploy(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
    ) -> DeploymentResult:
        """Execute the deployment. Updates external Databricks resources."""
        ...

    async def deactivate(self, deployment: AgentDeployment) -> None:
        """Tear down external resources. MUST be idempotent.

        Implementations MUST treat ``404`` / ``NotFound`` as success: the
        cleanup lifecycle (plan Section N.2) retries up to 3 times before
        transitioning to ``CLEANUP_FAILED``, and orphan detection requires
        re-running ``deactivate()`` on a row that may have already been
        partially cleaned up.
        """
        ...


@runtime_checkable
class InlineDeploymentTranslator(DeploymentTranslator, Protocol):
    """Sub-protocol for translators that support synchronous OBO-scoped deploy.

    Distinct from ``DeploymentTranslator.deploy()`` which runs asynchronously
    via ``DeploymentJobRunner`` with the app's service-principal credentials.
    ``deploy_inline()`` runs synchronously in the request handler using the
    user's OBO-scoped ``WorkspaceClient``.

    Translators that do NOT support inline deploy (``MlflowAgentTranslator``,
    ``InAppTranslator``) do NOT implement this sub-protocol — callers detect
    support via ``isinstance(translator, InlineDeploymentTranslator)`` or
    ``getattr(translator, "deploy_inline", None)``.

    Translators that DO support it (e.g., ``ShellAppExporter`` after US-403)
    declare both ``DeploymentTranslator`` and ``InlineDeploymentTranslator``
    conformance by implementing ``deploy_inline()``.
    """

    async def deploy_inline(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
        workspace_client: Any,  # WorkspaceClient — kept Any to avoid hard SDK dep at this layer
    ) -> DeploymentResult:
        """Inline-synchronous deploy using a request-scoped WorkspaceClient.

        Default raises NotImplementedError. Translators that support OBO-scoped
        inline deploy (ShellAppExporter, BatchTranslator in future) override this.

        Distinct from ``deploy()`` which runs async via DeploymentJobRunner with
        app SP credentials — ``deploy_inline()`` runs synchronously in the request
        handler with the user's OBO token.

        Args:
            artifact: The deployable artifact produced by ``translate()``.
            config: Mode-specific deployment configuration.
            deployment: The ``AgentDeployment`` row being deployed.
            workspace_client: A request-scoped ``WorkspaceClient`` built from the
                user's OBO token.  MUST NOT be cached across requests.

        Returns:
            ``DeploymentResult`` indicating success or failure.

        Raises:
            NotImplementedError: If the translator does not override this method.
        """
        ...
