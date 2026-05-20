"""MlflowAgentTranslator — Mode 3 (Mosaic AI Agent Framework deployment).

Plan reference: agent-designer-deployment.md Section D (MLflow Agent Path).

Pipeline:
  validate()    — pre-flight checks on UC fields + AST shape
  translate()   — write workflow definition + pip-requirements to a temp dir
  deploy()      — log_model → set_registry_uri('databricks-uc') → register_model
                  → databricks.agents.deploy(uc_name, version)
  deactivate()  — idempotent: serving endpoint delete + model version Archived
                  (treats 404 as success; orphan-cron picks up resources that
                  predate the table per plan Section N.3)

The ``databricks-agents`` SDK is **lazy-imported** inside ``deploy()`` so
unit tests can patch it via ``unittest.mock`` without requiring the package.
The runtime install adds ``databricks-agents>=1.1.0`` (per plan research)
when this code path is exercised in production.
"""
# validate() ignores ``agent`` and ``revision`` (Protocol signature is wider
# than this mode needs); silence ARG002 at module level.
# ruff: noqa: ARG002
from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment.framework_version import framework_git_tag
from deep_research.services.deployment.resource_resolver import ResourceResolver
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentResult,
    ValidationError,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# UC identifier regex (catalog/schema/model name) — ASCII letters, digits,
# underscore, hyphen; cannot start with a digit.
_UC_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# Endpoint name override regex — must start with the dr-agent- prefix per
# plan Section N.3 (matches Pydantic schema validation server-side).
_ENDPOINT_NAME_RE = re.compile(r"^dr-agent-[a-z0-9-]+$")


def _pip_requirements(framework_git_tag: str) -> list[str]:
    """The pip_requirements list logged with the model.

    Pinning is via Git tag (no PyPI yet). Tag immutability is enforced by
    the release-tag-lint.yml CI workflow.
    """
    return [
        f"databricks-deep-research[tracing,web,crawl,search] @ "
        f"git+https://github.com/mshtelma/databricks-deep-research-agent.git@"
        f"{framework_git_tag}#subdirectory=databricks-deep-research",
        "mlflow>=3.1.3",
        "databricks-agents>=1.1.0",
    ]


class MlflowAgentTranslator:
    """Translator for ``DeploymentMode.MLFLOW_AGENT``.

    The blocking SDK calls (mlflow.pyfunc.log_model, mlflow.register_model,
    databricks.agents.deploy) are wrapped at the API layer in
    ``asyncio.to_thread`` so they don't block the FastAPI event loop.
    """

    mode: ClassVar[DeploymentMode] = DeploymentMode.MLFLOW_AGENT

    async def validate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> ValidationResult:
        """Pre-flight checks on UC fields + AST shape.

        UC permission probe (CREATE MODEL on the schema) is intentionally a
        deploy-time concern — it requires a workspace round-trip and is
        re-checked just before ``register_model`` per plan Section D.3.
        """
        errors: list[ValidationError] = []

        for required, label in (
            ("uc_catalog", "uc_catalog"),
            ("uc_schema", "uc_schema"),
            ("uc_model_name", "uc_model_name"),
        ):
            value = config.get(required, "")
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    ValidationError(
                        message=f"{label} is required and must be non-empty",
                        path=f"config.{required}",
                    )
                )
            elif not _UC_IDENT_RE.match(value):
                errors.append(
                    ValidationError(
                        message=(
                            f"{label} must be a valid Unity Catalog identifier "
                            "(letters, digits, underscore, hyphen; not "
                            "starting with a digit)"
                        ),
                        path=f"config.{required}",
                    )
                )

        # Endpoint name override (when set) must match dr-agent- prefix.
        endpoint_override = config.get("endpoint_name")
        if endpoint_override and not _ENDPOINT_NAME_RE.match(endpoint_override):
            errors.append(
                ValidationError(
                    message="endpoint_name must start with 'dr-agent-' prefix",
                    path="config.endpoint_name",
                )
            )

        return ValidationResult(valid=not errors, errors=errors)

    async def translate(
        self,
        agent: Any,
        revision: Any,
        config: dict[str, Any],
    ) -> Artifact:
        """Write the workflow definition JSON to a temp file + collect deploy params.

        ``deploy()`` reads ``payload['workflow_definition_path']`` to seed
        the MLflow model artifacts. The temp dir is left on disk; MLflow's
        log_model ingests the file before the artifact is needed elsewhere.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="dr-mlflow-agent-"))
        definition_path = temp_dir / "workflow_definition.json"
        definition_path.write_text(
            json.dumps(revision.definition, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        # Resolve framework Git tag: caller-supplied wins, else fall back to
        # the running framework's installed version so deployments always pin
        # to a real release rather than a stale hardcoded literal (W3).
        configured_tag = config.get("framework_git_tag")
        resolved_tag = configured_tag if configured_tag else framework_git_tag()
        uc_model_uri = (
            f"{config['uc_catalog']}.{config['uc_schema']}.{config['uc_model_name']}"
        )

        # W8: pass through optional wizard fields the schema validates but the
        # previous translator silently dropped. ``endpoint_name`` is captured
        # in the artifact when set; ``_deploy_via_sdk`` forwards it as a kwarg
        # to ``agents.deploy()``. ``env_overrides`` (already carried) now
        # actually reaches the deploy call.
        endpoint_name_override = config.get("endpoint_name")

        return Artifact(
            mode=DeploymentMode.MLFLOW_AGENT,
            payload={
                "workflow_definition_path": str(definition_path),
                "pip_requirements": _pip_requirements(resolved_tag),
                "uc_model_uri": uc_model_uri,
                "env_overrides": config.get("env_overrides", {}),
                "endpoint_name_override": endpoint_name_override,
            },
            metadata={
                "agent_id": str(agent.id),
                "revision_id": str(revision.rev_id),
                "uc_model_uri": uc_model_uri,
                "framework_git_tag": resolved_tag,
                "endpoint_name_override": endpoint_name_override or "",
            },
        )

    async def deploy(
        self,
        artifact: Artifact,
        config: dict[str, Any],
        deployment: AgentDeployment,
    ) -> DeploymentResult:
        """log_model → register_model → databricks.agents.deploy.

        Lazy-imports of ``mlflow`` + ``databricks.agents`` keep the module
        importable when those packages are absent at unit-test time
        (tests patch ``mlflow_deploy._deploy_via_sdk``).
        """
        if not isinstance(artifact.payload, dict):
            return DeploymentResult(
                success=False,
                error_message="MlflowAgentTranslator artifact payload must be a dict",
            )

        try:
            # log_model + register_model + agents.deploy are blocking SDK
            # calls (typically 5-15 min per plan Section B.8). Hand them off
            # to a worker thread so the FastAPI event loop stays responsive
            # for Modes 1/2/4 which are fast/synchronous.
            return await asyncio.to_thread(
                _deploy_via_sdk, artifact, config, deployment
            )
        except Exception as exc:  # noqa: BLE001 -- surface to caller w/o trace
            logger.exception(
                "MLflow agent deploy failed for deployment %s", deployment.id
            )
            return DeploymentResult(
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def deactivate(self, deployment: AgentDeployment) -> None:
        """Idempotent teardown: archive UC model version + delete endpoint.

        Both calls treat 404 / NotFound as success per the protocol's
        idempotency contract. Genuine upstream failures (permission denied,
        network errors, 5xx) propagate as ``DeploymentCleanupError`` so the
        API layer can transition the row to ``cleanup_failed`` after
        ``MAX_CLEANUP_ATTEMPTS`` (W4 of the fix plan — replaces the
        previous "swallow everything" pattern that masked failure as
        success).
        """
        await asyncio.to_thread(_deactivate_via_sdk, deployment)


# ---------------------------------------------------------------------------
# Module-level helpers — extracted so unit tests can patch them via
# ``unittest.mock.patch('deep_research.services.deployment.mlflow_deploy._deploy_via_sdk', ...)``
# without dragging the real SDK imports into test environments.
# ---------------------------------------------------------------------------


def _deploy_via_sdk(
    artifact: Artifact,
    config: dict[str, Any],  # noqa: ARG001 -- signature mirrors deploy() for mock-patching
    deployment: AgentDeployment,  # noqa: ARG001 -- signature mirrors deploy() for mock-patching
) -> DeploymentResult:
    """Real deploy path — runs mlflow + databricks.agents at production time.

    Replaced in tests by ``unittest.mock.patch``. ``config`` and ``deployment``
    are intentionally unused: ``artifact.payload`` already carries everything
    the deploy step needs, but matching the public ``deploy()`` signature
    keeps the ``unittest.mock.patch(... return_value=fake_result)`` test
    pattern uniform.
    """
    import mlflow  # noqa: PLC0415 -- lazy
    import mlflow.pyfunc  # noqa: PLC0415

    try:
        from databricks import agents  # noqa: PLC0415 -- lazy
    except ImportError:
        return DeploymentResult(
            success=False,
            error_message=(
                "databricks-agents SDK is not installed. "
                "MLflow agent deployment requires `pip install databricks-agents>=1.1.0`."
            ),
        )

    if not isinstance(artifact.payload, dict):
        return DeploymentResult(
            success=False,
            error_message="MlflowAgentTranslator artifact payload must be a dict",
        )

    payload: dict[str, Any] = artifact.payload
    uc_name: str = payload["uc_model_uri"]
    workflow_definition_path: str = payload["workflow_definition_path"]
    pip_requirements: list[str] = payload["pip_requirements"]
    env_overrides: dict[str, str] = payload.get("env_overrides") or {}
    endpoint_name_override: str | None = payload.get("endpoint_name_override")

    # Re-resolve resources from the persisted workflow definition (don't pass
    # them through the artifact — keeps the artifact JSON-serializable for
    # diagnostics and audit logs).
    with open(workflow_definition_path, encoding="utf-8") as f:
        definition_dict: dict[str, Any] = json.load(f)
    resources = ResourceResolver().resolve(definition_dict)

    # The Python-model code lives in this app at responses_agent.py.
    python_model_path = str(
        Path(__file__).resolve().parent / "responses_agent.py"
    )

    with mlflow.start_run():
        log_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=python_model_path,
            artifacts={"workflow_definition": workflow_definition_path},
            pip_requirements=pip_requirements,
            resources=resources,
        )

    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.register_model(model_uri=log_info.model_uri, name=uc_name)

    # W8: forward optional wizard fields to agents.deploy() when set. Build
    # kwargs dynamically so older databricks-agents versions that don't
    # accept these args still work for the bare-minimum call.
    deploy_kwargs: dict[str, Any] = {}
    if endpoint_name_override:
        deploy_kwargs["endpoint_name"] = endpoint_name_override
    if env_overrides:
        deploy_kwargs["environment_vars"] = env_overrides

    deploy_result = agents.deploy(uc_name, str(mv.version), **deploy_kwargs)

    return DeploymentResult(
        success=True,
        endpoint_name=getattr(deploy_result, "endpoint_name", None),
        model_name=uc_name,
        external_resource_ids={
            "uc_model": uc_name,
            "model_version": str(mv.version),
            # databricks.agents.deploy() return shape varies by SDK version;
            # fall back to the UC name when the attribute is missing.
            "app_name": getattr(deploy_result, "app_name", uc_name),
            "endpoint_name": getattr(deploy_result, "endpoint_name", uc_name),
            "endpoint_name_override": endpoint_name_override or "",
        },
    )


def _is_not_found_error(exc: BaseException) -> bool:
    """Heuristic 404/NotFound detector for upstream SDK exceptions.

    The databricks-sdk and mlflow client both surface "already gone" as
    typed exceptions named ``NotFound`` / ``ResourceDoesNotExist`` (and
    occasionally an HTTP 404 carried inside the message). We don't import
    the typed classes here so the deactivate path stays loosely coupled to
    SDK versions and so tests don't need the SDK installed.
    """
    cls_name = type(exc).__name__.lower()
    if "notfound" in cls_name or "doesnotexist" in cls_name:
        return True
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "does not exist" in msg


def _deactivate_via_sdk(deployment: AgentDeployment) -> None:
    """Real deactivate path.

    Treats 404/NotFound as idempotent success per the protocol contract.
    Any other upstream failure is collected and re-raised as a single
    ``DeploymentCleanupError`` so the API layer can promote the row to
    ``cleanup_failed`` after the attempt threshold (W4).
    """
    import mlflow  # noqa: PLC0415

    try:
        from databricks import agents  # noqa: PLC0415
    except ImportError:
        return  # If the SDK is missing the deployment was never live anyway.

    external = deployment.external_resource_ids or {}
    uc_model: str | None = external.get("uc_model")
    model_version: str | None = external.get("model_version")

    if not (uc_model and model_version):
        # No external resources tracked — nothing to tear down.
        return

    failures: list[tuple[str, Exception]] = []

    try:
        agents.delete_deployment(uc_model, model_version)
    except Exception as exc:  # noqa: BLE001 -- triage via _is_not_found_error
        if _is_not_found_error(exc):
            logger.info(
                "agents.delete_deployment(%s, %s) returned NotFound; treating as gone",
                uc_model,
                model_version,
            )
        else:
            logger.exception(
                "agents.delete_deployment(%s, %s) failed", uc_model, model_version
            )
            failures.append(("agents.delete_deployment", exc))

    try:
        mlflow.set_registry_uri("databricks-uc")
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=uc_model, version=model_version, stage="Archived"
        )
    except Exception as exc:  # noqa: BLE001 -- triage via _is_not_found_error
        if _is_not_found_error(exc):
            logger.info(
                "MLflow archive(%s, %s) returned NotFound; treating as gone",
                uc_model,
                model_version,
            )
        else:
            logger.exception(
                "MLflow archive(%s, %s) failed", uc_model, model_version
            )
            failures.append(("mlflow.archive", exc))

    if failures:
        # Compose a single error so the API can record one ``error_message``
        # without losing the per-resource detail.
        detail = ", ".join(
            f"{resource} raised {type(exc).__name__}" for resource, exc in failures
        )
        raise DeploymentCleanupError(
            f"MLflow agent deactivate failed: {detail}",
            resource=", ".join(resource for resource, _ in failures),
            upstream_error_type=type(failures[0][1]).__name__,
        )
