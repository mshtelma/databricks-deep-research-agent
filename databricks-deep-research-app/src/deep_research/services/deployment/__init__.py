"""Deployment translators package.

Each ``DeploymentTranslator`` implementation handles validation, artifact
generation, deployment, and idempotent deactivation for one deployment mode.
Mirrors plan Section M.

Phase 1 ships only ``InAppTranslator``. ShellAppExporter (Mode 2),
MlflowDeployService (Mode 3), and BatchTranslator (Mode 4) land in later
phases.
"""
from deep_research.models.agent_deployment import DeploymentMode
from deep_research.services.deployment.batch import BatchTranslator
from deep_research.services.deployment.in_app import InAppTranslator
from deep_research.services.deployment.mlflow_deploy import MlflowAgentTranslator
from deep_research.services.deployment.resource_resolver import ResourceResolver
from deep_research.services.deployment.shell_app import ShellAppExporter
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentResult,
    DeploymentTranslator,
    ValidationError,
    ValidationResult,
)

_TRANSLATORS: dict[DeploymentMode, DeploymentTranslator] = {
    DeploymentMode.IN_APP: InAppTranslator(),
    DeploymentMode.SHELL_APP: ShellAppExporter(),
    DeploymentMode.MLFLOW_AGENT: MlflowAgentTranslator(),
    DeploymentMode.BATCH: BatchTranslator(),
}


def translator_for(mode: DeploymentMode) -> DeploymentTranslator:
    """Return the singleton translator for ``mode``.

    Centralized so both the API layer and the force-delete cascade in
    ``AgentV2Service`` (W9) dispatch through the same map without
    duplicating the per-mode wiring.
    """
    return _TRANSLATORS[mode]


__all__ = [
    "Artifact",
    "BatchTranslator",
    "DeploymentCleanupError",
    "DeploymentResult",
    "DeploymentTranslator",
    "InAppTranslator",
    "MlflowAgentTranslator",
    "ResourceResolver",
    "ShellAppExporter",
    "ValidationError",
    "ValidationResult",
    "translator_for",
]
