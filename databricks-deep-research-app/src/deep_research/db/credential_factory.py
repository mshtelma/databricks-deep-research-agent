"""Factory for creating Lakebase credential providers.

Auto-detects the backend type (Provisioned vs Autoscaling) based on
environment variables and settings.
"""

import logging
import os
from typing import TYPE_CHECKING

from deep_research.db.credential_provider import (
    BaseLakebaseCredentialProvider,
    LakebaseBackend,
)

if TYPE_CHECKING:
    from deep_research.core.config import Settings

logger = logging.getLogger(__name__)


# Sentinel values written by the bundle template before a full deploy
# resolves real connection details (see databricks.yml + Makefile step 5).
# Treating these as "unset" prevents the app from booting into the wrong
# credential path with a placeholder endpoint/host name, which previously
# manifested as "Database instance 'deep-research-lakebase' not found" or
# "permission denied for database deep_research" at startup.
_PLACEHOLDER_ENV: frozenset[str] = frozenset({"pending", "tbd", "todo"})


def _meaningful(value: str | None, *, source: str | None = None) -> str | None:
    """Return ``value`` if non-empty and not a known placeholder, else ``None``.

    ``source`` is logged when a non-empty placeholder is filtered out so the
    operator can pinpoint a bad deploy from the app's first log line.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in _PLACEHOLDER_ENV:
        if source:
            logger.warning(
                "PLACEHOLDER_ENV_FILTERED name=%s raw=%r — treating as unset. "
                "If this is a deployed app, re-run `make deploy TARGET=<env>` "
                "so bundle variables resolve to real values.",
                source, stripped,
            )
        return None
    return stripped


def detect_lakebase_backend(settings: "Settings") -> LakebaseBackend | None:
    """Auto-detect which Lakebase backend to use.

    Detection priority:
    1. ENDPOINT_NAME env var or settings → Autoscaling
    2. LAKEBASE_INSTANCE_NAME or PGHOST → Provisioned
    3. Neither → None (not using Lakebase)

    Placeholder values written by the bundle template (e.g. ``pending``,
    ``tbd``, empty/whitespace) are treated as unset on every input — they
    never flow into the chosen backend.

    Edge case: If BOTH are set, ENDPOINT_NAME wins with a warning log.
    This can happen during migration from Provisioned to Autoscaling.
    """
    endpoint = (
        _meaningful(settings.endpoint_name, source="settings.endpoint_name")
        or _meaningful(os.environ.get("ENDPOINT_NAME"), source="ENDPOINT_NAME")
    )
    instance = (
        _meaningful(settings.lakebase_instance_name, source="settings.lakebase_instance_name")
        or _meaningful(os.environ.get("PGHOST"), source="PGHOST")
    )

    if endpoint and instance:
        logger.warning(
            "LAKEBASE_BACKEND_CONFLICT both ENDPOINT_NAME and LAKEBASE_INSTANCE_NAME/PGHOST set. "
            "Using Autoscaling (ENDPOINT_NAME takes priority)."
        )
    if endpoint:
        return "autoscaling"
    if instance:
        return "provisioned"
    return None


def create_credential_provider(
    settings: "Settings",
) -> BaseLakebaseCredentialProvider | None:
    """Create the appropriate credential provider based on auto-detection.

    Args:
        settings: Application settings.

    Returns:
        Credential provider for the detected backend, or None if no
        Lakebase backend is configured.
    """
    backend = detect_lakebase_backend(settings)

    if backend == "autoscaling":
        from deep_research.db.autoscaling_auth import AutoscalingCredentialProvider

        logger.info("LAKEBASE_BACKEND_SELECTED backend=autoscaling")
        return AutoscalingCredentialProvider(settings)
    elif backend == "provisioned":
        from deep_research.db.lakebase_auth import LakebaseCredentialProvider

        logger.info("LAKEBASE_BACKEND_SELECTED backend=provisioned")
        return LakebaseCredentialProvider(settings)

    logger.info("LAKEBASE_BACKEND_SELECTED backend=none")
    return None
