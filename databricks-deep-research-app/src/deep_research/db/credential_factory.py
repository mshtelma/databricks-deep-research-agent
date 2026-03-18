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


def detect_lakebase_backend(settings: "Settings") -> LakebaseBackend | None:
    """Auto-detect which Lakebase backend to use.

    Detection priority:
    1. ENDPOINT_NAME env var or settings → Autoscaling
    2. LAKEBASE_INSTANCE_NAME or PGHOST → Provisioned
    3. Neither → None (not using Lakebase)

    Edge case: If BOTH are set, ENDPOINT_NAME wins with a warning log.
    This can happen during migration from Provisioned to Autoscaling.
    """
    endpoint = settings.endpoint_name or os.environ.get("ENDPOINT_NAME")
    instance = settings.lakebase_instance_name or os.environ.get("PGHOST")

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
