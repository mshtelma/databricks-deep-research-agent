"""Helper for resolving the framework Git ref used by generated deployments.

Used by both the deployment-defaults API endpoint
(`api/v1/config.py:get_deployment_defaults`) and the Mode 3 (MLflow agent)
translator (`mlflow_deploy.py`) so the wizard default and the actual
deploy pin agree on a single source of truth.

The default intentionally tracks ``main`` while the framework package is
iterating faster than release tags are published.
"""
from __future__ import annotations

DEFAULT_FRAMEWORK_GIT_REF = "main"


def framework_git_tag(fallback: str = DEFAULT_FRAMEWORK_GIT_REF) -> str:
    """Return the default framework Git ref.

    The function name and response field remain ``framework_git_tag`` for API
    compatibility with existing deployment configs.
    """
    return fallback
