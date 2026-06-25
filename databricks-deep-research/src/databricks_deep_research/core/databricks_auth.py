"""Reusable Databricks ``WorkspaceClient`` construction for OBO and SP auth.

Every host that runs framework Databricks tools (the main app, the per-agent
shell-app) needs to decide whether a tool talks to Databricks as the
**service principal** (SP) or **on-behalf-of** (OBO) the calling user. This
module owns that single decision so hosts do not each re-implement it.

OBO clients force ``auth_type="pat"``: the user's OAuth access token is used
as a static bearer token, which stops the SDK from auto-detecting the app
SP's OAuth-M2M environment credentials (``DATABRICKS_CLIENT_ID`` /
``DATABRICKS_CLIENT_SECRET``) and silently losing the OBO identity.

Framework-only: imports ``databricks.sdk`` (a hard dependency) but never the
host application packages.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


def build_obo_workspace_client(*, host: str, user_token: str) -> Any:
    """Build a ``WorkspaceClient`` that authenticates AS THE USER.

    ``auth_type="pat"`` is mandatory — it forces token auth so the SDK cannot
    fall back to the service-principal OAuth env vars and silently drop the
    on-behalf-of identity.
    """
    return WorkspaceClient(host=host, token=user_token, auth_type="pat")


def resolve_workspace_client(
    *, sp_client: Any | None, user_token: str | None
) -> Any | None:
    """Pick the right ``WorkspaceClient`` for one request.

    Returns an OBO client (built from ``user_token``; host derived from
    ``sp_client.config.host``) when a token is present; otherwise returns
    ``sp_client`` unchanged (service-principal / default auth).

    Policy-light by design: when no token is present this falls back to the
    service principal rather than failing. Hosts that must *require* OBO for
    UC-gated workflows enforce that themselves (fail-closed) before calling
    the runner — see ``workflow_requires_databricks`` in
    :mod:`databricks_deep_research.tools.builtins.databricks_runner`.
    """
    if user_token and sp_client is not None:
        host = sp_client.config.host
        if host:
            return build_obo_workspace_client(host=host, user_token=user_token)
        logger.warning(
            "OBO_CLIENT_HOST_UNRESOLVED could not derive host from "
            "sp_client.config; falling back to the provided client"
        )
    return sp_client
