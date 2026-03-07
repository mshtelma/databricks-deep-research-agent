"""
Lakebase Connection Utilities
=============================

Provides utilities for getting correct Lakebase connection information.
These functions handle the proper hostname and username resolution for
Lakebase PostgreSQL connections.

The patterns here are based on the reference implementation in
deep_research.db.lakebase_auth.LakebaseCredentialProvider.
"""

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LakebaseConnectionInfo:
    """Connection information for a Lakebase instance."""

    host: str
    port: int
    username: str
    token: str


def get_lakebase_host(
    instance_name: str,
    workspace_client: Any | None = None,
) -> str:
    """Get the actual hostname for a Lakebase instance.

    In Databricks Apps environment, PGHOST is automatically injected by the
    platform and should be used directly. Falls back to API lookup for local
    development when PGHOST is not available.

    Args:
        instance_name: Name of the Lakebase instance (used for API lookup)
        workspace_client: Optional WorkspaceClient (created if not provided)

    Returns:
        The actual DNS hostname for the instance (from read_write_dns).

    Raises:
        ValueError: If hostname cannot be determined.
    """
    # Priority 1: Use PGHOST if set (Databricks Apps auto-injects this)
    pghost = os.environ.get("PGHOST")
    if pghost:
        logger.info(f"Using PGHOST from environment: {pghost}")
        return pghost

    # Priority 2: Fall back to API lookup (for local development)
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    logger.info(f"Looking up hostname for Lakebase instance: {instance_name}")

    try:
        # Use get_database_instance() - matches Databricks Apps Cookbook pattern
        # This only requires permission on the specific instance, not workspace-level list
        inst = workspace_client.database.get_database_instance(name=instance_name)

        if not inst.read_write_dns:
            raise ValueError(
                f"Instance '{instance_name}' has no read_write_dns configured"
            )

        logger.info(f"Found Lakebase host: {inst.read_write_dns}")
        return inst.read_write_dns

    except Exception as e:
        raise ValueError(
            f"Failed to get Lakebase instance '{instance_name}': {e}. "
            f"Check LAKEBASE_INSTANCE_NAME and service principal permissions."
        ) from e


def extract_username_from_token(token: str) -> str:
    """Get username for Lakebase connection from JWT token or environment.

    In Databricks Apps environment, PGUSER is automatically injected by the
    platform and should be used directly. Falls back to extracting from JWT
    token's 'sub' claim for local development.

    Args:
        token: JWT token from Databricks (used for fallback extraction).

    Returns:
        Username (service principal client ID or user email).

    Raises:
        ValueError: If username cannot be determined.
    """
    # Priority 1: Use PGUSER if set (Databricks Apps auto-injects this)
    pguser = os.environ.get("PGUSER")
    if pguser:
        logger.info(f"Using PGUSER from environment: {pguser}")
        return pguser

    # Priority 2: Extract from JWT token's 'sub' claim
    try:
        # JWT format: header.payload.signature
        payload_b64 = token.split(".")[1]
        # Add padding if needed
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        username = payload.get("sub", "")
        if username:
            logger.info(f"Extracted username from token: {username}")
            return username
    except Exception as e:
        logger.warning(f"Failed to extract username from token: {e}")

    raise ValueError(
        "Could not determine username for Lakebase authentication. "
        "Set PGUSER or ensure token contains 'sub' claim."
    )


def discover_autoscaling_endpoint(
    project_id: str,
    workspace_client: Any | None = None,
) -> str:
    """Discover the auto-created endpoint for an Autoscaling project.

    When a project is created, Databricks auto-creates a production branch
    and a primary read_write endpoint. This function discovers that endpoint.

    Args:
        project_id: The Autoscaling project ID (e.g., "deep-research-dev").
        workspace_client: Optional WorkspaceClient (created if not provided).

    Returns:
        Full endpoint resource path (projects/<id>/branches/<id>/endpoints/<id>).

    Raises:
        ValueError: If no endpoint can be found.
    """
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    logger.info(f"Discovering endpoint for project: {project_id}")

    # List branches for this project
    try:
        branches = list(workspace_client.postgres.list_branches(parent=f"projects/{project_id}"))
    except Exception as e:
        raise ValueError(
            f"Failed to list branches for project '{project_id}': {e}"
        ) from e

    if not branches:
        raise ValueError(f"No branches found for project '{project_id}'")

    # Find the production branch (auto-created)
    production_branch = None
    for branch in branches:
        if (branch.status is not None and branch.status.default) or (
            branch.name is not None and branch.name.endswith("/production")
        ):
            production_branch = branch
            break

    # Fall back to first branch if no production branch found
    if production_branch is None:
        production_branch = branches[0]
        logger.warning(
            f"No 'production' branch found, using first branch: {production_branch.name}"
        )

    branch_id = production_branch.name.split("/")[-1] if production_branch.name else production_branch.uid
    logger.info(f"Found branch: {branch_id} (name={production_branch.name})")

    # List endpoints for this branch
    try:
        endpoints = list(
            workspace_client.postgres.list_endpoints(
                parent=f"projects/{project_id}/branches/{branch_id}",
            )
        )
    except Exception as e:
        raise ValueError(
            f"Failed to list endpoints for project '{project_id}', "
            f"branch '{branch_id}': {e}"
        ) from e

    if not endpoints:
        raise ValueError(
            f"No endpoints found for project '{project_id}', branch '{branch_id}'"
        )

    # Use the first read_write endpoint, or fall back to first endpoint
    endpoint = endpoints[0]
    for ep in endpoints:
        if ep.spec is not None and "READ_WRITE" in str(ep.spec.endpoint_type):
            endpoint = ep
            break

    endpoint_name = endpoint.name
    logger.info(f"Discovered endpoint: {endpoint_name}")
    return endpoint_name


def get_lakebase_connection_info(
    instance_name: str | None = None,
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> LakebaseConnectionInfo:
    """Get complete connection info for either Provisioned or Autoscaling backend.

    Convenience function that combines host lookup, credential generation,
    and username extraction.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        workspace_client: Optional WorkspaceClient (created if not provided)
        endpoint_name: Autoscaling endpoint (format: projects/<id>/branches/<id>/endpoints/<id>)

    Returns:
        LakebaseConnectionInfo with host, port, username, and token.

    Raises:
        ValueError: If connection info cannot be determined.
    """
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    # Get port from environment or default
    port = int(os.environ.get("PGPORT", "5432"))

    if endpoint_name:
        # Autoscaling path
        cred = workspace_client.postgres.generate_database_credential(  # type: ignore[attr-defined]
            endpoint=endpoint_name,
        )
        host = os.environ.get("PGHOST")
        if not host:
            raise ValueError("PGHOST is required for Autoscaling backend")
    else:
        # Provisioned path (existing logic)
        if not instance_name:
            raise ValueError(
                "instance_name is required for Provisioned backend "
                "(or provide endpoint_name for Autoscaling)"
            )

        # Get the actual hostname
        host = get_lakebase_host(instance_name, workspace_client)

        # Generate credential (request_id required by Databricks API)
        cred = workspace_client.database.generate_database_credential(
            instance_names=[instance_name],
            request_id=str(uuid.uuid4()),
        )

    if not cred.token:
        raise ValueError("No token returned from Databricks")

    # Get username from env or token
    username = extract_username_from_token(cred.token)

    return LakebaseConnectionInfo(
        host=host,
        port=port,
        username=username,
        token=cred.token,
    )
