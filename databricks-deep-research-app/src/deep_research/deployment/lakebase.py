"""
Lakebase Health and Wait Utilities
==================================

Provides utilities for checking Lakebase health and waiting
for instances to become available. Supports both Provisioned and Autoscaling backends.
"""

import asyncio
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


async def check_lakebase_health(
    instance_name: str | None = None,
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Check if a Lakebase instance is healthy and connectable.

    Uses the WorkspaceClient to generate a database credential,
    which validates that the instance is ready to accept connections.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        workspace_client: Optional WorkspaceClient (creates one if not provided)
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if instance is healthy, False otherwise
    """
    try:
        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        if endpoint_name:
            # Autoscaling health check
            response = workspace_client.postgres.generate_database_credential(
                endpoint=endpoint_name,
            )
        else:
            # Provisioned health check
            response = workspace_client.database.generate_database_credential(
                instance_names=[instance_name] if instance_name else [],
                request_id=str(uuid.uuid4()),
            )

        # Check that we got a valid token back
        return bool(response and response.token)

    except Exception as e:
        logger.debug(
            "Lakebase health check failed: %s",
            str(e),
        )
        return False


async def wait_for_lakebase(
    instance_name: str | None = None,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Wait for a Lakebase instance to become available.

    Polls the instance health until it becomes ready or timeout is reached.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        timeout_seconds: Maximum time to wait (default 5 minutes)
        poll_interval_seconds: Time between health checks (default 5 seconds)
        workspace_client: Optional WorkspaceClient (creates one if not provided)
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if instance became available, False if timeout reached
    """
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    identifier = endpoint_name or instance_name
    logger.info(
        "Waiting for Lakebase '%s' to become available...",
        identifier,
    )

    elapsed = 0
    while elapsed < timeout_seconds:
        is_healthy = await check_lakebase_health(
            instance_name, workspace_client, endpoint_name=endpoint_name,
        )
        if is_healthy:
            logger.info(
                "Lakebase '%s' is ready after %d seconds",
                identifier,
                elapsed,
            )
            return True

        logger.debug(
            "Lakebase not ready yet, waiting %d seconds...",
            poll_interval_seconds,
        )
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    logger.error(
        "Timeout waiting for Lakebase '%s' after %d seconds",
        identifier,
        timeout_seconds,
    )
    return False


def wait_for_lakebase_sync(
    instance_name: str | None = None,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
    workspace_client: Any | None = None,
    *,
    endpoint_name: str | None = None,
) -> bool:
    """Synchronous version of wait_for_lakebase.

    Args:
        instance_name: Provisioned instance name (required if no endpoint_name)
        timeout_seconds: Maximum time to wait
        poll_interval_seconds: Time between health checks
        workspace_client: Optional WorkspaceClient
        endpoint_name: Autoscaling endpoint name

    Returns:
        True if instance became available, False if timeout reached
    """
    return asyncio.run(
        wait_for_lakebase(
            instance_name=instance_name,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for waiting on Lakebase.

    Usage:
        python -m deep_research.deployment.lakebase wait <instance_name> [--timeout 300]
        python -m deep_research.deployment.lakebase wait --endpoint-name <ep> [--timeout 300]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Wait for Lakebase instance to become available"
    )
    parser.add_argument("command", choices=["wait"], help="Command to execute")
    parser.add_argument("instance_name", nargs="?", help="Lakebase instance name (Provisioned)")
    parser.add_argument(
        "--endpoint-name",
        help="Autoscaling endpoint name (alternative to instance_name)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Poll interval in seconds (default: 5)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "wait":
        success = wait_for_lakebase_sync(
            instance_name=args.instance_name,
            timeout_seconds=args.timeout,
            poll_interval_seconds=args.poll_interval,
            endpoint_name=args.endpoint_name,
        )
        if not success:
            exit(1)


if __name__ == "__main__":
    main()
