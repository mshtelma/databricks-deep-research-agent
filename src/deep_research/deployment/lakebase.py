"""
Lakebase Health and Wait Utilities
==================================

Provides utilities for checking Lakebase health and waiting
for instances to become available.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def check_lakebase_health(
    instance_name: str,
    workspace_client: Any | None = None,
) -> bool:
    """Check if a Lakebase instance is healthy and connectable.

    Uses the WorkspaceClient to generate a database credential,
    which validates that the instance is ready to accept connections.

    Args:
        instance_name: Name of the Lakebase instance
        workspace_client: Optional WorkspaceClient (creates one if not provided)

    Returns:
        True if instance is healthy, False otherwise
    """
    try:
        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        # Try to generate a credential - this validates the instance is ready
        response = workspace_client.database.generate_database_credential(
            instance_names=[instance_name]
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
    instance_name: str,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
    workspace_client: Any | None = None,
) -> bool:
    """Wait for a Lakebase instance to become available.

    Polls the instance health until it becomes ready or timeout is reached.

    Args:
        instance_name: Name of the Lakebase instance
        timeout_seconds: Maximum time to wait (default 5 minutes)
        poll_interval_seconds: Time between health checks (default 5 seconds)
        workspace_client: Optional WorkspaceClient (creates one if not provided)

    Returns:
        True if instance became available, False if timeout reached
    """
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    logger.info(
        "Waiting for Lakebase instance '%s' to become available...",
        instance_name,
    )

    elapsed = 0
    while elapsed < timeout_seconds:
        is_healthy = await check_lakebase_health(instance_name, workspace_client)
        if is_healthy:
            logger.info(
                "Lakebase instance '%s' is ready after %d seconds",
                instance_name,
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
        "Timeout waiting for Lakebase instance '%s' after %d seconds",
        instance_name,
        timeout_seconds,
    )
    return False


def wait_for_lakebase_sync(
    instance_name: str,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
    workspace_client: Any | None = None,
) -> bool:
    """Synchronous version of wait_for_lakebase.

    Useful for CLI commands and scripts that don't use asyncio.

    Args:
        instance_name: Name of the Lakebase instance
        timeout_seconds: Maximum time to wait
        poll_interval_seconds: Time between health checks
        workspace_client: Optional WorkspaceClient

    Returns:
        True if instance became available, False if timeout reached
    """
    return asyncio.run(
        wait_for_lakebase(
            instance_name=instance_name,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            workspace_client=workspace_client,
        )
    )


# CLI entry point
def main() -> None:
    """CLI entry point for waiting on Lakebase.

    Usage:
        python -m deep_research.deployment.lakebase wait <instance_name> [--timeout 300]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Wait for Lakebase instance to become available"
    )
    parser.add_argument("command", choices=["wait"], help="Command to execute")
    parser.add_argument("instance_name", help="Lakebase instance name")
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
        )
        if not success:
            exit(1)


if __name__ == "__main__":
    main()
