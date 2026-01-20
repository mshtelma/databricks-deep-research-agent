"""
Deployment Module
=================

Provides utilities for deploying deep research applications
to Databricks Apps.

This module enables:
- Lakebase health checks and waiting
- Database creation and management
- Migration execution with multi-path support
- Permission granting to app service principals
- Connection info utilities for Lakebase instances

Example usage:
    from deep_research.deployment import (
        wait_for_lakebase,
        create_database,
        run_migrations,
        grant_to_app,
        get_lakebase_connection_info,
    )

    # Wait for Lakebase to be ready
    await wait_for_lakebase("instance-name")

    # Create database
    await create_database("instance-name", "my_database")

    # Run migrations
    await run_migrations(["src/db/migrations", "my_app/migrations"])

    # Grant permissions
    await grant_to_app("instance-name", "my_database", "my-app-name")

    # Get connection info (correct hostname and username)
    info = get_lakebase_connection_info("instance-name")
    print(f"Host: {info.host}, Username: {info.username}")
"""

from deep_research.deployment.lakebase import (
    check_lakebase_health,
    wait_for_lakebase,
)
from deep_research.deployment.database import (
    create_database,
    database_exists,
    ensure_database_exists,
)
from deep_research.deployment.migrations import (
    run_migrations,
    get_migration_paths,
)
from deep_research.deployment.permissions import (
    grant_to_app,
    get_app_service_principal,
)
from deep_research.deployment.lakebase_connection import (
    get_lakebase_host,
    extract_username_from_token,
    get_lakebase_connection_info,
    LakebaseConnectionInfo,
)

__all__ = [
    # Lakebase
    "wait_for_lakebase",
    "check_lakebase_health",
    # Database
    "create_database",
    "database_exists",
    "ensure_database_exists",
    # Migrations
    "run_migrations",
    "get_migration_paths",
    # Permissions
    "grant_to_app",
    "get_app_service_principal",
    # Connection utilities
    "get_lakebase_host",
    "extract_username_from_token",
    "get_lakebase_connection_info",
    "LakebaseConnectionInfo",
]
