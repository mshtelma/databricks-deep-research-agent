#!/bin/bash
# Grant table permissions to app service principal
# Usage: ./scripts/grant-app-permissions.sh <instance_name> <profile> <database_name> <app_name>
#
# After migrations create tables (owned by developer), this script grants
# the app's service principal permission to access those tables.
#
# This is necessary because:
# - Developer runs migrations → tables are owned by developer
# - App has CAN_CONNECT_AND_CREATE on database (not on tables)
# - Explicit GRANT statements are needed for app to SELECT/INSERT/UPDATE/DELETE

set -e

INSTANCE_NAME="$1"
PROFILE="$2"
DATABASE_NAME="${3:-deep_research}"
APP_NAME="${4:-deep-research-agent}"

if [ -z "$INSTANCE_NAME" ] || [ -z "$PROFILE" ]; then
    echo "Usage: $0 <instance_name> <profile> [database_name] [app_name]"
    echo "  instance_name: Lakebase instance name (e.g., deep-research-lakebase-dre-dev)"
    echo "  profile: Databricks CLI profile (e.g., e2-demo-west)"
    echo "  database_name: Database to grant permissions on (default: deep_research)"
    echo "  app_name: Databricks App name to grant permissions to (default: deep-research-agent)"
    exit 1
fi

echo "Granting table permissions to app '$APP_NAME'..."
echo "  Instance: $INSTANCE_NAME"
echo "  Profile: $PROFILE"
echo "  Database: $DATABASE_NAME"

DATABRICKS_CONFIG_PROFILE="$PROFILE" \
LAKEBASE_INSTANCE_NAME="$INSTANCE_NAME" \
LAKEBASE_DATABASE="$DATABASE_NAME" \
uv run python -c "
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from src.db.grant_permissions import grant_permissions_to_app
asyncio.run(grant_permissions_to_app('$APP_NAME'))
"

echo ""
echo "SUCCESS: Permissions granted to app '$APP_NAME'!"
