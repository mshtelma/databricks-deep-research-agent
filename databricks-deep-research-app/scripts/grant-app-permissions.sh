#!/bin/bash
# Grant table permissions to app service principal
# Usage: ./scripts/grant-app-permissions.sh <instance_name> <profile> <database_name> <app_name>
#
# Supports both Provisioned and Autoscaling backends:
# - Provisioned: Pass instance_name and profile as positional args
# - Autoscaling: Set ENDPOINT_NAME env var before calling
#
# After migrations create tables (owned by developer), this script grants
# the app's service principal permission to access those tables.

set -e

INSTANCE_NAME="$1"
PROFILE="$2"
DATABASE_NAME="${3:-deep_research}"
APP_NAME="${4:-deep-research-agent}"

# Auto-detect backend
if [ -n "$ENDPOINT_NAME" ]; then
    echo "Detected Autoscaling backend (ENDPOINT_NAME set)"
    if [ -z "$PROFILE" ]; then
        echo "Usage: ENDPOINT_NAME=... $0 '' <profile> [database_name] [app_name]"
        exit 1
    fi
elif [ -z "$INSTANCE_NAME" ] || [ -z "$PROFILE" ]; then
    echo "Usage: $0 <instance_name> <profile> [database_name] [app_name]"
    echo "  instance_name: Lakebase instance name (e.g., deep-research-lakebase-dev)"
    echo "  profile: Databricks CLI profile (e.g., e2-demo-west)"
    echo "  database_name: Database to grant permissions on (default: deep_research)"
    echo "  app_name: Databricks App name to grant permissions to (default: deep-research-agent)"
    echo ""
    echo "For Autoscaling: set ENDPOINT_NAME env var"
    exit 1
fi

echo "Granting table permissions to app '$APP_NAME'..."
echo "  Profile: $PROFILE"
if [ -n "$ENDPOINT_NAME" ]; then
    echo "  Endpoint: $ENDPOINT_NAME"
else
    echo "  Instance: $INSTANCE_NAME"
fi
echo "  Database: $DATABASE_NAME"

# Build environment
export DATABRICKS_CONFIG_PROFILE="$PROFILE"
export LAKEBASE_DATABASE="$DATABASE_NAME"
if [ -n "$INSTANCE_NAME" ]; then
    export LAKEBASE_INSTANCE_NAME="$INSTANCE_NAME"
fi
# ENDPOINT_NAME is already exported if set

uv run python -c "
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from deep_research.db.grant_permissions import grant_permissions_to_app
asyncio.run(grant_permissions_to_app('$APP_NAME'))
"

echo ""
echo "SUCCESS: Permissions granted to app '$APP_NAME'!"
