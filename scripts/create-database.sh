#!/bin/bash
# Create the deep_research database on Lakebase
# Usage: ./scripts/create-database.sh <instance_name> <profile>
#
# This script creates the deep_research database and research schema.
# Must be run AFTER the Lakebase instance is ready (use wait-for-lakebase.sh first).

set -e

INSTANCE_NAME="$1"
PROFILE="$2"
DATABASE_NAME="${3:-deep_research}"

if [ -z "$INSTANCE_NAME" ] || [ -z "$PROFILE" ]; then
    echo "Usage: $0 <instance_name> <profile> [database_name]"
    echo "  instance_name: Lakebase instance name (e.g., deep-research-lakebase-dre-dev)"
    echo "  profile: Databricks CLI profile (e.g., e2-demo-west)"
    echo "  database_name: Database to create (default: deep_research)"
    exit 1
fi

echo "Creating database '$DATABASE_NAME' on Lakebase instance '$INSTANCE_NAME'..."
echo "  Profile: $PROFILE"

DATABRICKS_CONFIG_PROFILE="$PROFILE" \
LAKEBASE_INSTANCE_NAME="$INSTANCE_NAME" \
LAKEBASE_DATABASE="$DATABASE_NAME" \
uv run python -c "
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from deep_research.db.bootstrap import ensure_database_exists
asyncio.run(ensure_database_exists())
"

echo ""
echo "SUCCESS: Database '$DATABASE_NAME' created!"
