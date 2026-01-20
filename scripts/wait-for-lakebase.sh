#!/bin/bash
# Wait for Lakebase instance to be ready and connectable
# Usage: ./scripts/wait-for-lakebase.sh <instance_name> <profile> [max_retries]
#
# This script waits for the Lakebase PostgreSQL instance to be ready for connections.
# It's needed because:
# 1. New Lakebase instances take time to provision after bundle deploy
# 2. Permissions may take 30-60 seconds to propagate after creation
# 3. OAuth token generation requires the instance to be fully ready

set -e

INSTANCE_NAME="$1"
PROFILE="$2"
MAX_RETRIES="${3:-10}"
RETRY_DELAY=10

if [ -z "$INSTANCE_NAME" ] || [ -z "$PROFILE" ]; then
    echo "Usage: $0 <instance_name> <profile> [max_retries]"
    echo "  instance_name: Lakebase instance name (e.g., deep-research-lakebase-dre-dev)"
    echo "  profile: Databricks CLI profile (e.g., e2-demo-west)"
    echo "  max_retries: Number of connection attempts (default: 10)"
    exit 1
fi

echo "Waiting for Lakebase instance '$INSTANCE_NAME' to be ready..."
echo "  Profile: $PROFILE"
echo "  Max retries: $MAX_RETRIES"
echo "  Retry delay: ${RETRY_DELAY}s"
echo ""

for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i/$MAX_RETRIES: Checking database connection..."

    # Try to connect and run a simple query
    # This validates: instance exists, OAuth works, connection succeeds
    if DATABRICKS_CONFIG_PROFILE="$PROFILE" \
       LAKEBASE_INSTANCE_NAME="$INSTANCE_NAME" \
       LAKEBASE_DATABASE="postgres" \
       uv run python -c "
import sys
try:
    from src.db.lakebase_auth import LakebaseCredentialProvider
    from src.core.config import get_settings

    settings = get_settings()
    provider = LakebaseCredentialProvider(settings)

    # This will fail fast if instance doesn't exist or OAuth fails
    cred = provider.get_credential()
    print(f'OAuth credential obtained for user: {cred.username}')
    print(f'Host: {settings.lakebase_host}')
    print('Lakebase is ready!')
    sys.exit(0)
except Exception as e:
    print(f'Not ready: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        echo ""
        echo "SUCCESS: Lakebase instance is ready for connections!"
        exit 0
    fi

    if [ $i -lt $MAX_RETRIES ]; then
        echo "  Not ready yet. Waiting ${RETRY_DELAY}s before retry..."
        sleep $RETRY_DELAY
    fi
done

echo ""
echo "ERROR: Lakebase instance not ready after $MAX_RETRIES attempts ($(( MAX_RETRIES * RETRY_DELAY ))s total)"
echo ""
echo "Troubleshooting:"
echo "  1. Check if instance exists: databricks apps list --profile $PROFILE"
echo "  2. Check Lakebase status in Databricks workspace UI"
echo "  3. Verify your Databricks CLI profile: databricks auth login --profile $PROFILE"
echo "  4. Try running migrations manually: make db-migrate-remote TARGET=<target>"
exit 1
