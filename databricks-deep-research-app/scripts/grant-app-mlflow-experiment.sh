#!/bin/bash
# Grant CAN_EDIT on an MLflow experiment to the app's service principal.
#
# Databricks Apps does NOT support mlflow_experiment as a bundle resource,
# so the app SP gets no ACL on the experiment by default. Without it,
# mlflow.set_experiment() fails inside the app and no traces are recorded.
#
# Usage: ./scripts/grant-app-mlflow-experiment.sh <experiment_path> <app_name> <profile> [permission]
#   experiment_path : MLflow experiment path (e.g., /Shared/deep-research-agent-experiments)
#   app_name        : Databricks App name (e.g., deep-research-agent-ais)
#   profile         : Databricks CLI profile (e.g., ais)
#   permission      : CAN_READ | CAN_EDIT | CAN_MANAGE (default: CAN_EDIT)
#
# Idempotent: re-running with the same arguments is a no-op grant refresh.

set -e

EXPERIMENT_PATH="$1"
APP_NAME="$2"
PROFILE="$3"
PERMISSION="${4:-CAN_EDIT}"

if [ -z "$EXPERIMENT_PATH" ] || [ -z "$APP_NAME" ] || [ -z "$PROFILE" ]; then
    echo "Usage: $0 <experiment_path> <app_name> <profile> [permission]"
    exit 1
fi

echo "Granting MLflow experiment permission to app '$APP_NAME'..."
echo "  Experiment: $EXPERIMENT_PATH"
echo "  Profile:    $PROFILE"
echo "  Level:      $PERMISSION"

EXP_JSON=$(DATABRICKS_CONFIG_PROFILE="$PROFILE" databricks experiments get-by-name "$EXPERIMENT_PATH" 2>/dev/null || true)
EXPERIMENT_ID=$(printf '%s' "$EXP_JSON" | jq -r '.experiment.experiment_id // empty')

if [ -z "$EXPERIMENT_ID" ]; then
    echo "  Experiment not found — creating $EXPERIMENT_PATH"
    # NOTE: the Databricks CLI takes the experiment name POSITIONALLY
    # (`create-experiment NAME`); the `--name` flag does not exist and emits a
    # usage error that previously corrupted the jq parse below.
    CREATE_OUT=$(DATABRICKS_CONFIG_PROFILE="$PROFILE" databricks experiments create-experiment \
        "$EXPERIMENT_PATH" 2>&1)
    EXPERIMENT_ID=$(printf '%s' "$CREATE_OUT" | jq -r '.experiment_id // empty' 2>/dev/null)
    if [ -z "$EXPERIMENT_ID" ]; then
        echo "ERROR: Failed to create experiment $EXPERIMENT_PATH"
        echo "$CREATE_OUT"
        exit 1
    fi
fi
echo "  Experiment ID: $EXPERIMENT_ID"

SP_CLIENT_ID=$(DATABRICKS_CONFIG_PROFILE="$PROFILE" databricks apps get "$APP_NAME" 2>/dev/null \
    | jq -r '.service_principal_client_id // empty')
if [ -z "$SP_CLIENT_ID" ]; then
    echo "ERROR: Could not resolve service_principal_client_id for app '$APP_NAME'"
    exit 1
fi
echo "  App SP:        $SP_CLIENT_ID"

DATABRICKS_CONFIG_PROFILE="$PROFILE" databricks permissions update experiments "$EXPERIMENT_ID" --json "$(cat <<EOF
{
  "access_control_list": [
    {"service_principal_name": "$SP_CLIENT_ID", "permission_level": "$PERMISSION"}
  ]
}
EOF
)" > /dev/null

echo ""
echo "SUCCESS: Granted $PERMISSION on $EXPERIMENT_PATH to $APP_NAME"
