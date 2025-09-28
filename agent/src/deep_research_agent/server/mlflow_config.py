import logging
import os
import subprocess
from pathlib import Path

import mlflow


def get_git_commit() -> str:
    """Get the current Git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent).decode("utf-8").strip()
        return commit[:8]  # Return short commit hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def setup_mlflow(register_model_to_uc: bool = False):
    """Setup MLflow configuration for the agent server."""
    logger = logging.getLogger(__name__)

    # Set MLflow tracking and registry URIs
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # Set experiment if provided
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
        logger.info(f"Using MLflow experiment: {experiment_id}")
    else:
        logger.warning("MLFLOW_EXPERIMENT_ID not set - traces may not be logged properly")

    # Get git commit for versioning
    commit_hash = get_git_commit()
    logger.info(f"Current git commit: {commit_hash}")

    # Set run tags
    mlflow.set_tags({
        "git_commit": commit_hash,
        "agent_type": "deep_research_agent",
        "framework": "langgraph",
    })

    # Register model to Unity Catalog if requested
    if register_model_to_uc:
        try:
            # TODO: Configure these values for your UC model
            catalog = os.getenv("UC_CATALOG", "your_catalog")
            schema = os.getenv("UC_SCHEMA", "your_schema")
            model_name = os.getenv("UC_MODEL_NAME", "deep_research_agent")

            model_uri = f"models:/{catalog}.{schema}.{model_name}/latest"
            logger.info(f"Registering model to UC: {model_uri}")

            # This would be implemented if using model registration
            # mlflow.register_model(model_uri=..., name=f"{catalog}.{schema}.{model_name}")

        except Exception as e:
            logger.error(f"Failed to register model to UC: {e}")

    logger.info("MLflow configuration complete")