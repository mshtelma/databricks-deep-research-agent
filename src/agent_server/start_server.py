"""Agent Server entry point.

This module provides the entry point for running the Databricks Agent Server.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI

from src.core.config import get_settings
from src.core.tracing import setup_tracing
from src.main import create_app

logger = logging.getLogger(__name__)


def create_agent_app() -> FastAPI:
    """Create the agent server FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    # Setup tracing first
    settings = get_settings()
    setup_tracing(experiment_name=settings.mlflow_experiment_name)

    # Create and return the app
    app = create_app()

    logger.info("Agent server initialized")
    return app


def main() -> None:
    """Run the agent server."""
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting agent server on {host}:{port}")

    # Create app
    app = create_agent_app()

    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
