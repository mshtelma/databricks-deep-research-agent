"""
Deep Research Agent Server with embedded UI serving.

This module provides a unified server that serves both the sophisticated
multi-agent research system and the React UI as static files.
"""

# Load environment variables first, before any other imports
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables with priority: .env.local > .env > system environment
env_root = Path(__file__).parent.parent.parent.parent  # Go up to agent/ directory
if (env_root / ".env.local").exists():
    load_dotenv(env_root / ".env.local", override=True)
    print(f"✅ Loaded environment from: {env_root / '.env.local'}")
elif (env_root / ".env").exists():
    load_dotenv(env_root / ".env")
    print(f"✅ Loaded environment from: {env_root / '.env'}")
else:
    print("ℹ️  No .env files found, using system environment variables")

import sys
from typing import Generator
import logging

import mlflow
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# Import from the current server package
from .mlflow_config import setup_mlflow
from .server import create_server, invoke, parse_server_args, stream

# Import your existing sophisticated research agent
from ..databricks_compatible_agent import DatabricksCompatibleAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################
# Agent Configuration
############################################

# Initialize the sophisticated multi-agent research system
try:
    # Find configuration file in the new structure
    current_file = Path(__file__)
    agent_root = current_file.parent.parent.parent.parent  # Go up to agent/ directory

    possible_config_paths = [
        agent_root / "conf" / "base.yaml",  # Enhanced config
        current_file.parent.parent / "agent_config.yaml",  # In deep_research_agent/
        current_file.parent.parent / "conf" / "base.yaml",  # Legacy location
    ]

    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break

    logger.info(f"Initializing DatabricksCompatibleAgent with config: {config_path}")

    RESEARCH_AGENT = DatabricksCompatibleAgent(
        yaml_path=str(config_path) if config_path else None,
    )

    logger.info("Successfully initialized Deep Research Agent with multi-agent architecture")

except Exception as e:
    logger.error(f"Failed to initialize research agent: {e}")
    # Create a minimal fallback agent for development
    logger.warning("Creating fallback agent - some features may not work")




############################################
# Agent Endpoint Functions
############################################

@invoke()
def predict(request: dict) -> ResponsesAgentResponse:
    """
    Non-streaming prediction endpoint.

    Processes research requests using the sophisticated multi-agent system
    with planning, research, fact-checking, and report generation.
    """
    logger.info("Processing non-streaming research request")

    try:
        # Convert dict to proper request format
        agent_request = ResponsesAgentRequest(**request)

        # Delegate to the sophisticated research agent
        result = RESEARCH_AGENT.predict(agent_request)

        logger.info("Successfully completed non-streaming research request")
        return result

    except Exception as e:
        logger.error(f"Error in predict: {e}")
        # Return error response in proper format
        return ResponsesAgentResponse(
            output=[{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"Error processing request: {str(e)}"}],
                "id": "error-1",
            }],
            custom_outputs={"error": str(e)}
        )


@stream()
def predict_stream(request: dict) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """
    Streaming prediction endpoint with real-time progress updates.

    Streams intermediate events from the multi-agent research workflow:
    - Coordinator agent routing
    - Background investigation
    - Planning with iterative refinement
    - Step-by-step research execution
    - Fact checking and grounding
    - Professional report generation
    """
    logger.info("Processing streaming research request")

    try:
        # Convert dict to proper request format
        agent_request = ResponsesAgentRequest(**request)

        # Stream from the sophisticated research agent
        # This provides real-time updates from all 5 agents:
        # Coordinator -> Planner -> Researcher -> Fact Checker -> Reporter
        yield from RESEARCH_AGENT.predict_stream(agent_request)

        logger.info("Successfully completed streaming research request")

    except Exception as e:
        logger.error(f"Error in predict_stream: {e}")
        # Return error event in proper streaming format
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"Error processing streaming request: {str(e)}"}],
                "id": "stream-error-1",
            }
        )


###########################################
# Required components to start the server #
###########################################

# Create the agent server with responses API type
agent_server = create_server("agent/v1/responses")
app = agent_server.app  # Required for multi-worker support


def main():
    """
    Main entry point for the Deep Research Agent Server.

    This function:
    1. Sets up MLflow configuration and tracing
    2. Configures the unified server (agent + UI)
    3. Starts the server with configurable options
    """
    args = parse_server_args()

    # Setup MLflow for tracing and experiments
    setup_mlflow()

    logger.info("=" * 60)
    logger.info("🚀 Deep Research Agent Server Starting")
    logger.info("=" * 60)
    logger.info(f"Server: agent + UI on port {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Agent: Multi-agent research system (5 agents)")
    logger.info(f"UI: React interface served at /")
    logger.info(f"API: /invocations endpoint for agent requests")
    logger.info(f"Health: /health endpoint for status checks")
    logger.info("=" * 60)

    # Start the unified server
    agent_server.run(
        "deep_research_agent.server.app:app",  # Import string for app defined above (supports workers)
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
