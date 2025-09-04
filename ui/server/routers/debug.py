"""Debug router for configuration testing and troubleshooting."""

import logging
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.agent_client import AgentClient, is_development_mode

router = APIRouter()
logger = logging.getLogger(__name__)


class ConfigResponse(BaseModel):
    """Configuration status response."""

    authentication: Dict[str, Any]
    agent_config: Dict[str, Any]
    workspace_connection: Dict[str, Any]
    agent_endpoint: Dict[str, Any]


class TestConnectionResponse(BaseModel):
    """Connection test response."""

    success: bool
    error: Optional[str] = None
    details: Dict[str, Any] = {}


@router.get("/config", response_model=ConfigResponse)
async def get_debug_config():
    """Get current configuration for debugging (no secrets exposed)."""
    try:
        logger.info("Debug config requested")

        # Authentication configuration
        auth_config = {
            "development_mode": is_development_mode(),
            "workspace_host": os.getenv("DATABRICKS_HOST"),
            "workspace_token_configured": bool(os.getenv("DATABRICKS_TOKEN")),
            "workspace_profile": os.getenv("DATABRICKS_CONFIG_PROFILE"),
            "agent_host": os.getenv("AGENT_DATABRICKS_HOST"),
            "agent_token_configured": bool(os.getenv("AGENT_DATABRICKS_TOKEN")),
            "agent_profile": os.getenv("AGENT_CONFIG_PROFILE"),
        }

        # Agent configuration
        agent_config = {
            "endpoint_name": os.getenv("AGENT_ENDPOINT_NAME", "deep-research-agent"),
            "endpoint_url": os.getenv("AGENT_ENDPOINT_URL"),
            "development_mode": is_development_mode(),
        }

        # Test workspace connection
        workspace_connection = {"status": "unknown", "error": None}
        try:
            client = AgentClient()
            workspace_connection = {
                "status": "success",
                "host": client.workspace_client.config.host,
                "error": None,
            }
        except Exception as e:
            workspace_connection = {"status": "failed", "error": str(e)}

        # Test agent endpoint
        agent_endpoint = {"status": "unknown", "url": None, "error": None}
        try:
            client = AgentClient()
            agent_endpoint = {"status": "configured", "url": client.agent_endpoint, "error": None}
        except Exception as e:
            agent_endpoint = {"status": "failed", "url": None, "error": str(e)}

        return ConfigResponse(
            authentication=auth_config,
            agent_config=agent_config,
            workspace_connection=workspace_connection,
            agent_endpoint=agent_endpoint,
        )

    except Exception as e:
        logger.error(f"Debug config failed: {str(e)}")
        logger.error(f"Debug config traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get debug config: {str(e)}")


@router.post("/test-workspace", response_model=TestConnectionResponse)
async def test_workspace_connection():
    """Test workspace connection and authentication."""
    try:
        logger.info("Testing workspace connection")

        # Try to create and test workspace client
        client = AgentClient()

        # Try to get current user to test authentication
        try:
            current_user = client.workspace_client.current_user.me()
            return TestConnectionResponse(
                success=True,
                details={
                    "user": current_user.user_name,
                    "workspace_host": client.workspace_client.config.host,
                    "authentication": "success",
                },
            )
        except Exception as e:
            return TestConnectionResponse(
                success=False,
                error=f"Authentication failed: {str(e)}",
                details={"workspace_host": client.workspace_client.config.host, "error_type": type(e).__name__},
            )

    except Exception as e:
        logger.error(f"Workspace connection test failed: {str(e)}")
        logger.error(f"Workspace test traceback: {traceback.format_exc()}")
        return TestConnectionResponse(
            success=False,
            error=f"Failed to create workspace client: {str(e)}",
            details={"error_type": type(e).__name__},
        )


@router.post("/test-agent", response_model=TestConnectionResponse)
async def test_agent_connection():
    """Test agent endpoint reachability (without sending actual message)."""
    try:
        logger.info("Testing agent endpoint connection")

        if is_development_mode():
            return TestConnectionResponse(
                success=True,
                details={
                    "mode": "development",
                    "endpoint": "mock",
                    "message": "Development mode - using simulated responses",
                },
            )

        # Create agent client
        client = AgentClient()

        # Test if endpoint is reachable (HEAD request)
        import httpx

        async with httpx.AsyncClient() as http_client:
            try:
                response = await http_client.head(
                    client.agent_endpoint.replace("/invocations", ""),
                    headers={"Authorization": f"Bearer {client.workspace_client.config.token}"},
                    timeout=10.0,
                )

                return TestConnectionResponse(
                    success=response.status_code in [200, 404],  # 404 is OK for HEAD to serving endpoint
                    details={
                        "endpoint_url": client.agent_endpoint,
                        "status_code": response.status_code,
                        "reachable": True,
                    },
                )
            except httpx.TimeoutException:
                return TestConnectionResponse(
                    success=False,
                    error="Connection timeout",
                    details={"endpoint_url": client.agent_endpoint, "timeout": "10s"},
                )
            except Exception as e:
                return TestConnectionResponse(
                    success=False,
                    error=f"Connection failed: {str(e)}",
                    details={"endpoint_url": client.agent_endpoint, "error_type": type(e).__name__},
                )

    except Exception as e:
        logger.error(f"Agent connection test failed: {str(e)}")
        logger.error(f"Agent test traceback: {traceback.format_exc()}")
        return TestConnectionResponse(
            success=False,
            error=f"Failed to test agent connection: {str(e)}",
            details={"error_type": type(e).__name__},
        )


@router.get("/logs")
async def get_recent_logs():
    """Get recent application logs for debugging."""
    try:
        log_file = "/tmp/databricks-app.log"
        if os.path.exists(log_file):
            # Read last 100 lines
            with open(log_file, "r") as f:
                lines = f.readlines()
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                return {"logs": recent_lines, "total_lines": len(lines)}
        else:
            return {"logs": [], "message": "Log file not found"}

    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


@router.get("/table-test")
async def get_table_test():
    """Get test table content for markdown rendering debugging."""
    test_content = """**Apple Inc. (AAPL) – Current Sentiment Overview (as of 2 Sept 2025)**

Below is a consolidated, data‑driven snapshot of how the market is feeling about Apple's stock today. The analysis pulls together three primary "sentiment streams" that investors rely on:

| Sentiment Source | Recent Data (last 30 days) | Bullish % | Neutral % | Bearish % | Key Drivers |
|------------------|---------------------------|-----------|-----------|-----------|-------------|
| Equity‑research consensus (30 analysts) | 2025‑Q3 earnings beat, new iPhone 17 launch, services‑revenue growth | 62 % (Buy) | 30 % (Hold) | 8 % (Sell) | Strong earnings, AI‑enhanced hardware, modest supply‑chain risk |
| Financial‑news sentiment (Bloomberg, Reuters, MarketWatch, WSJ) – NLP‑derived tone on 5 000 articles | 4 800 positive / 600 negative mentions | 78 % positive | 15 % neutral | 7 % negative | Positive coverage of Apple Vision Pro 2, services expansion, macro‑rate outlook |
| Social‑media sentiment (Twitter, Reddit r/investing, StockTwits) – sentiment‑score model (VADER + finBERT) on 120 000 posts | 68 % positive / 22 % neutral / 10 % negative | 68 % bullish | 22 % neutral | 10 % bearish | Retail enthusiasm for new product ecosystem; some concern over China‑sales slowdown |
| Sentiment‑index (Sentix/Thomson Reuters) – "Investor Confidence" gauge for U.S. tech | Index at +12.4 pts (above 10‑yr avg) | — | — | — | Reflects broader optimism in U.S. tech despite higher rates |
| Short‑interest & options positioning | Short‑float ≈ 3.1 % (down 0.5 pts YoY); Put‑call ratio ≈ 0.62 (bullish) | — | — | — | Low short pressure, net‑long options market |

All percentages are rounded to the nearest whole number."""

    return {"content": test_content}
