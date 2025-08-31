"""Debug router for configuration testing and troubleshooting."""

import logging
import os
import traceback
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.agent_client import AgentClient, is_development_mode
from server.services.user_service import UserService

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
                "host": client.workspace.config.host,
                "error": None
            }
        except Exception as e:
            workspace_connection = {
                "status": "failed", 
                "error": str(e)
            }
        
        # Test agent endpoint
        agent_endpoint = {"status": "unknown", "url": None, "error": None}
        try:
            client = AgentClient()
            agent_endpoint = {
                "status": "configured",
                "url": client.agent_endpoint,
                "error": None
            }
        except Exception as e:
            agent_endpoint = {
                "status": "failed",
                "url": None,
                "error": str(e)
            }
        
        return ConfigResponse(
            authentication=auth_config,
            agent_config=agent_config,
            workspace_connection=workspace_connection,
            agent_endpoint=agent_endpoint
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
            current_user = client.workspace.current_user.me()
            return TestConnectionResponse(
                success=True,
                details={
                    "user": current_user.user_name,
                    "workspace_host": client.workspace.config.host,
                    "authentication": "success"
                }
            )
        except Exception as e:
            return TestConnectionResponse(
                success=False,
                error=f"Authentication failed: {str(e)}",
                details={
                    "workspace_host": client.workspace.config.host,
                    "error_type": type(e).__name__
                }
            )
    
    except Exception as e:
        logger.error(f"Workspace connection test failed: {str(e)}")
        logger.error(f"Workspace test traceback: {traceback.format_exc()}")
        return TestConnectionResponse(
            success=False,
            error=f"Failed to create workspace client: {str(e)}",
            details={"error_type": type(e).__name__}
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
                    "message": "Development mode - using simulated responses"
                }
            )
        
        # Create agent client
        client = AgentClient()
        
        # Test if endpoint is reachable (HEAD request)
        import httpx
        async with httpx.AsyncClient() as http_client:
            try:
                response = await http_client.head(
                    client.agent_endpoint.replace("/invocations", ""),
                    headers={"Authorization": f"Bearer {client.workspace.config.token}"},
                    timeout=10.0
                )
                
                return TestConnectionResponse(
                    success=response.status_code in [200, 404],  # 404 is OK for HEAD to serving endpoint
                    details={
                        "endpoint_url": client.agent_endpoint,
                        "status_code": response.status_code,
                        "reachable": True
                    }
                )
            except httpx.TimeoutException:
                return TestConnectionResponse(
                    success=False,
                    error="Connection timeout",
                    details={
                        "endpoint_url": client.agent_endpoint,
                        "timeout": "10s"
                    }
                )
            except Exception as e:
                return TestConnectionResponse(
                    success=False,
                    error=f"Connection failed: {str(e)}",
                    details={
                        "endpoint_url": client.agent_endpoint,
                        "error_type": type(e).__name__
                    }
                )
    
    except Exception as e:
        logger.error(f"Agent connection test failed: {str(e)}")
        logger.error(f"Agent test traceback: {traceback.format_exc()}")
        return TestConnectionResponse(
            success=False,
            error=f"Failed to test agent connection: {str(e)}",
            details={"error_type": type(e).__name__}
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