"""Agent client for communicating with deployed Deep Research Agent."""

import json
import logging
import os
import traceback
from typing import AsyncGenerator, Dict, List, Optional
import httpx
from databricks.sdk import WorkspaceClient
from pydantic import BaseModel
from fastapi import HTTPException


# Set up logger
logger = logging.getLogger(__name__)


def is_development_mode() -> bool:
    """Check if development mode is enabled."""
    dev_mode = os.getenv("DEVELOPMENT_MODE", "false").lower()
    return dev_mode in ("true", "1", "yes", "on")


class AgentMessage(BaseModel):
    """Message format for agent communication."""
    role: str  # "user" | "assistant" 
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None


class ResearchMetadata(BaseModel):
    """Research-specific metadata from agent."""
    search_queries: List[str] = []
    sources: List[Dict] = []
    research_iterations: int = 0
    confidence_score: Optional[float] = None
    reasoning_steps: List[str] = []


class StreamEvent(BaseModel):
    """Real-time stream event structure."""
    type: str  # "message_start" | "content_delta" | "research_update" | "message_complete"
    content: Optional[str] = None
    metadata: Optional[ResearchMetadata] = None


class AgentClient:
    """Client for communicating with deployed Deep Research Agent."""
    
    def __init__(self, workspace_client: Optional[WorkspaceClient] = None):
        """Initialize agent client with workspace client."""
        self.workspace = workspace_client or self._create_workspace_client()
        self.agent_endpoint = self._get_agent_endpoint()
        self.http_client = httpx.AsyncClient(timeout=300.0)  # 5min timeout for research
        
    def _get_auth_headers(self) -> dict:
        """Get authentication headers for HTTP requests."""
        # Check if we have a direct token (PAT authentication)
        if self.workspace.config.token:
            logger.debug(f"Using PAT authentication (token length: {len(self.workspace.config.token)})")
            return {"Authorization": f"Bearer {self.workspace.config.token}"}
        
        # For CLI profile authentication, the SDK handles auth automatically
        # but we need to get the token manually for direct HTTP requests
        if self.workspace.config.auth_type == "databricks-cli":
            try:
                import subprocess
                import json
                
                # Determine which profile to use
                profile = os.getenv("AGENT_CONFIG_PROFILE") or os.getenv("DATABRICKS_CONFIG_PROFILE")
                
                logger.debug(f"Getting token for CLI profile: {profile}")
                
                # Get token using databricks CLI
                cmd = ['databricks', 'auth', 'token']
                if profile:
                    cmd.extend(['--profile', profile])
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Try to parse as JSON first (newer CLI versions)
                try:
                    token_data = json.loads(result.stdout.strip())
                    token = token_data.get('access_token', result.stdout.strip())
                except json.JSONDecodeError:
                    # Fallback to raw output (older CLI versions)
                    token = result.stdout.strip()
                
                logger.info(f"Retrieved token via CLI profile '{profile}' (length: {len(token)})")
                return {"Authorization": f"Bearer {token}"}
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to get token via CLI: {e}")
                logger.error(f"CLI stderr: {e.stderr}")
                raise ValueError(f"Failed to get authentication token via CLI: {e}")
            except Exception as e:
                logger.error(f"Unexpected error getting token: {str(e)}")
                raise ValueError(f"Failed to get authentication token: {str(e)}")
        
        # Fallback - this shouldn't happen but provides useful error info
        logger.error(f"Unsupported auth_type: {self.workspace.config.auth_type}")
        raise ValueError(f"Unsupported authentication type: {self.workspace.config.auth_type}")
    
    def _create_workspace_client(self) -> WorkspaceClient:
        """Create workspace client with agent-specific authentication fallback."""
        try:
            # Check agent-specific credentials first
            if os.getenv("AGENT_DATABRICKS_TOKEN") and os.getenv("AGENT_DATABRICKS_HOST"):
                logger.info(f"Using agent-specific PAT authentication for host: {os.getenv('AGENT_DATABRICKS_HOST')}")
                return WorkspaceClient(
                    host=os.getenv("AGENT_DATABRICKS_HOST"),
                    token=os.getenv("AGENT_DATABRICKS_TOKEN")
                )
            elif os.getenv("AGENT_CONFIG_PROFILE"):
                logger.info(f"Using agent-specific profile authentication: {os.getenv('AGENT_CONFIG_PROFILE')}")
                from databricks.sdk.core import Config
                return WorkspaceClient(config=Config(profile=os.getenv("AGENT_CONFIG_PROFILE")))
            elif os.getenv("DATABRICKS_TOKEN") and os.getenv("DATABRICKS_HOST"):
                logger.info(f"Using workspace PAT authentication for host: {os.getenv('DATABRICKS_HOST')}")
                return WorkspaceClient(
                    host=os.getenv("DATABRICKS_HOST"),
                    token=os.getenv("DATABRICKS_TOKEN")
                )
            elif os.getenv("DATABRICKS_CONFIG_PROFILE"):
                logger.info(f"Using workspace profile authentication: {os.getenv('DATABRICKS_CONFIG_PROFILE')}")
                from databricks.sdk.core import Config
                return WorkspaceClient(config=Config(profile=os.getenv("DATABRICKS_CONFIG_PROFILE")))
            else:
                logger.info("Using default SDK authentication (environment variables or default profile)")
                return WorkspaceClient()
        except Exception as e:
            logger.error(f"Failed to create workspace client: {str(e)}")
            logger.error(f"Authentication config: AGENT_TOKEN={'***' if os.getenv('AGENT_DATABRICKS_TOKEN') else None}, "
                        f"AGENT_HOST={os.getenv('AGENT_DATABRICKS_HOST')}, "
                        f"AGENT_PROFILE={os.getenv('AGENT_CONFIG_PROFILE')}, "
                        f"WORKSPACE_TOKEN={'***' if os.getenv('DATABRICKS_TOKEN') else None}, "
                        f"WORKSPACE_HOST={os.getenv('DATABRICKS_HOST')}, "
                        f"WORKSPACE_PROFILE={os.getenv('DATABRICKS_CONFIG_PROFILE')}")
            logger.error(f"Workspace client creation traceback: {traceback.format_exc()}")
            raise ValueError(f"Authentication failed: {str(e)}. Check your Databricks credentials and configuration.")
    
    def _get_agent_endpoint(self) -> str:
        """Construct agent endpoint URL from host and endpoint name."""
        try:
            # Check for direct endpoint URL in environment (backward compatibility)
            endpoint_url = os.getenv("AGENT_ENDPOINT_URL")
            if endpoint_url:
                logger.info(f"Using direct agent endpoint URL: {endpoint_url}")
                return endpoint_url
            
            # For development mode, provide a mock endpoint
            if is_development_mode():
                logger.info("Development mode enabled - using mock agent endpoint")
                return "http://localhost:8001/mock-agent"
            
            # Get host - prefer agent-specific, fall back to workspace
            agent_host = os.getenv("AGENT_DATABRICKS_HOST")
            if not agent_host:
                try:
                    agent_host = self.workspace.config.host
                    logger.info(f"Using workspace host for agent: {agent_host}")
                except Exception as e:
                    logger.error(f"Failed to get workspace host: {str(e)}")
                    raise ValueError(f"Failed to get workspace host: {str(e)}")
                    
                if agent_host and not agent_host.startswith("https://"):
                    agent_host = f"https://{agent_host}"
            else:
                logger.info(f"Using agent-specific host: {agent_host}")
            
            # Get endpoint name
            endpoint_name = os.getenv("AGENT_ENDPOINT_NAME", "deep-research-agent")
            logger.info(f"Using agent endpoint name: {endpoint_name}")
            
            # Construct the serving endpoint URL
            if agent_host:
                endpoint_url = f"{agent_host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
                logger.info(f"Constructed agent endpoint URL: {endpoint_url}")
                return endpoint_url
            else:
                error_msg = "No agent host configured. Please set AGENT_DATABRICKS_HOST or configure workspace authentication."
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Failed to get agent endpoint: {str(e)}")
            logger.error(f"Endpoint configuration: AGENT_ENDPOINT_URL={os.getenv('AGENT_ENDPOINT_URL')}, "
                        f"AGENT_HOST={os.getenv('AGENT_DATABRICKS_HOST')}, "
                        f"ENDPOINT_NAME={os.getenv('AGENT_ENDPOINT_NAME')}")
            logger.error(f"Agent endpoint traceback: {traceback.format_exc()}")
            raise
    
    async def send_message(
        self, 
        messages: List[AgentMessage],
        config: Optional[Dict] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send message to agent and yield streaming responses."""
        
        # Convert to agent request format (Databricks schema)
        request_data = {
            "input": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_output_tokens": config.get("max_tokens", 4000) if config else 4000,
            "temperature": config.get("temperature", 0.7) if config else 0.7,
            "stream": True
        }
        
        try:
            # For development mode, simulate streaming response
            if is_development_mode():
                async for event in self._simulate_agent_response(messages[-1].content):
                    yield event
                return
            
            # Real agent endpoint streaming
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            async with self.http_client.stream(
                "POST",
                self.agent_endpoint,
                json=request_data,
                headers=headers
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code, 
                        detail=f"Agent request failed: {error_text.decode()}"
                    )
                
                # Parse streaming response
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        raw_data = line[6:]  # Remove "data: " prefix
                        
                        # Handle [DONE] marker
                        if raw_data.strip() == "[DONE]":
                            yield StreamEvent(type="stream_end")
                            break
                        
                        try:
                            data = json.loads(raw_data)
                            yield self._parse_stream_event(data)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error parsing stream event: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Agent communication failed: {str(e)}")
            logger.error(f"Request data: {json.dumps(request_data, indent=2)}")
            logger.error(f"Agent endpoint: {self.agent_endpoint}")
            logger.error(f"Agent communication traceback: {traceback.format_exc()}")
            
            # Determine error type for better user messaging
            error_type = "Unknown error"
            if "Authentication" in str(e) or "401" in str(e):
                error_type = "Authentication error"
            elif "404" in str(e):
                error_type = "Agent endpoint not found"
            elif "timeout" in str(e).lower() or "TimeoutError" in str(type(e).__name__):
                error_type = "Request timeout"
            elif "Connection" in str(e) or "Network" in str(e):
                error_type = "Network error"
            
            # Yield error event with detailed information
            yield StreamEvent(
                type="error", 
                content=f"{error_type}: {str(e)}"
            )
    
    async def _simulate_agent_response(self, query: str) -> AsyncGenerator[StreamEvent, None]:
        """Simulate agent response for development mode."""
        import asyncio
        
        # Start event
        yield StreamEvent(type="stream_start")
        await asyncio.sleep(0.5)
        
        # Research phases
        phases = [
            ("querying", "Analyzing your query and generating search strategies..."),
            ("searching", "Searching through multiple sources..."),
            ("analyzing", "Analyzing search results and extracting insights..."),
            ("synthesizing", "Synthesizing comprehensive response...")
        ]
        
        full_response = ""
        
        for phase, description in phases:
            # Progress update
            yield StreamEvent(
                type="research_update",
                metadata=ResearchMetadata(
                    search_queries=["query 1", "query 2"] if phase != "querying" else [],
                    sources=[{"url": "example.com", "title": "Example Source"}] if phase in ["analyzing", "synthesizing"] else [],
                    research_iterations=1 if phase == "synthesizing" else 0
                )
            )
            await asyncio.sleep(1.0)
            
            # Content delta
            content_chunk = f"\n\n**{phase.title()} Phase:**\n{description}\n"
            full_response += content_chunk
            yield StreamEvent(type="content_delta", content=content_chunk)
            await asyncio.sleep(0.5)
        
        # Final response content
        final_content = f"\n\nBased on my research, here's what I found about: {query}\n\nThis is a simulated response for development. The actual agent would provide comprehensive research-backed answers with citations and detailed analysis."
        full_response += final_content
        
        for chunk in [final_content[i:i+50] for i in range(0, len(final_content), 50)]:
            yield StreamEvent(type="content_delta", content=chunk)
            await asyncio.sleep(0.1)
        
        # Complete event
        yield StreamEvent(
            type="message_complete",
            metadata=ResearchMetadata(
                search_queries=["primary query", "secondary query"],
                sources=[
                    {"url": "https://example.com", "title": "Example Research Source", "relevanceScore": 0.95},
                    {"url": "https://research.org", "title": "Research Paper", "relevanceScore": 0.87}
                ],
                research_iterations=2,
                confidence_score=0.92
            )
        )
    
    def _parse_stream_event(self, data: Dict) -> StreamEvent:
        """Parse agent stream event to UI format."""
        # Handle Databricks agent response format
        response_type = data.get("type", "")
        
        # Streaming content delta
        if response_type == "response.output_text.delta":
            content = data.get("delta", "")
            return StreamEvent(type="content_delta", content=content)
        
        # Complete content (for final response)
        elif response_type == "response.output_item.done":
            item = data.get("item", {})
            content_list = item.get("content", [])
            if content_list and len(content_list) > 0:
                content = content_list[0].get("text", "")
                return StreamEvent(type="message_complete", content=content)
            return StreamEvent(type="message_complete", content="")
        
        # Legacy format support (for backward compatibility)
        elif "content" in data:
            return StreamEvent(type="content_delta", content=data["content"])
        elif "metadata" in data:
            return StreamEvent(type="research_update", metadata=self._extract_metadata(data))
        
        # Default case
        else:
            return StreamEvent(type="content_delta", content="")
    
    def _extract_metadata(self, data: Dict) -> ResearchMetadata:
        """Extract research metadata from agent response."""
        metadata = data.get("metadata", {})
        return ResearchMetadata(
            search_queries=metadata.get("search_queries", []),
            sources=metadata.get("sources", []),
            research_iterations=metadata.get("iterations", 0),
            confidence_score=metadata.get("confidence", None),
            reasoning_steps=metadata.get("reasoning", [])
        )
    
    async def send_simple_message(
        self, 
        messages: List[AgentMessage],
        config: Optional[Dict] = None
    ) -> Dict:
        """Send message to agent and return complete response (non-streaming)."""
        full_response = ""
        final_metadata = None
        
        async for event in self.send_message(messages, config):
            if event.type == "content_delta" and event.content:
                full_response += event.content
            elif event.type == "message_complete" and event.metadata:
                final_metadata = event.metadata
            elif event.type == "error":
                raise HTTPException(status_code=500, detail=event.content)
        
        return {
            "message": {"role": "assistant", "content": full_response},
            "metadata": final_metadata.dict() if final_metadata else None
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()