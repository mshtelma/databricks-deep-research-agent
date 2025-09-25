"""Agent client for communicating with deployed Deep Research Agent."""

import json
import logging
import os
import traceback
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from databricks.sdk import WorkspaceClient
from fastapi import HTTPException
from pydantic import BaseModel

# Set up logger
logger = logging.getLogger(__name__)


def is_databricks_app_environment() -> bool:
    """Detect if running inside a Databricks Apps container.

    Databricks sets several app-specific environment variables that are
    guaranteed to be present inside an App container but are unlikely to be
    set in local development or general Databricks jobs / notebooks.  The
    following variables are currently observed:

    * ``DATABRICKS_APP_NAME``
    * ``DATABRICKS_APP_URL``
    * ``DATABRICKS_APP_PORT``

    Presence of any one of them is treated as a positive signal that the code
    is running inside a Databricks App.
    """
    return bool(os.getenv("DATABRICKS_APP_NAME") or os.getenv("DATABRICKS_APP_URL") or os.getenv("DATABRICKS_APP_PORT"))


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

    # Enhanced progress tracking fields
    total_sources_found: int = 0
    phase: str = "processing"
    progress_percentage: float = 0.0
    elapsed_time: float = 0.0
    current_node: str = ""
    vector_results_count: int = 0

    # Multi-agent and plan visualization fields
    plan_details: Optional[Dict] = None
    factuality_score: Optional[float] = None
    report_style: Optional[str] = None
    verification_level: Optional[str] = None
    grounding: Optional[Dict] = None
    current_agent: Optional[str] = None


class StreamEvent(BaseModel):
    """Real-time stream event structure."""

    type: str  # "message_start" | "content_delta" | "research_update" | "message_complete"
    content: Optional[str] = None
    metadata: Optional[ResearchMetadata] = None
    event: Optional[Dict] = None  # For intermediate events
    events: Optional[List[Dict]] = None  # For event batches
    batch_size: Optional[int] = None  # For event batches


class AgentClient:
    """Client for communicating with deployed Deep Research Agent."""

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None):
        """Initialize agent client with workspace client."""
        self.workspace_client = workspace_client or self._create_workspace_client()
        self.agent_endpoint = self._get_agent_endpoint()
        self.http_client = httpx.AsyncClient(timeout=300.0)  # 5min timeout for research

    def _get_auth_headers(self) -> dict:
        """Get authentication headers for HTTP requests."""

        # Check if we have a direct token (PAT authentication)
        if self.workspace_client.config.token:
            logger.info(f"Using PAT authentication (token length: {len(self.workspace_client.config.token)})")
            headers = {"Authorization": f"Bearer {self.workspace_client.config.token}"}
            self._log_headers("PAT Authentication", headers)
            return headers

        # Try to explicitly authenticate via the SDK which, inside Databricks
        # (including Apps), can mint a short-lived PAT on demand.  This method
        # is no-op outside the platform and raises if auth fails.
        if hasattr(self.workspace_client.config, "authenticate"):
            try:
                return self.workspace_client.config.authenticate()
            except Exception as e:
                logger.info(f"workspace.authenticate() raised an exception: {e}")

        # Handle OAuth (service principal / M2M) authentication that Databricks SDK
        # automatically selects inside Databricks jobs, model-serving and Apps. In
        # this mode ``config.token`` is ``None`` and ``auth_type`` starts with
        # "oauth" (e.g. "oauth-m2m").  The SDK stores a credential helper able
        # to produce an access token via ``get_access_token``.
        if self.workspace_client.config.auth_type and self.workspace_client.config.auth_type.startswith("oauth"):
            credentials = getattr(self.workspace_client.config, "credentials", None)
            try:
                if credentials is not None:
                    if hasattr(credentials, "get_access_token"):
                        token = credentials.get_access_token()
                    elif callable(credentials):  # older SDK interface
                        token = credentials().get("access_token")
                    else:
                        token = None

                    if token:
                        logger.info(
                            f"OAuth cred helper produced token (preview={token[:6]}...{token[-4:]}) auth_type={self.workspace_client.config.auth_type}"
                        )
                        logger.info(
                            f"Using OAuth ({self.workspace_client.config.auth_type}) authentication with dynamic token"
                        )
                        headers = {"Authorization": f"Bearer {token}"}
                        self._log_headers("OAuth Authentication", headers)
                        return headers
                    else:
                        # No token but OAuth mode - use empty headers (implicit auth)
                        logger.info(
                            f"OAuth mode ({self.workspace_client.config.auth_type}) but no token - using implicit authentication"
                        )
                        return {}
            except Exception as e:
                logger.warning(
                    f"Failed to obtain OAuth token via SDK credentials object: {e}. Proceeding without explicit header."
                )
                # For in-workspace calls an explicit header isn't mandatory; return empty headers
                logger.info("Using implicit OAuth authentication (no explicit token in header)")
                return {}

        # For CLI profile authentication, the SDK handles auth automatically
        # but we need to get the token manually for direct HTTP requests
        if self.workspace_client.config.auth_type == "databricks-cli":
            try:
                import json
                import subprocess

                # Determine which profile to use
                profile = os.getenv("AGENT_CONFIG_PROFILE") or os.getenv("DATABRICKS_CONFIG_PROFILE")

                logger.debug(f"Getting token for CLI profile: {profile}")

                # Get token using databricks CLI
                cmd = ["databricks", "auth", "token"]
                if profile:
                    cmd.extend(["--profile", profile])

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Try to parse as JSON first (newer CLI versions)
                try:
                    token_data = json.loads(result.stdout.strip())
                    token = token_data.get("access_token", result.stdout.strip())
                except json.JSONDecodeError:
                    # Fallback to raw output (older CLI versions)
                    token = result.stdout.strip()

                logger.info(f"Retrieved token via CLI profile '{profile}' (length: {len(token)})")
                headers = {"Authorization": f"Bearer {token}"}
                self._log_headers("CLI Profile Authentication", headers)
                return headers

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to get token via CLI: {e}")
                logger.error(f"CLI stderr: {e.stderr}")
                raise ValueError(f"Failed to get authentication token via CLI: {e}")
            except Exception as e:
                logger.error(f"Unexpected error getting token: {str(e)}")
                raise ValueError(f"Failed to get authentication token: {str(e)}")

        # Fallback - this shouldn't happen but provides useful error info
        logger.error(f"Unsupported auth_type: {self.workspace_client.config.auth_type}")
        raise ValueError(f"Unsupported authentication type: {self.workspace_client.config.auth_type}")

    def _create_workspace_client(self) -> WorkspaceClient:
        """Create workspace client with environment-aware authentication.

        In Databricks Apps we can instantiate ``WorkspaceClient()`` without
        parameters – the platform injects all required credentials via
        environment variables / workload identity.  Outside of Apps we keep
        the existing fallback logic to support local development as well as
        generic Databricks jobs.
        """
        # Fast-path: Databricks Apps container – rely on automatic auth
        if is_databricks_app_environment():
            logger.info("Detected Databricks App environment – using default WorkspaceClient() with auto-auth")
            return WorkspaceClient()

        try:
            # Check agent-specific credentials first
            if os.getenv("AGENT_DATABRICKS_TOKEN") and os.getenv("AGENT_DATABRICKS_HOST"):
                logger.info(f"Using agent-specific PAT authentication for host: {os.getenv('AGENT_DATABRICKS_HOST')}")
                return WorkspaceClient(
                    host=os.getenv("AGENT_DATABRICKS_HOST"), token=os.getenv("AGENT_DATABRICKS_TOKEN")
                )
            elif os.getenv("AGENT_CONFIG_PROFILE"):
                logger.info(f"Using agent-specific profile authentication: {os.getenv('AGENT_CONFIG_PROFILE')}")
                from databricks.sdk.core import Config

                return WorkspaceClient(config=Config(profile=os.getenv("AGENT_CONFIG_PROFILE")))
            elif os.getenv("DATABRICKS_TOKEN") and os.getenv("DATABRICKS_HOST"):
                logger.info(f"Using workspace PAT authentication for host: {os.getenv('DATABRICKS_HOST')}")
                return WorkspaceClient(host=os.getenv("DATABRICKS_HOST"), token=os.getenv("DATABRICKS_TOKEN"))
            elif os.getenv("DATABRICKS_CONFIG_PROFILE"):
                logger.info(f"Using workspace profile authentication: {os.getenv('DATABRICKS_CONFIG_PROFILE')}")
                from databricks.sdk.core import Config

                return WorkspaceClient(config=Config(profile=os.getenv("DATABRICKS_CONFIG_PROFILE")))
            else:
                logger.info("Using default SDK authentication (environment variables or default profile)")
                return WorkspaceClient()
        except Exception as e:
            logger.error(f"Failed to create workspace client: {str(e)}")
            logger.error(
                f"Authentication config: AGENT_TOKEN={'***' if os.getenv('AGENT_DATABRICKS_TOKEN') else None}, "
                f"AGENT_HOST={os.getenv('AGENT_DATABRICKS_HOST')}, "
                f"AGENT_PROFILE={os.getenv('AGENT_CONFIG_PROFILE')}, "
                f"WORKSPACE_TOKEN={'***' if os.getenv('DATABRICKS_TOKEN') else None}, "
                f"WORKSPACE_HOST={os.getenv('DATABRICKS_HOST')}, "
                f"WORKSPACE_PROFILE={os.getenv('DATABRICKS_CONFIG_PROFILE')}"
            )
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

            # Get host - prefer agent-specific, fall back to workspace
            agent_host = os.getenv("AGENT_DATABRICKS_HOST")
            if not agent_host:
                try:
                    agent_host = self.workspace_client.config.host
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
                error_msg = (
                    "No agent host configured. Please set AGENT_DATABRICKS_HOST or configure workspace authentication."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Failed to get agent endpoint: {str(e)}")
            logger.error(
                f"Endpoint configuration: AGENT_ENDPOINT_URL={os.getenv('AGENT_ENDPOINT_URL')}, "
                f"AGENT_HOST={os.getenv('AGENT_DATABRICKS_HOST')}, "
                f"ENDPOINT_NAME={os.getenv('AGENT_ENDPOINT_NAME')}"
            )
            logger.error(f"Agent endpoint traceback: {traceback.format_exc()}")
            raise

    async def send_message(
        self, messages: List[AgentMessage], config: Optional[Dict] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send message to agent and yield streaming responses."""
        # Convert to agent request format (Databricks schema)
        request_data = {
            "input": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_output_tokens": config.get("max_tokens", 4000) if config else 4000,
            "temperature": config.get("temperature", 0.7) if config else 0.7,
            "stream": True,
        }

        try:
            # Real agent endpoint streaming
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            headers["User-Agent"] = "Databricks-Deep-Research-UI/1.0"
            headers["Connection"] = "keep-alive"
            headers["Keep-Alive"] = "timeout=300"

            # Log full request details for debugging
            self._log_request_details("POST", self.agent_endpoint, request_data, headers)

            async with self.http_client.stream(
                "POST", self.agent_endpoint, json=request_data, headers=headers
            ) as response:
                # Log response details immediately
                self._log_response_details(response)
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code, detail=f"Agent request failed: {error_text.decode()}"
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
                            logger.debug(
                                f"🔴 [AGENT_CLIENT] Raw event received: type={data.get('type')}, has_delta={bool(data.get('delta'))}"
                            )
                            
                            # Extensive debug logging for events
                            if data.get("type") == "response.output_text.delta" and "[PHASE:" in data.get("delta", ""):
                                logger.info(f"Progress marker detected in delta: {data.get('delta', '')[:100]}")
                            
                            # Log intermediate events
                            if data.get("type") == "response.intermediate_event":
                                logger.info(f"🔴 [AGENT_CLIENT] Intermediate event: {json.dumps(data, indent=2)}")
                            
                            # Log event batches
                            if data.get("type") == "response.event_batch":
                                logger.info(f"🔴 [AGENT_CLIENT] Event batch with {len(data.get('events', []))} events")
                                for i, evt in enumerate(data.get('events', [])):
                                    logger.info(f"🔴 [AGENT_CLIENT] Event {i}: {evt.get('event_type', 'unknown')}")
                            
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
            yield StreamEvent(type="error", content=f"{error_type}: {str(e)}")

    def _parse_stream_event(self, data: Dict) -> StreamEvent:
        """Parse agent stream event to UI format with enhanced event support."""
        # Add defensive check for data
        if not data or not isinstance(data, dict):
            return StreamEvent(type="content_delta", content="")

        # Handle Databricks agent response format
        response_type = data.get("type", "")
        logger.info(f"🔴 [AGENT_CLIENT] Parsing stream event type: {response_type}")

        # Check for enhanced intermediate events first
        if response_type == "response.intermediate_event" or response_type == "intermediate_event" or "event" in data:
            logger.info(f"🔴 [AGENT_CLIENT] Processing intermediate event")
            logger.debug("🔴 [AGENT_CLIENT] Intermediate event payload: %s", json.dumps(data, indent=2))
            return self._parse_intermediate_event(data)
        elif response_type == "response.event_batch" or response_type == "event_batch" or "events" in data:
            logger.info(f"🔴 [AGENT_CLIENT] Processing event batch")
            logger.debug("🔴 [AGENT_CLIENT] Event batch payload: %s", json.dumps(data, indent=2))
            return self._parse_event_batch(data)

        # Streaming content delta
        elif response_type == "response.output_text.delta":
            content = data.get("delta", "")
            logger.info(
                "🔴 [AGENT_CLIENT] Streaming delta len=%s preview=%s",
                len(content),
                content[:200].replace("\n", "\\n") if isinstance(content, str) else content,
            )

            # Always send content delta, even if it contains progress markers
            # The UI will handle filtering out markers from displayed content
            return StreamEvent(type="content_delta", content=content)

        # Complete content (for final response)
        elif response_type == "response.metadata":
            event_metadata = data.get("metadata", {})
            logger.info(
                "🔴 [AGENT_CLIENT] Metadata event keys=%s",
                list(event_metadata.keys()) if isinstance(event_metadata, dict) else "non-dict",
            )
            metadata = self._convert_metadata_to_research_metadata(event_metadata)
            logger.debug("🔴 [AGENT_CLIENT] Metadata payload: %s", json.dumps(event_metadata, indent=2))
            return StreamEvent(type="research_update", metadata=metadata)

        elif response_type == "response.output_item.done":
            item = data.get("item", {})
            content_list = item.get("content", [])
            if content_list and len(content_list) > 0 and content_list[0]:
                content = content_list[0].get("text", "")
            else:
                content = ""

            logger.info(
                "🔴 [AGENT_CLIENT] Output item done len=%s preview=%s",
                len(content),
                content[:200].replace("\n", "\\n") if isinstance(content, str) else content,
            )

            # Extract metadata from the event if present
            event_metadata = data.get("metadata", {})
            metadata = None
            if event_metadata:
                # Convert metadata to ResearchMetadata format for UI
                metadata = self._convert_metadata_to_research_metadata(event_metadata, default_phase="complete", default_agent="reporter")
                logger.debug("🔴 [AGENT_CLIENT] Output item metadata payload: %s", json.dumps(event_metadata, indent=2))

            return StreamEvent(type="message_complete", content=content, metadata=metadata)

        # Legacy format support (for backward compatibility)
        elif "content" in data:
            logger.info(
                "🔴 [AGENT_CLIENT] Legacy content len=%s preview=%s",
                len(data.get("content", "")),
                str(data.get("content", ""))[:200].replace("\n", "\\n"),
            )
            return StreamEvent(type="content_delta", content=data["content"])
        elif "metadata" in data:
            logger.info(
                "🔴 [AGENT_CLIENT] Legacy metadata keys=%s",
                list(data.get("metadata", {}).keys()) if isinstance(data.get("metadata", {}), dict) else "non-dict",
            )
            return StreamEvent(type="research_update", metadata=self._convert_metadata_to_research_metadata(data.get("metadata", {})))

        # Default case
        else:
            logger.debug("🔴 [AGENT_CLIENT] Unrecognized payload: %s", json.dumps(data, indent=2))
            return StreamEvent(type="content_delta", content="")

    def _parse_intermediate_event(self, data: Dict) -> StreamEvent:
        """Parse enhanced intermediate event from agent."""
        if not data or not isinstance(data, dict):
            return StreamEvent(type="intermediate_event", event={})

        event_data = data.get("event", data)  # Handle both formats
        if not event_data or not isinstance(event_data, dict):
            return StreamEvent(type="intermediate_event", event={})

        # Create StreamEvent with enhanced event data
        return StreamEvent(
            type="intermediate_event",
            event={
                "id": event_data.get("id", "unknown"),
                "timestamp": event_data.get("timestamp", 0),
                "correlation_id": event_data.get("correlation_id"),
                "sequence": event_data.get("sequence", 0),
                "event_type": event_data.get("event_type", "action_start"),
                "data": event_data.get("data", {}),
                "meta": event_data.get("meta", {}),
                # Enhanced fields for UI
                "category": event_data.get("category"),
                "title": event_data.get("title"),
                "description": event_data.get("description"),
                "confidence": event_data.get("confidence"),
                "reasoning": event_data.get("reasoning"),
                "alternatives_considered": event_data.get("alternatives_considered", []),
                "related_event_ids": event_data.get("related_event_ids", []),
                "priority": event_data.get("priority", 5),
            },
        )

    def _parse_event_batch(self, data: Dict) -> StreamEvent:
        """Parse batch of intermediate events from agent."""
        events = data.get("events", [])

        # Process each event in the batch
        processed_events = []
        for event_data in events:
            processed_event = {
                "id": event_data.get("id", "unknown"),
                "timestamp": event_data.get("timestamp", 0),
                "correlation_id": event_data.get("correlation_id"),
                "sequence": event_data.get("sequence", 0),
                "event_type": event_data.get("event_type", "action_start"),
                "data": event_data.get("data", {}),
                "meta": event_data.get("meta", {}),
                # Enhanced fields for UI
                "category": event_data.get("category"),
                "title": event_data.get("title"),
                "description": event_data.get("description"),
                "confidence": event_data.get("confidence"),
                "reasoning": event_data.get("reasoning"),
                "alternatives_considered": event_data.get("alternatives_considered", []),
                "related_event_ids": event_data.get("related_event_ids", []),
                "priority": event_data.get("priority", 5),
            }
            processed_events.append(processed_event)

        return StreamEvent(type="event_batch", events=processed_events, batch_size=len(processed_events))

    def _parse_progress_delta(self, content: str) -> StreamEvent:
        """Parse progress markers from delta text to extract research phase information."""
        import re

        logger.debug(f"Parsing progress delta: {content[:200]}...")

        # Extract phase from content
        phase_match = re.search(r"\[PHASE:(\w+)\]", content)
        phase = phase_match.group(1).lower() if phase_match else "processing"

        # Direct 1:1 mapping - UI already has meaningful labels for all phases
        ui_phase_map = {
            "querying": "querying",
            "preparing": "preparing",
            "searching": "searching",
            "searching_internal": "searching_internal",
            "aggregating": "aggregating",
            "analyzing": "analyzing",
            "synthesizing": "synthesizing",
            "processing": "processing",
        }
        current_phase = ui_phase_map.get(phase, "processing")

        logger.info(f"Progress parsed: raw_phase={phase}, ui_phase={current_phase}")

        # Extract metadata from META markers
        metadata = {}
        for match in re.finditer(r"\[META:(\w+):([^\]]+)\]", content):
            key, value = match.groups()
            # Convert numeric values
            try:
                metadata[key] = float(value) if "." in value else int(value)
            except ValueError:
                metadata[key] = value

        # Extract human-readable message (remove markers for display)
        message_match = re.search(r"\[PHASE:\w+\]\s+([^\[]+?)(?:\[|$)", content, re.DOTALL)
        message = message_match.group(1).strip() if message_match else ""

        # Remove emoji if present but preserve the description
        if message:
            # Remove leading emoji but keep the descriptive text
            message = re.sub(r"^[🔍📋🌐🗄️📊🤔✍️⚙️]\s*", "", message).strip()

        # Build research metadata for the UI
        research_metadata = ResearchMetadata(
            search_queries=[],  # Can be populated from actual data later
            sources=[],
            confidence_score=None,
            research_iterations=metadata.get("queries", 0),
            total_sources_found=metadata.get("results", 0),
            phase=current_phase,
            progress_percentage=metadata.get("progress", 0.0),
            elapsed_time=metadata.get("elapsed", 0.0),
            current_node=metadata.get("node", ""),
            vector_results_count=metadata.get("vector_results", 0),
        )

        # Return as research update event
        logger.info(
            f"Creating research_update event: phase={current_phase}, progress={metadata.get('progress', 0)}%, metadata_keys={list(metadata.keys())}"
        )
        return StreamEvent(
            type="research_update",
            content=message,  # Clean human-readable message
            metadata=research_metadata,
        )

    def _convert_metadata_to_research_metadata(
        self,
        event_metadata: Dict,
        *,
        default_phase: Optional[str] = None,
        default_agent: Optional[str] = None,
    ) -> ResearchMetadata:
        """Normalize agent metadata payload into ResearchMetadata."""
        if not isinstance(event_metadata, dict):
            return ResearchMetadata()

        # Support both camelCase and snake_case keys
        plan_details = (
            event_metadata.get("planDetails")
            or event_metadata.get("plan_details")
            or event_metadata.get("plan")
        )

        return ResearchMetadata(
            search_queries=event_metadata.get("searchQueries")
            or event_metadata.get("search_queries")
            or [],
            sources=event_metadata.get("sources", []),
            research_iterations=event_metadata.get("researchIterations")
            or event_metadata.get("research_iterations")
            or 0,
            confidence_score=event_metadata.get("confidenceScore")
            or event_metadata.get("confidence_score"),
            reasoning_steps=event_metadata.get("reasoningSteps")
            or event_metadata.get("reasoning_steps")
            or [],
            total_sources_found=event_metadata.get("totalSourcesFound")
            or event_metadata.get("total_sources_found")
            or len(event_metadata.get("sources", [])),
            phase=event_metadata.get("phase") or default_phase or "processing",
            progress_percentage=event_metadata.get("progressPercentage")
            or event_metadata.get("progress_percentage")
            or 0.0,
            elapsed_time=event_metadata.get("elapsedTime")
            or event_metadata.get("elapsed_time")
            or 0.0,
            current_node=event_metadata.get("currentNode")
            or event_metadata.get("current_node")
            or "",
            vector_results_count=event_metadata.get("vectorResultsCount")
            or event_metadata.get("vector_results_count")
            or 0,
            plan_details=plan_details,
            factuality_score=event_metadata.get("factualityScore")
            or event_metadata.get("factuality_score"),
            report_style=event_metadata.get("reportStyle")
            or event_metadata.get("report_style"),
            verification_level=event_metadata.get("verificationLevel")
            or event_metadata.get("verification_level"),
            grounding=event_metadata.get("grounding"),
            current_agent=event_metadata.get("currentAgent")
            or event_metadata.get("current_agent")
            or default_agent,
        )

    async def send_simple_message(self, messages: List[AgentMessage], config: Optional[Dict] = None) -> Dict:
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
            "metadata": final_metadata.dict() if final_metadata else None,
        }

    def _log_headers(self, context: str, headers: dict) -> None:
        """Log headers for debugging (with sensitive data masked)."""
        safe_headers = {}
        for key, value in headers.items():
            if key.lower() in ["authorization", "x-api-key", "cookie"]:
                # Mask sensitive headers
                safe_headers[key] = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
            else:
                safe_headers[key] = value

        logger.info(f"{context} - Headers: {safe_headers}")

    def _log_request_details(self, method: str, url: str, data: dict, headers: dict) -> None:
        """Log detailed request information for debugging."""
        logger.info(f"🚀 HTTP REQUEST: {method} {url}")
        logger.info(f"📤 Request payload keys: {list(data.keys())}")
        logger.info(f"📤 Request payload size: {len(str(data))} chars")

        # Log first few input messages for context (without full content)
        if "input" in data and isinstance(data["input"], list):
            input_messages = data["input"]
            logger.info(f"📤 Input messages: {len(input_messages)} messages")
            for i, msg in enumerate(input_messages[:2]):  # Log first 2 messages
                content_preview = (
                    msg.get("content", "")[:100] + "..."
                    if len(msg.get("content", "")) > 100
                    else msg.get("content", "")
                )
                logger.info(f"   Message {i+1}: {msg.get('role', 'unknown')} - '{content_preview}'")

        # Log all headers
        self._log_headers("Request", headers)

    def _log_response_details(self, response) -> None:
        """Log detailed response information for debugging."""
        logger.info(f"📥 HTTP RESPONSE: {response.status_code} {response.reason_phrase}")
        logger.info(f"📥 Response headers: {dict(response.headers)}")

        # Check for specific headers that indicate endpoint status
        if "x-served-by" in response.headers:
            logger.info(f"📥 Served by: {response.headers['x-served-by']}")

        if "x-request-id" in response.headers:
            logger.info(f"📥 Request ID: {response.headers['x-request-id']}")

        # Check for rate limiting headers
        rate_limit_headers = ["x-ratelimit-remaining", "x-ratelimit-reset", "x-ratelimit-limit"]
        for header in rate_limit_headers:
            if header in response.headers:
                logger.info(f"📥 Rate limit - {header}: {response.headers[header]}")

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()
