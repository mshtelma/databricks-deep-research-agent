"""
Pure Async FastAPI Server for Deep Research Agent

This module provides a native async implementation that uses EnhancedResearchAgent
directly without any sync/async bridges, maximizing performance for Databricks Apps deployment.

Key Features:
- 100% async execution (no AsyncExecutor bridges)
- Server-Sent Events for streaming
- MLflow schema compatibility (for future migration)
- Full multi-agent workflow support
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, AsyncIterator
import json
import logging
from pathlib import Path
import time

from ..enhanced_research_agent import EnhancedResearchAgent
from ..core.multi_agent_state import StateManager
from ..core.report_styles import ReportStyle
from ..core.grounding import VerificationLevel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel as PydanticBaseModel
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# ==================== Serialization Helpers ====================

def make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format.
    Handles: Pydantic models, datetime objects, enums, custom objects
    """
    if obj is None:
        return None

    # Handle Pydantic models
    if isinstance(obj, PydanticBaseModel):
        try:
            return obj.model_dump(mode='json')
        except:
            # Fallback to dict
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}

    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Handle enums
    if isinstance(obj, Enum):
        return obj.value

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]

    # Handle primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # For unknown types, try to convert to dict or str
    if hasattr(obj, '__dict__'):
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}

    # Last resort: convert to string
    return str(obj)

# ==================== Request/Response Models ====================

class Message(BaseModel):
    """Message format compatible with MLflow schema"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ResearchRequest(BaseModel):
    """
    Research request matching Databricks Agent Framework schema
    Compatible with MLflow ResponsesAgentRequest for easy migration
    """
    input: List[Dict[str, str]] = Field(
        ...,
        description="List of messages in conversation format"
    )
    custom_inputs: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters like report_style, verification_level"
    )

class OutputContent(BaseModel):
    """Output content structure"""
    type: str = Field(default="output_text")
    text: str

class OutputMessage(BaseModel):
    """Output message structure"""
    type: str = Field(default="message")
    role: str = Field(default="assistant")
    content: List[OutputContent]
    id: str

class ResearchResponse(BaseModel):
    """
    Non-streaming response matching MLflow ResponsesAgentResponse schema
    """
    output: List[Dict[str, Any]] = Field(
        ...,
        description="List of output items (messages, function calls, etc.)"
    )
    custom_outputs: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata like factuality_score, citations, plan"
    )

# ==================== Agent Initialization ====================

# Global agent instance (singleton pattern for performance)
_agent_instance = None
_agent_lock = None

async def get_agent() -> EnhancedResearchAgent:
    """
    Lazy initialization of agent (called on first request)
    Thread-safe singleton pattern
    """
    global _agent_instance, _agent_lock

    if _agent_instance is None:
        if _agent_lock is None:
            import asyncio
            _agent_lock = asyncio.Lock()

        async with _agent_lock:
            if _agent_instance is None:
                # Auto-discover config from standard locations
                config_path = None

                search_paths = [
                    Path(__file__).parent.parent.parent.parent / "conf" / "base.yaml",
                    Path(__file__).parent.parent / "conf" / "base.yaml",
                    Path(__file__).parent.parent / "agent_config.yaml",
                ]

                for path in search_paths:
                    if path.exists():
                        config_path = str(path)
                        logger.info(f"Found config at: {config_path}")
                        break

                _agent_instance = EnhancedResearchAgent(
                    config_path=config_path
                )
                logger.info("✅ Initialized EnhancedResearchAgent for FastAPI server")

    return _agent_instance

# ==================== FastAPI Application ====================

app = FastAPI(
    title="Deep Research Agent API",
    description="Multi-agent research system with native async streaming",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on deployment environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Static File Serving ====================

def setup_static_files():
    """
    Setup static file serving for the React UI
    Looks for UI build in standard locations and mounts as static files
    """
    # Look for UI build in multiple possible locations
    possible_ui_paths = [
        Path(__file__).parent.parent.parent.parent / "ui" / "static",  # agent/ui/static (built)
        Path(__file__).parent.parent.parent.parent / "ui" / "build",   # agent/ui/build
        Path(__file__).parent.parent.parent.parent / "ui" / "client" / "build",  # Legacy
        Path(__file__).parent / "ui" / "static",  # Local fallback
    ]

    ui_dist_path = None
    for path in possible_ui_paths:
        if path.exists():
            ui_dist_path = path
            break

    if ui_dist_path and ui_dist_path.exists():
        logger.info(f"✅ Serving UI from: {ui_dist_path}")

        # Mount assets directory if it exists
        assets_path = ui_dist_path / "assets"
        if assets_path.exists():
            app.mount(
                "/assets", StaticFiles(directory=str(assets_path)), name="assets"
            )
            logger.info(f"✅ Mounted /assets from: {assets_path}")

        # Serve main UI file at root
        index_path = ui_dist_path / "index.html"
        if index_path.exists():
            @app.get("/")
            async def serve_ui():
                """Serve the React UI on root path"""
                return FileResponse(str(index_path))
            logger.info(f"✅ Serving index.html at / from: {index_path}")

        # Serve other static files (favicon, etc.)
        for static_file in ["favicon.ico", "databricks.svg", "vite.svg"]:
            static_path = ui_dist_path / static_file
            if static_path.exists():
                # Create closure to capture file path
                def make_static_handler(file_path):
                    async def serve_static_file():
                        return FileResponse(str(file_path))
                    return serve_static_file

                app.get(f"/{static_file}")(make_static_handler(static_path))
                logger.info(f"✅ Serving /{static_file}")
    else:
        logger.warning(
            f"⚠️  UI dist folder not found at any of these locations: {[str(p) for p in possible_ui_paths]}. "
            f"UI will not be served. Only API endpoints available."
        )

# Setup static files on module load
setup_static_files()

# ==================== Health & Info Endpoints ====================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    agent = await get_agent()
    return {
        "status": "healthy",
        "agent": "ready",
        "version": "2.0.0",
        "deployment": "fastapi-async",
        "multi_agent": True
    }

@app.get("/info")
async def agent_info():
    """
    Get agent configuration and capabilities
    """
    agent = await get_agent()

    return {
        "agents": [
            "Coordinator",
            "Background Investigation",
            "Planner",
            "Researcher",
            "Fact Checker",
            "Reporter"
        ],
        "features": {
            "background_investigation": agent.config.get("workflow", {}).get("enable_background_investigation", True),
            "iterative_planning": agent.config.get("planning", {}).get("enable_iterative_planning", True),
            "grounding": agent.config.get("grounding", {}).get("enabled", True),
            "reflexion": agent.config.get("reflexion", {}).get("enabled", True)
        },
        "report_styles": [
            "default",
            "professional",
            "academic",
            "technical",
            "executive_summary",
            "conversational",
            "detailed"
        ]
    }

# ==================== Research Endpoints ====================

@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """
    Non-streaming research endpoint

    Uses native async execution without any sync/async bridges.
    Returns complete research results with report, citations, and metadata.

    Request format matches MLflow ResponsesAgentRequest for compatibility.
    """
    try:
        # Extract query from input messages
        query = ""
        if request.input:
            # Find the last user message
            for msg in reversed(request.input):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

        if not query:
            raise HTTPException(
                status_code=400,
                detail="No user message found in input"
            )

        logger.info(f"Processing research request: {query[:100]}...")

        # Get agent instance
        research_agent = await get_agent()

        # Initialize state using StateManager
        initial_state = StateManager.initialize_state(
            research_topic=query,
            config=research_agent.config
        )

        # Add user message to state
        initial_state["messages"] = [HumanMessage(content=query)]

        # Apply custom inputs if provided
        if request.custom_inputs:
            if "report_style" in request.custom_inputs:
                initial_state["report_style"] = request.custom_inputs["report_style"]
            if "verification_level" in request.custom_inputs:
                initial_state["verification_level"] = request.custom_inputs["verification_level"]

        # Execute async research (NATIVE ASYNC - NO BRIDGES!)
        config = {
            "configurable": {"thread_id": f"research_{time.time()}"},
            "recursion_limit": int(
                research_agent.config.get("workflow", {}).get("recursion_limit", 500)
            )
        }

        logger.info("Starting LangGraph workflow execution...")
        final_state = await research_agent.graph.ainvoke(initial_state, config)
        logger.info("LangGraph workflow completed")

        # Extract results
        final_report = final_state.get("final_report", "No report generated")

        # Build response matching MLflow schema
        response = ResearchResponse(
            output=[{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_report}],
                "id": f"msg_{time.time()}"
            }],
            custom_outputs={
                "factuality_score": final_state.get("factuality_score"),
                "confidence_score": final_state.get("confidence_score"),
                "citations": final_state.get("citations", [])[:20],  # Limit for response size
                "plan": {
                    "total_steps": len(final_state.get("completed_steps", [])),
                    "quality": final_state.get("current_plan", {}).get("quality") if isinstance(final_state.get("current_plan"), dict) else None
                },
                "research_iterations": final_state.get("research_loops", 0),
                "fact_check_iterations": final_state.get("fact_check_loops", 0)
            }
        )

        logger.info("Research request completed successfully")
        return response

    except Exception as e:
        logger.error(f"Research error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}"
        )


@app.post("/research/stream")
async def research_stream_endpoint(request: ResearchRequest):
    """
    Streaming research endpoint with Server-Sent Events (SSE)

    Native async streaming - no sync bridges!
    Streams real-time updates from all 5 agents as the workflow progresses.

    Event format:
    - data: <json_event>

    Where json_event contains LangGraph state updates that can be consumed by UI.
    """

    async def event_generator() -> AsyncIterator[str]:
        """
        Generator that streams LangGraph events as Server-Sent Events
        """
        try:
            # Extract query
            query = ""
            if request.input:
                for msg in reversed(request.input):
                    if msg.get("role") == "user":
                        query = msg.get("content", "")
                        break

            if not query:
                error_event = json.dumps({
                    "type": "error",
                    "error": "No user message found in input"
                })
                yield f"data: {error_event}\n\n"
                return

            logger.info(f"Starting streaming research: {query[:100]}...")

            # Get agent instance
            research_agent = await get_agent()

            # Initialize state
            initial_state = StateManager.initialize_state(
                research_topic=query,
                config=research_agent.config
            )
            initial_state["messages"] = [HumanMessage(content=query)]

            # Apply custom inputs
            if request.custom_inputs:
                if "report_style" in request.custom_inputs:
                    initial_state["report_style"] = request.custom_inputs["report_style"]
                if "verification_level" in request.custom_inputs:
                    initial_state["verification_level"] = request.custom_inputs["verification_level"]

            # Stream events from graph (NATIVE ASYNC - NO BRIDGES!)
            config = {
                "configurable": {"thread_id": f"stream_{time.time()}"},
                "recursion_limit": int(
                    research_agent.config.get("workflow", {}).get("recursion_limit", 500)
                )
            }

            logger.info("Starting LangGraph streaming workflow...")

            # Track what we've sent to prevent duplicates
            sent_report = False
            sent_event_ids = set()  # Track intermediate event IDs to prevent duplicates
            final_state = {}  # Accumulate complete state for fallback

            # This is the key - direct async streaming without any sync bridge
            async for event in research_agent.graph.astream(initial_state, config):
                # Convert to SSE format
                # Event is a dict like: {"node_name": {...state_updates...}}
                try:
                    for node_name, state_update in event.items():
                        if isinstance(state_update, dict):
                            # Update accumulated state for final fallback
                            final_state.update(state_update)

                            # ✅ CRITICAL FIX: Check for final_report in EVERY state update
                            if not sent_report and 'final_report' in state_update:
                                report_content = state_update['final_report']

                                # Validate report before sending
                                if report_content and isinstance(report_content, str) and len(report_content.strip()) > 0:
                                    logger.info(f"📝 Found final_report in {node_name} update, length: {len(report_content)}")

                                    # Send as MLflow output_text.delta (text chunk)
                                    item_id = f"msg_{time.time()}"
                                    delta_event = {
                                        "type": "response.output_text.delta",
                                        "delta": report_content,
                                        "item_id": item_id
                                    }
                                    yield f"data: {json.dumps(delta_event)}\n\n"

                                    # Send MLflow output_item.done (completion signal)
                                    done_event = {
                                        "type": "response.output_item.done",
                                        "item": {
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [{"type": "output_text", "text": report_content}],
                                            "id": item_id
                                        }
                                    }
                                    yield f"data: {json.dumps(done_event)}\n\n"

                                    sent_report = True
                                    logger.info("✅ Successfully sent final report via MLflow schema events")

                            # ✅ IMPROVED: Send intermediate events with deduplication
                            if 'intermediate_events' in state_update:
                                intermediate_events = state_update['intermediate_events']
                                # Send each intermediate event individually wrapped in ResponsesAgent format
                                for ie in intermediate_events:
                                    # Use event ID for deduplication (or create one from content hash)
                                    event_id = ie.get('id') or ie.get('event_id') or str(hash(json.dumps(ie, sort_keys=True)))

                                    # Only send if we haven't sent this event before
                                    if event_id not in sent_event_ids:
                                        sent_event_ids.add(event_id)
                                        # Convert to JSON-serializable format
                                        serializable_ie = make_json_serializable(ie)
                                        # Wrap in ResponsesAgentStreamEvent format expected by UI
                                        wrapped_event = {
                                            "type": "intermediate_event",
                                            "intermediate_event": serializable_ie
                                        }
                                        event_data = json.dumps(wrapped_event)
                                        yield f"data: {event_data}\n\n"

                            # Also send node completion as an event (for debugging)
                            node_event = {
                                'event_type': 'node_complete',
                                'node': node_name,
                                'timestamp': time.time()
                            }
                            event_data = json.dumps(node_event)
                            yield f"data: {event_data}\n\n"

                except Exception as e:
                    logger.warning(f"Could not serialize event: {e}", exc_info=True)
                    continue

            # ✅ FALLBACK: If no report sent during stream, try final_state
            if not sent_report:
                report_content = final_state.get("final_report")
                if report_content and isinstance(report_content, str) and len(report_content.strip()) > 0:
                    logger.warning("⚠️ Report not sent during stream, sending from final_state")
                    item_id = f"msg_{time.time()}"

                    delta_event = {
                        "type": "response.output_text.delta",
                        "delta": report_content,
                        "item_id": item_id
                    }
                    yield f"data: {json.dumps(delta_event)}\n\n"

                    done_event = {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": report_content}],
                            "id": item_id
                        }
                    }
                    yield f"data: {json.dumps(done_event)}\n\n"
                    sent_report = True
                else:
                    # No report generated - send error
                    logger.error("❌ Workflow completed but no final_report found in state")
                    error_event = {
                        "type": "error",
                        "error": "Research completed but no report was generated",
                        "error_type": "MissingReportError"
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

            # Send completion event
            completion_event = json.dumps({
                "type": "workflow_complete",
                "timestamp": time.time(),
                "report_sent": sent_report
            })
            yield f"data: {completion_event}\n\n"

            logger.info(f"Streaming research completed successfully, report_sent={sent_report}")

        except Exception as e:
            logger.error(f"Stream error: {type(e).__name__}: {str(e)}", exc_info=True)
            error_event = json.dumps({
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


# ==================== Legacy MLflow Compatibility Endpoints ====================

@app.post("/invocations")
async def mlflow_invocations_endpoint(request: Request):
    """
    MLflow-compatible invocations endpoint with streaming support

    This provides compatibility with MLflow serving API for easy migration.
    Supports both streaming and non-streaming modes.
    """
    try:
        body = await request.json()

        # Check if streaming is requested in the raw body
        is_streaming = body.get("stream", False)

        # Convert MLflow request to our format
        research_request = ResearchRequest(**body)

        # Route to appropriate endpoint
        if is_streaming:
            # Use streaming endpoint
            return await research_stream_endpoint(research_request)
        else:
            # Use non-streaming endpoint
            result = await research_endpoint(research_request)
            return result

    except Exception as e:
        logger.error(f"MLflow invocations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Application Lifecycle ====================

@app.on_event("startup")
async def startup_event():
    """
    Initialize agent on startup
    """
    logger.info("=" * 60)
    logger.info("🚀 Deep Research Agent FastAPI Server Starting")
    logger.info("=" * 60)
    logger.info("Deployment Type: FastAPI Native Async")
    logger.info("Agent Architecture: Multi-Agent (5 agents)")
    logger.info("Streaming: Server-Sent Events (SSE)")
    logger.info("MLflow Compatibility: Yes (for future migration)")
    logger.info("=" * 60)

    # Pre-initialize agent to avoid first request latency
    try:
        await get_agent()
        logger.info("✅ Agent pre-initialized successfully")
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {e}")
        # Don't raise - let it retry on first request


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    logger.info("Shutting down Deep Research Agent FastAPI Server...")
    # Cleanup if needed
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn

    # For development/testing
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
