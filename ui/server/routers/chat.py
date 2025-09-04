"""Chat router for Deep Research Agent communication."""

import json
import logging
import time
import traceback
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.services.agent_client import AgentClient, AgentMessage
from server.services.chat_logger import ChatLogger, ChatRequestContext
from server.services.user_service import UserService

router = APIRouter()
logger = logging.getLogger(__name__)
chat_logger = ChatLogger()


class ChatRequest(BaseModel):
    """Chat request from frontend."""

    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]
    config: Optional[Dict] = None  # Research configuration


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    message: Dict[str, str]
    metadata: Optional[Dict] = None


async def get_agent_client() -> AgentClient:
    """Dependency to get agent client with workspace authentication."""
    try:
        user_service = UserService()
        workspace_client = user_service.get_workspace_client()
        logger.info("Successfully created user service workspace client")
        return AgentClient(workspace_client)
    except Exception as e:
        logger.warning(f"Failed to create user service workspace client: {str(e)}")
        logger.warning("Falling back to direct AgentClient initialization")
        try:
            return AgentClient()
        except Exception as e2:
            logger.error(f"Failed to create AgentClient: {str(e2)}")
            logger.error(f"AgentClient creation traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize agent client: {str(e2)}")


@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest, 
    req: Request,
    agent_client: AgentClient = Depends(get_agent_client)
):
    """Send message to agent (non-streaming response) with comprehensive logging."""
    # Extract user information
    user_service = UserService()
    user_info = user_service.get_user_from_request(req)
    
    # Get the latest user prompt
    prompt = request.messages[-1]["content"] if request.messages else ""
    
    # Create logging context
    with ChatRequestContext(chat_logger, user_info, prompt, request.messages) as ctx:
        try:
            # Validate request
            if not request.messages or len(request.messages) == 0:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Convert to agent message format
            messages = []
            for msg in request.messages:
                if "role" not in msg or "content" not in msg:
                    raise HTTPException(status_code=400, detail="Invalid message format")
                messages.append(AgentMessage(role=msg["role"], content=msg["content"]))

            # Send to agent
            response = await agent_client.send_simple_message(messages, request.config)
            
            # Capture response for logging
            if response and "message" in response:
                ctx.response_content = response["message"].get("content", "")

            return ChatResponse(message=response["message"], metadata=response["metadata"])

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat request failed: {str(e)}")
            logger.error(f"Chat request traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")
        finally:
            await agent_client.close()


@router.post("/stream")
async def stream_message(
    request: ChatRequest,
    req: Request,
    agent_client: AgentClient = Depends(get_agent_client)
):
    """Stream message to agent (real-time responses) with comprehensive logging."""
    # Extract user information
    user_service = UserService()
    user_info = user_service.get_user_from_request(req)
    
    # Get the latest user prompt
    prompt = request.messages[-1]["content"] if request.messages else ""
    
    # Create logging context
    ctx = ChatRequestContext(chat_logger, user_info, prompt, request.messages)
    ctx.__enter__()  # Start logging

    async def event_stream():
        try:
            # Validate request
            if not request.messages or len(request.messages) == 0:
                error_data = {"type": "error", "error": "No messages provided"}
                yield f"data: {json.dumps(error_data)}\n\n"
                ctx.__exit__(ValueError, ValueError("No messages provided"), None)
                return

            # Convert to agent message format
            messages = []
            for msg in request.messages:
                if "role" not in msg or "content" not in msg:
                    error_data = {"type": "error", "error": "Invalid message format"}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    ctx.__exit__(ValueError, ValueError("Invalid message format"), None)
                    return
                messages.append(AgentMessage(role=msg["role"], content=msg["content"]))

            # Start streaming
            yield f"data: {json.dumps({'type': 'stream_start'})}\n\n"

            # Stream from agent
            async for event in agent_client.send_message(messages, request.config):
                logger.debug(
                    f"Streaming event to UI: type={event.type}, has_content={bool(event.content)}, has_metadata={bool(event.metadata)}"
                )
                
                # Track streaming events in context
                ctx.add_stream_event(
                    event.type,
                    event.content or "",
                    event.metadata.dict() if event.metadata else None
                )
                
                if event.type == "research_update":
                    logger.info(
                        f"Research update: phase={event.metadata.phase if event.metadata else 'none'}, progress={event.metadata.progress_percentage if event.metadata else 0}%"
                    )

                # Convert to SSE format
                event_data = {
                    "type": event.type,
                    "content": event.content,
                    "metadata": event.metadata.dict() if event.metadata else None,
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            # End stream
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            
            # Complete logging context successfully
            ctx.__exit__(None, None, None)

        except Exception as e:
            logger.error(f"Stream message failed: {str(e)}")
            logger.error(f"Stream request traceback: {traceback.format_exc()}")
            error_data = {"type": "error", "error": str(e), "error_type": "server_error"}
            yield f"data: {json.dumps(error_data)}\n\n"
            
            # Complete logging context with error
            ctx.__exit__(type(e), e, None)

        finally:
            await agent_client.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        },
    )


@router.get("/debug/progress-test")
async def debug_progress_test():
    """Test endpoint that emits sample progress events."""
    import asyncio

    async def generate():
        try:
            # Start streaming
            yield f"data: {json.dumps({'type': 'stream_start'})}\n\n"

            phases = [
                "QUERYING",
                "PREPARING",
                "SEARCHING",
                "SEARCHING_INTERNAL",
                "AGGREGATING",
                "ANALYZING",
                "SYNTHESIZING",
            ]
            for i, phase in enumerate(phases):
                progress = int((i + 1) * 100 / len(phases))
                # Create progress event like the agent does (without percentages in text)
                descriptions = {
                    "QUERYING": "🔍 Analyzing your query and generating search strategies",
                    "PREPARING": "📋 Preparing search execution",
                    "SEARCHING": "🌐 Searching across multiple sources",
                    "SEARCHING_INTERNAL": "🗄️ Searching internal knowledge base",
                    "AGGREGATING": "📊 Aggregating search results",
                    "ANALYZING": "🤔 Analyzing search results and extracting insights",
                    "SYNTHESIZING": "✍️ Synthesizing comprehensive response",
                }
                description = descriptions.get(phase, f"⚙️ Processing {phase.lower()}")
                delta = f"[PHASE:{phase}] {description}\n[META:progress:{progress}]\n[META:node:{phase.lower()}]\n---\n"
                event_data = {
                    "type": "response.output_text.delta",
                    "delta": delta,
                    "item_id": "debug-test-123",
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(1)

            # Add some actual content
            content_chunks = ["This is ", "a test ", "response ", "from the ", "debug endpoint."]
            for chunk in content_chunks:
                event_data = {
                    "type": "response.output_text.delta",
                    "delta": chunk,
                    "item_id": "debug-test-123",
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(0.2)

            # Done event
            done_event = {
                "type": "response.output_item.done",
                "item": {
                    "id": "debug-test-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "This is a test response from the debug endpoint."}],
                },
            }
            yield f"data: {json.dumps(done_event)}\n\n"

            # End stream
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

        except Exception as e:
            logger.error(f"Debug progress test failed: {str(e)}")
            error_data = {"type": "error", "error": str(e), "error_type": "debug_error"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        },
    )


@router.get("/health")
async def chat_health():
    """Health check for chat service."""
    return {"status": "healthy", "service": "chat"}


@router.get("/config")
async def get_chat_config():
    """Get chat configuration options."""
    return {
        "max_tokens": 4000,
        "temperature_range": [0.1, 1.0],
        "supported_models": ["default"],
        "research_modes": [
            {"id": "quick", "name": "Quick Research", "description": "Fast overview with key points"},
            {
                "id": "standard",
                "name": "Standard Research",
                "description": "Comprehensive analysis with sources",
            },
            {"id": "deep", "name": "Deep Research", "description": "Extensive multi-iteration research"},
        ],
    }
