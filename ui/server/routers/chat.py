"""Chat router for Deep Research Agent communication."""

import json
import logging
import traceback
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.services.agent_client import AgentClient, AgentMessage, ResearchMetadata
from server.services.user_service import UserService


router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Chat request from frontend."""
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]
    config: Optional[Dict] = None   # Research configuration


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
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize agent client: {str(e2)}"
            )


@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest, 
    agent_client: AgentClient = Depends(get_agent_client)
):
    """Send message to agent (non-streaming response)."""
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
        
        return ChatResponse(
            message=response["message"],
            metadata=response["metadata"]
        )
    
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
    agent_client: AgentClient = Depends(get_agent_client)
):
    """Stream message to agent (real-time responses)."""
    
    async def event_stream():
        try:
            # Validate request
            if not request.messages or len(request.messages) == 0:
                error_data = {"type": "error", "error": "No messages provided"}
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # Convert to agent message format
            messages = []
            for msg in request.messages:
                if "role" not in msg or "content" not in msg:
                    error_data = {"type": "error", "error": "Invalid message format"}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                messages.append(AgentMessage(role=msg["role"], content=msg["content"]))
            
            # Start streaming
            yield f"data: {json.dumps({'type': 'stream_start'})}\n\n"
            
            # Stream from agent
            async for event in agent_client.send_message(messages, request.config):
                # Convert to SSE format
                event_data = {
                    "type": event.type,
                    "content": event.content,
                    "metadata": event.metadata.dict() if event.metadata else None
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            
            # End stream
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream message failed: {str(e)}")
            logger.error(f"Stream request traceback: {traceback.format_exc()}")
            error_data = {
                "type": "error", 
                "error": str(e),
                "error_type": "server_error"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        
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
        }
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
            {"id": "standard", "name": "Standard Research", "description": "Comprehensive analysis with sources"},
            {"id": "deep", "name": "Deep Research", "description": "Extensive multi-iteration research"}
        ]
    }