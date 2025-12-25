"""Agent endpoint (Databricks-compatible)."""

from uuid import uuid4

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from src.middleware.auth import CurrentUser
from src.schemas.agent import AgentQueryRequest, AgentQueryResponse

router = APIRouter()


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
    request: AgentQueryRequest,  # noqa: ARG001 - used when fully implemented
    user: CurrentUser,  # noqa: ARG001 - used when fully implemented
    accept: str | None = Header(None),
) -> AgentQueryResponse | StreamingResponse:
    """Submit a research query (Databricks agent pattern).

    Accepts a query with optional conversation context and returns a research
    response. Supports streaming via SSE when Accept header includes
    text/event-stream.
    """
    # Check if streaming is requested
    if accept and "text/event-stream" in accept:
        # Return SSE stream
        async def generate():
            # TODO: Implement streaming with run_research()
            yield "data: {'event_type': 'agent_started', 'agent': 'coordinator'}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming response
    # TODO: Implement with run_research()
    session_id = uuid4()

    return AgentQueryResponse(
        response="Research functionality coming soon...",
        sources=[],
        query_classification=None,
        research_session_id=session_id,
    )
