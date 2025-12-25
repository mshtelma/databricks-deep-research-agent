"""Agent server utilities.

Utilities for transforming between internal events and Databricks Agent protocol.
"""

from typing import Any

from src.agent.state import SourceInfo


def extract_messages(request: dict[str, Any]) -> list[dict[str, str]]:
    """Extract messages from request.

    Args:
        request: Request dict with messages array.

    Returns:
        List of message dicts with role and content.
    """
    messages = request.get("messages", [])
    if not messages:
        # Try alternative format
        query = request.get("query") or request.get("input")
        if query:
            return [{"role": "user", "content": str(query)}]
        return []

    return [
        {
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        }
        for msg in messages
        if msg.get("content")
    ]


def extract_user_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract user context from request context.

    Args:
        context: Optional context dict.

    Returns:
        Extracted user context.
    """
    if not context:
        return {}

    return {
        "user_id": context.get("user_id"),
        "trace_id": context.get("trace_id"),
        "request_id": context.get("request_id"),
    }


def transform_event_to_databricks(event: dict[str, Any]) -> dict[str, Any]:
    """Transform internal StreamEvent to Databricks Agent protocol.

    Args:
        event: Internal event dict.

    Returns:
        Databricks-compatible event dict.
    """
    event_type = event.get("type", "unknown")

    # Map internal event types to Databricks protocol
    if event_type == "content":
        return {
            "type": "text",
            "content": event.get("data", {}).get("content", ""),
        }

    elif event_type == "agent_start":
        return {
            "type": "status",
            "status": "agent_started",
            "agent": event.get("data", {}).get("agent"),
            "message": event.get("data", {}).get("message"),
        }

    elif event_type == "agent_complete":
        return {
            "type": "status",
            "status": "agent_completed",
            "agent": event.get("data", {}).get("agent"),
            "message": event.get("data", {}).get("message"),
        }

    elif event_type == "step_start":
        return {
            "type": "step",
            "status": "started",
            "step_index": event.get("data", {}).get("step_index"),
            "step_title": event.get("data", {}).get("title"),
        }

    elif event_type == "step_complete":
        return {
            "type": "step",
            "status": "completed",
            "step_index": event.get("data", {}).get("step_index"),
            "observation": event.get("data", {}).get("observation"),
        }

    elif event_type == "plan_created":
        plan_data = event.get("data", {}).get("plan", {})
        return {
            "type": "plan",
            "plan": {
                "title": plan_data.get("title"),
                "reasoning": plan_data.get("reasoning"),
                "steps": [
                    {
                        "index": s.get("index"),
                        "title": s.get("title"),
                        "description": s.get("description"),
                    }
                    for s in plan_data.get("steps", [])
                ],
            },
        }

    elif event_type == "reflection":
        return {
            "type": "reflection",
            "decision": event.get("data", {}).get("decision"),
            "reasoning": event.get("data", {}).get("reasoning"),
            "suggested_action": event.get("data", {}).get("suggested_action"),
        }

    elif event_type == "search_results":
        return {
            "type": "search",
            "query": event.get("data", {}).get("query"),
            "results_count": event.get("data", {}).get("count", 0),
        }

    elif event_type == "sources_added":
        sources = event.get("data", {}).get("sources", [])
        return {
            "type": "sources",
            "sources": [
                {
                    "url": s.get("url"),
                    "title": s.get("title"),
                    "snippet": s.get("snippet"),
                }
                for s in sources
            ],
        }

    elif event_type == "error":
        return {
            "type": "error",
            "content": event.get("data", {}).get("message", "Unknown error"),
            "error_type": event.get("data", {}).get("error_type"),
        }

    elif event_type == "complete":
        return {
            "type": "complete",
            "metadata": {
                "session_id": str(event.get("data", {}).get("session_id", "")),
                "research_time": event.get("data", {}).get("research_time"),
                "steps_completed": event.get("data", {}).get("steps_completed"),
            },
        }

    else:
        # Pass through unknown events
        return {
            "type": event_type,
            "data": event.get("data"),
        }


def format_sources_for_response(sources: list[SourceInfo]) -> list[dict[str, Any]]:
    """Format sources for API response.

    Args:
        sources: List of SourceInfo objects.

    Returns:
        List of source dicts.
    """
    return [
        {
            "url": s.url,
            "title": s.title,
            "snippet": s.snippet,
            "relevance_score": s.relevance_score,
        }
        for s in sources
    ]


def get_obo_token(context: dict[str, Any] | None) -> str | None:
    """Extract OBO token from context for authentication.

    Args:
        context: Request context.

    Returns:
        OBO token if present, None otherwise.
    """
    if not context:
        return None

    # Try various token locations
    token = context.get("authorization")
    if token and token.startswith("Bearer "):
        return token[7:]

    return context.get("obo_token") or context.get("token")
