"""Debug endpoints for troubleshooting authentication and user isolation."""

import logging
from typing import Any

from fastapi import APIRouter, Request

from deep_research.middleware.auth import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["Debug"])


@router.get("/me")
async def get_current_user_identity(user: CurrentUser) -> dict[str, Any]:
    """Return current authenticated user identity.

    Use this endpoint to verify which user identity is being resolved.
    If OBO is working correctly, this should return your real Databricks user.
    If falling back to Service Principal, you'll see the SP identity instead.

    Returns:
        Dictionary with user_id, email, and display_name.
    """
    logger.info(f"Debug /me called - user_id={user.user_id}, email={user.email}")
    return {
        "user_id": user.user_id,
        "email": user.email,
        "display_name": user.display_name,
    }


@router.get("/headers")
async def get_request_headers(request: Request, user: CurrentUser) -> dict[str, Any]:
    """Return relevant authentication headers for debugging.

    Shows which auth-related headers are present in the request.
    Useful for diagnosing OBO token forwarding issues.

    Note: Sensitive token values are partially masked for security.
    """
    headers = dict(request.headers)

    # Headers relevant to authentication
    auth_header_names = [
        "x-forwarded-access-token",
        "authorization",
        "x-forwarded-user",
        "x-forwarded-email",
        "x-forwarded-for",
    ]

    auth_headers: dict[str, str | None] = {}
    for name in auth_header_names:
        value = headers.get(name)
        if value:
            # Mask sensitive token values (show first/last 8 chars)
            if "token" in name.lower() or "authorization" in name.lower():
                if len(value) > 20:
                    auth_headers[name] = f"{value[:8]}...{value[-8:]}"
                else:
                    auth_headers[name] = "***masked***"
            else:
                auth_headers[name] = value
        else:
            auth_headers[name] = None

    logger.info(f"Debug /headers called - OBO token present: {auth_headers.get('x-forwarded-access-token') is not None}")

    return {
        "current_user": {
            "user_id": user.user_id,
            "email": user.email,
            "display_name": user.display_name,
        },
        "auth_headers": auth_headers,
        "obo_token_present": auth_headers.get("x-forwarded-access-token") is not None,
        "total_headers_count": len(headers),
    }
