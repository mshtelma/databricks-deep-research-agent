"""Debug endpoints for troubleshooting authentication and user isolation."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from deep_research.core.config import Settings, get_settings
from deep_research.middleware.auth import CurrentUser

logger = logging.getLogger(__name__)


def _require_non_production(
    settings: Settings = Depends(get_settings),
) -> None:
    """Block debug endpoints in production at request time.

    Defense-in-depth: the router is already excluded at import time,
    but this guard catches settings drift or misconfiguration.
    """
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )


router = APIRouter(prefix="/debug", tags=["Debug"], dependencies=[Depends(_require_non_production)])


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
            # Mask sensitive token values — log presence and length only
            if "token" in name.lower() or "authorization" in name.lower():
                auth_headers[name] = f"present (length={len(value)})"
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
