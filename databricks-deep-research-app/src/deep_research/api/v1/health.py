"""Health check endpoint."""

import logging

from fastapi import APIRouter

from deep_research.schemas.common import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring and load balancer health checks.

    Note: Database check removed to allow health checks to pass even if DB is down.
    Database connectivity is verified during Alembic migrations at startup.
    """
    logger.info("Health check called")
    return HealthResponse(
        status="healthy",
        database="not_checked",
        version="1.0.0",
    )
