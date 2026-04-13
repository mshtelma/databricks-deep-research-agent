"""API v1 router."""

from fastapi import APIRouter

from deep_research.api.v1 import (
    agent,
    chats,
    citations,
    config,
    custom_agents,
    data_sources,
    debug,
    discovery,
    files,
    health,
    jobs,
    messages,
    preferences,
    research,
    templates,
    user,
)

router = APIRouter()

# Include sub-routers
router.include_router(health.router, tags=["Health"])
router.include_router(agent.router, prefix="/agent", tags=["Agent"])
router.include_router(chats.router, prefix="/chats", tags=["Chats"])
router.include_router(messages.router, tags=["Messages"])
# Research routes are mounted under /chats to match frontend expectations
router.include_router(research.router, prefix="/chats", tags=["Research"])
router.include_router(preferences.router, prefix="/preferences", tags=["Preferences"])
# Citation verification routes
router.include_router(citations.router, tags=["Citations"])
# Background job management routes
router.include_router(jobs.router, tags=["Jobs"])
# Debug endpoints for troubleshooting auth (non-production only)
from deep_research.core.config import get_settings as _get_settings  # noqa: E402

if not _get_settings().is_production:
    router.include_router(debug.router, tags=["Debug"])
# User profile routes
router.include_router(user.router, tags=["User"])
# Data source management routes (007-enterprise-data-sources)
router.include_router(data_sources.router, tags=["Data Sources"])
# Data source discovery routes (US9a - auto-discover available sources)
router.include_router(discovery.router, tags=["Discovery"])
# Prompt template management routes (US5)
router.include_router(templates.router, tags=["Templates"])
# Custom agent management routes (US6)
router.include_router(custom_agents.router, tags=["Custom Agents"])
# File upload management routes (US7)
router.include_router(files.router, tags=["Files"])
# Configuration catalog routes (009-custom-agent-config)
router.include_router(config.router, tags=["Config"])
