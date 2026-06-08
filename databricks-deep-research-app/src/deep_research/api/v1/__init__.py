"""API v1 router."""

from fastapi import APIRouter

from deep_research.api.v1 import (
    agent,
    agent_designer,
    agents_v2,
    chats,
    citations,
    config,
    data_sources,
    debug,
    deployments,
    discovery,
    files,
    health,
    hitl,
    jobs,
    messages,
    metrics,
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
# File upload management routes (US7)
router.include_router(files.router, tags=["Files"])
# Configuration catalog routes (009-custom-agent-config)
router.include_router(config.router, tags=["Config"])
# HITL approval routes (Phase 2)
router.include_router(hitl.router, tags=["HITL"])
# Client metrics ingest endpoint (US-614)
router.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
# Agent Designer routes (US-106)
router.include_router(agent_designer.router, prefix="/agent-designer", tags=["Agent Designer"])
# AgentV2 CRUD routes (US-105)
router.include_router(agents_v2.router, prefix="/agents-v2", tags=["Agents V2"])
# Deployment routes (Phase 1 backend foundation)
router.include_router(deployments.router, prefix="/deployments", tags=["Deployments"])
