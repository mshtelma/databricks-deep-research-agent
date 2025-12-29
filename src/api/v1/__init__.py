"""API v1 router."""

from fastapi import APIRouter

from src.api.v1 import agent, chats, citations, health, messages, preferences, research

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
