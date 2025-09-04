# Generic router module for the Databricks app template
# Add your FastAPI routes here

from fastapi import APIRouter

from .chat import router as chat_router
from .debug import router as debug_router
from .test import router as test_router
from .test_table import router as test_table_router
from .user import router as user_router

router = APIRouter()
router.include_router(user_router, prefix="/user", tags=["user"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(debug_router, prefix="/debug", tags=["debug"])
router.include_router(test_table_router, tags=["test"])
router.include_router(test_router, tags=["test"])
