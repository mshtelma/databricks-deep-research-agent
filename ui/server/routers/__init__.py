# Generic router module for the Databricks app template
# Add your FastAPI routes here

from fastapi import APIRouter

from .user import router as user_router
from .chat import router as chat_router
from .debug import router as debug_router

router = APIRouter()
router.include_router(user_router, prefix='/user', tags=['user'])
router.include_router(chat_router, prefix='/chat', tags=['chat'])
router.include_router(debug_router, prefix='/debug', tags=['debug'])
