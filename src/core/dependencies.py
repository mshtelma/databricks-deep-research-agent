"""FastAPI dependency injection for shared services."""

from typing import cast

from fastapi import Request

from src.agent.tools.web_crawler import WebCrawler
from src.services.llm.client import LLMClient
from src.services.llm.config import ModelConfig
from src.services.search.brave import BraveSearchClient


def get_llm_client(request: Request) -> LLMClient:
    """Get LLM client from app state."""
    return cast(LLMClient, request.app.state.llm_client)


def get_model_config(request: Request) -> ModelConfig:
    """Get model configuration from app state."""
    return cast(ModelConfig, request.app.state.model_config)


def get_brave_client(request: Request) -> BraveSearchClient:
    """Get Brave Search client from app state."""
    return cast(BraveSearchClient, request.app.state.brave_client)


def get_web_crawler(request: Request) -> WebCrawler:
    """Get web crawler from app state."""
    return cast(WebCrawler, request.app.state.web_crawler)
