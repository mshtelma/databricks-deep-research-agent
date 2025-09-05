# Agent authoring package for Databricks deployment

# Import core components that exist
from .core import (
    get_logger,
    SearchResult,
    SearchResultType,
    Citation,
    ResearchContext
)

# Define constants locally to avoid backend dependencies
UC_TOOL_NAMES = ["system.ai.python_exec"]
AGENT = None  # Will be initialized when needed

__all__ = [
    "get_logger",
    "SearchResult",
    "SearchResultType",
    "Citation",
    "ResearchContext",
    "UC_TOOL_NAMES"
]