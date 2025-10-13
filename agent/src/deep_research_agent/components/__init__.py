"""
Components module for the research agent.

This module provides shared components for message conversion, content extraction,
tool management, and response building.
"""


from .tool_manager import (
    ToolFactory,
    TavilyToolFactory,
    VectorSearchToolFactory,
    UCFunctionToolFactory,
    ToolRegistry,
    create_tool_registry
)

__version__ = "1.0.0"

__all__ = [
    
    # Tool Manager
    "ToolFactory",
    "TavilyToolFactory",
    "VectorSearchToolFactory", 
    "UCFunctionToolFactory",
    "ToolRegistry",
    "create_tool_registry",
]