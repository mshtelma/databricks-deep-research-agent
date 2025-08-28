"""
Components module for the research agent.

This module provides shared components for message conversion, content extraction,
tool management, and response building.
"""

from .message_converter import MessageConverter, message_converter
from .content_extractor import ContentExtractor, content_extractor
from .tool_manager import (
    ToolFactory,
    TavilyToolFactory,
    VectorSearchToolFactory,
    UCFunctionToolFactory,
    ToolRegistry,
    create_tool_registry
)
from .response_builder import ResponseBuilder, response_builder

__version__ = "1.0.0"

__all__ = [
    # Message Converter
    "MessageConverter",
    "message_converter",
    
    # Content Extractor
    "ContentExtractor", 
    "content_extractor",
    
    # Tool Manager
    "ToolFactory",
    "TavilyToolFactory",
    "VectorSearchToolFactory", 
    "UCFunctionToolFactory",
    "ToolRegistry",
    "create_tool_registry",
    
    # Response Builder
    "ResponseBuilder",
    "response_builder",
]