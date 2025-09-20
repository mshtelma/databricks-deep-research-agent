"""
Base classes for search tools with centralized secret resolution.

This module provides abstract base classes that handle common functionality
like API key resolution, validation, and initialization for all search tools.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .logging import get_logger
from .utils import resolve_secret
from .exceptions import ToolInitializationError

logger = get_logger(__name__)


class BaseSearchInput(BaseModel):
    """Base input schema for search tools."""
    query: str = Field(description="Search query to execute", min_length=1)
    max_results: int = Field(default=5, description="Maximum results to return", ge=1, le=20)


class BaseSearchTool(BaseTool, ABC):
    """
    Abstract base class for search tools with centralized secret resolution.
    
    This class provides common functionality for all search tools:
    - Automatic API key resolution using the centralized SecretResolver
    - Input validation
    - Error handling
    - Consistent initialization patterns
    """
    
    args_schema: type[BaseModel] = BaseSearchInput
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    test_mode: bool = Field(default=False)
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        test_mode: bool = False,
        **kwargs
    ):
        """
        Initialize the search tool with automatic secret resolution.
        
        Args:
            api_key: API key for the search service (can be a secret reference)
            base_url: Base URL for the search service
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retries
            test_mode: If True, skip API key validation for testing
            **kwargs: Additional keyword arguments
        """
        # Set defaults if not provided
        if api_key is None:
            api_key = self._get_default_api_key()
        if base_url is None:
            base_url = self._get_default_base_url()
        
        # Resolve secret if needed
        resolved_api_key = resolve_secret(api_key) if api_key else ""
        
        # Initialize with resolved values
        super().__init__(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            test_mode=test_mode,
            **kwargs
        )
        
        # Validate the resolved API key (skip in test mode)
        if not test_mode:
            self._validate_api_key()
        
        logger.info(
            f"Initialized {self.__class__.__name__}",
            has_api_key=bool(self.api_key and not self.api_key.startswith("{{secrets/")),
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            test_mode=test_mode
        )
    
    @abstractmethod
    def _get_default_api_key(self) -> str:
        """Get the default API key (usually from environment variable)."""
        pass
    
    @abstractmethod
    def _get_default_base_url(self) -> str:
        """Get the default base URL for the search service."""
        pass
    
    @abstractmethod
    def _get_service_name(self) -> str:
        """Get the service name for error messages."""
        pass
    
    def _validate_api_key(self):
        """Validate that the API key is properly resolved and available."""
        if not self.api_key:
            raise ToolInitializationError(
                f"{self._get_service_name()} API key is required but not provided. "
                f"Set the appropriate environment variable or configure the secret."
            )
        
        if self.api_key.startswith("{{secrets/"):
            raise ToolInitializationError(
                f"{self._get_service_name()} API key could not be resolved from secret: {self.api_key}. "
                f"Ensure the secret is available or set the appropriate environment variable."
            )
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Execute search query with input validation."""
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if max_results < 1 or max_results > 20:
            raise ValueError("max_results must be between 1 and 20")
        
        # Call the implementation-specific search method
        return self.search(query.strip(), max_results)
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform the actual search operation.
        
        Args:
            query: Cleaned search query string
            max_results: Number of results to return
            
        Returns:
            List of search results
        """
        pass
    
    def execute(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute method for consistency with researcher expectations.
        
        This wraps the search method to provide a uniform interface
        across all search tools, ensuring compatibility with agents
        that expect an execute() method.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (max_results, etc.)
            
        Returns:
            List of search results
        """
        max_results = kwargs.get('max_results', 5)
        return self.search(query, max_results)


class BraveSearchInput(BaseSearchInput):
    """Input schema for Brave search."""
    pass


class TavilySearchInput(BaseSearchInput):
    """Input schema for Tavily search."""
    pass