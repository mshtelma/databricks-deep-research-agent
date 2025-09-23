"""
Search validation utilities to detect and handle search tool failures.
"""

import os
from typing import Dict, List, Optional
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class SearchValidator:
    """Validates search tool configuration and availability."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_search_tools(self, config: Dict) -> Dict[str, bool]:
        """
        Validate that search tools are properly configured.
        
        Returns:
            Dict mapping tool names to availability status
        """
        results = {}
        search_config = config.get('search', {})
        providers = search_config.get('providers', {})
        
        # Check Brave configuration
        brave_config = providers.get('brave', {})
        if brave_config.get('enabled', False):
            brave_api_key = os.getenv('BRAVE_API_KEY')
            results['brave'] = bool(brave_api_key)
            if not brave_api_key:
                logger.error("Brave search enabled but BRAVE_API_KEY environment variable not set")
        else:
            results['brave'] = False
            logger.info("Brave search disabled in configuration")
        
        # Check Tavily configuration  
        tavily_config = providers.get('tavily', {})
        if tavily_config.get('enabled', False):
            tavily_api_key = os.getenv('TAVILY_API_KEY')
            results['tavily'] = bool(tavily_api_key)
            if not tavily_api_key:
                logger.error("Tavily search enabled but TAVILY_API_KEY environment variable not set")
        else:
            results['tavily'] = False
            logger.info("Tavily search disabled in configuration")
        
        # Check if any search tool is available
        any_available = any(results.values())
        if not any_available:
            logger.critical("NO SEARCH TOOLS AVAILABLE - This will cause research failures!")
            logger.critical("Please set BRAVE_API_KEY or TAVILY_API_KEY environment variable")
        
        self.validation_results = results
        return results
    
    def get_available_tools(self) -> List[str]:
        """Get list of available search tools."""
        return [tool for tool, available in self.validation_results.items() if available]
    
    def has_any_tools(self) -> bool:
        """Check if any search tools are available."""
        return any(self.validation_results.values())

