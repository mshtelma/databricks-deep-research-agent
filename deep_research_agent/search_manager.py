"""
Unified search management for the research agent.

This module provides a high-level interface for managing multiple search providers
with automatic fallback, load balancing, and intelligent provider selection.
"""

from typing import Dict, List, Any, Optional
import logging

from deep_research_agent.core.search_provider import (
    UnifiedSearchManager, RateLimiter, RateLimitConfig, SearchResult
)
from deep_research_agent.core.unified_config import get_config_manager
from deep_research_agent.core.cache_manager import global_cache_manager
from deep_research_agent.core.error_handler import safe_call

from deep_research_agent.tools_brave_refactored import create_brave_search_provider
from deep_research_agent.tools_tavily_refactored import create_tavily_search_provider

logger = logging.getLogger(__name__)


class ResearchSearchManager:
    """
    High-level search manager for the research agent.
    
    This class integrates with the agent's configuration system and provides
    intelligent search provider selection based on query characteristics and
    provider performance.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the research search manager.
        
        Args:
            config_manager: Optional config manager instance
        """
        self.config_manager = config_manager or get_config_manager()
        self.search_manager = None
        self._provider_cache = {}
        
        # Initialize search providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available search providers based on configuration."""
        providers = {}
        
        # Try to initialize Brave search provider
        try:
            brave_config = self.config_manager.get_tool_config("brave_search")
            if brave_config.enabled:
                brave_provider = create_brave_search_provider(
                    api_key=brave_config.config.get("api_key"),
                    base_url=brave_config.config.get("base_url"),
                    timeout_seconds=brave_config.timeout_seconds or 30,
                    max_retries=brave_config.max_retries or 3
                )
                providers["brave"] = brave_provider
                logger.info("Initialized Brave search provider")
            else:
                logger.info("Brave search provider disabled in configuration")
        except Exception as e:
            logger.warning(f"Failed to initialize Brave search provider: {e}")
        
        # Try to initialize Tavily search provider
        try:
            tavily_config = self.config_manager.get_tool_config("tavily_search")
            if tavily_config.enabled:
                tavily_provider = create_tavily_search_provider(
                    api_key=tavily_config.config.get("api_key"),
                    base_url=tavily_config.config.get("base_url"),
                    timeout_seconds=tavily_config.timeout_seconds or 30,
                    max_retries=tavily_config.max_retries or 3
                )
                providers["tavily"] = tavily_provider
                logger.info("Initialized Tavily search provider")
            else:
                logger.info("Tavily search provider disabled in configuration")
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily search provider: {e}")
        
        if not providers:
            logger.warning("No search providers available - searches will fail")
            return
        
        # Create rate limiter with configuration
        rate_limit_config = RateLimitConfig(
            max_concurrent=self.config_manager.get("rate_limiting.max_concurrent_searches", 2, int),
            requests_per_minute=self.config_manager.get("rate_limiting.requests_per_minute", 30, int),
            burst_allowance=self.config_manager.get("rate_limiting.burst_allowance", 5, int),
            cooldown_seconds=self.config_manager.get("rate_limiting.batch_delay_seconds", 1.0, float),
            adaptive=self.config_manager.get("rate_limiting.adaptive", True, bool)
        )
        
        rate_limiter = RateLimiter(rate_limit_config)
        
        # Create unified search manager
        self.search_manager = UnifiedSearchManager(providers, rate_limiter)
        
        # Set primary provider based on configuration
        primary_provider = self.config_manager.get("research.search_provider", "tavily", str)
        if primary_provider in providers:
            self.search_manager.set_primary_provider(primary_provider)
            logger.info(f"Set primary search provider to: {primary_provider}")
        else:
            logger.warning(f"Configured primary provider '{primary_provider}' not available")
    
    @safe_call("intelligent_search", fallback=[])
    def search(
        self, 
        query: str, 
        max_results: int = 5,
        use_cache: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform intelligent search with caching and provider selection.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results if available
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        if not self.search_manager:
            logger.error("No search providers available")
            return []
        
        if not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        # Check cache first if enabled
        if use_cache:
            cache_key = f"search_{hash(query)}_{max_results}"
            cached_results = global_cache_manager.get(cache_key, "search_results")
            if cached_results:
                logger.info(f"Returning cached search results for: {query[:50]}...")
                return cached_results
        
        # Perform search with fallback
        try:
            results = self.search_manager.search(query, max_results, **kwargs)
            
            # Cache results if successful
            if results and use_cache:
                global_cache_manager.set(
                    cache_key, 
                    results, 
                    "search_results", 
                    ttl=1800  # Cache for 30 minutes
                )
            
            logger.info(
                f"Search completed successfully",
                extra={
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "results_count": len(results),
                    "cached": False
                }
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def search_async(
        self, 
        query: str, 
        max_results: int = 5,
        use_cache: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform async intelligent search with caching and provider selection.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results if available
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        if not self.search_manager:
            logger.error("No search providers available")
            return []
        
        if not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        # Check cache first if enabled
        if use_cache:
            cache_key = f"search_{hash(query)}_{max_results}"
            cached_results = global_cache_manager.get(cache_key, "search_results")
            if cached_results:
                logger.info(f"Returning cached search results for: {query[:50]}...")
                return cached_results
        
        # Perform async search with fallback
        try:
            results = await self.search_manager.search_with_fallback(query, max_results, **kwargs)
            
            # Cache results if successful
            if results and use_cache:
                global_cache_manager.set(
                    cache_key, 
                    results, 
                    "search_results", 
                    ttl=1800  # Cache for 30 minutes
                )
            
            logger.info(
                f"Async search completed successfully",
                extra={
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "results_count": len(results),
                    "cached": False
                }
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Async search failed for query '{query}': {e}")
            return []
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all search providers."""
        if not self.search_manager:
            return {"status": "no_providers", "providers": {}}
        
        return self.search_manager.get_provider_health()
    
    def optimize_providers(self):
        """Optimize provider order based on performance."""
        if self.search_manager:
            self.search_manager.reorder_providers_by_performance()
            logger.info("Optimized search provider order based on performance")
    
    def get_best_provider(self) -> Optional[str]:
        """Get the name of the best-performing provider."""
        if not self.search_manager:
            return None
        return self.search_manager.get_best_provider()
    
    def clear_cache(self):
        """Clear search result cache."""
        global_cache_manager.clear_category("search_results")
        logger.info("Cleared search result cache")


# Global search manager instance
_global_search_manager: Optional[ResearchSearchManager] = None


def get_search_manager(config_manager=None) -> ResearchSearchManager:
    """Get or create global search manager instance."""
    global _global_search_manager
    
    if _global_search_manager is None:
        _global_search_manager = ResearchSearchManager(config_manager)
    
    return _global_search_manager


# Convenience functions
def search_web(
    query: str, 
    max_results: int = 5, 
    **kwargs
) -> List[SearchResult]:
    """
    Convenience function for web search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        List of SearchResult objects
    """
    return get_search_manager().search(query, max_results, **kwargs)


async def search_web_async(
    query: str, 
    max_results: int = 5, 
    **kwargs
) -> List[SearchResult]:
    """
    Convenience function for async web search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        List of SearchResult objects
    """
    return await get_search_manager().search_async(query, max_results, **kwargs)


def get_search_status() -> Dict[str, Any]:
    """Get search provider status."""
    return get_search_manager().get_provider_status()