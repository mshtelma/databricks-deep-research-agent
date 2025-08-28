"""
Refactored Tavily web search tool using the new search provider interface.

This module implements the Tavily search provider using the unified search provider
interface with proper error handling, rate limiting, and result normalization.
"""

import requests
from typing import Dict, List, Any, Optional
import logging

from deep_research_agent.core.search_provider import SearchProvider, SearchProviderType, SearchResult
from deep_research_agent.core.error_handler import retry
from deep_research_agent.core.unified_config import get_config_value

logger = logging.getLogger(__name__)


class TavilySearchProvider(SearchProvider):
    """Tavily search provider implementation using unified interface."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        # Get configuration values with fallbacks
        api_key = api_key or get_config_value("tools.tavily_search.api_key")
        base_url = base_url or get_config_value(
            "tools.tavily_search.base_url", 
            "https://api.tavily.com"
        )
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            provider_type=SearchProviderType.TAVILY,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds
        )
        
        # Validate API key
        if not self.api_key or self.api_key.startswith("{{secrets/"):
            raise ValueError("Tavily Search API key is required and must be configured")
    
    @retry("tavily_search", "search")
    async def search_async(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform async search using Tavily Search API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        if not query.strip():
            logger.warning("Empty query provided to Tavily search")
            return []
        
        url = f"{self.base_url}/search"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Tavily search parameters
        payload = {
            "api_key": self.api_key,
            "query": query.strip(),
            "search_depth": kwargs.get("search_depth", "basic"),  # "basic" or "advanced"
            "include_images": kwargs.get("include_images", False),
            "include_answer": kwargs.get("include_answer", False),
            "include_raw_content": kwargs.get("include_raw_content", False),
            "max_results": min(max_results, 20),  # Tavily API practical limit
            "include_domains": kwargs.get("include_domains", []),
            "exclude_domains": kwargs.get("exclude_domains", [])
        }
        
        # Remove empty lists and None values
        payload = {k: v for k, v in payload.items() if v is not None and (not isinstance(v, list) or v)}
        
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Extract results
                    results = data.get("results", [])
                    
                    if not results:
                        logger.info(f"No results found for query: {query}")
                        return []
                    
                    # Normalize results
                    normalized_results = self.normalize_results(results)
                    
                    logger.info(
                        f"Tavily search completed successfully",
                        extra={
                            "query": query,
                            "results_count": len(normalized_results),
                            "raw_count": len(results),
                            "search_depth": payload.get("search_depth", "basic")
                        }
                    )
                    
                    return normalized_results
        
        except ImportError:
            # Fallback to synchronous requests
            import requests
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                logger.info(f"No results found for query: {query}")
                return []
            
            normalized_results = self.normalize_results(results)
            
            logger.info(
                f"Tavily search completed successfully (sync)",
                extra={
                    "query": query,
                    "results_count": len(normalized_results),
                    "raw_count": len(results)
                }
            )
            
            return normalized_results
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Tavily search rate limited for query: {query}")
                raise Exception("Rate limit exceeded") from e
            elif e.response.status_code == 401:
                logger.error("Tavily search authentication failed - check API key")
                raise Exception("Authentication failed") from e
            elif e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("detail", "Bad request")
                    logger.error(f"Tavily search bad request: {error_msg}")
                    raise Exception(f"Bad request: {error_msg}") from e
                except:
                    logger.error("Tavily search bad request - invalid parameters")
                    raise Exception("Bad request - invalid parameters") from e
            else:
                logger.error(f"Tavily search HTTP error: {e}")
                raise Exception(f"HTTP error: {e.response.status_code}") from e
        
        except requests.exceptions.Timeout:
            logger.error(f"Tavily search timed out for query: {query}")
            raise Exception("Search request timed out")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Tavily search connection error: {e}")
            raise Exception("Connection error to Tavily API")
        
        except Exception as e:
            logger.error(f"Tavily search failed for query: {query}, error: {e}")
            raise
    
    def _normalize_result(self, result: Dict[str, Any]) -> SearchResult:
        """
        Normalize Tavily search result to standard format.
        
        Args:
            result: Raw result from Tavily API
            
        Returns:
            Normalized SearchResult object
        """
        if not isinstance(result, dict):
            return None
        
        # Extract basic fields
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        
        # Skip results without essential fields
        if not title or not url:
            return None
        
        # Use content if available, otherwise fall back to title
        if not content:
            content = title
        
        # Extract additional metadata
        metadata = {
            "raw_content": result.get("raw_content"),
            "published_date": result.get("published_date"),
        }
        
        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Extract published date
        published_date = result.get("published_date")
        
        # Extract relevance score (Tavily provides scores)
        score = float(result.get("score", 0.5))
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        return SearchResult(
            title=title,
            url=url,
            content=content,
            score=score,
            published_date=published_date,
            source="tavily",
            metadata=metadata
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider-specific information."""
        return {
            "name": "Tavily Search",
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "supports_async": True,
            "supports_search_depth": True,
            "supports_domain_filtering": True,
            "supports_images": True,
            "supports_answers": True,
            "max_results_limit": 20,
            "rate_limit_info": "1000 requests per month (free tier)",
            "search_depth_options": ["basic", "advanced"],
            "supported_features": [
                "domain_include", 
                "domain_exclude", 
                "images", 
                "answers", 
                "raw_content"
            ]
        }


# Factory function for easy instantiation
def create_tavily_search_provider(
    api_key: Optional[str] = None,
    **kwargs
) -> TavilySearchProvider:
    """
    Create a Tavily search provider instance.
    
    Args:
        api_key: Tavily Search API key (optional, will use config if not provided)
        **kwargs: Additional configuration options
        
    Returns:
        TavilySearchProvider instance
    """
    return TavilySearchProvider(api_key=api_key, **kwargs)


# Legacy compatibility wrapper
class TavilySearchTool:
    """Legacy compatibility wrapper for the old TavilySearchTool interface."""
    
    def __init__(self, **kwargs):
        self.provider = create_tavily_search_provider(**kwargs)
        self.name = "tavily_search"
        self.description = "Search the web for current information using Tavily Search"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Legacy search method that returns dictionaries instead of SearchResult objects.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of result dictionaries
        """
        try:
            results = self.provider.search(query, max_results)
            return [result.to_dict() for result in results]
        except Exception as e:
            logger.error(f"Legacy Tavily search failed: {e}")
            return []