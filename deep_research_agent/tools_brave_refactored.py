"""
Refactored Brave web search tool using the new search provider interface.

This module implements the Brave search provider using the unified search provider
interface with proper error handling, rate limiting, and result normalization.
"""

import requests
from typing import Dict, List, Any, Optional
import logging

from deep_research_agent.core.search_provider import SearchProvider, SearchProviderType, SearchResult
from deep_research_agent.core.error_handler import retry
from deep_research_agent.core.unified_config import get_config_value

logger = logging.getLogger(__name__)


class BraveSearchProvider(SearchProvider):
    """Brave search provider implementation using unified interface."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        # Get configuration values with fallbacks
        api_key = api_key or get_config_value("tools.brave_search.api_key")
        base_url = base_url or get_config_value(
            "tools.brave_search.base_url", 
            "https://api.search.brave.com/res/v1"
        )
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            provider_type=SearchProviderType.BRAVE,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds
        )
        
        # Validate API key
        if not self.api_key or self.api_key.startswith("{{secrets/"):
            raise ValueError("Brave Search API key is required and must be configured")
    
    @retry("brave_search", "search")
    async def search_async(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform async search using Brave Search API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        if not query.strip():
            logger.warning("Empty query provided to Brave search")
            return []
        
        url = f"{self.base_url}/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query.strip(),
            "count": min(max_results, 20),  # Brave API limit
            "search_lang": kwargs.get("search_lang", "en"),
            "country": kwargs.get("country", "us"),
            "safesearch": kwargs.get("safesearch", "moderate"),
            "freshness": kwargs.get("freshness"),  # "pd" for past day, "pw" for past week
            "text_decorations": kwargs.get("text_decorations", False),
            "spellcheck": kwargs.get("spellcheck", True)
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Extract web results
                    web_results = data.get("web", {}).get("results", [])
                    
                    if not web_results:
                        logger.info(f"No results found for query: {query}")
                        return []
                    
                    # Normalize results
                    normalized_results = self.normalize_results(web_results)
                    
                    logger.info(
                        f"Brave search completed successfully",
                        extra={
                            "query": query,
                            "results_count": len(normalized_results),
                            "raw_count": len(web_results)
                        }
                    )
                    
                    return normalized_results
        
        except ImportError:
            # Fallback to synchronous requests
            import requests
            response = requests.get(
                url, 
                headers=headers, 
                params=params, 
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
            
            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                logger.info(f"No results found for query: {query}")
                return []
            
            normalized_results = self.normalize_results(web_results)
            
            logger.info(
                f"Brave search completed successfully (sync)",
                extra={
                    "query": query,
                    "results_count": len(normalized_results),
                    "raw_count": len(web_results)
                }
            )
            
            return normalized_results
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Brave search rate limited for query: {query}")
                raise Exception("Rate limit exceeded") from e
            elif e.response.status_code == 401:
                logger.error("Brave search authentication failed - check API key")
                raise Exception("Authentication failed") from e
            elif e.response.status_code == 403:
                logger.error("Brave search access forbidden - check API key permissions")
                raise Exception("Access forbidden") from e
            else:
                logger.error(f"Brave search HTTP error: {e}")
                raise Exception(f"HTTP error: {e.response.status_code}") from e
        
        except requests.exceptions.Timeout:
            logger.error(f"Brave search timed out for query: {query}")
            raise Exception("Search request timed out")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Brave search connection error: {e}")
            raise Exception("Connection error to Brave API")
        
        except Exception as e:
            logger.error(f"Brave search failed for query: {query}, error: {e}")
            raise
    
    def _normalize_result(self, result: Dict[str, Any]) -> SearchResult:
        """
        Normalize Brave search result to standard format.
        
        Args:
            result: Raw result from Brave API
            
        Returns:
            Normalized SearchResult object
        """
        if not isinstance(result, dict):
            return None
        
        # Extract basic fields
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        description = result.get("description", "").strip()
        
        # Skip results without essential fields
        if not title or not url:
            return None
        
        # Extract additional metadata
        metadata = {
            "age": result.get("age"),
            "language": result.get("language"),
            "family_friendly": result.get("family_friendly", True),
            "type": result.get("type", "web"),
            "subtype": result.get("subtype"),
            "deep_results": result.get("deep_results", {}),
            "thumbnails": result.get("thumbnail", {}),
            "extra_snippets": result.get("extra_snippets", [])
        }
        
        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Extract published date if available
        published_date = None
        if "age" in result and result["age"]:
            # Brave sometimes provides relative dates like "2 days ago"
            published_date = result["age"]
        
        # Calculate relevance score (Brave doesn't provide explicit scores)
        # Use position-based scoring with title/description quality factors
        score = 0.8  # Base score
        
        # Adjust based on result quality indicators
        if len(description) > 100:
            score += 0.1  # Longer descriptions often indicate better content
        
        if any(keyword in title.lower() for keyword in ["official", "wikipedia", "news"]):
            score += 0.1  # Authoritative sources
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        return SearchResult(
            title=title,
            url=url,
            content=description,
            score=score,
            published_date=published_date,
            source="brave",
            metadata=metadata
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider-specific information."""
        return {
            "name": "Brave Search",
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "supports_async": True,
            "supports_freshness": True,
            "supports_safesearch": True,
            "max_results_limit": 20,
            "rate_limit_info": "20 requests per second, 2000 per month",
            "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
            "supported_countries": ["us", "ca", "gb", "au", "de", "fr", "es", "it", "jp"]
        }


# Factory function for easy instantiation
def create_brave_search_provider(
    api_key: Optional[str] = None,
    **kwargs
) -> BraveSearchProvider:
    """
    Create a Brave search provider instance.
    
    Args:
        api_key: Brave Search API key (optional, will use config if not provided)
        **kwargs: Additional configuration options
        
    Returns:
        BraveSearchProvider instance
    """
    return BraveSearchProvider(api_key=api_key, **kwargs)


# Legacy compatibility wrapper
class BraveSearchTool:
    """Legacy compatibility wrapper for the old BraveSearchTool interface."""
    
    def __init__(self, **kwargs):
        self.provider = create_brave_search_provider(**kwargs)
        self.name = "brave_search"
        self.description = "Search the web for current information using Brave Search"
    
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
            logger.error(f"Legacy Brave search failed: {e}")
            return []