"""
Brave web search tool for research agent.
"""

import os
import time
import random
import requests
from typing import Dict, List, Any, Optional
from pydantic import Field

from .core.base_tools import BaseSearchTool, BraveSearchInput
from .core.logging import get_logger
from .core.global_rate_limiter import global_rate_limiter

logger = get_logger(__name__)


class BraveSearchTool(BaseSearchTool):
    """Tool for performing web searches using Brave Search API."""
    
    name: str = "brave_search"
    description: str = "Search the web for current information using Brave Search"
    args_schema: type[BraveSearchInput] = BraveSearchInput
    
    def _get_default_api_key(self) -> str:
        """Get the default API key from environment variable."""
        return os.getenv("BRAVE_API_KEY", "")
    
    def _get_default_base_url(self) -> str:
        """Get the default base URL for Brave Search API."""
        return "https://api.search.brave.com/res/v1"
    
    def _get_service_name(self) -> str:
        """Get the service name for error messages."""
        return "Brave Search"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using Brave Search API with global rate limiting coordination.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, url, content
        """
        start_time = time.time()
        
        # Log search initiation
        logger.info(f"BRAVE_SEARCH: Starting search for query: '{query[:50]}...'")
        logger.info(f"BRAVE_SEARCH: Max results requested: {max_results}")
        
        # Acquire permission from global rate limiter with timing
        logger.info("BRAVE_SEARCH: Acquiring rate limiter permission...")
        if not global_rate_limiter.acquire('brave'):
            logger.error("BRAVE_SEARCH: Rate limiter rejected request (circuit breaker open)")
            raise Exception("Brave search service is temporarily unavailable (circuit breaker open)")
        
        wait_time = time.time() - start_time
        logger.info(f"BRAVE_SEARCH: Acquired rate limit slot after {wait_time:.1f}s delay")
        
        url = f"{self.base_url}/web/search"
        
        params = {
            "q": query,
            "count": max_results,
            "search_lang": "en",
            "country": "us"
        }
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        logger.info(f"BRAVE_SEARCH: Making API request to {url}")
        logger.info(f"BRAVE_SEARCH: Request params: {params}")
        
        # Use retry mechanism
        response = self._search_with_retry(url, params, headers)
        
        # Process successful response
        return self._process_response(response, max_results, query)
    
    def _search_with_retry(self, url: str, params: dict, headers: dict, max_retries: int = 5):
        """Execute search with intelligent retry on rate limiting"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"BRAVE_SEARCH: Attempt {attempt + 1}/{max_retries}")
                response = requests.get(url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 10))
                    backoff_delay = min(retry_after * (2 ** attempt), 60)  # Max 60s
                    
                    logger.warning(f"BRAVE_SEARCH: Rate limited on attempt {attempt + 1}. Waiting {backoff_delay}s")
                    global_rate_limiter.report_failure('brave', is_rate_limit=True)
                    
                    # Update cooldown in rate limiter
                    global_rate_limiter.update_cooldown('brave', backoff_delay)
                    
                    time.sleep(backoff_delay)
                    continue
                
                # Check for other HTTP errors
                response.raise_for_status()
                
                # Success!
                global_rate_limiter.report_success('brave')
                logger.info(f"BRAVE_SEARCH: SUCCESS on attempt {attempt + 1}")
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    global_rate_limiter.report_failure('brave', is_rate_limit=False)
                    logger.error(f"BRAVE_SEARCH: FAILED after {max_retries} attempts: {e}")
                    raise Exception(f"Brave Search API request failed after {max_retries} attempts: {str(e)}")
                
                # Intermediate failure - wait and retry
                backoff_time = min(2 ** attempt, 10)  # Max 10s exponential backoff
                logger.warning(f"BRAVE_SEARCH: Request failed on attempt {attempt + 1}: {e}, retrying in {backoff_time}s")
                time.sleep(backoff_time)
        
        # Should not reach here
        raise Exception(f"Failed after {max_retries} attempts")
    
    def _process_response(self, response, max_results: int, query: str) -> List[Dict[str, Any]]:
        """Process successful API response"""
        
        try:
            data = response.json()
            logger.info("BRAVE_SEARCH: Successfully parsed JSON response")
            
            # Extract web results
            results = data.get("web", {}).get("results", [])
            logger.info(f"BRAVE_SEARCH: Found {len(results)} raw results in API response")
            
            # Format results consistently with Tavily output
            formatted_results = []
            for i, result in enumerate(results[:max_results]):
                # NO TRUNCATION - preserve full content for comprehensive research
                content = result.get("description", "")
                # Commented out content truncation - it was destroying research quality
                # if len(content) > 1000:  # Limit to 1000 characters
                #     content = content[:1000] + "..."
                
                title = result.get("title", "")
                # Keep title truncation as titles are usually short anyway
                if len(title) > 500:  # Increased title limit
                    title = title[:500] + "..."
                
                formatted_results.append({
                    "title": title,
                    "url": result.get("url", ""),
                    "content": content,
                    "score": result.get("relevance_score", 0.0) if "relevance_score" in result else 0.0,
                    "published_date": result.get("age") if "age" in result else None,
                })
                
                logger.debug(f"BRAVE_SEARCH: Processed result {i+1}: {title[:50]}...")
            
            logger.info(f"BRAVE_SEARCH: Successfully formatted {len(formatted_results)} search results")
            return formatted_results
            
        except Exception as e:
            global_rate_limiter.report_failure('brave', is_rate_limit=False)
            logger.error(f"BRAVE_SEARCH: Error processing response: {e}")
            raise Exception(f"Error processing Brave Search response: {str(e)}")


def create_brave_tool(api_key: Optional[str] = None) -> BraveSearchTool:
    """Factory function to create Brave search tool."""
    return BraveSearchTool(api_key=api_key)