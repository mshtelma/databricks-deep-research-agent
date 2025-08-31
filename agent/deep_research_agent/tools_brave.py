"""
Brave web search tool for research agent.
"""

import os
import requests
from typing import Dict, List, Any, Optional
from pydantic import Field

from deep_research_agent.core.base_tools import BaseSearchTool, BraveSearchInput


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
        Search the web using Brave Search API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, url, content
        """
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
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                import time
                retry_after = response.headers.get('Retry-After', '5')
                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = 5
                
                raise Exception(f"Rate limited. Retry after {wait_time} seconds")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract web results
            results = data.get("web", {}).get("results", [])
            
            # Format results consistently with Tavily output
            formatted_results = []
            for result in results[:max_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("description", ""),
                    "score": result.get("relevance_score", 0.0) if "relevance_score" in result else 0.0,
                    "published_date": result.get("age") if "age" in result else None,
                })
            
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Brave Search API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing Brave Search response: {str(e)}")


def create_brave_tool(api_key: Optional[str] = None) -> BraveSearchTool:
    """Factory function to create Brave search tool."""
    return BraveSearchTool(api_key=api_key)