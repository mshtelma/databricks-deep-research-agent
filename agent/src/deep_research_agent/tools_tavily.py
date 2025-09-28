"""
Tavily web search tool for research agent.
"""

import os
import requests
from typing import Dict, List, Any, Optional
from pydantic import Field

from .core.base_tools import BaseSearchTool, TavilySearchInput


class TavilySearchTool(BaseSearchTool):
    """Tool for performing web searches using Tavily API."""
    
    name: str = "tavily_search"
    description: str = "Search the web for current information using Tavily"
    args_schema: type[TavilySearchInput] = TavilySearchInput
    
    def _get_default_api_key(self) -> str:
        """Get the default API key from environment variable."""
        return os.getenv("TAVILY_API_KEY", "")
    
    def _get_default_base_url(self) -> str:
        """Get the default base URL for Tavily API."""
        return "https://api.tavily.com"
    
    def _get_service_name(self) -> str:
        """Get the service name for error messages."""
        return "Tavily Search"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using Tavily API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, url, content
        """
        url = f"{self.base_url}/search"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Format results consistently
            formatted_results = []
            for result in results:
                # Limit content size to prevent memory bloat
                content = result.get("content", "")
                # NO TRUNCATION - preserve full content for comprehensive research
                # if len(content) > 1000:  # Limit to 1000 characters
                #     content = content[:1000] + "..."
                
                title = result.get("title", "")
                # Keep title truncation but increase limit
                if len(title) > 500:  # Increased title limit
                    title = title[:500] + "..."
                
                formatted_results.append({
                    "title": title,
                    "url": result.get("url", ""),
                    "content": content,
                    "score": result.get("score", 0.0),
                    "published_date": result.get("published_date"),
                })
            
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Tavily API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing Tavily response: {str(e)}")


def create_tavily_tool(api_key: Optional[str] = None) -> TavilySearchTool:
    """Factory function to create Tavily search tool."""
    return TavilySearchTool(api_key=api_key)