"""
Refactored tools with improved error handling and architecture.

This module provides robust implementations of search tools using the new
core libraries and components.
"""

import os
import requests
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

from langchain_core.tools import BaseTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field

try:
    from databricks_langchain import DatabricksVectorSearch
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    DatabricksVectorSearch = None

from core import (
    get_logger,
    TavilySearchError,
    VectorSearchError,
    ToolInitializationError,
    retry_with_exponential_backoff,
    CircuitBreaker,
    timeout_after,
    validate_url,
    safe_json_loads,
    truncate_text
)

logger = get_logger(__name__)


class TavilySearchInput(BaseModel):
    """Input schema for Tavily search with validation."""
    query: str = Field(description="Search query to execute", min_length=1)
    max_results: int = Field(default=5, description="Maximum results to return", ge=1, le=20)


class EnhancedTavilySearchTool(BaseTool):
    """
    Enhanced Tavily search tool with robust error handling and validation.
    """
    
    name: str = "tavily_search"
    description: str = "Search the web for current information using Tavily API"
    args_schema: type[BaseModel] = TavilySearchInput
    api_key: str = Field(default="")
    base_url: str = Field(default="https://api.tavily.com")
    timeout_seconds: int = Field(default=30)
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize enhanced Tavily search tool.
        
        Args:
            api_key: Tavily API key
            **kwargs: Additional configuration
        """
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY", "")
        
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=TavilySearchError
        )
        
        logger.info(
            "Initialized enhanced Tavily search tool",
            has_api_key=bool(self.api_key),
            timeout=self.timeout_seconds
        )
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Execute search query with enhanced error handling."""
        return self.search(query, max_results)
    
    @retry_with_exponential_backoff(max_retries=2, exceptions=(requests.RequestException,))
    @timeout_after(30)  # Overall timeout
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using Tavily API with robust error handling.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
            
        Raises:
            TavilySearchError: If search fails
        """
        # Validation
        if not query or not query.strip():
            raise TavilySearchError("Search query cannot be empty")
        
        if not self.api_key:
            logger.warning("Tavily API key not provided")
            raise TavilySearchError("Tavily API key not configured")
        
        # Sanitize query
        query = query.strip()[:500]  # Limit query length
        
        # Use circuit breaker
        try:
            return self.circuit_breaker.call(self._perform_search, query, max_results)
        except Exception as e:
            logger.error("Tavily search failed", error=e, query=query[:50])
            raise TavilySearchError(f"Search failed: {str(e)}")
    
    def _perform_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform the actual search request."""
        url = urljoin(self.base_url, "/search")
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ResearchAgent/1.0"
        }
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": min(max_results, 20),  # API limit
            "include_domains": [],
            "exclude_domains": []
        }
        
        try:
            logger.debug("Making Tavily API request", query=query, max_results=max_results)
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds
            )
            
            # Check response status
            if response.status_code == 401:
                raise TavilySearchError("Invalid API key")
            elif response.status_code == 429:
                raise TavilySearchError("Rate limit exceeded")
            elif response.status_code >= 500:
                raise TavilySearchError(f"Server error: {response.status_code}")
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            raw_results = data.get("results", [])
            
            # Process and validate results
            processed_results = self._process_results(raw_results)
            
            logger.info(
                "Tavily search completed",
                query=truncate_text(query, 50),
                results_count=len(processed_results),
                status_code=response.status_code
            )
            
            return processed_results
        except requests.exceptions.Timeout:
            raise TavilySearchError("Search request timed out")
        except requests.exceptions.ConnectionError:
            raise TavilySearchError("Could not connect to Tavily API")
        except requests.exceptions.RequestException as e:
            raise TavilySearchError(f"HTTP request failed: {str(e)}")
        except ValueError as e:
            raise TavilySearchError(f"Invalid JSON response: {str(e)}")
    
    def _process_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and validate search results."""
        processed_results = []
        
        for result in raw_results:
            try:
                # Extract and validate fields
                processed_result = {
                    "title": str(result.get("title", "")).strip(),
                    "url": str(result.get("url", "")).strip(),
                    "content": str(result.get("content", "")).strip(),
                    "score": float(result.get("score", 0.0)),
                    "published_date": result.get("published_date")
                }
                
                # Validate URL
                if processed_result["url"] and not validate_url(processed_result["url"]):
                    logger.debug("Invalid URL in result", url=processed_result["url"])
                    processed_result["url"] = ""
                
                # Ensure we have some content
                if not processed_result["title"] and not processed_result["content"]:
                    logger.debug("Skipping result with no title or content")
                    continue
                
                # Truncate long content
                if len(processed_result["content"]) > 2000:
                    processed_result["content"] = truncate_text(processed_result["content"], 2000)
                
                processed_results.append(processed_result)
            except Exception as e:
                logger.debug("Failed to process search result", error=e, result=str(result)[:200])
                continue
        
        return processed_results


class EnhancedVectorSearchRetriever(BaseRetriever):
    """
    Enhanced vector search retriever with robust error handling.
    """
    
    def __init__(
        self, 
        index_name: str, 
        k: int = 5,
        text_column: str = "content",
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize enhanced vector search retriever.
        
        Args:
            index_name: Vector search index name
            k: Number of documents to retrieve
            text_column: Name of the text column
            columns: Additional columns to retrieve
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        
        self.index_name = index_name
        self.k = k
        self.text_column = text_column
        self.columns = columns or ["source", "title", "url"]
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=VectorSearchError
        )
        
        # Initialize vector search client
        self.vector_search = None
        if DATABRICKS_AVAILABLE:
            try:
                self.vector_search = DatabricksVectorSearch(
                    index_name=index_name,
                    text_column=text_column,
                    columns=self.columns
                )
                logger.info(
                    "Initialized vector search retriever",
                    index_name=index_name,
                    k=k,
                    text_column=text_column
                )
            except Exception as e:
                logger.error("Failed to initialize vector search", error=e)
                raise ToolInitializationError(f"Vector search initialization failed: {e}")
        else:
            logger.warning("Databricks vector search not available")
            raise ToolInitializationError("databricks_langchain not available")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents with enhanced error handling.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
            
        Raises:
            VectorSearchError: If retrieval fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to vector search")
            return []
        
        if not self.vector_search:
            logger.warning("Vector search not initialized")
            return []
        
        try:
            # Use circuit breaker
            return self.circuit_breaker.call(self._perform_search, query.strip())
        except Exception as e:
            logger.error("Vector search failed", error=e, query=truncate_text(query, 50))
            # Return empty list instead of raising exception for graceful degradation
            return []
    
    @retry_with_exponential_backoff(max_retries=2)
    @timeout_after(20)
    def _perform_search(self, query: str) -> List[Document]:
        """Perform the actual vector search."""
        try:
            logger.debug("Performing vector search", query=truncate_text(query, 50), k=self.k)
            
            # Perform similarity search
            results = self.vector_search.similarity_search(query, k=self.k)
            
            # Process results
            processed_docs = []
            for result in results:
                if isinstance(result, Document):
                    # Already a Document object
                    processed_docs.append(result)
                else:
                    # Convert dict to Document
                    content = str(result.get(self.text_column, ""))
                    metadata = {}
                    
                    # Extract metadata
                    for col in self.columns:
                        if col in result:
                            metadata[col] = result[col]
                    
                    # Add search metadata
                    metadata.update({
                        "index_name": self.index_name,
                        "search_query": query,
                        "score": result.get("score", 0.0)
                    })
                    
                    doc = Document(page_content=content, metadata=metadata)
                    processed_docs.append(doc)
            
            logger.info(
                "Vector search completed",
                query=truncate_text(query, 50),
                results_count=len(processed_docs)
            )
            
            return processed_docs
        except Exception as e:
            logger.error("Vector search operation failed", error=e)
            raise VectorSearchError(f"Vector search failed: {str(e)}")


def create_enhanced_tavily_tool(
    api_key: Optional[str] = None,
    timeout_seconds: int = 30,
    base_url: str = "https://api.tavily.com"
) -> EnhancedTavilySearchTool:
    """
    Create enhanced Tavily search tool with configuration.
    
    Args:
        api_key: Tavily API key
        timeout_seconds: Request timeout
        base_url: Tavily API base URL
        
    Returns:
        Configured EnhancedTavilySearchTool
    """
    return EnhancedTavilySearchTool(
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        base_url=base_url
    )


def create_enhanced_vector_search_retriever(
    index_name: str,
    k: int = 5,
    text_column: str = "content",
    columns: Optional[List[str]] = None
) -> Optional[EnhancedVectorSearchRetriever]:
    """
    Create enhanced vector search retriever with configuration.
    
    Args:
        index_name: Vector search index name
        k: Number of documents to retrieve
        text_column: Name of the text column
        columns: Additional columns to retrieve
        
    Returns:
        Configured EnhancedVectorSearchRetriever or None if unavailable
    """
    if not index_name:
        logger.info("No vector search index provided")
        return None
    
    if not DATABRICKS_AVAILABLE:
        logger.warning("Databricks vector search not available")
        return None
    
    try:
        return EnhancedVectorSearchRetriever(
            index_name=index_name,
            k=k,
            text_column=text_column,
            columns=columns
        )
    except Exception as e:
        logger.error("Failed to create vector search retriever", error=e)
        return None