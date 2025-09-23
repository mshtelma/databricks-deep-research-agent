"""
Unified tool wrapper for consistent tool interface across all agents.

This module provides a wrapper that standardizes tool invocation across
different tool types and interfaces, handling the complexity of different
method names (search, execute, invoke, run) in a single place.
"""

from typing import Any, Dict, List, Optional, Union
from langchain_core.tools import BaseTool

from deep_research_agent.core import get_logger
from deep_research_agent.core.types import SearchResult, SearchResultType

logger = get_logger(__name__)


class UnifiedToolWrapper:
    """
    Wrapper that provides a consistent interface for all tool types.
    
    This wrapper handles the complexity of different tool interfaces:
    - search() method for custom search tools
    - execute() method for legacy tools
    - invoke() method for LangChain tools
    - run() method for older tools
    - callable interface for mock tools
    """
    
    def __init__(self, tool: Any):
        """
        Initialize the wrapper with a tool instance.
        
        Args:
            tool: Any tool instance (search tool, LangChain tool, etc.)
        """
        self.tool = tool
        self.tool_name = getattr(tool, 'name', tool.__class__.__name__)
        self.tool_type = self._identify_tool_type()
        
    def _identify_tool_type(self) -> str:
        """Identify the type of tool to optimize method selection."""
        if hasattr(self.tool, 'search'):
            return "search_tool"
        elif hasattr(self.tool, 'execute'):
            return "execute_tool"
        elif hasattr(self.tool, 'invoke'):
            return "langchain_tool"
        elif hasattr(self.tool, 'run'):
            return "legacy_tool"
        elif callable(self.tool):
            return "callable"
        else:
            return "unknown"
            
    def execute(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Unified execution method that adapts to different tool interfaces.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (e.g., max_results)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            AttributeError: If tool has no executable method
            Exception: If tool execution fails
        """
        try:
            # Priority order based on tool type for performance
            if self.tool_type == "search_tool":
                # Direct search method (BraveSearchTool, TavilySearchTool)
                max_results = kwargs.get('max_results', 5)
                results = self.tool.search(query, max_results)
                
            elif self.tool_type == "execute_tool":
                # Custom execute method
                results = self.tool.execute(query, **kwargs)
                
            elif self.tool_type == "langchain_tool":
                # LangChain's standard invoke method
                input_dict = {"query": query, **kwargs}
                results = self.tool.invoke(input_dict)
                
            elif self.tool_type == "legacy_tool":
                # Legacy run method
                results = self.tool.run(query, **kwargs)
                
            elif self.tool_type == "callable":
                # Handle mock tools or callables
                results = self.tool(query, **kwargs)
                
            else:
                raise AttributeError(
                    f"Tool {self.tool_name} has no executable method "
                    "(search, execute, invoke, run, or callable)"
                )
            
            # Convert results to SearchResult objects if needed
            normalized_results = self._normalize_results(results)
            
            logger.debug(f"Tool {self.tool_name} executed successfully, "
                        f"returned {len(normalized_results)} results")
            
            return normalized_results
            
        except Exception as e:
            logger.error(f"Tool execution failed for {self.tool_name}: {e}")
            # Re-raise to maintain error propagation
            raise
            
    def _normalize_results(self, results: Any) -> List[SearchResult]:
        """
        Convert various result formats to SearchResult objects.
        
        Args:
            results: Raw results from tool execution
            
        Returns:
            List of normalized SearchResult objects
        """
        if not results:
            return []
            
        normalized = []
        
        try:
            # Handle different result formats
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, SearchResult):
                        # Already a SearchResult
                        normalized.append(result)
                    elif isinstance(result, dict):
                        # Convert dictionary to SearchResult
                        normalized.append(self._dict_to_search_result(result))
                    elif isinstance(result, str):
                        # Handle string results (fallback)
                        normalized.append(SearchResult(
                            content=result,
                            source=f"{self.tool_name}_result",
                            result_type=SearchResultType.WEB
                        ))
                    else:
                        logger.warning(f"Unknown result type: {type(result)}")
                        
            elif isinstance(results, dict):
                # Single dictionary result
                normalized.append(self._dict_to_search_result(results))
                
            elif isinstance(results, str):
                # Single string result
                normalized.append(SearchResult(
                    content=results,
                    source=f"{self.tool_name}_result",
                    result_type=SearchResultType.WEB
                ))
                
            else:
                logger.warning(f"Unknown results type: {type(results)}")
                
        except Exception as e:
            logger.error(f"Error normalizing results from {self.tool_name}: {e}")
            # Return empty list rather than failing
            
        return normalized
        
    def _dict_to_search_result(self, result_dict: Dict[str, Any]) -> SearchResult:
        """
        Convert a dictionary to a SearchResult object.
        
        Args:
            result_dict: Dictionary with search result data
            
        Returns:
            SearchResult object
        """
        return SearchResult(
            content=result_dict.get('content') or result_dict.get('description', ''),
            source=result_dict.get('source') or result_dict.get('url', ''),
            url=result_dict.get('url', ''),
            title=result_dict.get('title', ''),
            score=float(result_dict.get('score', 0.0)),
            relevance_score=float(result_dict.get('relevance_score', result_dict.get('score', 0.0))),
            published_date=result_dict.get('published_date'),
            result_type=SearchResultType.WEB,
            metadata=result_dict.get('metadata', {})
        )


def wrap_tool(tool: Any) -> UnifiedToolWrapper:
    """
    Convenience function to wrap a tool.
    
    Args:
        tool: Tool instance to wrap
        
    Returns:
        UnifiedToolWrapper instance
    """
    return UnifiedToolWrapper(tool)


def execute_tool_safely(tool: Any, query: str, **kwargs) -> List[SearchResult]:
    """
    Execute a tool safely with unified interface.
    
    Args:
        tool: Tool instance
        query: Search query
        **kwargs: Additional parameters
        
    Returns:
        List of SearchResult objects (empty list on failure)
    """
    try:
        wrapper = wrap_tool(tool)
        return wrapper.execute(query, **kwargs)
    except Exception as e:
        logger.error(f"Safe tool execution failed: {e}")
        return []