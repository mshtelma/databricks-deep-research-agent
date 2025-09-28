"""
Tool management for the research agent.

This module provides centralized tool initialization, registration,
and management for all tools used by the research agent.
"""

from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod

try:
    from databricks_langchain import UCFunctionToolkit, VectorSearchRetrieverTool
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    UCFunctionToolkit = None
    VectorSearchRetrieverTool = None
    DatabricksFunctionClient = None

from langchain_core.tools import BaseTool
from langchain_core.retrievers import BaseRetriever

from ..core import (
    get_logger,
    ToolInitializationError,
    ToolType,
    retry_with_exponential_backoff,
    CircuitBreaker
)
from ..core.unified_config import (
    get_config_manager,
    ToolConfigSchema as ToolConfiguration  # Alias for compatibility
)

logger = get_logger(__name__)


class ToolFactory(ABC):
    """Abstract base class for tool factories."""
    
    @abstractmethod
    def create_tool(self, config: ToolConfiguration) -> Optional[BaseTool]:
        """Create tool instance from configuration."""
        pass
    
    @abstractmethod
    def get_tool_type(self) -> ToolType:
        """Get the tool type this factory creates."""
        pass
    
    @abstractmethod
    def validate_config(self, config: ToolConfiguration) -> List[str]:
        """Validate tool configuration and return list of issues."""
        pass


class BraveToolFactory(ToolFactory):
    """Factory for creating Brave search tools."""
    
    def get_tool_type(self) -> ToolType:
        """Get tool type."""
        return ToolType.BRAVE_SEARCH
    
    def validate_config(self, config: ToolConfiguration) -> List[str]:
        """Validate Brave configuration."""
        issues = []
        
        if not config.enabled:
            return issues  # Skip validation if disabled
        
        api_key = config.config.get("api_key")
        if not api_key:
            issues.append("Brave API key is required")
        elif api_key.startswith("{{secrets/"):
            # Placeholder for secrets - assume valid in deployment
            pass
        
        max_results = config.config.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            issues.append("max_results must be a positive integer")
        
        return issues
    
    def create_tool(self, config: ToolConfiguration) -> Optional[BaseTool]:
        """Create Brave search tool."""
        try:
            from ..tools_brave import BraveSearchTool
            
            print(f"[BraveToolFactory] Creating tool, enabled: {config.enabled}")
            logger.info(f"BraveToolFactory.create_tool: enabled={config.enabled}")
            
            if not config.enabled:
                print("[BraveToolFactory] Tool disabled in configuration")
                logger.info("Brave tool disabled in configuration")
                return None
            
            api_key = config.config.get("api_key")
            print(f"[BraveToolFactory] Raw API key provided: {bool(api_key)}")
            if api_key:
                is_secret_ref = api_key.startswith('{{secrets/')
                key_preview = api_key[:20] + "..." if len(api_key) > 20 else api_key
                print(f"[BraveToolFactory] API key format - is secret ref: {is_secret_ref}, preview: {key_preview}")
            
            # Create tool with configuration - the BaseSearchTool will handle secret resolution
            tool = BraveSearchTool(
                api_key=api_key,
                base_url=config.config.get("base_url"),
                timeout_seconds=config.timeout_seconds or 30,
                max_retries=config.max_retries or 3
            )
            
            print(f"[BraveToolFactory] Successfully created Brave search tool")
            logger.info("Created Brave search tool", base_url=config.config.get("base_url"))
            return tool
        except Exception as e:
            print(f"[BraveToolFactory] ERROR: Exception during tool creation: {e}")
            
            # Check if this is an API key resolution issue - if so, return None instead of raising
            error_message = str(e)
            if "API key could not be resolved" in error_message or "API key is required but not provided" in error_message:
                print("[BraveToolFactory] API key resolution failed - returning None for graceful degradation")
                logger.info("Brave tool unavailable due to missing API key - returning None")
                return None
            
            # For other errors, still raise the exception
            logger.error("Failed to create Brave tool", error=e)
            raise ToolInitializationError(f"Failed to create Brave tool: {e}")


class TavilyToolFactory(ToolFactory):
    """Factory for creating Tavily search tools."""
    
    def get_tool_type(self) -> ToolType:
        """Get tool type."""
        return ToolType.TAVILY_SEARCH
    
    def validate_config(self, config: ToolConfiguration) -> List[str]:
        """Validate Tavily configuration."""
        issues = []
        
        if not config.enabled:
            return issues  # Skip validation if disabled
        
        api_key = config.config.get("api_key")
        if not api_key:
            issues.append("Tavily API key is required")
        elif api_key.startswith("{{secrets/"):
            # Placeholder for secrets - assume valid in deployment
            pass
        
        max_results = config.config.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            issues.append("max_results must be a positive integer")
        
        return issues
    
    def create_tool(self, config: ToolConfiguration) -> Optional[BaseTool]:
        """Create Tavily search tool."""
        try:
            # Check if tool is disabled first
            if not config.enabled:
                logger.info("Tavily tool disabled in configuration")
                return None
            
            # Check if API key is available - Tavily requires API key
            api_key = config.config.get("api_key")
            if not api_key or api_key.startswith("{{secrets/"):
                logger.info("Tavily tool disabled - no API key available")
                return None
            
            from ..tools_tavily import TavilySearchTool
            
            # Create tool with configuration - the BaseSearchTool will handle secret resolution
            tool = TavilySearchTool(
                api_key=api_key,
                base_url=config.config.get("base_url"),
                timeout_seconds=config.timeout_seconds or 30,
                max_retries=config.max_retries or 3
            )
            
            logger.info("Created Tavily search tool", api_key_provided=bool(api_key))
            return tool
        except Exception as e:
            logger.info(f"Tavily tool not available: {e}")
            # Don't raise an exception for missing Tavily, just return None
            return None


class VectorSearchToolFactory(ToolFactory):
    """Factory for creating Vector Search tools."""
    
    def get_tool_type(self) -> ToolType:
        """Get tool type."""
        return ToolType.VECTOR_SEARCH
    
    def validate_config(self, config: ToolConfiguration) -> List[str]:
        """Validate Vector Search configuration."""
        issues = []
        
        if not config.enabled:
            return issues
        
        if not DATABRICKS_AVAILABLE:
            issues.append("databricks_langchain not available for Vector Search")
            return issues
        
        index_name = config.config.get("index_name")
        if not index_name:
            issues.append("Vector Search index name is required")
        
        k = config.config.get("k", 5)
        if not isinstance(k, int) or k < 1:
            issues.append("k must be a positive integer")
        
        return issues
    
    def create_tool(self, config: ToolConfiguration) -> Optional[BaseRetriever]:
        """Create Vector Search retriever."""
        try:
            if not config.enabled:
                logger.info("Vector Search tool disabled in configuration")
                return None
            
            if not DATABRICKS_AVAILABLE:
                logger.warning("databricks_langchain not available, Vector Search disabled")
                return None
            
            from ..tools_vector_search import setup_vector_search_retriever
            
            index_name = config.config.get("index_name")
            k = config.config.get("k", 5)
            
            retriever = setup_vector_search_retriever(index_name, k)
            
            if retriever:
                logger.info("Created Vector Search retriever", index_name=index_name, k=k)
            else:
                logger.warning("Failed to create Vector Search retriever")
            
            return retriever
        except Exception as e:
            logger.error("Failed to create Vector Search tool", error=e)
            raise ToolInitializationError(f"Failed to create Vector Search tool: {e}")


class UCFunctionToolFactory(ToolFactory):
    """Factory for creating Unity Catalog function tools."""
    
    def get_tool_type(self) -> ToolType:
        """Get tool type."""
        return ToolType.UC_FUNCTION
    
    def validate_config(self, config: ToolConfiguration) -> List[str]:
        """Validate UC Function configuration."""
        issues = []
        
        if not config.enabled:
            return issues
        
        if not DATABRICKS_AVAILABLE:
            issues.append("databricks_langchain not available for UC Functions")
            return issues
        
        function_names = config.config.get("function_names", [])
        if not function_names:
            issues.append("At least one function name is required")
        
        return issues
    
    def create_tool(self, config: ToolConfiguration) -> Optional[List[BaseTool]]:
        """Create Unity Catalog function tools."""
        try:
            if not config.enabled:
                logger.info("UC Function tools disabled in configuration")
                return None
            
            if not DATABRICKS_AVAILABLE:
                logger.warning("databricks_langchain not available, UC Functions disabled")
                return None
            
            function_names = config.config.get("function_names", ["system.ai.python_exec"])
            
            # Initialize UC function client
            try:
                uc_client = DatabricksFunctionClient()
                from databricks_langchain import set_uc_function_client
                set_uc_function_client(uc_client)
            except Exception as e:
                logger.warning("Failed to initialize UC function client", error=e)
                return None
            
            # Create toolkit
            toolkit = UCFunctionToolkit(function_names=function_names)
            tools = toolkit.tools
            
            logger.info(
                f"Created {len(tools)} UC function tools", 
                function_names=function_names,
                tool_count=len(tools)
            )
            
            return tools
        except Exception as e:
            logger.error("Failed to create UC function tools", error=e)
            raise ToolInitializationError(f"Failed to create UC function tools: {e}")


class ToolRegistry:
    """Registry for managing all tools."""
    
    def __init__(self, config_manager=None):
        """
        Initialize tool registry.
        
        Args:
            config_manager: Configuration manager instance
        """
        if config_manager is None:
            config_manager = get_config_manager()
        self.config_manager = config_manager
        self.factories: Dict[ToolType, ToolFactory] = {}
        self.tools: Dict[ToolType, Any] = {}
        self.circuit_breakers: Dict[ToolType, CircuitBreaker] = {}
        
        # Register built-in factories
        self._register_builtin_factories()
    
    def _register_builtin_factories(self):
        """Register built-in tool factories."""
        self.register_factory(TavilyToolFactory())
        self.register_factory(BraveToolFactory())
        self.register_factory(VectorSearchToolFactory())
        self.register_factory(UCFunctionToolFactory())
    
    def register_factory(self, factory: ToolFactory):
        """
        Register a tool factory.
        
        Args:
            factory: Tool factory instance
        """
        tool_type = factory.get_tool_type()
        self.factories[tool_type] = factory
        
        # Create circuit breaker for this tool type
        self.circuit_breakers[tool_type] = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60
        )
        
        logger.debug(f"Registered factory for {tool_type.value}")
    
    def get_tool(self, tool_type: ToolType) -> Optional[Any]:
        """
        Get tool instance, creating it if necessary.
        
        Args:
            tool_type: Type of tool to get
            
        Returns:
            Tool instance or None if unavailable
        """
        # Return cached tool if available
        if tool_type in self.tools:
            return self.tools[tool_type]
        
        # Create tool if factory is available
        if tool_type in self.factories:
            try:
                tool = self._create_tool_with_circuit_breaker(tool_type)
                if tool is not None:
                    self.tools[tool_type] = tool
                return tool
            except Exception as e:
                logger.error(f"Failed to create tool {tool_type.value}", error=e)
                return None
        
        logger.warning(f"No factory registered for tool type: {tool_type.value}")
        return None
    
    def _create_tool_with_circuit_breaker(self, tool_type: ToolType) -> Optional[Any]:
        """Create tool with circuit breaker protection."""
        circuit_breaker = self.circuit_breakers[tool_type]
        factory = self.factories[tool_type]
        config = self.config_manager.get_tool_config(tool_type)
        
        # Pre-check if tool is disabled to avoid unnecessary circuit breaker calls
        if hasattr(config, 'enabled') and not config.enabled:
            logger.debug(f"Tool {tool_type.value} is disabled in configuration")
            return None
        
        try:
            return circuit_breaker.call(factory.create_tool, config)
        except Exception as e:
            # Don't log as error for Tavily if it's just unavailable
            if tool_type.value == 'tavily_search':
                logger.info(f"Tavily tool not available: {e}")
            else:
                logger.error(f"Tool creation failed for {tool_type.value}", error=e)
            return None
    
    def get_all_tools(self) -> Dict[ToolType, Any]:
        """
        Get all available tools.
        
        Returns:
            Dictionary mapping tool types to tool instances
        """
        all_tools = {}
        
        for tool_type in self.factories.keys():
            tool = self.get_tool(tool_type)
            if tool is not None:
                all_tools[tool_type] = tool
        
        logger.info(f"Retrieved {len(all_tools)} available tools")
        return all_tools
    
    def validate_all_configurations(self) -> Dict[ToolType, List[str]]:
        """
        Validate configurations for all registered tools.
        
        Returns:
            Dictionary mapping tool types to validation issues
        """
        validation_results = {}
        
        for tool_type, factory in self.factories.items():
            try:
                config = self.config_manager.get_tool_config(tool_type)
                issues = factory.validate_config(config)
                validation_results[tool_type] = issues
            except Exception as e:
                validation_results[tool_type] = [f"Configuration error: {e}"]
        
        return validation_results
    
    def get_tool_status(self) -> Dict[str, Any]:
        """
        Get status of all tools.
        
        Returns:
            Dictionary with tool status information
        """
        status = {
            "total_registered": len(self.factories),
            "total_available": len([t for t in self.tools.values() if t is not None]),
            "tools": {}
        }
        
        for tool_type in self.factories.keys():
            tool_status = {
                "available": tool_type in self.tools and self.tools[tool_type] is not None,
                "circuit_breaker_state": self.circuit_breakers[tool_type].state,
                "failure_count": self.circuit_breakers[tool_type].failure_count
            }
            
            try:
                config = self.config_manager.get_tool_config(tool_type)
                tool_status["enabled"] = config.enabled
            except Exception:
                tool_status["enabled"] = False
            
            status["tools"][tool_type.value] = tool_status
        
        return status
    
    def get_tools_by_type(self, tool_type_name: str) -> List[Any]:
        """
        Get all tools of a specific type (e.g., "search").
        
        Args:
            tool_type_name: Name of the tool type (e.g., "search")
            
        Returns:
            List of tool instances matching the type
        """
        search_tools = []
        
        # Map common type names to actual tool types
        if tool_type_name.lower() == "search":
            # Get all search-related tools
            search_tool_types = [ToolType.TAVILY_SEARCH, ToolType.BRAVE_SEARCH]
            for tool_type in search_tool_types:
                if tool_type in self.factories:
                    tool = self.get_tool(tool_type)
                    if tool is not None:
                        search_tools.append(tool)
        
        return search_tools
    
    @retry_with_exponential_backoff(max_retries=2)
    def health_check(self, tool_type: ToolType) -> bool:
        """
        Perform health check on a specific tool.
        
        Args:
            tool_type: Type of tool to check
            
        Returns:
            True if tool is healthy, False otherwise
        """
        try:
            tool = self.get_tool(tool_type)
            if tool is None:
                return False
            
            # Basic health check - tool exists and is callable
            if hasattr(tool, '__call__') or hasattr(tool, 'invoke'):
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Health check failed for {tool_type.value}", error=e)
            return False


def create_tool_registry(config_manager=None) -> ToolRegistry:
    """
    Create tool registry with default configuration.
    
    Args:
        config_manager: Optional configuration manager
        
    Returns:
        Configured ToolRegistry instance
    """
    if config_manager is None:
        config_manager = get_config_manager()
    
    return ToolRegistry(config_manager)