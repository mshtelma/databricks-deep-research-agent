"""
LLM Factory with Dependency Injection for Clean Testing

This factory allows injecting different LLM implementations without modifying production code.
Tests can register mock factories while production uses real LLMs.
"""

from typing import Any, Dict, Callable, Optional, Protocol
from abc import ABC, abstractmethod
import os
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def invoke(self, messages, **kwargs) -> Any:
        """Synchronous LLM invocation."""
        ...
    
    async def ainvoke(self, messages, **kwargs) -> Any:
        """Asynchronous LLM invocation."""
        ...


class LLMFactory(ABC):
    """Abstract factory for creating LLMs."""
    
    @abstractmethod
    def create_llm(self, config: Dict[str, Any]) -> LLMProvider:
        """Create an LLM instance."""
        pass


class ProductionLLMFactory(LLMFactory):
    """Production LLM factory using real Databricks endpoints."""
    
    def create_llm(self, config: Dict[str, Any]) -> LLMProvider:
        """Create a real LLM instance."""
        from deep_research_agent.core.model_manager import ModelManager
        
        try:
            model_manager = ModelManager()
            llm = model_manager.get_chat_model("default")
            
            if llm is None:
                raise ValueError("ModelManager returned None for default LLM")
                
            logger.info("Successfully created production LLM")
            return llm
            
        except Exception as e:
            error_msg = f"Failed to initialize production LLM: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


class MockLLMFactory(LLMFactory):
    """Mock LLM factory for testing."""
    
    def create_llm(self, config: Dict[str, Any]) -> LLMProvider:
        """Create a mock LLM instance."""
        from deep_research_agent.core.mock_llm import MockLLM
        logger.info("Created mock LLM for testing")
        return MockLLM(config)


class LLMRegistry:
    """
    Registry for LLM factories with dependency injection.
    
    This allows tests to register mock factories without touching production code.
    """
    
    _instance: Optional['LLMRegistry'] = None
    _factory: Optional[LLMFactory] = None
    
    def __new__(cls) -> 'LLMRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_factory(self, factory: LLMFactory) -> None:
        """Register an LLM factory."""
        self._factory = factory
        logger.debug(f"Registered LLM factory: {factory.__class__.__name__}")
    
    def get_factory(self) -> LLMFactory:
        """Get the registered factory or default to production."""
        if self._factory is None:
            # Default to production factory
            self._factory = ProductionLLMFactory()
            logger.debug("Using default production LLM factory")
        
        return self._factory
    
    def create_llm(self, config: Dict[str, Any]) -> LLMProvider:
        """Create an LLM using the registered factory."""
        factory = self.get_factory()
        return factory.create_llm(config)
    
    def reset(self) -> None:
        """Reset to default factory (for testing cleanup)."""
        self._factory = None


def get_llm_registry() -> LLMRegistry:
    """Get the global LLM registry instance."""
    return LLMRegistry()


def create_llm(config: Dict[str, Any]) -> LLMProvider:
    """
    Create an LLM using the registered factory.
    
    This is the main entry point that production code should use.
    Tests can inject different factories without changing this code.
    """
    registry = get_llm_registry()
    return registry.create_llm(config)