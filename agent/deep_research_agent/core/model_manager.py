"""
Multi-model management for Databricks research agent.

This module provides flexible model configuration and management,
allowing different models to be used for different workflow nodes
while maintaining backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum

try:
    from databricks_langchain import ChatDatabricks
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    ChatDatabricks = None

from .types import WorkflowNodeType
from .logging import get_logger
from .exceptions import ResearchAgentError, LLMEndpointError

logger = get_logger(__name__)


class ModelRole(str, Enum):
    """Roles that models can play in the workflow."""
    QUERY_GENERATION = "query_generation"
    WEB_RESEARCH = "web_research"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"
    EMBEDDING = "embedding"
    DEFAULT = "default"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    endpoint: str
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout_seconds: int = 30
    max_retries: int = 3

@dataclass 
class NodeModelConfiguration:
    """Configuration for per-node model assignments."""
    
    # Per-node model configurations
    default_model: ModelConfig = field(default_factory=lambda: ModelConfig(endpoint="databricks-gpt-oss-20b"))
    query_generation_model: Optional[ModelConfig] = None
    web_research_model: Optional[ModelConfig] = None
    reflection_model: Optional[ModelConfig] = None
    synthesis_model: Optional[ModelConfig] = None
    
    # Embedding model for deduplication and similarity
    embedding_model: ModelConfig = field(default_factory=lambda: ModelConfig(endpoint="databricks-bge-large-en"))
    
    # Global fallback configuration
    enable_fallback: bool = True
    
    def get_model_for_node(self, node_type: WorkflowNodeType) -> ModelConfig:
        """
        Get the model configuration for a specific workflow node.
        
        Args:
            node_type: The workflow node type
            
        Returns:
            ModelConfig to use for this node
        """
        node_models = {
            WorkflowNodeType.QUERY_GENERATION: self.query_generation_model,
            WorkflowNodeType.WEB_RESEARCH: self.web_research_model,
            WorkflowNodeType.REFLECTION: self.reflection_model,
            WorkflowNodeType.SYNTHESIS: self.synthesis_model,
        }
        
        specific_model = node_models.get(node_type)
        return specific_model if specific_model is not None else self.default_model
    
    def get_model_for_role(self, role: ModelRole) -> ModelConfig:
        """
        Get the model configuration for a specific role.
        
        Args:
            role: The model role
            
        Returns:
            ModelConfig to use for this role
        """
        role_models = {
            ModelRole.QUERY_GENERATION: self.query_generation_model,
            ModelRole.WEB_RESEARCH: self.web_research_model,
            ModelRole.REFLECTION: self.reflection_model,
            ModelRole.SYNTHESIS: self.synthesis_model,
            ModelRole.EMBEDDING: self.embedding_model,
            ModelRole.DEFAULT: self.default_model,
        }
        
        specific_model = role_models.get(role)
        if specific_model is not None:
            return specific_model
        
        # For non-embedding roles, fall back to default
        if role != ModelRole.EMBEDDING:
            return self.default_model
        
        return self.embedding_model
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.default_model or not self.default_model.endpoint:
            raise ValueError("default_model endpoint must be specified")
        
        if not self.embedding_model or not self.embedding_model.endpoint:
            raise ValueError("embedding_model endpoint must be specified")
        
        # Validate all model configurations
        models_to_validate = [self.default_model, self.embedding_model]
        
        # Add optional models that are configured
        for model in [self.query_generation_model, self.web_research_model, 
                      self.reflection_model, self.synthesis_model]:
            if model is not None:
                models_to_validate.append(model)
        
        for model in models_to_validate:
            if hasattr(model, 'temperature') and (model.temperature < 0 or model.temperature > 2):
                raise ValueError(f"temperature must be between 0 and 2 for {model.endpoint}")
            
            if hasattr(model, 'max_tokens') and model.max_tokens < 1:
                raise ValueError(f"max_tokens must be positive for {model.endpoint}")
            
            if hasattr(model, 'timeout_seconds') and model.timeout_seconds < 1:
                raise ValueError(f"timeout_seconds must be positive for {model.endpoint}")


class ModelManager:
    """
    Manages LLM instances for different workflow nodes.
    
    Provides caching, fallback logic, and health monitoring
    for model endpoints used throughout the research workflow.
    """
    
    def __init__(self, config: Optional[NodeModelConfiguration] = None):
        """
        Initialize the model manager.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or NodeModelConfiguration()
        self.config.validate()
        
        # Cache for model instances
        self._model_cache: Dict[str, ChatDatabricks] = {}
        
        # Usage tracking
        self._usage_stats: Dict[str, int] = {}
        
        # Health status
        self._model_health: Dict[str, bool] = {}
        
        if not DATABRICKS_AVAILABLE:
            raise LLMEndpointError("databricks_langchain not available")
        
        logger.info(
            "Initialized model manager",
            default_model=self.config.default_model,
            embedding_model=self.config.embedding_model,
            fallback_enabled=self.config.enable_fallback
        )
    
    def get_llm_for_node(self, node_type: WorkflowNodeType) -> ChatDatabricks:
        """
        Get an LLM instance for a specific workflow node.
        
        Args:
            node_type: The workflow node type
            
        Returns:
            ChatDatabricks instance for the node
            
        Raises:
            LLMEndpointError: If model initialization fails
        """
        model_config = self.config.get_model_for_node(node_type)
        return self._get_or_create_model(model_config, node_type)
    
    def get_llm_for_role(self, role: ModelRole) -> ChatDatabricks:
        """
        Get an LLM instance for a specific role.
        
        Args:
            role: The model role
            
        Returns:
            ChatDatabricks instance for the role
            
        Raises:
            LLMEndpointError: If model initialization fails
        """
        model_config = self.config.get_model_for_role(role)
        return self._get_or_create_model(model_config, role)
    
    def get_chat_model(self, model_name: str = "default") -> ChatDatabricks:
        """
        Get a chat model instance by name.
        
        Args:
            model_name: Name of the model ("default", "synthesis", etc.)
            
        Returns:
            ChatDatabricks instance
            
        Raises:
            LLMEndpointError: If model initialization fails
        """
        # Map string names to model roles
        role_mapping = {
            "default": ModelRole.DEFAULT,
            "synthesis": ModelRole.SYNTHESIS,
            "web_research": ModelRole.WEB_RESEARCH,
            "query_generation": ModelRole.QUERY_GENERATION,
            "reflection": ModelRole.REFLECTION,
        }
        
        role = role_mapping.get(model_name.lower(), ModelRole.DEFAULT)
        return self.get_llm_for_role(role)
    
    def _get_or_create_model(self, model_config: ModelConfig, context: Any = None) -> ChatDatabricks:
        """
        Get or create a model instance with caching.
        
        Args:
            model_config: Model configuration
            context: Context for logging (node type or role)
            
        Returns:
            ChatDatabricks instance
        """
        model_endpoint = model_config.endpoint
        
        # Check cache first
        if model_endpoint in self._model_cache:
            # Update usage stats
            self._usage_stats[model_endpoint] = self._usage_stats.get(model_endpoint, 0) + 1
            return self._model_cache[model_endpoint]
        
        # Create new model instance
        try:
            model = self._create_model_instance(model_config)
            
            # Test the model with a simple call
            if self._test_model_health(model):
                self._model_cache[model_endpoint] = model
                self._model_health[model_endpoint] = True
                self._usage_stats[model_endpoint] = 1
                
                logger.info(
                    "Created and cached model instance",
                    model=model_endpoint,
                    context=str(context)
                )
                
                return model
            else:
                # Model failed health check
                self._model_health[model_endpoint] = False
                if self.config.enable_fallback and model_endpoint != self.config.default_model.endpoint:
                    logger.warning(
                        "Model failed health check, using fallback",
                        failed_model=model_endpoint,
                        fallback_model=self.config.default_model.endpoint,
                        context=str(context)
                    )
                    return self._get_or_create_model(self.config.default_model, context)
                else:
                    raise LLMEndpointError(f"Model {model_endpoint} failed health check and no fallback available")
        
        except Exception as e:
            logger.error(
                "Failed to create model instance",
                model=model_endpoint,
                error=str(e),
                context=str(context)
            )
            
            # Try fallback if enabled and not already using fallback
            if (self.config.enable_fallback and 
                model_endpoint != self.config.default_model.endpoint):
                logger.warning(
                    "Using fallback model due to creation failure",
                    failed_model=model_endpoint,
                    fallback_model=self.config.default_model.endpoint,
                    context=str(context)
                )
                return self._get_or_create_model(self.config.default_model, context)
            
            raise LLMEndpointError(f"Failed to create model {model_endpoint}: {e}")
    
    def _create_model_instance(self, model_config: ModelConfig) -> ChatDatabricks:
        """Create a new ChatDatabricks instance from a ModelConfig."""
        return ChatDatabricks(
            endpoint=model_config.endpoint,
            temperature=getattr(model_config, 'temperature', 0.7),
            max_tokens=getattr(model_config, 'max_tokens', 4000),
            max_retries=getattr(model_config, 'max_retries', 3)
        )
    
    def _test_model_health(self, model: ChatDatabricks, timeout_seconds: int = 10) -> bool:
        """
        Test if a model is healthy by making a simple call.
        
        Args:
            model: Model instance to test
            timeout_seconds: Timeout for the test
            
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            # Simple test message
            from langchain_core.messages import HumanMessage
            
            test_message = HumanMessage(content="Hello")
            
            # Use a shorter timeout for health check
            response = model.invoke([test_message])
            
            # Check if we got a valid response
            return response and hasattr(response, 'content') and response.content
            
        except Exception as e:
            logger.debug(f"Model health check failed: {e}")
            return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics and health status.
        
        Returns:
            Dictionary with model statistics
        """
        return {
            "cached_models": list(self._model_cache.keys()),
            "usage_stats": self._usage_stats.copy(),
            "health_status": self._model_health.copy(),
            "config": {
                "default_model": self.config.default_model.endpoint,
                "embedding_model": self.config.embedding_model.endpoint,
                "fallback_enabled": self.config.enable_fallback
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        logger.info("Clearing model cache", cached_count=len(self._model_cache))
        self._model_cache.clear()
        self._usage_stats.clear()
        self._model_health.clear()
    
    def preload_models(self) -> None:
        """
        Preload all configured models to warm the cache.
        
        This can be useful for reducing latency on first use.
        """
        logger.info("Preloading models")
        
        # Preload models for all node types
        for node_type in WorkflowNodeType:
            try:
                model_config = self.config.get_model_for_node(node_type)
                self._get_or_create_model(model_config, node_type)
            except Exception as e:
                logger.warning(
                    "Failed to preload model for node",
                    node_type=str(node_type),
                    error=str(e)
                )
        
        # Preload embedding model
        embedding_model_config = self.config.get_model_for_role(ModelRole.EMBEDDING)
        try:
            self._get_or_create_model(embedding_model_config, ModelRole.EMBEDDING)
        except Exception as e:
            logger.warning(
                "Failed to preload embedding model",
                error=str(e)
            )
        
        logger.info(
            "Model preloading completed",
            preloaded_count=len(self._model_cache),
            models=list(self._model_cache.keys())
        )