"""
Multi-model management for Databricks research agent.

This module provides flexible model configuration and management,
allowing different models to be used for different workflow nodes
while maintaining backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import os

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
    # Original roles (kept for backward compatibility)
    QUERY_GENERATION = "query_generation"
    WEB_RESEARCH = "web_research"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"
    EMBEDDING = "embedding"
    DEFAULT = "default"

    # New complexity-based roles
    SIMPLE = "simple"           # Quick tasks: classification, query generation, extraction
    ANALYTICAL = "analytical"   # Medium tasks: research, fact-checking, planning
    COMPLEX = "complex"         # Heavy tasks: final synthesis, deep analysis


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    endpoint: str
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout_seconds: int = 30
    max_retries: int = 3
    # Reasoning-specific configuration
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" for GPT-OSS models
    reasoning_budget: Optional[int] = None  # Token budget for reasoning (for hybrid models like Claude)

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
        # First check if we have directly configured models from _all_models
        if hasattr(self, '_all_models'):
            # Check for direct role mapping in config
            if role == ModelRole.SIMPLE and 'simple' in self._all_models:
                return self._all_models['simple']
            elif role == ModelRole.ANALYTICAL and 'analytical' in self._all_models:
                return self._all_models['analytical']
            elif role == ModelRole.COMPLEX and 'complex' in self._all_models:
                return self._all_models['complex']

        role_models = {
            ModelRole.QUERY_GENERATION: self.query_generation_model,
            ModelRole.WEB_RESEARCH: self.web_research_model,
            ModelRole.REFLECTION: self.reflection_model,
            ModelRole.SYNTHESIS: self.synthesis_model,
            ModelRole.EMBEDDING: self.embedding_model,
            ModelRole.DEFAULT: self.default_model,

            # Map complexity-based roles to specific models
            # SIMPLE uses query_generation model (or web_research as fallback)
            ModelRole.SIMPLE: self.query_generation_model or self.web_research_model,
            # ANALYTICAL uses web_research model (or default as fallback)
            ModelRole.ANALYTICAL: self.web_research_model or self.default_model,
            # COMPLEX uses synthesis model (or default as fallback)
            ModelRole.COMPLEX: self.synthesis_model or self.default_model,
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
    
    def __init__(
        self,
        config: Optional[NodeModelConfiguration] = None,
        model_selector: Optional['ModelSelector'] = None
    ):
        """
        Initialize the model manager.

        Args:
            config: Model configuration (uses defaults if None)
            model_selector: Optional ModelSelector for rate limiting
        """
        self.config = config or NodeModelConfiguration()
        self.config.validate()

        # Store model selector for rate limiting
        self.model_selector = model_selector

        # Store all model configs if available for dynamic role lookup
        if hasattr(config, '_all_models'):
            self._all_models = config._all_models
        else:
            self._all_models = {}

        # Cache for model instances
        self._model_cache: Dict[str, ChatDatabricks] = {}

        # Usage tracking
        self._usage_stats: Dict[str, int] = {}

        # Health status
        self._model_health: Dict[str, bool] = {}

        # Log initialization
        logger.info(
            f"ModelManager initialized | "
            f"Rate limiting: {'ENABLED' if model_selector else 'DISABLED'}"
        )
        
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
    
    def get_chat_model(self, model_name: str = "default"):
        """
        Get chat model with standardized tier-based naming.

        Tier Names (NEW STANDARD):
        - "simple": Lightweight (query gen, validation)
        - "analytical": Medium (planning, research, fact checking)
        - "complex": Heavy (report generation, tables)
        - "default": Alias for analytical

        Legacy Names (DEPRECATED):
        - "synthesis" → use "complex"
        - "web_research" → use "analytical"
        - "query_generation" → use "simple"
        - "reflection" → use "analytical"

        Args:
            model_name: Model/tier name

        Returns:
            Rate-limited chat model if ModelSelector available, else ChatDatabricks
        """
        # DEPRECATION: Map old functional names to new tier names
        DEPRECATED_ALIASES = {
            "synthesis": "complex",
            "web_research": "analytical",
            "query_generation": "simple",
            "reflection": "analytical",
            "default": "analytical",  # Safe default
        }

        original_name = model_name
        if model_name.lower() in DEPRECATED_ALIASES:
            model_name = DEPRECATED_ALIASES[model_name.lower()]
            if original_name != model_name:
                logger.warning(
                    f"⚠️ DEPRECATED: '{original_name}' → use '{model_name}' "
                    f"(tier-based naming)"
                )

        # Validate tier name
        valid_tiers = ["micro", "simple", "analytical", "complex", "structured"]
        if model_name not in valid_tiers:
            logger.error(f"❌ Invalid tier: {model_name}, using 'analytical'")
            model_name = "analytical"

        # Return rate-limited or plain model
        if self.model_selector:
            from .rate_limited_chat_model import RateLimitedChatModel

            logger.info(
                f"📦 Creating rate-limited model | "
                f"Tier: {model_name} | "
                f"Original name: {original_name}"
            )

            return RateLimitedChatModel(
                tier=model_name,
                model_selector=self.model_selector,
                operation=original_name  # Use original for operation tracking
            )
        else:
            logger.warning(
                f"⚠️ No ModelSelector available | "
                f"Returning plain ChatDatabricks (no rate limiting)"
            )
            # Fallback to legacy behavior
            role_mapping = {
                "simple": ModelRole.SIMPLE,
                "analytical": ModelRole.ANALYTICAL,
                "complex": ModelRole.COMPLEX,
            }
            role = role_mapping.get(model_name, ModelRole.DEFAULT)
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

            # Skip health check if we've already validated this endpoint recently
            # or if health checks are disabled (for performance)
            skip_health_check = (
                model_endpoint in self._model_health or
                os.environ.get("SKIP_MODEL_HEALTH_CHECKS", "false").lower() == "true"
            )

            if skip_health_check:
                # Trust the model is healthy if we've seen it before or checks are disabled
                self._model_cache[model_endpoint] = model
                self._model_health[model_endpoint] = True
                self._usage_stats[model_endpoint] = 1

                logger.info(
                    "Created and cached model instance (health check skipped)",
                    model=model_endpoint,
                    context=str(context)
                )

                return model

            # Test the model with a simple call (only for new endpoints)
            if self._test_model_health(model):
                self._model_cache[model_endpoint] = model
                self._model_health[model_endpoint] = True
                self._usage_stats[model_endpoint] = 1

                logger.info(
                    "Created and cached model instance (health check passed)",
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
        # Determine if this is a reasoning model that needs special configuration
        endpoint_name = model_config.endpoint.lower()
        is_reasoning_model = 'gpt-oss' in endpoint_name
        is_claude_model = 'claude' in endpoint_name

        # Configure extra_params based on model type and config
        extra_params = {}

        # Use configuration values if provided
        if model_config.reasoning_effort:
            extra_params["reasoning_effort"] = model_config.reasoning_effort
            logger.info(
                f"Using configured reasoning_effort={model_config.reasoning_effort} "
                f"for model: {model_config.endpoint}"
            )
        elif is_reasoning_model:
            # Smart default based on model size and usage context
            # 20b models default to low, 120b models default to medium
            # This provides a good balance between speed and quality
            if '20b' in endpoint_name:
                default_effort = "low"
            elif '120b' in endpoint_name:
                default_effort = "medium"  # Better default for larger models
            else:
                default_effort = "low"  # Conservative default

            extra_params["reasoning_effort"] = default_effort
            logger.info(
                f"Using default reasoning_effort={default_effort} for GPT-OSS model: {model_config.endpoint}"
            )

        # For models that support reasoning budget (future enhancement for Claude)
        if model_config.reasoning_budget:
            if is_claude_model:
                # Claude uses thinking configuration with budget_tokens
                extra_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": model_config.reasoning_budget
                }
                logger.info(
                    f"Using configured reasoning_budget={model_config.reasoning_budget} "
                    f"for Claude model: {model_config.endpoint}"
                )
            else:
                # For other models, could add as a parameter if supported
                logger.debug(
                    f"reasoning_budget configured but not applied for {model_config.endpoint} "
                    f"(not a Claude model)"
                )

        # Create the ChatDatabricks instance with optional extra_params
        # IMPORTANT: Set max_retries=0 to disable OpenAI SDK's built-in retry logic (which uses too-short delays like 0.5s, 1s, 2s)
        # Instead, we rely on ModelSelector's invoke_with_fallback which uses our configured delays (15s, 30s, 60s, etc.) from rate_limiting config
        return ChatDatabricks(
            endpoint=model_config.endpoint,
            temperature=getattr(model_config, 'temperature', 0.7),
            max_tokens=getattr(model_config, 'max_tokens', 4000),
            max_retries=0,  # CHANGED: Disable SDK retries (was: getattr(model_config, 'max_retries', 3)) - ModelSelector handles retries with proper delays
            extra_params=extra_params if extra_params else None
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