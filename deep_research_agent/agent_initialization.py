"""
Initialization components for the research agent.

This module handles LLM and Phase 2 component initialization.
"""

from typing import Optional

from deep_research_agent.core import (
    get_logger,
    LLMEndpointError,
    DatabricksEmbeddingClient,
    ModelManager,
    NodeModelConfiguration,
    SemanticDeduplicator,
    QueryAnalyzer,
    ResultEvaluator,
    AdaptiveQueryGenerator
)

try:
    from databricks_langchain import ChatDatabricks
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    ChatDatabricks = None

logger = get_logger(__name__)


class AgentInitializer:
    """Handles initialization of agent components."""
    
    @staticmethod
    def initialize_llm(agent_config, temperature=None, max_tokens=None):
        """Initialize the language model."""
        if not DATABRICKS_AVAILABLE:
            raise LLMEndpointError("databricks_langchain not available")
        
        try:
            llm = ChatDatabricks(
                endpoint=agent_config.llm_endpoint,
                temperature=temperature or agent_config.temperature,
                max_tokens=max_tokens or agent_config.max_tokens
            )
            
            logger.info("Initialized LLM", endpoint=agent_config.llm_endpoint)
            return llm
        except Exception as e:
            logger.error("Failed to initialize LLM", error=e)
            raise LLMEndpointError(f"Failed to initialize LLM: {e}")
    
    @staticmethod
    def initialize_phase2_components(config_manager):
        """Initialize Phase 2 components for enhanced research capabilities.
        
        Returns:
            tuple: (embedding_client, model_manager, deduplicator, query_analyzer, 
                   result_evaluator, adaptive_generator)
        """
        try:
            # Load model configuration from YAML
            model_config_dict = config_manager.get_model_config()
            databricks_config = config_manager.get_databricks_config()
            
            # Initialize embedding client for deduplication and evaluation
            logger.debug("Initializing embedding client...")
            embedding_config = model_config_dict.get("embedding", {"endpoint": "databricks-bge-large-en"})
            embedding_endpoint = embedding_config.get("endpoint") if isinstance(embedding_config, dict) else embedding_config
            
            embedding_client = DatabricksEmbeddingClient(
                endpoint_name=embedding_endpoint,
                workspace_url=databricks_config.get("workspace_url"),
                token=databricks_config.get("token"),
                profile=databricks_config.get("workspace_profile")
            )
            logger.debug("Embedding client initialized successfully")
            
            # Initialize model manager for per-node model configuration
            logger.debug("Initializing model manager...")
            from deep_research_agent.core.model_manager import ModelConfig
            
            def create_model_config(model_dict):
                """Create ModelConfig from dictionary if not None."""
                if model_dict is None:
                    return None
                if isinstance(model_dict, str):
                    return ModelConfig(endpoint=model_dict)
                return ModelConfig(
                    endpoint=model_dict.get("endpoint"),
                    temperature=model_dict.get("temperature", 0.7),
                    max_tokens=model_dict.get("max_tokens", 4000),
                    timeout_seconds=model_dict.get("timeout_seconds", 30),
                    max_retries=model_dict.get("max_retries", 3)
                )
            
            model_config = NodeModelConfiguration(
                default_model=create_model_config(model_config_dict.get("default", {"endpoint": "databricks-claude-3-7-sonnet"})),
                query_generation_model=create_model_config(model_config_dict.get("query_generation")),
                web_research_model=create_model_config(model_config_dict.get("web_research")),
                reflection_model=create_model_config(model_config_dict.get("reflection")),
                synthesis_model=create_model_config(model_config_dict.get("synthesis")),
                embedding_model=create_model_config(model_config_dict.get("embedding", {"endpoint": "databricks-bge-large-en"}))
            )
            model_manager = ModelManager(model_config)
            logger.debug("Model manager initialized successfully")
            
            # Initialize semantic deduplicator
            logger.debug("Initializing semantic deduplicator...")
            deduplicator = SemanticDeduplicator(
                embedding_client=embedding_client,
                similarity_threshold=0.85,  # Conservative threshold
                min_content_length=50
            )
            logger.debug("Semantic deduplicator initialized successfully")
            
            # Initialize query analyzer
            logger.debug("Initializing query analyzer...")
            query_analyzer = QueryAnalyzer()
            logger.debug("Query analyzer initialized successfully")
            
            # Initialize result evaluator
            logger.debug("Initializing result evaluator...")
            result_evaluator = ResultEvaluator(
                embedding_client=embedding_client,
                query_analyzer=query_analyzer,
                coverage_threshold=0.7
            )
            logger.debug("Result evaluator initialized successfully")
            
            # Initialize adaptive query generator
            logger.debug("Initializing adaptive query generator...")
            adaptive_generator = AdaptiveQueryGenerator(
                query_analyzer=query_analyzer,
                result_evaluator=result_evaluator,
                max_adaptive_queries=3
            )
            logger.debug("Adaptive query generator initialized successfully")
            
            logger.info("Initialized Phase 2 components successfully")
            
            return (
                embedding_client,
                model_manager,
                deduplicator,
                query_analyzer,
                result_evaluator,
                adaptive_generator
            )
            
        except Exception as e:
            import traceback
            logger.warning(f"Failed to initialize some Phase 2 components: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            # Return fallback components to prevent failures
            return (
                None,  # embedding_client
                None,  # model_manager
                None,  # deduplicator
                QueryAnalyzer(),  # query_analyzer - doesn't require external dependencies
                None,  # result_evaluator
                None   # adaptive_generator
            )