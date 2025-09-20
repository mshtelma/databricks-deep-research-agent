"""
Core module for the research agent.

This module provides foundational utilities, types, configuration management,
logging, and error handling for the research agent.
"""

# Import constants first (they may be used by other modules)
from ..constants import (
    AGENT_CONFIG_FILENAME,
    AGENT_CONFIG_BACKUP,
    AGENT_CONFIG_PATH,
    AGENT_CONFIG_OVERRIDE_ENV,
    CONFIG_SEARCH_ORDER
)

from .config import ConfigManager, get_default_config, get_config_from_dict, validate_configuration

# Add compatibility imports for unified config system
from .unified_config import get_config_manager as UnifiedConfigManager
from .unified_config import ToolConfigSchema

# Provide compatibility aliases
def create_config_manager(*args, **kwargs):
    """Compatibility function that creates unified config manager."""
    import warnings
    warnings.warn("ConfigManager is deprecated, use get_config_manager() instead", DeprecationWarning, stacklevel=2)
    if args or kwargs:
        return UnifiedConfigManager(override_config=args[0] if args else None, yaml_path=kwargs.get('yaml_path'))
    return UnifiedConfigManager()
from .exceptions import (
    ResearchAgentError,
    ConfigurationError,
    LLMEndpointError,
    SearchToolError,
    VectorSearchError,
    TavilySearchError,
    WorkflowError,
    MessageConversionError,
    ContentExtractionError,
    ToolInitializationError,
    ValidationError,
    TimeoutError,
    CircuitBreakerError,
    RetryExhaustedError,
)
from .logging import (
    AgentLogger,
    MLflowLogger,
    setup_logging,
    get_logger,
    log_execution_time,
    log_function_calls,
    default_logger,
)
from .types import (
    SearchResultType,
    AgentRole,
    WorkflowNodeType,
    ToolType,
    SearchResult,
    Citation,
    ResearchQuery,
    ResearchContext,
    AgentConfiguration,
    ToolConfiguration,
    SearchTaskState,
    DeduplicationMetrics,
    CoverageAnalysis,
    QueryComplexity,
    WorkflowMetrics,
    MessageList,
    ConfigDict,
    MetadataDict,
    OutputDict,
    ContentType,
    ResponseContent,
)
from .utils import (
    CircuitBreaker,
    URLResolver,
    retry_with_exponential_backoff,
    validate_url,
    sanitize_filename,
    extract_domain,
    generate_hash,
    safe_json_loads,
    safe_json_dumps,
    truncate_text,
    validate_required_fields,
    format_duration,
    parse_search_keywords,
    chunk_list,
    merge_dicts,
    timeout_after,
    insert_citation_markers,
    SecretResolver,
    resolve_secret,
    clear_secret_cache,
    get_secret_cache_status,
)

# Phase 2 components
from .databricks_embeddings import DatabricksEmbeddingClient, DatabricksEmbeddingError
from .embedding_manager import EmbeddingManager
from .model_manager import ModelManager, NodeModelConfiguration, ModelRole
from .deduplication import SemanticDeduplicator, SimilarityCluster, DeduplicationError
from .query_intelligence import QueryAnalyzer
from .result_evaluation import ResultEvaluator
from .adaptive_generation import AdaptiveQueryGenerator, AdaptationStrategy
from .section_models import SectionResearchResult

__version__ = "1.0.0"

__all__ = [
    # Constants
    "AGENT_CONFIG_FILENAME",
    "AGENT_CONFIG_BACKUP",
    "AGENT_CONFIG_PATH",
    "AGENT_CONFIG_OVERRIDE_ENV",
    "CONFIG_SEARCH_ORDER",
    
    # Config
    "ConfigManager",
    "get_default_config", 
    "get_config_from_dict",
    "validate_configuration",
    
    # Exceptions
    "ResearchAgentError",
    "ConfigurationError",
    "LLMEndpointError",
    "SearchToolError",
    "VectorSearchError",
    "TavilySearchError",
    "WorkflowError",
    "MessageConversionError",
    "ContentExtractionError",
    "ToolInitializationError",
    "ValidationError",
    "TimeoutError",
    "CircuitBreakerError",
    "RetryExhaustedError",
    
    # Logging
    "AgentLogger",
    "MLflowLogger",
    "setup_logging",
    "get_logger",
    "log_execution_time",
    "log_function_calls",
    "default_logger",
    
    # Types
    "SearchResultType",
    "AgentRole",
    "WorkflowNodeType",
    "ToolType",
    "SearchResult",
    "Citation",
    "ResearchQuery",
    "ResearchContext",
    "AgentConfiguration",
    "ToolConfiguration",
    "SearchTaskState",
    "DeduplicationMetrics",
    "CoverageAnalysis", 
    "QueryComplexity",
    "WorkflowMetrics",
    "MessageList",
    "ConfigDict",
    "MetadataDict",
    "OutputDict",
    "ContentType",
    "ResponseContent",
    
    # Utils
    "CircuitBreaker",
    "URLResolver",
    "retry_with_exponential_backoff",
    "validate_url",
    "sanitize_filename",
    "extract_domain",
    "generate_hash",
    "safe_json_loads",
    "safe_json_dumps",
    "truncate_text",
    "validate_required_fields",
    "format_duration",
    "parse_search_keywords",
    "chunk_list",
    "merge_dicts",
    "timeout_after",
    "insert_citation_markers",
    "SecretResolver",
    "resolve_secret",
    "clear_secret_cache",
    "get_secret_cache_status",
    
    # Phase 2 exports
    "DatabricksEmbeddingClient",
    "DatabricksEmbeddingError",
    "EmbeddingManager",
    "ModelManager",
    "NodeModelConfiguration", 
    "ModelRole",
    "SemanticDeduplicator",
    "SimilarityCluster",
    "DeduplicationError",
    "QueryAnalyzer",
    "ResultEvaluator",
    "AdaptiveQueryGenerator",
    "AdaptationStrategy",
    "SectionResearchResult",
]
