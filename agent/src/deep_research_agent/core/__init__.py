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

# Compatibility aliases removed - deprecated create_config_manager function eliminated
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
    EnrichedSearchResult,
    FetchingConfig,
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

# State capture for testing
from .state_capture import state_capture

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
    "EnrichedSearchResult",
    "FetchingConfig",

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

    # State capture
    "state_capture",
]
