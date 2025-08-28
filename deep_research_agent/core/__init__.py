"""
Core module for the research agent.

This module provides foundational utilities, types, configuration management,
logging, and error handling for the research agent.
"""

from .config import ConfigManager, get_default_config, get_config_from_dict, validate_configuration
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
from .model_manager import ModelManager, NodeModelConfiguration, ModelRole
from .deduplication import SemanticDeduplicator, SimilarityCluster, DeduplicationError
from .query_intelligence import QueryAnalyzer
from .result_evaluation import ResultEvaluator
from .adaptive_generation import AdaptiveQueryGenerator, AdaptationStrategy

__version__ = "1.0.0"

__all__ = [
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
]