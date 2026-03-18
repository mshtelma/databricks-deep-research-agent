/**
 * Discovery types for automatic data source discovery.
 *
 * Corresponds to backend schemas in:
 * - src/deep_research/schemas/discovery.py
 * - src/deep_research/schemas/query_config.py
 *
 * API Contract: /specs/007-enterprise-data-sources/contracts/discovery.yaml
 */

// =============================================================================
// Enums
// =============================================================================

export type DataSourceType =
  | 'vector_search'
  | 'genie'
  | 'knowledge_assistant'
  | 'web_search'
  | 'uploaded_file'
  | 'custom';

export type DiscoveryStatus = 'ready' | 'syncing' | 'unavailable' | 'error';

export type QueryType = 'ANN' | 'HYBRID' | 'FULL_TEXT';

export type FilterOperator =
  | '='
  | '!='
  | '<'
  | '<='
  | '>'
  | '>='
  | 'LIKE'
  | 'NOT LIKE'
  | 'IN';

export type FilterSyntax = 'sql' | 'dict';

// =============================================================================
// Type-Specific Metadata
// =============================================================================

export interface FilterColumnInfo {
  name: string;
  data_type: 'string' | 'integer' | 'float' | 'timestamp' | 'boolean';
  operators: string[];
}

export interface VectorSearchMetadata {
  index_name: string;
  endpoint_name: string;
  primary_key: string;
  index_type: 'DELTA_SYNC' | 'DIRECT_ACCESS';
  embedding_columns: string[];
  embedding_dimension?: number;
  embedding_model?: string;
  filter_columns: FilterColumnInfo[];
  supported_query_types: QueryType[];
  supports_reranking: boolean;
  row_count?: number;
  is_ready: boolean;
}

export interface GenieSpaceMetadata {
  space_id: string;
  title: string;
  description?: string;
  warehouse_id?: string;
  owner?: string;
  created_at?: string;
  capabilities: string[];
}

export interface ServingEndpointMetadata {
  endpoint_name: string;
  endpoint_type: string;
  state: 'READY' | 'NOT_READY' | 'PENDING';
  tags: Record<string, string>;
  is_knowledge_assistant: boolean;
  assistant_type?: string;
  creator?: string;
}

// =============================================================================
// Core Discovery Models
// =============================================================================

export interface DiscoveredSource {
  source_id: string;
  source_type: DataSourceType;
  name: string;
  endpoint_name: string;
  description?: string;
  status: DiscoveryStatus;
  capabilities: string[];
  metadata: Record<string, unknown>;
  discovered_at: string;
  cached_until?: string;
}

export interface DiscoveryError {
  source_type: DataSourceType;
  error_code: string;
  error_message: string;
  retryable: boolean;
}

export interface DiscoveryResponse {
  sources: DiscoveredSource[];
  total_count: number;
  by_type: Record<string, DiscoveredSource[]>;
  discovered_at: string;
  cached: boolean;
  cache_expires_at?: string;
  errors?: DiscoveryError[];
}

export interface SourceMetadataResponse {
  source: DiscoveredSource;
  vector_search?: VectorSearchMetadata;
  genie?: GenieSpaceMetadata;
  serving_endpoint?: ServingEndpointMetadata;
  saved_config?: VectorSearchQueryConfig;
}

// =============================================================================
// Query Configuration
// =============================================================================

export interface FilterExpression {
  column: string;
  operator: FilterOperator;
  value: string | number | (string | number)[];
}

export interface VectorSearchQueryConfig {
  query_type: QueryType;
  num_results: number;
  score_threshold?: number;
  columns?: string[];
  enable_reranking: boolean;
  columns_to_rerank?: string[];
  filters: FilterExpression[];
  filter_syntax: FilterSyntax;
}

export interface QueryConfigValidationResult {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface QueryConfigResponse {
  source_id: string;
  config: VectorSearchQueryConfig;
  validation?: QueryConfigValidationResult;
}

// =============================================================================
// Request Types
// =============================================================================

export interface RefreshDiscoveryRequest {
  source_types?: DataSourceType[];
}

export interface UpdateQueryConfigRequest {
  query_type?: QueryType;
  num_results?: number;
  score_threshold?: number;
  columns?: string[];
  enable_reranking?: boolean;
  columns_to_rerank?: string[];
  filters?: FilterExpression[];
  filter_syntax?: FilterSyntax;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get display label for a data source type.
 */
export function getSourceTypeLabel(type: DataSourceType): string {
  const labels: Record<DataSourceType, string> = {
    vector_search: 'Vector Search',
    genie: 'Genie',
    knowledge_assistant: 'Knowledge Assistant',
    web_search: 'Web Search',
    uploaded_file: 'Uploaded File',
    custom: 'Custom',
  };
  return labels[type] || type;
}

/**
 * Get icon name for a data source type.
 */
export function getSourceTypeIcon(type: DataSourceType): string {
  const icons: Record<DataSourceType, string> = {
    vector_search: 'database',
    genie: 'sparkles',
    knowledge_assistant: 'user-circle',
    web_search: 'globe',
    uploaded_file: 'document',
    custom: 'puzzle',
  };
  return icons[type] || 'question-mark-circle';
}

/**
 * Get display label for discovery status.
 */
export function getStatusLabel(status: DiscoveryStatus): string {
  const labels: Record<DiscoveryStatus, string> = {
    ready: 'Ready',
    syncing: 'Syncing',
    unavailable: 'Unavailable',
    error: 'Error',
  };
  return labels[status] || status;
}

/**
 * Get color class for discovery status.
 */
export function getStatusColor(status: DiscoveryStatus): string {
  const colors: Record<DiscoveryStatus, string> = {
    ready: 'text-green-600',
    syncing: 'text-yellow-600',
    unavailable: 'text-gray-500',
    error: 'text-red-600',
  };
  return colors[status] || 'text-gray-500';
}

/**
 * Get display label for query type.
 */
export function getQueryTypeLabel(type: QueryType): string {
  const labels: Record<QueryType, string> = {
    ANN: 'Vector Search (ANN)',
    HYBRID: 'Hybrid (Vector + Keyword)',
    FULL_TEXT: 'Full-Text (Keyword)',
  };
  return labels[type] || type;
}

/**
 * Get description for query type.
 */
export function getQueryTypeDescription(type: QueryType): string {
  const descriptions: Record<QueryType, string> = {
    ANN: 'Fast approximate nearest neighbor search using vector embeddings',
    HYBRID:
      'Combines vector similarity with keyword matching for better precision (max 200 results)',
    FULL_TEXT: 'Traditional keyword-based search without vectors (max 200 results, beta)',
  };
  return descriptions[type] || '';
}

/**
 * Parse type-specific metadata from a discovered source.
 */
export function parseSourceMetadata(source: DiscoveredSource): {
  vectorSearch?: VectorSearchMetadata;
  genie?: GenieSpaceMetadata;
  servingEndpoint?: ServingEndpointMetadata;
} {
  const result: {
    vectorSearch?: VectorSearchMetadata;
    genie?: GenieSpaceMetadata;
    servingEndpoint?: ServingEndpointMetadata;
  } = {};

  if (source.source_type === 'vector_search') {
    result.vectorSearch = source.metadata as unknown as VectorSearchMetadata;
  } else if (source.source_type === 'genie') {
    result.genie = source.metadata as unknown as GenieSpaceMetadata;
  } else if (source.source_type === 'knowledge_assistant') {
    result.servingEndpoint = source.metadata as unknown as ServingEndpointMetadata;
  }

  return result;
}

/**
 * Create default query config for a source.
 */
export function createDefaultQueryConfig(
  metadata?: VectorSearchMetadata
): VectorSearchQueryConfig {
  return {
    query_type: 'ANN',
    num_results: 10,
    enable_reranking: false,
    filters: [],
    filter_syntax: 'sql',
    ...(metadata?.supports_reranking && {
      enable_reranking: true,
      columns_to_rerank: metadata.filter_columns
        .filter((c) => c.data_type === 'string')
        .map((c) => c.name)
        .slice(0, 3), // Default to first 3 text columns
    }),
  };
}
