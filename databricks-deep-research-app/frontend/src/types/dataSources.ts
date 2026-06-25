/**
 * Data Source types for enterprise data source management.
 *
 * This module defines types for:
 * - Data source definitions (Vector Search, Genie, Knowledge Assistant)
 * - Source scope configuration
 * - Genie query results
 * - Plan review workflow
 */

/** Data source types */
export type DataSourceType =
  | 'vector_search'
  | 'delta_table'
  | 'genie'
  | 'knowledge_assistant'
  | 'web_search'
  | 'uploaded_file'
  | 'mcp_server'
  | 'custom';

/** Data source visibility options */
export type DataSourceVisibility = 'private' | 'workspace';

/** Validation status for data sources */
export type ValidationStatus = 'pending' | 'valid' | 'invalid' | 'expired';

/** Source scope options */
export type SourceScope = 'enterprise_only' | 'web_only' | 'all';

/** Base data source configuration */
export interface DataSourceConfig {
  [key: string]: unknown;
}

/** Vector Search specific configuration */
export interface VectorSearchConfig extends DataSourceConfig {
  endpoint_name: string;
  index_name: string;
  column_schema?: Record<string, string>;
  filterable_columns?: string[];
  num_results?: number;
  enable_reranking?: boolean;
  query_type?: 'ann' | 'hybrid';
}

/** Genie specific configuration */
export interface GenieConfig extends DataSourceConfig {
  space_id: string;
  description?: string;
  example_questions?: string[];
  max_rows?: number;
}

/** Knowledge Assistant specific configuration */
export interface KnowledgeAssistantConfig extends DataSourceConfig {
  endpoint_name: string;
  description?: string;
  pass_context?: boolean;
}

/** User data source definition */
export interface UserDataSource {
  id: string;
  ownerId: string;
  type: DataSourceType;
  name: string;
  description: string | null;
  endpointIdentifier: string | null;
  config: DataSourceConfig;
  visibility: DataSourceVisibility;
  validationStatus: ValidationStatus;
  lastValidatedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

/** Request to create a Vector Search data source */
export interface CreateVectorSearchSourceRequest {
  name: string;
  description?: string;
  endpoint_name: string;
  index_name: string;
  visibility?: DataSourceVisibility;
  num_results?: number;
  enable_reranking?: boolean;
  query_type?: 'ann' | 'hybrid';
}

/** Request to create a Genie data source */
export interface CreateGenieSourceRequest {
  name: string;
  description?: string;
  space_id: string;
  example_questions?: string[];
  visibility?: DataSourceVisibility;
  max_rows?: number;
}

/** Request to create a Knowledge Assistant data source */
export interface CreateKnowledgeAssistantSourceRequest {
  name: string;
  description?: string;
  endpoint_name: string;
  pass_context?: boolean;
  visibility?: DataSourceVisibility;
}

/** Union type for all create requests */
export type CreateDataSourceRequest =
  | CreateVectorSearchSourceRequest
  | CreateGenieSourceRequest
  | CreateKnowledgeAssistantSourceRequest;

/** API response for data source */
export interface DataSourceResponse extends UserDataSource {}

/** API response for data source list */
export interface DataSourceListResponse {
  items: UserDataSource[];
  sources: DataSource[];
  total: number;
  user_sources: number;
  workspace_sources: number;
}

/** Validation result */
export interface DataSourceValidationResult {
  isValid: boolean;
  message: string;
  details?: Record<string, unknown>;
}

// =============================================================================
// Genie Query Results
// =============================================================================

/** Column definition in Genie result */
export interface GenieColumn {
  name: string;
  type: string;
}

/** Genie query result row */
export type GenieRow = Record<string, unknown>;

/** Genie query result */
export interface GenieResult {
  columns: GenieColumn[];
  rows: GenieRow[];
  totalRows: number;
  truncated: boolean;
  generatedSql: string | null;
  narrativeSummary: string | null;
  executionTimeMs: number | null;
}

// =============================================================================
// Knowledge Assistant Results
// =============================================================================

/** Confidence level for assistant responses */
export type AssistantConfidenceLevel = 'high' | 'medium' | 'low';

/** Source reference from knowledge assistant */
export interface AssistantSourceReference {
  title: string;
  url?: string | null;
  snippet?: string | null;
}

/** Knowledge Assistant response */
export interface KnowledgeAssistantResult {
  answer: string;
  confidenceLevel: AssistantConfidenceLevel;
  sources: AssistantSourceReference[];
  includedContext: boolean;
}

// =============================================================================
// Source Scope Configuration
// =============================================================================

/** Source scope configuration for research */
export interface SourceScopeConfig {
  scope: SourceScope;
  enabledSources: string[];
  disabledSources: string[];
}

/** Available source for selection */
export interface AvailableSource {
  id: string;
  name: string;
  type: DataSourceType;
  description: string | null;
  relevanceHint?: string;
  isEnabled: boolean;
}

// =============================================================================
// Plan Review Types
// =============================================================================

/** Source hint for a plan step */
export interface StepSourceHint {
  sourceName: string;
  sourceType: DataSourceType;
  priority: 1 | 2 | 3;
  queryHint?: string;
  filters?: Record<string, unknown>;
}

/** Plan step with source hints */
export interface PlanStepWithSources {
  id: string;
  title: string;
  description: string;
  stepType: 'research' | 'analysis';
  needsSearch: boolean;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  sourceHints: StepSourceHint[];
  excludeSources?: string[];
  requireAllSources?: boolean;
}

/** Plan with source hints for review */
export interface PlanWithSources {
  id: string;
  title: string;
  thought: string;
  steps: PlanStepWithSources[];
  iteration: number;
  createdAt: string;
}

/** Edited plan from user */
export interface EditedPlan {
  steps: PlanStepWithSources[];
}

/** Plan review event from SSE */
export interface PlanReviewEvent {
  eventType: 'plan_review';
  timestamp: string;
  planId: string;
  plan: PlanWithSources;
  timeoutSeconds: number;
}

/** Plan review response options */
export type PlanReviewAction = 'approve' | 'approve_with_edits' | 'reject';

// =============================================================================
// Display Utilities
// =============================================================================

/** Human-readable labels for data source types */
export const DATA_SOURCE_TYPE_LABELS: Record<DataSourceType, string> = {
  vector_search: 'Vector Search',
  delta_table: 'Delta Table',
  genie: 'Genie',
  knowledge_assistant: 'Knowledge Assistant',
  web_search: 'Web Search',
  uploaded_file: 'Uploaded File',
  mcp_server: 'MCP Server',
  custom: 'Custom',
};

/** Human-readable labels for source scope */
export const SOURCE_SCOPE_LABELS: Record<SourceScope, string> = {
  enterprise_only: 'Enterprise Only',
  web_only: 'Web Only',
  all: 'All Sources',
};

/** Color mapping for confidence levels */
export const CONFIDENCE_COLORS: Record<AssistantConfidenceLevel, string> = {
  high: 'green',
  medium: 'amber',
  low: 'red',
};

/** Icons/colors for source types */
export const SOURCE_TYPE_COLORS: Record<DataSourceType, string> = {
  vector_search: 'blue',
  delta_table: 'cyan',
  genie: 'purple',
  knowledge_assistant: 'emerald',
  web_search: 'orange',
  uploaded_file: 'gray',
  mcp_server: 'indigo',
  custom: 'slate',
};

// =============================================================================
// Type Aliases for API/Hooks Compatibility
// =============================================================================

/** Alias for CreateVectorSearchSourceRequest */
export type CreateVectorSearchRequest = CreateVectorSearchSourceRequest;

/** Alias for CreateGenieSourceRequest */
export type CreateGenieRequest = CreateGenieSourceRequest;

/** Alias for CreateKnowledgeAssistantSourceRequest */
export type CreateKnowledgeAssistantRequest = CreateKnowledgeAssistantSourceRequest;

/** Request to update a data source */
export interface UpdateDataSourceRequest {
  name?: string;
  description?: string;
  visibility?: DataSourceVisibility;
  enable_hybrid?: boolean;
  enable_reranking?: boolean;
  num_results?: number;
  example_questions?: string[];
  pass_context?: boolean;
  query_type?: 'ann' | 'hybrid';
  max_rows?: number;
}

/** Request to validate connection before creating */
export interface ValidateConnectionRequest {
  type: DataSourceType;
  endpoint_name?: string;
  index_name?: string;
  space_id?: string;
}

/** Response from connection validation */
export interface ValidateConnectionResponse {
  has_access: boolean;
  error_message: string | null;
  detected_columns?: string[];
  detected_text_columns?: string[];
}

/** Validation status type alias */
export type DataSourceValidationStatus = ValidationStatus;

/** Data source capability type */
export type DataSourceCapability =
  | 'semantic_search'
  | 'keyword_search'
  | 'metadata_filtering'
  | 'sql_analytics'
  | 'aggregations'
  | 'follow_up'
  | 'domain_expertise'
  | 'current_events'
  | 'document_search';

/**
 * Full data source response from API.
 * Compatible with both UserDataSource and the API response format.
 */
export interface DataSource {
  id: string;
  owner_id: string;
  type: DataSourceType;
  name: string;
  description: string | null;
  endpoint_identifier: string;
  config: DataSourceConfig;
  visibility: DataSourceVisibility;
  validation_status: DataSourceValidationStatus;
  last_validated_at: string | null;
  created_at: string;
  updated_at: string;
  capabilities: DataSourceCapability[];
  source_origin: 'system' | 'plugin' | 'user';
}

/** Response from data source validation */
export interface DataSourceValidationResponse {
  source_id: string;
  has_access: boolean;
  error_message: string | null;
  validated_at: string;
  detected_columns?: string[];
  detected_text_columns?: string[];
}

// Helper functions
export function getSourceTypeLabel(type: DataSourceType): string {
  return DATA_SOURCE_TYPE_LABELS[type] || type;
}

export function getCapabilityLabel(capability: DataSourceCapability): string {
  const labels: Record<DataSourceCapability, string> = {
    semantic_search: 'Semantic Search',
    keyword_search: 'Keyword Search',
    metadata_filtering: 'Metadata Filtering',
    sql_analytics: 'SQL Analytics',
    aggregations: 'Aggregations',
    follow_up: 'Follow-up Questions',
    domain_expertise: 'Domain Expertise',
    current_events: 'Current Events',
    document_search: 'Document Search',
  };
  return labels[capability] || capability;
}

export function getValidationStatusColor(
  status: DataSourceValidationStatus
): 'green' | 'yellow' | 'red' | 'gray' {
  const colors: Record<DataSourceValidationStatus, 'green' | 'yellow' | 'red' | 'gray'> = {
    valid: 'green',
    pending: 'yellow',
    expired: 'yellow',
    invalid: 'red',
  };
  return colors[status] || 'gray';
}

export function getValidationStatusLabel(status: DataSourceValidationStatus): string {
  const labels: Record<DataSourceValidationStatus, string> = {
    valid: 'Valid',
    pending: 'Pending',
    expired: 'Expired',
    invalid: 'Invalid',
  };
  return labels[status] || status;
}
