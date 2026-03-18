/**
 * Types, constants, and factory functions for data source E2E tests.
 *
 * Enterprise resource IDs are config-driven via environment variables
 * with defaults matching the integration test values.
 */

import { generateTestId } from './test-data';

// ---------------------------------------------------------------------------
// Types (mirror API contract from schemas/data_source.py)
// ---------------------------------------------------------------------------

export type DataSourceType = 'vector_search' | 'genie' | 'knowledge_assistant';
export type DataSourceVisibility = 'private' | 'workspace';
export type DataSourceValidationStatus = 'pending' | 'valid' | 'invalid' | 'expired';

export interface CreateVSRequest {
  name: string;
  endpoint_name: string;
  index_name: string;
  description?: string;
  visibility?: DataSourceVisibility;
  enable_hybrid?: boolean;
  enable_reranking?: boolean;
  num_results?: number;
}

export interface CreateGenieRequest {
  name: string;
  space_id: string;
  description?: string;
  example_questions?: string[];
  visibility?: DataSourceVisibility;
}

export interface CreateKARequest {
  name: string;
  endpoint_name: string;
  description?: string;
  pass_context?: boolean;
  visibility?: DataSourceVisibility;
}

export interface UpdateDataSourceRequest {
  name?: string;
  description?: string;
  visibility?: DataSourceVisibility;
  enable_hybrid?: boolean;
  enable_reranking?: boolean;
  num_results?: number;
  example_questions?: string[];
  pass_context?: boolean;
}

export interface DataSourceConfig {
  endpoint_name?: string;
  index_name?: string;
  columns?: string[];
  columns_to_rerank?: string[];
  enable_hybrid?: boolean;
  enable_reranking?: boolean;
  num_results?: number;
  space_id?: string;
  example_questions?: string[];
  pass_context?: boolean;
}

export interface DataSourceResponse {
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
  capabilities: string[];
  source_origin: string;
}

export interface DataSourceListResponse {
  sources: DataSourceResponse[];
  total: number;
  user_sources: number;
  workspace_sources: number;
}

export interface DataSourceValidationResponse {
  source_id: string;
  has_access: boolean;
  error_message: string | null;
  validated_at: string;
  detected_columns: string[] | null;
  detected_text_columns: string[] | null;
}

export interface QueryConfigResponse {
  source_id: string;
  config: Record<string, unknown>;
  validation: { is_valid: boolean; errors: string[]; warnings: string[] } | null;
}

export interface UpdateQueryConfigRequest {
  query_type?: string;
  num_results?: number;
  score_threshold?: number;
  columns?: string[];
  enable_reranking?: boolean;
  columns_to_rerank?: string[];
  filters?: Array<{ column: string; operator: string; value: unknown }>;
  filter_syntax?: string;
}

// ---------------------------------------------------------------------------
// Enterprise resource IDs (env-driven with integration test defaults)
// ---------------------------------------------------------------------------

export const VS_ENDPOINT =
  process.env.E2E_VS_ENDPOINT ?? 'databricks-gte-large-en';

export const VS_INDEX =
  process.env.E2E_VS_INDEX ?? 'anthony_ivan.demo-toolsapp.pdf_chunks_index';

export const GENIE_SPACE =
  process.env.E2E_GENIE_SPACE ?? '01f0b5ab5b841281858ae25da3f58125';

export const KA_ENDPOINT =
  process.env.E2E_KA_ENDPOINT ?? 'ka-99a12b9d-endpoint';

// ---------------------------------------------------------------------------
// Timeout Constants (milliseconds)
// ---------------------------------------------------------------------------

export const DS_TIMEOUTS = {
  /** API CRUD operations */
  api: 15_000,
  /** Validation operations (OBO round-trip) */
  validation: 30_000,
} as const;

// ---------------------------------------------------------------------------
// Factory Functions
// ---------------------------------------------------------------------------

/** Create a Vector Search source config with defaults. */
export function makeVSSource(namePrefix = 'E2E VS'): CreateVSRequest {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    endpoint_name: VS_ENDPOINT,
    index_name: VS_INDEX,
    description: 'E2E test Vector Search source',
    enable_hybrid: true,
    enable_reranking: true,
    num_results: 10,
  };
}

/** Create a Genie source config with defaults. */
export function makeGenieSource(namePrefix = 'E2E Genie'): CreateGenieRequest {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    space_id: GENIE_SPACE,
    description: 'E2E test Genie source',
    example_questions: ['What are the top 5 items?'],
  };
}

/** Create a Knowledge Assistant source config with defaults. */
export function makeKASource(namePrefix = 'E2E KA'): CreateKARequest {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    endpoint_name: KA_ENDPOINT,
    description: 'E2E test Knowledge Assistant source',
    pass_context: true,
  };
}
