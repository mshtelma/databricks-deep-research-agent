/**
 * TypeScript types for the Agent Designer Deployment Feature (Phase 1+2).
 *
 * Mirrors backend Pydantic schemas in
 * `src/deep_research/schemas/deployment.py`. Keep in sync.
 *
 * The four mode-specific config interfaces form a discriminated union on the
 * `mode` field. Use `DeploymentConfig` as the union type for any code that
 * dispatches by mode.
 */

export type DeploymentMode =
  | 'in_app'
  | 'shell_app'
  | 'mlflow_agent'
  | 'batch'

export type DeploymentStatus =
  | 'pending'
  | 'deploying'
  | 'active'
  | 'failed'
  | 'deactivated'
  | 'cleanup_failed'

// ---------------------------------------------------------------------------
// Per-mode configs (discriminated union)
// ---------------------------------------------------------------------------

export interface InAppDeploymentConfig {
  mode: 'in_app'
}

export interface ShellAppDeploymentConfig {
  mode: 'shell_app'
  /** Databricks App name, 2-30 chars, must start with `dr-shell-`. */
  app_name: string
  /** Git ref of databricks-deep-research-agent. */
  framework_git_tag: string
  /** Databricks bundle target. Defaults to `dev` server-side. */
  target?: string
  /** Optional Databricks secret scope containing BRAVE_API_KEY. */
  brave_secret_scope?: string | null
  /** Optional key within brave_secret_scope. */
  brave_secret_key?: string | null
}

export interface MlflowAgentDeploymentConfig {
  mode: 'mlflow_agent'
  uc_catalog: string
  uc_schema: string
  uc_model_name: string
  /** Optional override; must start with `dr-agent-` if set. */
  endpoint_name?: string
  env_overrides?: Record<string, string>
}

export interface BatchDeploymentConfig {
  mode: 'batch'
  /**
   * Serving endpoint to call via `ai_query`. May be a Mode 3 deployment name
   * OR any pre-existing serving endpoint -- decoupled from Mode 3 per plan
   * Section F.1.
   */
  target_endpoint: string
  /** 3-level Unity Catalog name: catalog.schema.table */
  input_table: string
  output_table: string
  prompt_column: string
  /** Optional `ai_query` responseFormat STRUCT spec. */
  response_format?: Record<string, unknown> | null
}

export type DeploymentConfig =
  | InAppDeploymentConfig
  | ShellAppDeploymentConfig
  | MlflowAgentDeploymentConfig
  | BatchDeploymentConfig

// ---------------------------------------------------------------------------
// Request / response shapes
// ---------------------------------------------------------------------------

export interface CreateDeploymentRequest {
  agent_id: string
  revision_id: string
  config: DeploymentConfig
}

export interface DeploymentResponse {
  id: string
  agent_id: string
  revision_id: string
  mode: DeploymentMode
  status: DeploymentStatus
  config: Record<string, unknown>
  endpoint_name: string | null
  model_name: string | null
  external_resource_ids: Record<string, unknown> | null
  error_message: string | null
  cleanup_attempts: number
  /** Set when DELETE has fired against a PENDING/DEPLOYING row;
   *  the worker resolves the cancel on its next heartbeat. */
  cancel_requested: boolean
  deployed_by: string
  created_at: string
  updated_at: string
  deactivated_at: string | null
}

export interface DeploymentListResponse {
  items: DeploymentResponse[]
  next_cursor: string | null
}

export interface DeploymentStatusResponse {
  status: DeploymentStatus
  updated_at: string
  error_message: string | null
  external_resource_ids: Record<string, unknown> | null
}

// ---------------------------------------------------------------------------
// Capability probes (split fast/slow per plan Section B.7)
// ---------------------------------------------------------------------------

export interface CanRunFastResponse {
  can_run: boolean
  reasons: string[]
}

export interface CanRunSlowResponse {
  can_run: boolean
  reasons: string[]
  cached: boolean
}

// ---------------------------------------------------------------------------
// Deletion guard error body (HTTP 409 from DELETE /agents-v2/{id})
// ---------------------------------------------------------------------------

export interface ActiveDeploymentSummary {
  id: string
  mode: DeploymentMode
  status: DeploymentStatus
  endpoint_name: string | null
}

export interface ActiveDeploymentsErrorResponse {
  error_kind: 'active_deployments_exist'
  active_count: number
  deployments: ActiveDeploymentSummary[]
  message: string
}

export interface DeploymentCleanupFailedErrorResponse {
  error_kind: 'deployment_cleanup_failed'
  message: string
  max_attempts?: number
}

// Surfaced when the agent-delete cascade leaves residual deployment rows that
// the FK ON DELETE RESTRICT still blocks (e.g., a new status the cascade
// hasn't learned to handle). Defense in depth against an opaque 500.
export interface DeploymentRowsBlockDeleteErrorResponse {
  error_kind: 'deployment_rows_block_delete'
  message: string
  blocking_deployments: ActiveDeploymentSummary[]
}

export interface DefaultRevisionNotDeployableErrorResponse {
  error_kind: 'default_revision_not_deployable'
  agent_id: string
  revision_id: string
  workflow_name: string
  root_child_summary: string[]
  message: string
}

// ---------------------------------------------------------------------------
// Filter shape for list endpoint
// ---------------------------------------------------------------------------

export interface DeploymentListFilters {
  mode?: DeploymentMode
  status?: DeploymentStatus
  /** Scope the list to deployments of a single agent.
   *  Server-side W9 authz still applies (deployer OR agent owner). */
  agent_id?: string
  cursor?: string
  limit?: number
}

// ---------------------------------------------------------------------------
// Deploy-here action error kinds
// ---------------------------------------------------------------------------

export type DeployHereErrorKind =
  | 'mode_does_not_support_inline_deploy'
  | 'deploy_already_in_progress'
  | 'missing_workspace_permission'
  | 'missing_obo_token'
  | 'artifact_too_large'
  | 'redeploy_requires_confirmation'
  | 'app_name_collision'
  | 'framework_tag_unreachable'
  | 'reachability_timeout'
  | 'reachability_failed'

export type DeployHereProbeStatus = 'ok' | 'denied' | 'unknown'

export interface CanDeployHereResponse {
  can_deploy: boolean
  reason: DeployHereErrorKind | null
  /** Optional for compatibility with app builds served by older backends. */
  probe_status?: DeployHereProbeStatus
  actor: 'obo' | 'sp_fallback'
}

// Terminal statuses — mirror backend `TERMINAL_STATUSES` (poll-stop set).
// Used by `useDeploymentStatusPoll` to stop refetching once the deployment
// has reached an end state. NOTE: `failed` is intentionally included so the
// UI does not poll forever on failed deploys (W2 of the fix plan). Audit
// trail is preserved server-side because the backend's separate
// `DELETABLE_STATUSES` excludes `failed`.
export const TERMINAL_STATUSES: ReadonlySet<DeploymentStatus> = new Set([
  'deactivated',
  'cleanup_failed',
  'failed',
])
