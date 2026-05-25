/**
 * Single source of truth for deployment status & mode display + the
 * status-x-mode action-label matrix. Consumed by StatusPanel, DeploymentRow,
 * DeploymentsSection, and UndeployConfirmDialog so every surface stays in
 * sync. Adding a new DeploymentStatus or DeploymentMode forces a TypeScript
 * compile error in `getAction` via `assertNever`.
 */

import type {
  DeploymentMode,
  DeploymentResponse,
  DeploymentStatus,
} from '@/types/deployment'

// ---------------------------------------------------------------------------
// Color + label tokens
// ---------------------------------------------------------------------------

export const STATUS_COLOR: Record<DeploymentStatus, string> = {
  pending: 'bg-yellow-500/15 text-yellow-700 border-yellow-300',
  deploying: 'bg-blue-500/15 text-blue-700 border-blue-300',
  active: 'bg-green-500/15 text-green-700 border-green-300',
  failed: 'bg-red-500/15 text-red-700 border-red-300',
  deactivated: 'bg-zinc-500/15 text-zinc-700 border-zinc-300',
  cleanup_failed: 'bg-zinc-500/15 text-zinc-700 border-zinc-300',
}

export const STATUS_LABEL: Record<DeploymentStatus, string> = {
  pending: 'Pending',
  deploying: 'Deploying',
  active: 'Active',
  failed: 'Failed',
  deactivated: 'Deactivated',
  cleanup_failed: 'Cleanup failed',
}

/** Background classes for the per-mode chip (mirrors DeployDropdown tones). */
export const MODE_COLOR: Record<DeploymentMode, string> = {
  in_app: 'bg-db-blue-100 text-db-blue-700',
  shell_app: 'bg-db-lava-100 text-db-lava-700',
  mlflow_agent: 'bg-db-oat-medium text-db-navy-800',
  batch: 'bg-db-green-300 text-db-navy-800',
}

export const MODE_LABEL: Record<DeploymentMode, string> = {
  in_app: 'In-App',
  shell_app: 'Shell App',
  mlflow_agent: 'MLflow Agent',
  batch: 'Batch',
}

// ---------------------------------------------------------------------------
// Action resolver
// ---------------------------------------------------------------------------

export type DeploymentActionKind =
  | 'cancel'
  | 'undeploy'
  | 'unregister'
  | 'cleanup'
  | 'retry-cleanup'

export interface DeploymentAction {
  kind: DeploymentActionKind
  label: string
  /** All current kinds are destructive — future-proofs styling. */
  destructive: boolean
}

const ACTION_LABEL: Record<DeploymentActionKind, string> = {
  cancel: 'Cancel',
  undeploy: 'Undeploy',
  unregister: 'Unregister',
  cleanup: 'Clean up',
  'retry-cleanup': 'Retry cleanup',
}

function action(kind: DeploymentActionKind): DeploymentAction {
  return { kind, label: ACTION_LABEL[kind], destructive: true }
}

function assertNever(x: never): never {
  throw new Error(`Unhandled deployment variant: ${JSON.stringify(x)}`)
}

/**
 * Returns the action that should be presented to the user, or `null` when no
 * action is meaningful (DEACTIVATED rows are inert).
 *
 * Matrix:
 *   pending / deploying    → Cancel
 *   active + in_app        → Unregister
 *   active + other         → Undeploy
 *   failed                 → Clean up
 *   cleanup_failed         → Retry cleanup
 *   deactivated            → null
 */
export function getAction(d: DeploymentResponse): DeploymentAction | null {
  switch (d.status) {
    case 'pending':
    case 'deploying':
      return action('cancel')
    case 'active':
      switch (d.mode) {
        case 'in_app':
          return action('unregister')
        case 'shell_app':
        case 'mlflow_agent':
        case 'batch':
          return action('undeploy')
        default:
          return assertNever(d.mode)
      }
    case 'failed':
      return action('cleanup')
    case 'cleanup_failed':
      return action('retry-cleanup')
    case 'deactivated':
      return null
    default:
      return assertNever(d.status)
  }
}

// ---------------------------------------------------------------------------
// Status-label derivations
// ---------------------------------------------------------------------------

/**
 * Effective status label that overlays `cancel_requested` so the user sees
 * "Cancelling…" instead of "Pending" / "Deploying" while the worker has
 * acknowledged the cancel but not yet landed the row.
 */
export function getEffectiveStatusLabel(d: DeploymentResponse): string {
  if (d.cancel_requested && (d.status === 'pending' || d.status === 'deploying')) {
    return 'Cancelling…'
  }
  return STATUS_LABEL[d.status]
}

// ---------------------------------------------------------------------------
// Typed accessor for resource identifiers
// ---------------------------------------------------------------------------

/**
 * Read a string-shaped resource identifier from `external_resource_ids` first,
 * falling back to `config`. Returns null if the key is absent or non-string.
 * Never throws — defense-in-depth against future per-mode key renames.
 */
export function readResourceId(
  d: DeploymentResponse,
  key: string,
): string | null {
  const fromExternal = d.external_resource_ids?.[key]
  if (typeof fromExternal === 'string' && fromExternal.length > 0) {
    return fromExternal
  }
  const fromConfig = d.config?.[key]
  if (typeof fromConfig === 'string' && fromConfig.length > 0) {
    return fromConfig
  }
  return null
}

// ---------------------------------------------------------------------------
// Per-row resource summary
// ---------------------------------------------------------------------------

/**
 * One-line resource summary for the row's middle column. Mode-specific but
 * driven only by the (mode, payload) signature — no hardcoded examples.
 */
export function getResourceSummary(d: DeploymentResponse): string {
  switch (d.mode) {
    case 'shell_app':
      return readResourceId(d, 'app_name') ?? '(app name pending)'
    case 'mlflow_agent':
      return (
        d.endpoint_name
        ?? readResourceId(d, 'endpoint_name')
        ?? readResourceId(d, 'endpoint_name_override')
        ?? '(endpoint pending)'
      )
    case 'in_app':
      return 'Chat picker'
    case 'batch':
      return 'Lakeflow pipeline (stub)'
    default:
      return assertNever(d.mode)
  }
}

// ---------------------------------------------------------------------------
// Confirm-dialog impact text
// ---------------------------------------------------------------------------

/**
 * Mode-specific destructive-consequence sentence shown in the confirm modal.
 * Honest about the in_app multi-deployment case and the batch Phase-3 stub.
 */
export function getImpactText(d: DeploymentResponse): string {
  if (d.status === 'pending' || d.status === 'deploying') {
    return 'Cancel this in-flight deployment? The worker will roll back any partially-created resources. The agent itself is preserved.'
  }
  switch (d.mode) {
    case 'shell_app': {
      const appName = readResourceId(d, 'app_name')
      const label = appName ? `\`${appName}\`` : 'the Databricks App'
      return `Databricks App ${label} will be deleted and its workspace files removed. The agent and its revisions are preserved.`
    }
    case 'mlflow_agent': {
      const endpoint = d.endpoint_name
        ?? readResourceId(d, 'endpoint_name')
        ?? readResourceId(d, 'endpoint_name_override')
      const modelName = d.model_name ?? readResourceId(d, 'uc_model')
      const endpointLabel = endpoint ? `\`${endpoint}\`` : 'the serving endpoint'
      const modelLabel = modelName ? ` and UC model version \`${modelName}\` archived` : ''
      return `Serving endpoint ${endpointLabel} will be deleted${modelLabel}. The agent is preserved.`
    }
    case 'in_app':
      return 'The agent will be unregistered from the chat picker (visibility flips to private when this is the last active in-app deployment). The agent itself is preserved.'
    case 'batch':
      return 'Lakeflow pipeline cleanup is not yet implemented (backend Phase 3). The deployment row will be marked deactivated; you may need to delete the pipeline manually.'
    default:
      return assertNever(d.mode)
  }
}
