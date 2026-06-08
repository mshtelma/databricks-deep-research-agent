/**
 * TanStack Query hooks for the Agent Designer Deployment API.
 *
 * Mirrors the surface in `src/api/deployments.ts`. The status-poll hook
 * stops auto-refetching once the deployment reaches a terminal status
 * (DEACTIVATED / CLEANUP_FAILED).
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  getDeploymentDefaults,
  type DeploymentDefaultsResponse,
} from '@/api/config'
import {
  canDeployHereAction,
  canRunFast,
  canRunSlow,
  createDeployment,
  deactivateDeployment,
  deployHereAction,
  getDeployment,
  getDeploymentStatus,
  listDeployments,
  refreshCanDeployHereAction,
} from '@/api/deployments'
import {
  TERMINAL_STATUSES,
  type CanDeployHereResponse,
  type CanRunFastResponse,
  type CanRunSlowResponse,
  type CreateDeploymentRequest,
  type DeploymentListFilters,
  type DeploymentListResponse,
  type DeploymentResponse,
  type DeploymentStatusResponse,
} from '@/types/deployment'

// =============================================================================
// Query Keys
// =============================================================================

export const deploymentKeys = {
  all: ['deployments'] as const,
  list: (filters: DeploymentListFilters) =>
    [...deploymentKeys.all, 'list', filters] as const,
  detail: (id: string) => [...deploymentKeys.all, 'detail', id] as const,
  status: (id: string) => [...deploymentKeys.all, 'status', id] as const,
  canRunFast: (agentId: string) =>
    [...deploymentKeys.all, 'can-run', 'fast', agentId] as const,
  canRunSlow: (agentId: string) =>
    [...deploymentKeys.all, 'can-run', 'slow', agentId] as const,
  defaults: () => [...deploymentKeys.all, 'defaults'] as const,
}

// =============================================================================
// List + detail
// =============================================================================

export function useDeployments(filters: DeploymentListFilters = {}) {
  return useQuery<DeploymentListResponse>({
    queryKey: deploymentKeys.list(filters),
    queryFn: () => listDeployments(filters),
  })
}

export function useDeployment(id: string | undefined | null) {
  return useQuery<DeploymentResponse>({
    queryKey: deploymentKeys.detail(id ?? ''),
    queryFn: () => getDeployment(id as string),
    enabled: !!id,
  })
}

// =============================================================================
// Status polling
// =============================================================================

const ACTIVE_POLL_INTERVAL_MS = 5_000

/**
 * Polls the lightweight status endpoint until a terminal status is reached.
 *
 * Use this in the StatusPanel after a deploy is initiated; pass `enabled`
 * false when the modal is closed to stop the query.
 */
export function useDeploymentStatusPoll(
  id: string | undefined | null,
  options: { enabled?: boolean } = {},
) {
  const { enabled = true } = options
  return useQuery<DeploymentStatusResponse>({
    queryKey: deploymentKeys.status(id ?? ''),
    queryFn: () => getDeploymentStatus(id as string),
    enabled: !!id && enabled,
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data) return ACTIVE_POLL_INTERVAL_MS
      if (TERMINAL_STATUSES.has(data.status)) return false
      return ACTIVE_POLL_INTERVAL_MS
    },
    refetchOnWindowFocus: false,
  })
}

// =============================================================================
// Mutations
// =============================================================================

export function useCreateDeployment() {
  const qc = useQueryClient()
  return useMutation<DeploymentResponse, Error, CreateDeploymentRequest>({
    mutationFn: (req) => createDeployment(req),
    onSuccess: (created) => {
      // Invalidate list views and seed detail cache.
      void qc.invalidateQueries({ queryKey: deploymentKeys.all })
      qc.setQueryData(deploymentKeys.detail(created.id), created)
    },
  })
}

export function useDeactivateDeployment() {
  const qc = useQueryClient()
  return useMutation<DeploymentResponse, Error, string>({
    mutationFn: (id) => deactivateDeployment(id),
    onSuccess: (updated) => {
      qc.setQueryData(deploymentKeys.detail(updated.id), updated)
      void qc.invalidateQueries({ queryKey: deploymentKeys.all })
    },
  })
}

// =============================================================================
// Capability probes
// =============================================================================

export function useCanRunFast(agentId: string | undefined | null) {
  return useQuery<CanRunFastResponse>({
    queryKey: deploymentKeys.canRunFast(agentId ?? ''),
    queryFn: () => canRunFast(agentId as string),
    enabled: !!agentId,
    staleTime: 0,
  })
}

export function useCanRunSlow(agentId: string | undefined | null) {
  return useQuery<CanRunSlowResponse>({
    queryKey: deploymentKeys.canRunSlow(agentId ?? ''),
    queryFn: () => canRunSlow(agentId as string),
    enabled: !!agentId,
    // Match backend cache TTL (5 min) — UC probe is the slow path and
    // re-query within this window is wasteful.
    staleTime: 5 * 60 * 1000,
  })
}

// =============================================================================
// Deploy-here action hooks
// =============================================================================

/**
 * Wraps deployHereAction for a known deploymentId. On settle (success or
 * error) it invalidates the status-poll query so the wizard's StatusPanel
 * picks up the latest state automatically.
 */
export function useDeployHere(deploymentId: string) {
  const qc = useQueryClient()
  return useMutation<
    DeploymentResponse,
    Error,
    { confirmRedeploy?: boolean } | undefined
  >({
    mutationFn: (opts) => deployHereAction(deploymentId, opts ?? undefined),
    onSettled: () => {
      void qc.invalidateQueries({
        queryKey: deploymentKeys.status(deploymentId),
      })
    },
  })
}

/**
 * Convenience hook for the two-step wizard flow:
 *   1. POST /deployments  (useCreateDeployment)
 *   2. POST /deployments/{id}/actions/deploy-here
 *
 * Variables: { agent_id, revision_id, config }
 * Returns the post-deploy-here DeploymentResponse. The usual status is
 * `deploying`; callers poll `/status` for completion.
 * Step 1 creates only the row; the inline action owns status transitions.
 */
export function useDeployHereFromConfig() {
  const qc = useQueryClient()
  return useMutation<
    DeploymentResponse,
    Error,
    CreateDeploymentRequest
  >({
    mutationFn: async (req) => {
      const pending = await createDeployment(req, { runAsync: false })
      // Seed detail cache so polls start with a known row
      qc.setQueryData(deploymentKeys.detail(pending.id), pending)
      return deployHereAction(pending.id)
    },
    onSuccess: (result) => {
      qc.setQueryData(deploymentKeys.detail(result.id), result)
      void qc.invalidateQueries({ queryKey: deploymentKeys.all })
    },
  })
}

// =============================================================================
// Wizard defaults
// =============================================================================

/**
 * Default values for the deployment wizards (currently: shell-app Git ref).
 *
 * The backend resolves the framework's installed version, so a release bump
 * propagates to the wizard with no frontend edit. Cached for 5 minutes since
 * it only changes on framework upgrade.
 */
export function useDeploymentDefaults() {
  return useQuery<DeploymentDefaultsResponse>({
    queryKey: deploymentKeys.defaults(),
    queryFn: getDeploymentDefaults,
    staleTime: 5 * 60 * 1000,
    gcTime: Infinity,
  })
}

// =============================================================================
// Can-deploy-here probe
// =============================================================================

/**
 * Probes whether the current actor can deploy in this workspace.
 * Cached for 60 s; will not re-fetch on window focus.
 */
export function useCanDeployHere() {
  return useQuery<CanDeployHereResponse>({
    queryKey: ['can-deploy-here'],
    queryFn: canDeployHereAction,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    refetchInterval: false,
  })
}

/**
 * Invalidates the server-side cache then re-probes.
 * On success the query cache for ['can-deploy-here'] is updated immediately.
 */
export function useRefreshCanDeployHere() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: refreshCanDeployHereAction,
    onSuccess: (data) => {
      qc.setQueryData(['can-deploy-here'], data)
    },
  })
}
