/**
 * useDeploymentAction — shared orchestration hook for the Cancel /
 * Undeploy / Unregister / Clean up / Retry cleanup flow.
 *
 * Wraps `useDeactivateDeployment` (which already wires cache invalidation
 * and DELETE /api/v1/deployments/{id}) with confirm-modal state, exposing
 * one `dialog` ReactNode that the caller mounts at a stable parent.
 *
 * Consumers: StatusPanel (inline post-deploy badge) and DeploymentsSection
 * (per-agent deployments list). Both share an identical user contract —
 * label resolution, dialog text, error surfacing, cache invalidation —
 * by going through this single source of truth.
 */

import * as React from 'react'

import { useDeactivateDeployment } from '@/hooks/useDeployments'
import type { DeploymentResponse } from '@/types/deployment'

import { UndeployConfirmDialog } from './UndeployConfirmDialog'

export interface UseDeploymentActionResult {
  /** Open the confirm dialog for a given deployment row. */
  openConfirm(d: DeploymentResponse): void
  /** Mount this ReactNode somewhere stable (section root). */
  dialog: React.ReactNode
  /** Deployment.id currently in flight, or null. */
  pendingFor: string | null
}

export function useDeploymentAction(): UseDeploymentActionResult {
  const [target, setTarget] = React.useState<DeploymentResponse | null>(null)
  const mutation = useDeactivateDeployment()

  const close = React.useCallback(() => {
    if (mutation.isPending) return
    setTarget(null)
    mutation.reset()
  }, [mutation])

  const openConfirm = React.useCallback((d: DeploymentResponse) => {
    setTarget(d)
    mutation.reset()
  }, [mutation])

  const confirm = React.useCallback(() => {
    if (!target || mutation.isPending) return
    mutation.mutate(target.id, {
      onSuccess: () => {
        setTarget(null)
      },
      // onError: keep dialog open so user can read the inline error banner.
    })
  }, [target, mutation])

  const dialog = target !== null ? (
    <UndeployConfirmDialog
      deployment={target}
      isPending={mutation.isPending}
      error={mutation.error}
      onConfirm={confirm}
      onCancel={close}
    />
  ) : null

  return {
    openConfirm,
    dialog,
    pendingFor: mutation.isPending && target ? target.id : null,
  }
}
