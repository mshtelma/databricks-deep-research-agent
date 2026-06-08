/**
 * UndeployConfirmDialog — destructive confirmation modal for the
 * Cancel / Undeploy / Unregister / Clean up / Retry cleanup actions.
 *
 * Cloned from AgentDesignerListPage.tsx::DeleteDialog (scrim + a11y +
 * destructive button styling) so this and DeleteDialog stay visually
 * consistent. Both consume the same Databricks tokens. A future
 * follow-up will extract a shared <ConfirmDialog> primitive.
 */

import * as React from 'react'

import { DeploymentApiError } from '@/api/deployments'
import type { DeploymentResponse } from '@/types/deployment'

import {
  type DeploymentAction,
  getAction,
  getImpactText,
} from './statusStyles'

interface CleanupFailedDetail {
  attempts: number
  max_attempts: number
  message: string
}

function parseCleanupFailed(error: unknown): CleanupFailedDetail | null {
  if (!(error instanceof DeploymentApiError)) return null
  const detail = error.detail
  // FastAPI wraps the dict in `{detail: ...}` — handle both shapes.
  const body =
    detail && typeof detail === 'object' && 'detail' in detail
      ? (detail as { detail: unknown }).detail
      : detail
  if (!body || typeof body !== 'object') return null
  const errKind = (body as { error_kind?: unknown }).error_kind
  if (errKind !== 'deployment_cleanup_failed') return null
  return {
    attempts: Number((body as { attempts?: unknown }).attempts ?? 0),
    max_attempts: Number((body as { max_attempts?: unknown }).max_attempts ?? 0),
    message: String((body as { message?: unknown }).message ?? ''),
  }
}

interface UndeployConfirmDialogProps {
  deployment: DeploymentResponse
  isPending: boolean
  error: unknown
  onConfirm: () => void
  onCancel: () => void
}

export function UndeployConfirmDialog({
  deployment,
  isPending,
  error,
  onConfirm,
  onCancel,
}: UndeployConfirmDialogProps): React.ReactElement | null {
  const cancelButtonRef = React.useRef<HTMLButtonElement | null>(null)

  // Focus the secondary "Keep" button on open (safer default for destructive).
  React.useEffect(() => {
    cancelButtonRef.current?.focus()
  }, [])

  // Close on Escape.
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isPending) onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isPending, onCancel])

  const action: DeploymentAction | null = getAction(deployment)
  // Defensive: dialog should never open for an inert deployment.
  if (action === null) return null

  const impactText = getImpactText(deployment)
  const cleanupFailed = parseCleanupFailed(error)
  const genericError = error && !cleanupFailed
    ? error instanceof Error ? error.message : 'Action failed.'
    : null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-db-navy-900/30 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
      aria-labelledby="undeploy-confirm-title"
      onClick={isPending ? undefined : onCancel}
      data-testid="undeploy-confirm-dialog"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="mx-4 w-full max-w-md rounded-db-lg border border-db-gray-lines bg-white p-5 shadow-db-xl"
      >
        <div
          id="undeploy-confirm-title"
          className="mb-1.5 text-[15px] font-medium text-db-navy-800"
        >
          {action.label}?
        </div>
        <p className="mb-5 text-[13px] leading-relaxed text-db-gray-text">
          {impactText}
        </p>
        {cleanupFailed && (
          <div
            role="alert"
            className="mb-4 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] leading-relaxed text-db-lava-700"
          >
            <div className="font-medium">
              Cleanup failed (attempt {cleanupFailed.attempts} of{' '}
              {cleanupFailed.max_attempts}). You can retry.
            </div>
            {cleanupFailed.message && (
              <div className="mt-1 font-db-mono text-[11px] opacity-80">
                {cleanupFailed.message}
              </div>
            )}
          </div>
        )}
        {genericError && (
          <div
            role="alert"
            className="mb-4 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] leading-relaxed text-db-lava-700"
          >
            {genericError}
          </div>
        )}
        <div className="flex justify-end gap-2">
          <button
            ref={cancelButtonRef}
            type="button"
            onClick={onCancel}
            disabled={isPending}
            className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium disabled:opacity-55"
            data-testid="undeploy-confirm-cancel"
          >
            Keep
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isPending}
            className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:opacity-55"
            data-testid="undeploy-confirm-action"
          >
            {isPending ? `${action.label}…` : action.label}
          </button>
        </div>
      </div>
    </div>
  )
}
