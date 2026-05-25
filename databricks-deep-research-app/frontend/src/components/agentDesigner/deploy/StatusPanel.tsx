/**
 * StatusPanel — renders a deployment status badge + last-update timestamp +
 * error_message (when present). Polls the lightweight `/status` endpoint
 * until the deployment reaches a terminal state.
 *
 * When the optional `deployment` prop is supplied the panel also exposes a
 * destructive action button (Cancel / Undeploy / Unregister / Clean up /
 * Retry cleanup) driven by `getAction` from `statusStyles.ts`. The button
 * opens the shared `UndeployConfirmDialog` via `useDeploymentAction`.
 */

import * as React from 'react'

import { Badge } from '@/components/ui/badge'
import { useDeploymentStatusPoll } from '@/hooks/useDeployments'
import type { DeploymentResponse } from '@/types/deployment'

import { STATUS_COLOR, STATUS_LABEL, getAction } from './statusStyles'
import { useDeploymentAction } from './useDeploymentAction'

interface StatusPanelProps {
  deploymentId: string
  /** When false (e.g. modal closed) the polling query is paused. */
  active?: boolean
  /** Full deployment object. When supplied, an action button + confirm
   *  dialog are rendered for Cancel / Undeploy / etc. Legacy callers may
   *  omit this and the panel behaves as before. */
  deployment?: DeploymentResponse
}

export function StatusPanel({
  deploymentId,
  active = true,
  deployment,
}: StatusPanelProps): React.ReactElement {
  const query = useDeploymentStatusPoll(deploymentId, { enabled: active })
  const action = useDeploymentAction()

  if (query.isLoading) {
    return (
      <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600">
        Loading deployment status…
      </div>
    )
  }

  if (query.isError || !query.data) {
    return (
      <div
        role="alert"
        className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
      >
        Could not load deployment status.
      </div>
    )
  }

  const { status, updated_at, error_message } = query.data
  const updatedDate = new Date(updated_at)

  // Action button needs the full row (mode + cancel_requested + resources).
  // Derive a merged view: live status from the poll, static fields from prop.
  const liveDeployment: DeploymentResponse | null = deployment
    ? { ...deployment, status, updated_at, error_message }
    : null
  const buttonAction = liveDeployment ? getAction(liveDeployment) : null

  return (
    <div
      data-testid="deployment-status-panel"
      className="space-y-2 rounded-md border border-zinc-200 bg-white px-3 py-2"
    >
      <div className="flex items-center justify-between gap-2">
        <Badge
          data-testid={`deployment-status-${status}`}
          className={STATUS_COLOR[status]}
        >
          {STATUS_LABEL[status]}
        </Badge>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">
            Updated {updatedDate.toLocaleTimeString()}
          </span>
          {liveDeployment && buttonAction && (
            <button
              type="button"
              onClick={() => action.openConfirm(liveDeployment)}
              disabled={action.pendingFor === liveDeployment.id}
              data-testid={`status-panel-action-${buttonAction.kind}`}
              className="rounded-db-md border border-db-gray-lines bg-white px-2 py-1 text-[11px] font-medium text-db-navy-800 transition-colors hover:bg-db-lava-100 hover:border-db-lava-300 hover:text-db-lava-700 disabled:opacity-55"
            >
              {action.pendingFor === liveDeployment.id
                ? `${buttonAction.label}…`
                : buttonAction.label}
            </button>
          )}
        </div>
      </div>
      {error_message ? (
        <p className="text-xs text-red-700" role="alert">
          {error_message}
        </p>
      ) : null}
      {action.dialog}
    </div>
  )
}
