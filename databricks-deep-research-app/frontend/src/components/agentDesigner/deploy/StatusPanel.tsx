/**
 * StatusPanel — renders a deployment status badge + last-update timestamp +
 * error_message (when present). Polls the lightweight `/status` endpoint until
 * the deployment reaches a terminal state.
 *
 * Used by the agent-designer Deploy flow after an `InAppWizard` (or future
 * mode wizard) submits a deployment.
 */

import * as React from 'react'

import { Badge } from '@/components/ui/badge'
import { useDeploymentStatusPoll } from '@/hooks/useDeployments'
import type { DeploymentStatus } from '@/types/deployment'

interface StatusPanelProps {
  deploymentId: string
  /** When false (e.g. modal closed) the polling query is paused. */
  active?: boolean
}

const STATUS_COLOR: Record<DeploymentStatus, string> = {
  pending: 'bg-yellow-500/15 text-yellow-700 border-yellow-300',
  deploying: 'bg-blue-500/15 text-blue-700 border-blue-300',
  active: 'bg-green-500/15 text-green-700 border-green-300',
  failed: 'bg-red-500/15 text-red-700 border-red-300',
  deactivated: 'bg-zinc-500/15 text-zinc-700 border-zinc-300',
  cleanup_failed: 'bg-zinc-500/15 text-zinc-700 border-zinc-300',
}

const STATUS_LABEL: Record<DeploymentStatus, string> = {
  pending: 'Pending',
  deploying: 'Deploying',
  active: 'Active',
  failed: 'Failed',
  deactivated: 'Deactivated',
  cleanup_failed: 'Cleanup failed',
}

export function StatusPanel({
  deploymentId,
  active = true,
}: StatusPanelProps): React.ReactElement {
  const query = useDeploymentStatusPoll(deploymentId, { enabled: active })

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
        <span className="text-xs text-zinc-500">
          Updated {updatedDate.toLocaleTimeString()}
        </span>
      </div>
      {error_message ? (
        <p className="text-xs text-red-700" role="alert">
          {error_message}
        </p>
      ) : null}
    </div>
  )
}
