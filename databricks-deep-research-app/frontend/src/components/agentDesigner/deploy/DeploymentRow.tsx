/**
 * DeploymentRow — pure presentation of a single AgentDeployment.
 *
 * No mode-specific logic; every label, color, and action decision flows from
 * `statusStyles.ts`. Adding a new DeploymentMode or DeploymentStatus only
 * requires updating the resolver, not this component.
 */

import * as React from 'react'

import { Badge } from '@/components/ui/badge'
import type { DeploymentResponse } from '@/types/deployment'

import {
  MODE_COLOR,
  MODE_LABEL,
  STATUS_COLOR,
  getAction,
  getEffectiveStatusLabel,
  getResourceSummary,
} from './statusStyles'

function formatRelative(input: string): string {
  const date = new Date(input)
  if (Number.isNaN(date.getTime())) return '—'
  const diff = Date.now() - date.getTime()
  const min = Math.round(diff / 60_000)
  if (min < 1) return 'just now'
  if (min < 60) return `${min}m ago`
  const hr = Math.round(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.round(hr / 24)
  if (day === 1) return 'Yesterday'
  if (day < 7) return `${day}d ago`
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

interface DeploymentRowProps {
  deployment: DeploymentResponse
  /** Called when the user clicks the right-side action button. */
  onActionClick(d: DeploymentResponse): void
  /** When true, the action button shows a spinner and is disabled. */
  isPending?: boolean
}

export function DeploymentRow({
  deployment,
  onActionClick,
  isPending = false,
}: DeploymentRowProps): React.ReactElement {
  const action = getAction(deployment)
  const isDeactivated = deployment.status === 'deactivated'

  return (
    <div
      data-testid={`deployment-row-${deployment.id}`}
      className={`flex items-center gap-3 rounded-db-md border border-db-gray-lines bg-white px-3 py-2.5 ${
        isDeactivated ? 'opacity-60' : ''
      }`}
    >
      <span
        className={`inline-flex shrink-0 items-center rounded-db-pill px-2 py-0.5 font-db-mono text-[10px] font-medium tracking-[0.02em] ${
          MODE_COLOR[deployment.mode]
        }`}
      >
        {MODE_LABEL[deployment.mode]}
      </span>
      <Badge
        data-testid={`deployment-status-${deployment.status}`}
        className={STATUS_COLOR[deployment.status]}
      >
        {getEffectiveStatusLabel(deployment)}
      </Badge>
      <div className="min-w-0 flex-1">
        <div className="truncate font-db-mono text-[12px] text-db-navy-800">
          {getResourceSummary(deployment)}
        </div>
        <div className="mt-0.5 text-[11px] text-db-gray-text">
          {isDeactivated && deployment.deactivated_at
            ? `Deactivated · ${formatRelative(deployment.deactivated_at)}`
            : `Updated ${formatRelative(deployment.updated_at)}`}
        </div>
      </div>
      {action !== null && (
        <button
          type="button"
          onClick={() => onActionClick(deployment)}
          disabled={isPending}
          data-testid={`deployment-action-${deployment.id}`}
          className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:bg-db-lava-100 hover:border-db-lava-300 hover:text-db-lava-700 disabled:opacity-55"
        >
          {isPending ? `${action.label}…` : action.label}
        </button>
      )}
    </div>
  )
}
