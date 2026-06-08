/**
 * DeploymentsSection — agent-scoped deployment list.
 *
 * Renders every deployment for one agent (active / pending / deploying /
 * failed / cleanup_failed / deactivated) and exposes per-row Cancel /
 * Undeploy / Unregister / Clean up / Retry cleanup actions through a
 * single shared `useDeploymentAction` instance (one dialog node total,
 * not N).
 *
 * Smart refetch: polls every 5s while any row is non-terminal so the
 * status badges stay fresh during in-flight deploys/undeploys, then
 * idles.
 */

import { useQuery } from '@tanstack/react-query'
import * as React from 'react'

import { listDeployments } from '@/api/deployments'
import { deploymentKeys } from '@/hooks/useDeployments'
import {
  TERMINAL_STATUSES,
  type DeploymentListResponse,
} from '@/types/deployment'

import { DeploymentRow } from './DeploymentRow'
import { useDeploymentAction } from './useDeploymentAction'

interface DeploymentsSectionProps {
  agentId: string
}

const REFETCH_WHILE_IN_FLIGHT_MS = 5_000
const PAGE_LIMIT = 50

export function DeploymentsSection({
  agentId,
}: DeploymentsSectionProps): React.ReactElement {
  const action = useDeploymentAction()
  const [pages, setPages] = React.useState<string[]>([])

  // Lead page = no cursor. "Load more" appends a cursor to `pages`.
  // We render one query per page so cache invalidation refreshes them all
  // consistently when a mutation fires.
  const filters = React.useMemo(
    () => ({ agent_id: agentId, limit: PAGE_LIMIT }),
    [agentId],
  )

  const query = useQuery<DeploymentListResponse>({
    queryKey: deploymentKeys.list(filters),
    queryFn: () => listDeployments(filters),
    refetchInterval: (q) => {
      const data = q.state.data
      if (!data) return REFETCH_WHILE_IN_FLIGHT_MS
      const hasNonTerminal = data.items.some(
        (d) => !TERMINAL_STATUSES.has(d.status),
      )
      return hasNonTerminal ? REFETCH_WHILE_IN_FLIGHT_MS : false
    },
  })

  // Page 2+ queries are conditional. Render them through a small inline
  // helper so each gets its own cache key.
  const additionalPages = pages.map((cursor) => ({ ...filters, cursor }))

  if (query.isLoading) {
    return (
      <div className="space-y-2">
        <div className="h-14 animate-pulse rounded-db-md border border-db-gray-lines bg-db-oat-light" />
        <div className="h-14 animate-pulse rounded-db-md border border-db-gray-lines bg-db-oat-light" />
      </div>
    )
  }

  if (query.isError || !query.data) {
    return (
      <div
        role="alert"
        className="rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-sm text-db-lava-700"
      >
        Could not load deployments.
      </div>
    )
  }

  const leadItems = query.data.items
  const leadCursor = query.data.next_cursor

  if (leadItems.length === 0 && additionalPages.length === 0) {
    return (
      <div className="rounded-db-md border border-dashed border-db-gray-lines bg-db-oat-light px-4 py-6 text-center text-sm text-db-gray-text">
        No deployments yet. Use <strong className="font-medium">Deploy</strong>{' '}
        above to publish this agent.
      </div>
    )
  }

  return (
    <div className="space-y-2" data-testid="deployments-section">
      {leadItems.map((d) => (
        <DeploymentRow
          key={d.id}
          deployment={d}
          onActionClick={action.openConfirm}
          isPending={action.pendingFor === d.id}
        />
      ))}
      {additionalPages.map((pageFilters) => (
        <DeploymentsPage
          key={pageFilters.cursor}
          filters={pageFilters}
          onAction={action.openConfirm}
          pendingFor={action.pendingFor}
        />
      ))}
      {leadCursor && additionalPages.length === 0 && (
        <LoadMoreButton onClick={() => setPages([leadCursor])} />
      )}
      {action.dialog}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Auxiliary subcomponents
// ---------------------------------------------------------------------------

interface DeploymentsPageProps {
  filters: { agent_id: string; limit: number; cursor: string }
  onAction(d: import('@/types/deployment').DeploymentResponse): void
  pendingFor: string | null
}

function DeploymentsPage({
  filters,
  onAction,
  pendingFor,
}: DeploymentsPageProps): React.ReactElement {
  const query = useQuery<DeploymentListResponse>({
    queryKey: deploymentKeys.list(filters),
    queryFn: () => listDeployments(filters),
  })
  if (!query.data) return <></>
  return (
    <>
      {query.data.items.map((d) => (
        <DeploymentRow
          key={d.id}
          deployment={d}
          onActionClick={onAction}
          isPending={pendingFor === d.id}
        />
      ))}
    </>
  )
}

function LoadMoreButton({
  onClick,
}: {
  onClick(): void
}): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-1 w-full rounded-db-md border border-db-gray-lines bg-white px-3 py-2 text-[12px] font-medium text-db-navy-800 hover:bg-db-oat-light"
      data-testid="deployments-load-more"
    >
      Load more
    </button>
  )
}
