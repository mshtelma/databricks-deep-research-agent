/**
 * RevisionList — displays the revision history for an agent.
 *
 * Fetches via TanStack Query and renders one row per revision.
 * Clicking a row calls onSelectRevision(rev_id) to trigger the preview panel.
 * The Restore button calls updateAgentV2 with the revision's definition.
 */

import * as React from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listRevisions, getRevision, updateAgentV2 } from '@/api/agentsV2'
import { useAgentEditorStore } from '@/stores/agentEditorStore'
import type { RevisionSummary } from '@/api/agentsV2'

// ---------------------------------------------------------------------------
// Inline relative-time helper (~10 lines, no date library)
// ---------------------------------------------------------------------------

function relativeTime(isoString: string): string {
  const diffMs = Date.now() - new Date(isoString).getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  const diffDay = Math.floor(diffHr / 24)
  return `${diffDay}d ago`
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface RevisionListProps {
  agentId: string
  onSelectRevision: (revId: string | null) => void
  selectedRevId: string | null
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function RevisionList({
  agentId,
  onSelectRevision,
  selectedRevId,
}: RevisionListProps): React.ReactElement {
  const queryClient = useQueryClient()
  const etag = useAgentEditorStore((s) => s.etag)

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['agent-revisions', agentId],
    queryFn: () => listRevisions(agentId),
  })

  const restoreMutation = useMutation({
    mutationFn: async (revId: string) => {
      const revision = await getRevision(agentId, revId)
      const currentEtag = etag ?? ''
      return updateAgentV2(
        agentId,
        { definition: revision.definition as unknown as Record<string, unknown> },
        currentEtag,
      )
    },
    onSuccess: ({ agent, etag: newEtag }) => {
      // Reload store with the restored agent
      useAgentEditorStore.getState().load({ agent, etag: newEtag })
      // Invalidate revisions list so the new restore shows up
      void queryClient.invalidateQueries({ queryKey: ['agent-revisions', agentId] })
    },
  })

  // -------------------------------------------------------------------------
  // Loading / error states
  // -------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-6 text-sm text-slate-500">
        Loading revisions…
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center p-6 text-sm text-red-500">
        {error instanceof Error ? error.message : 'Failed to load revisions'}
      </div>
    )
  }

  const items: RevisionSummary[] = data?.items ?? []

  // -------------------------------------------------------------------------
  // Empty state
  // -------------------------------------------------------------------------

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center p-6 text-sm text-slate-500">
        No prior revisions
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Revision rows
  // -------------------------------------------------------------------------

  return (
    <div className="flex flex-col divide-y divide-slate-100 overflow-auto">
      {items.map((rev) => {
        const isSelected = rev.rev_id === selectedRevId
        return (
          <div
            key={rev.rev_id}
            role="row"
            aria-selected={isSelected}
            onClick={() => onSelectRevision(rev.rev_id)}
            className={`flex cursor-pointer items-center gap-3 px-4 py-3 hover:bg-slate-50 ${
              isSelected ? 'bg-blue-50' : ''
            }`}
          >
            {/* Timestamp */}
            <span className="min-w-[4rem] text-xs font-medium text-slate-500">
              {relativeTime(rev.created_at)}
            </span>

            {/* Author */}
            <span className="flex-1 truncate text-sm text-slate-700">{rev.created_by}</span>

            {/* ETag prefix */}
            <span className="font-mono text-xs text-slate-400">
              {rev.etag.replace(/^"|"$/g, '').slice(0, 8)}
            </span>

            {/* Restore button */}
            <button
              type="button"
              aria-label={`Restore revision ${rev.rev_id}`}
              onClick={(e) => {
                e.stopPropagation()
                restoreMutation.mutate(rev.rev_id)
              }}
              disabled={restoreMutation.isPending}
              className="rounded border border-slate-300 bg-white px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Restore
            </button>
          </div>
        )
      })}
    </div>
  )
}
