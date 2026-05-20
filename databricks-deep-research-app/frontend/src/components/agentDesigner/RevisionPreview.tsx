/**
 * RevisionPreview — renders a read-only view of a specific agent revision's AST.
 *
 * Fetches via TanStack Query. Renders a minimal recursive read-only block stack.
 * Mount-to-first-render delta is measured via performance.now().
 */

import * as React from 'react'
import { useQuery } from '@tanstack/react-query'
import { getRevision } from '@/api/agentsV2'
import type { Block } from '@/types/ast'

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface RevisionPreviewProps {
  agentId: string
  revId: string
}

// ---------------------------------------------------------------------------
// Recursive read-only block renderer
// ---------------------------------------------------------------------------

function ReadOnlyBlock({ block, depth }: { block: Block; depth: number }): React.ReactElement {
  const children = block.children ?? []
  return (
    <div
      className={`rounded border border-slate-200 bg-white p-2 ${depth > 0 ? 'ml-4' : ''}`}
      data-testid={`readonly-block-${block.id}`}
    >
      <div className="flex items-center gap-2">
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono text-slate-500">
          {block.type}
        </span>
        <span className="text-sm font-medium text-slate-700">{block.label}</span>
      </div>
      {children.length > 0 && (
        <div className="mt-2 flex flex-col gap-1">
          {children.map((child) => (
            <ReadOnlyBlock key={child.id} block={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function RevisionPreview({ agentId, revId }: RevisionPreviewProps): React.ReactElement {
  const mountTimeRef = React.useRef<number>(performance.now())

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['agent-revision', agentId, revId],
    queryFn: () => getRevision(agentId, revId),
  })

  // Log render delta when data arrives (target < 200ms from mount)
  React.useEffect(() => {
    if (data) {
      const delta = performance.now() - mountTimeRef.current
      if (delta > 200) {
        console.warn(`[RevisionPreview] First render took ${delta.toFixed(1)}ms (target < 200ms)`)
      }
    }
  }, [data])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-6 text-sm text-slate-500">
        Loading revision…
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center p-6 text-sm text-red-500">
        {error instanceof Error ? error.message : 'Failed to load revision'}
      </div>
    )
  }

  if (!data) {
    return <div className="p-6 text-sm text-slate-400">No data</div>
  }

  return (
    <div className="flex flex-col gap-2 overflow-auto p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Revision preview
        </span>
        <span className="font-mono text-xs text-slate-400">
          {data.etag.replace(/^"|"$/g, '').slice(0, 8)}
        </span>
      </div>
      <ReadOnlyBlock block={data.definition.root} depth={0} />
    </div>
  )
}
