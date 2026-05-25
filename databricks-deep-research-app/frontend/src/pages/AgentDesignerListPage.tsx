/**
 * AgentDesignerListPage — lists all agents_v2 with the Databricks Agentic
 * Designer chrome (header bar + cards/list grid + CTA banner).
 *
 * Uses the shared <AppShell/> so the chat sidebar stays visible while
 * browsing agents (matches the design's intent for cross-page navigation).
 */

import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { parseAgentDeleteError } from '@/api/agentsV2'
import { useAgentsV2List, useDeleteAgentV2 } from '@/hooks/useAgentsV2'
import type { AgentV2Summary, AgentVisibility } from '@/types/agentDesigner'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compact relative time string ("2m ago", "Yesterday", "3d ago", "May 1"). */
function relativeTime(dateStr: string): string {
  const now = Date.now()
  const then = new Date(dateStr).getTime()
  if (Number.isNaN(then)) return ''
  const diffMs = now - then
  const min = Math.round(diffMs / 60_000)
  if (min < 1) return 'just now'
  if (min < 60) return `${min}m ago`
  const hr = Math.round(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.round(hr / 24)
  if (day === 1) return 'Yesterday'
  if (day < 7) return `${day}d ago`
  return new Date(dateStr).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

const VISIBILITY_META: Record<AgentVisibility, { label: string; color: string; bg: string }> = {
  private: { label: 'Private', color: 'var(--db-gray-text)', bg: 'var(--db-oat-medium)' },
  workspace: { label: 'Workspace', color: 'var(--db-blue-700)', bg: 'var(--db-blue-100)' },
  system: { label: 'System', color: 'var(--db-maroon-600)', bg: 'var(--db-maroon-300)' },
}

// ---------------------------------------------------------------------------
// Delete confirmation dialog (Databricks-styled scrim modal)
// ---------------------------------------------------------------------------

interface DeleteDialogProps {
  agentName: string
  isPending: boolean
  error: unknown
  onConfirm: () => void
  onForceConfirm: () => void
  onCancel: () => void
}

function DeleteDialog({
  agentName,
  isPending,
  error,
  onConfirm,
  onForceConfirm,
  onCancel,
}: DeleteDialogProps) {
  const deleteError = parseAgentDeleteError(error)
  const genericError = error && !deleteError
    ? error instanceof Error
      ? error.message
      : 'Delete failed.'
    : null
  const forceLabel = deleteError?.error_kind === 'deployment_cleanup_failed'
    ? 'Retry cleanup'
    : 'Delete and deactivate deployments'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-db-navy-900/30 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
      aria-label="Delete confirmation"
      onClick={onCancel}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="mx-4 w-full max-w-sm rounded-db-lg border border-db-gray-lines bg-white p-5 shadow-db-xl"
      >
        <div className="mb-1.5 text-[15px] font-medium text-db-navy-800">Delete agent?</div>
        <p className="mb-5 text-[13px] leading-relaxed text-db-gray-text">
          <strong className="font-medium text-db-navy-800">{agentName}</strong> will be permanently
          removed. This cannot be undone.
        </p>
        {deleteError?.error_kind === 'active_deployments_exist' && (
          <div
            role="alert"
            className="mb-4 rounded-db-md border border-db-yellow-700 bg-db-yellow-300 px-3 py-2 text-[12px] leading-relaxed text-db-yellow-800"
          >
            <div className="font-medium">Active deployments block deletion.</div>
            <ul className="mt-1 list-disc pl-4">
              {deleteError.deployments.map((deployment) => (
                <li key={deployment.id}>
                  <span className="font-db-mono">{deployment.mode}</span>
                  {' · '}
                  <span className="font-db-mono">{deployment.status}</span>
                  {' · '}
                  {deployment.endpoint_name ?? deployment.id.slice(0, 8)}
                </li>
              ))}
            </ul>
          </div>
        )}
        {deleteError?.error_kind === 'deployment_cleanup_failed' && (
          <div
            role="alert"
            className="mb-4 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] leading-relaxed text-db-lava-700"
          >
            {deleteError.message}
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
            type="button"
            onClick={onCancel}
            disabled={isPending}
            className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium disabled:opacity-55"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isPending}
            className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:opacity-55"
          >
            {isPending ? 'Deleting…' : 'Delete'}
          </button>
          {deleteError && (
            <button
              type="button"
              onClick={onForceConfirm}
              disabled={isPending}
              className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:opacity-55"
            >
              {isPending ? 'Deleting…' : forceLabel}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Card view
// ---------------------------------------------------------------------------

interface AgentCardProps {
  agent: AgentV2Summary
  onOpen: (id: string) => void
  onDeleteRequest: (agent: AgentV2Summary) => void
}

function AgentCard({ agent, onOpen, onDeleteRequest }: AgentCardProps) {
  const meta = VISIBILITY_META[agent.visibility] ?? VISIBILITY_META.private
  const [showMenu, setShowMenu] = React.useState(false)
  const menuRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!showMenu) return
    const onDoc = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setShowMenu(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [showMenu])

  return (
    <div
      onClick={() => onOpen(agent.id)}
      className="group flex cursor-pointer flex-col gap-3 rounded-db-md border border-db-gray-lines bg-white p-[18px] transition-all hover:border-db-navy-300 hover:shadow-db-sm"
    >
      {/* Header */}
      <div className="flex items-start gap-2.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-db-md bg-db-lava-100">
          <BotIcon className="h-4 w-4 text-db-lava-600" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[14px] font-medium leading-[1.3] text-db-navy-800">
            {agent.name}
          </div>
          <div className="mt-1 flex items-center gap-1.5">
            <span
              className="inline-flex items-center gap-1 rounded-db-pill px-1.5 py-px font-db-mono text-[10px] font-semibold uppercase tracking-[0.04em]"
              style={{ background: meta.bg, color: meta.color }}
            >
              <span
                className="h-[5px] w-[5px] rounded-full"
                style={{ background: meta.color }}
              />
              {meta.label}
            </span>
            <span className="text-[11px] text-db-gray-text">· {relativeTime(agent.updated_at)}</span>
          </div>
        </div>
        <div ref={menuRef} className="relative">
          <button
            type="button"
            aria-label={`Actions for ${agent.name}`}
            onClick={(e) => {
              e.stopPropagation()
              setShowMenu((v) => !v)
            }}
            className="rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
          >
            <MoreIcon className="h-3.5 w-3.5" />
          </button>
          {showMenu && (
            <div
              className="absolute right-0 top-full z-20 mt-1 w-40 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-db-md"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={() => {
                  setShowMenu(false)
                  onOpen(agent.id)
                }}
                className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
              >
                Open
              </button>
              <div className="my-1 h-px bg-db-gray-lines" />
              <button
                type="button"
                onClick={() => {
                  setShowMenu(false)
                  onDeleteRequest(agent)
                }}
                className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-[13px] font-medium text-db-lava-700 transition-colors hover:bg-db-lava-100"
              >
                Delete
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Description */}
      {agent.description ? (
        <p className="m-0 line-clamp-3 text-[12.5px] leading-[1.5] text-db-gray-text">
          {agent.description}
        </p>
      ) : (
        <p className="m-0 text-[12.5px] italic leading-[1.5] text-db-navy-300">No description</p>
      )}

      {/* Footer */}
      <div className="flex items-center gap-2.5 border-t border-db-gray-lines pt-3 text-[11px] text-db-gray-text">
        <span className="inline-flex items-center gap-1">
          <BlocksIcon className="h-3 w-3" />
          {agent.node_count} {agent.node_count === 1 ? 'block' : 'blocks'}
        </span>
        <span
          className="ml-auto truncate font-db-mono text-[10px] text-db-navy-400"
          title={agent.id}
        >
          {agent.id.slice(0, 12)}
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// List view (dense)
// ---------------------------------------------------------------------------

interface AgentRowProps {
  agent: AgentV2Summary
  onOpen: (id: string) => void
  onDeleteRequest: (agent: AgentV2Summary) => void
}

function AgentRow({ agent, onOpen, onDeleteRequest }: AgentRowProps) {
  const meta = VISIBILITY_META[agent.visibility] ?? VISIBILITY_META.private
  return (
    <div
      onClick={() => onOpen(agent.id)}
      className="grid cursor-pointer grid-cols-[1.6fr_120px_120px_100px_auto] items-center gap-3.5 border-b border-db-gray-lines px-4 py-3.5 transition-colors last:border-b-0 hover:bg-db-oat-light"
    >
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded bg-db-lava-100">
          <BotIcon className="h-3.5 w-3.5 text-db-lava-600" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-[13px] font-medium leading-[1.3] text-db-navy-800">
            {agent.name}
          </div>
          {agent.description && (
            <div className="truncate text-[11px] text-db-gray-text">{agent.description}</div>
          )}
        </div>
      </div>
      <span
        className="justify-self-start rounded-db-pill px-1.5 py-px font-db-mono text-[10px] font-semibold uppercase tracking-[0.04em]"
        style={{ background: meta.bg, color: meta.color }}
      >
        {meta.label}
      </span>
      <span
        className="truncate font-db-mono text-[12px] text-db-navy-400"
        title={agent.id}
      >
        {agent.id.slice(0, 12)}
      </span>
      <span className="font-db-mono text-[12px] text-db-navy-800">
        {relativeTime(agent.updated_at)}
      </span>
      <button
        type="button"
        aria-label={`Delete ${agent.name}`}
        onClick={(e) => {
          e.stopPropagation()
          onDeleteRequest(agent)
        }}
        className="justify-self-end rounded p-1 text-db-gray-text transition-colors hover:bg-db-lava-100 hover:text-db-lava-700"
      >
        <TrashIcon className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

type ViewMode = 'grid' | 'list'

export function AgentDesignerListPage() {
  const navigate = useNavigate()
  const { data, isLoading, error } = useAgentsV2List()
  const deleteMutation = useDeleteAgentV2()

  const [pendingDelete, setPendingDelete] = React.useState<AgentV2Summary | null>(null)
  const [deleteError, setDeleteError] = React.useState<unknown>(null)
  const [search, setSearch] = React.useState('')
  const [view, setView] = React.useState<ViewMode>('grid')

  const handleOpen = (id: string) => {
    void navigate(`/designer/${id}`)
  }
  const handleDeleteRequest = (agent: AgentV2Summary) => {
    setDeleteError(null)
    setPendingDelete(agent)
  }
  const handleDeleteConfirm = (force = false) => {
    if (!pendingDelete) return
    setDeleteError(null)
    deleteMutation.mutate(force ? { id: pendingDelete.id, force: true } : pendingDelete.id, {
      onSuccess: () => {
        setPendingDelete(null)
        setDeleteError(null)
      },
      onError: (err) => {
        setDeleteError(err)
      },
    })
  }
  const handleCreateNew = () => {
    void navigate('/designer/new')
  }

  const items = React.useMemo(() => data?.items ?? [], [data])
  const filtered = React.useMemo(() => {
    if (!search.trim()) return items
    const q = search.toLowerCase()
    return items.filter(
      (a) =>
        a.name.toLowerCase().includes(q) ||
        (a.description ?? '').toLowerCase().includes(q),
    )
  }, [items, search])

  return (
    <AppShell>
      {/* Header bar */}
      <header className="flex h-14 shrink-0 items-center gap-3.5 border-b border-db-gray-lines bg-white px-6">
        <div className="flex items-center gap-2 text-[13px] text-db-gray-text">
          <span>workspace</span>
          <span className="text-db-navy-300">/</span>
          <span className="font-medium text-db-navy-800">Agents</span>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-db-pill bg-db-oat-medium px-2.5 py-0.5 font-db-mono text-[11px] text-db-gray-text">
          <span className="h-[5px] w-[5px] rounded-full bg-db-green-700" />
          {items.length} {items.length === 1 ? 'agent' : 'agents'}
        </span>
        <div className="ml-auto flex items-center gap-2.5">
          <button
            type="button"
            disabled
            title="Import — coming soon"
            className="inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 opacity-55"
          >
            <FileIcon className="h-3.5 w-3.5" /> Import
          </button>
          <button
            type="button"
            onClick={handleCreateNew}
            className="inline-flex items-center gap-1.5 rounded-db-md bg-db-navy-800 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-navy-900"
          >
            <PlusIcon className="h-3 w-3" /> New Agent
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-auto bg-db-oat-light">
        <div className="mx-auto max-w-7xl px-7 py-7">
          {/* Title */}
          <div className="mb-[22px]">
            <h1 className="m-0 font-db-sans text-[26px] font-medium leading-[1.2] tracking-[-0.015em] text-db-navy-800">
              Agents
            </h1>
            <p className="mt-1 max-w-2xl text-[13px] leading-[1.5] text-db-gray-text">
              Compose, deploy, and manage multi-step agent workflows. Each agent is a typed block
              tree that runs against your governed data and tools.
            </p>
          </div>

          {/* Toolbar */}
          <div className="mb-[14px] flex items-center gap-2.5">
            <div className="flex w-[280px] items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 transition-colors focus-within:border-db-navy-400 focus-within:shadow-db-focus">
              <SearchIcon className="h-3.5 w-3.5 text-db-navy-400" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search agents…"
                className="flex-1 border-0 bg-transparent text-[13px] font-normal text-db-navy-800 outline-none placeholder:text-db-gray-text"
              />
              {search && (
                <button
                  type="button"
                  onClick={() => setSearch('')}
                  aria-label="Clear search"
                  className="text-db-gray-text hover:text-db-navy-800"
                >
                  <CloseIcon className="h-3 w-3" />
                </button>
              )}
            </div>
            <div className="ml-auto flex items-center gap-0 overflow-hidden rounded-db-md border border-db-gray-lines bg-white">
              <button
                type="button"
                title="Grid view"
                onClick={() => setView('grid')}
                className={`px-2.5 py-1.5 ${
                  view === 'grid'
                    ? 'bg-db-oat-medium text-db-navy-800'
                    : 'bg-transparent text-db-gray-text hover:text-db-navy-800'
                }`}
              >
                <GridIcon className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                title="List view"
                onClick={() => setView('list')}
                className={`px-2.5 py-1.5 ${
                  view === 'list'
                    ? 'bg-db-oat-medium text-db-navy-800'
                    : 'bg-transparent text-db-gray-text hover:text-db-navy-800'
                }`}
              >
                <ListIcon className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* States */}
          {isLoading && (
            <div className="flex items-center justify-center py-20 text-[13px] text-db-gray-text">
              Loading agents…
            </div>
          )}
          {error && !isLoading && (
            <div className="flex items-center justify-center py-20 text-[13px] text-db-lava-700">
              {error instanceof Error ? error.message : 'Failed to load agents'}
            </div>
          )}

          {/* Empty state */}
          {!isLoading && !error && items.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-4 rounded-db-md border border-dashed border-db-gray-lines bg-white py-20 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-db-md bg-db-lava-100">
                <BotIcon className="h-5 w-5 text-db-lava-600" />
              </div>
              <div>
                <div className="text-[14px] font-medium text-db-navy-800">No agents yet</div>
                <div className="mt-1 text-[12px] text-db-gray-text">
                  Create your first agent — a typed block tree of researchers, planners, and tools.
                </div>
              </div>
              <button
                type="button"
                onClick={handleCreateNew}
                className="inline-flex items-center gap-1.5 rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700"
              >
                <PlusIcon className="h-3 w-3" /> New Agent
              </button>
            </div>
          )}

          {/* Filtered empty */}
          {!isLoading && !error && items.length > 0 && filtered.length === 0 && (
            <div className="rounded-db-md border border-db-gray-lines bg-white p-10 text-center text-[13px] text-db-gray-text">
              No agents match “{search}”.
            </div>
          )}

          {/* Grid view */}
          {!isLoading && !error && filtered.length > 0 && view === 'grid' && (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(360px,1fr))] gap-3.5">
              {filtered.map((agent) => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  onOpen={handleOpen}
                  onDeleteRequest={handleDeleteRequest}
                />
              ))}
            </div>
          )}

          {/* List view */}
          {!isLoading && !error && filtered.length > 0 && view === 'list' && (
            <div className="overflow-hidden rounded-db-md border border-db-gray-lines bg-white">
              <div className="grid grid-cols-[1.6fr_120px_120px_100px_auto] gap-3.5 border-b border-db-gray-lines bg-db-oat-light px-4 py-2.5 font-db-sans text-[10px] font-semibold uppercase tracking-[0.04em] text-db-gray-text">
                <span>Agent</span>
                <span>Visibility</span>
                <span>Owner</span>
                <span>Updated</span>
                <span />
              </div>
              {filtered.map((agent) => (
                <AgentRow
                  key={agent.id}
                  agent={agent}
                  onOpen={handleOpen}
                  onDeleteRequest={handleDeleteRequest}
                />
              ))}
            </div>
          )}

          {/* CTA banner */}
          {!isLoading && !error && (
            <div className="mt-[22px] flex items-center gap-3.5 rounded-db-md border border-dashed border-db-gray-lines bg-white px-[22px] py-5">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-db-md bg-db-lava-600">
                <SparkleIcon className="h-4 w-4 text-white" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-medium leading-[1.3] text-db-navy-800">
                  Start a new agent from a description
                </div>
                <div className="mt-0.5 text-[12px] leading-[1.5] text-db-gray-text">
                  Describe what you want — Designer Chat will scaffold the block tree, tool
                  bindings, and a starter prompt.
                </div>
              </div>
              <button
                type="button"
                onClick={handleCreateNew}
                className="inline-flex shrink-0 items-center gap-1.5 rounded-db-md bg-db-navy-800 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-navy-900"
              >
                <SparkleIcon className="h-3 w-3" /> Generate with Designer
              </button>
            </div>
          )}
        </div>
      </div>

      {pendingDelete && (
        <DeleteDialog
          agentName={pendingDelete.name}
          isPending={deleteMutation.isPending}
          error={deleteError}
          onConfirm={() => handleDeleteConfirm(false)}
          onForceConfirm={() => handleDeleteConfirm(true)}
          onCancel={() => {
            setPendingDelete(null)
            setDeleteError(null)
          }}
        />
      )}
    </AppShell>
  )
}

export default AgentDesignerListPage

// =====================================================================
// Icons
// =====================================================================

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M5 12h14M12 5v14" />
    </svg>
  )
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  )
}

function MoreIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="5" cy="12" r="1" />
      <circle cx="12" cy="12" r="1" />
      <circle cx="19" cy="12" r="1" />
    </svg>
  )
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  )
}

function GridIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
    </svg>
  )
}

function ListIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  )
}

function FileIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zM14 2v6h6" />
    </svg>
  )
}

function BotIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect x="4" y="8" width="16" height="12" rx="3" />
      <path d="M12 8V4M9 13v1M15 13v1M9 17h6M2 14h2M20 14h2" />
    </svg>
  )
}

function BlocksIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M4 6h16M4 12h16M4 18h16M4 6l3-2M4 12l3-2M4 18l3-2" />
    </svg>
  )
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 6h18M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
      <line x1="10" x2="10" y1="11" y2="17" />
      <line x1="14" x2="14" y1="11" y2="17" />
    </svg>
  )
}

function SparkleIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6L12 3z" />
      <path d="M19 17l.7 2.1L22 20l-2.3.9L19 23l-.7-2.1L16 20l2.3-.9L19 17z" />
    </svg>
  )
}
