/**
 * Tests for RevisionList and the revisions_tab_opened metric.
 */

import * as React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import '@testing-library/jest-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

// ---------------------------------------------------------------------------
// Mocks — set up BEFORE any imports that transitively load these modules
// ---------------------------------------------------------------------------

vi.mock('@/api/agentsV2', () => ({
  listRevisions: vi.fn(),
  getRevision: vi.fn(),
  updateAgentV2: vi.fn(),
  createAgentV2: vi.fn(),
  getAgentV2WithEtag: vi.fn(),
  deleteAgentV2: vi.fn(),
  EtagConflictError: class EtagConflictError extends Error {
    current_etag: string
    constructor(current_etag: string, message = 'ETag conflict') {
      super(message)
      this.name = 'EtagConflictError'
      this.current_etag = current_etag
    }
  },
}))

vi.mock('@/lib/clientMetrics', () => ({
  emit: vi.fn(),
  flush: vi.fn(),
  startClientMetricsPipeline: vi.fn(() => () => undefined),
  CLIENT_SIGNAL_NAMES: [
    'block_render_count',
    'dnd_drop_failed',
    'widget_fallback',
    'token_refresh_attempts',
    'token_refresh_failures',
    'revisions_tab_opened',
  ],
}))

vi.mock('@/api/agentDesigner', () => ({
  getRegistry: vi.fn(),
  validateWorkflow: vi.fn(),
  clearRegistryCache: vi.fn(),
}))

vi.mock('@/components/agentDesigner/BlockEditor', () => ({
  BlockEditor: () => <div data-testid="block-editor" />,
}))
vi.mock('@/components/agentDesigner/ConfigPanel', () => ({
  ConfigPanel: () => <div data-testid="config-panel" />,
}))
vi.mock('@/components/agentDesigner/ToolsPanel', () => ({
  ToolsPanel: () => <div data-testid="tools-panel" />,
}))
vi.mock('@/components/agentDesigner/ChatPanel', () => ({
  ChatPanel: () => <div data-testid="chat-panel" />,
}))
vi.mock('@/components/agentDesigner/RevisionPreview', () => ({
  RevisionPreview: ({ revId }: { agentId: string; revId: string }) => (
    <div data-testid={`revision-preview-${revId}`}>Preview {revId}</div>
  ),
}))

// ---------------------------------------------------------------------------
// Imports after mocks
// ---------------------------------------------------------------------------

import { listRevisions, getRevision, updateAgentV2 } from '@/api/agentsV2'
import * as clientMetrics from '@/lib/clientMetrics'
import { getRegistry } from '@/api/agentDesigner'
import { RevisionList } from '../RevisionList'
import { AgentDesignerPage } from '@/pages/AgentDesignerPage'
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore'
import { createDraftWorkflow } from '@/lib/workflowAst'
import type { RegistryResponse, AgentV2Response } from '@/types/agentDesigner'
import type { AST } from '@/types/ast'

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NOW = new Date().toISOString()
const TWO_HOURS_AGO = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()

const FAKE_REVISIONS = [
  {
    rev_id: 'rev-1',
    etag: '"etag-aabbccdd"',
    created_at: TWO_HOURS_AGO,
    created_by: 'alice@example.com',
  },
  {
    rev_id: 'rev-2',
    etag: '"etag-eeff0011"',
    created_at: NOW,
    created_by: 'bob@example.com',
  },
  {
    rev_id: 'rev-3',
    etag: '"etag-22334455"',
    created_at: new Date(Date.now() - 60000).toISOString(),
    created_by: 'carol@example.com',
  },
]

const FAKE_AST: AST = {
  ...createDraftWorkflow('Workflow'),
  root: {
    id: 'root-id',
    type: 'sequence',
    label: 'Workflow',
    config: {},
    children: [],
  },
}

const FAKE_REGISTRY: RegistryResponse = {
  node_types: [
    {
      type: 'sequence',
      label: 'Sequence',
      icon: 'list',
      category: 'composite',
      is_composite: true,
      config_schema: null,
    },
  ],
  agent_subtypes: [],
  tool_kinds: [],
  model_tiers: ['simple', 'analytical', 'complex'],
  version: '1.0.0',
}

const FAKE_AGENT: AgentV2Response = {
  id: 'agent-abc',
  owner_id: 'user-1',
  name: 'Test Agent',
  description: null,
  avatar_url: null,
  visibility: 'private',
  definition: FAKE_AST as unknown as Record<string, unknown>,
  schema_version: 1,
  etag: '"etag-v1"',
  created_at: NOW,
  updated_at: NOW,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeQC(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
}

function renderRevisionList(overrides: Partial<React.ComponentProps<typeof RevisionList>> = {}) {
  const onSelectRevision = vi.fn()
  const props = {
    agentId: 'agent-abc',
    selectedRevId: null,
    onSelectRevision,
    ...overrides,
  }
  const qc = makeQC()
  const result = render(
    <QueryClientProvider client={qc}>
      <RevisionList {...props} />
    </QueryClientProvider>,
  )
  return { ...result, onSelectRevision }
}

function renderPage(path: string, qc?: QueryClient) {
  const client = qc ?? makeQC()
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/designer/:id" element={<AgentDesignerPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks()
  useAgentEditorStore.setState(initialState)
  vi.mocked(getRegistry).mockResolvedValue(FAKE_REGISTRY)
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('RevisionList', () => {
  it('test_renders_empty_state — shows "No prior revisions" when list is empty', async () => {
    vi.mocked(listRevisions).mockResolvedValue({ items: [], total: 0 })

    renderRevisionList()

    await screen.findByText('No prior revisions')
    expect(screen.getByText('No prior revisions')).toBeInTheDocument()
  })

  it('test_renders_rows_with_relative_time — 3 revisions render 3 rows with timestamps', async () => {
    vi.mocked(listRevisions).mockResolvedValue({ items: FAKE_REVISIONS, total: 3 })

    renderRevisionList()

    // Wait for rows to appear (any of the authors)
    await screen.findByText('alice@example.com')
    expect(screen.getByText('alice@example.com')).toBeInTheDocument()
    expect(screen.getByText('bob@example.com')).toBeInTheDocument()
    expect(screen.getByText('carol@example.com')).toBeInTheDocument()

    // Timestamps: "2h ago" for TWO_HOURS_AGO, "0s ago" or similar for NOW
    expect(screen.getByText('2h ago')).toBeInTheDocument()

    // All 3 rows present
    const rows = screen.getAllByRole('row')
    expect(rows).toHaveLength(3)
  })

  it('test_click_revision_triggers_preview — clicking a row calls onSelectRevision with rev_id', async () => {
    vi.mocked(listRevisions).mockResolvedValue({ items: FAKE_REVISIONS, total: 3 })

    const { onSelectRevision } = renderRevisionList()

    await screen.findByText('alice@example.com')

    // Click the first row
    const rows = screen.getAllByRole('row')
    fireEvent.click(rows[0]!)

    expect(onSelectRevision).toHaveBeenCalledTimes(1)
    expect(onSelectRevision).toHaveBeenCalledWith('rev-1')
  })

  it('test_restore_button_calls_updateAgentV2 — clicking Restore calls updateAgentV2 with revision definition', async () => {
    vi.mocked(listRevisions).mockResolvedValue({ items: [FAKE_REVISIONS[0]!], total: 1 })
    vi.mocked(getRevision).mockResolvedValue({
      ...FAKE_REVISIONS[0]!,
      definition: FAKE_AST,
    })
    vi.mocked(updateAgentV2).mockResolvedValue({ agent: FAKE_AGENT, etag: '"etag-v2"' })

    // Set etag in store so the restore call uses it
    useAgentEditorStore.setState({ ...initialState, etag: '"etag-v1"' })

    renderRevisionList()

    await screen.findByText('alice@example.com')

    const restoreBtn = screen.getByRole('button', { name: /restore revision rev-1/i })
    fireEvent.click(restoreBtn)

    await waitFor(() => {
      expect(updateAgentV2).toHaveBeenCalledTimes(1)
    })

    const [calledId, calledReq, calledEtag] = vi.mocked(updateAgentV2).mock.calls[0]!
    expect(calledId).toBe('agent-abc')
    expect(calledReq.definition).toEqual(FAKE_AST)
    expect(calledEtag).toBe('"etag-v1"')
  })

  it('test_revisions_tab_opened_metric_emitted — switching to Revisions tab emits metric exactly once', async () => {
    // getAgentV2WithEtag needed for edit flow
    const { getAgentV2WithEtag } = await import('@/api/agentsV2')
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({ agent: FAKE_AGENT, etag: '"etag-v1"' })
    vi.mocked(listRevisions).mockResolvedValue({ items: [], total: 0 })

    renderPage('/designer/agent-abc')

    // Wait for editor to load
    await screen.findByTestId('block-editor')

    // metric should not have been emitted yet
    expect(clientMetrics.emit).not.toHaveBeenCalledWith('revisions_tab_opened', undefined, expect.anything())

    // Switch to Revisions tab
    const revisionsTab = screen.getByRole('tab', { name: /revisions/i })
    fireEvent.click(revisionsTab)

    await waitFor(() => {
      expect(clientMetrics.emit).toHaveBeenCalledWith('revisions_tab_opened', undefined, { agent_id: 'agent-abc' })
    })

    expect(clientMetrics.emit).toHaveBeenCalledTimes(1)
  })
})

describe('AgentDesignerPage — revisions tab', () => {
  it('test_revisions_tab_renders_list — switching to Revisions tab renders RevisionList', async () => {
    const { getAgentV2WithEtag } = await import('@/api/agentsV2')
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({ agent: FAKE_AGENT, etag: '"etag-v1"' })
    vi.mocked(listRevisions).mockResolvedValue({ items: [], total: 0 })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    fireEvent.click(screen.getByRole('tab', { name: /revisions/i }))

    // RevisionList renders (empty state)
    await screen.findByText('No prior revisions')
    expect(screen.getByText('No prior revisions')).toBeInTheDocument()
  })

  it('test_revisions_tab_metric_emitted_once — opening tab twice emits metric twice (once per open transition)', async () => {
    const { getAgentV2WithEtag } = await import('@/api/agentsV2')
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({ agent: FAKE_AGENT, etag: '"etag-v1"' })
    vi.mocked(listRevisions).mockResolvedValue({ items: [], total: 0 })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    const revisionsTab = screen.getByRole('tab', { name: /revisions/i })
    const editTab = screen.getByRole('tab', { name: /^edit$/i })

    // First open
    fireEvent.click(revisionsTab)
    await waitFor(() => {
      expect(clientMetrics.emit).toHaveBeenCalledWith('revisions_tab_opened', undefined, { agent_id: 'agent-abc' })
    })
    expect(vi.mocked(clientMetrics.emit)).toHaveBeenCalledTimes(1)

    // Go back to Edit
    fireEvent.click(editTab)

    // Second open
    fireEvent.click(revisionsTab)
    await waitFor(() => {
      expect(vi.mocked(clientMetrics.emit)).toHaveBeenCalledTimes(2)
    })
  })
})
