import '@testing-library/jest-dom/vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

// ---------------------------------------------------------------------------
// Mock the API module before importing the page
// ---------------------------------------------------------------------------

vi.mock('@/api/agentsV2', () => ({
  listAgentsV2: vi.fn(),
  getAgentV2WithEtag: vi.fn(),
  createAgentV2: vi.fn(),
  updateAgentV2: vi.fn(),
  deleteAgentV2: vi.fn(),
  parseAgentDeleteError: vi.fn((error: unknown) =>
    error && typeof error === 'object' && 'error_kind' in error ? error : null,
  ),
  EtagConflictError: class EtagConflictError extends Error {},
}))

// mock useNavigate so we can assert on calls
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

import { listAgentsV2, deleteAgentV2 } from '@/api/agentsV2'
import { AgentDesignerListPage } from '../AgentDesignerListPage'
import type { AgentV2ListResponse } from '@/types/agentDesigner'

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function withProviders(ui: ReactNode, initialEntries = ['/designer']) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={initialEntries}>{ui}</MemoryRouter>
    </QueryClientProvider>
  )
}

const EMPTY_LIST: AgentV2ListResponse = { items: [], total: 0 }

const TWO_AGENTS: AgentV2ListResponse = {
  items: [
    {
      id: 'ffc304b0-a67d-458a-8fb2-7a433cd36107',
      name: 'Alpha Agent',
      description: 'First test agent',
      visibility: 'private',
      owner_id: 'user-abc-123',
      updated_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2h ago
      node_count: 3,
      in_app_active: false,
    },
    {
      id: 'a0d7eaf0-7e8f-4a1a-9db1-bf48652dfa14',
      name: 'Beta Agent',
      description: null,
      visibility: 'workspace',
      owner_id: 'user-xyz-456',
      updated_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5m ago
      node_count: 1,
      in_app_active: true,
    },
  ],
  total: 2,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AgentDesignerListPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders empty state when list is empty', async () => {
    vi.mocked(listAgentsV2).mockResolvedValueOnce(EMPTY_LIST)

    render(withProviders(<AgentDesignerListPage />))

    await screen.findByText('No agents yet')
    expect(screen.getByText('No agents yet')).toBeInTheDocument()
    // The header "New Agent" button + the empty-state "New Agent" button.
    const createButtons = screen.getAllByRole('button', { name: /new agent/i })
    expect(createButtons.length).toBeGreaterThanOrEqual(1)
  })

  it('renders rows when list contains agents', async () => {
    vi.mocked(listAgentsV2).mockResolvedValueOnce(TWO_AGENTS)

    render(withProviders(<AgentDesignerListPage />))

    await screen.findByText('Alpha Agent')
    expect(screen.getByText('Alpha Agent')).toBeInTheDocument()
    expect(screen.getByText('Beta Agent')).toBeInTheDocument()

    // Visibility badges
    expect(screen.getByText('Private')).toBeInTheDocument()
    expect(screen.getByText('Workspace')).toBeInTheDocument()

    // Agent IDs (NOT owner IDs) shown in the footer, truncated to 12 chars.
    // The full UUID is in the title attribute for hover-to-copy.
    expect(screen.getAllByText('ffc304b0-a67')).toHaveLength(1)
    expect(screen.getAllByText('a0d7eaf0-7e8')).toHaveLength(1)
    // Owner IDs MUST NOT be rendered — single-user workspaces would otherwise
    // see the same opaque user_id repeated on every card (the bug this fixes).
    expect(screen.queryByText('user-abc-123')).not.toBeInTheDocument()
    expect(screen.queryByText('user-xyz-456')).not.toBeInTheDocument()
  })

  it('clicking an agent card navigates to /designer/:id', async () => {
    vi.mocked(listAgentsV2).mockResolvedValueOnce(TWO_AGENTS)

    render(withProviders(<AgentDesignerListPage />))

    // The whole card is clickable — find it by the agent name text and walk up.
    const nameEl = await screen.findByText('Alpha Agent')
    const card = nameEl.closest('[class*="cursor-pointer"]') as HTMLElement | null
    expect(card).not.toBeNull()
    fireEvent.click(card!)

    expect(mockNavigate).toHaveBeenCalledWith('/designer/ffc304b0-a67d-458a-8fb2-7a433cd36107')
  })

  it('Delete confirm calls deleteAgentV2 with the correct id', async () => {
    vi.mocked(listAgentsV2).mockResolvedValue(TWO_AGENTS)
    vi.mocked(deleteAgentV2).mockResolvedValue(undefined)

    render(withProviders(<AgentDesignerListPage />))

    // Open the overflow menu on the first card
    const moreButton = await screen.findByRole('button', { name: /actions for alpha agent/i })
    fireEvent.click(moreButton)

    // Click the Delete menu item (a button containing "Delete" text inside the open menu)
    const deleteMenuItem = await screen.findByRole('button', { name: /^delete$/i })
    fireEvent.click(deleteMenuItem)

    // Confirmation dialog should appear
    await screen.findByRole('dialog')

    // Find the confirm Delete button inside the dialog footer
    const dialog = screen.getByRole('dialog')
    const confirmBtn = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent?.trim().toLowerCase() === 'delete',
    )
    expect(confirmBtn).toBeDefined()
    fireEvent.click(confirmBtn!)

    await waitFor(() => {
      expect(deleteAgentV2).toHaveBeenCalledWith('ffc304b0-a67d-458a-8fb2-7a433cd36107')
    })
  })

  it('normal delete success closes the confirmation dialog', async () => {
    vi.mocked(listAgentsV2).mockResolvedValue(TWO_AGENTS)
    vi.mocked(deleteAgentV2).mockResolvedValue(undefined)

    render(withProviders(<AgentDesignerListPage />))

    fireEvent.click(await screen.findByRole('button', { name: /actions for alpha agent/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^delete$/i }))
    await screen.findByRole('dialog')

    const dialog = screen.getByRole('dialog')
    const confirmBtn = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent?.trim().toLowerCase() === 'delete',
    )
    fireEvent.click(confirmBtn!)

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })
  })

  it('active-deployment delete error keeps dialog open and force deletes explicitly', async () => {
    vi.mocked(listAgentsV2).mockResolvedValue(TWO_AGENTS)
    const activeError = Object.assign(new Error('Active deployments exist'), {
      error_kind: 'active_deployments_exist',
      active_count: 1,
      deployments: [
        {
          id: 'deployment-1',
          mode: 'shell_app',
          status: 'active',
          endpoint_name: 'dr-shell-alpha',
        },
      ],
    })
    vi.mocked(deleteAgentV2)
      .mockRejectedValueOnce(activeError)
      .mockResolvedValueOnce(undefined)

    render(withProviders(<AgentDesignerListPage />))

    fireEvent.click(await screen.findByRole('button', { name: /actions for alpha agent/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^delete$/i }))
    await screen.findByRole('dialog')

    const dialog = screen.getByRole('dialog')
    const confirmBtn = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent?.trim().toLowerCase() === 'delete',
    )
    fireEvent.click(confirmBtn!)

    await screen.findByText(/active deployments block deletion/i)
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole('button', { name: /delete and deactivate deployments/i }),
    )

    await waitFor(() => {
      expect(deleteAgentV2).toHaveBeenLastCalledWith('ffc304b0-a67d-458a-8fb2-7a433cd36107', { force: true })
    })
  })

  it('generic delete errors do not close the dialog silently', async () => {
    vi.mocked(listAgentsV2).mockResolvedValue(TWO_AGENTS)
    vi.mocked(deleteAgentV2).mockRejectedValue(new Error('database is busy'))

    render(withProviders(<AgentDesignerListPage />))

    fireEvent.click(await screen.findByRole('button', { name: /actions for alpha agent/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^delete$/i }))
    await screen.findByRole('dialog')

    const dialog = screen.getByRole('dialog')
    const confirmBtn = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent?.trim().toLowerCase() === 'delete',
    )
    fireEvent.click(confirmBtn!)

    await screen.findByText('database is busy')
    expect(screen.getByRole('dialog')).toBeInTheDocument()
  })
})
