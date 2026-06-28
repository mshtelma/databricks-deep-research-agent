/**
 * AgentDesignerPage tests — vitest + @testing-library/react
 *
 * Each test wraps the component in MemoryRouter + QueryClientProvider.
 * API modules are fully mocked. Zustand store is reset in beforeEach.
 */

import '@testing-library/jest-dom/vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// ---------------------------------------------------------------------------
// Mock API modules BEFORE importing the page
// ---------------------------------------------------------------------------

vi.mock('@/api/agentsV2', () => {
  class EtagConflictError extends Error {
    current_etag: string
    constructor(current_etag: string, message = 'ETag conflict') {
      super(message)
      this.name = 'EtagConflictError'
      this.current_etag = current_etag
    }
  }
  class AgentCriticError extends Error {
    critique: Record<string, unknown> | null
    constructor(critique: Record<string, unknown> | null, message = 'Critic blocked') {
      super(message)
      this.name = 'AgentCriticError'
      this.critique = critique
    }
  }
  return {
    getAgentV2WithEtag: vi.fn(),
    createAgentV2: vi.fn(),
    updateAgentV2: vi.fn(),
    listRevisions: vi.fn(),
    getRevision: vi.fn(),
    deleteAgentV2: vi.fn(),
    EtagConflictError,
    AgentCriticError,
    parseAgentCriticError: (error: unknown) =>
      error instanceof AgentCriticError ? error : null,
  }
})

vi.mock('@/api/agentDesigner', () => ({
  getRegistry: vi.fn(),
  validateWorkflow: vi.fn(),
  clearRegistryCache: vi.fn(),
}))

// Mock heavy child components so tests stay fast and focused on page logic
vi.mock('@/components/agentDesigner/BlockEditor', () => ({
  BlockEditor: () => <div data-testid="block-editor" />,
}))
vi.mock('@/components/agentDesigner/ConfigPanel', () => ({
  ConfigPanel: () => <div data-testid="config-panel" />,
}))
vi.mock('@/components/agentDesigner/ToolsPanel', () => ({
  ToolsPanel: () => <div data-testid="tools-panel" />,
}))
// The separate Designer Chat column was removed in the Direction 2 redesign —
// the co-pilot is now a tab *inside* ConfigPanel (which is mocked as a stub
// here), so the page no longer renders ChatPanel directly.

// ---------------------------------------------------------------------------
// Imports (after mocks are set up)
// ---------------------------------------------------------------------------

import { getAgentV2WithEtag, updateAgentV2, EtagConflictError, AgentCriticError } from '@/api/agentsV2'
import { getRegistry, validateWorkflow } from '@/api/agentDesigner'
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore'
import { createDraftWorkflow } from '@/lib/workflowAst'
import { AgentDesignerPage } from '../AgentDesignerPage'
import type { RegistryResponse, AgentV2Response } from '@/types/agentDesigner'
import type { AST } from '@/types/ast'

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

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
    {
      type: 'agent',
      label: 'Agent',
      icon: 'bot',
      category: 'leaf',
      is_composite: false,
      config_schema: null,
    },
  ],
  agent_subtypes: [],
  tool_kinds: [],
  model_tiers: ['simple', 'analytical', 'complex'],
  version: '1.0.0',
}

const FAKE_AST: AST = {
  ...createDraftWorkflow('Workflow'),
  root: {
    id: 'root-id',
    type: 'sequence',
    label: 'Workflow',
    config: {},
    children: [
      {
        id: 'agent-id',
        type: 'agent',
        label: 'Researcher',
        config: { subtype: 'researcher' },
        children: [],
      },
    ],
  },
}

const FAKE_AGENT: AgentV2Response = {
  id: 'agent-abc',
  owner_id: 'user-1',
  name: 'Test Agent',
  description: 'A test agent',
  avatar_url: null,
  visibility: 'private',
  definition: FAKE_AST as unknown as Record<string, unknown>,
  schema_version: 1,
  etag: '"etag-v1"',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
}

function renderPage(path: string, qc?: QueryClient): ReturnType<typeof render> {
  const client = qc ?? makeQueryClient()
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/designer/:id" element={<AgentDesignerPage />} />
          {/* Sentinel route used by Test-run tests to verify navigation. */}
          <Route path="/chat" element={<div data-testid="chat-route-sentinel" />} />
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
  // Reset zustand store to initial state before each test
  useAgentEditorStore.setState(initialState)

  // Default mock for getRegistry — most tests need it
  vi.mocked(getRegistry).mockResolvedValue(FAKE_REGISTRY)
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AgentDesignerPage', () => {
  // Test 1: new flow initializes empty AST and renders the editor
  it('new flow initializes default AST and renders editor', async () => {
    renderPage('/designer/new')

    // Wait for registry to load
    await screen.findByTestId('block-editor')

    // Check the inspector renders. The dedicated ToolsPanel was removed in the
    // Databricks redesign (its workspace-tools role lives inside ConfigPanel's
    // no-selection view), and the separate Designer Chat column was folded into
    // ConfigPanel as a "Co-pilot" tab — so the inspector is the only docked
    // side panel the page renders now.
    expect(screen.getByTestId('config-panel')).toBeInTheDocument()

    // Store should have a default AST (sequence root)
    const { ast } = useAgentEditorStore.getState()
    expect(ast).not.toBeNull()
    expect(ast?.root.type).toBe('sequence')
    expect(ast?.tools).toHaveLength(0)
  })

  // Test 2: edit flow loads agent via getAgentV2WithEtag, asserts AST in store
  it('edit flow calls getAgentV2WithEtag and loads agent into store', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })

    renderPage('/designer/agent-abc')

    // Wait for panels to appear (registry + agent both loaded)
    await screen.findByTestId('block-editor')

    expect(getAgentV2WithEtag).toHaveBeenCalledWith('agent-abc')

    // Agent should be loaded into the store
    await waitFor(() => {
      const { ast, etag, agentId } = useAgentEditorStore.getState()
      expect(ast).not.toBeNull()
      expect(etag).toBe('"etag-v1"')
      expect(agentId).toBe('agent-abc')
    })
  })

  // Test 3: Save calls validateWorkflow first, then updateAgentV2
  it('Save calls validateWorkflow before updateAgentV2', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: true,
      errors: [],
      workflow_summary: null,
    })
    vi.mocked(updateAgentV2).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v2"',
    })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    // Mark as dirty so the Save button is enabled
    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const saveBtn = await screen.findByRole('button', { name: /save agent/i })
    expect(saveBtn).not.toBeDisabled()

    fireEvent.click(saveBtn)

    await waitFor(() => {
      expect(validateWorkflow).toHaveBeenCalled()
      expect(updateAgentV2).toHaveBeenCalled()
    })

    // Verify call order: validateWorkflow before updateAgentV2
    const validateOrder = vi.mocked(validateWorkflow).mock.invocationCallOrder[0]!
    const updateOrder = vi.mocked(updateAgentV2).mock.invocationCallOrder[0]!
    expect(validateOrder).toBeLessThan(updateOrder)
  })

  // Test 4: Invalid AST stores validation errors and does NOT call updateAgentV2
  it('Save with invalid AST stores errors and skips updateAgentV2', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: false,
      errors: [{ message: 'root is required', path: null, line: null, kind: 'validation' }],
      workflow_summary: null,
    })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    // Mark dirty to enable Save
    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const saveBtn = await screen.findByRole('button', { name: /save agent/i })
    fireEvent.click(saveBtn)

    await waitFor(() => {
      expect(validateWorkflow).toHaveBeenCalled()
    })

    // updateAgentV2 must NOT have been called
    expect(updateAgentV2).not.toHaveBeenCalled()

    // Validation errors should be stored
    await waitFor(() => {
      const { validationErrors } = useAgentEditorStore.getState()
      expect(validationErrors).toHaveLength(1)
      expect(validationErrors[0]?.message).toBe('root is required')
    })

    // Error badge should appear in the header
    expect(await screen.findByText(/1 error/i)).toBeInTheDocument()
  })

  // Test 5: 409 from updateAgentV2 opens EtagConflictModal
  it('409 from save opens EtagConflictModal', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: true,
      errors: [],
      workflow_summary: null,
    })
    vi.mocked(updateAgentV2).mockRejectedValue(
      new EtagConflictError('"etag-server"'),
    )

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    // Mark dirty
    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const saveBtn = await screen.findByRole('button', { name: /save agent/i })
    fireEvent.click(saveBtn)

    // Modal should appear with the conflict title
    await screen.findByText('Agent was modified elsewhere')
    expect(screen.getByText(/Another user or session modified this agent/)).toBeInTheDocument()

    // All three action buttons should be visible
    expect(screen.getByRole('button', { name: /reload agent from server/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /force overwrite remote agent/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument()
  })

  // Test 5b: AgentCriticError from updateAgentV2 shows a visible error banner
  it('AgentCriticError from save is surfaced as a visible banner', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: true,
      errors: [],
      workflow_summary: null,
    })
    vi.mocked(updateAgentV2).mockRejectedValue(
      new AgentCriticError({ verdict: 'fail', summary: 'off-topic workflow' } as import('@/api/agentsV2').WorkflowValidationResult),
    )

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const saveBtn = await screen.findByRole('button', { name: /save agent/i })
    fireEvent.click(saveBtn)

    // A visible alert banner containing the critic summary must appear
    await waitFor(() => {
      const alerts = screen.getAllByRole('alert')
      const saveAlert = alerts.find((el) => el.textContent?.includes('off-topic workflow'))
      expect(saveAlert).toBeTruthy()
    })
  })

  // Test 6: Save button is disabled when !isDirty (existing agent with no changes)
  it('Save button is disabled when agent is clean (no unsaved changes)', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    // Wait for agent to load into store (isDirty = false after load)
    await waitFor(() => {
      expect(useAgentEditorStore.getState().agentId).toBe('agent-abc')
    })

    const saveBtn = screen.getByRole('button', { name: /save agent/i })
    expect(saveBtn).toBeDisabled()
  })

  // Test 7: Test run on saved (clean) agent writes localStorage and navigates to /chat
  it('Test run writes selected agent id to localStorage and navigates to /chat', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })

    // The vitest jsdom localStorage shim is partial — install an in-memory
    // stub that fully implements the contract for this test.
    const store = new Map<string, string>()
    const setItem = vi.fn((key: string, value: string) => {
      store.set(key, value)
    })
    const localStorageStub = {
      getItem: (key: string) => store.get(key) ?? null,
      setItem,
      removeItem: (key: string) => {
        store.delete(key)
      },
      clear: () => store.clear(),
      key: (i: number) => Array.from(store.keys())[i] ?? null,
      get length() {
        return store.size
      },
    }
    const originalLocalStorage = Object.getOwnPropertyDescriptor(window, 'localStorage')
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: localStorageStub,
    })

    try {
      renderPage('/designer/agent-abc')
      await screen.findByTestId('block-editor')

      // Wait for clean state
      await waitFor(() => {
        expect(useAgentEditorStore.getState().agentId).toBe('agent-abc')
      })

      const runBtn = screen.getByRole('button', { name: /test run agent/i })
      expect(runBtn).not.toBeDisabled()
      fireEvent.click(runBtn)

      // Should navigate to /chat (sentinel renders) and the localStorage key
      // is set to the agent id so MessageInput picks it up.
      await screen.findByTestId('chat-route-sentinel')
      expect(setItem).toHaveBeenCalledWith('deep-research-selected-agent', 'agent-abc')
    } finally {
      if (originalLocalStorage) {
        Object.defineProperty(window, 'localStorage', originalLocalStorage)
      }
    }
  })

  // Test 8: Test run on a dirty agent saves first then navigates
  it('Test run saves a dirty agent before navigating', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: true,
      errors: [],
      workflow_summary: null,
    })
    vi.mocked(updateAgentV2).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v2"',
    })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    // Mark dirty so the run handler will trigger a save first
    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const runBtn = await screen.findByRole('button', { name: /test run agent/i })
    fireEvent.click(runBtn)

    // Save was called first, then navigation happened
    await waitFor(() => {
      expect(updateAgentV2).toHaveBeenCalled()
    })
    await screen.findByTestId('chat-route-sentinel')
  })

  // Test 9: Test run is disabled on a brand-new draft with no name
  it('Test run is disabled on /designer/new until a name is entered', async () => {
    renderPage('/designer/new')
    await screen.findByTestId('block-editor')

    const runBtn = await screen.findByRole('button', { name: /test run agent/i })
    expect(runBtn).toBeDisabled()
  })

  // Tests 10-12 tested the legacy visibility-flip Deploy button which was
  // removed in D1. The DeployDropdown (new deploy path) is tested via its
  // own component tests in components/agentDesigner/deploy/__tests__/.
  // Chat-picker visibility is now handled by the D2-shim in deployments.py.

  // ---------------------------------------------------------------------------
  // US-301: validation directives in saveNotice banner
  // ---------------------------------------------------------------------------

  // Test US-301-a: advisory save with directives renders each directive line
  it('advisory save with directives renders directive lines in the banner', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: true,
      errors: [],
      workflow_summary: null,
    })

    const validationResult = {
      verdict: 'needs_revision',
      summary: 'Researcher lacks a tool',
      directives: [
        {
          node_path: 'root.children.0',
          issue: 'No web_search tool bound',
          suggested_action: 'Bind web_search to the Researcher agent',
          severity: 'blocking' as const,
          tool_hint: 'web_search',
        },
        {
          node_path: 'root.children.1',
          issue: 'Missing system prompt',
          suggested_action: 'Add a user_prompt_template',
          severity: 'advisory' as const,
          tool_hint: null,
        },
      ],
      semantic_hash: 'abc',
      intent_hash: 'def',
      validator_version: '1.0',
      source: 'fresh' as const,
      cache_hit: false,
      cacheable: true,
    } satisfies import('@/api/agentsV2').WorkflowValidationResult

    vi.mocked(updateAgentV2).mockResolvedValue({
      agent: { ...FAKE_AGENT, validation: validationResult },
      etag: '"etag-v2"',
    })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const saveBtn = await screen.findByRole('button', { name: /save agent/i })
    fireEvent.click(saveBtn)

    // The advisory banner should appear
    await waitFor(() => {
      const alerts = screen.getAllByRole('alert')
      const banner = alerts.find((el) => el.textContent?.includes('needs_revision'))
      expect(banner).toBeTruthy()
    })

    // Directive lines should be rendered in the banner
    expect(await screen.findByText(/No web_search tool bound/)).toBeInTheDocument()
    expect(await screen.findByText(/Bind web_search to the Researcher agent/)).toBeInTheDocument()
    expect(await screen.findByText(/Missing system prompt/)).toBeInTheDocument()
  })

  // Test US-301-b: "Ask designer to fix" seeds the store and dismisses the banner
  it('"Ask designer to fix" seeds agentEditorStore.pendingChatSeed and dismisses banner', async () => {
    vi.mocked(getAgentV2WithEtag).mockResolvedValue({
      agent: FAKE_AGENT,
      etag: '"etag-v1"',
    })
    vi.mocked(validateWorkflow).mockResolvedValue({
      valid: true,
      errors: [],
      workflow_summary: null,
    })

    const validationResult = {
      verdict: 'needs_revision',
      summary: 'Researcher lacks a tool',
      directives: [
        {
          node_path: 'root.children.0',
          issue: 'No web_search tool bound',
          suggested_action: 'Bind web_search to the Researcher agent',
          severity: 'blocking' as const,
          tool_hint: null,
        },
      ],
      semantic_hash: 'abc',
      intent_hash: 'def',
      validator_version: '1.0',
      source: 'fresh' as const,
      cache_hit: false,
      cacheable: true,
    } satisfies import('@/api/agentsV2').WorkflowValidationResult

    vi.mocked(updateAgentV2).mockResolvedValue({
      agent: { ...FAKE_AGENT, validation: validationResult },
      etag: '"etag-v2"',
    })

    renderPage('/designer/agent-abc')
    await screen.findByTestId('block-editor')

    act(() => {
      useAgentEditorStore.getState().setAst(FAKE_AST)
    })

    const saveBtn = await screen.findByRole('button', { name: /save agent/i })
    fireEvent.click(saveBtn)

    // Wait for the advisory banner
    const fixBtn = await screen.findByRole('button', { name: /ask designer to fix/i })
    expect(fixBtn).toBeInTheDocument()

    fireEvent.click(fixBtn)

    // Banner should be dismissed
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /ask designer to fix/i })).toBeNull()
    })

    // pendingChatSeed should be set in the store with verdict + summary + directives
    const { pendingChatSeed } = useAgentEditorStore.getState()
    expect(pendingChatSeed).toContain('needs_revision')
    expect(pendingChatSeed).toContain('Researcher lacks a tool')
    expect(pendingChatSeed).toContain('[root.children.0]')
    expect(pendingChatSeed).toContain('No web_search tool bound')
    expect(pendingChatSeed).toContain('Bind web_search to the Researcher agent')
  })
})
