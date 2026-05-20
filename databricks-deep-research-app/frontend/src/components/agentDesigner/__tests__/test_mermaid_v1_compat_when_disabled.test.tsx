/**
 * Tests for MermaidPreview — V1 compatibility when the server-mode flag is
 * disabled (default) vs. enabled.
 *
 * Strategy
 * --------
 * * ``VITE_AGENT_DESIGNER_MERMAID_SERVER`` is read at module import time as a
 *   compile-time constant in production, but Vitest executes in Node where
 *   ``import.meta.env`` is mutable.  We manipulate the env object **before**
 *   (re-)importing the component so each describe block sees the right value.
 *
 * * The ``mermaid`` package is mocked so tests never invoke a real renderer;
 *   this also ensures the dynamic import does not fail in jsdom.
 *
 * * ``fetch`` is replaced with ``vi.fn()`` so we can assert call counts and
 *   control the returned text without a real network.
 */

import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import '@testing-library/jest-dom'

// ---------------------------------------------------------------------------
// Mock the mermaid package (dynamic import inside MermaidPreview)
// ---------------------------------------------------------------------------

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg data-testid="mermaid-svg" />' }),
  },
}))

// ---------------------------------------------------------------------------
// Helper to (re)load the component module with a specific env value.
//
// Vitest module cache is cleared between describe blocks via vi.resetModules()
// so each block gets a fresh MermaidPreview that reads the env at import time.
// ---------------------------------------------------------------------------

async function loadComponent(serverMode: string | undefined): Promise<{
  MermaidPreview: typeof import('../MermaidPreview').MermaidPreview
}> {
  // Stub the env value before the module is loaded.
  // vi.stubEnv is the Vitest-idiomatic way to mutate import.meta.env without
  // violating TypeScript's readonly constraint on the generated interface.
  if (serverMode !== undefined) {
    vi.stubEnv('VITE_AGENT_DESIGNER_MERMAID_SERVER', serverMode)
  } else {
    vi.unstubAllEnvs()
  }
  vi.resetModules()
  const mod = await import('../MermaidPreview')
  return { MermaidPreview: mod.MermaidPreview }
}

// ---------------------------------------------------------------------------
// describe 1: flag NOT set — local-fallback mode
// ---------------------------------------------------------------------------

describe('MermaidPreview — server flag DISABLED (default)', () => {
  let MermaidPreview: Awaited<ReturnType<typeof loadComponent>>['MermaidPreview']

  beforeEach(async () => {
    vi.stubGlobal('fetch', vi.fn())
    ;({ MermaidPreview } = await loadComponent(undefined))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('renders the local-fallback placeholder without fetching /mermaid', () => {
    render(<MermaidPreview agentId="some-agent-id" />)

    // Placeholder must be visible
    expect(screen.getByTestId('mermaid-preview-placeholder')).toBeInTheDocument()

    // No fetch calls to the mermaid endpoint
    expect(vi.mocked(fetch)).not.toHaveBeenCalled()
  })

  it('renders placeholder even when agentId is undefined', () => {
    render(<MermaidPreview />)
    expect(screen.getByTestId('mermaid-preview-placeholder')).toBeInTheDocument()
    expect(vi.mocked(fetch)).not.toHaveBeenCalled()
  })
})

// ---------------------------------------------------------------------------
// describe 2: flag SET to '1' — server mode
// ---------------------------------------------------------------------------

describe('MermaidPreview — server flag ENABLED', () => {
  let MermaidPreview: Awaited<ReturnType<typeof loadComponent>>['MermaidPreview']

  const SAMPLE_MERMAID = [
    'flowchart TD',
    '  subgraph test_agent',
    '    root["Agent"]',
    '  end',
  ].join('\n')

  beforeEach(async () => {
    // Stub fetch to return a valid Mermaid document
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(SAMPLE_MERMAID),
      }),
    )
    ;({ MermaidPreview } = await loadComponent('1'))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('makes exactly one fetch to /agents-v2/{id}/mermaid when agentId is set', async () => {
    render(<MermaidPreview agentId="agent-uuid-123" />)

    await waitFor(() => {
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    })

    const [url] = vi.mocked(fetch).mock.calls[0] as [string, ...unknown[]]
    expect(url).toMatch(/\/agents-v2\/agent-uuid-123\/mermaid$/)
  })

  it('does NOT fetch when agentId is not provided', () => {
    render(<MermaidPreview />)
    // No fetch — no agentId means nothing to load
    expect(vi.mocked(fetch)).not.toHaveBeenCalled()
  })

  it('shows loading state initially then diagram container after fetch', async () => {
    // Delay the fetch slightly so the loading state is observable
    vi.mocked(fetch).mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                ok: true,
                text: () => Promise.resolve(SAMPLE_MERMAID),
              } as Response),
            10,
          ),
        ),
    )

    render(<MermaidPreview agentId="agent-uuid-456" />)

    // Loading indicator should appear immediately
    expect(screen.getByTestId('mermaid-preview-loading')).toBeInTheDocument()

    // After fetch resolves, the diagram container should replace it
    await waitFor(() => {
      expect(screen.queryByTestId('mermaid-preview-loading')).not.toBeInTheDocument()
    })
  })

  it('shows error banner when fetch returns non-OK status', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 404,
      text: () => Promise.resolve('not found'),
    } as Response)

    render(<MermaidPreview agentId="missing-agent" />)

    await waitFor(() => {
      expect(screen.getByTestId('mermaid-preview-error')).toBeInTheDocument()
    })
    expect(screen.getByRole('alert')).toHaveTextContent('HTTP 404')
  })
})
