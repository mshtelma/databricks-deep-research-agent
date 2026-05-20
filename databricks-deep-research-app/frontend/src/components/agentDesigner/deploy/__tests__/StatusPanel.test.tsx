/**
 * Tests for StatusPanel — verifies status badge rendering for each of the 6
 * statuses + error_message rendering when present + loading/error states.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { StatusPanel } from '../StatusPanel'
import type { DeploymentStatus } from '@/types/deployment'

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  }
}

function mockStatus(
  status: DeploymentStatus,
  errorMessage: string | null = null,
): void {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(
      JSON.stringify({
        status,
        updated_at: '2026-05-09T00:00:00Z',
        error_message: errorMessage,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ),
  )
}

describe('StatusPanel', () => {
  let originalFetch: typeof globalThis.fetch
  beforeEach(() => {
    originalFetch = globalThis.fetch
  })
  afterEach(() => {
    globalThis.fetch = originalFetch
    vi.restoreAllMocks()
  })

  it.each<[DeploymentStatus, RegExp]>([
    ['pending', /Pending/i],
    ['deploying', /Deploying/i],
    ['active', /Active/i],
    ['failed', /Failed/i],
    ['deactivated', /Deactivated/i],
    ['cleanup_failed', /Cleanup failed/i],
  ])('renders %s status badge with the right label', async (status, label) => {
    mockStatus(status)
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <StatusPanel deploymentId="dep-1" />
      </Wrapper>,
    )
    expect(await screen.findByText(label)).toBeInTheDocument()
    expect(
      await screen.findByTestId(`deployment-status-${status}`),
    ).toBeInTheDocument()
  })

  it('renders error_message when present', async () => {
    mockStatus('failed', 'serving endpoint quota exhausted')
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <StatusPanel deploymentId="dep-1" />
      </Wrapper>,
    )
    expect(
      await screen.findByText(/serving endpoint quota exhausted/i),
    ).toBeInTheDocument()
  })

  it('shows error UI when the status fetch fails', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('boom', { status: 500 }),
    )
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <StatusPanel deploymentId="dep-1" />
      </Wrapper>,
    )
    expect(
      await screen.findByText(/Could not load deployment status/i),
    ).toBeInTheDocument()
  })
})
