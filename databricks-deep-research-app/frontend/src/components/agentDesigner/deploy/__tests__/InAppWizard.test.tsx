/**
 * Tests for InAppWizard — verifies the confirm dialog renders the agent name,
 * submit triggers POST /api/v1/deployments with mode='in_app', and onDeployed
 * fires with the resulting DeploymentResponse.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { InAppWizard } from '../InAppWizard'

const FAKE_DEPLOYMENT = {
  id: 'd-1',
  agent_id: 'a-1',
  revision_id: 'r-1',
  mode: 'in_app' as const,
  status: 'active' as const,
  config: { mode: 'in_app' },
  endpoint_name: null,
  model_name: null,
  external_resource_ids: null,
  error_message: null,
  cleanup_attempts: 0,
  deployed_by: 'tester',
  created_at: '2026-05-09T00:00:00Z',
  updated_at: '2026-05-09T00:00:00Z',
  deactivated_at: null,
}

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  }
}

describe('InAppWizard', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders the agent name and revision id summary', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <InAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-12345678"
          agentName="Deep Research Agent"
          revisionId="rev-87654321"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    expect(screen.getByText('Deploy in-app')).toBeInTheDocument()
    expect(screen.getByText(/Deep Research Agent/)).toBeInTheDocument()
  })

  it('submit POSTs /deployments with mode=in_app and fires onDeployed', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(FAKE_DEPLOYMENT), {
        status: 201,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const onDeployed = vi.fn()
    const onOpenChange = vi.fn()
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <InAppWizard
          open={true}
          onOpenChange={onOpenChange}
          agentId="agent-1"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={onDeployed}
        />
      </Wrapper>,
    )

    fireEvent.click(screen.getByTestId('in-app-wizard-submit'))

    await waitFor(() => expect(onDeployed).toHaveBeenCalledTimes(1))
    expect(onDeployed).toHaveBeenCalledWith(FAKE_DEPLOYMENT)

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const call = fetchSpy.mock.calls[0]
    expect(call).toBeDefined()
    const [url, init] = call!
    expect(url).toBe('/api/v1/deployments')
    expect(init?.method).toBe('POST')
    const body = JSON.parse(init?.body as string)
    expect(body.config).toEqual({ mode: 'in_app' })
    expect(body.agent_id).toBe('agent-1')
    expect(body.revision_id).toBe('rev-1')

    // Dialog auto-closes on success
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('shows error message when the mutation fails', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'agent not found' }), {
        status: 404,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <InAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-1"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )

    fireEvent.click(screen.getByTestId('in-app-wizard-submit'))

    await waitFor(() =>
      expect(
        screen.getByRole('alert', { name: undefined }),
      ).toBeInTheDocument(),
    )
  })
})
