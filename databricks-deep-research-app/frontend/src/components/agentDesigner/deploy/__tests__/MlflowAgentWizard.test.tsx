/**
 * Tests for MlflowAgentWizard — verifies the 5 inputs render with sensible
 * defaults; client-side validation rejects invalid endpoint_name and bad
 * env_overrides JSON; submit POSTs /deployments with mode='mlflow_agent'
 * and the right config.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { MlflowAgentWizard } from '../MlflowAgentWizard'

const FAKE_DEPLOYMENT = {
  id: 'd-mlflow-1',
  agent_id: 'a-1',
  revision_id: 'r-1',
  mode: 'mlflow_agent' as const,
  status: 'deploying' as const,
  config: {
    mode: 'mlflow_agent',
    uc_catalog: 'main',
    uc_schema: 'agents',
    uc_model_name: 'dr_agent_1',
  },
  endpoint_name: null,
  model_name: 'main.agents.dr_agent_1',
  external_resource_ids: null,
  error_message: null,
  cleanup_attempts: 0,
  deployed_by: 'tester',
  created_at: '2026-05-10T00:00:00Z',
  updated_at: '2026-05-10T00:00:00Z',
  deactivated_at: null,
}

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  }
}

describe('MlflowAgentWizard', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders 5 inputs with sensible defaults (uc_catalog=main, uc_schema=agents)', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <MlflowAgentWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-deadbeef-1234"
          agentName="Research Agent"
          revisionId="rev-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    expect(
      (screen.getByTestId('mlflow-uc-catalog-input') as HTMLInputElement).value,
    ).toBe('main')
    expect(
      (screen.getByTestId('mlflow-uc-schema-input') as HTMLInputElement).value,
    ).toBe('agents')
    // Default model name is dr_<sanitized first-8 of agent id>
    expect(
      (screen.getByTestId('mlflow-uc-model-name-input') as HTMLInputElement).value,
    ).toBe('dr_agent-de')
    expect(
      screen.getByTestId('mlflow-endpoint-name-input'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('mlflow-env-overrides-input'),
    ).toBeInTheDocument()
  })

  it('disables submit when endpoint_name override is malformed', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <MlflowAgentWizard
          open={true}
          onOpenChange={() => {}}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.change(screen.getByTestId('mlflow-endpoint-name-input'), {
      target: { value: 'custom-endpoint' },
    })
    expect(
      screen.getByTestId('mlflow-agent-wizard-submit'),
    ).toBeDisabled()
  })

  it('rejects malformed env_overrides JSON', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <MlflowAgentWizard
          open={true}
          onOpenChange={() => {}}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.change(screen.getByTestId('mlflow-env-overrides-input'), {
      target: { value: '{ not valid json' },
    })
    expect(
      screen.getByTestId('mlflow-agent-wizard-submit'),
    ).toBeDisabled()
  })

  it('submit POSTs /deployments with mode=mlflow_agent and the config', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(FAKE_DEPLOYMENT), {
        status: 201,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const onDeployed = vi.fn()
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <MlflowAgentWizard
          open={true}
          onOpenChange={() => {}}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={onDeployed}
        />
      </Wrapper>,
    )

    fireEvent.click(screen.getByTestId('mlflow-agent-wizard-submit'))

    await waitFor(() => expect(onDeployed).toHaveBeenCalledTimes(1))
    expect(onDeployed).toHaveBeenCalledWith(FAKE_DEPLOYMENT)

    const call = fetchSpy.mock.calls[0]
    expect(call).toBeDefined()
    const [url, init] = call!
    expect(url).toBe('/api/v1/deployments')
    expect(init?.method).toBe('POST')
    const body = JSON.parse(init?.body as string)
    expect(body.config.mode).toBe('mlflow_agent')
    expect(body.config.uc_catalog).toBe('main')
    expect(body.config.uc_schema).toBe('agents')
    expect(body.config.uc_model_name).toBe('dr_a-1')
  })
})
