/**
 * Tests for SparkBatchWizard — verifies the form fields render, the OBO
 * warning banner is present (per plan Section F.4), 3-level UC name
 * validation rejects bad input, and submit POSTs /deployments with
 * mode='batch' and the right config.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { SparkBatchWizard } from '../SparkBatchWizard'

const FAKE_DEPLOYMENT = {
  id: 'd-batch-1',
  agent_id: 'a-1',
  revision_id: 'r-1',
  mode: 'batch' as const,
  status: 'active' as const,
  config: {
    mode: 'batch',
    target_endpoint: 'databricks-claude-sonnet-4-5',
    input_table: 'main.research.queries',
    output_table: 'main.research.results',
    prompt_column: 'query',
    response_format: null,
  },
  endpoint_name: null,
  model_name: null,
  external_resource_ids: { sql_artifact_sha256: 'a'.repeat(64) },
  error_message: null,
  cleanup_attempts: 0,
  deployed_by: 'tester',
  created_at: '2026-05-09T00:00:00Z',
  updated_at: '2026-05-09T00:00:00Z',
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

describe('SparkBatchWizard', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders all form fields and the OBO warning banner', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <SparkBatchWizard
          open={true}
          onOpenChange={() => {}}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    expect(
      screen.getByTestId('spark-batch-endpoint-input'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('spark-batch-input-table-input'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('spark-batch-output-table-input'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('spark-batch-prompt-column-input'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('spark-batch-response-format-input'),
    ).toBeInTheDocument()

    // OBO warning banner per plan Section F.4
    const banner = screen.getByTestId('spark-batch-obo-banner')
    expect(banner).toBeInTheDocument()
    expect(banner.textContent).toMatch(/OBO not supported/i)
    expect(banner.textContent).toMatch(/CAN_QUERY/)
  })

  it('disables submit when input_table is not a 3-level UC name', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <SparkBatchWizard
          open={true}
          onOpenChange={() => {}}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    const inputTable = screen.getByTestId('spark-batch-input-table-input')
    fireEvent.change(inputTable, { target: { value: 'not-three-level' } })
    const submit = screen.getByTestId(
      'spark-batch-wizard-submit',
    ) as HTMLButtonElement
    expect(submit).toBeDisabled()
  })

  it('rejects malformed JSON in the response_format field', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <SparkBatchWizard
          open={true}
          onOpenChange={() => {}}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    const responseFmt = screen.getByTestId(
      'spark-batch-response-format-input',
    )
    fireEvent.change(responseFmt, { target: { value: '{ not: valid' } })
    const submit = screen.getByTestId(
      'spark-batch-wizard-submit',
    ) as HTMLButtonElement
    expect(submit).toBeDisabled()
  })

  it('submit POSTs /deployments with mode=batch and the config', async () => {
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
        <SparkBatchWizard
          open={true}
          onOpenChange={onOpenChange}
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={onDeployed}
        />
      </Wrapper>,
    )

    fireEvent.click(screen.getByTestId('spark-batch-wizard-submit'))

    await waitFor(() => expect(onDeployed).toHaveBeenCalledTimes(1))
    expect(onDeployed).toHaveBeenCalledWith(FAKE_DEPLOYMENT)

    expect(fetchSpy).toHaveBeenCalled()
    const call = fetchSpy.mock.calls[0]
    expect(call).toBeDefined()
    const [url, init] = call!
    expect(url).toBe('/api/v1/deployments')
    expect(init?.method).toBe('POST')
    const body = JSON.parse(init?.body as string)
    expect(body.config.mode).toBe('batch')
    expect(body.config.target_endpoint).toBe('databricks-claude-sonnet-4-5')
    expect(body.config.input_table).toBe('main.research.queries')

    // Wizard auto-closes on success
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
