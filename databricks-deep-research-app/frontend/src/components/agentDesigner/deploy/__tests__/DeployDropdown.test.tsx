/**
 * Tests for DeployDropdown — verifies all 4 mode entries render and are
 * wired. As of Phase 3, every entry (in_app, shell_app, mlflow_agent, batch)
 * opens its corresponding wizard. No 'Phase 3' / disabled entries remain.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

vi.mock('@/api/agentsV2', () => ({
  getRevision: vi.fn(),
}))

import { getRevision } from '@/api/agentsV2'
import { DeployDropdown } from '../DeployDropdown'

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

const REVISION_DETAIL: Awaited<ReturnType<typeof getRevision>> = {
  rev_id: 'r-1',
  etag: 'etag-1',
  created_at: '2026-05-16T00:00:00Z',
  created_by: 'tester',
  definition: {
    id: 'wf-1',
    name: 'Research Agent',
    description: 'Designed workflow',
    version: 1,
    root: {
      id: 'root',
      type: 'sequence',
      label: 'Workflow',
      config: {},
      children: [
        { id: 'coordinator', type: 'agent', label: 'Coordinator', config: {} },
      ],
    },
    tools: [],
  },
}

describe('DeployDropdown', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getRevision).mockResolvedValue(REVISION_DETAIL)
  })

  it('renders the trigger button', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    expect(screen.getByTestId('deploy-dropdown-trigger')).toBeInTheDocument()
  })

  it('opens the menu and renders all 4 mode entries', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    expect(screen.getByTestId('deploy-dropdown-in_app')).toBeInTheDocument()
    expect(
      screen.getByTestId('deploy-dropdown-shell_app'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('deploy-dropdown-mlflow_agent'),
    ).toBeInTheDocument()
    expect(screen.getByTestId('deploy-dropdown-batch')).toBeInTheDocument()
  })

  it('Phase 3 — every entry is enabled (no Phase 3 badges remain)', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    expect(screen.getByTestId('deploy-dropdown-in_app')).not.toBeDisabled()
    expect(
      screen.getByTestId('deploy-dropdown-shell_app'),
    ).not.toBeDisabled()
    expect(
      screen.getByTestId('deploy-dropdown-mlflow_agent'),
    ).not.toBeDisabled()
    expect(screen.getByTestId('deploy-dropdown-batch')).not.toBeDisabled()
    // No 'Phase 3' badge text anywhere.
    expect(
      screen.queryByText(/Phase 3/i),
    ).not.toBeInTheDocument()
  })

  it('clicking mlflow_agent opens the MlflowAgentWizard (Phase 3)', async () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    fireEvent.click(screen.getByTestId('deploy-dropdown-mlflow_agent'))
    // MlflowAgentWizard renders the UC catalog input + endpoint name input.
    expect(
      await screen.findByTestId('mlflow-uc-catalog-input'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('mlflow-endpoint-name-input'),
    ).toBeInTheDocument()
  })

  it('keeps in_app, shell_app, mlflow_agent, and batch entries enabled', () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    expect(screen.getByTestId('deploy-dropdown-in_app')).not.toBeDisabled()
    expect(
      screen.getByTestId('deploy-dropdown-shell_app'),
    ).not.toBeDisabled()
    expect(
      screen.getByTestId('deploy-dropdown-mlflow_agent'),
    ).not.toBeDisabled()
    expect(screen.getByTestId('deploy-dropdown-batch')).not.toBeDisabled()
  })

  it('clicking batch opens the SparkBatchWizard (Phase 2-B)', async () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    fireEvent.click(screen.getByTestId('deploy-dropdown-batch'))
    // SparkBatchWizard renders its OBO banner + endpoint input.
    expect(await screen.findByTestId('spark-batch-obo-banner')).toBeInTheDocument()
    expect(
      screen.getByTestId('spark-batch-endpoint-input'),
    ).toBeInTheDocument()
  })

  it('clicking shell_app opens the ShellAppWizard (Phase 2-B)', async () => {
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    fireEvent.click(screen.getByTestId('deploy-dropdown-shell_app'))
    expect(await screen.findByTestId('shell-app-name-input')).toBeInTheDocument()
    expect(screen.getByTestId('shell-app-git-tag-input')).toBeInTheDocument()
  })

  it('W5: onBeforeDeploy runs before opening the wizard and supplies the new revisionId', async () => {
    const onBeforeDeploy = vi.fn(async () => 'r-saved')
    vi.mocked(getRevision).mockResolvedValue({
      ...REVISION_DETAIL,
      rev_id: 'r-saved',
    })
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-stale"
          onBeforeDeploy={onBeforeDeploy}
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    fireEvent.click(screen.getByTestId('deploy-dropdown-in_app'))

    await waitFor(() => expect(onBeforeDeploy).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(getRevision).toHaveBeenCalledWith('a-1', 'r-saved'))
    // The InAppWizard should be open with the freshly-resolved revisionId.
    // We assert at least that the wizard surface is visible (deeper coupling
    // would belong in a parent-component integration test).
    await waitFor(() =>
      expect(screen.queryByTestId('in-app-wizard-submit')).toBeInTheDocument(),
    )
    expect(screen.getByTestId('revision-provenance-card')).toHaveTextContent('r-saved')
  })

  it('W5: onBeforeDeploy returning null aborts wizard open (e.g. save failed)', async () => {
    const onBeforeDeploy = vi.fn(async () => null)
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <DeployDropdown
          agentId="a-1"
          agentName="Research Agent"
          revisionId="r-stale"
          onBeforeDeploy={onBeforeDeploy}
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    fireEvent.click(screen.getByTestId('deploy-dropdown-trigger'))
    fireEvent.click(screen.getByTestId('deploy-dropdown-shell_app'))

    await waitFor(() => expect(onBeforeDeploy).toHaveBeenCalledTimes(1))
    // Wizard must NOT have opened — no shell-app name input rendered.
    expect(screen.queryByTestId('shell-app-name-input')).not.toBeInTheDocument()
  })
})
