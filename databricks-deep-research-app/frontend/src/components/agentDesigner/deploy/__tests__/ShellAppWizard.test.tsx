/**
 * Tests for ShellAppWizard — verifies the 3 inputs render, app_name regex
 * validation rejects bad input, submit POSTs /deployments with mode='shell_app',
 * and the post-success Download zip button is present.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { ShellAppWizard } from '../ShellAppWizard'

const FAKE_DEFAULT_GIT_TAG = 'v9.9.9'

const FAKE_DEPLOYMENT = {
  id: 'd-shell-1',
  agent_id: 'a-1',
  revision_id: 'r-1',
  mode: 'shell_app' as const,
  status: 'active' as const,
  config: {
    mode: 'shell_app',
    app_name: 'dr-shell-research',
    framework_git_tag: FAKE_DEFAULT_GIT_TAG,
    target: 'dev',
  },
  endpoint_name: 'dr-shell-research',
  model_name: null,
  external_resource_ids: { app_name: 'dr-shell-research' },
  error_message: null,
  cleanup_attempts: 0,
  deployed_by: 'tester',
  created_at: '2026-05-09T00:00:00Z',
  updated_at: '2026-05-09T00:00:00Z',
  deactivated_at: null,
}

/** Stub fetch: route /config/deployment-defaults to a fixed tag, fall through to override. */
function mockFetch(
  override?: (url: string, init?: RequestInit) => Response,
) {
  return vi
    .spyOn(globalThis, 'fetch')
    .mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : (input as Request).url
      if (url.endsWith('/config/deployment-defaults')) {
        return new Response(
          JSON.stringify({ frameworkGitTag: FAKE_DEFAULT_GIT_TAG }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        )
      }
      if (override) return override(url, init)
      return new Response('not mocked', { status: 500 })
    })
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

describe('ShellAppWizard', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders the 3 input fields with the dr-shell-* default app name', async () => {
    mockFetch()
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <ShellAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-12345678abcdef"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    const appNameInput = screen.getByTestId(
      'shell-app-name-input',
    ) as HTMLInputElement
    expect(appNameInput).toBeInTheDocument()
    expect(appNameInput.value).toBe('dr-shell-agent-12')
    const gitTagInput = screen.getByTestId(
      'shell-app-git-tag-input',
    ) as HTMLInputElement
    // Defaults query resolves and populates the field with the framework tag.
    await waitFor(() =>
      expect(gitTagInput.value).toBe(FAKE_DEFAULT_GIT_TAG),
    )
    expect(screen.getByTestId('shell-app-target-input')).toBeInTheDocument()
  })

  it('disables submit when app_name does not match the dr-shell-* prefix', () => {
    mockFetch()
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <ShellAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-1"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    const appNameInput = screen.getByTestId('shell-app-name-input')
    fireEvent.change(appNameInput, { target: { value: 'my-app-no-prefix' } })

    const submit = screen.getByTestId(
      'shell-app-wizard-submit',
    ) as HTMLButtonElement
    expect(submit).toBeDisabled()
    // Validation hint must surface to the user.
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('disables submit when app_name exceeds the Databricks Apps limit', () => {
    mockFetch()
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <ShellAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-1"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )
    const appNameInput = screen.getByTestId('shell-app-name-input')
    fireEvent.change(appNameInput, {
      target: { value: 'dr-shell-a528c6b059bb421587b49de' },
    })

    const submit = screen.getByTestId(
      'shell-app-wizard-submit',
    ) as HTMLButtonElement
    expect(submit).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('30 chars or fewer')
  })

  it('renders default revision blocker as a blocking error', async () => {
    mockFetch((url) => {
      if (url.endsWith('/deployments')) {
        return new Response(
          JSON.stringify({
            detail: {
              error_kind: 'default_revision_not_deployable',
              agent_id: 'agent-1',
              revision_id: 'revision-default',
              workflow_name: 'Untitled Agent',
              root_child_summary: [
                'coordinator:agent:Coordinator',
                'plan-and-execute:plan_and_execute:Plan & Execute',
                'synthesizer:agent:Synthesizer',
              ],
              message: 'Save or select a designed workflow revision before deploying.',
            },
          }),
          { status: 422, headers: { 'content-type': 'application/json' } },
        )
      }
      return new Response('not mocked', { status: 500 })
    })
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <ShellAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-1"
          agentName="Untitled Agent"
          revisionId="revision-default"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )

    await waitFor(() =>
      expect((screen.getByTestId('shell-app-git-tag-input') as HTMLInputElement).value)
        .toBe(FAKE_DEFAULT_GIT_TAG),
    )
    fireEvent.click(screen.getByTestId('shell-app-wizard-submit'))

    await screen.findByText(/Revision revision \(Untitled Agent\) is not deployable/i)
    expect(screen.getByTestId('shell-app-wizard-submit')).toBeInTheDocument()
  })

  it('submit POSTs /deployments with mode=shell_app and reveals Download zip', async () => {
    const fetchSpy = mockFetch(
      () =>
        new Response(JSON.stringify(FAKE_DEPLOYMENT), {
          status: 201,
          headers: { 'content-type': 'application/json' },
        }),
    )
    const onDeployed = vi.fn()
    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <ShellAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-1"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={onDeployed}
        />
      </Wrapper>,
    )

    // Wait for the defaults query to populate the git-tag field; submit is
    // disabled while it's empty, so clicking before that is a no-op.
    const gitTagInput = screen.getByTestId(
      'shell-app-git-tag-input',
    ) as HTMLInputElement
    await waitFor(() =>
      expect(gitTagInput.value).toBe(FAKE_DEFAULT_GIT_TAG),
    )

    fireEvent.click(
      screen.getByRole('tab', { name: /export for another workspace/i }),
    )
    await waitFor(() =>
      expect(screen.getByTestId('shell-app-wizard-submit')).toBeInTheDocument(),
    )
    fireEvent.click(screen.getByTestId('shell-app-wizard-submit'))

    await waitFor(() => expect(onDeployed).toHaveBeenCalledTimes(1))
    expect(onDeployed).toHaveBeenCalledWith(FAKE_DEPLOYMENT)

    const deployCall = fetchSpy.mock.calls.find(
      ([url]) => url === '/api/v1/deployments',
    )
    expect(deployCall).toBeDefined()
    const [url, init] = deployCall!
    expect(url).toBe('/api/v1/deployments')
    expect(init?.method).toBe('POST')
    const body = JSON.parse(init?.body as string)
    expect(body.config.mode).toBe('shell_app')
    expect(body.config.app_name).toBe('dr-shell-agent-1')
    expect(body.config.framework_git_tag).toBe(FAKE_DEFAULT_GIT_TAG)

    // Post-success: Download zip button appears
    expect(
      await screen.findByTestId('shell-app-download-button'),
    ).toBeInTheDocument()
  })

  it('preserves user-typed git tag when defaults resolve late (regression: W1)', async () => {
    // Stage the defaults response so we can resolve it *after* the user
    // types a custom tag — exercising the race window codex flagged.
    let resolveDefaults: ((value: Response) => void) | undefined
    const defaultsPromise = new Promise<Response>((resolve) => {
      resolveDefaults = resolve
    })
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : (input as Request).url
      if (url.endsWith('/config/deployment-defaults')) {
        return defaultsPromise
      }
      return new Response('not mocked', { status: 500 })
    })

    const Wrapper = makeWrapper()
    render(
      <Wrapper>
        <ShellAppWizard
          open={true}
          onOpenChange={() => {}}
          agentId="agent-1"
          agentName="Deep Research Agent"
          revisionId="rev-1"
          onDeployed={() => {}}
        />
      </Wrapper>,
    )

    const gitTagInput = screen.getByTestId(
      'shell-app-git-tag-input',
    ) as HTMLInputElement

    // While defaults are in flight, the user types a custom tag.
    fireEvent.change(gitTagInput, { target: { value: 'v1.5.0-rc1' } })
    expect(gitTagInput.value).toBe('v1.5.0-rc1')

    // Now resolve the defaults query — the on-open effect refires with the
    // new defaultGitTag but must NOT overwrite the user's input.
    resolveDefaults!(
      new Response(JSON.stringify({ frameworkGitTag: FAKE_DEFAULT_GIT_TAG }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )
    // Give React a tick to flush the post-query render.
    await waitFor(() =>
      expect(gitTagInput.value).toBe('v1.5.0-rc1'),
    )
  })
})
