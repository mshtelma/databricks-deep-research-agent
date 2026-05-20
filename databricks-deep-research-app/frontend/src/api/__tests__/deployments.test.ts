/**
 * Tests for the deployments API client.
 *
 * Mocks fetch directly (no msw); each test restores the mock after use.
 * Verifies request URL/method shaping and response parsing for the 7 helpers.
 */

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type MockInstance,
} from 'vitest'

import {
  canRunFast,
  canRunSlow,
  createDeployment,
  deactivateDeployment,
  DeploymentApiError,
  parseDefaultRevisionNotDeployableError,
  getDeployment,
  getDeploymentStatus,
  listDeployments,
} from '../deployments'

function makeResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

const FAKE_DEPLOYMENT = {
  id: '00000000-0000-0000-0000-000000000001',
  agent_id: '00000000-0000-0000-0000-000000000002',
  revision_id: '00000000-0000-0000-0000-000000000003',
  mode: 'in_app',
  status: 'active',
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

describe('deployments API client', () => {
  let fetchSpy: MockInstance<Parameters<typeof fetch>, ReturnType<typeof fetch>>

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch')
  })
  afterEach(() => {
    fetchSpy.mockRestore()
  })

  describe('createDeployment', () => {
    it('POSTs to /api/v1/deployments with the given body', async () => {
      fetchSpy.mockResolvedValueOnce(makeResponse(FAKE_DEPLOYMENT, 201))
      const result = await createDeployment({
        agent_id: FAKE_DEPLOYMENT.agent_id,
        revision_id: FAKE_DEPLOYMENT.revision_id,
        config: { mode: 'in_app' },
      })
      expect(fetchSpy).toHaveBeenCalledTimes(1)
      const [url, init] = fetchSpy.mock.calls[0]!
      expect(url).toBe('/api/v1/deployments')
      expect(init?.method).toBe('POST')
      expect(JSON.parse(init?.body as string)).toEqual({
        agent_id: FAKE_DEPLOYMENT.agent_id,
        revision_id: FAKE_DEPLOYMENT.revision_id,
        config: { mode: 'in_app' },
      })
      expect(result.id).toBe(FAKE_DEPLOYMENT.id)
    })

    it('can create a row without starting the async runner', async () => {
      fetchSpy.mockResolvedValueOnce(makeResponse(FAKE_DEPLOYMENT, 202))
      await createDeployment(
        {
          agent_id: FAKE_DEPLOYMENT.agent_id,
          revision_id: FAKE_DEPLOYMENT.revision_id,
          config: { mode: 'in_app' },
        },
        { runAsync: false },
      )
      const [url] = fetchSpy.mock.calls[0]!
      expect(url).toBe('/api/v1/deployments?run_async=false')
    })

    it('throws DeploymentApiError on non-2xx', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({ detail: 'agent not found' }, 404),
      )
      await expect(
        createDeployment({
          agent_id: 'a',
          revision_id: 'r',
          config: { mode: 'in_app' },
        }),
      ).rejects.toBeInstanceOf(DeploymentApiError)
    })

    it('parses default revision blocker responses', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse(
          {
            detail: {
              error_kind: 'default_revision_not_deployable',
              agent_id: 'agent-1',
              revision_id: 'revision-1',
              workflow_name: 'Untitled Agent',
              root_child_summary: ['coordinator:agent:Coordinator'],
              message: 'Save or select a designed workflow revision before deploying.',
            },
          },
          422,
        ),
      )

      await expect(
        createDeployment({
          agent_id: 'agent-1',
          revision_id: 'revision-1',
          config: { mode: 'in_app' },
        }),
      ).rejects.toSatisfy((error: unknown) => {
        const parsed = parseDefaultRevisionNotDeployableError(error)
        return (
          parsed?.error_kind === 'default_revision_not_deployable' &&
          parsed.revision_id === 'revision-1' &&
          parsed.workflow_name === 'Untitled Agent'
        )
      })
    })
  })

  describe('listDeployments', () => {
    it('serializes filters into the query string', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({ items: [], next_cursor: null }),
      )
      await listDeployments({
        mode: 'batch',
        status: 'active',
        cursor: 'abc',
        limit: 25,
      })
      const url = fetchSpy.mock.calls[0]![0] as string
      expect(url).toContain('mode=batch')
      expect(url).toContain('status=active')
      expect(url).toContain('cursor=abc')
      expect(url).toContain('limit=25')
    })

    it('omits the question mark when no filters set', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({ items: [], next_cursor: null }),
      )
      await listDeployments()
      const url = fetchSpy.mock.calls[0]![0] as string
      expect(url).toBe('/api/v1/deployments')
    })
  })

  describe('getDeployment', () => {
    it('GETs /api/v1/deployments/{id}', async () => {
      fetchSpy.mockResolvedValueOnce(makeResponse(FAKE_DEPLOYMENT))
      const result = await getDeployment(FAKE_DEPLOYMENT.id)
      expect(fetchSpy.mock.calls[0]![0]).toBe(
        `/api/v1/deployments/${FAKE_DEPLOYMENT.id}`,
      )
      expect(result.status).toBe('active')
    })
  })

  describe('deactivateDeployment', () => {
    it('DELETEs /api/v1/deployments/{id}', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({ ...FAKE_DEPLOYMENT, status: 'deactivated' }),
      )
      const result = await deactivateDeployment(FAKE_DEPLOYMENT.id)
      const init = fetchSpy.mock.calls[0]![1]
      expect(init?.method).toBe('DELETE')
      expect(result.status).toBe('deactivated')
    })
  })

  describe('getDeploymentStatus', () => {
    it('returns the lightweight status response', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({
          status: 'active',
          updated_at: '2026-05-09T00:00:00Z',
          error_message: null,
        }),
      )
      const result = await getDeploymentStatus(FAKE_DEPLOYMENT.id)
      expect(result.status).toBe('active')
      expect(fetchSpy.mock.calls[0]![0]).toBe(
        `/api/v1/deployments/${FAKE_DEPLOYMENT.id}/status`,
      )
    })
  })

  describe('canRunFast', () => {
    it('GETs /api/v1/deployments/can-run/fast/{agentId}', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({ can_run: true, reasons: [] }),
      )
      const result = await canRunFast(FAKE_DEPLOYMENT.agent_id)
      expect(fetchSpy.mock.calls[0]![0]).toBe(
        `/api/v1/deployments/can-run/fast/${FAKE_DEPLOYMENT.agent_id}`,
      )
      expect(result.can_run).toBe(true)
    })
  })

  describe('canRunSlow', () => {
    it('GETs /api/v1/deployments/can-run/slow/{agentId} and exposes cached', async () => {
      fetchSpy.mockResolvedValueOnce(
        makeResponse({ can_run: true, reasons: [], cached: false }),
      )
      const result = await canRunSlow(FAKE_DEPLOYMENT.agent_id)
      expect(fetchSpy.mock.calls[0]![0]).toBe(
        `/api/v1/deployments/can-run/slow/${FAKE_DEPLOYMENT.agent_id}`,
      )
      expect(result.cached).toBe(false)
    })
  })
})
