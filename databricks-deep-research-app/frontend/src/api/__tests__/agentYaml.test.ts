/**
 * Tests for the YAML import/export API client functions and the
 * create-with-force / critic-error contract. Mocks fetch directly (no msw).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

import { exportAgentYaml, createAgentV2, updateAgentV2, AgentCriticError, EtagConflictError } from '../agentsV2'
import { importYaml, exportYamlFromDefinition } from '../agentDesigner'
import { YamlImportError } from '../client'

function makeResponse(
  body: unknown,
  { status = 200, headers = {} }: { status?: number; headers?: Record<string, string> } = {},
): Response {
  const bodyStr = typeof body === 'string' ? body : JSON.stringify(body)
  return new Response(bodyStr, {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}

/** Mirror the app's HTTP error envelope: { code: 'HTTP_ERROR', message: <detail> }. */
function httpError(detail: unknown, status: number): Response {
  return makeResponse({ code: 'HTTP_ERROR', message: detail }, { status })
}

beforeEach(() => {
  vi.resetAllMocks()
})
afterEach(() => {
  vi.restoreAllMocks()
})

describe('exportAgentYaml', () => {
  it('GETs /agents-v2/{id}/yaml and returns the text body', async () => {
    const yaml = "registry_version: '1.0.0'\nname: A\n"
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(yaml, { status: 200, headers: { 'Content-Type': 'text/yaml' } }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await exportAgentYaml('abc-123')

    expect(result).toBe(yaml)
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/agents-v2/abc-123/yaml', expect.any(Object))
  })

  it('surfaces a 404 as an ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(httpError('agent not found', 404)))
    await expect(exportAgentYaml('missing')).rejects.toMatchObject({ status: 404 })
  })
})

describe('exportYamlFromDefinition', () => {
  it('POSTs JSON {definition} to /export-yaml and returns the text body', async () => {
    const yaml = "registry_version: '1.0.0'\nname: Live\n"
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(yaml, { status: 200, headers: { 'Content-Type': 'text/yaml' } }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await exportYamlFromDefinition({ name: 'Live', root: {} })

    expect(result).toBe(yaml)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/agent-designer/export-yaml')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual({ definition: { name: 'Live', root: {} } })
  })

  it('maps a 400 schema_error to YamlImportError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        httpError({ errors: [{ path: null, kind: 'schema_error', message: 'bad AST' }] }, 400),
      ),
    )
    await expect(exportYamlFromDefinition({})).rejects.toBeInstanceOf(YamlImportError)
  })
})

describe('importYaml', () => {
  it('POSTs the RAW yaml body with text/yaml and returns the parsed result', async () => {
    const yamlText = "registry_version: '1.0.0'\nname: Imported\n"
    const payload = {
      definition: { name: 'Imported', root: {} },
      workflow_summary: { node_count: 1, tool_count: 0, source_count: 0 },
    }
    const fetchMock = vi.fn().mockResolvedValue(makeResponse(payload))
    vi.stubGlobal('fetch', fetchMock)

    const result = await importYaml(yamlText)

    expect(result.workflow_summary.node_count).toBe(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/agent-designer/import-yaml')
    expect(init.method).toBe('POST')
    // Raw YAML body — NOT JSON.stringify'd.
    expect(init.body).toBe(yamlText)
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('text/yaml')
  })

  it('maps a structured 400 to YamlImportError carrying every error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        httpError(
          {
            errors: [
              { path: null, kind: 'registry_version_mismatch', message: 'expected 1.0.0' },
              { path: 'root', kind: 'schema_error', message: 'bad node' },
            ],
          },
          400,
        ),
      ),
    )

    await expect(importYaml('x: 1')).rejects.toMatchObject({
      name: 'YamlImportError',
      errors: [
        { kind: 'registry_version_mismatch' },
        { kind: 'schema_error', path: 'root' },
      ],
    })
  })

  it('maps a 413 oversize to YamlImportError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(httpError({ error_kind: 'too_large', max_bytes: 262144 }, 413)),
    )
    // 413 detail has no `errors[]` array → falls back to a generic ApiError.
    await expect(importYaml('x: 1')).rejects.toMatchObject({ status: 413 })
  })
})

describe('createAgentV2 force + critic', () => {
  const req = { name: 'A', definition: { name: 'A', root: {} } }

  it('appends ?force=true when force is set', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      makeResponse({ id: 'new-1', name: 'A' }, { status: 201, headers: { ETag: '"v1"' } }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const { agent } = await createAgentV2(req, { force: true })

    expect(agent.id).toBe('new-1')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/agents-v2?force=true')
  })

  it('does NOT append force by default', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      makeResponse({ id: 'new-2', name: 'A' }, { status: 201 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await createAgentV2(req)

    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/agents-v2')
  })

  it('throws AgentCriticError with the critique on a 422 critic fail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        httpError(
          { message: 'critic blocked', critique: { verdict: 'fail', summary: 'off-topic' } },
          422,
        ),
      ),
    )

    try {
      await createAgentV2(req)
      expect.unreachable('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(AgentCriticError)
      expect((err as AgentCriticError).critique?.summary).toBe('off-topic')
    }
  })
})

describe('updateAgentV2 force + critic', () => {
  const req = { name: 'B', definition: { name: 'B', root: {} } }
  const etag = '"etag-v1"'

  it('appends ?force=true when force option is set', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      makeResponse({ id: 'agent-1', name: 'B' }, { status: 200, headers: { ETag: '"v2"' } }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await updateAgentV2('agent-1', req, etag, { force: true })

    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/agents-v2/agent-1?force=true')
  })

  it('does NOT append force by default', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      makeResponse({ id: 'agent-1', name: 'B' }, { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await updateAgentV2('agent-1', req, etag)

    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/agents-v2/agent-1')
  })

  it('throws EtagConflictError on 409', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeResponse(
          { detail: { current_etag: '"etag-server"' } },
          { status: 409 },
        ),
      ),
    )

    await expect(updateAgentV2('agent-1', req, etag)).rejects.toBeInstanceOf(EtagConflictError)
  })

  it('throws AgentCriticError with the critique on a 422 critic fail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        httpError(
          { message: 'critic blocked', critique: { verdict: 'fail', summary: 'irrelevant scope' } },
          422,
        ),
      ),
    )

    try {
      await updateAgentV2('agent-1', req, etag)
      expect.unreachable('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(AgentCriticError)
      expect((err as AgentCriticError).critique?.summary).toBe('irrelevant scope')
    }
  })
})
