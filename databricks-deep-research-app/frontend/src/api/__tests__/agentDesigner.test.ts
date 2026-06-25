/**
 * Tests for agentsV2 + agentDesigner API clients.
 *
 * Mocks fetch directly (no msw). Each test restores the mock after use.
 *
 * NOTE: @/types/ast is created by US-302. A local type alias is used here
 * so tests can run independently.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// ---------------------------------------------------------------------------
// Mock @/types/ast so this test file compiles without US-302's file
// ---------------------------------------------------------------------------
vi.mock('@/types/ast', () => ({
  // no runtime values needed; types are erased at compile-time
}))

import {
  listAgentsV2,
  getAgentV2WithEtag,
  createAgentV2,
  updateAgentV2,
  deleteAgentV2,
  EtagConflictError,
  AgentDeleteError,
  parseAgentDeleteError,
} from '../agentsV2'

import {
  getRegistry,
  clearRegistryCache,
  chatStream,
  slimWireMessages,
  listDesignerResources,
  startDesignerSqlWarehouse,
} from '../agentDesigner'
import type { ChatMessage } from '../agentDesigner'

import { ApiError } from '../client'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal Response-like object that vitest fetch mocks expect. */
function makeResponse(
  body: unknown,
  {
    status = 200,
    headers = {},
  }: { status?: number; headers?: Record<string, string> } = {}
): Response {
  const bodyStr = typeof body === 'string' ? body : JSON.stringify(body)
  return new Response(bodyStr, {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}

/** Build an SSE-formatted string from an array of [eventType, payload] pairs. */
function makeSSEStream(frames: Array<[string, unknown]>): ReadableStream<Uint8Array> {
  const text = frames
    .map(([type, payload]) => `event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`)
    .join('')
  const encoder = new TextEncoder()
  const bytes = encoder.encode(text)

  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes)
      controller.close()
    },
  })
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  clearRegistryCache()
  vi.resetAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// 1. getRegistry — returns parsed shape
// ---------------------------------------------------------------------------

describe('getRegistry', () => {
  it('returns the parsed registry on success', async () => {
    const payload = {
      node_types: [{ type: 'agent', label: 'Agent', icon: 'robot', category: 'leaf', is_composite: false, config_schema: null }],
      agent_subtypes: [{ id: 'coordinator', label: 'Coordinator', icon: 'star' }],
      tool_kinds: [{ kind: 'web_search', label: 'Web Search', icon: 'tool' }],
      model_tiers: ['simple', 'analytical', 'complex', 'synthesis'],
      version: '1.0.0',
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(makeResponse(payload)))

    const result = await getRegistry()

    expect(result.version).toBe('1.0.0')
    expect(result.model_tiers).toEqual(['simple', 'analytical', 'complex', 'synthesis'])
    expect(result.node_types[0]?.type).toBe('agent')
    expect(result.tool_kinds[0]?.kind).toBe('web_search')
  })

  // 2. getRegistry is cached on second call
  it('issues only one fetch call even when called twice', async () => {
    const payload = {
      node_types: [],
      agent_subtypes: [],
      tool_kinds: [],
      model_tiers: [],
      version: '1.0.0',
    }
    const fetchMock = vi.fn().mockResolvedValue(makeResponse(payload))
    vi.stubGlobal('fetch', fetchMock)

    await getRegistry()
    await getRegistry()

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('designer resources', () => {
  it('lists SQL warehouse resources by source kind', async () => {
    const payload = {
      resources: [
        {
          kind: 'sql_warehouse',
          source_id: 'wh-1',
          name: 'Starter Warehouse',
          full_name: 'Starter Warehouse',
          description: null,
          status: 'STOPPED',
          capabilities: ['sql'],
          metadata: { warehouse_id: 'wh-1', state: 'STOPPED' },
        },
      ],
      total: 1,
    }
    const fetchMock = vi.fn().mockResolvedValue(makeResponse(payload))
    vi.stubGlobal('fetch', fetchMock)

    const result = await listDesignerResources(['sql_warehouse'])

    expect(result.resources[0]?.metadata.warehouse_id).toBe('wh-1')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/agent-designer/resources?kinds=sql_warehouse',
      { headers: { 'Content-Type': 'application/json' } },
    )
  })

  it('starts a selected SQL warehouse', async () => {
    const payload = {
      kind: 'sql_warehouse',
      source_id: 'wh-1',
      name: 'Starter Warehouse',
      full_name: 'Starter Warehouse',
      description: null,
      status: 'STARTING',
      capabilities: ['sql'],
      metadata: { warehouse_id: 'wh-1', state: 'STARTING' },
    }
    const fetchMock = vi.fn().mockResolvedValue(makeResponse(payload))
    vi.stubGlobal('fetch', fetchMock)

    const result = await startDesignerSqlWarehouse('wh/1')

    expect(result.status).toBe('STARTING')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/agent-designer/resources/sql-warehouses/wh%2F1/start',
      { method: 'POST', headers: { 'Content-Type': 'application/json' } },
    )
  })
})

// ---------------------------------------------------------------------------
// 3. listAgentsV2 — returns array
// ---------------------------------------------------------------------------

describe('listAgentsV2', () => {
  it('returns the items array from the list response', async () => {
    const payload = {
      items: [
        {
          id: 'aaaa-bbbb',
          name: 'My Agent',
          description: null,
          visibility: 'private',
          owner_id: 'user1',
          updated_at: '2026-01-01T00:00:00Z',
          node_count: 3,
        },
      ],
      total: 1,
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(makeResponse(payload)))

    const result = await listAgentsV2()

    expect(result.total).toBe(1)
    expect(result.items).toHaveLength(1)
    expect(result.items[0]?.name).toBe('My Agent')
  })
})

// ---------------------------------------------------------------------------
// 4. updateAgentV2 — sends If-Match header
// ---------------------------------------------------------------------------

describe('updateAgentV2', () => {
  it('sends the If-Match header with the provided etag', async () => {
    const agentPayload = {
      id: 'aaaa',
      owner_id: 'user1',
      name: 'Updated',
      description: null,
      avatar_url: null,
      visibility: 'private',
      definition: {},
      schema_version: 1,
      etag: 'new-etag-456',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-02T00:00:00Z',
    }
    const fetchMock = vi
      .fn()
      .mockResolvedValue(makeResponse(agentPayload, { headers: { ETag: 'new-etag-456' } }))
    vi.stubGlobal('fetch', fetchMock)

    await updateAgentV2('aaaa', { name: 'Updated' }, 'old-etag-123')

    expect(fetchMock).toHaveBeenCalledOnce()
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    const headers = init.headers as Record<string, string>
    expect(headers['If-Match']).toBe('old-etag-123')
  })

  // 5. updateAgentV2 on 409 throws EtagConflictError with current_etag
  it('throws EtagConflictError carrying current_etag on 409', async () => {
    const conflictBody = { detail: { current_etag: 'server-etag-789' } }
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeResponse(conflictBody, { status: 409 }))
    )

    await expect(
      updateAgentV2('aaaa', { name: 'Stale' }, 'stale-etag')
    ).rejects.toSatisfy((err: unknown) => {
      return err instanceof EtagConflictError && err.current_etag === 'server-etag-789'
    })
  })
})

// ---------------------------------------------------------------------------
// slimWireMessages — trims redundant resent transcript payloads on the wire
// ---------------------------------------------------------------------------

describe('slimWireMessages', () => {
  const big = 'R'.repeat(4000) // above the ~2 KB per-message summarize threshold

  it('summarizes OLD oversized tool content but preserves role + tool_call_id', () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'start' },
      { role: 'assistant', content: 'ok' },
      { role: 'tool', content: big, tool_call_id: 'c1' }, // old + oversized
      { role: 'user', content: 'u0' },
      { role: 'user', content: 'u1' },
      { role: 'user', content: 'u2' },
      { role: 'user', content: 'u3' },
      { role: 'user', content: 'u4' },
      { role: 'user', content: 'CURRENT' },
    ]
    const slimmed = slimWireMessages(messages)
    const oldTool = slimmed[2]!
    expect(oldTool.role).toBe('tool')
    expect(oldTool.tool_call_id).toBe('c1') // wire-shape pairing preserved
    expect(oldTool.content.startsWith('[tool content summarized')).toBe(true)
    expect(oldTool.content.length).toBeLessThan(200)
    expect(slimmed[slimmed.length - 1]!.content).toBe('CURRENT') // final msg untouched
  })

  it('keeps the recent window verbatim (recent oversized tool result survives)', () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'u0' },
      { role: 'tool', content: big, tool_call_id: 'recent' }, // inside last 6
      { role: 'user', content: 'CURRENT' },
    ]
    const tool = slimWireMessages(messages).find((m) => m.tool_call_id === 'recent')
    expect(tool?.content).toBe(big)
  })

  it('leaves small messages unchanged', () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'user', content: 'bye' },
    ]
    expect(slimWireMessages(messages)).toEqual(messages)
  })
})

// ---------------------------------------------------------------------------
// 6. chatStream — parses message + done events
// ---------------------------------------------------------------------------

describe('chatStream', () => {
  it('parses a message event followed by a done event', async () => {
    const stream = makeSSEStream([
      ['message', { content: 'Hello world' }],
      ['done', {}],
    ])

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      )
    )

    const events = []
    for await (const { event } of chatStream({
      messages: [{ role: 'user', content: 'Hi' }],
      current_ast: null,
    })) {
      events.push(event)
    }

    expect(events).toHaveLength(2)
    expect(events[0]).toMatchObject({ type: 'message', content: 'Hello world' })
    expect(events[1]).toMatchObject({ type: 'done' })
  })

  it('strips UI-only tool_name before sending chat history', async () => {
    const stream = makeSSEStream([['done', {}]])
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(stream, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const events = []
    for await (const { event } of chatStream({
      messages: [
        {
          role: 'tool',
          content: '{"schema":"prompt_grounding.v1"}',
          tool_call_id: 'prompt_grounding:init',
          tool_name: 'prompt_grounding',
        },
      ],
      current_ast: null,
    })) {
      events.push(event)
    }

    expect(events).toHaveLength(1)
    expect(fetchMock).toHaveBeenCalledOnce()
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string) as {
      messages: Array<Record<string, unknown>>
    }
    expect(body.messages[0]).toEqual({
      role: 'tool',
      content: '{"schema":"prompt_grounding.v1"}',
      tool_call_id: 'prompt_grounding:init',
    })
    expect(body.messages[0]).not.toHaveProperty('tool_name')
  })

  // 7. chatStream throws on 413 BEFORE yielding any events
  it('throws ApiError with status 413 before yielding any events', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeResponse({ detail: 'messages exceeds 20 turns' }, { status: 413 })
      )
    )

    const iter = chatStream({
      messages: [{ role: 'user', content: 'too long' }],
      current_ast: null,
    })

    await expect(iter[Symbol.asyncIterator]().next()).rejects.toSatisfy((err: unknown) => {
      return err instanceof ApiError && err.status === 413 && err.code === 'request_too_large'
    })
  })

  // 7b. chatStream surfaces the REAL 413 reason from the { code, message } envelope
  it('surfaces the real 413 reason from the { code, message } error envelope', async () => {
    // The backend wraps every HTTPException as { code: 'HTTP_ERROR', message: <detail> }.
    // The banner must show that reason, not the generic fallback string.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeResponse(
          { code: 'HTTP_ERROR', message: 'total payload exceeds 524288 bytes (600000)' },
          { status: 413 }
        )
      )
    )

    const iter = chatStream({
      messages: [{ role: 'user', content: 'too big' }],
      current_ast: null,
    })

    await expect(iter[Symbol.asyncIterator]().next()).rejects.toSatisfy((err: unknown) => {
      return (
        err instanceof ApiError &&
        err.status === 413 &&
        err.message === 'total payload exceeds 524288 bytes (600000)'
      )
    })
  })

  // 8. chatStream yields tool_call then mutation_proposed from a multi-event stream
  it('yields tool_call then mutation_proposed from a multi-event stream', async () => {
    const oldAst = { id: 'draft', name: 'Old', root: {} }
    const newAst = { id: 'draft', name: 'New', root: {} }

    const stream = makeSSEStream([
      [
        'tool_call',
        { tool_name: 'propose_workflow', tool_call_id: 'tc1', args: { intent: 'research' } },
      ],
      [
        'mutation_proposed',
        {
          tool_call_id: 'tc1',
          old_ast: oldAst,
          new_ast: newAst,
          validation_errors: [],
          summary: { node_count: 1, tool_count: 0, source_count: 0 },
        },
      ],
      ['done', {}],
    ])

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      )
    )

    const events = []
    for await (const { event } of chatStream({
      messages: [{ role: 'user', content: 'Build me an agent' }],
      current_ast: null,
    })) {
      events.push(event)
    }

    expect(events).toHaveLength(3)
    expect(events[0]).toMatchObject({ type: 'tool_call', tool_name: 'propose_workflow' })
    expect(events[1]).toMatchObject({ type: 'mutation_proposed', tool_call_id: 'tc1' })
    expect(events[2]).toMatchObject({ type: 'done' })
  })

  // Bonus: chatStream skips a malformed frame (non-JSON data) without yielding it
  it('skips malformed (non-JSON) SSE data frames and continues', async () => {
    // Manually build SSE text with one malformed frame
    const encoder = new TextEncoder()
    const sseText = [
      'event: message\ndata: {"content":"first"}\n\n',
      'event: message\ndata: THIS IS NOT JSON\n\n',
      'event: done\ndata: {}\n\n',
    ].join('')

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(sseText))
        controller.close()
      },
    })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      )
    )

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    const events = []
    for await (const { event } of chatStream({
      messages: [{ role: 'user', content: 'test' }],
      current_ast: null,
    })) {
      events.push(event)
    }

    // Malformed frame is skipped — only 2 events: message + done
    expect(events).toHaveLength(2)
    expect(events[0]).toMatchObject({ type: 'message', content: 'first' })
    expect(events[1]).toMatchObject({ type: 'done' })
    expect(warnSpy).toHaveBeenCalledOnce()
  })
})

// ---------------------------------------------------------------------------
// Additional coverage: deleteAgentV2 and createAgentV2
// ---------------------------------------------------------------------------

describe('deleteAgentV2', () => {
  it('resolves without error on 204', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    )
    await expect(deleteAgentV2('aaaa')).resolves.toBeUndefined()
  })

  it('supports force delete query parameter', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetchMock)

    await deleteAgentV2('aaaa', { force: true })

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/v1/agents-v2/aaaa?force=true')
    expect(init.method).toBe('DELETE')
  })

  it('parses active-deployment 409 into AgentDeleteError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeResponse(
          {
            detail: {
              error_kind: 'active_deployments_exist',
              active_count: 1,
              deployments: [
                {
                  id: 'dep-1',
                  mode: 'shell_app',
                  status: 'active',
                  endpoint_name: 'dr-shell-alpha',
                },
              ],
              message: 'Deactivate all deployments before deleting.',
            },
          },
          { status: 409 },
        ),
      ),
    )

    await expect(deleteAgentV2('aaaa')).rejects.toSatisfy((error: unknown) => {
      const parsed = parseAgentDeleteError(error)
      return (
        parsed instanceof AgentDeleteError &&
        parsed.error_kind === 'active_deployments_exist' &&
        parsed.deployments[0]?.id === 'dep-1'
      )
    })
  })

  it('parses app-wrapped {code: "HTTP_ERROR", message: <object>} 409 into AgentDeleteError', async () => {
    // The app's global http_exception_handler wraps every FastAPI HTTPException
    // as {code: "HTTP_ERROR", message: <exc.detail>} instead of FastAPI's default
    // {detail: <exc.detail>}. unwrapDetail must recognise both shapes.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeResponse(
          {
            code: 'HTTP_ERROR',
            message: {
              error_kind: 'active_deployments_exist',
              active_count: 1,
              deployments: [
                {
                  id: 'dep-wrap-1',
                  mode: 'shell_app',
                  status: 'active',
                  endpoint_name: 'dr-shell-beta',
                },
              ],
              message: 'Deactivate all deployments before deleting, or use ?force=true',
            },
          },
          { status: 409 },
        ),
      ),
    )

    await expect(deleteAgentV2('aaaa')).rejects.toSatisfy((error: unknown) => {
      const parsed = parseAgentDeleteError(error)
      return (
        parsed instanceof AgentDeleteError &&
        parsed.error_kind === 'active_deployments_exist' &&
        parsed.deployments[0]?.id === 'dep-wrap-1'
      )
    })
  })

  it('falls through to generic ApiError when wrapper message is a plain string', async () => {
    // Guard: the additive branch only matches when message is an object.
    // Plain-string messages must NOT be misparsed as an AgentDeleteError.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        makeResponse(
          { code: 'HTTP_ERROR', message: 'some opaque server error' },
          { status: 409 },
        ),
      ),
    )

    await expect(deleteAgentV2('aaaa')).rejects.toSatisfy((error: unknown) => {
      return (
        error instanceof ApiError &&
        !(error instanceof AgentDeleteError) &&
        error.status === 409
      )
    })
  })
})

describe('createAgentV2', () => {
  it('returns the created agent and etag', async () => {
    const agentPayload = {
      id: 'new-id',
      owner_id: 'user1',
      name: 'Brand New',
      description: null,
      avatar_url: null,
      visibility: 'private',
      definition: { root: {} },
      schema_version: 1,
      etag: 'etag-001',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeResponse(agentPayload, { status: 201, headers: { ETag: 'etag-001' } }))
    )

    const { agent, etag } = await createAgentV2({
      name: 'Brand New',
      definition: { root: {} },
    })

    expect(agent.id).toBe('new-id')
    expect(etag).toBe('etag-001')
  })
})

describe('getAgentV2WithEtag', () => {
  it('returns the agent and etag from headers', async () => {
    const agentPayload = {
      id: 'abc',
      owner_id: 'u1',
      name: 'Test',
      description: null,
      avatar_url: null,
      visibility: 'private',
      definition: {},
      schema_version: 1,
      etag: 'etag-xyz',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeResponse(agentPayload, { headers: { ETag: 'etag-xyz' } }))
    )

    const { agent, etag } = await getAgentV2WithEtag('abc')

    expect(agent.name).toBe('Test')
    expect(etag).toBe('etag-xyz')
  })
})
