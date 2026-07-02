/**
 * API client for the Agent Designer endpoints.
 *
 * Endpoints:
 *   GET  /api/v1/agent-designer/registry  — registry (cached)
 *   POST /api/v1/agent-designer/validate  — validate a workflow AST
 *   POST /api/v1/agent-designer/chat      — SSE stream of DesignerSSEEvents
 */

import { ApiError, fetchText, throwForErrorResponse } from './client'
import type { AST } from '@/types/ast'
import type {
  RegistryResponse,
  ValidateResponse,
  ChatMessage,
  DesignerSSEEvent,
  DesignerAsset,
  DesignerResource,
  DesignerResourcesResponse,
  WorkflowSummary,
} from '../types/agentDesigner'

export type {
  RegistryResponse,
  ValidateResponse,
  ChatMessage,
  DesignerSSEEvent,
  DesignerAsset,
}

export interface RefreshCatalogResponse {
  definition: AST
}

export interface ProbeSample {
  sample_input: Record<string, unknown>
  sample_output: string
  probed_at: string
  status: 'ok' | 'error' | 'skipped'
  reason?: string | null
}

export interface ProbeToolsResponse {
  samples: ProbeSample[]
  definition: AST
  persist: boolean
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// ---------------------------------------------------------------------------
// Registry (module-level cache)
// ---------------------------------------------------------------------------

let _registryCache: RegistryResponse | null = null

/** Clear the in-memory registry cache. Useful in tests. */
export function clearRegistryCache(): void {
  _registryCache = null
}

/**
 * Fetch the agent designer registry.
 * The result is cached in module memory after the first successful fetch.
 */
export async function getRegistry(): Promise<RegistryResponse> {
  if (_registryCache !== null) {
    return _registryCache
  }

  const response = await fetch(`${API_BASE_URL}/agent-designer/registry`, {
    headers: { 'Content-Type': 'application/json' },
  })

  if (!response.ok) {
    let errorData: { code?: string; message?: string } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    throw new ApiError(
      response.status,
      errorData.code ?? 'UNKNOWN',
      errorData.message ?? 'Failed to fetch registry'
    )
  }

  const data = (await response.json()) as RegistryResponse
  _registryCache = data
  return data
}

/** Global feature-gate states used to annotate gated knobs in the inspector. */
export interface DesignerCapabilities {
  skill_scripts_global: boolean
  cross_session_memory_global: boolean
  live_search_global: boolean
}

let _capabilitiesCache: DesignerCapabilities | null = null

/**
 * Fetch global feature-gate states (cached). Used by the inspector to warn when
 * a per-agent gated knob (e.g. allow_skill_scripts) is on but its global switch
 * is off — so the toggle never silently no-ops.
 */
export async function getDesignerCapabilities(): Promise<DesignerCapabilities> {
  if (_capabilitiesCache !== null) {
    return _capabilitiesCache
  }
  const response = await fetch(`${API_BASE_URL}/agent-designer/capabilities`, {
    headers: { 'Content-Type': 'application/json' },
  })
  if (!response.ok) {
    throw new ApiError(response.status, 'UNKNOWN', 'Failed to fetch capabilities')
  }
  const data = (await response.json()) as DesignerCapabilities
  _capabilitiesCache = data
  return data
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

/**
 * Validate a workflow AST against the framework loader.
 */
export async function validateWorkflow(
  definition: AST | Record<string, unknown>
): Promise<ValidateResponse> {
  const response = await fetch(`${API_BASE_URL}/agent-designer/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ definition }),
  })

  if (!response.ok) {
    let errorData: { code?: string; message?: string } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    throw new ApiError(
      response.status,
      errorData.code ?? 'UNKNOWN',
      errorData.message ?? 'Validation request failed'
    )
  }

  return response.json() as Promise<ValidateResponse>
}

// ---------------------------------------------------------------------------
// YAML import / export
// ---------------------------------------------------------------------------

/**
 * One structured warning about a designer-metadata key emitted by the import
 * carriage (invalid metadata is dropped/pruned/recomputed, never silently).
 */
export interface ImportMetadataWarning {
  key: string
  code: 'invalid_shape' | 'consistency_mismatch' | 'recomputed_divergent' | 'stale_entries_pruned'
  action: 'dropped' | 'recomputed' | 'pruned'
  message: string
  detail?: string[]
}

export interface ImportYamlResponse {
  definition: AST
  workflow_summary: WorkflowSummary
  /** Optional so responses from older servers still parse. */
  warnings?: ImportMetadataWarning[]
}

/**
 * Validate a raw YAML workflow document (structural + version + safe-parse).
 * Does NOT persist — the caller chains the returned `definition` into create.
 *
 * The body is the raw YAML string with `Content-Type: text/yaml` — NOT JSON.
 *
 * @throws {import('./client').YamlImportError} on safe-parse, registry_version,
 *   structural, or oversize failures (inspect `.errors`).
 */
export async function importYaml(yamlText: string): Promise<ImportYamlResponse> {
  const response = await fetch(`${API_BASE_URL}/agent-designer/import-yaml`, {
    method: 'POST',
    headers: { 'Content-Type': 'text/yaml' },
    body: yamlText,
  })
  if (!response.ok) {
    await throwForErrorResponse(response)
  }
  return response.json() as Promise<ImportYamlResponse>
}

/**
 * Serialize an in-memory workflow AST to a YAML document (text/yaml).
 * Used to export the live editor canvas, including unsaved / brand-new agents.
 *
 * @throws {import('./client').YamlImportError} when the AST fails framework
 *   validation (HTTP 400) so we never download an un-importable document.
 */
export async function exportYamlFromDefinition(
  definition: AST | Record<string, unknown>,
): Promise<string> {
  return fetchText('/agent-designer/export-yaml', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ definition }),
  })
}

// ---------------------------------------------------------------------------
// Tool Catalog
// ---------------------------------------------------------------------------

export async function refreshCatalog({
  definition,
  agentId,
  forceRegen = true,
}: {
  definition: AST | Record<string, unknown>
  agentId?: string | null
  forceRegen?: boolean
}): Promise<RefreshCatalogResponse> {
  const response = await fetch(`${API_BASE_URL}/agent-designer/refresh-catalog`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      definition,
      agent_id: agentId ?? null,
      force_regen: forceRegen,
    }),
  })

  if (!response.ok) {
    let errorData: { code?: string; message?: string; detail?: unknown } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    const message =
      typeof errorData.message === 'string'
        ? errorData.message
        : typeof errorData.detail === 'string'
          ? errorData.detail
          : 'Catalog refresh failed'
    throw new ApiError(response.status, errorData.code ?? 'UNKNOWN', message)
  }

  return response.json() as Promise<RefreshCatalogResponse>
}

export async function probeTools({
  definition,
  agentId,
  toolNames,
  userQuery,
  persist = false,
}: {
  definition: AST | Record<string, unknown>
  agentId?: string | null
  toolNames?: string[] | null
  userQuery?: string | null
  persist?: boolean
}): Promise<ProbeToolsResponse> {
  const response = await fetch(`${API_BASE_URL}/agent-designer/probe-tools`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      definition,
      agent_id: agentId ?? null,
      tool_names: toolNames ?? null,
      user_query: userQuery ?? null,
      persist,
    }),
  })

  if (!response.ok) {
    let errorData: { code?: string; message?: string; detail?: unknown } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    const message =
      typeof errorData.message === 'string'
        ? errorData.message
        : typeof errorData.detail === 'string'
          ? errorData.detail
          : 'Tool probe failed'
    throw new ApiError(response.status, errorData.code ?? 'UNKNOWN', message)
  }

  return response.json() as Promise<ProbeToolsResponse>
}

// ---------------------------------------------------------------------------
// Designer Resources
// ---------------------------------------------------------------------------

export async function listDesignerResources(
  kinds?: string[]
): Promise<DesignerResourcesResponse> {
  const params = new URLSearchParams()
  for (const kind of kinds ?? []) {
    params.append('kinds', kind)
  }
  const qs = params.toString()
  const response = await fetch(`${API_BASE_URL}/agent-designer/resources${qs ? `?${qs}` : ''}`, {
    headers: { 'Content-Type': 'application/json' },
  })

  if (!response.ok) {
    let errorData: { code?: string; message?: string; detail?: unknown } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    const message =
      typeof errorData.message === 'string'
        ? errorData.message
        : typeof errorData.detail === 'string'
          ? errorData.detail
          : 'Failed to fetch Designer resources'
    throw new ApiError(response.status, errorData.code ?? 'UNKNOWN', message)
  }

  return response.json() as Promise<DesignerResourcesResponse>
}

export async function startDesignerSqlWarehouse(
  warehouseId: string
): Promise<DesignerResource> {
  const response = await fetch(
    `${API_BASE_URL}/agent-designer/resources/sql-warehouses/${encodeURIComponent(warehouseId)}/start`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    }
  )

  if (!response.ok) {
    let errorData: { code?: string; message?: string; detail?: unknown } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    const message =
      typeof errorData.message === 'string'
        ? errorData.message
        : typeof errorData.detail === 'string'
          ? errorData.detail
          : 'Failed to start SQL warehouse'
    throw new ApiError(response.status, errorData.code ?? 'UNKNOWN', message)
  }

  return response.json() as Promise<DesignerResource>
}

// ---------------------------------------------------------------------------
// Chat SSE stream
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Wire payload slimming
// ---------------------------------------------------------------------------

/** Keep this many most-recent messages verbatim on the wire (current + prior turn). */
const WIRE_KEEP_RECENT = 6
/** Summarize historical tool/assistant message content longer than this (chars). */
const WIRE_CONTENT_MAX_CHARS = 2048

/**
 * Slim the transcript sent on the wire. Old `tool`/`assistant` messages carry
 * large discovery dumps and full AST snapshots that the server never consumes
 * as runtime input (it uses `current_ast`, sent separately) — they are stale
 * LLM context only. Replacing their oversized `content` with a compact summary
 * keeps the request under the server size cap and the LLM context lean without
 * dropping turns. The DISPLAYED transcript is untouched (this affects only the
 * outgoing copy). `role`, `tool_calls`, and `tool_call_id` are preserved so the
 * gateway's tool pairing stays valid; the recent window and the final user
 * message are always kept verbatim.
 */
export function slimWireMessages(messages: ChatMessage[]): ChatMessage[] {
  const n = messages.length
  return messages.map((message, i) => {
    const isRecent = i >= n - WIRE_KEEP_RECENT
    if (
      !isRecent &&
      (message.role === 'tool' || message.role === 'assistant') &&
      typeof message.content === 'string' &&
      message.content.length > WIRE_CONTENT_MAX_CHARS
    ) {
      return {
        ...message,
        content: `[${message.role} content summarized: ${message.content.length} chars omitted to fit the request budget]`,
      }
    }
    return message
  })
}

/**
 * One item from a designer SSE stream: a parsed event plus its sequence id (the
 * SSE `id:` line). `seq` is null for connection-scoped frames (`turn_started`)
 * and never set for keepalive comments — only buffered events carry a sequence,
 * which the consumer tracks as `lastSeq` to resume a reconnect from `since`.
 */
export interface DesignerStreamChunk {
  event: DesignerSSEEvent
  seq: number | null
}

/**
 * Read an SSE response body and yield parsed event frames with their sequence
 * id. Comment lines (`:keepalive`) and any line that is not `event:`/`data:`/`id:`
 * are ignored. Shared by `chatStream` (POST) and `reconnectChatStream` (GET).
 */
async function* _readSSE(response: Response): AsyncIterable<DesignerStreamChunk> {
  if (!response.body) {
    throw new ApiError(0, 'NO_BODY', 'Response body is missing')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let currentEventType: string | null = null
  let currentData: string | null = null
  let currentId: number | null = null

  const flushFrame = (): DesignerStreamChunk | null => {
    const chunk =
      currentEventType !== null && currentData !== null
        ? ((): DesignerStreamChunk | null => {
            const event = _parseFrame(currentEventType, currentData)
            return event !== null ? { event, seq: currentId } : null
          })()
        : null
    currentEventType = null
    currentData = null
    currentId = null
    return chunk
  }

  const applyLine = (rawLine: string): void => {
    // Strip trailing \r (CRLF streams).
    const trimmed = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
    if (trimmed.startsWith('event:')) {
      currentEventType = trimmed.slice('event:'.length).trim()
    } else if (trimmed.startsWith('data:')) {
      currentData = trimmed.slice('data:'.length).trim()
    } else if (trimmed.startsWith('id:')) {
      const parsed = Number.parseInt(trimmed.slice('id:'.length).trim(), 10)
      currentId = Number.isNaN(parsed) ? null : parsed
    }
    // Comment lines (':keepalive') and unknown fields are intentionally skipped.
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // Split on newlines; keep the last (possibly partial) line in the buffer.
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line === '' || line === '\r') {
          // Blank line → end of SSE frame.
          const chunk = flushFrame()
          if (chunk !== null) yield chunk
          continue
        }
        applyLine(line)
      }
    }
    // Flush any trailing buffered line, then a final complete frame.
    if (buffer) applyLine(buffer)
    const tail = flushFrame()
    if (tail !== null) yield tail
  } finally {
    reader.releaseLock()
  }
}

/**
 * Open a streaming chat session with the agent designer.
 *
 * Yields `DesignerStreamChunk` objects (event + sequence id) as they arrive. The
 * first frame is `turn_started` (carrying the turn_id for reconnect); the stream
 * ends with a `{ type: 'done' }` event.
 *
 * Throws `ApiError` with status 413 if the payload exceeds server size limits
 * (HTTP 413 is returned BEFORE any SSE frames).
 *
 * @param messages     Conversation history (at least one message).
 * @param current_ast  Current workflow AST, or null for a fresh session.
 * @param session_id   Optional correlation ID (no server-side state is stored).
 * @param assets       Optional structured assets selected for this design turn.
 * @param signal       AbortSignal to cancel the stream.
 */
export async function* chatStream({
  messages,
  current_ast,
  session_id,
  assets,
  skill_names,
  signal,
}: {
  messages: ChatMessage[]
  current_ast: AST | null
  session_id?: string | null
  assets?: DesignerAsset[]
  skill_names?: string[]
  signal?: AbortSignal
}): AsyncIterable<DesignerStreamChunk> {
  // The backend ChatMessage schema is extra="forbid", and the Databricks
  // gateway only permits tool_calls on assistant turns and tool_call_id on tool
  // turns. Replaying a tool_calls key on a user/tool message (or an empty []) —
  // which the prior unconditional map did — triggers a 400
  // "messages.N.tool_calls: Extra inputs are not permitted" on multi-turn edits.
  // Attach tool fields only where the gateway permits them, and never an empty
  // array. (The framework also flattens history defensively, but stripping at
  // the source keeps payloads small and the wire shape valid.)
  const wireMessages = slimWireMessages(messages).map((message) => {
    const wire: {
      role: ChatMessage['role']
      content: string
      tool_calls?: NonNullable<ChatMessage['tool_calls']>
      tool_call_id?: NonNullable<ChatMessage['tool_call_id']>
    } = { role: message.role, content: message.content }
    if (
      message.role === 'assistant' &&
      message.tool_calls &&
      message.tool_calls.length > 0
    ) {
      wire.tool_calls = message.tool_calls
    }
    if (message.role === 'tool' && message.tool_call_id) {
      wire.tool_call_id = message.tool_call_id
    }
    return wire
  })
  const response = await fetch(`${API_BASE_URL}/agent-designer/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: wireMessages,
      current_ast,
      session_id,
      assets: assets ?? [],
      skill_names: skill_names ?? [],
    }),
    signal,
  })

  // 413 must be surfaced as ApiError BEFORE yielding any events.
  if (response.status === 413) {
    // The backend wraps every HTTPException as { code, message }; older/raw
    // shapes use { detail }. Read `message` first so the banner shows the real
    // reason (e.g. "total payload exceeds 524288 bytes (…)") instead of a
    // generic string. Fall back to the literal for non-string (object) bodies.
    let detail = 'Request payload too large'
    try {
      const body = (await response.json()) as { message?: unknown; detail?: unknown }
      const reason = body.message ?? body.detail
      if (typeof reason === 'string' && reason) detail = reason
    } catch {
      // ignore
    }
    throw new ApiError(413, 'request_too_large', detail)
  }

  if (!response.ok) {
    let errorData: { code?: string; message?: string } = {}
    try {
      errorData = (await response.json()) as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    throw new ApiError(
      response.status,
      errorData.code ?? 'UNKNOWN',
      errorData.message ?? 'Chat request failed'
    )
  }

  yield* _readSSE(response)
}


/**
 * Resume a designer chat turn after its streamed connection was severed (the
 * Databricks Apps gateway's absolute ~4-minute response cap). Streams the
 * buffered events from sequence `since` via the GET resume route, so the client
 * picks up exactly where it left off — including a terminal `mutation_proposed`
 * + `done` produced while disconnected.
 *
 * Throws `ApiError` with status 404 when the turn is unknown or expired (the FE
 * surfaces this as "please resend" rather than retrying forever).
 */
export async function* reconnectChatStream({
  turnId,
  since,
  signal,
}: {
  turnId: string
  since: number
  signal?: AbortSignal
}): AsyncIterable<DesignerStreamChunk> {
  const response = await fetch(
    `${API_BASE_URL}/agent-designer/chat/${encodeURIComponent(turnId)}/events?since=${since}`,
    { method: 'GET', signal },
  )
  if (!response.ok) {
    let code = 'UNKNOWN'
    let detail = response.statusText || 'Reconnect failed'
    try {
      const body = (await response.json()) as {
        code?: string
        message?: unknown
        detail?: unknown
      }
      const reason = body.message ?? body.detail
      if (typeof reason === 'string' && reason) detail = reason
      if (typeof body.code === 'string') code = body.code
    } catch {
      // ignore — fall back to statusText
    }
    throw new ApiError(response.status, code, detail)
  }
  yield* _readSSE(response)
}

/**
 * Parse a single SSE frame into a `DesignerSSEEvent`.
 * Returns null if the data is malformed JSON (a warning is logged).
 */
function _parseFrame(eventType: string, data: string): DesignerSSEEvent | null {
  let payload: Record<string, unknown>
  try {
    payload = JSON.parse(data) as Record<string, unknown>
  } catch {
    console.warn(`[agentDesigner] Skipping malformed SSE frame (event=${eventType}):`, data)
    return null
  }

  // Merge the event type discriminant back in (the backend strips it via model_dump(exclude={'type'})).
  return { type: eventType, ...payload } as DesignerSSEEvent
}
