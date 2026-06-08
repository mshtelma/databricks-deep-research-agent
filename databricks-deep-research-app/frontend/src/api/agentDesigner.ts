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

export interface ImportYamlResponse {
  definition: AST
  workflow_summary: WorkflowSummary
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

/**
 * Open a streaming chat session with the agent designer.
 *
 * Yields `DesignerSSEEvent` objects as they arrive. The stream ends with a
 * `{ type: 'done' }` event.
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
  signal,
}: {
  messages: ChatMessage[]
  current_ast: AST | null
  session_id?: string | null
  assets?: DesignerAsset[]
  signal?: AbortSignal
}): AsyncIterable<DesignerSSEEvent> {
  const wireMessages = messages.map((message) => ({
    role: message.role,
    content: message.content,
    tool_calls: message.tool_calls,
    tool_call_id: message.tool_call_id,
  }))
  const response = await fetch(`${API_BASE_URL}/agent-designer/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: wireMessages, current_ast, session_id, assets: assets ?? [] }),
    signal,
  })

  // 413 must be surfaced as ApiError BEFORE yielding any events.
  if (response.status === 413) {
    let detail = 'Request payload too large'
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
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

  if (!response.body) {
    throw new ApiError(0, 'NO_BODY', 'Response body is missing')
  }

  // Read the SSE stream incrementally.
  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  // Accumulate text across chunks; frames are delimited by blank lines.
  let buffer = ''

  // State for the current incomplete SSE frame.
  let currentEventType: string | null = null
  let currentData: string | null = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Split on newlines and process complete lines.
      // We keep the last partial line in the buffer.
      const lines = buffer.split('\n')
      // The last element may be an incomplete line — hold it back.
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (line === '' || line === '\r') {
          // Blank line → end of SSE frame.
          if (currentEventType !== null && currentData !== null) {
            const event = _parseFrame(currentEventType, currentData)
            if (event !== null) {
              yield event
            }
          }
          currentEventType = null
          currentData = null
          continue
        }

        // Strip trailing \r (CRLF streams)
        const trimmed = line.endsWith('\r') ? line.slice(0, -1) : line

        if (trimmed.startsWith('event:')) {
          currentEventType = trimmed.slice('event:'.length).trim()
        } else if (trimmed.startsWith('data:')) {
          currentData = trimmed.slice('data:'.length).trim()
        }
        // Lines not starting with 'event:' or 'data:' are intentionally skipped.
      }
    }

    // Flush any remaining buffered content after the stream ends.
    if (buffer) {
      const trimmed = buffer.endsWith('\r') ? buffer.slice(0, -1) : buffer
      if (trimmed.startsWith('event:')) {
        currentEventType = trimmed.slice('event:'.length).trim()
      } else if (trimmed.startsWith('data:')) {
        currentData = trimmed.slice('data:'.length).trim()
      }
    }

    // Yield the final frame if complete.
    if (currentEventType !== null && currentData !== null) {
      const event = _parseFrame(currentEventType, currentData)
      if (event !== null) {
        yield event
      }
    }
  } finally {
    reader.releaseLock()
  }
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
