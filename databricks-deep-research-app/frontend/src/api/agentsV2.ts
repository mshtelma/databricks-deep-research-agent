/**
 * API client for the Agents V2 CRUD endpoints.
 *
 * Endpoints:
 *   GET    /api/v1/agents-v2       — list
 *   POST   /api/v1/agents-v2       — create  (returns ETag header)
 *   GET    /api/v1/agents-v2/{id}  — get     (returns ETag header)
 *   PATCH  /api/v1/agents-v2/{id}  — update  (requires If-Match; 428 if missing; 409 on stale)
 *   DELETE /api/v1/agents-v2/{id}  — delete  (204)
 */

import { ApiError } from './client'
import type {
  AgentV2ListResponse,
  AgentV2Response,
  CreateAgentV2Request,
  UpdateAgentV2Request,
} from '../types/agentDesigner'
import type {
  ActiveDeploymentSummary,
  ActiveDeploymentsErrorResponse,
  DeploymentCleanupFailedErrorResponse,
} from '../types/deployment'

export type { AgentV2ListResponse, AgentV2Response, CreateAgentV2Request, UpdateAgentV2Request }

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'
const DEFAULT_TIMEOUT_MS = 30000

// ---------------------------------------------------------------------------
// ETag conflict error
// ---------------------------------------------------------------------------

/**
 * Thrown by `updateAgentV2` when the server responds with 409 (stale ETag).
 * `current_etag` is the server's authoritative ETag at the time of conflict.
 */
export class EtagConflictError extends Error {
  constructor(
    public readonly current_etag: string,
    message = 'ETag conflict: the agent was modified by another client'
  ) {
    super(message)
    this.name = 'EtagConflictError'
  }
}

export type AgentDeleteErrorKind =
  | 'active_deployments_exist'
  | 'deployment_cleanup_failed'

export class AgentDeleteError extends ApiError {
  constructor(
    public readonly error_kind: AgentDeleteErrorKind,
    status: number,
    message: string,
    public readonly deployments: ActiveDeploymentSummary[] = [],
    public readonly active_count = 0,
    public readonly max_attempts?: number,
  ) {
    super(status, error_kind, message, {
      error_kind,
      deployments,
      active_count,
      ...(max_attempts !== undefined ? { max_attempts } : {}),
    })
    this.name = 'AgentDeleteError'
  }
}

// ---------------------------------------------------------------------------
// Internal raw-fetch helper that returns (response, etag)
// ---------------------------------------------------------------------------

interface RawResult {
  response: Response
  etag: string | null
}

async function rawFetch(
  endpoint: string,
  init: RequestInit & { timeout?: number } = {}
): Promise<RawResult> {
  const { timeout = DEFAULT_TIMEOUT_MS, ...fetchInit } = init
  const url = `${API_BASE_URL}${endpoint}`

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  const mergedInit: RequestInit = {
    ...fetchInit,
    headers: {
      'Content-Type': 'application/json',
      ...(fetchInit.headers as Record<string, string> | undefined),
    },
    signal: controller.signal,
  }

  let response: Response
  try {
    response = await fetch(url, mergedInit)
  } catch (error) {
    clearTimeout(timeoutId)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError(0, 'TIMEOUT', `Request timed out after ${timeout}ms`)
    }
    throw error
  } finally {
    clearTimeout(timeoutId)
  }

  return { response, etag: response.headers.get('ETag') }
}

async function throwOnError(response: Response): Promise<void> {
  if (!response.ok) {
    let errorData: { code?: string; message?: string; detail?: unknown; details?: Record<string, unknown> }
    try {
      errorData = await response.json() as typeof errorData
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    const message =
      typeof errorData.message === 'string'
        ? errorData.message
        : typeof errorData.detail === 'string'
          ? errorData.detail
          : 'An error occurred'
    throw new ApiError(
      response.status,
      errorData.code || 'UNKNOWN',
      message,
      errorData.details
    )
  }
}

async function readErrorBody(response: Response): Promise<unknown> {
  try {
    return await response.json() as unknown
  } catch {
    return { message: response.statusText }
  }
}

function unwrapDetail(body: unknown): unknown {
  if (body && typeof body === 'object' && 'detail' in body) {
    return (body as { detail: unknown }).detail
  }
  return body
}

function errorMessageFromDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (detail && typeof detail === 'object') {
    const message = (detail as { message?: unknown }).message
    if (typeof message === 'string' && message.trim()) return message
  }
  return fallback
}

export function parseAgentDeleteError(error: unknown): AgentDeleteError | null {
  return error instanceof AgentDeleteError ? error : null
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** List all agents visible to the current user. */
export async function listAgentsV2(): Promise<AgentV2ListResponse> {
  const { response } = await rawFetch('/agents-v2')
  await throwOnError(response)
  return response.json() as Promise<AgentV2ListResponse>
}

/** Get a single agent by ID. */
export async function getAgentV2(id: string): Promise<AgentV2Response> {
  const { agent } = await getAgentV2WithEtag(id)
  return agent
}

/** Get a single agent by ID, also returning the server ETag. */
export async function getAgentV2WithEtag(
  id: string
): Promise<{ agent: AgentV2Response; etag: string | null }> {
  const { response, etag } = await rawFetch(`/agents-v2/${id}`)
  await throwOnError(response)
  const agent = (await response.json()) as AgentV2Response
  return { agent, etag }
}

/** Create a new agent. Returns the created agent and its initial ETag. */
export async function createAgentV2(
  req: CreateAgentV2Request
): Promise<{ agent: AgentV2Response; etag: string | null }> {
  const { response, etag } = await rawFetch('/agents-v2', {
    method: 'POST',
    body: JSON.stringify(req),
  })
  await throwOnError(response)
  const agent = (await response.json()) as AgentV2Response
  return { agent, etag }
}

/**
 * Update an existing agent using optimistic locking.
 *
 * @param id - Agent UUID.
 * @param req - Partial update payload.
 * @param etag - The ETag obtained from the last GET or create/update response.
 * @throws {EtagConflictError} When the server returns 409 (stale ETag).
 * @throws {ApiError} For 404, 428, or other HTTP errors.
 */
export async function updateAgentV2(
  id: string,
  req: UpdateAgentV2Request,
  etag: string
): Promise<{ agent: AgentV2Response; etag: string | null }> {
  const { response, etag: newEtag } = await rawFetch(`/agents-v2/${id}`, {
    method: 'PATCH',
    headers: { 'If-Match': etag },
    body: JSON.stringify(req),
  })

  if (response.status === 409) {
    // Body: { "message": "Etag conflict", "current_etag": "<value>" }
    let body: { detail?: { current_etag?: string }; current_etag?: string } = {}
    try {
      body = (await response.json()) as typeof body
    } catch {
      // ignore parse failure — we'll still throw EtagConflictError
    }
    const currentEtag =
      body.detail?.current_etag ?? body.current_etag ?? ''
    throw new EtagConflictError(currentEtag)
  }

  await throwOnError(response)
  const agent = (await response.json()) as AgentV2Response
  return { agent, etag: newEtag }
}

export interface DeleteAgentV2Options {
  force?: boolean
}

/** Delete an agent by ID. */
export async function deleteAgentV2(
  id: string,
  options: DeleteAgentV2Options = {},
): Promise<void> {
  const endpoint = `/agents-v2/${id}${options.force ? '?force=true' : ''}`
  const { response } = await rawFetch(endpoint, { method: 'DELETE' })
  if (response.ok) return

  const body = await readErrorBody(response)
  const detail = unwrapDetail(body)
  if (response.status === 409 && detail && typeof detail === 'object') {
    const errorKind = (detail as { error_kind?: unknown }).error_kind
    if (errorKind === 'active_deployments_exist') {
      const typed = detail as Partial<ActiveDeploymentsErrorResponse>
      throw new AgentDeleteError(
        'active_deployments_exist',
        response.status,
        errorMessageFromDetail(
          detail,
          'Active deployments must be deactivated before deleting this agent.',
        ),
        Array.isArray(typed.deployments) ? typed.deployments : [],
        typeof typed.active_count === 'number' ? typed.active_count : 0,
      )
    }
    if (errorKind === 'deployment_cleanup_failed') {
      const typed = detail as Partial<DeploymentCleanupFailedErrorResponse>
      throw new AgentDeleteError(
        'deployment_cleanup_failed',
        response.status,
        errorMessageFromDetail(
          detail,
          'Deployment cleanup failed. Retry delete and deactivate deployments.',
        ),
        [],
        0,
        typeof typed.max_attempts === 'number' ? typed.max_attempts : undefined,
      )
    }
  }

  throw new ApiError(
    response.status,
    'UNKNOWN',
    errorMessageFromDetail(detail, 'An error occurred'),
    detail && typeof detail === 'object'
      ? detail as Record<string, unknown>
      : undefined,
  )
}

// ---------------------------------------------------------------------------
// Revisions
// ---------------------------------------------------------------------------

/** Lightweight revision metadata returned in list responses. */
export interface RevisionSummary {
  rev_id: string
  etag: string
  created_at: string // ISO
  created_by: string
}

/** Full revision detail including the workflow AST. */
export interface RevisionDetail extends RevisionSummary {
  definition: import('../types/ast').AST
}

/** List revisions for an agent (newest first). */
export async function listRevisions(
  agentId: string,
  limit = 20,
  offset = 0,
): Promise<{ items: RevisionSummary[]; total: number }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  const { response } = await rawFetch(`/agents-v2/${agentId}/revisions?${params.toString()}`)
  await throwOnError(response)
  return response.json() as Promise<{ items: RevisionSummary[]; total: number }>
}

/** Get a single revision by ID. */
export async function getRevision(agentId: string, revId: string): Promise<RevisionDetail> {
  const { response } = await rawFetch(`/agents-v2/${agentId}/revisions/${revId}`)
  await throwOnError(response)
  return response.json() as Promise<RevisionDetail>
}
