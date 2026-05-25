/**
 * API client for the Agent Designer Deployment endpoints (Phase 1+2 backend).
 *
 * Endpoints (mirror `src/deep_research/api/v1/deployments.py`):
 *   POST   /api/v1/deployments                          create
 *   GET    /api/v1/deployments                          list (cursor paginated)
 *   GET    /api/v1/deployments/{id}                     detail
 *   DELETE /api/v1/deployments/{id}                     deactivate
 *   GET    /api/v1/deployments/{id}/status              lightweight status poll
 *   GET    /api/v1/deployments/can-run/fast/{agent_id}  visibility probe
 *   GET    /api/v1/deployments/can-run/slow/{agent_id}  UC probe (Phase-3 stub)
 */

import type {
  CanDeployHereResponse,
  CanRunFastResponse,
  CanRunSlowResponse,
  CreateDeploymentRequest,
  DeployHereErrorKind,
  DeploymentListFilters,
  DeploymentListResponse,
  DeploymentResponse,
  DeploymentStatusResponse,
  DefaultRevisionNotDeployableErrorResponse,
} from '@/types/deployment'

const API_BASE = '/api/v1'

async function readErrorDetail(response: Response): Promise<unknown> {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text) as unknown
  } catch {
    return text
  }
}

/**
 * Thin wrapper around `fetch` that JSON-parses successful responses and
 * throws on non-2xx. Mirrors the simple style used by `discovery.ts`.
 */
async function fetchJson<T>(
  endpoint: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new DeploymentApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export class DeploymentApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: unknown,
  ) {
    super(formatDeploymentErrorMessage(status, detail))
    this.name = 'DeploymentApiError'
  }
}

function unwrapFastApiDetail(detail: unknown): unknown {
  if (detail && typeof detail === 'object' && 'detail' in detail) {
    return (detail as { detail: unknown }).detail
  }
  return detail
}

function formatDeploymentErrorMessage(status: number, detail: unknown): string {
  const unwrapped = unwrapFastApiDetail(detail)
  if (typeof unwrapped === 'string' && unwrapped.trim()) {
    return unwrapped
  }
  if (unwrapped && typeof unwrapped === 'object') {
    const message = (unwrapped as { message?: unknown }).message
    if (typeof message === 'string' && message.trim()) {
      return message
    }
  }
  return `Deployment API error: HTTP ${status}`
}

export function parseDefaultRevisionNotDeployableError(
  error: unknown,
): DefaultRevisionNotDeployableErrorResponse | null {
  if (!(error instanceof DeploymentApiError)) return null
  const detail = unwrapFastApiDetail(error.detail)
  if (!detail || typeof detail !== 'object') return null
  const body = detail as Partial<DefaultRevisionNotDeployableErrorResponse>
  if (body.error_kind !== 'default_revision_not_deployable') return null
  return {
    error_kind: 'default_revision_not_deployable',
    agent_id: String(body.agent_id ?? ''),
    revision_id: String(body.revision_id ?? ''),
    workflow_name: String(body.workflow_name ?? ''),
    root_child_summary: Array.isArray(body.root_child_summary)
      ? body.root_child_summary.map(String)
      : [],
    message:
      typeof body.message === 'string' && body.message.trim()
        ? body.message
        : 'Save or select a designed workflow revision before deploying.',
  }
}

export function formatDefaultRevisionNotDeployableError(
  error: unknown,
): string | null {
  const blocked = parseDefaultRevisionNotDeployableError(error)
  if (!blocked) return null
  const revision = blocked.revision_id ? blocked.revision_id.slice(0, 8) : 'unknown'
  const workflowName = blocked.workflow_name || 'Untitled Agent'
  return `${blocked.message} Revision ${revision} (${workflowName}) is not deployable.`
}

export class DeploymentActionError extends DeploymentApiError {
  constructor(
    public readonly error_kind: DeployHereErrorKind | string,
    status: number,
    detail: string,
    public readonly externalResourceIds: Record<string, unknown> | null = null,
  ) {
    super(status, detail)
    this.name = 'DeploymentActionError'
    this.message =
      `Deployment action error (${error_kind}): HTTP ${status}` +
      (detail ? ` — ${detail}` : '')
  }
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

export function createDeployment(
  body: CreateDeploymentRequest,
  options: { runAsync?: boolean } = {},
): Promise<DeploymentResponse> {
  const endpoint =
    options.runAsync === false
      ? '/deployments?run_async=false'
      : '/deployments'
  return fetchJson<DeploymentResponse>(endpoint, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function listDeployments(
  filters: DeploymentListFilters = {},
): Promise<DeploymentListResponse> {
  const params = new URLSearchParams()
  if (filters.mode) params.set('mode', filters.mode)
  if (filters.status) params.set('status', filters.status)
  if (filters.agent_id) params.set('agent_id', filters.agent_id)
  if (filters.cursor) params.set('cursor', filters.cursor)
  if (filters.limit !== undefined) params.set('limit', String(filters.limit))
  const qs = params.toString()
  return fetchJson<DeploymentListResponse>(
    `/deployments${qs ? `?${qs}` : ''}`,
  )
}

export function getDeployment(id: string): Promise<DeploymentResponse> {
  return fetchJson<DeploymentResponse>(`/deployments/${id}`)
}

export function deactivateDeployment(
  id: string,
): Promise<DeploymentResponse> {
  return fetchJson<DeploymentResponse>(`/deployments/${id}`, {
    method: 'DELETE',
  })
}

export function getDeploymentStatus(
  id: string,
): Promise<DeploymentStatusResponse> {
  return fetchJson<DeploymentStatusResponse>(`/deployments/${id}/status`)
}

// ---------------------------------------------------------------------------
// Capability probes
// ---------------------------------------------------------------------------

export function canRunFast(agentId: string): Promise<CanRunFastResponse> {
  return fetchJson<CanRunFastResponse>(
    `/deployments/can-run/fast/${agentId}`,
  )
}

export function canRunSlow(agentId: string): Promise<CanRunSlowResponse> {
  return fetchJson<CanRunSlowResponse>(
    `/deployments/can-run/slow/${agentId}`,
  )
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/**
 * POST /api/v1/deployments/{id}/actions/deploy-here
 *
 * Returns the updated DeploymentResponse. Deploy-here normally returns
 * 'deploying' immediately; callers poll /status for 'active' or 'failed'.
 *
 * Throws DeploymentActionError on 4xx with a structured error_kind so the
 * calling wizard can switch on it.
 */
export async function deployHereAction(
  deploymentId: string,
  opts?: { confirmRedeploy?: boolean },
): Promise<DeploymentResponse> {
  const qs = opts?.confirmRedeploy ? '?confirm_redeploy=1' : ''
  const response = await fetch(
    `${API_BASE}/deployments/${deploymentId}/actions/deploy-here${qs}`,
    { method: 'POST', credentials: 'include' },
  )
  let body: Record<string, unknown>
  try {
    body = (await response.json()) as Record<string, unknown>
  } catch {
    throw new DeploymentApiError(response.status, response.statusText)
  }
  if (!response.ok) {
    const msg = body.message as Record<string, unknown> | undefined
    const error_kind = (msg?.error_kind ?? '') as string
    const detail = (msg?.detail ?? '') as string
    // Some error kinds carry extra context in deployment.external_resource_ids
    const deployment = body.deployment as Record<string, unknown> | undefined
    const externalResourceIds =
      (deployment?.external_resource_ids as Record<string, unknown> | null | undefined) ??
      null
    throw new DeploymentActionError(
      error_kind as DeployHereErrorKind,
      response.status,
      detail,
      externalResourceIds,
    )
  }
  return body as unknown as DeploymentResponse
}

// ---------------------------------------------------------------------------
// Can-deploy-here probe
// ---------------------------------------------------------------------------

/**
 * GET /api/v1/deployments/can-deploy-here
 *
 * Probes whether the current actor can deploy in this workspace.
 * Always returns 200. Explicit denials use can_deploy=false with a reason;
 * probe_status="unknown" means the advisory probe failed but deploy can try.
 */
export async function canDeployHereAction(): Promise<CanDeployHereResponse> {
  const resp = await fetch(`${API_BASE}/deployments/can-deploy-here`, {
    method: 'GET',
    credentials: 'include',
  })
  if (!resp.ok) {
    const detail = await readErrorDetail(resp)
    throw new DeploymentApiError(resp.status, detail)
  }
  return (await resp.json()) as CanDeployHereResponse
}

/**
 * POST /api/v1/deployments/can-deploy-here/refresh
 *
 * Invalidates the server-side cache then re-probes. Same response shape.
 */
export async function refreshCanDeployHereAction(): Promise<CanDeployHereResponse> {
  const resp = await fetch(`${API_BASE}/deployments/can-deploy-here/refresh`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!resp.ok) {
    const detail = await readErrorDetail(resp)
    throw new DeploymentApiError(resp.status, detail)
  }
  return (await resp.json()) as CanDeployHereResponse
}
