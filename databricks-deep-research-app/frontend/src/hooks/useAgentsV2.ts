/**
 * TanStack Query hooks for the Agents V2 CRUD API.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listAgentsV2,
  getAgentV2WithEtag,
  createAgentV2,
  updateAgentV2,
  deleteAgentV2,
  type DeleteAgentV2Options,
} from '@/api/agentsV2'
import { ApiError } from '@/api/client'
import type {
  AgentV2Summary,
  AgentV2Response,
  CreateAgentV2Request,
  UpdateAgentV2Request,
} from '@/types/agentDesigner'

export type { AgentV2Summary, AgentV2Response }

export type DeleteAgentV2Variables =
  | string
  | ({ id: string } & DeleteAgentV2Options)

function normalizeDeleteVariables(
  variables: DeleteAgentV2Variables,
): { id: string; force?: boolean } {
  return typeof variables === 'string' ? { id: variables } : variables
}

// =============================================================================
// Query Keys
// =============================================================================

export const agentsV2Keys = {
  all: ['agents-v2'] as const,
  list: () => [...agentsV2Keys.all, 'list'] as const,
  detail: (id: string) => [...agentsV2Keys.all, 'detail', id] as const,
}

// =============================================================================
// Query Hooks
// =============================================================================

/**
 * Fetches all agents_v2 visible to the current user.
 */
export function useAgentsV2List() {
  return useQuery({
    queryKey: agentsV2Keys.list(),
    queryFn: listAgentsV2,
  })
}

/**
 * Fetches a single agent by ID.
 * Disabled when `id` is undefined.
 */
export function useAgentV2(id: string | undefined) {
  return useQuery({
    queryKey: agentsV2Keys.detail(id ?? ''),
    queryFn: () => getAgentV2WithEtag(id!).then(({ agent }) => agent),
    enabled: id !== undefined,
  })
}

// =============================================================================
// Mutation Hooks
// =============================================================================

/**
 * Creates a new agent. Invalidates the list on success.
 */
export function useCreateAgentV2() {
  const queryClient = useQueryClient()

  return useMutation<
    { agent: AgentV2Response; etag: string | null },
    ApiError,
    CreateAgentV2Request & { force?: boolean }
  >({
    mutationFn: ({ force, ...req }) => createAgentV2(req, { force }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: agentsV2Keys.list() })
    },
  })
}

/**
 * Updates an existing agent using optimistic locking.
 * Variables: `{ id, req, etag }`.
 * Invalidates both the list and the detail on success.
 */
export function useUpdateAgentV2() {
  const queryClient = useQueryClient()

  return useMutation<
    { agent: AgentV2Response; etag: string | null },
    ApiError,
    { id: string; req: UpdateAgentV2Request; etag: string }
  >({
    mutationFn: ({ id, req, etag }) => updateAgentV2(id, req, etag),
    onSuccess: (_, { id }) => {
      void queryClient.invalidateQueries({ queryKey: agentsV2Keys.list() })
      void queryClient.invalidateQueries({ queryKey: agentsV2Keys.detail(id) })
    },
  })
}

/**
 * Deletes an agent by ID. Invalidates the list and removes the detail on success.
 */
export function useDeleteAgentV2() {
  const queryClient = useQueryClient()

  return useMutation<void, ApiError, DeleteAgentV2Variables>({
    mutationFn: (variables) => {
      const { id, force } = normalizeDeleteVariables(variables)
      return force ? deleteAgentV2(id, { force: true }) : deleteAgentV2(id)
    },
    onSuccess: (_, variables) => {
      const { id } = normalizeDeleteVariables(variables)
      void queryClient.invalidateQueries({ queryKey: agentsV2Keys.list() })
      queryClient.removeQueries({ queryKey: agentsV2Keys.detail(id) })
    },
  })
}
