/**
 * React Query hooks for Custom Agents.
 *
 * Provides hooks for:
 * - Fetching and managing custom agents
 * - Managing preset steps
 * - Fetching prompt templates
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { customAgentsApi } from '../api/customAgents';
import type {
  CreateCustomAgentRequest,
  UpdateCustomAgentRequest,
  ListAgentsParams,
  CreatePresetStepRequest,
  UpdatePresetStepRequest,
  ReorderPresetStepsRequest,
  PromptTemplateType,
} from '../types/customAgents';

// =============================================================================
// Query Keys
// =============================================================================

const CUSTOM_AGENTS_KEY = ['custom-agents'];
const PRESET_STEPS_KEY = ['preset-steps'];
const PROMPT_TEMPLATES_KEY = ['prompt-templates'];

// =============================================================================
// Custom Agents Hooks
// =============================================================================

/**
 * Hook to fetch all custom agents accessible to the user.
 */
export function useCustomAgents(
  params?: ListAgentsParams,
  options?: { enabled?: boolean }
) {
  return useQuery({
    queryKey: [...CUSTOM_AGENTS_KEY, params],
    queryFn: () => customAgentsApi.list(params),
    enabled: options?.enabled !== false,
    staleTime: 5 * 60 * 1000, // Custom agents change infrequently in chat flow
    gcTime: Infinity,
  });
}

/**
 * Hook to fetch a single custom agent by ID.
 */
export function useCustomAgent(agentId: string | undefined) {
  return useQuery({
    queryKey: [...CUSTOM_AGENTS_KEY, agentId],
    queryFn: () => (agentId ? customAgentsApi.get(agentId) : null),
    enabled: !!agentId,
    gcTime: Infinity,
  });
}

/**
 * Hook to create a new custom agent.
 */
export function useCreateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateCustomAgentRequest) => customAgentsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CUSTOM_AGENTS_KEY });
    },
  });
}

/**
 * Hook to update an existing custom agent.
 */
export function useUpdateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: UpdateCustomAgentRequest }) =>
      customAgentsApi.update(agentId, data),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: CUSTOM_AGENTS_KEY });
      queryClient.invalidateQueries({ queryKey: [...CUSTOM_AGENTS_KEY, agentId] });
    },
  });
}

/**
 * Hook to delete a custom agent.
 */
export function useDeleteAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (agentId: string) => customAgentsApi.delete(agentId),
    onSuccess: (_, agentId) => {
      queryClient.invalidateQueries({ queryKey: CUSTOM_AGENTS_KEY });
      queryClient.removeQueries({ queryKey: [...CUSTOM_AGENTS_KEY, agentId] });
      // Also remove preset steps queries for this agent
      queryClient.removeQueries({ queryKey: [...PRESET_STEPS_KEY, agentId] });
    },
  });
}

/**
 * Hook to duplicate a custom agent.
 */
export function useDuplicateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (agentId: string) => customAgentsApi.duplicate(agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CUSTOM_AGENTS_KEY });
    },
  });
}

// =============================================================================
// Preset Steps Hooks
// =============================================================================

/**
 * Hook to fetch preset steps for an agent.
 */
export function useAgentPresetSteps(agentId: string | undefined) {
  return useQuery({
    queryKey: [...PRESET_STEPS_KEY, agentId],
    queryFn: () => (agentId ? customAgentsApi.listPresetSteps(agentId) : null),
    enabled: !!agentId,
    gcTime: Infinity,
  });
}

/**
 * Hook to create a new preset step.
 */
export function useCreatePresetStep() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: CreatePresetStepRequest }) =>
      customAgentsApi.createPresetStep(agentId, data),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: [...PRESET_STEPS_KEY, agentId] });
    },
  });
}

/**
 * Hook to update a preset step.
 */
export function useUpdatePresetStep() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      agentId,
      stepId,
      data,
    }: {
      agentId: string;
      stepId: string;
      data: UpdatePresetStepRequest;
    }) => customAgentsApi.updatePresetStep(agentId, stepId, data),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: [...PRESET_STEPS_KEY, agentId] });
    },
  });
}

/**
 * Hook to delete a preset step.
 */
export function useDeletePresetStep() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, stepId }: { agentId: string; stepId: string }) =>
      customAgentsApi.deletePresetStep(agentId, stepId),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: [...PRESET_STEPS_KEY, agentId] });
    },
  });
}

/**
 * Hook to reorder preset steps.
 */
export function useReorderPresetSteps() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: ReorderPresetStepsRequest }) =>
      customAgentsApi.reorderPresetSteps(agentId, data),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: [...PRESET_STEPS_KEY, agentId] });
    },
  });
}

// =============================================================================
// Prompt Templates Hooks
// =============================================================================

/**
 * Hook to fetch available prompt templates.
 */
export function usePromptTemplates(type?: PromptTemplateType) {
  return useQuery({
    queryKey: [...PROMPT_TEMPLATES_KEY, type],
    queryFn: () => customAgentsApi.listPromptTemplates(type ? { type } : undefined),
    gcTime: Infinity,
    staleTime: 5 * 60 * 1000, // 5 minutes - templates don't change often
  });
}

// =============================================================================
// Convenience Hooks
// =============================================================================

/**
 * Convenience hook that returns agents grouped by visibility.
 * Useful for the AgentSelector component.
 */
export function useGroupedAgents(params?: ListAgentsParams) {
  const { data, isLoading, error } = useCustomAgents(params);

  const grouped = {
    systemAgents: [] as import('../types/customAgents').CustomAgentSummary[],
    workspaceAgents: [] as import('../types/customAgents').CustomAgentSummary[],
    userAgents: [] as import('../types/customAgents').CustomAgentSummary[],
  };

  if (data?.agents) {
    for (const agent of data.agents) {
      // System agents have a special ownerId or no owner
      if (!agent.ownerId || agent.ownerId === 'system') {
        grouped.systemAgents.push(agent);
      } else if (agent.inAppActive) {
        grouped.workspaceAgents.push(agent);
      } else {
        grouped.userAgents.push(agent);
      }
    }
  }

  return {
    grouped,
    agents: data?.agents ?? [],
    total: data?.total ?? 0,
    userAgents: data?.userAgents ?? 0,
    workspaceAgents: data?.workspaceAgents ?? 0,
    systemAgents: data?.systemAgents ?? 0,
    isLoading,
    error,
  };
}

// =============================================================================
// Export Query Keys
// =============================================================================

export { CUSTOM_AGENTS_KEY, PRESET_STEPS_KEY, PROMPT_TEMPLATES_KEY };
