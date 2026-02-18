/**
 * API Client for Custom Agents.
 *
 * Provides methods for:
 * - CRUD operations on custom agents
 * - Managing preset steps
 * - Fetching prompt templates
 */

import type {
  CustomAgent,
  CustomAgentListResponse,
  CreateCustomAgentRequest,
  UpdateCustomAgentRequest,
  ListAgentsParams,
  PresetStep,
  PresetStepsResponse,
  CreatePresetStepRequest,
  UpdatePresetStepRequest,
  ReorderPresetStepsRequest,
  PromptTemplatesResponse,
  PromptTemplateType,
} from '../types/customAgents';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const DEFAULT_TIMEOUT_MS = 30000;

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
  timeout?: number;
}

class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { params, timeout = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options;

  // Build URL with query params
  let url = `${API_BASE_URL}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      url += `?${queryString}`;
    }
  }

  // Set default headers
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  // Setup timeout with AbortController
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  let response: Response;
  try {
    response = await fetch(url, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError(0, 'TIMEOUT', `Request timed out after ${timeout}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let errorData: { code?: string; message?: string; details?: Record<string, unknown> };
    try {
      errorData = await response.json();
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText };
    }
    throw new ApiError(
      response.status,
      errorData.code || 'UNKNOWN',
      errorData.message || 'An error occurred',
      errorData.details
    );
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

/**
 * Custom Agents API client.
 */
export const customAgentsApi = {
  // ==========================================================================
  // Custom Agents
  // ==========================================================================

  /**
   * List all custom agents accessible to the user.
   */
  list: (params?: ListAgentsParams) =>
    request<CustomAgentListResponse>('/custom-agents', {
      params: params as Record<string, string | number | boolean | undefined>,
    }),

  /**
   * Get a single custom agent by ID.
   */
  get: (agentId: string) => request<CustomAgent>(`/custom-agents/${agentId}`),

  /**
   * Create a new custom agent.
   */
  create: (data: CreateCustomAgentRequest) =>
    request<CustomAgent>('/custom-agents', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update an existing custom agent.
   */
  update: (agentId: string, data: UpdateCustomAgentRequest) =>
    request<CustomAgent>(`/custom-agents/${agentId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a custom agent.
   */
  delete: (agentId: string) =>
    request<void>(`/custom-agents/${agentId}`, {
      method: 'DELETE',
    }),

  /**
   * Duplicate an existing custom agent.
   */
  duplicate: (agentId: string) =>
    request<CustomAgent>(`/custom-agents/${agentId}/duplicate`, {
      method: 'POST',
    }),

  // ==========================================================================
  // Preset Steps
  // ==========================================================================

  /**
   * List preset steps for an agent.
   */
  listPresetSteps: (agentId: string) =>
    request<PresetStepsResponse>(`/custom-agents/${agentId}/preset-steps`),

  /**
   * Get a single preset step.
   */
  getPresetStep: (agentId: string, stepId: string) =>
    request<PresetStep>(`/custom-agents/${agentId}/preset-steps/${stepId}`),

  /**
   * Create a new preset step.
   */
  createPresetStep: (agentId: string, data: CreatePresetStepRequest) =>
    request<PresetStep>(`/custom-agents/${agentId}/preset-steps`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update a preset step.
   */
  updatePresetStep: (agentId: string, stepId: string, data: UpdatePresetStepRequest) =>
    request<PresetStep>(`/custom-agents/${agentId}/preset-steps/${stepId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a preset step.
   */
  deletePresetStep: (agentId: string, stepId: string) =>
    request<void>(`/custom-agents/${agentId}/preset-steps/${stepId}`, {
      method: 'DELETE',
    }),

  /**
   * Reorder preset steps.
   */
  reorderPresetSteps: (agentId: string, data: ReorderPresetStepsRequest) =>
    request<PresetStepsResponse>(`/custom-agents/${agentId}/preset-steps/reorder`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // ==========================================================================
  // Prompt Templates
  // ==========================================================================

  /**
   * List available prompt templates.
   */
  listPromptTemplates: (params?: { type?: PromptTemplateType }) =>
    request<PromptTemplatesResponse>('/prompt-templates', {
      params: params as Record<string, string | number | boolean | undefined>,
    }),
};

export { ApiError };
