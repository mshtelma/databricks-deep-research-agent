// API Client for Templates

import type {
  Template,
  TemplateListResponse,
  TemplateListParams,
  CreateTemplateRequest,
  UpdateTemplateRequest,
  RenderTemplateRequest,
  RenderTemplateResponse,
  TemplateType,
} from '../types/templates';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const DEFAULT_TIMEOUT_MS = 30000; // 30 seconds

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
 * Templates API client.
 */
export const templatesApi = {
  /**
   * List all templates accessible to the user.
   */
  list: (params?: TemplateListParams) =>
    request<TemplateListResponse>('/templates', {
      params: {
        type: params?.type,
        visibility: params?.visibility,
        search: params?.search,
        tags: params?.tags?.join(','),
        include_system: params?.includeSystem,
      },
    }),

  /**
   * Get a single template by ID.
   */
  get: (templateId: string) => request<Template>(`/templates/${templateId}`),

  /**
   * Create a new template.
   */
  create: (data: CreateTemplateRequest) =>
    request<Template>('/templates', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update an existing template.
   */
  update: (templateId: string, data: UpdateTemplateRequest) =>
    request<Template>(`/templates/${templateId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a template.
   */
  delete: (templateId: string) =>
    request<void>(`/templates/${templateId}`, {
      method: 'DELETE',
    }),

  /**
   * Render a template with provided variables.
   */
  render: (data: RenderTemplateRequest) =>
    request<RenderTemplateResponse>('/templates/render', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Get the default template for a specific type.
   */
  getDefault: (type: TemplateType) =>
    request<Template | null>(`/templates/default/${type}`),

  /**
   * Set a template as the default for its type.
   */
  setDefault: (templateId: string) =>
    request<Template>(`/templates/${templateId}/set-default`, {
      method: 'POST',
    }),

  /**
   * Clone an existing template.
   */
  clone: (templateId: string, newName?: string) =>
    request<Template>(`/templates/${templateId}/clone`, {
      method: 'POST',
      body: JSON.stringify({ name: newName }),
    }),
};

export { ApiError };
