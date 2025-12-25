// API Client for Deep Research Agent

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'
const DEFAULT_TIMEOUT_MS = 30000 // 30 seconds

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>
  timeout?: number
}

class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: Record<string, unknown>
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { params, timeout = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options

  // Build URL with query params
  let url = `${API_BASE_URL}${endpoint}`
  if (params) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value))
      }
    })
    const queryString = searchParams.toString()
    if (queryString) {
      url += `?${queryString}`
    }
  }

  // Set default headers
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  }

  // Setup timeout with AbortController
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  let response: Response
  try {
    response = await fetch(url, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    })
  } catch (error) {
    clearTimeout(timeoutId)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError(0, 'TIMEOUT', `Request timed out after ${timeout}ms`)
    }
    throw error
  } finally {
    clearTimeout(timeoutId)
  }

  if (!response.ok) {
    let errorData: { code?: string; message?: string; details?: Record<string, unknown> }
    try {
      errorData = await response.json()
    } catch {
      errorData = { code: 'UNKNOWN', message: response.statusText }
    }
    throw new ApiError(
      response.status,
      errorData.code || 'UNKNOWN',
      errorData.message || 'An error occurred',
      errorData.details
    )
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

// Chats API
export const chatsApi = {
  list: (params?: { status?: string; search?: string; limit?: number; offset?: number }) =>
    request<import('../types').PaginatedResponse<import('../types').Chat>>('/chats', { params }),

  get: (chatId: string, includeMessages = true) =>
    request<import('../types').Chat>(`/chats/${chatId}`, {
      params: { includeMessages },
    }),

  create: (data?: { title?: string }) =>
    request<import('../types').Chat>('/chats', {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    }),

  update: (chatId: string, data: { title?: string; status?: string }) =>
    request<import('../types').Chat>(`/chats/${chatId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  delete: (chatId: string) =>
    request<void>(`/chats/${chatId}`, {
      method: 'DELETE',
    }),

  restore: (chatId: string) =>
    request<import('../types').Chat>(`/chats/${chatId}/restore`, {
      method: 'POST',
    }),

  export: async (chatId: string, format: 'markdown' | 'json'): Promise<{ content: string; filename: string }> => {
    const url = `${API_BASE_URL}/chats/${chatId}/export?format=${format}`
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
    })
    if (!response.ok) {
      let errorData: { code?: string; message?: string }
      try {
        errorData = await response.json()
      } catch {
        errorData = { code: 'UNKNOWN', message: response.statusText }
      }
      throw new ApiError(
        response.status,
        errorData.code || 'UNKNOWN',
        errorData.message || 'Export failed'
      )
    }
    const content = await response.text()
    const contentDisposition = response.headers.get('Content-Disposition') || ''
    const filenameMatch = contentDisposition.match(/filename="([^"]+)"/)
    const filename = filenameMatch?.[1] ?? `chat-${chatId}.${format === 'markdown' ? 'md' : 'json'}`
    return { content, filename }
  },
}

// Messages API
export const messagesApi = {
  list: (chatId: string, params?: { limit?: number; offset?: number }) =>
    request<import('../types').PaginatedResponse<import('../types').Message>>(
      `/chats/${chatId}/messages`,
      { params }
    ),

  send: (chatId: string, data: { content: string; research_depth?: string }) =>
    request<import('../types').SendMessageResponse>(`/chats/${chatId}/messages`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  edit: (chatId: string, messageId: string, data: { content: string }) =>
    request<{ message: import('../types').Message; removed_message_count: number }>(
      `/chats/${chatId}/messages/${messageId}`,
      {
        method: 'PATCH',
        body: JSON.stringify(data),
      }
    ),

  regenerate: (chatId: string, messageId: string) =>
    request<{ new_message_id: string; research_session_id: string }>(
      `/chats/${chatId}/messages/${messageId}/regenerate`,
      { method: 'POST' }
    ),

  submitFeedback: (
    chatId: string,
    messageId: string,
    data: { rating: -1 | 1; error_report?: string }
  ) =>
    request<import('../types').Source>(`/chats/${chatId}/messages/${messageId}/feedback`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
}

// Research API
export const researchApi = {
  cancel: (sessionId: string) =>
    request<{ session_id: string; status: string; partial_results?: string }>(
      `/research/${sessionId}/cancel`,
      { method: 'POST' }
    ),

  streamUrl: (chatId: string) => `${API_BASE_URL}/research/chats/${chatId}/stream`,
}

// Preferences API
export const preferencesApi = {
  get: () => request<import('../types').UserPreferences>('/preferences'),

  update: (data: {
    system_instructions?: string
    default_depth?: string
    ui_preferences?: Record<string, unknown>
  }) =>
    request<import('../types').UserPreferences>('/preferences', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
}

// Health API
export const healthApi = {
  check: () => request<{ status: string; database: string; version: string }>('/health'),
}

export { ApiError }
