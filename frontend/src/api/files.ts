/**
 * API Client for File Upload operations.
 *
 * Provides functions for uploading, listing, previewing, and deleting files
 * associated with research sessions.
 */

import type {
  UploadedFile,
  FileListResponse,
  FilePreview,
} from '../types/files';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const DEFAULT_TIMEOUT_MS = 60000; // 60 seconds for uploads

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

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
  timeout?: number;
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

  // Setup timeout with AbortController
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  let response: Response;
  try {
    response = await fetch(url, {
      ...fetchOptions,
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
 * Upload a file for a session.
 * Uses FormData for multipart upload with progress tracking support.
 */
export async function uploadFile(
  file: File,
  sessionId: string,
  onProgress?: (progress: number) => void
): Promise<UploadedFile> {
  const formData = new FormData();
  formData.append('files', file);

  // Build query string for session_id (backend expects Query param, not form body)
  const params = new URLSearchParams();
  if (sessionId) params.set('session_id', sessionId);
  const qs = params.toString();

  // For progress tracking, we need to use XMLHttpRequest
  if (onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const items: UploadedFile[] = JSON.parse(xhr.responseText);
            const first = items[0];
            if (!first) {
              reject(new ApiError(xhr.status, 'EMPTY_RESPONSE', 'No files in upload response'));
              return;
            }
            resolve(first);
          } catch {
            reject(new ApiError(xhr.status, 'PARSE_ERROR', 'Failed to parse response'));
          }
        } else {
          let errorData: { code?: string; message?: string };
          try {
            errorData = JSON.parse(xhr.responseText);
          } catch {
            errorData = { code: 'UNKNOWN', message: xhr.statusText };
          }
          reject(
            new ApiError(
              xhr.status,
              errorData.code || 'UNKNOWN',
              errorData.message || 'Upload failed'
            )
          );
        }
      });

      xhr.addEventListener('error', () => {
        reject(new ApiError(0, 'NETWORK_ERROR', 'Network error during upload'));
      });

      xhr.addEventListener('abort', () => {
        reject(new ApiError(0, 'ABORTED', 'Upload was aborted'));
      });

      const uploadUrl = `${API_BASE_URL}/files/upload${qs ? `?${qs}` : ''}`;
      xhr.open('POST', uploadUrl);
      xhr.send(formData);
    });
  }

  // Simple fetch without progress
  const path = `/files/upload${qs ? `?${qs}` : ''}`;
  const items = await request<UploadedFile[]>(path, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header - browser will set it with boundary for FormData
  });
  const first = items[0];
  if (!first) {
    throw new ApiError(0, 'EMPTY_RESPONSE', 'No files in upload response');
  }
  return first;
}

/**
 * List all files for a session.
 */
export function listFiles(sessionId: string): Promise<FileListResponse> {
  return request<FileListResponse>('/files', {
    params: { session_id: sessionId },
  });
}

/**
 * Get a specific file by ID.
 */
export function getFile(fileId: string): Promise<UploadedFile> {
  return request<UploadedFile>(`/files/${fileId}`);
}

/**
 * Get file preview with sample chunks.
 */
export function getFilePreview(fileId: string): Promise<FilePreview> {
  return request<FilePreview>(`/files/${fileId}/preview`);
}

/**
 * Delete a file.
 */
export function deleteFile(fileId: string): Promise<void> {
  return request<void>(`/files/${fileId}`, {
    method: 'DELETE',
  });
}

/**
 * Files API client object for consistent usage pattern.
 */
export const filesApi = {
  /**
   * Upload a file for a session with optional progress tracking.
   */
  upload: uploadFile,

  /**
   * List all files for a session.
   */
  list: listFiles,

  /**
   * Get a specific file by ID.
   */
  get: getFile,

  /**
   * Get file preview with sample chunks.
   */
  getPreview: getFilePreview,

  /**
   * Delete a file.
   */
  delete: deleteFile,
};

export { ApiError };
