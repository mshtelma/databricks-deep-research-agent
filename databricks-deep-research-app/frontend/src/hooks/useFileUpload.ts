/**
 * React hooks for file upload operations.
 *
 * Provides hooks for:
 * - Listing session files
 * - Uploading files with progress tracking
 * - Deleting files
 * - Fetching file previews
 */

import { useCallback, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { filesApi } from '../api/files';
import type {
  UploadedFile,
  UploadProgress,
} from '../types/files';

/** Query key for session files */
const SESSION_FILES_KEY = ['session-files'];

/**
 * Hook to fetch files for a session.
 */
export function useSessionFiles(
  sessionId: string | undefined,
  options?: { enabled?: boolean }
) {
  return useQuery({
    queryKey: [...SESSION_FILES_KEY, sessionId],
    queryFn: () => (sessionId ? filesApi.list(sessionId) : null),
    enabled: !!sessionId && options?.enabled !== false,
    // Refetch periodically to check processing status
    refetchInterval: (query) => {
      // Only refetch if there are files still processing
      const data = query.state.data;
      if (data?.items?.some((f) => f.processingStatus === 'pending' || f.processingStatus === 'processing')) {
        return 3000; // 3 seconds
      }
      return false;
    },
  });
}

/**
 * Hook to fetch a single file.
 */
export function useFile(fileId: string | undefined) {
  return useQuery({
    queryKey: [...SESSION_FILES_KEY, 'file', fileId],
    queryFn: () => (fileId ? filesApi.get(fileId) : null),
    enabled: !!fileId,
  });
}

/**
 * Hook to fetch file preview.
 */
export function useFilePreview(fileId: string | undefined) {
  return useQuery({
    queryKey: [...SESSION_FILES_KEY, 'preview', fileId],
    queryFn: () => (fileId ? filesApi.getPreview(fileId) : null),
    enabled: !!fileId,
    gcTime: Infinity, // Keep preview data cached
  });
}

/**
 * Hook for file upload with progress tracking.
 */
export function useUploadFile() {
  const queryClient = useQueryClient();
  const [uploadProgress, setUploadProgress] = useState<Map<string, UploadProgress>>(new Map());

  const mutation = useMutation({
    mutationFn: async ({ file, sessionId }: { file: File; sessionId: string }) => {
      // Create a temporary ID for tracking
      const tempId = `temp-${Date.now()}-${file.name}`;

      // Initialize progress tracking
      setUploadProgress((prev) => {
        const next = new Map(prev);
        next.set(tempId, {
          fileId: tempId,
          filename: file.name,
          progress: 0,
          status: 'uploading',
        });
        return next;
      });

      try {
        const response = await filesApi.upload(file, sessionId, (progress) => {
          setUploadProgress((prev) => {
            const next = new Map(prev);
            const current = next.get(tempId);
            if (current) {
              next.set(tempId, {
                ...current,
                progress,
                status: progress === 100 ? 'processing' : 'uploading',
              });
            }
            return next;
          });
        });

        // Update with real file ID
        const status =
          response.processingStatus === 'ready'
            ? 'completed'
            : response.processingStatus === 'failed'
              ? 'failed'
              : 'processing';

        setUploadProgress((prev) => {
          const next = new Map(prev);
          next.delete(tempId);
          next.set(response.id, {
            fileId: response.id,
            filename: file.name,
            progress: 100,
            status,
          });
          return next;
        });

        // Clear finished progress after a delay; keep "processing" visible
        // until the session files query reflects current backend status.
        if (status === 'completed' || status === 'failed') {
          setTimeout(() => {
            setUploadProgress((prev) => {
              const next = new Map(prev);
              next.delete(response.id);
              return next;
            });
          }, 2000);
        }

        return response;
      } catch (error) {
        // Mark as failed
        setUploadProgress((prev) => {
          const next = new Map(prev);
          const current = next.get(tempId);
          if (current) {
            next.set(tempId, {
              ...current,
              status: 'failed',
              error: error instanceof Error ? error.message : 'Upload failed',
            });
          }
          return next;
        });
        throw error;
      }
    },
    onSuccess: (_, { sessionId }) => {
      // Invalidate the session files query to refetch
      queryClient.invalidateQueries({ queryKey: [...SESSION_FILES_KEY, sessionId] });
    },
  });

  const clearProgress = useCallback((fileId: string) => {
    setUploadProgress((prev) => {
      const next = new Map(prev);
      next.delete(fileId);
      return next;
    });
  }, []);

  const clearAllProgress = useCallback(() => {
    setUploadProgress(new Map());
  }, []);

  return {
    ...mutation,
    uploadProgress: Array.from(uploadProgress.values()),
    clearProgress,
    clearAllProgress,
  };
}

/**
 * Hook to delete a file.
 */
export function useDeleteFile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (fileId: string) => filesApi.delete(fileId),
    onSuccess: () => {
      // Invalidate all session files queries
      queryClient.invalidateQueries({ queryKey: SESSION_FILES_KEY });
    },
  });
}

/**
 * Convenience hook that combines all file operations.
 */
export function useFileUpload(
  sessionId: string | undefined,
  options?: { enabled?: boolean }
) {
  const {
    data: filesData,
    isLoading: isLoadingFiles,
    error: filesError,
    refetch: refetchFiles,
  } = useSessionFiles(sessionId, options);
  const uploadMutation = useUploadFile();
  const deleteMutation = useDeleteFile();

  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!sessionId) {
        throw new Error('Session ID is required');
      }

      const results: UploadedFile[] = [];
      const errors: { file: string; error: Error }[] = [];

      for (const file of files) {
        try {
          const result = await uploadMutation.mutateAsync({ file, sessionId });
          results.push(result);
        } catch (error) {
          errors.push({
            file: file.name,
            error: error instanceof Error ? error : new Error('Upload failed'),
          });
        }
      }

      return { results, errors };
    },
    [sessionId, uploadMutation]
  );

  const deleteFile = useCallback(
    async (fileId: string): Promise<void> => {
      return deleteMutation.mutateAsync(fileId);
    },
    [deleteMutation]
  );

  return {
    // File list
    files: filesData?.items ?? [],
    totalFiles: filesData?.total ?? 0,
    isLoadingFiles,
    filesError,
    refetchFiles,

    // Upload operations
    uploadFiles,
    uploadProgress: uploadMutation.uploadProgress,
    isUploading: uploadMutation.isPending,
    uploadError: uploadMutation.error,
    clearUploadProgress: uploadMutation.clearProgress,
    clearAllUploadProgress: uploadMutation.clearAllProgress,

    // Delete operations
    deleteFile,
    isDeleting: deleteMutation.isPending,
    deleteError: deleteMutation.error,
  };
}

// Export query key for external use
export { SESSION_FILES_KEY };
