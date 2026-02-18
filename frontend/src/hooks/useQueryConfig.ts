/**
 * React hooks for query configuration management.
 *
 * Provides per-source config state management with:
 * - Save/load from API
 * - Validation before save
 * - Optimistic updates
 *
 * Part of US9b (T010x).
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getQueryConfig, updateQueryConfig } from '@/api/discovery';
import type {
  VectorSearchQueryConfig,
  UpdateQueryConfigRequest,
  QueryConfigResponse,
} from '@/types/discovery';

// Query keys for cache management
export const queryConfigKeys = {
  all: ['queryConfig'] as const,
  bySource: (sourceId: string) => [...queryConfigKeys.all, sourceId] as const,
};

// Cache settings
const STALE_TIME = 5 * 60 * 1000; // 5 minutes
const CACHE_TIME = 10 * 60 * 1000; // 10 minutes

/**
 * Hook to get query configuration for a data source.
 *
 * @param sourceId - Data source ID
 * @param options - Optional parameters
 * @param options.enabled - Whether to enable the query
 * @param options.validate - Whether to validate config against source capabilities
 * @returns Query result with query config
 */
export function useQueryConfig(
  sourceId: string | undefined,
  options?: {
    enabled?: boolean;
    validate?: boolean;
  }
) {
  return useQuery({
    queryKey: sourceId ? queryConfigKeys.bySource(sourceId) : queryConfigKeys.all,
    queryFn: () =>
      sourceId
        ? getQueryConfig(sourceId, options?.validate)
        : Promise.reject(new Error('No source ID')),
    staleTime: STALE_TIME,
    gcTime: CACHE_TIME,
    enabled: (options?.enabled !== false) && !!sourceId,
  });
}

/**
 * Hook to update query configuration for a data source.
 *
 * @returns Mutation for updating query config
 */
export function useUpdateQueryConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      sourceId,
      request,
      validate = true,
    }: {
      sourceId: string;
      request: UpdateQueryConfigRequest;
      validate?: boolean;
    }) => updateQueryConfig(sourceId, request, validate),

    onMutate: async ({ sourceId, request }) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: queryConfigKeys.bySource(sourceId) });

      // Snapshot the previous value
      const previousConfig = queryClient.getQueryData<QueryConfigResponse>(
        queryConfigKeys.bySource(sourceId)
      );

      // Optimistically update to the new value
      if (previousConfig) {
        queryClient.setQueryData<QueryConfigResponse>(queryConfigKeys.bySource(sourceId), {
          ...previousConfig,
          config: {
            ...previousConfig.config,
            ...request,
            // Handle filters specially - use provided filters or keep existing
            filters: request.filters ?? previousConfig.config.filters,
          } as VectorSearchQueryConfig,
        });
      }

      return { previousConfig };
    },

    onError: (_err, { sourceId }, context) => {
      // Rollback to the previous value on error
      if (context?.previousConfig) {
        queryClient.setQueryData(queryConfigKeys.bySource(sourceId), context.previousConfig);
      }
    },

    onSuccess: (data, { sourceId }) => {
      // Update cache with server response
      queryClient.setQueryData(queryConfigKeys.bySource(sourceId), data);
    },
  });
}

/**
 * Hook to manage query config state locally before saving.
 *
 * Provides a local state for editing with save/reset functionality.
 *
 * @param sourceId - Data source ID
 * @returns Local config state and actions
 */
export function useLocalQueryConfig(sourceId: string | undefined) {
  const { data, isLoading, error } = useQueryConfig(sourceId);
  const updateMutation = useUpdateQueryConfig();

  return {
    // State
    config: data?.config,
    validation: data?.validation,
    isLoading,
    error,
    isSaving: updateMutation.isPending,
    saveError: updateMutation.error,

    // Actions
    save: async (config: VectorSearchQueryConfig, validate = true) => {
      if (!sourceId) throw new Error('No source ID');

      return updateMutation.mutateAsync({
        sourceId,
        request: {
          query_type: config.query_type,
          num_results: config.num_results,
          score_threshold: config.score_threshold,
          columns: config.columns,
          enable_reranking: config.enable_reranking,
          columns_to_rerank: config.columns_to_rerank,
          filters: config.filters,
          filter_syntax: config.filter_syntax,
        },
        validate,
      });
    },
  };
}

/**
 * Hook to prefetch query config for better UX.
 *
 * @param sourceId - Data source ID to prefetch
 */
export function usePrefetchQueryConfig(sourceId: string | undefined) {
  const queryClient = useQueryClient();

  return () => {
    if (sourceId) {
      queryClient.prefetchQuery({
        queryKey: queryConfigKeys.bySource(sourceId),
        queryFn: () => getQueryConfig(sourceId),
        staleTime: STALE_TIME,
      });
    }
  };
}
