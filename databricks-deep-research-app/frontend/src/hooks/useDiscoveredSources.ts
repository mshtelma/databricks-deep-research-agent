/**
 * React hooks for data source discovery.
 *
 * Uses TanStack Query for data fetching with 5-minute stale time
 * matching the backend cache TTL.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  discoverSources,
  getSourceMetadata,
  refreshDiscovery,
  getDiscoveryCacheStats,
} from '@/api/discovery';
import type {
  DataSourceType,
  DiscoveredSource,
  RefreshDiscoveryRequest,
} from '@/types/discovery';

// Query keys for cache management
export const discoveryKeys = {
  all: ['discovery'] as const,
  sources: () => [...discoveryKeys.all, 'sources'] as const,
  sourcesByType: (type: DataSourceType) => [...discoveryKeys.sources(), type] as const,
  metadata: (sourceId: string) => [...discoveryKeys.all, 'metadata', sourceId] as const,
  stats: () => [...discoveryKeys.all, 'stats'] as const,
};

// Cache settings (match backend TTL)
const STALE_TIME = 5 * 60 * 1000; // 5 minutes
const CACHE_TIME = 10 * 60 * 1000; // 10 minutes

/**
 * Hook to discover all available data sources.
 *
 * Always fetches the full superset (includeAllEndpoints=true) so there is
 * exactly ONE TanStack Query cache entry for discovery sources. Client-side
 * filtering handles the "show all endpoints" toggle in the UI.
 *
 * @param options - Optional parameters
 * @param options.sourceType - Filter by source type
 * @param options.enabled - Whether to enable the query
 * @returns Query result with discovered sources
 */
export function useDiscoveredSources(options?: {
  sourceType?: DataSourceType;
  enabled?: boolean;
}) {
  const queryKey = options?.sourceType
    ? discoveryKeys.sourcesByType(options.sourceType)
    : discoveryKeys.sources();

  return useQuery({
    queryKey,
    queryFn: () =>
      discoverSources({
        sourceType: options?.sourceType,
        includeAllEndpoints: true, // Always fetch superset
      }),
    staleTime: STALE_TIME,
    gcTime: CACHE_TIME,
    enabled: options?.enabled !== false,
  });
}

/**
 * Hook to get detailed metadata for a specific source.
 *
 * @param sourceId - Source identifier
 * @param enabled - Whether to enable the query
 * @returns Query result with source metadata
 */
export function useSourceMetadata(sourceId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: discoveryKeys.metadata(sourceId || ''),
    queryFn: () => getSourceMetadata(sourceId!),
    staleTime: STALE_TIME,
    gcTime: CACHE_TIME,
    enabled: enabled && !!sourceId,
  });
}

/**
 * Hook to get discovery cache statistics.
 *
 * @returns Query result with cache stats
 */
export function useDiscoveryCacheStats() {
  return useQuery({
    queryKey: discoveryKeys.stats(),
    queryFn: getDiscoveryCacheStats,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook to refresh discovery cache.
 *
 * @returns Mutation for refreshing discovery
 */
export function useRefreshDiscovery() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request?: RefreshDiscoveryRequest) => refreshDiscovery(request),
    onSuccess: (data) => {
      // Update the sources cache with fresh data
      queryClient.setQueryData(discoveryKeys.sources(), data);

      // Invalidate all discovery queries to refetch
      queryClient.invalidateQueries({ queryKey: discoveryKeys.all });
    },
  });
}

// =============================================================================
// Utility Hooks
// =============================================================================

/**
 * Hook to get sources grouped by type.
 *
 * @returns Grouped sources and loading state
 */
export function useSourcesByType() {
  const { data, isLoading, error } = useDiscoveredSources();

  return {
    vectorSearch: data?.by_type?.vector_search || [],
    genie: data?.by_type?.genie || [],
    knowledgeAssistant: data?.by_type?.knowledge_assistant || [],
    isLoading,
    error,
    totalCount: data?.total_count || 0,
    cached: data?.cached || false,
  };
}

/**
 * Hook to find a specific source by ID.
 *
 * @param sourceId - Source identifier to find
 * @returns The source if found, or undefined
 */
export function useSourceById(sourceId: string | undefined): {
  source: DiscoveredSource | undefined;
  isLoading: boolean;
} {
  const { data, isLoading } = useDiscoveredSources();

  const source = sourceId ? data?.sources?.find((s) => s.source_id === sourceId) : undefined;

  return { source, isLoading };
}

/**
 * Hook to get sources filtered by status.
 *
 * @param status - Status to filter by ('ready', 'syncing', 'unavailable', 'error')
 * @returns Filtered sources
 */
export function useSourcesByStatus(status: string) {
  const { data, isLoading } = useDiscoveredSources();

  const sources = data?.sources?.filter((s) => s.status === status) || [];

  return { sources, isLoading };
}

/**
 * Hook to check if discovery has any errors.
 *
 * @returns Error information if any
 */
export function useDiscoveryErrors() {
  const { data, isLoading, error: queryError } = useDiscoveredSources();

  return {
    hasErrors: (data?.errors?.length || 0) > 0 || !!queryError,
    errors: data?.errors || [],
    queryError,
    isLoading,
  };
}
