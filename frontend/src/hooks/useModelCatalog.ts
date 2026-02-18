/**
 * TanStack Query hook for the model endpoint catalog.
 *
 * Part of 009-custom-agent-config (T025).
 */

import { useQuery } from '@tanstack/react-query';
import { getModelCatalog, getServingEndpoints } from '../api/config';
import type {
  EndpointCatalogResponse,
  ModelCategoryInfo,
  EndpointInfo,
  ServingEndpointsResponse,
} from '../api/config';

const MODEL_CATALOG_KEY = ['model-catalog'];
const SERVING_ENDPOINTS_KEY = ['serving-endpoints'];

/**
 * Hook to fetch the model endpoint catalog and workspace serving endpoints.
 *
 * Returns categories (model tiers) and endpoints for populating
 * model override inputs in the agent editor, plus workspace endpoints
 * for autocomplete suggestions.
 */
export function useModelCatalog() {
  const { data: catalog, isLoading, error } = useQuery<EndpointCatalogResponse>({
    queryKey: MODEL_CATALOG_KEY,
    queryFn: getModelCatalog,
    staleTime: 5 * 60 * 1000, // 5 minutes — catalog rarely changes
    gcTime: Infinity,
  });

  const { data: workspace, isLoading: isLoadingWorkspace } = useQuery<ServingEndpointsResponse>({
    queryKey: SERVING_ENDPOINTS_KEY,
    queryFn: getServingEndpoints,
    staleTime: 2 * 60 * 1000, // Match backend cache TTL
    gcTime: 10 * 60 * 1000,
    retry: 1, // Don't retry aggressively — autocomplete is optional
  });

  return {
    categories: (catalog?.categories ?? {}) as Record<string, ModelCategoryInfo>,
    endpoints: (catalog?.endpoints ?? {}) as Record<string, EndpointInfo>,
    workspaceEndpoints: workspace?.endpoints ?? [],
    configEndpointNames: workspace?.configEndpointNames ?? [],
    isLoading,
    isLoadingWorkspace,
    error,
  };
}

export { MODEL_CATALOG_KEY, SERVING_ENDPOINTS_KEY };
