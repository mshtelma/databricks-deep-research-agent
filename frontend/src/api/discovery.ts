/**
 * Discovery API client for data source auto-discovery.
 *
 * Provides methods to interact with the discovery API:
 * - discoverSources: Get all available data sources
 * - getSourceMetadata: Get detailed metadata for a specific source
 * - refreshDiscovery: Force cache refresh
 *
 * API Contract: /specs/007-enterprise-data-sources/contracts/discovery.yaml
 */

import type {
  DataSourceType,
  DiscoveryResponse,
  RefreshDiscoveryRequest,
  SourceMetadataResponse,
  UpdateQueryConfigRequest,
  QueryConfigResponse,
} from '@/types/discovery';

const API_BASE = '/api/v1';

/**
 * Discover all available data sources for the current user.
 *
 * Results are cached for 5 minutes per user. Use refresh=true to bypass cache.
 *
 * @param options - Optional parameters
 * @param options.sourceType - Filter by source type
 * @param options.refresh - Force cache refresh
 * @param options.includeAllEndpoints - Include all serving endpoints, not just detected KAs
 * @returns Discovery response with all available sources
 */
export async function discoverSources(options?: {
  sourceType?: DataSourceType;
  refresh?: boolean;
  includeAllEndpoints?: boolean;
}): Promise<DiscoveryResponse> {
  const params = new URLSearchParams();

  if (options?.sourceType) {
    params.append('source_type', options.sourceType);
  }
  if (options?.refresh) {
    params.append('refresh', 'true');
  }
  if (options?.includeAllEndpoints) {
    params.append('include_all_endpoints', 'true');
  }

  const url = `${API_BASE}/discovery/sources${params.toString() ? `?${params.toString()}` : ''}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Discovery failed' }));
    throw new Error(error.detail || `Discovery failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Get detailed metadata for a specific discovered source.
 *
 * @param sourceId - Source identifier (e.g., 'vs:catalog.schema.index')
 * @returns Source metadata response
 */
export async function getSourceMetadata(sourceId: string): Promise<SourceMetadataResponse> {
  const url = `${API_BASE}/discovery/sources/${encodeURIComponent(sourceId)}/metadata`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  });

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Source not found: ${sourceId}`);
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to get metadata' }));
    throw new Error(error.detail || `Failed to get metadata: ${response.status}`);
  }

  return response.json();
}

/**
 * Force refresh the discovery cache.
 *
 * @param request - Optional request with source types to refresh
 * @returns Fresh discovery response
 */
export async function refreshDiscovery(
  request?: RefreshDiscoveryRequest
): Promise<DiscoveryResponse> {
  const url = `${API_BASE}/discovery/refresh`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: request ? JSON.stringify(request) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Refresh failed' }));
    throw new Error(error.detail || `Refresh failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Get discovery cache statistics.
 *
 * @returns Cache statistics
 */
export async function getDiscoveryCacheStats(): Promise<{
  total_entries: number;
  expired_entries: number;
  active_entries: number;
  by_source_type: Record<string, number>;
  ttl_seconds: number;
}> {
  const url = `${API_BASE}/discovery/stats`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to get stats' }));
    throw new Error(error.detail || `Failed to get stats: ${response.status}`);
  }

  return response.json();
}

// =============================================================================
// Query Configuration API
// =============================================================================

/**
 * Get query configuration for a data source.
 *
 * @param sourceId - Data source ID (UUID)
 * @param validate - Whether to validate config against source capabilities
 * @returns Query configuration response
 */
export async function getQueryConfig(
  sourceId: string,
  validate = false
): Promise<QueryConfigResponse> {
  const params = new URLSearchParams();
  if (validate) {
    params.append('validate', 'true');
  }

  const url = `${API_BASE}/data-sources/${sourceId}/query-config${params.toString() ? `?${params.toString()}` : ''}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to get config' }));
    throw new Error(error.detail || `Failed to get config: ${response.status}`);
  }

  return response.json();
}

/**
 * Update query configuration for a data source.
 *
 * @param sourceId - Data source ID (UUID)
 * @param request - Configuration updates
 * @param validate - Whether to validate config before saving
 * @returns Updated query configuration
 */
export async function updateQueryConfig(
  sourceId: string,
  request: UpdateQueryConfigRequest,
  validate = true
): Promise<QueryConfigResponse> {
  const params = new URLSearchParams();
  if (!validate) {
    params.append('validate', 'false');
  }

  const url = `${API_BASE}/data-sources/${sourceId}/query-config${params.toString() ? `?${params.toString()}` : ''}`;

  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to update config' }));
    throw new Error(error.detail || `Failed to update config: ${response.status}`);
  }

  return response.json();
}
