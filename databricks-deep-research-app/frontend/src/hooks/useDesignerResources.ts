import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import {
  getUcFunctionSignature,
  listDesignerResources,
  startDesignerSqlWarehouse,
} from '@/api/agentDesigner';
import type { DesignerResourcesResponse } from '@/types/agentDesigner';

export const designerResourceKeys = {
  all: ['agent-designer', 'resources'] as const,
  byKinds: (kinds: string[]) => [...designerResourceKeys.all, [...kinds].sort().join(',')] as const,
};

const STALE_TIME = 5 * 60 * 1000;
const CACHE_TIME = 10 * 60 * 1000;

export function useDesignerResources(kinds: string[], enabled = true) {
  return useQuery({
    queryKey: designerResourceKeys.byKinds(kinds),
    queryFn: () => listDesignerResources(kinds),
    staleTime: STALE_TIME,
    gcTime: CACHE_TIME,
    enabled: enabled && kinds.length > 0,
  });
}

export const ucBrowseKeys = {
  all: ['agent-designer', 'uc-browse'] as const,
  level: (kind: string, parent: string) => [...ucBrowseKeys.all, kind, parent] as const,
  signature: (fqn: string) => [...ucBrowseKeys.all, 'signature', fqn] as const,
}

export type UcBrowseKind = 'uc_catalog' | 'uc_schema' | 'uc_function'

/**
 * Browse one level of the Unity Catalog cascade. Keyed on (kind, parent) only —
 * the full child list is fetched once per parent and the native <datalist>
 * filters as the user types, so keystrokes never trigger fetches. A child level
 * stays disabled until its parent is passed (the caller gates on a *committed*
 * parent so a half-typed name never fires a doomed query).
 */
export function useUcBrowse(kind: UcBrowseKind, parent: string | undefined, enabled = true) {
  const ready = enabled && (kind === 'uc_catalog' || Boolean(parent))
  return useQuery({
    queryKey: ucBrowseKeys.level(kind, parent ?? ''),
    queryFn: () => listDesignerResources([kind], { parent }),
    staleTime: STALE_TIME,
    gcTime: CACHE_TIME,
    enabled: ready,
  })
}

/**
 * Prefix search for UC functions. `parent` is either `catalog.schema` (single
 * SHOW USER FUNCTIONS) or a bare catalog (budgeted server-side fan-out; the
 * response's `warning` reports truncation). Keyed on (parent, prefix) with a
 * short staleTime — the server keeps its own 60s cache per user/scope/prefix.
 */
export function useUcFunctionSearch(
  parent: string | undefined,
  prefix: string,
  enabled = true,
) {
  return useQuery({
    queryKey: [...ucBrowseKeys.all, 'search', parent ?? '', prefix] as const,
    queryFn: () => listDesignerResources(['uc_function'], { parent, query: prefix }),
    staleTime: 30 * 1000,
    gcTime: CACHE_TIME,
    enabled: enabled && Boolean(parent),
    placeholderData: (previous: DesignerResourcesResponse | undefined) => previous,
  });
}

/** Live signature for a chosen UC function (drives auto parameter mapping). */
export function useUcFunctionSignature(fqn: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ucBrowseKeys.signature(fqn ?? ''),
    queryFn: () => getUcFunctionSignature(fqn ?? ''),
    staleTime: STALE_TIME,
    gcTime: CACHE_TIME,
    enabled: enabled && Boolean(fqn),
  })
}

export function useStartDesignerSqlWarehouse() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: startDesignerSqlWarehouse,
    onSuccess: (resource) => {
      queryClient.setQueryData<DesignerResourcesResponse>(
        designerResourceKeys.byKinds(['sql_warehouse']),
        (current) => {
          if (!current) return current;
          const nextResources = current.resources.map((item) => {
            const itemId = resourceWarehouseId(item);
            const resourceId = resourceWarehouseId(resource);
            return itemId && itemId === resourceId ? resource : item;
          });
          return { ...current, resources: nextResources };
        },
      );
      void queryClient.invalidateQueries({
        queryKey: designerResourceKeys.byKinds(['sql_warehouse']),
      });
    },
  });
}

function resourceWarehouseId(resource: {
  source_id?: string | null;
  metadata: Record<string, unknown>;
}) {
  const metadataId = resource.metadata['warehouse_id'];
  if (typeof metadataId === 'string' && metadataId.length > 0) return metadataId;
  return resource.source_id ?? '';
}
