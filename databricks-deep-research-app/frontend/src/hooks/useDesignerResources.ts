import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { listDesignerResources, startDesignerSqlWarehouse } from '@/api/agentDesigner';
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
