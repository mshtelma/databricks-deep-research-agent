import { useQuery } from '@tanstack/react-query';

import { listDesignerResources } from '@/api/agentDesigner';

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
