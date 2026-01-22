import { useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { messagesApi } from '@/api/client';

const PREFETCH_DEBOUNCE_MS = 150; // Avoid rapid-fire prefetches

/**
 * Hook for prefetching messages on hover, providing instant chat switching.
 * Uses debouncing to avoid excessive API calls during rapid cursor movement.
 */
export function usePrefetchMessages() {
  const queryClient = useQueryClient();
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastPrefetchedRef = useRef<string | null>(null);

  const prefetchMessages = useCallback((chatId: string) => {
    // Skip if already prefetched recently
    if (lastPrefetchedRef.current === chatId) return;

    // Skip if already cached
    const cached = queryClient.getQueryData(['messages', chatId, undefined]);
    if (cached) return;

    // Clear pending prefetch
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Debounced prefetch
    timeoutRef.current = setTimeout(() => {
      queryClient.prefetchQuery({
        queryKey: ['messages', chatId, undefined],
        queryFn: () => messagesApi.list(chatId),
        staleTime: Infinity,
      });
      lastPrefetchedRef.current = chatId;
    }, PREFETCH_DEBOUNCE_MS);
  }, [queryClient]);

  return { prefetchMessages };
}
