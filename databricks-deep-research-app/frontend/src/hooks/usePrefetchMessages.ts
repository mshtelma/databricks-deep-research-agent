import { useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { chatsApi } from '@/api/client';
import { CHAT_FULL_KEY } from './useChatFull';

const PREFETCH_DEBOUNCE_MS = 150; // Avoid rapid-fire prefetches

/**
 * Hook for prefetching full chat data on hover, providing instant chat switching.
 * Prefetches via GET /chats/{id}/full (messages + claims + sources in one call).
 */
export function usePrefetchMessages() {
  const queryClient = useQueryClient();
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastPrefetchedRef = useRef<string | null>(null);

  const prefetchMessages = useCallback((chatId: string) => {
    // Skip if already prefetched recently
    if (lastPrefetchedRef.current === chatId) return;

    // Skip if already cached
    const cached = queryClient.getQueryData([...CHAT_FULL_KEY, chatId]);
    if (cached) return;

    // Clear pending prefetch
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Debounced prefetch
    timeoutRef.current = setTimeout(() => {
      queryClient.prefetchQuery({
        queryKey: [...CHAT_FULL_KEY, chatId],
        queryFn: () => chatsApi.getFull(chatId),
        staleTime: Infinity,
      });
      lastPrefetchedRef.current = chatId;
    }, PREFETCH_DEBOUNCE_MS);
  }, [queryClient]);

  return { prefetchMessages };
}
