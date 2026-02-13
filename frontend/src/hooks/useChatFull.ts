import { useCallback, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { chatsApi } from '@/api/client';

export const CHAT_FULL_KEY = ['chatFull'];

/**
 * Hook to load a complete chat with all messages, sessions, sources, and claims.
 * Replaces the waterfall of useMessages + useCitations for initial page load.
 */
export function useChatFull(chatId: string | undefined) {
  return useQuery({
    queryKey: [...CHAT_FULL_KEY, chatId],
    queryFn: () => (chatId ? chatsApi.getFull(chatId) : null),
    enabled: !!chatId,
    staleTime: 2 * 60 * 1000, // Match useMessages staleTime
    gcTime: Infinity,         // Match useMessages gcTime
  });
}

const PREFETCH_DEBOUNCE_MS = 150;

/**
 * Hook for prefetching full chat data on hover, providing instant chat switching.
 * Replaces usePrefetchMessages for the chatFull pattern.
 */
export function usePrefetchChatFull() {
  const queryClient = useQueryClient();
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastPrefetchedRef = useRef<string | null>(null);

  const prefetchChatFull = useCallback((chatId: string) => {
    if (lastPrefetchedRef.current === chatId) return;
    const cached = queryClient.getQueryData([...CHAT_FULL_KEY, chatId]);
    if (cached) return;

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      queryClient.prefetchQuery({
        queryKey: [...CHAT_FULL_KEY, chatId],
        queryFn: () => chatsApi.getFull(chatId),
        staleTime: Infinity,
      });
      lastPrefetchedRef.current = chatId;
    }, PREFETCH_DEBOUNCE_MS);
  }, [queryClient]);

  return { prefetchChatFull };
}
