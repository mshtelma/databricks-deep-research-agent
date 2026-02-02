/**
 * Hook for managing incognito chats.
 *
 * Provides queries and mutations for:
 * - Listing incognito chats
 * - Getting session status
 * - Creating incognito chats
 * - Converting incognito chats to permanent
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { incognitoApi } from '@/api/client'
import type { Chat, IncognitoSessionStatus } from '@/types'

// Query keys
export const incognitoKeys = {
  all: ['incognito'] as const,
  chats: () => [...incognitoKeys.all, 'chats'] as const,
  session: () => [...incognitoKeys.all, 'session'] as const,
}

interface IncognitoChatsResponse {
  items: Chat[]
  total: number
  sessionExpiresAt: string | null
}

/**
 * Query for listing incognito chats for the current browser session.
 */
export function useIncognitoChats() {
  return useQuery<IncognitoChatsResponse>({
    queryKey: incognitoKeys.chats(),
    queryFn: () => incognitoApi.list(),
    staleTime: 30 * 1000, // 30 seconds - shorter stale time for session data
    refetchInterval: 60 * 1000, // Refresh every minute to update expiry
  })
}

/**
 * Query for incognito session status (quota info, expiry).
 */
export function useIncognitoSessionStatus() {
  return useQuery<IncognitoSessionStatus>({
    queryKey: incognitoKeys.session(),
    queryFn: () => incognitoApi.getSessionStatus(),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refresh every minute
  })
}

/**
 * Mutation for creating an incognito chat.
 */
export function useCreateIncognitoChat() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data?: { title?: string }) => incognitoApi.create(data),
    onSuccess: () => {
      // Immediately refetch (not just invalidate) for instant UI update
      // This ensures the IncognitoSection renders immediately after chat creation
      queryClient.refetchQueries({ queryKey: incognitoKeys.chats() })
      queryClient.refetchQueries({ queryKey: incognitoKeys.session() })
    },
  })
}

/**
 * Mutation for converting an incognito chat to a permanent chat.
 */
export function useConvertToRegular() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (chatId: string) => incognitoApi.convert(chatId),
    onSuccess: () => {
      // Invalidate both incognito and regular chats lists
      queryClient.invalidateQueries({ queryKey: incognitoKeys.chats() })
      queryClient.invalidateQueries({ queryKey: incognitoKeys.session() })
      queryClient.invalidateQueries({ queryKey: ['chats'] })
    },
  })
}

/**
 * Check if user can create more incognito chats (under quota).
 */
export function useCanCreateIncognito(): boolean {
  const { data: status } = useIncognitoSessionStatus()

  if (!status) {
    // No session yet, can create
    return true
  }

  return status.chatCount < status.maxChats
}

/**
 * Get remaining incognito chat slots.
 */
export function useIncognitoQuota(): { used: number; max: number; remaining: number } | null {
  const { data: status } = useIncognitoSessionStatus()

  if (!status) {
    return null
  }

  return {
    used: status.chatCount,
    max: status.maxChats,
    remaining: status.maxChats - status.chatCount,
  }
}
