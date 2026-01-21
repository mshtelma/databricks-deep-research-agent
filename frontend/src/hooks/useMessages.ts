import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { messagesApi } from '../api/client'
import type { ResearchDepth } from '../types'

const MESSAGES_KEY = ['messages']

export function useMessages(
  chatId: string | undefined,
  params?: { limit?: number; offset?: number }
) {
  return useQuery({
    queryKey: [...MESSAGES_KEY, chatId, params],
    queryFn: () => (chatId ? messagesApi.list(chatId, params) : null),
    enabled: !!chatId,
    // Keep gcTime: Infinity to prevent garbage collection (memory benefit)
    // Remove staleTime: Infinity so TanStack Query will background-refetch
    // This gives: instant display from cache + eventual consistency from refetch
    gcTime: Infinity,
  })
}

export function useSendMessage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      chatId,
      content,
      researchDepth,
    }: {
      chatId: string
      content: string
      researchDepth?: ResearchDepth
    }) =>
      messagesApi.send(chatId, {
        content,
        research_depth: researchDepth,
      }),
    onSuccess: (_, { chatId }) => {
      queryClient.invalidateQueries({ queryKey: [...MESSAGES_KEY, chatId] })
      queryClient.invalidateQueries({ queryKey: ['chats', chatId] })
    },
  })
}

export function useEditMessage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      chatId,
      messageId,
      content,
    }: {
      chatId: string
      messageId: string
      content: string
    }) => messagesApi.edit(chatId, messageId, { content }),
    onSuccess: (_, { chatId }) => {
      queryClient.invalidateQueries({ queryKey: [...MESSAGES_KEY, chatId] })
    },
  })
}

export function useRegenerateMessage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ chatId, messageId }: { chatId: string; messageId: string }) =>
      messagesApi.regenerate(chatId, messageId),
    onSuccess: (_, { chatId }) => {
      queryClient.invalidateQueries({ queryKey: [...MESSAGES_KEY, chatId] })
    },
  })
}

export function useSubmitFeedback() {
  return useMutation({
    mutationFn: ({
      chatId,
      messageId,
      rating,
      errorReport,
    }: {
      chatId: string
      messageId: string
      rating: -1 | 1
      errorReport?: string
    }) =>
      messagesApi.submitFeedback(chatId, messageId, {
        rating,
        error_report: errorReport,
      }),
  })
}
