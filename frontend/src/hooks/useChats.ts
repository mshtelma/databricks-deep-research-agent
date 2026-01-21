import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { chatsApi } from '../api/client'
import { clearStreamingState } from '@/stores/chatStreamingState'
import type { ChatStatus } from '../types'

const CHATS_KEY = ['chats']

export function useChats(params?: {
  status?: ChatStatus | 'all'
  search?: string
  limit?: number
  offset?: number
}) {
  return useQuery({
    queryKey: [...CHATS_KEY, params],
    queryFn: () => chatsApi.list(params),
    // Keep gcTime: Infinity to prevent garbage collection (memory benefit)
    // Remove staleTime: Infinity - allow background refetch for consistency
    gcTime: Infinity,
  })
}

export function useChat(chatId: string | undefined) {
  return useQuery({
    queryKey: [...CHATS_KEY, chatId],
    queryFn: () => (chatId ? chatsApi.get(chatId) : null),
    enabled: !!chatId,
    // Keep gcTime: Infinity to prevent garbage collection (memory benefit)
    // Remove staleTime: Infinity - allow background refetch for consistency
    gcTime: Infinity,
  })
}

export function useCreateChat() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data?: { title?: string }) => chatsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CHATS_KEY })
    },
  })
}

export function useUpdateChat() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ chatId, data }: { chatId: string; data: { title?: string; status?: string } }) =>
      chatsApi.update(chatId, data),
    onSuccess: (_, { chatId }) => {
      queryClient.invalidateQueries({ queryKey: CHATS_KEY })
      queryClient.invalidateQueries({ queryKey: [...CHATS_KEY, chatId] })
    },
  })
}

export function useDeleteChat() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (chatId: string) => chatsApi.delete(chatId),
    onSuccess: (_, chatId) => {
      queryClient.invalidateQueries({ queryKey: CHATS_KEY })
      // Remove messages cache for deleted chat (prevents memory leak)
      queryClient.removeQueries({ queryKey: ['messages', chatId] })
      // Clean up streaming state cache
      clearStreamingState(chatId)
    },
  })
}

export function useRestoreChat() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (chatId: string) => chatsApi.restore(chatId),
    onSuccess: (_, chatId) => {
      queryClient.invalidateQueries({ queryKey: CHATS_KEY })
      queryClient.invalidateQueries({ queryKey: [...CHATS_KEY, chatId] })
    },
  })
}

export function useExportChat() {
  return useMutation({
    mutationFn: async ({ chatId, format }: { chatId: string; format: 'markdown' | 'json' }) => {
      const { content, filename } = await chatsApi.export(chatId, format)

      // Trigger download
      const blob = new Blob([content], {
        type: format === 'markdown' ? 'text/markdown' : 'application/json',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      return { filename }
    },
  })
}
