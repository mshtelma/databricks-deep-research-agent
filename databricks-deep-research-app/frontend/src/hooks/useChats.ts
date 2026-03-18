import { useQuery, useMutation, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { chatsApi } from '../api/client'
import { CHAT_FULL_KEY } from './useChatFull'
import { clearStreamingState } from '@/stores/chatStreamingState'
import type { Chat, ChatStatus, PaginatedResponse, UpdateChatRequest } from '../types'

const CHATS_KEY = ['chats']

type ChatsListResponse = PaginatedResponse<Chat>
type ChatsQueryKey = readonly [string, { status?: ChatStatus | 'all'; search?: string; limit?: number; offset?: number }?]

function getChatsQueryEntries(queryClient: QueryClient) {
  return queryClient.getQueriesData<ChatsListResponse>({ queryKey: CHATS_KEY })
}

function matchesSearch(chat: Chat, search?: string): boolean {
  if (!search) return true
  return (chat.title || '').toLowerCase().includes(search.toLowerCase())
}

function matchesStatus(chat: Chat, status?: ChatStatus | 'all'): boolean {
  if (!status || status === 'all') return true
  return chat.status === status
}

function shouldIncludeChat(chat: Chat, params?: { status?: ChatStatus | 'all'; search?: string }) {
  return matchesStatus(chat, params?.status) && matchesSearch(chat, params?.search)
}

function updateChatInList(
  data: ChatsListResponse | undefined,
  updatedChat: Chat,
  params?: { status?: ChatStatus | 'all'; search?: string }
): ChatsListResponse | undefined {
  if (!data) return data

  const existingIndex = data.items.findIndex((chat) => chat.id === updatedChat.id)
  const include = shouldIncludeChat(updatedChat, params)

  if (existingIndex === -1) {
    if (!include) return data
    return {
      ...data,
      items: [updatedChat, ...data.items],
      total: data.total + 1,
    }
  }

  if (!include) {
    return {
      ...data,
      items: data.items.filter((chat) => chat.id !== updatedChat.id),
      total: Math.max(0, data.total - 1),
    }
  }

  const items = [...data.items]
  items[existingIndex] = updatedChat
  return { ...data, items }
}

function removeChatFromList(data: ChatsListResponse | undefined, chatId: string): ChatsListResponse | undefined {
  if (!data) return data
  const exists = data.items.some((chat) => chat.id === chatId)
  if (!exists) return data
  return {
    ...data,
    items: data.items.filter((chat) => chat.id !== chatId),
    total: Math.max(0, data.total - 1),
  }
}

async function snapshotChatsQueries(queryClient: QueryClient) {
  await queryClient.cancelQueries({ queryKey: CHATS_KEY })
  return getChatsQueryEntries(queryClient)
}

function restoreChatsQueries(
  queryClient: QueryClient,
  previous: Array<[readonly unknown[], ChatsListResponse | undefined]> | undefined
) {
  previous?.forEach(([queryKey, data]) => {
    queryClient.setQueryData(queryKey, data)
  })
}

function patchExistingChatAcrossQueries(
  queryClient: QueryClient,
  chatId: string,
  nextChat: Chat | null
) {
  getChatsQueryEntries(queryClient).forEach(([queryKey, data]) => {
    const params = (queryKey as ChatsQueryKey)[1]
    if (!data) return
    if (nextChat === null) {
      queryClient.setQueryData(queryKey, removeChatFromList(data, chatId))
      return
    }
    queryClient.setQueryData(queryKey, updateChatInList(data, nextChat, params))
  })
}

export function useChats(params?: {
  status?: ChatStatus | 'all'
  search?: string
  limit?: number
  offset?: number
}) {
  return useQuery({
    queryKey: [...CHATS_KEY, params],
    queryFn: () => chatsApi.list(params),
    staleTime: 2 * 60 * 1000, // Chat list rarely changes unexpectedly within a session
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
    staleTime: 2 * 60 * 1000,
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
    mutationFn: ({ chatId, data }: { chatId: string; data: UpdateChatRequest }) =>
      chatsApi.update(chatId, data),
    onMutate: async ({ chatId, data }) => {
      const previousChats = await snapshotChatsQueries(queryClient)
      const previousChat = queryClient.getQueryData<Chat | null>([...CHATS_KEY, chatId])

      if (previousChat) {
        const optimisticChat: Chat = {
          ...previousChat,
          title: data.title ?? previousChat.title,
          status: (data.status as ChatStatus | undefined) ?? previousChat.status,
          updatedAt: new Date().toISOString(),
        }
        patchExistingChatAcrossQueries(queryClient, chatId, optimisticChat)
        queryClient.setQueryData<Chat | null>([...CHATS_KEY, chatId], optimisticChat)
      }

      return { previousChats, previousChat, chatId }
    },
    onError: (_error, { chatId }, context) => {
      restoreChatsQueries(queryClient, context?.previousChats)
      if (context?.previousChat !== undefined) {
        queryClient.setQueryData([...CHATS_KEY, chatId], context.previousChat)
      }
    },
    onSuccess: (updatedChat, { chatId }) => {
      patchExistingChatAcrossQueries(queryClient, chatId, updatedChat)
      queryClient.setQueryData([...CHATS_KEY, chatId], updatedChat)
    },
    onSettled: (_data, _error, { chatId }) => {
      queryClient.invalidateQueries({ queryKey: CHATS_KEY })
      queryClient.invalidateQueries({ queryKey: [...CHATS_KEY, chatId] })
    },
  })
}

export function useDeleteChat() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (chatId: string) => chatsApi.delete(chatId),
    onMutate: async (chatId: string) => {
      const previousChats = await snapshotChatsQueries(queryClient)
      const previousChat = queryClient.getQueryData<Chat | null>([...CHATS_KEY, chatId])

      patchExistingChatAcrossQueries(queryClient, chatId, null)

      return { previousChats, previousChat, chatId }
    },
    onError: (_error, chatId, context) => {
      restoreChatsQueries(queryClient, context?.previousChats)
      if (context?.previousChat !== undefined) {
        queryClient.setQueryData([...CHATS_KEY, chatId], context.previousChat)
      }
    },
    onSuccess: (_, chatId) => {
      queryClient.invalidateQueries({ queryKey: CHATS_KEY })
      // Remove messages cache for deleted chat (prevents memory leak)
      queryClient.removeQueries({ queryKey: ['messages', chatId] })
      // Remove chatFull cache for deleted chat
      queryClient.removeQueries({ queryKey: [...CHAT_FULL_KEY, chatId] })
      // Remove chat detail cache
      queryClient.removeQueries({ queryKey: [...CHATS_KEY, chatId] })
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
