import { useCallback } from 'react';
import { useUpdateChat, useDeleteChat, useExportChat } from './useChats';

interface UseChatActionsOptions {
  currentChatId?: string;
  onNavigateAway?: () => void;
}

/**
 * Custom hook that encapsulates all chat management mutations.
 * Provides clean, reusable actions for rename, archive, delete, restore, and export.
 */
export function useChatActions({ currentChatId, onNavigateAway }: UseChatActionsOptions = {}) {
  const updateMutation = useUpdateChat();
  const deleteMutation = useDeleteChat();
  const exportMutation = useExportChat();

  const renameChat = useCallback((chatId: string, newTitle: string) => {
    updateMutation.mutate({ chatId, data: { title: newTitle } });
  }, [updateMutation]);

  const archiveChat = useCallback((chatId: string) => {
    updateMutation.mutate({ chatId, data: { status: 'archived' } });
    // Navigate away if archiving current chat
    if (chatId === currentChatId) {
      onNavigateAway?.();
    }
  }, [updateMutation, currentChatId, onNavigateAway]);

  const deleteChat = useCallback((chatId: string) => {
    deleteMutation.mutate(chatId);
    // Navigate away if deleting current chat
    if (chatId === currentChatId) {
      onNavigateAway?.();
    }
  }, [deleteMutation, currentChatId, onNavigateAway]);

  const restoreChat = useCallback((chatId: string) => {
    // Use update mutation to change status back to active (unarchive)
    // The restore endpoint is for undeleting soft-deleted chats, not for unarchiving
    updateMutation.mutate({ chatId, data: { status: 'active' } });
  }, [updateMutation]);

  const exportChat = useCallback((chatId: string, format: 'markdown' | 'json') => {
    exportMutation.mutate({ chatId, format });
  }, [exportMutation]);

  return {
    renameChat,
    archiveChat,
    deleteChat,
    restoreChat,
    exportChat,
    isDeleting: deleteMutation.isPending,
    isExporting: exportMutation.isPending,
    isUpdating: updateMutation.isPending,
  };
}
