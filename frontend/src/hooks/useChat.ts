import * as React from 'react';
import type { Message } from '../types';
import { useMessages, useEditMessage, useRegenerateMessage } from './useMessages';
import { useChat as useChatQuery } from './useChats';

/**
 * Hook for managing chat state with message invalidation handling.
 * Handles the UI state changes when messages are edited or regenerated.
 */
export function useChatWithInvalidation(chatId: string | undefined) {
  const { data: chat, isLoading: isChatLoading } = useChatQuery(chatId);
  const { data: messagesData, isLoading: isMessagesLoading } = useMessages(chatId);
  const editMessageMutation = useEditMessage();
  const regenerateMessageMutation = useRegenerateMessage();

  // Track messages that are being invalidated (waiting for new response)
  const [invalidatingAfterMessageId, setInvalidatingAfterMessageId] = React.useState<string | null>(null);

  // Get messages, filtering out invalidated ones
  const messages = React.useMemo(() => {
    if (!messagesData?.items) return [];

    // If we're invalidating, filter out messages after the edited one
    if (invalidatingAfterMessageId) {
      const index = messagesData.items.findIndex((m: Message) => m.id === invalidatingAfterMessageId);
      if (index >= 0) {
        return messagesData.items.slice(0, index + 1);
      }
    }

    return messagesData.items;
  }, [messagesData?.items, invalidatingAfterMessageId]);

  // Edit a message and invalidate subsequent messages
  const editMessage = React.useCallback(
    async (messageId: string, newContent: string) => {
      if (!chatId) return;

      // Mark this message as the point after which messages are invalidated
      setInvalidatingAfterMessageId(messageId);

      try {
        await editMessageMutation.mutateAsync({
          chatId,
          messageId,
          content: newContent,
        });
      } finally {
        // Clear invalidation state - new messages will be fetched
        setInvalidatingAfterMessageId(null);
      }
    },
    [chatId, editMessageMutation]
  );

  // Regenerate a response (will create new research session)
  const regenerateResponse = React.useCallback(
    async (messageId: string) => {
      if (!chatId) return;

      // Find the message and mark subsequent ones as invalidated
      const messages = messagesData?.items || [];
      const index = messages.findIndex((m: Message) => m.id === messageId);
      if (index > 0) {
        // Set the previous message as the cutoff point
        const previousMessage = messages[index - 1];
        if (previousMessage) {
          setInvalidatingAfterMessageId(previousMessage.id);
        }
      }

      try {
        await regenerateMessageMutation.mutateAsync({
          chatId,
          messageId,
        });
      } finally {
        setInvalidatingAfterMessageId(null);
      }
    },
    [chatId, messagesData?.items, regenerateMessageMutation]
  );

  // Check if a specific message is being processed
  const isMessageInvalidating = React.useCallback(
    (messageId: string) => {
      if (!invalidatingAfterMessageId) return false;
      const messages = messagesData?.items || [];
      const invalidatingIndex = messages.findIndex((m: Message) => m.id === invalidatingAfterMessageId);
      const messageIndex = messages.findIndex((m: Message) => m.id === messageId);
      return messageIndex > invalidatingIndex;
    },
    [invalidatingAfterMessageId, messagesData?.items]
  );

  return {
    chat,
    messages,
    isLoading: isChatLoading || isMessagesLoading,
    isEditingMessage: editMessageMutation.isPending,
    isRegenerating: regenerateMessageMutation.isPending,
    isInvalidating: !!invalidatingAfterMessageId,
    editMessage,
    regenerateResponse,
    isMessageInvalidating,
  };
}
