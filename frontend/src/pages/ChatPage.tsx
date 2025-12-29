import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ChatSidebar, MessageList, MessageInput } from '@/components/chat';
import { AgentStatusIndicator, PlanProgress } from '@/components/research';
import { useChats, useMessages, useStreamingQuery, useCreateChat } from '@/hooks';
import { formatActivityLabel, getActivityColor } from '@/utils/activityLabels';
import type { Chat, Message } from '@/types';

export default function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Data hooks
  const { data: chatsData, isLoading: isLoadingChats } = useChats();
  const { data: messagesData, isLoading: isLoadingMessages, error: messagesError } = useMessages(chatId);
  const createChatMutation = useCreateChat();

  // Redirect to /chat if chat doesn't exist (404 error)
  useEffect(() => {
    if (messagesError && 'status' in messagesError && messagesError.status === 404) {
      // Chat not found - redirect to new chat
      navigate('/chat', { replace: true });
    }
  }, [messagesError, navigate]);

  const chats: Chat[] = chatsData?.items ?? [];
  const apiMessages = messagesData?.items ?? [];

  // Local state for pending user message (displayed while waiting for API persistence)
  const [pendingUserMessage, setPendingUserMessage] = useState<Message | null>(null);

  // Callback when streaming completes - refresh messages to enable citation rendering
  const handleStreamComplete = useCallback(() => {
    if (chatId) {
      // Invalidate messages query to refetch from DB with real UUIDs
      // This allows AgentMessageWithCitations to fetch claims
      queryClient.invalidateQueries({ queryKey: ['messages', chatId] });
      // Also clear the pending user message as it's now persisted
      setPendingUserMessage(null);
    }
  }, [chatId, queryClient]);

  const {
    streamingContent,
    isStreaming,
    agentStatus,
    currentPlan,
    currentStepIndex,
    sendQuery: originalSendQuery,
    stopStream,
    events,
    completedMessages,
    agentMessageId,
  } = useStreamingQuery(chatId, { onStreamComplete: handleStreamComplete });

  // Build messages list combining API messages, completed in-session messages, and pending
  const messages: Message[] = useMemo(() => {
    // Start with API messages (from database)
    const baseMessages = [...apiMessages];

    // Add completed messages from this session that aren't in API yet
    for (const msg of completedMessages) {
      // Check if this message is already in apiMessages (by content match)
      const exists = baseMessages.some(
        (m) => m.content === msg.content && m.role === msg.role
      );
      if (!exists) {
        // Map 'assistant' to 'agent' for MessageRole compatibility
        const role = msg.role === 'assistant' ? 'agent' : msg.role;
        // For agent messages, use real UUID from backend if available for citation fetching
        // For user messages, use placeholder ID (they don't need citation support)
        const messageId = msg.role === 'assistant' && agentMessageId
          ? agentMessageId
          : `session-${Date.now()}-${baseMessages.length}`;
        baseMessages.push({
          id: messageId,
          chat_id: chatId || '',
          role: role as 'user' | 'agent',
          content: msg.content,
          created_at: new Date().toISOString(),
          is_edited: false,
        });
      }
    }

    // Add pending user message if exists
    if (pendingUserMessage) {
      // Check it's not already in the list
      const exists = baseMessages.some(
        (m) => m.content === pendingUserMessage.content && m.role === 'user'
      );
      if (!exists) {
        baseMessages.push(pendingUserMessage);
      }
    }

    return baseMessages;
  }, [apiMessages, completedMessages, pendingUserMessage, chatId, agentMessageId]);

  // Wrapped sendQuery that also sets the pending user message
  const sendQuery = useCallback(
    (query: string) => {
      // Create a pending user message
      setPendingUserMessage({
        id: `pending-${Date.now()}`,
        chat_id: chatId || '',
        role: 'user',
        content: query,
        created_at: new Date().toISOString(),
        is_edited: false,
      });

      // The hook now automatically tracks conversation history
      // and the backend loads history from DB
      originalSendQuery(query);
    },
    [chatId, originalSendQuery]
  );

  // Pending query ref - used to send query after navigation
  const pendingQueryRef = useRef<string | null>(null);

  // Handle pending query after chat creation and navigation
  useEffect(() => {
    if (chatId && pendingQueryRef.current) {
      const query = pendingQueryRef.current;
      pendingQueryRef.current = null;
      sendQuery(query);
    }
  }, [chatId, sendQuery]);

  // Clear pending user message when chat changes
  useEffect(() => {
    setPendingUserMessage(null);
  }, [chatId]);

  // Navigate to new chat (without creating - chat is created on first message submit)
  const handleNewChat = useCallback(() => {
    // Just navigate to /chat - chat will be created when user sends first message
    navigate('/chat', { replace: true });
  }, [navigate]);

  // Auto-select first chat when navigating to /chat without chatId
  // NOTE: We do NOT auto-create empty chats - lazy chat creation happens in handleSendMessage
  useEffect(() => {
    // Skip if we already have a chatId or still loading
    if (chatId || isLoadingChats) return;

    const firstChat = chats[0];
    if (firstChat) {
      // Auto-select the most recent chat
      navigate(`/chat/${firstChat.id}`, { replace: true });
    }
    // If no chats exist, stay on /chat - user will see empty input
    // Chat is created when user sends first message (lazy creation)
  }, [chatId, chats, isLoadingChats, navigate]);

  // Send message (creates chat on-demand if needed)
  const handleSendMessage = async (content: string) => {
    if (!chatId) {
      // Lazy chat creation - create with title from first message
      const title = content.length > 50 ? content.slice(0, 47) + '...' : content;
      const newChat = await createChatMutation.mutateAsync({ title });
      if (newChat) {
        // Store query and navigate - the effect will send after navigation
        pendingQueryRef.current = content;
        navigate(`/chat/${newChat.id}`);
      }
    } else {
      sendQuery(content);
    }
  };

  // Select chat
  const handleSelectChat = (id: string) => {
    navigate(`/chat/${id}`);
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <ChatSidebar
        chats={chats}
        currentChatId={chatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        isLoading={isLoadingChats}
      />

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header with status */}
        <header className="flex items-center justify-between px-4 py-2 border-b">
          <h1 className="font-semibold truncate">
            {chatId
              ? chats.find((c) => c.id === chatId)?.title || 'New chat'
              : 'Deep Research Agent'}
          </h1>
          <AgentStatusIndicator status={agentStatus} />
        </header>

        {/* Messages area */}
        <div className="flex-1 flex min-h-0">
          {/* Messages */}
          <div className="flex-1 flex flex-col min-w-0">
            <MessageList
              messages={messages}
              streamingContent={streamingContent}
              isStreaming={isStreaming}
              isLoading={isStreaming}
              className="flex-1"
            />

            {/* Input */}
            <MessageInput
              onSubmit={handleSendMessage}
              onStop={stopStream}
              isLoading={isStreaming}
              disabled={isLoadingMessages}
            />
          </div>

          {/* Research progress panel (when active) */}
          {(isStreaming || currentPlan) && (
            <aside className="w-72 border-l p-4 overflow-y-auto bg-muted/20">
              <PlanProgress
                plan={currentPlan}
                currentStepIndex={currentStepIndex}
              />

              {/* Recent events log */}
              {events.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-xs font-medium text-muted-foreground mb-2">
                    Research Activity
                  </h4>
                  <div className="space-y-1 text-xs max-h-40 overflow-y-auto">
                    {events.slice(-10).map((event, i) => (
                      <div
                        key={`${event.event_type}-${events.length - 10 + i}`}
                        className={`truncate ${getActivityColor(event)}`}
                      >
                        {formatActivityLabel(event)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
