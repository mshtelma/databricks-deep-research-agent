import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { ChatSidebar, MessageList, MessageInput } from '@/components/chat';
import { AgentStatusIndicator, PlanProgress } from '@/components/research';
import { useChats, useMessages, useStreamingQuery, useCreateChat } from '@/hooks';
import { formatActivityLabel, getActivityColor } from '@/utils/activityLabels';
import type { Chat, Message } from '@/types';

export default function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();

  // Data hooks
  const { data: chatsData, isLoading: isLoadingChats } = useChats();
  const { data: messagesData, isLoading: isLoadingMessages } = useMessages(chatId);
  const createChatMutation = useCreateChat();

  const chats: Chat[] = chatsData?.items ?? [];
  const apiMessages = messagesData?.items ?? [];

  // Local state for pending user message (displayed while waiting for API persistence)
  const [pendingUserMessage, setPendingUserMessage] = useState<Message | null>(null);

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
  } = useStreamingQuery(chatId);

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
        baseMessages.push({
          id: `session-${Date.now()}-${baseMessages.length}`,
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
  }, [apiMessages, completedMessages, pendingUserMessage, chatId]);

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

  // Create new chat
  const handleNewChat = useCallback(async () => {
    const newChat = await createChatMutation.mutateAsync({});
    if (newChat) {
      navigate(`/chat/${newChat.id}`);
    }
  }, [createChatMutation, navigate]);

  // Auto-select first chat or create new one when navigating to /chat without chatId
  useEffect(() => {
    // Skip if we already have a chatId or still loading
    if (chatId || isLoadingChats) return;

    const firstChat = chats[0];
    if (firstChat) {
      // Auto-select the most recent chat
      navigate(`/chat/${firstChat.id}`, { replace: true });
    } else {
      // Auto-create a new chat for new users
      handleNewChat();
    }
  }, [chatId, chats, isLoadingChats, navigate, handleNewChat]);

  // Send message
  const handleSendMessage = async (content: string) => {
    // If no chat, create one first
    if (!chatId) {
      const newChat = await createChatMutation.mutateAsync({});
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
