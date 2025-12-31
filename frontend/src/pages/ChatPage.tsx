import { useParams, useNavigate, useLocation, useSearchParams } from 'react-router-dom';
import { useEffect, useState, useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ChatSidebar, MessageList, MessageInput, DeleteChatDialog, ExportChatDialog, type ExportFormat } from '@/components/chat';
import { AgentStatusIndicator, PlanProgress } from '@/components/research';
import { useChats, useMessages, useStreamingQuery, useChatActions, useDraftChats } from '@/hooks';
import { formatActivityLabel, getActivityColor } from '@/utils/activityLabels';
import type { Chat, Message, PersistenceCompletedEvent } from '@/types';

export default function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [_searchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // Draft chat management
  const { createDraft, removeDraft, isDraft: isDraftChat, getDraftList, clearStaleDrafts } = useDraftChats();

  // Data hooks
  const { data: chatsData, isLoading: isLoadingChats } = useChats();
  // Skip messages fetch for drafts (they don't exist in DB)
  const shouldFetchMessages = chatId && !isDraftChat(chatId);
  const { data: messagesData, isLoading: isLoadingMessages, error: messagesError } = useMessages(
    shouldFetchMessages ? chatId : undefined
  );

  // Chat actions hook (rename, archive, delete, restore, export)
  const handleNavigateAway = useCallback(() => {
    navigate('/chat', { replace: true });
  }, [navigate]);

  const chatActions = useChatActions({
    currentChatId: chatId,
    onNavigateAway: handleNavigateAway,
  });

  // Dialog state for delete confirmation and export format selection
  const [deleteDialog, setDeleteDialog] = useState<{ isOpen: boolean; chatId: string | null; title: string | null }>({
    isOpen: false, chatId: null, title: null
  });
  const [exportDialog, setExportDialog] = useState<{ isOpen: boolean; chatId: string | null; title: string | null }>({
    isOpen: false, chatId: null, title: null
  });

  // Sidebar filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'active' | 'archived' | 'all'>('active');

  // Redirect to /chat if chat doesn't exist (404 error)
  // Skip for draft chats - they don't exist in DB yet
  useEffect(() => {
    if (chatId && isDraftChat(chatId)) return; // Draft chats won't be in DB
    if (messagesError && 'status' in messagesError && messagesError.status === 404) {
      // Chat not found - redirect to new chat
      navigate('/chat', { replace: true });
    }
  }, [messagesError, navigate, chatId, isDraftChat]);

  // Merge draft chats with API chats (drafts appear at top)
  // Filter out drafts that already exist in API (handles race condition during persistence)
  const chats: Chat[] = useMemo(() => {
    const apiChats = chatsData?.items ?? [];
    const draftChats = getDraftList();
    // Deduplicate: filter out drafts that have been persisted to API
    const apiChatIds = new Set(apiChats.map(c => c.id));
    const uniqueDrafts = draftChats.filter(d => !apiChatIds.has(d.id));
    // Cast DraftChat to Chat (they're compatible for display)
    return [...uniqueDrafts as unknown as Chat[], ...apiChats];
  }, [chatsData, getDraftList]);

  // Sync localStorage drafts with API state on load
  // Removes stale drafts (older than 60s) that don't exist in the database
  // This prevents "phantom" chats from appearing after DB is cleaned
  useEffect(() => {
    if (chatsData?.items && !isLoadingChats) {
      const apiChatIds = new Set(chatsData.items.map(c => c.id));
      clearStaleDrafts(apiChatIds);
    }
  }, [chatsData, isLoadingChats, clearStaleDrafts]);

  const apiMessages = messagesData?.items ?? [];

  // Local state for pending user message (displayed while waiting for API persistence)
  const [pendingUserMessage, setPendingUserMessage] = useState<Message | null>(null);

  // Track last query for retry functionality
  const [lastQuery, setLastQuery] = useState<string>('');

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
    persistenceResult,
    persistenceFailed,
    hydrateFromSession,
  } = useStreamingQuery(chatId, { onStreamComplete: handleStreamComplete });

  // Handle persistence completion - convert draft to real chat
  const handlePersistenceComplete = useCallback((event: PersistenceCompletedEvent) => {
    if (event.was_draft && chatId) {
      // Remove from local draft storage
      removeDraft(chatId);
      // Navigate to real URL (remove ?draft=1)
      navigate(`/chat/${chatId}`, { replace: true });
      // Invalidate chats list to fetch the new chat from API
      queryClient.invalidateQueries({ queryKey: ['chats'] });
    }
  }, [chatId, removeDraft, navigate, queryClient]);

  // Effect for persistence result
  useEffect(() => {
    if (persistenceResult) {
      handlePersistenceComplete(persistenceResult);
    }
  }, [persistenceResult, handlePersistenceComplete]);

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
      // Track query for retry functionality
      setLastQuery(query);

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

  // Handle pending query from router state after chat creation and navigation
  useEffect(() => {
    if (chatId && location.state?.pendingQuery) {
      const query = location.state.pendingQuery;
      // Clear state immediately to prevent re-sending on refresh
      window.history.replaceState({}, document.title);
      sendQuery(query);
    }
  }, [chatId, location.state?.pendingQuery, sendQuery]);

  // Clear pending user message when chat changes
  useEffect(() => {
    setPendingUserMessage(null);
  }, [chatId]);

  // Hydrate research panel from persisted session on page reload
  // This restores the research panel state from the database
  useEffect(() => {
    // Skip if streaming (live state takes precedence)
    if (isStreaming) return;

    // Skip if we already have a plan (already hydrated or active session)
    if (currentPlan) return;

    // Find the most recent agent message with research_session
    const agentMessageWithSession = apiMessages
      .slice()
      .reverse()
      .find(m => m.role === 'agent' && m.research_session);

    if (agentMessageWithSession?.research_session) {
      hydrateFromSession(agentMessageWithSession.research_session);
    }
  }, [apiMessages, isStreaming, currentPlan, hydrateFromSession]);

  // Warn user before leaving a draft chat with unsent content
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (chatId && isDraftChat(chatId) && (isStreaming || pendingUserMessage)) {
        e.preventDefault();
        e.returnValue = 'You have an unsaved draft. Are you sure you want to leave?';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [chatId, isDraftChat, isStreaming, pendingUserMessage]);

  // Create new draft chat - instant, no API call
  // Chat will be persisted when first message is successfully processed
  const handleNewChat = useCallback(() => {
    const draft = createDraft();
    navigate(`/chat/${draft.id}?draft=1`);
  }, [createDraft, navigate]);

  // Auto-select first chat when navigating to /chat without chatId
  // NOTE: We do NOT auto-create empty chats - lazy chat creation happens in handleSendMessage
  useEffect(() => {
    // Skip if we already have a chatId or still loading
    if (chatId || isLoadingChats) return;

    // Skip auto-select if user explicitly requested new chat mode
    // This allows "New Chat" button to show empty input without redirecting
    if (location.state?.newChat) return;

    const firstChat = chats[0];
    if (firstChat) {
      // Auto-select the most recent chat
      navigate(`/chat/${firstChat.id}`, { replace: true });
    }
    // If no chats exist, stay on /chat - user will see empty input
    // Chat is created when user sends first message (lazy creation)
  }, [chatId, chats, isLoadingChats, navigate, location.state?.newChat]);

  // Send message - for draft chats, backend will persist chat on success
  const handleSendMessage = async (content: string) => {
    if (!chatId) {
      // No chat selected - create a draft and navigate
      const draft = createDraft();
      navigate(`/chat/${draft.id}?draft=1`, { state: { pendingQuery: content } });
    } else {
      // Chat exists (draft or real) - just send the query
      // Backend handles persistence for drafts via deferred materialization
      sendQuery(content);
    }
  };

  // Select chat
  const handleSelectChat = (id: string) => {
    navigate(`/chat/${id}`);
  };

  // Chat action handlers
  const handleRenameChat = useCallback((targetChatId: string, newTitle: string) => {
    chatActions.renameChat(targetChatId, newTitle);
  }, [chatActions]);

  const handleArchiveChat = useCallback((targetChatId: string) => {
    chatActions.archiveChat(targetChatId);
  }, [chatActions]);

  const handleRestoreChat = useCallback((targetChatId: string) => {
    chatActions.restoreChat(targetChatId);
  }, [chatActions]);

  // Dialog openers (show confirmation/selection UI before executing action)
  const handleDeleteClick = useCallback((targetChatId: string) => {
    const chat = chats.find(c => c.id === targetChatId);
    setDeleteDialog({ isOpen: true, chatId: targetChatId, title: chat?.title || null });
  }, [chats]);

  const handleExportClick = useCallback((targetChatId: string) => {
    const chat = chats.find(c => c.id === targetChatId);
    setExportDialog({ isOpen: true, chatId: targetChatId, title: chat?.title || null });
  }, [chats]);

  // Dialog confirmations (execute the actual action)
  const handleConfirmDelete = useCallback(() => {
    if (deleteDialog.chatId) {
      chatActions.deleteChat(deleteDialog.chatId);
    }
    setDeleteDialog({ isOpen: false, chatId: null, title: null });
  }, [deleteDialog.chatId, chatActions]);

  const handleConfirmExport = useCallback((format: ExportFormat, _includeMetadata: boolean) => {
    // Note: includeMetadata is not currently supported by the backend API
    if (exportDialog.chatId) {
      chatActions.exportChat(exportDialog.chatId, format);
    }
    setExportDialog({ isOpen: false, chatId: null, title: null });
  }, [exportDialog.chatId, chatActions]);

  const handleCloseDeleteDialog = useCallback(() => {
    setDeleteDialog({ isOpen: false, chatId: null, title: null });
  }, []);

  const handleCloseExportDialog = useCallback(() => {
    setExportDialog({ isOpen: false, chatId: null, title: null });
  }, []);

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <ChatSidebar
        chats={chats}
        currentChatId={chatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onRenameChat={handleRenameChat}
        onArchiveChat={handleArchiveChat}
        onRestoreChat={handleRestoreChat}
        onDeleteChat={handleDeleteClick}
        onExportChat={handleExportClick}
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        statusFilter={statusFilter}
        onStatusFilterChange={setStatusFilter}
        isLoading={isLoadingChats}
      />

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header with status */}
        <header className="flex items-center justify-between px-4 py-2 border-b">
          <h1 className="font-semibold truncate">
            {chatId
              ? chats.find((c) => c.id === chatId)?.title || (isDraftChat(chatId || '') ? 'New chat...' : 'New chat')
              : 'Deep Research Agent'}
          </h1>
          <AgentStatusIndicator status={agentStatus} />
        </header>

        {/* Persistence failure banner for draft chats */}
        {persistenceFailed && chatId && isDraftChat(chatId) && (
          <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-2 flex items-center justify-between">
            <span className="text-sm text-destructive">
              Failed to save your research. Your content is preserved.
            </span>
            <button
              onClick={() => lastQuery && sendQuery(lastQuery)}
              className="text-sm text-destructive underline hover:no-underline"
            >
              Retry
            </button>
          </div>
        )}

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

      {/* Delete confirmation dialog */}
      <DeleteChatDialog
        isOpen={deleteDialog.isOpen}
        chatTitle={deleteDialog.title || 'this chat'}
        onClose={handleCloseDeleteDialog}
        onConfirm={handleConfirmDelete}
        isDeleting={chatActions.isDeleting}
      />

      {/* Export format selection dialog */}
      <ExportChatDialog
        isOpen={exportDialog.isOpen}
        chatTitle={exportDialog.title}
        onClose={handleCloseExportDialog}
        onExport={handleConfirmExport}
        isExporting={chatActions.isExporting}
      />
    </div>
  );
}
