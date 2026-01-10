import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState, useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ChatSidebar, MessageList, MessageInput, DeleteChatDialog, ExportChatDialog, type ExportFormat } from '@/components/chat';
import { AgentStatusIndicator, ResearchPanel } from '@/components/research';
import { useChats, useMessages, useStreamingQuery, useChatActions, useDraftChats, useCitations } from '@/hooks';
import { useResearchReconnection } from '@/hooks/useResearchReconnection';
import type { Chat, Message, PersistenceCompletedEvent, QueryMode } from '@/types';

export default function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();
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

  // Note: currentQueryMode is now managed in useStreamingQuery hook (not local state)
  // This ensures it persists correctly throughout the streaming session

  // Callback when streaming completes - refresh messages to enable citation rendering
  const handleStreamComplete = useCallback(() => {
    if (chatId) {
      // Force immediate refetch from DB with real UUIDs
      // This allows AgentMessageWithCitations to fetch claims
      // Using refetchQueries instead of invalidateQueries for more aggressive refresh
      queryClient.refetchQueries({ queryKey: ['messages', chatId] });
      // Note: pendingUserMessage is cleared reactively when API confirms the message
      // (see the useEffect that watches apiMessages, not here)
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
    processExternalEvent,
    setIsStreaming,
    setStreamingContent,
    startTime,
    currentAgent,
    currentQueryMode,
    setCurrentQueryMode,
  } = useStreamingQuery(chatId, { onStreamComplete: handleStreamComplete });

  // Reconnection hook for crash resilience
  const {
    isReconnecting,
    reconnectionError,
    checkAndReconnect,
  } = useResearchReconnection({
    chatId: chatId ?? null,
    onEvent: processExternalEvent,
    onComplete: (finalReport) => {
      // Set the final content and mark as complete
      setStreamingContent(finalReport);
      setIsStreaming(false);
      // Refresh messages to get the fully persisted state
      queryClient.invalidateQueries({ queryKey: ['messages', chatId] });
    },
    onError: (error) => {
      console.error('[Reconnection] Error:', error);
      setIsStreaming(false);
    },
    // Only enable reconnection when not already streaming
    enabled: !isStreaming,
  });

  // Check for active research on page load (reconnection)
  useEffect(() => {
    if (!chatId || isStreaming || isDraftChat(chatId)) return;

    // Check if there's an active research session to reconnect to
    checkAndReconnect().then((result) => {
      if (result.reconnected) {
        console.log('[ChatPage] Reconnected to active research session:', result);

        // Restore query mode for panel visibility (CRITICAL!)
        if (result.queryMode) {
          setCurrentQueryMode(result.queryMode as QueryMode);
        }

        // Enable streaming UI to show live updates
        setIsStreaming(true);
      }
    });
  }, [chatId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Get the latest agent message ID for citation fetching
  const latestAgentMessageId = useMemo(() => {
    // During streaming, use the backend-provided agentMessageId
    if (agentMessageId) return agentMessageId;
    // After completion, find the most recent agent message
    const agentMessages = apiMessages.filter(m => m.role === 'agent');
    return agentMessages[agentMessages.length - 1]?.id ?? null;
  }, [agentMessageId, apiMessages]);

  // Fetch claims for the latest agent message (for Research Panel)
  const {
    claims,
    verificationSummary,
  } = useCitations(latestAgentMessageId);

  // Extract all sources from the latest research session
  // Note: Source URLs are only available after persistence (not in streaming events)
  const allSources = useMemo(() => {
    // Get from the most recent research_session (after persistence)
    const latestAgentMessage = apiMessages
      .slice()
      .reverse()
      .find(m => m.role === 'agent' && m.research_session);

    if (latestAgentMessage?.research_session?.sources) {
      return latestAgentMessage.research_session.sources.map((s) => ({
        url: s.url,
        title: s.title,
        snippet: s.snippet,
        is_cited: false,  // Default - will be enriched by citation data
        step_index: undefined as number | undefined,
        crawl_status: undefined as 'success' | 'failed' | 'timeout' | 'blocked' | undefined,
      }));
    }

    // During streaming, source URLs are not available in events
    // Return empty array - ResearchPanel will show placeholder
    return [];
  }, [apiMessages]);

  // ===== DEBUG LOGGING START =====
  // Track component mount/unmount
  useEffect(() => {
    const mountId = Math.random().toString(36).slice(2, 8);
    console.log('[ChatPage] MOUNTED - mountId:', mountId, 'chatId:', chatId);
    return () => {
      console.log('[ChatPage] UNMOUNTING - mountId:', mountId, 'chatId:', chatId);
    };
  }, []);

  // Track all panel visibility conditions
  useEffect(() => {
    const isDeepOrWeb = currentQueryMode === 'deep_research' || currentQueryMode === 'web_search';
    const hasActivity = isStreaming || currentPlan !== null || events.length > 0;
    const shouldShow = isDeepOrWeb && hasActivity;

    console.log('[ChatPage] Panel visibility:', {
      currentQueryMode,
      isDeepOrWeb,
      isStreaming,
      currentPlan: currentPlan ? `Plan with ${currentPlan.steps?.length} steps` : null,
      eventsCount: events.length,
      hasActivity,
      shouldShow,
      time: new Date().toISOString(),
    });

    if (!shouldShow && (isStreaming || currentPlan || events.length > 0)) {
      console.error('[ChatPage] WARNING: Panel hidden but has activity!', {
        reason: !isDeepOrWeb ? `currentQueryMode is "${currentQueryMode}"` : 'hasActivity is false',
      });
    }
  }, [currentQueryMode, isStreaming, currentPlan, events.length]);

  // Track currentQueryMode specifically
  useEffect(() => {
    console.log('[ChatPage] currentQueryMode changed:', currentQueryMode);
    if (currentQueryMode === null) {
      console.warn('[ChatPage] currentQueryMode is NULL - panel will not show!');
    }
  }, [currentQueryMode]);
  // ===== DEBUG LOGGING END =====

  // Redirect to /chat if chat doesn't exist (404 error)
  // Skip for draft chats - they don't exist in DB yet
  // Skip if streaming - chat may not be persisted yet during active research
  useEffect(() => {
    if (chatId && isDraftChat(chatId)) return; // Draft chats won't be in DB
    if (isStreaming) return; // Don't redirect during streaming - chat persistence may be pending
    if (messagesError && 'status' in messagesError && messagesError.status === 404) {
      // Chat not found - redirect to new chat
      console.log('[ChatPage] 404 redirect - chat not found:', chatId);
      navigate('/chat', { replace: true });
    }
  }, [messagesError, navigate, chatId, isDraftChat, isStreaming]);

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

    // Add pending user message if exists and belongs to current chat
    // The chat_id check prevents showing stale pending messages in wrong chat
    if (pendingUserMessage && pendingUserMessage.chat_id === chatId) {
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
    (query: string, queryMode?: QueryMode, researchDepth?: string, verifySources?: boolean) => {
      // Track query for retry functionality
      setLastQuery(query);

      // Note: queryMode is now tracked in useStreamingQuery hook

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
      originalSendQuery(query, queryMode, researchDepth, verifySources);
    },
    [chatId, originalSendQuery]
  );

  // Handle pending query from router state after chat creation and navigation
  useEffect(() => {
    if (chatId && location.state?.pendingQuery) {
      const query = location.state.pendingQuery;
      const queryMode = location.state.queryMode as QueryMode | undefined;
      const researchDepth = location.state.researchDepth as string | undefined;
      const verifySources = location.state.verifySources as boolean | undefined;
      // Clear state immediately to prevent re-sending on refresh
      window.history.replaceState({}, document.title);
      sendQuery(query, queryMode, researchDepth, verifySources);
    }
  }, [chatId, location.state?.pendingQuery, sendQuery]);

  // Clear pending user message when it appears in API messages
  // This prevents the race condition where it's cleared before API returns
  // (Previously this effect eagerly cleared on chatId change, but that ran
  // AFTER the effect that sets pendingUserMessage, causing it to vanish)
  useEffect(() => {
    if (pendingUserMessage && apiMessages.length > 0) {
      const exists = apiMessages.some(
        (m) => m.content === pendingUserMessage.content && m.role === 'user'
      );
      if (exists) {
        console.log('[ChatPage] pendingUserMessage cleared by API confirmation');
        setPendingUserMessage(null);
      }
    }
  }, [apiMessages, pendingUserMessage]);

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
  const handleSendMessage = async (content: string, queryMode?: QueryMode, researchDepth?: string, verifySources?: boolean) => {
    if (!chatId) {
      // No chat selected - create a draft and navigate
      const draft = createDraft();
      navigate(`/chat/${draft.id}?draft=1`, { state: { pendingQuery: content, queryMode, researchDepth, verifySources } });
    } else {
      // Chat exists (draft or real) - just send the query
      // Backend handles persistence for drafts via deferred materialization
      sendQuery(content, queryMode, researchDepth, verifySources);
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

        {/* Reconnection indicator */}
        {isReconnecting && (
          <div className="bg-blue-500/10 border-b border-blue-500/20 px-4 py-2">
            <span className="text-sm text-blue-600 animate-pulse">
              Reconnecting to research session...
            </span>
          </div>
        )}

        {/* Reconnection error */}
        {reconnectionError && (
          <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-2">
            <span className="text-sm text-destructive">
              Failed to reconnect: {reconnectionError}
            </span>
          </div>
        )}

        {/* Messages area */}
        <div className="flex-1 flex min-h-0">
          {/* Messages */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Compute whether to show research panel (for hiding duplicate sources) */}
            {(() => {
              const showResearchPanel = (currentQueryMode === 'deep_research' || currentQueryMode === 'web_search') &&
                (isStreaming || !!currentPlan || events.length > 0 || claims.length > 0);

              return (
                <MessageList
                  messages={messages}
                  streamingContent={streamingContent}
                  isStreaming={isStreaming}
                  isLoading={isStreaming}
                  className="flex-1"
                  hideAgentSourcesSection={showResearchPanel}
                  researchPanel={
                    showResearchPanel ? (
                      <ResearchPanel
                        isStreaming={isStreaming}
                        events={events}
                        plan={currentPlan}
                        currentStepIndex={currentStepIndex}
                        startTime={startTime ?? undefined}
                        currentAgent={currentAgent ?? undefined}
                        claims={claims}
                        allSources={allSources}
                        verificationSummary={verificationSummary}
                      />
                    ) : null
                  }
                />
              );
            })()}

            {/* Input */}
            <MessageInput
              onSubmit={handleSendMessage}
              onStop={stopStream}
              isLoading={isStreaming}
              disabled={isLoadingMessages}
            />
          </div>

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
