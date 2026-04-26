import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ChatSidebar, MessageList, MessageInput, DeleteChatDialog, ExportChatDialog, type ExportFormat } from '@/components/chat';
import { AgentStatusIndicator, ResearchPanel } from '@/components/research';
import { useChats, useStreamingQuery, useChatActions, useDraftChats, usePrefetchMessages, useChatFull, CHAT_FULL_KEY } from '@/hooks';
import { useChatActiveJob } from '@/hooks/useResearchJobs';
import { ComponentRegistry } from '@/core/plugins';
import { PlanReviewModal } from '@/components/research/PlanReviewModal';
import { usePlanReview } from '@/hooks/usePlanReview';
import { useDiscoveredSources } from '@/hooks/useDiscoveredSources';
import type { Chat, Message, MessageRole, PersistenceCompletedEvent } from '@/types';
import type { Claim } from '@/types/citation';
import type { QuerySubmission } from '@/types/querySubmission';
import type { AvailableSource } from '@/types/dataSources';

export default function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  // Draft chat management
  const { createDraft, removeDraft, isDraft: isDraftChat, getDraftList, clearStaleDrafts } = useDraftChats();
  // Pre-allocate a draft session ID so composer/upload works even before chat selection.
  const preallocatedDraftIdRef = useRef<string>(crypto.randomUUID());

  // Prefetch messages on hover for instant chat switching
  const { prefetchMessages } = usePrefetchMessages();

  // Data hooks
  const { data: chatsData, isLoading: isLoadingChats } = useChats();
  // Skip messages fetch for drafts (they don't exist in DB)
  const shouldFetchMessages = chatId && !isDraftChat(chatId);
  const { data: chatFullData, error: chatFullError } = useChatFull(
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

  // Get plugin-provided input configuration (controls mode selector visibility, etc.)
  const inputConfig = useMemo(() => ComponentRegistry.getInputConfig(), []);

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

  // Derive Message[] from chatFullData (FullMessage is a superset of Message)
  const apiMessages: Message[] = useMemo(() => {
    if (!chatFullData?.messages) return [];
    return chatFullData.messages.map((m) => ({
      id: m.id,
      chatId: m.chatId,
      role: m.role as MessageRole,
      content: m.content,
      createdAt: m.createdAt,
      isEdited: m.isEdited,
      researchSession: m.researchSession,
    }));
  }, [chatFullData?.messages]);

  // Local state for pending user message (displayed while waiting for API persistence)
  const [pendingUserMessage, setPendingUserMessage] = useState<Message | null>(null);

  // Track last query for retry functionality
  const [lastQuery, setLastQuery] = useState<string>('');

  // Track whether we've triggered refetch for current session
  // This prevents duplicate refetches when both persistence_completed and agentStatus='complete' fire
  const hasTriggeredCompletionRefetchRef = useRef(false);

  // Note: currentQueryMode is now managed in useStreamingQuery hook (not local state)
  // This ensures it persists correctly throughout the streaming session

  // Callback when streaming completes - DON'T refetch messages immediately
  // The streaming view with streamingClaims will continue to show colored citations
  // until persistence_completed triggers the refetch (prevents grey citations)
  const handleStreamComplete = useCallback(() => {
    // Note: pendingUserMessage is cleared reactively when API confirms the message
    // (see the useEffect that watches apiMessages, not here)
  }, []);

  // Callback when job submission fails - clear pending message since job didn't start
  const handleJobSubmissionError = useCallback(() => {
    // Clear pending message since job failed to submit
    // This prevents the message from being stuck in "pending" state
    setPendingUserMessage(null);
  }, []);

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
    startTime,
    currentAgent,
    currentQueryMode,
    activeSessionId,
    reconnectToJob,
    // Streaming claims for real-time citation display
    streamingClaims,
    streamingVerificationSummary,
    // Error details for error display
    errorDetails,
    clearErrorDetails,
    // Plan review event bridge
    planReviewEvent,
    clearPlanReviewEvent,
  } = useStreamingQuery(chatId, {
    onStreamComplete: handleStreamComplete,
    onJobSubmissionError: handleJobSubmissionError,
  });

  // Plan review hook
  const planReview = usePlanReview({
    sessionId: activeSessionId ?? null,
  });

  // Discover sources only when plan review is active.
  const { data: discoveryData } = useDiscoveredSources({
    enabled: planReview.isReviewPending,
  });
  const availableSourcesForReview: AvailableSource[] = useMemo(() => {
    if (!discoveryData?.sources) return [];
    return discoveryData.sources
      .filter((s) => s.status === 'ready')
      .map((source) => ({
        id: source.source_id,
        name: source.name,
        type: source.source_type,
        description: source.description ?? null,
        isEnabled: true,
      }));
  }, [discoveryData?.sources]);

  // Bridge plan_review SSE events to usePlanReview hook
  const { handlePlanReviewEvent } = planReview;
  useEffect(() => {
    if (planReviewEvent) {
      handlePlanReviewEvent(planReviewEvent);
      clearPlanReviewEvent();
    }
  }, [planReviewEvent, handlePlanReviewEvent, clearPlanReviewEvent]);

  // Check for active background job for this chat (job-based reconnection)
  // Skip while already streaming/connected to avoid redundant polling.
  const activeJobChatId = chatId && !isDraftChat(chatId) && !isStreaming && !activeSessionId ? chatId : null;
  const { data: activeJob, isLoading: isLoadingActiveJob } = useChatActiveJob(activeJobChatId);

  // Job-based reconnection
  useEffect(() => {
    if (!chatId || isStreaming || isDraftChat(chatId) || isLoadingActiveJob) return;
    // Skip if we already have an active session connected
    if (activeSessionId) return;

    if (activeJob && activeJob.status === 'in_progress') {
      reconnectToJob(activeJob.sessionId);
    }
  }, [chatId, isStreaming, activeJob, isLoadingActiveJob, activeSessionId, reconnectToJob, isDraftChat]);

  // Get the latest agent message with research session from chatFullData
  // This provides claims, verification summary, and sources inline (no separate API call)
  const latestAgentFullMessage = useMemo(() => {
    if (!chatFullData?.messages) return null;
    return chatFullData.messages
      .slice()
      .reverse()
      .find((m) => m.role === 'agent' && m.researchSession) ?? null;
  }, [chatFullData?.messages]);

  // Claims and verification summary from inline data (replaces useCitations for page load)
  // During streaming, streamingClaims from SSE events are used instead
  const claims = latestAgentFullMessage?.claims ?? [];
  const verificationSummary = latestAgentFullMessage?.verificationSummary ?? null;

  // Extract all sources from the latest research session (now populated from DB)
  const allSources = useMemo(() => {
    const session = latestAgentFullMessage?.researchSession;
    if (!session?.sources?.length) return [];
    return session.sources.map((s) => ({
      url: s.url,
      title: s.title,
      snippet: s.snippet,
      is_cited: s.isCited ?? false,
      step_index: undefined as number | undefined,
      crawl_status: undefined as 'success' | 'failed' | 'timeout' | 'blocked' | undefined,
    }));
  }, [latestAgentFullMessage]);

  // Convert streaming claims for ResearchPanel during streaming
  const panelClaims = useMemo((): Claim[] => {
    // Prefer DB claims (complete data) when available
    if (claims.length > 0) return claims;
    // Fall back to streaming claims during active streaming
    if (streamingClaims.length === 0) return [];
    // Convert StreamingClaim[] → Claim[] with all required fields
    return streamingClaims.map((sc): Claim => ({
      id: sc.id,
      claimText: sc.claimText,
      claimType: 'general',
      confidenceLevel: sc.confidenceLevel,
      positionStart: sc.positionStart,
      positionEnd: sc.positionEnd,
      verificationVerdict: sc.verificationVerdict,
      verificationReasoning: sc.reasoning,
      abstained: false,
      citations: sc.citationKey ? [{
        evidenceSpan: {
          id: `evidence-${sc.id}`,
          sourceId: `source-${sc.citationKey}`,
          quoteText: sc.evidencePreview || '',
          startOffset: null,
          endOffset: null,
          sectionHeading: null,
          relevanceScore: null,
          hasNumericContent: false,
          source: {
            id: `source-${sc.citationKey}`,
            title: sc.citationKey,
            url: null,
            author: null,
            publishedDate: null,
            contentType: null,
          },
        },
        confidenceScore: null,
        isPrimary: true,
      }] : [],
      corrections: [],
      numericDetail: null,
      citationKey: sc.citationKey,
      citationKeys: sc.citationKeys,
    }));
  }, [claims, streamingClaims]);

  // Redirect to /chat if chat doesn't exist (404 error)
  // Skip for draft chats - they don't exist in DB yet
  // Skip if streaming - chat may not be persisted yet during active research
  useEffect(() => {
    if (chatId && isDraftChat(chatId)) return; // Draft chats won't be in DB
    if (isStreaming) return; // Don't redirect during streaming - chat persistence may be pending
    if (activeSessionId) return; // Don't redirect while a job is still attached to this view
    if (pendingUserMessage) return; // Don't redirect while user message is pending persistence
    if (chatFullError && 'status' in chatFullError && chatFullError.status === 404) {
      // Chat not found - redirect to new chat
      navigate('/chat', { replace: true });
    }
  }, [
    chatFullError,
    navigate,
    chatId,
    isDraftChat,
    isStreaming,
    activeSessionId,
    pendingUserMessage,
  ]);

  // Handle persistence completion - convert draft to real chat and refetch messages
  // This is the ONLY place we refetch messages after streaming completes,
  // ensuring claims are already persisted when we switch from streaming view to DB view
  const handlePersistenceComplete = useCallback((event: PersistenceCompletedEvent) => {
    // NOW refetch since persistence is complete — chatFull includes claims inline
    queryClient.invalidateQueries({ queryKey: ['messages', event.chatId] });
    queryClient.invalidateQueries({ queryKey: [...CHAT_FULL_KEY, event.chatId] });
    // Invalidate sidebar list unconditionally so derived title (from backend)
    // becomes visible on non-draft chats without a page reload. Without this,
    // post-research renames silently fail to repaint the sidebar.
    queryClient.invalidateQueries({ queryKey: ['chats'] });

    if (event.wasDraft) {
      // Remove from local draft storage
      removeDraft(event.chatId);
      // Navigate to real URL (remove ?draft=1)
      navigate(`/chat/${event.chatId}`, { replace: true });
    }
  }, [removeDraft, navigate, queryClient]);

  // Effect for persistence result
  useEffect(() => {
    if (persistenceResult) {
      handlePersistenceComplete(persistenceResult);
    }
  }, [persistenceResult, handlePersistenceComplete]);

  // Reset completion refetch tracking when starting new streaming session
  useEffect(() => {
    if (isStreaming) {
      hasTriggeredCompletionRefetchRef.current = false;
    }
  }, [isStreaming]);

  // Trigger message refetch when agent completes
  // This handles the case where persistence_completed event is lost due to SSE race condition
  // (job_completed from DB status can close connection before persistence_completed arrives)
  useEffect(() => {
    if (agentStatus !== 'complete' || !chatId || hasTriggeredCompletionRefetchRef.current) {
      return;
    }

    // persistence_completed already triggers a refresh for this chat.
    if (persistenceResult?.chatId === chatId) {
      hasTriggeredCompletionRefetchRef.current = true;
      return;
    }

    hasTriggeredCompletionRefetchRef.current = true;
    const timer = setTimeout(() => {
      queryClient.invalidateQueries({ queryKey: ['messages', chatId] });
      queryClient.invalidateQueries({ queryKey: [...CHAT_FULL_KEY, chatId] });
      // Also invalidate sidebar list so post-research title shows even when
      // persistence_completed was lost to SSE race (see block comment above).
      queryClient.invalidateQueries({ queryKey: ['chats'] });
    }, 1200);

    return () => clearTimeout(timer);
  }, [agentStatus, chatId, persistenceResult?.chatId, queryClient]);

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
          chatId: chatId || '',
          role: role as 'user' | 'agent',
          content: msg.content,
          createdAt: new Date().toISOString(),
          isEdited: false,
        });
      }
    }

    // Add pending user message if exists and belongs to current chat
    // The chatId check prevents showing stale pending messages in wrong chat
    if (pendingUserMessage && pendingUserMessage.chatId === chatId) {
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
    (submission: QuerySubmission) => {
      // Track query for retry functionality
      setLastQuery(submission.message);

      // Note: queryMode is now tracked in useStreamingQuery hook

      // Create a pending user message
      setPendingUserMessage({
        id: `pending-${Date.now()}`,
        chatId: chatId || '',
        role: 'user',
        content: submission.message,
        createdAt: new Date().toISOString(),
        isEdited: false,
      });

      // The hook now automatically tracks conversation history
      // and the backend loads history from DB
      originalSendQuery(submission);
    },
    [chatId, originalSendQuery]
  );

  // Handle pending query from router state after chat creation and navigation
  useEffect(() => {
    if (chatId && location.state?.pendingSubmission) {
      const submission = location.state.pendingSubmission as QuerySubmission;
      // Clear state immediately to prevent re-sending on refresh
      window.history.replaceState({}, document.title);
      sendQuery(submission);
    }
  }, [chatId, location.state?.pendingSubmission, sendQuery]);

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

    // Find the most recent agent message with researchSession
    const agentMessageWithSession = apiMessages
      .slice()
      .reverse()
      .find(m => m.role === 'agent' && m.researchSession);

    if (agentMessageWithSession?.researchSession) {
      hydrateFromSession(agentMessageWithSession.researchSession);
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
    const draft = chatId ? createDraft() : createDraft(preallocatedDraftIdRef.current);
    if (!chatId) {
      preallocatedDraftIdRef.current = crypto.randomUUID();
    }
    navigate(`/chat/${draft.id}?draft=1`);
  }, [createDraft, navigate, chatId]);

  // Send message - for draft chats, backend will persist chat on success
  const handleSendMessage = async (submission: QuerySubmission) => {
    if (!chatId) {
      // No chat selected - create a draft and navigate
      const draft = createDraft(preallocatedDraftIdRef.current);
      preallocatedDraftIdRef.current = crypto.randomUUID();
      navigate(`/chat/${draft.id}?draft=1`, {
        state: { pendingSubmission: submission },
      });
    } else {
      // Chat exists (draft or real) - just send the query
      // Backend handles persistence for drafts via deferred materialization
      sendQuery(submission);
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
        onHoverChat={prefetchMessages}
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
              onClick={() => lastQuery && sendQuery({ message: lastQuery })}
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
            {/* Compute whether to show research panel (for hiding duplicate sources) */}
            {(() => {
              const showResearchPanel = (
                currentQueryMode === 'deep_research' || currentQueryMode === 'web_search' ||
                // Fallback for page reload: show if research session exists with content
                (!!latestAgentFullMessage?.researchSession && (claims.length > 0 || allSources.length > 0))
              ) && (isStreaming || !!currentPlan || events.length > 0 || claims.length > 0 || allSources.length > 0);

              return (
                <MessageList
                  messages={messages}
                  streamingContent={streamingContent}
                  isStreaming={isStreaming}
                  isLoading={isStreaming}
                  className="flex-1"
                  hideAgentSourcesSection={showResearchPanel}
                  // Pass streaming claims for real-time citation display during streaming
                  streamingClaims={streamingClaims}
                  streamingVerificationSummary={streamingVerificationSummary}
                  // Error display with stack trace
                  errorDetails={errorDetails}
                  onRetry={errorDetails?.recoverable ? () => lastQuery && sendQuery({ message: lastQuery }) : undefined}
                  onDismissError={clearErrorDetails}
                  researchPanel={
                    showResearchPanel ? (
                      <ResearchPanel
                        isStreaming={isStreaming}
                        events={events}
                        plan={currentPlan}
                        currentStepIndex={currentStepIndex}
                        startTime={startTime ?? undefined}
                        currentAgent={currentAgent ?? undefined}
                        claims={panelClaims}
                        allSources={allSources}
                        verificationSummary={verificationSummary ?? streamingVerificationSummary}
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
              sessionId={chatId ?? preallocatedDraftIdRef.current}
              inputConfig={inputConfig}
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

      {/* Plan review modal */}
      <PlanReviewModal
        isOpen={planReview.isReviewPending}
        plan={planReview.planForReview}
        availableSources={availableSourcesForReview}
        timeoutSeconds={planReview.timeoutSeconds}
        onApprove={planReview.handleApprove}
        onApproveWithEdits={planReview.handleApproveWithEdits}
        onReject={planReview.handleReject}
        onClose={planReview.clearReview}
      />
    </div>
  );
}
