import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  ChatSidebar,
  MessageList,
  MessageInput,
  DeleteChatDialog,
  ExportChatDialog,
  type ExportFormat,
} from '@/components/chat';
import { SourceBrowserModal } from '@/components/chat/SourceBrowserModal';
import { AgentStatusIndicator, ResearchPanel } from '@/components/research';
import {
  useChats,
  useStreamingQuery,
  useChatActions,
  useDraftChats,
  usePrefetchMessages,
  useChatFull,
  CHAT_FULL_KEY,
  useSourceScope,
} from '@/hooks';
import { useChatActiveJob } from '@/hooks/useResearchJobs';
import { ComponentRegistry } from '@/core/plugins';
import { PlanReviewModal } from '@/components/research/PlanReviewModal';
import { usePlanReview } from '@/hooks/usePlanReview';
import {
  useDiscoveredSources,
  useRefreshDiscovery,
} from '@/hooks/useDiscoveredSources';
import { useAgentV2 } from '@/hooks/useAgentsV2';
import { AgentSurfacePanel } from '@/components/surface/AgentSurfacePanel';
import { MarkdownRenderer } from '@/components/common';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { buildCitationDataMap } from '@/lib/citations';
import { extractSurfaceFromAgentDefinition } from '@/lib/agentSurface';
import { enrichSurfaceRunState } from '@/lib/surfaceEnrichment';
import {
  messageIdsNeedingLiveFill,
  liveFillPollTick,
  initialLiveFillPollState,
} from '@/lib/surfaceLiveFill';
import {
  mapJobStatusToRunStatus,
  toPersistedActionRun,
  actionRunsNeedingReconcile,
  surfaceRunsNeedingLiveReconcile,
  computeCaptureRun,
  surfaceRunScopeKey,
  surfaceRunScopeMatches,
  surfaceRunStateFromPersistedActionRuns,
  type SurfaceRunScope,
} from '@/lib/surfaceRunReconcile';
import type {
  Chat,
  Message,
  MessageRole,
  PersistenceCompletedEvent,
} from '@/types';
import type { Claim } from '@/types/citation';
import type { QuerySubmission } from '@/types/querySubmission';
import type { AvailableSource } from '@/types/dataSources';
import type { CustomAgentSummary } from '@/types/customAgents';
import type { Surface } from '@/types/surface';
import type { RunReference } from '@/types/surface';
import type { CompiledSurfaceSubmission } from '@/lib/surfaceCompile';
import { shouldFetchChatFullForChat } from './chatPageUtils';
import { chatsApi, jobsApi, messagesApi } from '@/api/client';
import type { ChatFullResponse } from '@/types';
import { writeEnabledEnterpriseSources } from '@/components/chat/sourceRouting';

export default function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  // Draft chat management
  const {
    createDraft,
    removeDraft,
    isDraft: isDraftChat,
    getDraftList,
    clearStaleDrafts,
  } = useDraftChats();
  // Pre-allocate a draft session ID so composer/upload works even before chat selection.
  const preallocatedDraftIdRef = useRef<string>(crypto.randomUUID());

  // Prefetch messages on hover for instant chat switching
  const { prefetchMessages } = usePrefetchMessages();

  // Data hooks
  const { data: chatsData, isLoading: isLoadingChats } = useChats();
  const apiChats = useMemo(() => chatsData?.items ?? [], [chatsData?.items]);
  const chatExistsInApi = !!chatId && apiChats.some((c) => c.id === chatId);
  // Skip messages fetch for drafts (they don't exist in DB)
  const shouldFetchMessages = shouldFetchChatFullForChat(
    chatId,
    chatId ? isDraftChat(chatId) : false,
    chatExistsInApi,
  );
  const { data: chatFullData, error: chatFullError } = useChatFull(
    shouldFetchMessages ? chatId : undefined,
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
  const [deleteDialog, setDeleteDialog] = useState<{
    isOpen: boolean;
    chatId: string | null;
    title: string | null;
  }>({
    isOpen: false,
    chatId: null,
    title: null,
  });
  const [exportDialog, setExportDialog] = useState<{
    isOpen: boolean;
    chatId: string | null;
    title: string | null;
  }>({
    isOpen: false,
    chatId: null,
    title: null,
  });
  const [showSurfaceSourceBrowser, setShowSurfaceSourceBrowser] =
    useState(false);

  // Sidebar filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<
    'active' | 'archived' | 'all'
  >('active');

  // Sidebar collapse state — shared with /agents and /designer/* pages via localStorage
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      return window.localStorage?.getItem?.('chatSidebarCollapsed') === '1';
    } catch {
      return false;
    }
  });
  const handleToggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem('chatSidebarCollapsed', next ? '1' : '0');
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  // Selected agent (kept in sync with MessageInput via onSelectedAgentChange)
  const [selectedAgent, setSelectedAgent] = useState<CustomAgentSummary | null>(
    null,
  );
  const surfaceScopeKey = useMemo(
    () => surfaceRunScopeKey(chatId, selectedAgent?.id),
    [chatId, selectedAgent?.id],
  );

  // Which primary view the main area shows when the agent has a surface UI.
  // Defaults to 'ui' (see the effect after the surface memo); the header
  // [ UI | Chat ] toggle switches it. Agents without a surface always show chat.
  const [viewMode, setViewMode] = useState<'ui' | 'chat'>('ui');

  // Run state for surface bindings: action → RunReference | null
  const [surfaceRunState, setSurfaceRunState] = useState<
    Record<string, RunReference | null>
  >({});

  // In-flight surface run: tracks which binding triggered it so we can patch on completion.
  // `sessionId` is stamped by the L0 capture effect once the live stream provides it, so the
  // error path can persist a reconcilable entry even after disconnectStream() clears activeSessionId.
  const surfaceInFlightRef = useRef<
    | ({
        action: string;
        outputTarget: string;
        sessionId?: string;
      } & SurfaceRunScope)
    | null
  >(null);

  // Tracks which chat+agent the page-level surfaceRunState currently belongs to.
  const surfaceRunScopeRef = useRef<string | null>(null);

  // Tracks whether run-state reconciliation has fired for the current chat+agent.
  const surfaceReconcileRef = useRef<string | null>(null);

  // Tracks whether the live completion heal (L1.5) has fired for the current
  // chat+agent completion. Re-armed when a new run starts (agentStatus leaves
  // 'complete').
  const surfaceLiveHealRef = useRef<string | null>(null);

  // Tracks the debounce timer for surface form state persistence.
  const surfacePersistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  useEffect(() => {
    return () => {
      if (surfacePersistTimerRef.current) {
        clearTimeout(surfacePersistTimerRef.current);
        surfacePersistTimerRef.current = null;
      }
    };
  }, [chatId, selectedAgent?.id]);

  // Bump key to remount AgentSurfacePanel (used to reset on drift-banner "Reset form").
  const [surfacePanelKey, setSurfacePanelKey] = useState(0);

  // Per-message citation maps for structured-output cells ([Key] markers).
  const resolveSurfaceCitations = useCallback(
    (messageId: string) => {
      const msg = chatFullData?.messages?.find((m) => m.id === messageId);
      if (!msg || msg.claims.length === 0) return undefined;
      return buildCitationDataMap(msg.claims);
    },
    [chatFullData?.messages],
  );

  // Active restructure poll timers, cleared on unmount so a bounded poll never
  // outlives the page.
  const restructurePollsRef = useRef<Set<ReturnType<typeof setInterval>>>(
    new Set(),
  );
  useEffect(() => {
    const timers = restructurePollsRef.current;
    return () => {
      timers.forEach((t) => clearInterval(t));
      timers.clear();
    };
  }, []);

  // Failed-slot retry: POST restructure, then poll chatFull (every 5s, ≤3 min)
  // until the requested slots leave "pending", so the panel swaps the skeleton
  // for the re-generated data without a manual refresh.
  const retryStructuring = useCallback(
    (messageId: string, slots: string[]) => {
      if (!chatId) return;
      const key = [...CHAT_FULL_KEY, chatId];
      messagesApi
        .restructure(chatId, messageId, slots)
        .then(() => {
          queryClient.invalidateQueries({ queryKey: key });
          let ticks = 0;
          const timer = setInterval(() => {
            ticks += 1;
            queryClient.invalidateQueries({ queryKey: key });
            const data = queryClient.getQueryData<ChatFullResponse>(key);
            const msg = data?.messages?.find((m) => m.id === messageId);
            const slotMeta = msg?.structuredOutput?.meta?.slots ?? {};
            const stillPending = slots.some(
              (s) => slotMeta[s]?.status === 'pending',
            );
            if (!stillPending || ticks >= 36) {
              clearInterval(timer);
              restructurePollsRef.current.delete(timer);
            }
          }, 5000);
          restructurePollsRef.current.add(timer);
        })
        .catch(() => {
          // Fail-soft: a rejected retry leaves the failed slot as-is (the
          // Retry button stays available for another attempt).
        });
    },
    [chatId, queryClient],
  );

  // Live fill: the structured-output wires persist the envelope ~30-40s AFTER
  // persistence_completed / the research message. Whenever a completed run's
  // message is missing its envelope (or still has pending slots), poll chatFull
  // (every 5s, ≤3 min) until it lands — reusing retryStructuring's bounded-poll
  // machinery (so unmount cleanup covers it). Driven by messageIdsNeedingLiveFill
  // (below the `surface` memo) so it covers BOTH explicit surface actions AND
  // regular composer runs under a surface agent (which never seed surfaceRunState).
  const liveFillPolledRef = useRef<Set<string>>(new Set());
  // agentStatus is destructured from useStreamingQuery further down; mirror it
  // into a ref so the long-lived poll interval can read the CURRENT status
  // (complete/error/idle ⇒ the run is terminal) without being re-created.
  const agentStatusRef = useRef<string>('idle');
  // Reset the per-message live-fill dedup on chat switch so a revisited chat
  // whose poll had expired can re-arm. (Stale timers self-retire; any that
  // linger only invalidate an inactive query, which fires no network request.)
  useEffect(() => {
    liveFillPolledRef.current.clear();
  }, [chatId]);
  const scheduleLiveFillPoll = useCallback(
    (messageId: string) => {
      if (!chatId) return;
      if (liveFillPolledRef.current.has(messageId)) return;
      liveFillPolledRef.current.add(messageId);
      const key = [...CHAT_FULL_KEY, chatId];
      // The bounded structuring budget only advances once the envelope stub
      // exists (see liveFillPollTick). This keeps the single poll alive across a
      // long research run — which can far exceed that budget — so it is still
      // ticking when the envelope finally lands (the pre-fix bug expired the
      // budget mid-research and the dedup guard blocked any restart).
      let state = initialLiveFillPollState();
      const timer = setInterval(() => {
        const data = queryClient.getQueryData<ChatFullResponse>(key);
        const m = data?.messages?.find((x) => x.id === messageId);
        const sm = m?.structuredOutput?.meta?.slots;
        const hasEnvelope = !!m?.structuredOutput && !!sm;
        const settled =
          hasEnvelope &&
          !!sm &&
          !Object.values(sm).some((s) => s.status === 'pending');
        const status = agentStatusRef.current;
        const runTerminal =
          status === 'complete' || status === 'error' || status === 'idle';
        const decision = liveFillPollTick(
          state,
          hasEnvelope,
          settled,
          runTerminal,
        );
        state = decision.next;
        if (decision.invalidate) {
          queryClient.invalidateQueries({ queryKey: key });
        }
        if (decision.stop) {
          clearInterval(timer);
          restructurePollsRef.current.delete(timer);
        }
      }, 5000);
      restructurePollsRef.current.add(timer);
    },
    [chatId, queryClient],
  );

  // Whether the drift banner has been dismissed for the current panel session.
  const [driftBannerDismissed, setDriftBannerDismissed] = useState(false);

  // Get plugin-provided input configuration (controls mode selector visibility, etc.)
  const inputConfig = useMemo(() => ComponentRegistry.getInputConfig(), []);

  // Fetch the full agent definition when the selected agent has a surface
  const { data: agentV2Detail } = useAgentV2(
    selectedAgent?.hasSurface ? selectedAgent.id : undefined,
  );

  // Extract surface with a structural guard. Agents may store the surface as
  // definition.surface or as a surface-like definition during shell export.
  const surface = useMemo((): Surface | null => {
    return extractSurfaceFromAgentDefinition(agentV2Detail?.definition);
  }, [agentV2Detail]);

  const surfacePanelIdentity = useMemo(
    () =>
      [
        chatId ?? 'no-chat',
        selectedAgent?.id ?? 'no-agent',
        agentV2Detail?.etag ?? 'no-etag',
        surfacePanelKey,
      ].join(':'),
    [agentV2Detail?.etag, chatId, selectedAgent?.id, surfacePanelKey],
  );

  // When an agent with a surface loads (or the selected agent changes), default
  // the main area to the UI view; the user can toggle to Chat from the header.
  useEffect(() => {
    if (surface) setViewMode('ui');
  }, [surface, selectedAgent?.id]);

  const pendingLiveFillMessageIds = useMemo(
    () =>
      messageIdsNeedingLiveFill(
        surfaceRunState,
        chatFullData?.messages,
        !!surface,
      ),
    [surfaceRunState, chatFullData?.messages, surface],
  );

  const pendingLiveFillMessageIdSet = useMemo(
    () => new Set(pendingLiveFillMessageIds),
    [pendingLiveFillMessageIds],
  );

  // Structured-output enrichment: completed refs gain the persisted message's
  // structuredOutput payload (data + evidence legend + per-slot status —
  // results-by-reference, never stored in surface_state) so output components
  // can read <target>/data/<slot>. While live-fill polling is waiting for a
  // missing envelope, synthesize pending refs so output slots show skeletons.
  const enrichedSurfaceRunState = useMemo<
    Record<string, RunReference | null>
  >(() => {
    const enriched = enrichSurfaceRunState(
      surfaceRunState,
      chatFullData?.messages,
      Date.now(),
      pendingLiveFillMessageIdSet,
    );
    if (!surface || pendingLiveFillMessageIds.length === 0) return enriched;

    const messageId =
      pendingLiveFillMessageIds[pendingLiveFillMessageIds.length - 1];
    if (!messageId) return enriched;

    let next = enriched;
    let cloned = false;
    const cloneOnce = (): void => {
      if (!cloned) {
        next = { ...enriched };
        cloned = true;
      }
    };

    for (const binding of surface.bindings) {
      if (next[binding.action] !== undefined && next[binding.action] !== null)
        continue;
      cloneOnce();
      next[binding.action] = {
        status: 'completed',
        message_id: messageId,
        pendingStructuredOutput: true,
      };
    }
    return next;
  }, [
    surfaceRunState,
    chatFullData?.messages,
    pendingLiveFillMessageIdSet,
    pendingLiveFillMessageIds,
    surface,
  ]);

  // Drive live-fill for BOTH explicit surface actions and regular composer runs
  // under a surface agent. Trigger 2 (regular runs) keys off `m.researchSession`:
  // the backend sources `structured_output` from `research_session.verification_data`
  // (api/v1/chats.py serializer), so an envelope always has a research session —
  // the gate is a strict superset of "has envelope" (no false-negative).
  useEffect(() => {
    for (const id of pendingLiveFillMessageIds) {
      scheduleLiveFillPoll(id);
    }
  }, [pendingLiveFillMessageIds, scheduleLiveFillPoll]);

  // Merge draft chats with API chats (drafts appear at top)
  // Filter out drafts that already exist in API (handles race condition during persistence)
  const chats: Chat[] = useMemo(() => {
    const draftChats = getDraftList();
    // Deduplicate: filter out drafts that have been persisted to API
    const apiChatIds = new Set(apiChats.map((c) => c.id));
    const uniqueDrafts = draftChats.filter((d) => !apiChatIds.has(d.id));
    // Cast DraftChat to Chat (they're compatible for display)
    return [...(uniqueDrafts as unknown as Chat[]), ...apiChats];
  }, [apiChats, getDraftList]);

  // Sync localStorage drafts with API state on load
  // Removes stale drafts (older than 60s) that don't exist in the database
  // This prevents "phantom" chats from appearing after DB is cleaned
  useEffect(() => {
    if (chatsData?.items && !isLoadingChats) {
      const apiChatIds = new Set(chatsData.items.map((c) => c.id));
      clearStaleDrafts(apiChatIds);
    }
  }, [chatsData, isLoadingChats, clearStaleDrafts]);

  // SSE-independent draft recovery. If the chat list proves the backend has
  // persisted this chat, local draft state must not keep blocking chatFull.
  useEffect(() => {
    if (!chatId || !chatExistsInApi || !isDraftChat(chatId)) return;

    removeDraft(chatId);
    queryClient.invalidateQueries({ queryKey: ['messages', chatId] });
    queryClient.invalidateQueries({ queryKey: [...CHAT_FULL_KEY, chatId] });
    queryClient.invalidateQueries({ queryKey: ['chats'] });

    if (location.search.includes('draft=1')) {
      navigate(`/chat/${chatId}`, { replace: true });
    }
  }, [
    chatId,
    chatExistsInApi,
    isDraftChat,
    location.search,
    navigate,
    queryClient,
    removeDraft,
  ]);

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
  const [pendingUserMessage, setPendingUserMessage] = useState<Message | null>(
    null,
  );

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

  // Discover sources when plan review needs them or a surface agent needs host
  // source controls. Surface controls reuse the same AvailableSource payload
  // shape as plan review and the composer source routing helpers.
  const {
    data: discoveryData,
    isLoading: isDiscoveryLoading,
    error: discoveryError,
    refetch: refetchDiscovery,
  } = useDiscoveredSources({
    enabled: planReview.isReviewPending || selectedAgent?.hasSurface === true,
  });
  const refreshDiscoveryMutation = useRefreshDiscovery();
  const validSourceIds = useMemo(() => {
    if (!discoveryData?.sources) return undefined;
    return discoveryData.sources
      .filter((source) => source.status === 'ready')
      .map((source) => source.source_id);
  }, [discoveryData?.sources]);
  const {
    disabledSources: surfaceDisabledSources,
    setDisabledSources: setSurfaceDisabledSources,
  } = useSourceScope({ validSourceIds });
  const availableReadySources: AvailableSource[] = useMemo(() => {
    if (!discoveryData?.sources) return [];
    return discoveryData.sources
      .filter((s) => s.status === 'ready')
      .map((source) => ({
        id: source.source_id,
        name: source.name,
        type: source.source_type,
        description: source.description ?? null,
        isEnabled: !surfaceDisabledSources.includes(source.source_id),
      }));
  }, [discoveryData?.sources, surfaceDisabledSources]);
  const selectedSurfaceSourceIds = useMemo(
    () =>
      availableReadySources
        .filter((source) => source.isEnabled)
        .map((source) => source.id),
    [availableReadySources],
  );
  const discoveredSources = useMemo(
    () => discoveryData?.sources ?? [],
    [discoveryData?.sources],
  );
  const handleApplySurfaceSources = useCallback(
    (ids: string[]) => {
      const allIds = availableReadySources.map((source) => source.id);
      setSurfaceDisabledSources(allIds.filter((id) => !ids.includes(id)));

      const enabledSet = new Set<string>();
      for (const id of ids) {
        const source = discoveredSources.find((item) => item.source_id === id);
        if (
          source &&
          source.source_type !== 'web_search' &&
          source.source_type !== 'uploaded_file'
        ) {
          enabledSet.add(id);
        }
      }
      writeEnabledEnterpriseSources(enabledSet);
    },
    [availableReadySources, discoveredSources, setSurfaceDisabledSources],
  );

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
  const activeJobChatId =
    chatId && !isDraftChat(chatId) && !isStreaming && !activeSessionId
      ? chatId
      : null;
  const { data: activeJob, isLoading: isLoadingActiveJob } =
    useChatActiveJob(activeJobChatId);

  // Job-based reconnection
  useEffect(() => {
    if (!chatId || isStreaming || isDraftChat(chatId) || isLoadingActiveJob)
      return;
    // Skip if we already have an active session connected
    if (activeSessionId) return;

    if (activeJob && activeJob.status === 'in_progress') {
      reconnectToJob(activeJob.sessionId);
    }
  }, [
    chatId,
    isStreaming,
    activeJob,
    isLoadingActiveJob,
    activeSessionId,
    reconnectToJob,
    isDraftChat,
  ]);

  // SSE-independent completion detection. The polled `useChatActiveJob` (3s
  // cadence) returns `null` once the backend marks the job complete. When that
  // transition is observed, invalidate the chat/message caches so the final
  // report appears even if the SSE persistence_completed event was lost.
  //
  // Why this matters: SSE drops more often than persistence completes. Without
  // this fallback, a user whose SSE connection broke mid-research stays on
  // stale React Query data (staleTime=2min, gcTime=Infinity) — manual refresh
  // helps only after staleTime elapses. Polling-based invalidation closes the
  // gap to the 3s `useChatActiveJob` cadence.
  const prevActiveJobIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!chatId || isDraftChat(chatId)) return;
    const previousJobId = prevActiveJobIdRef.current;
    const currentJobId = activeJob?.sessionId ?? null;
    prevActiveJobIdRef.current = currentJobId;
    // Transition from "job in progress" → "no active job" means the backend
    // completed (or failed) the workflow. Refresh chat-scoped queries so the
    // newly-persisted final report becomes visible.
    if (previousJobId && !currentJobId) {
      queryClient.invalidateQueries({ queryKey: ['messages', chatId] });
      queryClient.invalidateQueries({ queryKey: [...CHAT_FULL_KEY, chatId] });
      queryClient.invalidateQueries({ queryKey: ['chats'] });
    }
  }, [activeJob?.sessionId, chatId, isDraftChat, queryClient]);

  // ---------------------------------------------------------------------------
  // Surface state persistence helpers
  // ---------------------------------------------------------------------------

  /** Persisted entry for the current agent from chatFullData.surfaceState. */
  const persistedEntry = selectedAgent
    ? (chatFullData?.surfaceState?.[selectedAgent.id] ?? undefined)
    : undefined;

  /**
   * PUT the given patch to the server.  No-op for draft chats (no DB row yet).
   * Fire-and-forget with console.warn on failure.
   */
  const persistSurfaceState = useCallback(
    (patch: Record<string, unknown>) => {
      if (!chatId || !selectedAgent) return;
      // Only persist for real (non-draft) chats.
      if (isDraftChat(chatId)) return;
      void chatsApi
        .putSurfaceState(chatId, { [selectedAgent.id]: patch })
        .catch(console.warn);
    },
    [chatId, selectedAgent, isDraftChat],
  );

  // Staleness guards for async run-state resolves: BOTH chat and agent, because
  // persistSurfaceState is agent-scoped ({[selectedAgent.id]: patch}) — an agent
  // switch mid-resolve would otherwise write to the wrong agent's entry.
  const chatIdRef = useRef(chatId);
  const agentIdRef = useRef(selectedAgent?.id);
  useEffect(() => {
    chatIdRef.current = chatId;
  }, [chatId]);
  useEffect(() => {
    agentIdRef.current = selectedAgent?.id;
  }, [selectedAgent?.id]);
  useEffect(() => {
    agentStatusRef.current = agentStatus;
  }, [agentStatus]);

  // Single writer for surface run-state transitions: update in-memory state and
  // persist the projection (toPersistedActionRun drops the enrichment payload so
  // surface_state never bloats toward the 128KB PUT cap).
  const setAndPersistRun = useCallback(
    (action: string, ref: RunReference) => {
      setSurfaceRunState((s) => ({ ...s, [action]: ref }));
      persistSurfaceState({
        action_runs: {
          [action]: toPersistedActionRun(ref, new Date().toISOString()),
        },
      });
    },
    [persistSurfaceState],
  );

  // Resolve a run's TRUE status from the server job, then set+persist it. Shared
  // by the error-path heal (L2) and the on-load reconcile (L3). Never throws into
  // React; an unreachable server leaves the entry as-is; a chat/agent switch
  // mid-flight drops the stale write.
  const resolveRunFromServer = useCallback(
    async (action: string, sessionId: string, messageId?: string) => {
      const chat = chatId;
      const agent = selectedAgent?.id;
      if (!chat) return;
      try {
        const job = await jobsApi.get(chat, sessionId);
        if (chatIdRef.current !== chat || agentIdRef.current !== agent) return; // switched away
        const status = mapJobStatusToRunStatus(job.status);
        console.warn(
          `[surface] resolve ${action} session=${sessionId}: server=${job.status} → ${status}`,
        );
        setAndPersistRun(action, {
          status,
          session_id: sessionId,
          ...(messageId ? { message_id: messageId } : {}),
        });
      } catch {
        // Unreachable server → leave the entry as-is (don't force a wrong terminal).
      }
    },
    [chatId, selectedAgent?.id, setAndPersistRun],
  );

  // L1 — fast completion path: persistenceResult carries the ids; mark completed.
  useEffect(() => {
    const inflight = surfaceInFlightRef.current;
    if (!inflight || !persistenceResult) return;
    if (
      !surfaceRunScopeMatches(
        inflight,
        chatIdRef.current,
        agentIdRef.current,
      ) ||
      persistenceResult.chatId !== inflight.chatId
    ) {
      surfaceInFlightRef.current = null;
      return;
    }
    surfaceInFlightRef.current = null;
    setAndPersistRun(inflight.action, {
      status: 'completed',
      message_id: persistenceResult.messageId,
      session_id: persistenceResult.researchSessionId,
    });
  }, [persistenceResult, setAndPersistRun]);

  // L2 — same-session heal: a client-side stream error is NOT the job's verdict.
  // Resolve the true status from the server (completed/failed/running) instead of
  // blindly persisting failed. No session id ⇒ the job never started ⇒ failed.
  useEffect(() => {
    const inflight = surfaceInFlightRef.current;
    if (!inflight || agentStatus !== 'error') return;
    if (
      !surfaceRunScopeMatches(inflight, chatIdRef.current, agentIdRef.current)
    ) {
      surfaceInFlightRef.current = null;
      return;
    }
    const { action, sessionId } = inflight;
    surfaceInFlightRef.current = null;
    if (sessionId) void resolveRunFromServer(action, sessionId);
    else setAndPersistRun(action, { status: 'failed' });
  }, [agentStatus, resolveRunFromServer, setAndPersistRun]);

  // L0 — capture the session id while the stream is live, so a run that later
  // drops its SSE (agentStatus→error, activeSessionId already cleared) still has
  // a session_id persisted for the server-truth heal. Order-independent: both the
  // setState and the persist are gated on the run being non-terminal, so a
  // post-completion reconnect re-fire can never overwrite a completed entry.
  useEffect(() => {
    const inflight = surfaceInFlightRef.current;
    if (!inflight || !activeSessionId) return;
    if (
      !surfaceRunScopeMatches(inflight, chatIdRef.current, agentIdRef.current)
    )
      return;
    if (inflight.sessionId === activeSessionId) return;
    inflight.sessionId = activeSessionId; // in-memory stamp (unconditional — L2 relies on it)
    let shouldPersist = false;
    setSurfaceRunState((s) => {
      const next = computeCaptureRun(s[inflight.action], activeSessionId);
      if (!next) return s;
      shouldPersist = true;
      return { ...s, [inflight.action]: next };
    });
    if (shouldPersist) {
      persistSurfaceState({
        action_runs: {
          [inflight.action]: {
            status: 'running',
            session_id: activeSessionId,
            updated_at: new Date().toISOString(),
          },
        },
      });
    }
  }, [activeSessionId, persistSurfaceState]);

  // Scope page-level run state to the current chat+agent. Common action names
  // like "run" must not leak completed refs into another surface or block that
  // surface's persisted action_runs from seeding.
  useEffect(() => {
    if (surfaceRunScopeRef.current === surfaceScopeKey) return;
    surfaceRunScopeRef.current = surfaceScopeKey;
    const keepInFlight = surfaceRunScopeMatches(
      surfaceInFlightRef.current,
      chatId,
      selectedAgent?.id,
    );
    if (!keepInFlight) surfaceInFlightRef.current = null;
    setSurfaceRunState((current) => (keepInFlight ? current : {}));
    setDriftBannerDismissed(false);
    surfaceReconcileRef.current = null;
    surfaceLiveHealRef.current = null;
  }, [chatId, selectedAgent?.id, surfaceScopeKey]);

  // Seed surfaceRunState from persistedEntry.action_runs when it arrives and
  // local run state is still empty (no in-progress run has set it yet).
  useEffect(() => {
    if (!persistedEntry?.action_runs) return;
    setSurfaceRunState((current) => {
      // Only seed when local state is empty to avoid overwriting live run state.
      if (Object.keys(current).length > 0) return current;
      return surfaceRunStateFromPersistedActionRuns(persistedEntry.action_runs);
    });
  }, [persistedEntry?.action_runs, surfaceScopeKey]);

  // L3 — on load, reconcile stale persisted running/failed entries (that carry a
  // session_id) against server truth, so a false-'failed' (or stuck-'running')
  // from a prior session/device self-heals.
  useEffect(() => {
    if (!chatId || !selectedAgent || !persistedEntry?.action_runs) return;
    if (isDraftChat(chatId)) return;

    // Guard: run at most once per chat+agent combination.
    const guardKey = `${chatId}:${selectedAgent.id}`;
    if (surfaceReconcileRef.current === guardKey) return;
    surfaceReconcileRef.current = guardKey;

    // Heal stale running/failed entries (that carry a session_id) from server truth.
    const toReconcile = actionRunsNeedingReconcile(persistedEntry.action_runs);
    if (toReconcile.length === 0) return;

    void (async () => {
      let activeJobSessionId: string | null = null;
      try {
        const activeJob = await jobsApi.getChatActiveJob(chatId);
        activeJobSessionId = activeJob?.sessionId ?? null;
      } catch {
        // Can't check the active job — reconcile the rest against the server.
      }

      for (const { action, sessionId, run } of toReconcile) {
        // If the live active job owns this session, leave it — streaming will finish it.
        if (activeJobSessionId && activeJobSessionId === sessionId) continue;
        await resolveRunFromServer(action, sessionId, run.message_id);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, selectedAgent?.id, persistedEntry?.action_runs]);

  // L1.5 — live completion heal (the live twin of L3). The ONLY live
  // 'running'→'completed' transition is L1, which self-cancels on a scope/chat-id
  // mismatch or a lost persistence_completed; the L3 heal runs only on mount. So
  // a run whose L1 missed leaves its ref stuck 'running' — and slot enrichment is
  // gated on 'completed' — so the surface shows skeletons until a manual reload.
  // When the stream reaches 'complete', reconcile any still-'running' scoped ref
  // from server truth so the surface fills live. Re-armed when the next run
  // starts (agentStatus leaves 'complete'); scope-guarded; safe for re-runs
  // (resolves the actual job by session_id, never guesses from messages).
  useEffect(() => {
    if (agentStatus !== 'complete') {
      surfaceLiveHealRef.current = null; // re-arm for the next completion
      return;
    }
    if (!chatId || !selectedAgent || isDraftChat(chatId)) return;
    const guardKey = `${chatId}:${selectedAgent.id}`;
    if (surfaceLiveHealRef.current === guardKey) return;
    const pending = surfaceRunsNeedingLiveReconcile(surfaceRunState);
    if (pending.length === 0) return;
    surfaceLiveHealRef.current = guardKey;
    void (async () => {
      for (const { action, sessionId } of pending) {
        await resolveRunFromServer(action, sessionId);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentStatus, chatId, selectedAgent?.id, surfaceRunState]);

  // Get the latest agent message with research session from chatFullData
  // This provides claims, verification summary, and sources inline (no separate API call)
  const latestAgentFullMessage = useMemo(() => {
    if (!chatFullData?.messages) return null;
    return (
      chatFullData.messages
        .slice()
        .reverse()
        .find((m) => m.role === 'agent' && m.researchSession) ?? null
    );
  }, [chatFullData?.messages]);

  // Claims and verification summary from inline data (replaces useCitations for page load)
  // During streaming, streamingClaims from SSE events are used instead
  const claims = useMemo(
    () => latestAgentFullMessage?.claims ?? [],
    [latestAgentFullMessage?.claims],
  );
  const verificationSummary =
    latestAgentFullMessage?.verificationSummary ?? null;

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
      crawl_status: undefined as
        'success' | 'failed' | 'timeout' | 'blocked' | undefined,
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
      citations: sc.citationKey
        ? [
            {
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
            },
          ]
        : [],
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
    if (
      chatFullError &&
      'status' in chatFullError &&
      chatFullError.status === 404
    ) {
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
  const handlePersistenceComplete = useCallback(
    (event: PersistenceCompletedEvent) => {
      // NOW refetch since persistence is complete — chatFull includes claims inline
      queryClient.invalidateQueries({ queryKey: ['messages', event.chatId] });
      queryClient.invalidateQueries({
        queryKey: [...CHAT_FULL_KEY, event.chatId],
      });
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
    },
    [removeDraft, navigate, queryClient],
  );

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
    if (
      agentStatus !== 'complete' ||
      !chatId ||
      hasTriggeredCompletionRefetchRef.current
    ) {
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
        (m) => m.content === msg.content && m.role === msg.role,
      );
      if (!exists) {
        // Map 'assistant' to 'agent' for MessageRole compatibility
        const role = msg.role === 'assistant' ? 'agent' : msg.role;
        // For agent messages, use real UUID from backend if available for citation fetching
        // For user messages, use placeholder ID (they don't need citation support)
        const messageId =
          msg.role === 'assistant' && agentMessageId
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
        (m) => m.content === pendingUserMessage.content && m.role === 'user',
      );
      if (!exists) {
        baseMessages.push(pendingUserMessage);
      }
    }

    return baseMessages;
  }, [
    apiMessages,
    completedMessages,
    pendingUserMessage,
    chatId,
    agentMessageId,
  ]);

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
    [chatId, originalSendQuery],
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
        (m) => m.content === pendingUserMessage.content && m.role === 'user',
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
      .find((m) => m.role === 'agent' && m.researchSession);

    if (agentMessageWithSession?.researchSession) {
      hydrateFromSession(agentMessageWithSession.researchSession);
    }
  }, [apiMessages, isStreaming, currentPlan, hydrateFromSession]);

  // Warn user before leaving a draft chat with unsent content
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (
        chatId &&
        isDraftChat(chatId) &&
        (isStreaming || pendingUserMessage)
      ) {
        e.preventDefault();
        e.returnValue =
          'You have an unsaved draft. Are you sure you want to leave?';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [chatId, isDraftChat, isStreaming, pendingUserMessage]);

  // Create new draft chat - instant, no API call
  // Chat will be persisted when first message is successfully processed
  const handleNewChat = useCallback(() => {
    const draft = chatId
      ? createDraft()
      : createDraft(preallocatedDraftIdRef.current);
    if (!chatId) {
      preallocatedDraftIdRef.current = crypto.randomUUID();
    }
    navigate(`/chat/${draft.id}?draft=1`);
  }, [createDraft, navigate, chatId]);

  // Handle a Run action triggered from the Agent UI surface panel
  const handleSurfaceRunAction = useCallback(
    (compiled: CompiledSurfaceSubmission) => {
      if (!selectedAgent) return;
      const binding = compiled.binding;
      const runChatId = chatId ?? preallocatedDraftIdRef.current;
      const submission: QuerySubmission = compiled.submission;
      // Stash the in-flight action before sending
      surfaceInFlightRef.current = {
        action: binding.action,
        outputTarget: binding.output.target,
        chatId: runChatId,
        agentId: selectedAgent.id,
      };
      setSurfaceRunState((s) => ({
        ...s,
        [binding.action]: { status: 'running' },
      }));
      persistSurfaceState({
        action_runs: {
          [binding.action]: {
            status: 'running',
            updated_at: new Date().toISOString(),
          },
        },
      });
      // Use the same send path as the composer
      void handleSendMessage(submission);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [chatId, selectedAgent, persistSurfaceState],
  );

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
  const handleRenameChat = useCallback(
    (targetChatId: string, newTitle: string) => {
      chatActions.renameChat(targetChatId, newTitle);
    },
    [chatActions],
  );

  const handleArchiveChat = useCallback(
    (targetChatId: string) => {
      chatActions.archiveChat(targetChatId);
    },
    [chatActions],
  );

  const handleRestoreChat = useCallback(
    (targetChatId: string) => {
      chatActions.restoreChat(targetChatId);
    },
    [chatActions],
  );

  // Dialog openers (show confirmation/selection UI before executing action)
  const handleDeleteClick = useCallback(
    (targetChatId: string) => {
      const chat = chats.find((c) => c.id === targetChatId);
      setDeleteDialog({
        isOpen: true,
        chatId: targetChatId,
        title: chat?.title || null,
      });
    },
    [chats],
  );

  const handleExportClick = useCallback(
    (targetChatId: string) => {
      const chat = chats.find((c) => c.id === targetChatId);
      setExportDialog({
        isOpen: true,
        chatId: targetChatId,
        title: chat?.title || null,
      });
    },
    [chats],
  );

  // Dialog confirmations (execute the actual action)
  const handleConfirmDelete = useCallback(() => {
    if (deleteDialog.chatId) {
      chatActions.deleteChat(deleteDialog.chatId);
    }
    setDeleteDialog({ isOpen: false, chatId: null, title: null });
  }, [deleteDialog.chatId, chatActions]);

  const handleConfirmExport = useCallback(
    (format: ExportFormat, _includeMetadata: boolean) => {
      // Note: includeMetadata is not currently supported by the backend API
      if (exportDialog.chatId) {
        chatActions.exportChat(exportDialog.chatId, format);
      }
      setExportDialog({ isOpen: false, chatId: null, title: null });
    },
    [exportDialog.chatId, chatActions],
  );

  const handleCloseDeleteDialog = useCallback(() => {
    setDeleteDialog({ isOpen: false, chatId: null, title: null });
  }, []);

  const handleCloseExportDialog = useCallback(() => {
    setExportDialog({ isOpen: false, chatId: null, title: null });
  }, []);

  return (
    <div className="db-root flex h-screen bg-db-oat-light font-db-sans text-db-navy-800">
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
        collapsed={sidebarCollapsed}
        onToggleCollapsed={handleToggleSidebar}
      />

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header with status */}
        <header className="flex items-center justify-between px-4 py-2 border-b">
          <h1 className="font-semibold truncate">
            {chatId
              ? chats.find((c) => c.id === chatId)?.title ||
                (isDraftChat(chatId || '') ? 'New chat...' : 'New chat')
              : 'Deep Research Agent'}
          </h1>
          <div className="flex items-center gap-3">
            {surface && (
              <div
                role="tablist"
                aria-label="View mode"
                className="inline-flex items-center rounded-db-md border border-db-gray-lines bg-white p-0.5 text-[12px] font-medium"
              >
                {(['ui', 'chat'] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    role="tab"
                    aria-selected={viewMode === mode}
                    onClick={() => setViewMode(mode)}
                    className={
                      viewMode === mode
                        ? 'rounded-[5px] bg-db-navy-800 px-3 py-1 text-white'
                        : 'px-3 py-1 text-db-navy-800 hover:text-db-navy-900'
                    }
                  >
                    {mode === 'ui' ? 'UI' : 'Chat'}
                  </button>
                ))}
              </div>
            )}
            <AgentStatusIndicator status={agentStatus} />
          </div>
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
          {/* Chat view — conversation + composer; shown when no surface or Chat toggle */}
          {(!surface || viewMode === 'chat') && (
            <div className="flex-1 flex flex-col min-w-0">
              {/* Compute whether to show research panel (for hiding duplicate sources) */}
              {(() => {
                const showResearchPanel =
                  (currentQueryMode === 'deep_research' ||
                    currentQueryMode === 'web_search' ||
                    // Fallback for page reload: show if research session exists with content
                    (!!latestAgentFullMessage?.researchSession &&
                      (claims.length > 0 || allSources.length > 0))) &&
                  (isStreaming ||
                    !!currentPlan ||
                    events.length > 0 ||
                    claims.length > 0 ||
                    allSources.length > 0);

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
                    onRetry={
                      errorDetails?.recoverable
                        ? () => lastQuery && sendQuery({ message: lastQuery })
                        : undefined
                    }
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
                          verificationSummary={
                            verificationSummary ?? streamingVerificationSummary
                          }
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
                onSelectedAgentChange={setSelectedAgent}
              />
            </div>
          )}

          {/* Agent UI — primary full-width view when the UI toggle is active */}
          {surface && viewMode === 'ui' && (
            <div className="flex-1 flex flex-col min-h-0 overflow-y-auto bg-white">
              {/* Drift banner: agent definition changed since last use */}
              {!driftBannerDismissed &&
                persistedEntry?.surface_etag &&
                agentV2Detail?.etag &&
                persistedEntry.surface_etag !== agentV2Detail.etag && (
                  <div className="flex shrink-0 items-center justify-between border-b border-amber-200 bg-amber-50 px-4 py-2">
                    <p className="text-[12px] text-amber-800">
                      This agent&apos;s UI was updated since you last used it.
                    </p>
                    <button
                      type="button"
                      className="ml-3 shrink-0 rounded px-2 py-1 text-[11px] font-medium text-amber-800 hover:bg-amber-100"
                      onClick={() => {
                        // Reset: PUT the surface default + new etag, remount panel.
                        persistSurfaceState({
                          data_model: surface.data_model,
                          surface_etag: agentV2Detail!.etag ?? null,
                        });
                        setSurfacePanelKey((k) => k + 1);
                        setDriftBannerDismissed(true);
                      }}
                    >
                      Reset form
                    </button>
                  </div>
                )}
              <ErrorBoundary
                name="Surface"
                fallback={
                  <div className="m-4 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm">
                    <p className="mb-1 font-medium text-amber-800">
                      This agent&apos;s form couldn&apos;t be displayed.
                    </p>
                    <p className="mb-3 text-amber-700">
                      There was a problem rendering the agent UI — you can still
                      use the chat.
                    </p>
                    <button
                      type="button"
                      onClick={() => setViewMode('chat')}
                      className="rounded-md bg-amber-600 px-3 py-1.5 font-medium text-white hover:bg-amber-700"
                    >
                      Switch to Chat
                    </button>
                  </div>
                }
              >
                <AgentSurfacePanel
                  key={surfacePanelIdentity}
                  agentName={selectedAgent?.name ?? ''}
                  surface={surface}
                  surfaceIdentity={surfacePanelIdentity}
                  selectedAgentId={selectedAgent?.id}
                  availableSources={availableReadySources}
                  disabledSourceIds={surfaceDisabledSources}
                  onBrowseSources={() => setShowSurfaceSourceBrowser(true)}
                  isDiscoveringSources={isDiscoveryLoading}
                  agentDefinesSources={selectedAgent?.hasSourceConfig === true}
                  initialDataModel={persistedEntry?.data_model}
                  onFormStateChange={(dataModel) => {
                    if (surfacePersistTimerRef.current) {
                      clearTimeout(surfacePersistTimerRef.current);
                    }
                    surfacePersistTimerRef.current = setTimeout(() => {
                      persistSurfaceState({
                        data_model: dataModel,
                        surface_etag: agentV2Detail?.etag ?? null,
                      });
                      surfacePersistTimerRef.current = null;
                    }, 800);
                  }}
                  onRunAction={handleSurfaceRunAction}
                  runDisabled={isStreaming || !!activeSessionId}
                  runState={enrichedSurfaceRunState}
                  resolveCitations={resolveSurfaceCitations}
                  retryStructuring={retryStructuring}
                  resolveRunReference={(ref) => {
                    if (!ref) return null;
                    if (ref.status === 'running') {
                      return (
                        <div className="flex items-center gap-2 text-[12px] text-db-gray-text">
                          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-db-navy-800 border-t-transparent" />
                          Running…
                        </div>
                      );
                    }
                    if (ref.status === 'failed') {
                      return (
                        <p className="text-[12px] text-db-lava-700">
                          Run failed
                        </p>
                      );
                    }
                    if (ref.status === 'completed' && ref.message_id) {
                      const msg = chatFullData?.messages?.find(
                        (m) => m.id === ref.message_id,
                      );
                      if (msg) {
                        return (
                          <div>
                            <div className="max-h-[40vh] overflow-auto">
                              <MarkdownRenderer content={msg.content} />
                            </div>
                            <p className="mt-1 text-[11px] text-db-gray-text">
                              Full report with citations is in the conversation
                            </p>
                          </div>
                        );
                      }
                      return (
                        <p className="text-[12px] text-db-gray-text">
                          Report is loading…
                        </p>
                      );
                    }
                    return null;
                  }}
                  onClose={() => setViewMode('chat')}
                />
              </ErrorBoundary>
            </div>
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

      {/* Plan review modal */}
      <PlanReviewModal
        isOpen={planReview.isReviewPending}
        plan={planReview.planForReview}
        availableSources={availableReadySources}
        timeoutSeconds={planReview.timeoutSeconds}
        onApprove={planReview.handleApprove}
        onApproveWithEdits={planReview.handleApproveWithEdits}
        onReject={planReview.handleReject}
        onClose={planReview.clearReview}
      />
      <SourceBrowserModal
        isOpen={showSurfaceSourceBrowser}
        onClose={() => setShowSurfaceSourceBrowser(false)}
        initialSelectedIds={selectedSurfaceSourceIds}
        onApply={handleApplySurfaceSources}
        sources={discoveredSources}
        isDiscoveryLoading={isDiscoveryLoading}
        discoveryError={discoveryError ?? null}
        onRefetch={() => refetchDiscovery()}
        onRefresh={() => refreshDiscoveryMutation.mutate({})}
        isRefreshing={refreshDiscoveryMutation.isPending}
      />
    </div>
  );
}
