import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  StreamEvent,
  PlanCreatedEvent,
  StepStartedEvent,
  StepCompletedEvent,
  ToolCallEvent,
  ToolResultEvent,
  SynthesisProgressEvent,
  StreamErrorEvent,
  ReflectionDecisionEvent,
  ClaimVerifiedEvent,
  VerificationSummaryEvent,
  ResearchStartedEvent,
  PersistenceCompletedEvent,
  ResearchSession,
  QueryMode,
} from '../types';
import type {
  VerificationSummary,
  VerificationVerdict,
  ConfidenceLevel,
} from '../types/citation';
import { parseStreamEvent } from '../schemas/streamEvents';
import { jobsApi } from '../api/client';

type AgentStatus =
  | 'idle'
  | 'classifying'
  | 'planning'
  | 'researching'
  | 'reflecting'
  | 'synthesizing'
  | 'verifying'
  | 'complete'
  | 'error';

interface Plan {
  title?: string;
  reasoning?: string;
  steps: Array<{
    index: number;
    title: string;
    description?: string;
    status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  }>;
}

interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

/** Streaming claim data (partial, built from events) */
export interface StreamingClaim {
  id: string;
  claimText: string;
  positionStart: number;
  positionEnd: number;
  verificationVerdict: VerificationVerdict | null;
  confidenceLevel: ConfidenceLevel | null;
  evidencePreview: string;
  reasoning: string | null;
  /** Primary citation key for citationData mapping (e.g., "Arxiv", "Zhipu") */
  citationKey: string | null;
  /** All citation keys for multi-source claims */
  citationKeys: string[] | null;
}

/** Current tool activity during ReAct research loop */
export interface ToolActivity {
  toolName: 'web_search' | 'web_crawl' | null;
  toolArgs: Record<string, unknown>;
  callNumber: number;
  sourcesCrawled: number;
}

/** @deprecated Use ResearchSession from '../types' instead */
export type PersistedResearchSession = ResearchSession;

/** Full error details including stack trace for debugging */
export interface ErrorDetails {
  error: Error;
  errorCode?: string;
  stackTrace?: string;
  errorType?: string;
  recoverable?: boolean;
}

interface UseStreamingQueryReturn {
  isStreaming: boolean;
  events: StreamEvent[];
  streamingContent: string;
  agentStatus: AgentStatus;
  currentPlan: Plan | null;
  currentStepIndex: number;
  sendQuery: (query: string, queryMode?: QueryMode, researchDepth?: string, verifySources?: boolean) => Promise<void>;
  stopStream: () => void;
  error: Error | null;
  /** Full error details including stack trace */
  errorDetails: ErrorDetails | null;
  /** Clear error state (dismiss error alert) */
  clearErrorDetails: () => void;
  /** The completed messages from this session (for tracking conversation) */
  completedMessages: ConversationMessage[];
  /** Claims verified during streaming */
  streamingClaims: StreamingClaim[];
  /** Verification summary from stream */
  streamingVerificationSummary: VerificationSummary | null;
  /** Number of citation corrections made */
  citationCorrectionCount: number;
  /** The agent message UUID from backend (for citation fetching) */
  agentMessageId: string | null;
  /** Current tool activity during ReAct research loop */
  toolActivity: ToolActivity | null;
  /** Persistence result when draft chat becomes real */
  persistenceResult: PersistenceCompletedEvent | null;
  /** Whether persistence failed (for retry UI) */
  persistenceFailed: boolean;
  /** Hydrate state from persisted research session (for page reload) */
  hydrateFromSession: (session: ResearchSession) => void;
  /** Process an external event (for reconnection replay) */
  processExternalEvent: (event: StreamEvent) => void;
  /** Set streaming state (for reconnection to enable streaming UI) */
  setIsStreaming: (value: boolean) => void;
  /** Set streaming content directly (for reconnection final report) */
  setStreamingContent: (content: string) => void;
  /** Timestamp when streaming started (for elapsed time display) */
  startTime: number | null;
  /** Current active agent name (for status display) */
  currentAgent: string | null;
  /** Current query mode for the active/last streaming session (for UI rendering) */
  currentQueryMode: QueryMode | null;
  /** Set current query mode (for reconnection restoration) */
  setCurrentQueryMode: (mode: QueryMode | null) => void;
  /** Active job session ID (for background job architecture) */
  activeSessionId: string | null;
  /** Reconnect to an existing job's event stream */
  reconnectToJob: (sessionId: string) => Promise<void>;
}

interface UseStreamingQueryOptions {
  /** Callback invoked when streaming completes successfully */
  onStreamComplete?: () => void;
}

export function useStreamingQuery(
  chatId?: string,
  options?: UseStreamingQueryOptions
): UseStreamingQueryReturn {
  const { onStreamComplete } = options ?? {};
  const [isStreaming, setIsStreaming] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [streamingContent, setStreamingContent] = useState('');
  const [agentStatus, setAgentStatus] = useState<AgentStatus>('idle');
  const [currentPlan, setCurrentPlan] = useState<Plan | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [error, setError] = useState<Error | null>(null);
  const [errorDetails, setErrorDetails] = useState<ErrorDetails | null>(null);

  // Clear error details (dismiss error alert)
  const clearErrorDetails = useCallback(() => {
    setError(null);
    setErrorDetails(null);
  }, []);

  // Track completed messages for conversation history
  const [completedMessages, setCompletedMessages] = useState<ConversationMessage[]>([]);
  // Track the current query for adding to completed messages
  const currentQueryRef = useRef<string>('');

  // Citation verification state
  const [streamingClaims, setStreamingClaims] = useState<StreamingClaim[]>([]);
  const [streamingVerificationSummary, setStreamingVerificationSummary] = useState<VerificationSummary | null>(null);
  const [citationCorrectionCount, setCitationCorrectionCount] = useState(0);

  // Agent message UUID from backend (for citation fetching with real IDs)
  const [agentMessageId, setAgentMessageId] = useState<string | null>(null);

  // Tool activity state for ReAct research loop
  const [toolActivity, setToolActivity] = useState<ToolActivity | null>(null);

  // Persistence state for draft chat support
  const [persistenceResult, setPersistenceResult] = useState<PersistenceCompletedEvent | null>(null);
  const [persistenceFailed, setPersistenceFailed] = useState(false);

  // Streaming time and agent tracking for activity panel
  const [startTime, setStartTime] = useState<number | null>(null);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);

  // Query mode for the current streaming session (used for UI visibility)
  const [currentQueryMode, setCurrentQueryMode] = useState<QueryMode | null>(null);

  // ===== DEBUG LOGGING =====
  useEffect(() => {
    console.log('[useStreamingQuery] currentQueryMode is now:', currentQueryMode);
  }, [currentQueryMode]);

  useEffect(() => {
    console.log('[useStreamingQuery] isStreaming is now:', isStreaming);
  }, [isStreaming]);

  useEffect(() => {
    console.log('[useStreamingQuery] currentPlan changed:', currentPlan ? `has plan with ${currentPlan.steps?.length} steps` : 'null');
  }, [currentPlan]);

  useEffect(() => {
    console.log('[useStreamingQuery] events count:', events.length);
  }, [events.length]);
  // ===== END DEBUG LOGGING =====

  // Event counter for stable unique keys (prevents blinking on re-renders)
  const eventCounterRef = useRef(0);

  // Track seen event keys for deduplication (reconnection scenario)
  const seenEventKeysRef = useRef<Set<string>>(new Set());

  // Track previous chatId to prevent false state resets during draft→real navigation
  const prevChatIdRef = useRef<string | undefined>(undefined);

  // Track SSE connection errors for resilient handling
  const sseErrorCountRef = useRef(0);
  const lastSseErrorTimeRef = useRef(0);

  const eventSourceRef = useRef<EventSource | null>(null);

  // Track active job session ID for background job architecture
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  // Track last sequence number for reconnection support
  const lastSequenceRef = useRef(0);

  /**
   * Check if an event has already been processed (for reconnection deduplication).
   * Uses sequence_number if available, otherwise falls back to eventType+timestamp.
   * Returns true if duplicate (should skip), false if new (should process).
   */
  const isDuplicateEvent = useCallback((data: StreamEvent): boolean => {
    // Build a unique key for this event
    // Handle camelCase runtime keys (sequenceNumber vs sequence_number)
    const seqNum = (data as unknown as { sequenceNumber?: number }).sequenceNumber
      ?? (data as unknown as { sequence_number?: number }).sequence_number;

    const key = seqNum !== undefined
      ? `seq:${seqNum}`
      : `${data.event_type}:${(data as unknown as { timestamp?: string }).timestamp || Date.now()}`;

    if (seenEventKeysRef.current.has(key)) {
      console.log('[Dedup] Skipping duplicate event:', key);
      return true;
    }

    seenEventKeysRef.current.add(key);
    return false;
  }, []);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
    setActiveSessionId(null);
  }, []);

  /**
   * Hydrate state from a persisted research session (for page reload).
   * This restores the research panel from data loaded via API.
   */
  const hydrateFromSession = useCallback((session: ResearchSession) => {
    // Only hydrate if we have plan data
    if (!session.plan?.steps) {
      console.log('[Hydrate] No plan steps to hydrate');
      return;
    }

    // Handle camelCase runtime keys for session-level properties
    const currentStepIdx = (session as unknown as { currentStepIndex?: number }).currentStepIndex ?? session.current_step_index;

    console.log('[Hydrate] Session:', {
      status: session.status,
      planSteps: session.plan.steps.length,
      stepsWithStatus: session.plan.steps.map(s => ({ title: s.title, status: s.status })),
      currentStepIndex: currentStepIdx,
    });

    // Convert persisted plan to UI plan format
    // Default to 'pending' if status is missing - safer than assuming 'completed'
    const plan: Plan = {
      title: session.plan.title,
      reasoning: session.plan.thought,
      steps: session.plan.steps.map((s, i) => ({
        index: i,
        title: s.title,
        description: s.description,
        status: (s.status as 'pending' | 'in_progress' | 'completed' | 'skipped') || 'pending',
      })),
    };

    setCurrentPlan(plan);
    setCurrentStepIndex(currentStepIdx ?? -1);

    // Set agent status based on session status
    if (session.status === 'completed') {
      setAgentStatus('complete');
    } else if (session.status === 'failed' || session.status === 'cancelled') {
      setAgentStatus('error');
    } else {
      setAgentStatus('idle');
    }
  }, []);

  /**
   * Process a single event and update state accordingly.
   * Used by both SSE handler and reconnection replay.
   * Handles deduplication internally.
   */
  const processExternalEvent = useCallback((data: StreamEvent) => {
    // Deduplication check (for reconnection scenarios)
    if (isDuplicateEvent(data)) {
      return; // Skip duplicate event
    }

    console.log('[External] Processing event:', data.event_type, data);

    // Add stable unique ID to each event for React keys
    const eventWithId = {
      ...data,
      _eventId: `${data.event_type}-${eventCounterRef.current++}-${Date.now()}`,
    } as StreamEvent;
    setEvents((prev) => [...prev, eventWithId]);

    // Process event based on type (same logic as SSE handler)
    switch (data.event_type) {
      case 'agent_started': {
        if ('agent' in data) {
          const agent = (data as { agent: string }).agent;
          setCurrentAgent(agent.charAt(0).toUpperCase() + agent.slice(1));
          if (agent === 'coordinator') setAgentStatus('classifying');
          else if (agent === 'planner') setAgentStatus('planning');
          else if (agent === 'researcher') setAgentStatus('researching');
          else if (agent === 'reflector') setAgentStatus('reflecting');
          else if (agent === 'synthesizer') setAgentStatus('synthesizing');
          else if (agent === 'verifier') setAgentStatus('verifying');
        }
        break;
      }

      case 'agent_completed': {
        setCurrentAgent(null);
        break;
      }

      case 'research_started': {
        const startedEvent = data as ResearchStartedEvent;
        const messageId = (startedEvent as unknown as { messageId?: string }).messageId ?? startedEvent.message_id;
        setAgentMessageId(messageId);
        break;
      }

      case 'plan_created': {
        setAgentStatus('researching');
        const planEvent = data as PlanCreatedEvent;
        setCurrentPlan({
          title: planEvent.title,
          reasoning: planEvent.thought,
          steps: planEvent.steps.map((s, i) => ({
            index: i,
            title: s.title,
            description: undefined,
            status: 'pending' as const,
          })),
        });
        break;
      }

      case 'step_started': {
        const stepEvent = data as StepStartedEvent;
        const stepIndex = (stepEvent as unknown as { stepIndex?: number }).stepIndex ?? stepEvent.step_index;
        setCurrentStepIndex(stepIndex);
        setCurrentPlan((prev) => {
          if (!prev) return prev;
          const steps = [...prev.steps];
          const step = steps[stepIndex];
          if (step) {
            steps[stepIndex] = { ...step, status: 'in_progress' };
          }
          return { ...prev, steps };
        });
        break;
      }

      case 'step_completed': {
        const stepEvent = data as StepCompletedEvent;
        const stepIndex = (stepEvent as unknown as { stepIndex?: number }).stepIndex ?? stepEvent.step_index;
        setCurrentPlan((prev) => {
          if (!prev) return prev;
          const steps = [...prev.steps];
          const step = steps[stepIndex];
          if (step) {
            steps[stepIndex] = { ...step, status: 'completed' };
          }
          return { ...prev, steps };
        });
        setCurrentStepIndex(-1);
        setToolActivity(null);
        break;
      }

      case 'tool_call': {
        const toolEvent = data as ToolCallEvent;
        const toolName = (toolEvent as unknown as { toolName?: string }).toolName ?? toolEvent.tool_name;
        const toolArgs = (toolEvent as unknown as { toolArgs?: Record<string, unknown> }).toolArgs ?? toolEvent.tool_args;
        const callNumber = (toolEvent as unknown as { callNumber?: number }).callNumber ?? toolEvent.call_number;
        setToolActivity({
          toolName: toolName as 'web_search' | 'web_crawl',
          toolArgs,
          callNumber,
          sourcesCrawled: 0,
        });
        break;
      }

      case 'tool_result': {
        const toolEvent = data as ToolResultEvent;
        const sourcesCrawled = (toolEvent as unknown as { sourcesCrawled?: number }).sourcesCrawled ?? toolEvent.sources_crawled;
        setToolActivity((prev) => prev ? { ...prev, toolName: null, sourcesCrawled } : null);
        break;
      }

      case 'reflection_decision': {
        const reflectionEvent = data as ReflectionDecisionEvent;
        setAgentStatus('reflecting');
        if (reflectionEvent.decision === 'complete') {
          setCurrentPlan((prev) => {
            if (!prev) return prev;
            const steps = prev.steps.map(step =>
              step.status === 'pending' ? { ...step, status: 'skipped' as const } : step
            );
            return { ...prev, steps };
          });
        }
        break;
      }

      case 'synthesis_started':
        setAgentStatus('synthesizing');
        break;

      case 'synthesis_progress': {
        const progressEvent = data as SynthesisProgressEvent;
        const contentChunk = (progressEvent as unknown as { contentChunk?: string }).contentChunk ?? progressEvent.content_chunk;
        setStreamingContent((prev) => prev + contentChunk);
        break;
      }

      case 'claim_verified': {
        const claimEvent = data as unknown as ClaimVerifiedEvent;
        setAgentStatus('verifying');
        setStreamingClaims((prev) => {
          const existingIndex = prev.findIndex(c => c.id === claimEvent.claimId);
          const newClaim: StreamingClaim = {
            id: claimEvent.claimId,
            claimText: claimEvent.claimText,
            positionStart: claimEvent.positionStart,
            positionEnd: claimEvent.positionEnd,
            verificationVerdict: claimEvent.verdict,
            confidenceLevel: claimEvent.confidenceLevel,
            evidencePreview: claimEvent.evidencePreview,
            reasoning: claimEvent.reasoning,
            // Citation keys for frontend citationData mapping
            citationKey: claimEvent.citationKey ?? null,
            citationKeys: claimEvent.citationKeys ?? null,
          };
          if (existingIndex >= 0) {
            const updated = [...prev];
            updated[existingIndex] = newClaim;
            return updated;
          }
          return [...prev, newClaim];
        });
        break;
      }

      case 'citation_corrected': {
        setCitationCorrectionCount((prev) => prev + 1);
        break;
      }

      case 'verification_summary': {
        const summaryEvent = data as unknown as VerificationSummaryEvent;
        setStreamingVerificationSummary({
          totalClaims: summaryEvent.totalClaims,
          supportedCount: summaryEvent.supported,
          partialCount: summaryEvent.partial,
          unsupportedCount: summaryEvent.unsupported,
          contradictedCount: summaryEvent.contradicted,
          abstainedCount: summaryEvent.abstainedCount,
          unsupportedRate: summaryEvent.totalClaims > 0 ? summaryEvent.unsupported / summaryEvent.totalClaims : 0,
          contradictedRate: summaryEvent.totalClaims > 0 ? summaryEvent.contradicted / summaryEvent.totalClaims : 0,
          warning: summaryEvent.warning,
        });
        break;
      }

      case 'content_revised': {
        const revisedEvent = data as { content?: string; revision_count?: number };
        const revisedContent = (revisedEvent as unknown as { content?: string }).content ?? '';
        if (revisedContent) {
          setStreamingContent(revisedContent);
        }
        break;
      }

      case 'research_completed':
        setAgentStatus('complete');
        // Note: Don't stop stream here - we're processing external events
        break;

      case 'error': {
        const errorEvent = data as StreamErrorEvent;
        // Handle both camelCase and snake_case for compatibility
        const errorMessage = (errorEvent as unknown as { errorMessage?: string }).errorMessage ?? errorEvent.error_message;
        const errorCode = (errorEvent as unknown as { errorCode?: string }).errorCode ?? errorEvent.error_code;
        const stackTrace = (errorEvent as unknown as { stackTrace?: string }).stackTrace ?? errorEvent.stack_trace;
        const errorType = (errorEvent as unknown as { errorType?: string }).errorType ?? errorEvent.error_type;
        const recoverable = errorEvent.recoverable;

        const err = new Error(errorMessage || 'Research failed');

        // Store full error details for display
        setErrorDetails({
          error: err,
          errorCode,
          stackTrace,
          errorType,
          recoverable,
        });

        if (!recoverable) {
          setError(err);
          setAgentStatus('error');
        }
        break;
      }
    }
  }, [isDuplicateEvent]);

  /**
   * Handle a single job event from the SSE stream.
   * Unwraps job event format and processes the event.
   */
  const handleJobEvent = useCallback((eventData: string): boolean => {
    try {
      const rawData = JSON.parse(eventData);

      // Handle job_completed event (final event from job stream)
      if (rawData.eventType === 'job_completed') {
        console.log('[SSE] Job completed:', rawData.status);
        if (rawData.status === 'completed') {
          setAgentStatus('complete');
          onStreamComplete?.();
        } else if (rawData.status === 'cancelled') {
          setAgentStatus('idle');
        } else if (rawData.status === 'failed') {
          setAgentStatus('error');
          const err = new Error('Research job failed');
          setError(err);
          setErrorDetails({ error: err, errorCode: 'JOB_FAILED', recoverable: false });
        }
        stopStream();
        return true; // Signal to close connection
      }

      // Update sequence number for reconnection
      if (rawData.sequenceNumber) {
        lastSequenceRef.current = rawData.sequenceNumber;
      }

      // Unwrap job event payload OR use direct event format
      let data: StreamEvent;
      if (rawData.payload && rawData.eventType) {
        // Job event format: unwrap payload and normalize event_type
        data = {
          ...rawData.payload,
          event_type: rawData.eventType,
          sequenceNumber: rawData.sequenceNumber,
        } as StreamEvent;
      } else if (rawData.event_type) {
        // Direct SSE format (old endpoint compatibility)
        data = parseStreamEvent(eventData) as StreamEvent;
        if (!data) return false;
      } else {
        // Try parsing with schema validator
        data = parseStreamEvent(eventData) as StreamEvent;
        if (!data) return false;
      }

      // Process the unwrapped event
      processExternalEvent(data);
      return false;
    } catch (err) {
      console.error('[SSE] Error processing job event:', err);
      return false;
    }
  }, [stopStream, onStreamComplete, processExternalEvent]);

  /**
   * Handle SSE connection errors with retry logic.
   */
  const handleSseError = useCallback((e: Event, eventSource: EventSource) => {
    const now = Date.now();
    const timeSinceLastError = now - lastSseErrorTimeRef.current;

    // Reset error count if it's been more than 5 seconds since last error
    if (timeSinceLastError > 5000) {
      sseErrorCountRef.current = 0;
    }

    sseErrorCountRef.current++;
    lastSseErrorTimeRef.current = now;

    console.error('[SSE] ERROR - Connection issue detected:', {
      error: e,
      errorCount: sseErrorCountRef.current,
      timeSinceLastError,
      readyState: eventSource.readyState
    });

    // EventSource auto-reconnects, so only give up after multiple rapid errors
    if (sseErrorCountRef.current >= 3 || eventSource.readyState === 2) {
      console.log('[SSE] ERROR - Multiple errors or connection closed, stopping stream');
      const err = new Error('Stream connection failed');
      setError(err);
      setErrorDetails({ error: err, errorCode: 'CONNECTION_FAILED', recoverable: true });
      setAgentStatus('error');
      stopStream();
    } else {
      console.log('[SSE] ERROR - Waiting for auto-reconnect (error count:', sseErrorCountRef.current, ')');
    }
  }, [stopStream]);

  const sendQuery = useCallback(
    async (query: string, queryMode?: QueryMode, researchDepth?: string, verifySources?: boolean) => {
      if (!chatId) {
        console.error('No chat ID provided');
        return;
      }

      // Close any existing connection
      stopStream();

      // Store current query for later
      currentQueryRef.current = query;

      // Reset state
      setEvents([]);
      setStreamingContent('');
      setError(null);
      setErrorDetails(null);
      // For simple mode, skip to synthesizing status immediately
      setAgentStatus(queryMode === 'simple' ? 'synthesizing' : 'classifying');
      setCurrentPlan(null);
      setCurrentStepIndex(-1);
      setIsStreaming(true);

      // Reset citation state
      setStreamingClaims([]);
      setStreamingVerificationSummary(null);
      setCitationCorrectionCount(0);
      setAgentMessageId(null);
      setToolActivity(null);

      // Reset persistence state
      setPersistenceResult(null);
      setPersistenceFailed(false);

      // Reset streaming time and agent tracking
      setStartTime(Date.now());
      setCurrentAgent(null);
      eventCounterRef.current = 0;
      sseErrorCountRef.current = 0;
      lastSseErrorTimeRef.current = 0;
      // Reset deduplication state for new query
      seenEventKeysRef.current.clear();
      // Reset sequence tracking for reconnection
      lastSequenceRef.current = 0;
      setActiveSessionId(null);

      // Store query mode for the session (used for activity panel visibility)
      console.log('[useStreamingQuery] sendQuery: Setting currentQueryMode to:', queryMode || 'simple');
      setCurrentQueryMode(queryMode || 'simple');

      // =========================================================================
      // NEW JOB-BASED FLOW: Submit job first, then connect to event stream
      // This fixes the duplicate research bug caused by SSE auto-reconnect
      // =========================================================================
      let sessionId: string;
      try {
        console.log('[useStreamingQuery] Submitting job via jobs API');
        const job = await jobsApi.submit({
          chatId,
          query,
          queryMode: queryMode || 'simple',
          researchDepth: researchDepth || 'auto',
          verifySources: verifySources ?? (queryMode === 'deep_research'),
        });
        sessionId = job.sessionId;
        setActiveSessionId(sessionId);
        console.log('[useStreamingQuery] Job submitted, sessionId:', sessionId);
      } catch (err) {
        console.error('[useStreamingQuery] Job submission failed:', err);
        const errorMessage = err instanceof Error ? err.message : 'Failed to submit research job';
        // Handle specific error types
        let submissionError: Error;
        let errorCode: string;
        if (errorMessage.includes('429') || errorMessage.includes('concurrent')) {
          submissionError = new Error('Maximum concurrent jobs reached. Please wait for a running job to complete.');
          errorCode = 'MAX_CONCURRENT_JOBS';
        } else if (errorMessage.includes('409') || errorMessage.includes('research_in_progress')) {
          submissionError = new Error('Research is already in progress for this chat. Please wait for it to complete or cancel it.');
          errorCode = 'RESEARCH_IN_PROGRESS';
        } else {
          submissionError = new Error(errorMessage);
          errorCode = 'SUBMISSION_FAILED';
        }
        setError(submissionError);
        setErrorDetails({ error: submissionError, errorCode, recoverable: true });
        setAgentStatus('error');
        setIsStreaming(false);
        return;
      }

      // Connect to job event stream (with sinceSequence=0 for new job)
      const streamUrl = jobsApi.streamUrl(sessionId, 0);
      console.log('[useStreamingQuery] Connecting to job stream:', streamUrl);

      const eventSource = new EventSource(streamUrl);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        // Use job event handler that unwraps the job event format
        handleJobEvent(event.data);
      };

      eventSource.onerror = (e) => handleSseError(e, eventSource);
    },
    [chatId, stopStream, handleJobEvent, handleSseError]
  );

  /**
   * Reconnect to an existing job's event stream.
   * Used when page reloads and there's an active job.
   */
  const reconnectToJob = useCallback(
    async (sessionId: string) => {
      console.log('[useStreamingQuery] Reconnecting to job:', sessionId);

      // Close any existing connection
      stopStream();

      // Fetch job to verify it's still in progress
      try {
        const job = await jobsApi.get(sessionId);
        if (job.status !== 'in_progress') {
          console.log('[useStreamingQuery] Job not in progress, skipping reconnection. Status:', job.status);
          return;
        }

        // Restore query mode from the job
        if (job.queryMode) {
          console.log('[useStreamingQuery] Restoring query mode from job:', job.queryMode);
          setCurrentQueryMode(job.queryMode as QueryMode);
        }
      } catch (err) {
        console.error('[useStreamingQuery] Failed to fetch job details:', err);
        return;
      }

      // Set job state
      setActiveSessionId(sessionId);
      setIsStreaming(true);
      setAgentStatus('researching');
      setStartTime(Date.now());

      // Clear deduplication state for fresh replay
      seenEventKeysRef.current.clear();
      eventCounterRef.current = 0;
      lastSequenceRef.current = 0;

      // Connect to job event stream (from beginning to replay all events)
      const streamUrl = jobsApi.streamUrl(sessionId, 0);
      console.log('[useStreamingQuery] Connecting to job stream for reconnection:', streamUrl);

      const eventSource = new EventSource(streamUrl);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        handleJobEvent(event.data);
      };

      eventSource.onerror = (e) => handleSseError(e, eventSource);
    },
    [stopStream, handleJobEvent, handleSseError]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('[useStreamingQuery] Cleanup on unmount');
      stopStream();
    };
  }, [stopStream]);

  // Reset when chat changes (but NOT for draft→real navigation with same chatId)
  useEffect(() => {
    console.log('[useStreamingQuery] Chat change effect triggered:', {
      prevChatId: prevChatIdRef.current,
      newChatId: chatId,
      willReset: prevChatIdRef.current !== chatId,
    });

    // Only reset if chatId actually changed to a DIFFERENT value
    // This prevents state reset when URL changes but chatId stays the same (draft→real)
    if (prevChatIdRef.current === chatId) {
      console.log('[useStreamingQuery] SKIPPING reset - same chatId');
      return; // Same chatId, don't reset - prevents flickering during navigation
    }

    console.log('[useStreamingQuery] RESETTING ALL STATE - chatId changed from', prevChatIdRef.current, 'to', chatId);
    prevChatIdRef.current = chatId;

    // CRITICAL: Close any open EventSource BEFORE resetting state!
    // This prevents events from old chat polluting new chat's state
    console.log('[useStreamingQuery] Stopping stream before chat change');
    stopStream();

    // Reset all state for new chat
    setEvents([]);
    setStreamingContent('');
    setAgentStatus('idle');
    setCurrentPlan(null);
    setCurrentStepIndex(-1);
    setError(null);
    setErrorDetails(null);
    setCompletedMessages([]);
    setStreamingClaims([]);
    setStreamingVerificationSummary(null);
    setCitationCorrectionCount(0);
    setAgentMessageId(null);
    setToolActivity(null);
    setPersistenceResult(null);
    setPersistenceFailed(false);
    // Reset streaming time and agent tracking
    setStartTime(null);
    setCurrentAgent(null);
    eventCounterRef.current = 0;
    // Reset deduplication state when switching chats
    seenEventKeysRef.current.clear();
    // Reset query mode when switching chats
    console.log('[useStreamingQuery] Resetting currentQueryMode to null');
    setCurrentQueryMode(null);
  }, [chatId, stopStream]);

  return {
    isStreaming,
    events,
    streamingContent,
    agentStatus,
    currentPlan,
    currentStepIndex,
    sendQuery,
    stopStream,
    error,
    errorDetails,
    clearErrorDetails,
    completedMessages,
    streamingClaims,
    streamingVerificationSummary,
    citationCorrectionCount,
    agentMessageId,
    toolActivity,
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
    activeSessionId,
    reconnectToJob,
  };
}
