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

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

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
interface StreamingClaim {
  id: string;
  claimText: string;
  positionStart: number;
  positionEnd: number;
  verificationVerdict: VerificationVerdict | null;
  confidenceLevel: ConfidenceLevel | null;
  evidencePreview: string;
  reasoning: string | null;
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

interface UseStreamingQueryReturn {
  isStreaming: boolean;
  events: StreamEvent[];
  streamingContent: string;
  agentStatus: AgentStatus;
  currentPlan: Plan | null;
  currentStepIndex: number;
  sendQuery: (query: string, queryMode?: QueryMode, researchDepth?: string, verifySources?: boolean) => void;
  stopStream: () => void;
  error: Error | null;
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
        const errorMessage = (errorEvent as unknown as { errorMessage?: string }).errorMessage ?? errorEvent.error_message;
        if (!errorEvent.recoverable) {
          setError(new Error(errorMessage || 'Research failed'));
          setAgentStatus('error');
        }
        break;
      }
    }
  }, [isDuplicateEvent]);

  const sendQuery = useCallback(
    (query: string, queryMode?: QueryMode, researchDepth?: string, verifySources?: boolean) => {
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

      // Store query mode for the session (used for activity panel visibility)
      console.log('[useStreamingQuery] sendQuery: Setting currentQueryMode to:', queryMode || 'simple');
      setCurrentQueryMode(queryMode || 'simple');

      // Build stream URL with query, query_mode, and research_depth parameters
      let streamUrl = `${API_BASE_URL}/chats/${chatId}/stream?query=${encodeURIComponent(query)}`;
      if (queryMode) {
        streamUrl += `&query_mode=${encodeURIComponent(queryMode)}`;
      }
      if (researchDepth && researchDepth !== 'auto') {
        streamUrl += `&research_depth=${encodeURIComponent(researchDepth)}`;
      }
      if (verifySources !== undefined) {
        streamUrl += `&verify_sources=${verifySources}`;
      }

      const eventSource = new EventSource(streamUrl);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        // Use Zod validation for safe parsing with graceful degradation
        const data = parseStreamEvent(event.data);

        if (!data) {
          // Malformed event - already logged by parseStreamEvent, skip processing
          return;
        }

        // Deduplication check (for reconnection scenarios)
        if (isDuplicateEvent(data as StreamEvent)) {
          return; // Skip duplicate event
        }

        console.log('[SSE] Event received:', data.event_type, data);

        // Add stable unique ID to each event for React keys (prevents blinking)
        const eventWithId = {
          ...data,
          _eventId: `${data.event_type}-${eventCounterRef.current++}-${Date.now()}`,
        } as StreamEvent;
        setEvents((prev) => [...prev, eventWithId]);

        // Update state based on event type
        switch (data.event_type) {
            case 'agent_started': {
              // Update status based on which agent started
              if ('agent' in data) {
                const agent = (data as { agent: string }).agent;
                // Track current agent for activity panel display
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
              // Agent completed, continue with flow
              const completedEvent = data as { agent: string; duration_ms?: number };
              console.log('[SSE] agent_completed:', {
                agent: completedEvent.agent,
                duration_ms: completedEvent.duration_ms,
                hasValidDuration: typeof completedEvent.duration_ms === 'number',
              });
              // Clear current agent (next agent_started will set new one)
              setCurrentAgent(null);
              break;
            }

            case 'research_started': {
              // Capture the real agent message UUID from backend for citation fetching
              const startedEvent = data as ResearchStartedEvent;
              // Handle camelCase runtime keys
              const messageId = (startedEvent as unknown as { messageId?: string }).messageId ?? startedEvent.message_id;
              console.log('[SSE] research_started:', { messageId });
              setAgentMessageId(messageId);
              break;
            }

            case 'plan_created': {
              setAgentStatus('researching');
              const planEvent = data as PlanCreatedEvent;
              console.log('[SSE] plan_created:', {
                title: planEvent.title,
                steps: planEvent.steps.length,
                stepTitles: planEvent.steps.map(s => s.title),
              });
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
              // Handle camelCase runtime keys (stepIndex instead of step_index)
              const stepIndex = (stepEvent as unknown as { stepIndex?: number }).stepIndex ?? stepEvent.step_index;
              const stepTitle = (stepEvent as unknown as { stepTitle?: string }).stepTitle ?? stepEvent.step_title;
              console.log('[SSE] step_started:', { stepIndex, stepTitle });
              setCurrentStepIndex(stepIndex);
              setCurrentPlan((prev) => {
                if (!prev) return prev;
                const steps = [...prev.steps];
                const step = steps[stepIndex];
                if (step) {
                  steps[stepIndex] = {
                    ...step,
                    status: 'in_progress',
                  };
                }
                return { ...prev, steps };
              });
              break;
            }

            case 'step_completed': {
              const stepEvent = data as StepCompletedEvent;
              // Handle camelCase runtime keys
              const stepIndex = (stepEvent as unknown as { stepIndex?: number }).stepIndex ?? stepEvent.step_index;
              const sourcesFound = (stepEvent as unknown as { sourcesFound?: number }).sourcesFound ?? stepEvent.sources_found;
              console.log('[SSE] step_completed:', { stepIndex, sourcesFound });
              setCurrentPlan((prev) => {
                if (!prev) return prev;
                const steps = [...prev.steps];
                const step = steps[stepIndex];
                if (step) {
                  steps[stepIndex] = {
                    ...step,
                    status: 'completed',
                  };
                  console.log('[State] Step marked completed:', {
                    stepIndex,
                    stepTitle: step.title,
                    newStatus: 'completed',
                  });
                }
                return { ...prev, steps };
              });
              // FIX: Clear currentStepIndex so completed step is no longer "active"
              // The next step_started event will set it to the correct value
              setCurrentStepIndex(-1);
              // Clear tool activity when step completes
              setToolActivity(null);
              break;
            }

            case 'tool_call': {
              const toolEvent = data as ToolCallEvent;
              // Handle camelCase runtime keys
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
              // Handle camelCase runtime keys
              const sourcesCrawled = (toolEvent as unknown as { sourcesCrawled?: number }).sourcesCrawled ?? toolEvent.sources_crawled;
              setToolActivity((prev) => prev ? {
                ...prev,
                toolName: null, // Clear active tool
                sourcesCrawled,
              } : null);
              break;
            }

            case 'reflection_decision': {
              const reflectionEvent = data as ReflectionDecisionEvent;
              setAgentStatus('reflecting');

              // If decision is 'complete', mark all remaining steps as 'skipped'
              if (reflectionEvent.decision === 'complete') {
                setCurrentPlan((prev) => {
                  if (!prev) return prev;
                  const steps = prev.steps.map(step =>
                    step.status === 'pending'
                      ? { ...step, status: 'skipped' as const }
                      : step
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
              // Handle camelCase runtime keys
              const contentChunk = (progressEvent as unknown as { contentChunk?: string }).contentChunk ?? progressEvent.content_chunk;
              setStreamingContent((prev) => prev + contentChunk);
              break;
            }

            // Citation verification events
            case 'claim_verified': {
              const claimEvent = data as unknown as ClaimVerifiedEvent;
              setAgentStatus('verifying');
              setStreamingClaims((prev) => {
                // Check if claim already exists (update) or is new (add)
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
              // Citation was corrected - increment counter
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
                unsupportedRate: summaryEvent.totalClaims > 0
                  ? summaryEvent.unsupported / summaryEvent.totalClaims
                  : 0,
                contradictedRate: summaryEvent.totalClaims > 0
                  ? summaryEvent.contradicted / summaryEvent.totalClaims
                  : 0,
                warning: summaryEvent.warning,
              });
              break;
            }

            case 'content_revised': {
              // Stage 7 has revised the content with softening for partial/unsupported claims
              // Replace the streamed content with the revised version
              const revisedEvent = data as { content?: string; revision_count?: number };
              // Handle camelCase runtime keys
              const revisedContent = (revisedEvent as unknown as { content?: string }).content ?? '';
              const revisionCount = (revisedEvent as unknown as { revisionCount?: number }).revisionCount ?? revisedEvent.revision_count ?? 0;
              console.log('[SSE] content_revised:', { contentLen: revisedContent.length, revisionCount });
              if (revisedContent) {
                setStreamingContent(revisedContent);
              }
              break;
            }

            case 'research_completed':
              setAgentStatus('complete');
              // DON'T add to completedMessages here - content may be revised after this event
              // The streamingContent will continue showing until persistence_completed triggers refetch
              stopStream();
              // DON'T call onStreamComplete here - wait for persistence_completed
              // to ensure we get the final message with all revisions/citations
              break;

            case 'persistence_completed': {
              // Database persistence succeeded - draft chat is now real
              const persistEvent = data as PersistenceCompletedEvent;
              setPersistenceResult(persistEvent);
              setPersistenceFailed(false);

              // Trigger refetch - streamingContent will naturally be hidden
              // once agent message appears in apiMessages (MessageList.tsx line 111)
              onStreamComplete?.();

              // DON'T clear streamingContent here - it acts as a fallback
              // until the refetch completes. The MessageList rendering logic
              // already hides it when agent messages exist in the array.
              // Content is cleared on: new query start (line 510) or chat change (line 932)
              break;
            }

          case 'error': {
            const errorEvent = data as StreamErrorEvent;
            // Handle camelCase runtime keys
            const errorMessage = (errorEvent as unknown as { errorMessage?: string }).errorMessage ?? errorEvent.error_message;
            console.log('[SSE] error:', { errorMessage, recoverable: errorEvent.recoverable });
            if (!errorEvent.recoverable) {
              const err = new Error(errorMessage || 'Research failed');
              setError(err);
              setAgentStatus('error');
              stopStream();
            }
            break;
          }
        }
        // Note: Removed try-catch - parseStreamEvent handles JSON parse errors with graceful degradation
      };

      eventSource.onerror = (e) => {
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
          readyState: eventSource.readyState // 0=CONNECTING, 1=OPEN, 2=CLOSED
        });

        // EventSource auto-reconnects, so only give up after multiple rapid errors
        // or if the connection is definitively closed
        if (sseErrorCountRef.current >= 3 || eventSource.readyState === 2) {
          console.log('[SSE] ERROR - Multiple errors or connection closed, stopping stream');
          const err = new Error('Stream connection failed');
          setError(err);
          setAgentStatus('error');
          stopStream();
        } else {
          console.log('[SSE] ERROR - Waiting for auto-reconnect (error count:', sseErrorCountRef.current, ')');
        }
      };
    },
    [chatId, stopStream, onStreamComplete, isDuplicateEvent]
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
  };
}
