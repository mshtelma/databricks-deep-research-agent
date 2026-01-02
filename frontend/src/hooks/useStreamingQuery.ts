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
} from '../types';
import type {
  VerificationSummary,
  VerificationVerdict,
  ConfidenceLevel,
} from '../types/citation';

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
  sendQuery: (query: string, researchDepth?: string) => void;
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

  const eventSourceRef = useRef<EventSource | null>(null);

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

  const sendQuery = useCallback(
    (query: string, researchDepth?: string) => {
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
      setAgentStatus('classifying');
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

      // Build stream URL with query and research_depth parameters
      let streamUrl = `${API_BASE_URL}/chats/${chatId}/stream?query=${encodeURIComponent(query)}`;
      if (researchDepth && researchDepth !== 'auto') {
        streamUrl += `&research_depth=${encodeURIComponent(researchDepth)}`;
      }

      const eventSource = new EventSource(streamUrl);
      eventSourceRef.current = eventSource;

      // Accumulate content for final message
      let accumulatedContent = '';

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StreamEvent;
          console.log('[SSE] Event received:', data.event_type, data);
          setEvents((prev) => [...prev, data]);

          // Update state based on event type
          switch (data.event_type) {
            case 'agent_started':
              // Update status based on which agent started
              if ('agent' in data) {
                const agent = (data as { agent: string }).agent;
                if (agent === 'coordinator') setAgentStatus('classifying');
                else if (agent === 'planner') setAgentStatus('planning');
                else if (agent === 'researcher') setAgentStatus('researching');
                else if (agent === 'reflector') setAgentStatus('reflecting');
                else if (agent === 'synthesizer') setAgentStatus('synthesizing');
                else if (agent === 'verifier') setAgentStatus('verifying');
              }
              break;

            case 'agent_completed': {
              // Agent completed, continue with flow
              const completedEvent = data as { agent: string; duration_ms?: number };
              console.log('[SSE] agent_completed:', {
                agent: completedEvent.agent,
                duration_ms: completedEvent.duration_ms,
                hasValidDuration: typeof completedEvent.duration_ms === 'number',
              });
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
              accumulatedContent += contentChunk;
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

            case 'research_completed':
              setAgentStatus('complete');

              // Add this exchange to completed messages
              if (currentQueryRef.current && accumulatedContent) {
                setCompletedMessages(prev => [
                  ...prev,
                  { role: 'user' as const, content: currentQueryRef.current },
                  { role: 'assistant' as const, content: accumulatedContent }
                ]);
              }

              stopStream();

              // Notify parent that streaming is complete - allows message refetch
              // This enables citation rendering after persistence
              onStreamComplete?.();

              // Clear streaming content after a short delay to allow message refetch
              // This prevents duplicate rendering of streaming placeholder and persisted message
              setTimeout(() => {
                setStreamingContent('');
              }, 150);
              break;

            case 'persistence_completed': {
              // Database persistence succeeded - draft chat is now real
              const persistEvent = data as PersistenceCompletedEvent;
              setPersistenceResult(persistEvent);
              setPersistenceFailed(false);
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
        } catch (e) {
          console.error('Failed to parse SSE event:', e);
          setError(e instanceof Error ? e : new Error('Failed to parse SSE event'));
          setAgentStatus('error');
          stopStream();
        }
      };

      eventSource.onerror = (e) => {
        console.error('SSE error:', e);
        const err = new Error('Stream connection failed');
        setError(err);
        setAgentStatus('error');
        stopStream();
      };
    },
    [chatId, stopStream]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, [stopStream]);

  // Reset when chat changes
  useEffect(() => {
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
  }, [chatId]);

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
  };
}
