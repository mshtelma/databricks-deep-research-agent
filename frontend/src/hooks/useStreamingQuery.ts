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

interface UseStreamingQueryReturn {
  isStreaming: boolean;
  events: StreamEvent[];
  streamingContent: string;
  agentStatus: AgentStatus;
  currentPlan: Plan | null;
  currentStepIndex: number;
  sendQuery: (query: string) => void;
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

  const eventSourceRef = useRef<EventSource | null>(null);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const sendQuery = useCallback(
    (query: string) => {
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

      // Build stream URL with query parameter
      // The backend will load history from DB, or we can pass it via POST
      const streamUrl = `${API_BASE_URL}/chats/${chatId}/stream?query=${encodeURIComponent(query)}`;

      const eventSource = new EventSource(streamUrl);
      eventSourceRef.current = eventSource;

      // Accumulate content for final message
      let accumulatedContent = '';

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StreamEvent;
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

            case 'agent_completed':
              // Agent completed, continue with flow
              break;

            case 'research_started': {
              // Capture the real agent message UUID from backend for citation fetching
              const startedEvent = data as ResearchStartedEvent;
              setAgentMessageId(startedEvent.message_id);
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
              setCurrentStepIndex(stepEvent.step_index);
              setCurrentPlan((prev) => {
                if (!prev) return prev;
                const steps = [...prev.steps];
                const step = steps[stepEvent.step_index];
                if (step) {
                  steps[stepEvent.step_index] = {
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
              setCurrentPlan((prev) => {
                if (!prev) return prev;
                const steps = [...prev.steps];
                const step = steps[stepEvent.step_index];
                if (step) {
                  steps[stepEvent.step_index] = {
                    ...step,
                    status: 'completed',
                  };
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
              setToolActivity({
                toolName: toolEvent.tool_name as 'web_search' | 'web_crawl',
                toolArgs: toolEvent.tool_args,
                callNumber: toolEvent.call_number,
                sourcesCrawled: 0,
              });
              break;
            }

            case 'tool_result': {
              const toolEvent = data as ToolResultEvent;
              setToolActivity((prev) => prev ? {
                ...prev,
                toolName: null, // Clear active tool
                sourcesCrawled: toolEvent.sources_crawled,
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
              accumulatedContent += progressEvent.content_chunk;
              setStreamingContent((prev) => prev + progressEvent.content_chunk);
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

            case 'error': {
              const errorEvent = data as StreamErrorEvent;
              if (!errorEvent.recoverable) {
                const err = new Error(errorEvent.error_message || 'Research failed');
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
  };
}
