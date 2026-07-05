// API Types

export type ChatStatus = 'active' | 'archived' | 'deleted'
export type ChatType = 'regular' | 'incognito'
export type MessageRole = 'user' | 'agent' | 'system'
export type ResearchDepth = 'auto' | 'light' | 'medium' | 'extended'
export type QueryMode = 'simple' | 'web_search' | 'deep_research'
export type ResearchStatus =
  | 'pending'
  | 'classifying'
  | 'clarifying'
  | 'planning'
  | 'researching'
  | 'reflecting'
  | 'synthesizing'
  | 'completed'
  | 'cancelled'
  | 'failed'

export interface Chat {
  id: string
  title: string | null
  status: ChatStatus
  chatType: ChatType
  createdAt: string
  updatedAt: string
  messageCount: number
}

// User Profile Types
export interface UserProfile {
  userId: string
  email: string
  displayName: string
  workspace: string | null
}

// Incognito Session Types
export interface IncognitoSessionStatus {
  hasSession: boolean
  chatCount: number
  maxChats: number
  expiresAt: string | null
}

export interface Message {
  id: string
  chatId: string
  role: MessageRole
  content: string
  createdAt: string
  isEdited: boolean
  researchSession?: ResearchSession | null
}

export type SourceType = 'web' | 'vector_search' | 'knowledge_assistant' | 'custom'

export interface Source {
  id: string
  url: string
  title: string | null
  snippet: string | null
  relevanceScore: number | null
  sourceType: SourceType
  sourceMetadata: Record<string, unknown> | null
  isCited?: boolean
}

export interface QueryClassification {
  complexity: 'simple' | 'moderate' | 'complex'
  followUpType: 'new_topic' | 'clarification' | 'complex_follow_up'
  isAmbiguous: boolean
  clarifyingQuestions: string[]
  reasoning: string
}

export interface PlanStep {
  id: string
  title: string
  description: string
  stepType: 'research' | 'analysis'
  needsSearch: boolean
  status: 'pending' | 'in_progress' | 'completed' | 'skipped'
  observation: string | null
}

export interface ResearchPlan {
  id: string
  title: string
  thought: string
  steps: PlanStep[]
  iteration: number
  createdAt: string
}

export interface ResearchSession {
  id: string
  queryClassification: QueryClassification | null
  researchDepth: ResearchDepth
  status: ResearchStatus
  currentAgent: string | null
  plan: ResearchPlan | null
  currentStepIndex: number | null
  planIterations: number
  startedAt: string
  completedAt: string | null
  sources: Source[]
}

export interface UserPreferences {
  systemInstructions: string | null
  defaultDepth: ResearchDepth
  defaultQueryMode: QueryMode
  uiPreferences: Record<string, unknown>
  updatedAt: string
}

// API Request/Response Types

export interface CreateChatRequest {
  title?: string
}

export interface UpdateChatRequest {
  title?: string
  status?: ChatStatus
}

export interface SendMessageRequest {
  content: string
  researchDepth?: ResearchDepth
}

export interface SendMessageResponse {
  userMessage: Message
  agentMessageId: string
  researchSessionId: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  limit: number
  offset: number
}

// Streaming Event Types

export type StreamEventType =
  | 'agent_started'
  | 'agent_completed'
  | 'research_started'
  | 'clarification_needed'
  | 'plan_created'
  | 'step_started'
  | 'step_completed'
  | 'tool_call'
  | 'tool_result'
  | 'reflection_decision'
  | 'synthesis_started'
  | 'synthesis_progress'
  | 'research_completed'
  | 'error'
  // Citation verification events
  | 'claim_generated'
  | 'claim_verified'
  | 'citation_corrected'
  | 'numeric_claim_detected'
  | 'verification_summary'
  // Stage 7 content revision event
  | 'content_revised'
  // Persistence events
  | 'persistence_completed'
  // Plan review event
  | 'plan_review'
  // Custom research phases (e.g. structured-output structuring pass)
  | 'phase_started'
  | 'phase_completed'
  | 'phase_error'

/** Custom research phase events (e.g. the structured-output structuring pass). */
export interface PhaseStartedEvent extends BaseStreamEvent {
  eventType: 'phase_started'
  phaseName?: string
  phase_name?: string
  description?: string
}

export interface PhaseCompletedEvent extends BaseStreamEvent {
  eventType: 'phase_completed'
  phaseName?: string
  phase_name?: string
  durationMs?: number
  duration_ms?: number
}

export interface PhaseErrorEvent extends BaseStreamEvent {
  eventType: 'phase_error'
  phaseName?: string
  phase_name?: string
  error?: string
  recoverable?: boolean
}

export interface BaseStreamEvent {
  eventType: StreamEventType
  timestamp: string
  /** Stable unique ID for React keys (added by useStreamingQuery) */
  _eventId?: string
}

export interface AgentStartedEvent extends BaseStreamEvent {
  eventType: 'agent_started'
  agent: string
  modelTier: string
}

export interface AgentCompletedEvent extends BaseStreamEvent {
  eventType: 'agent_completed'
  agent: string
  durationMs: number
}

export interface ResearchStartedEvent extends BaseStreamEvent {
  eventType: 'research_started'
  messageId: string
  researchSessionId: string
}

export interface ClarificationNeededEvent extends BaseStreamEvent {
  eventType: 'clarification_needed'
  questions: string[]
  round: number
}

export interface PlanCreatedEvent extends BaseStreamEvent {
  eventType: 'plan_created'
  planId: string
  title: string
  thought: string
  steps: { id: string; title: string; stepType: string; needsSearch: boolean }[]
  iteration: number
}

export interface StepStartedEvent extends BaseStreamEvent {
  eventType: 'step_started'
  stepIndex: number
  stepId: string
  stepTitle: string
  stepType: string
}

export interface StepCompletedEvent extends BaseStreamEvent {
  eventType: 'step_completed'
  stepIndex: number
  stepId: string
  observationSummary: string
  sourcesFound: number
  fileSourcesFound?: number
}

export interface ToolCallEvent extends BaseStreamEvent {
  eventType: 'tool_call'
  toolName: string // 'web_search' | 'web_crawl' | enterprise tool names
  toolArgs: Record<string, unknown>
  callNumber: number
  sourceType?: string // genie, vector_search, knowledge_assistant, web_search, web_crawl
}

export interface ToolResultEvent extends BaseStreamEvent {
  eventType: 'tool_result'
  toolName: string
  resultPreview: string
  sourcesCrawled: number
  sourcesAdded?: number
  sourceType?: string // genie, vector_search, knowledge_assistant, web_search, web_crawl
}

export interface ReflectionDecisionEvent extends BaseStreamEvent {
  eventType: 'reflection_decision'
  decision: 'continue' | 'adjust' | 'complete'
  reasoning: string
  suggestedChanges: string[] | null
}

export interface SynthesisStartedEvent extends BaseStreamEvent {
  eventType: 'synthesis_started'
  totalObservations: number
  totalSources: number
}

export interface SynthesisProgressEvent extends BaseStreamEvent {
  eventType: 'synthesis_progress'
  contentChunk: string
}

export interface ResearchCompletedEvent extends BaseStreamEvent {
  eventType: 'research_completed'
  sessionId: string
  totalStepsExecuted: number
  totalStepsSkipped: number
  planIterations: number
  totalDurationMs: number
  // Content fields for structured output types (both snake_case and camelCase for compatibility)
  final_report?: string | null
  finalReport?: string | null
  structured_output?: Record<string, unknown> | null
  structuredOutput?: Record<string, unknown> | null
}

export interface StreamErrorEvent extends BaseStreamEvent {
  eventType: 'error'
  errorCode: string
  errorMessage: string
  recoverable: boolean
  /** Full Python traceback for debugging */
  stackTrace?: string
  /** Exception class name (e.g., "ValueError") */
  errorType?: string
}

export interface PersistenceCompletedEvent extends BaseStreamEvent {
  eventType: 'persistence_completed'
  chatId: string
  messageId: string
  researchSessionId: string
  chatTitle: string
  wasDraft: boolean
  counts: Record<string, number>
}

// Plan review event - sent when enable_plan_review is true
export interface PlanReviewEvent extends BaseStreamEvent {
  eventType: 'plan_review'
  plan: unknown
  timeoutSeconds?: number
  reviewId?: string
}

// Stage 7 content revision event - sent after verification retrieval applies softening
export interface ContentRevisedEvent extends BaseStreamEvent {
  eventType: 'content_revised'
  content: string
  revisionCount: number
}

// Import citation types for ChatFullResponse
import type { Claim, VerificationSummary } from './citation';

/** One evidence source in the envelope legend; source_refs index into these. */
export interface StructuredSourceRef {
  ref: string;
  url: string;
  title?: string | null;
}

/** Per-slot fill status (v2 native per-slot wire generation). */
export type SlotStatus = 'ok' | 'failed' | 'pending' | 'empty';

export interface SlotMeta {
  status: SlotStatus;
  error?: string;
  attempts?: number;
  duration_ms?: number;
  dropped_unsourced?: number;
}

/**
 * Agent-surface structured-output envelope (per-slot wire generation).
 *
 * Stored verbatim in `verification_data["structured_output"]` and passed
 * through the API as a raw `dict`, so — unlike the rest of this module — its
 * keys are SNAKE_CASE on the wire (the same convention as the surface JSON in
 * `types/surface.ts`). The `data` payload carries arbitrary user-defined
 * column keys plus each item's `source_refs`, and is never camelized.
 */
export interface StructuredOutputEnvelope {
  version: number;
  /** The surface binding action whose slots this payload fills. */
  binding: string;
  /** Owning agent id — the restructure endpoint reloads the surface from it. */
  agent_id?: string | null;
  surface_etag?: string | null;
  generated_at?: string;
  /** Slot name → payload (table rows / metric cards / string items). */
  data: Record<string, unknown>;
  meta?: {
    /** Per-slot state machine: pending → ok/empty/failed. */
    slots?: Record<string, SlotMeta>;
    /** Legend resolving item source_refs to their URL/title. */
    sources?: StructuredSourceRef[];
    warnings?: { code: string; message: string; slot?: string }[];
    stripped_citation_keys?: string[];
    truncated_slots?: string[];
    attempts?: number;
    duration_ms?: number;
    model_tier?: string;
    evidence?: string;
  };
}

/** Message with inline claims (from /chats/{id}/full endpoint) */
export interface FullMessage {
  id: string;
  chatId: string;
  role: 'user' | 'agent';
  content: string;
  createdAt: string;
  isEdited: boolean;
  researchSession: ResearchSession | null;
  claims: Claim[];
  verificationSummary: VerificationSummary | null;
  structuredOutput?: StructuredOutputEnvelope | null;
}

/** Per-action run reference persisted in surface_state.action_runs. */
export interface PersistedActionRun {
  session_id?: string;
  message_id?: string;
  status?: string;
  updated_at?: string;
}

/** Per-agent surface state entry persisted server-side. */
export interface SurfaceStateEntry {
  data_model?: Record<string, unknown>;
  action_runs?: Record<string, PersistedActionRun>;
  surface_etag?: string | null;
}

/** Complete chat payload from GET /chats/{id}/full */
export interface ChatFullResponse {
  id: string;
  title: string | null;
  status: ChatStatus;
  chatType: string;
  createdAt: string;
  updatedAt: string;
  messages: FullMessage[];
  messageCount: number;
  /** Per-agent surface state keyed by agent id. Wire name: surfaceState (camelCase). */
  surfaceState?: Record<string, SurfaceStateEntry> | null;
}

// Re-export citation stream events from citation types
export type {
  ClaimGeneratedEvent,
  ClaimVerifiedEvent,
  CitationCorrectedEvent,
  NumericClaimDetectedEvent,
  VerificationSummaryEvent,
  CitationStreamEvent,
} from './citation';

// Import citation event types for use in StreamEvent union
import type {
  ClaimGeneratedEvent,
  ClaimVerifiedEvent,
  CitationCorrectedEvent,
  NumericClaimDetectedEvent,
  VerificationSummaryEvent,
} from './citation';

export type StreamEvent =
  | AgentStartedEvent
  | AgentCompletedEvent
  | ResearchStartedEvent
  | ClarificationNeededEvent
  | PlanCreatedEvent
  | StepStartedEvent
  | StepCompletedEvent
  | ToolCallEvent
  | ToolResultEvent
  | ReflectionDecisionEvent
  | SynthesisStartedEvent
  | SynthesisProgressEvent
  | ResearchCompletedEvent
  | StreamErrorEvent
  // Citation verification events
  | ClaimGeneratedEvent
  | ClaimVerifiedEvent
  | CitationCorrectedEvent
  | NumericClaimDetectedEvent
  | VerificationSummaryEvent
  // Stage 7 content revision event
  | ContentRevisedEvent
  // Plan review event
  | PlanReviewEvent
  // Persistence events
  | PersistenceCompletedEvent
  // Custom research phases (e.g. structured-output structuring pass)
  | PhaseStartedEvent
  | PhaseCompletedEvent
  | PhaseErrorEvent
