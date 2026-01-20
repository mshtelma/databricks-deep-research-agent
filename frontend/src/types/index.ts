// API Types

export type ChatStatus = 'active' | 'archived' | 'deleted'
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
  createdAt: string
  updatedAt: string
  messageCount: number
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
}

export interface ToolCallEvent extends BaseStreamEvent {
  eventType: 'tool_call'
  toolName: string // 'web_search' | 'web_crawl'
  toolArgs: Record<string, unknown>
  callNumber: number
}

export interface ToolResultEvent extends BaseStreamEvent {
  eventType: 'tool_result'
  toolName: string
  resultPreview: string
  sourcesCrawled: number
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

// Stage 7 content revision event - sent after verification retrieval applies softening
export interface ContentRevisedEvent extends BaseStreamEvent {
  eventType: 'content_revised'
  content: string
  revisionCount: number
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
  // Persistence events
  | PersistenceCompletedEvent
