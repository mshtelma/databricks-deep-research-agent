// API Types

export type ChatStatus = 'active' | 'archived' | 'deleted'
export type MessageRole = 'user' | 'agent' | 'system'
export type ResearchDepth = 'auto' | 'light' | 'medium' | 'extended'
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
  created_at: string
  updated_at: string
  message_count: number
}

export interface Message {
  id: string
  chat_id: string
  role: MessageRole
  content: string
  created_at: string
  is_edited: boolean
  research_session?: ResearchSession | null
}

export interface Source {
  id: string
  url: string
  title: string | null
  snippet: string | null
  relevance_score: number | null
}

export interface QueryClassification {
  complexity: 'simple' | 'moderate' | 'complex'
  follow_up_type: 'new_topic' | 'clarification' | 'complex_follow_up'
  is_ambiguous: boolean
  clarifying_questions: string[]
  reasoning: string
}

export interface PlanStep {
  id: string
  title: string
  description: string
  step_type: 'research' | 'analysis'
  needs_search: boolean
  status: 'pending' | 'in_progress' | 'completed' | 'skipped'
  observation: string | null
}

export interface ResearchPlan {
  id: string
  title: string
  thought: string
  steps: PlanStep[]
  iteration: number
  created_at: string
}

export interface ResearchSession {
  id: string
  query_classification: QueryClassification | null
  research_depth: ResearchDepth
  status: ResearchStatus
  current_agent: string | null
  plan: ResearchPlan | null
  current_step_index: number | null
  plan_iterations: number
  started_at: string
  completed_at: string | null
  sources: Source[]
}

export interface UserPreferences {
  system_instructions: string | null
  default_depth: ResearchDepth
  ui_preferences: Record<string, unknown>
  updated_at: string
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
  research_depth?: ResearchDepth
}

export interface SendMessageResponse {
  user_message: Message
  agent_message_id: string
  research_session_id: string
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
  // Persistence events
  | 'persistence_completed'

export interface BaseStreamEvent {
  event_type: StreamEventType
  timestamp: string
}

export interface AgentStartedEvent extends BaseStreamEvent {
  event_type: 'agent_started'
  agent: string
  model_tier: string
}

export interface AgentCompletedEvent extends BaseStreamEvent {
  event_type: 'agent_completed'
  agent: string
  duration_ms: number
}

export interface ResearchStartedEvent extends BaseStreamEvent {
  event_type: 'research_started'
  message_id: string
  research_session_id: string
}

export interface ClarificationNeededEvent extends BaseStreamEvent {
  event_type: 'clarification_needed'
  questions: string[]
  round: number
}

export interface PlanCreatedEvent extends BaseStreamEvent {
  event_type: 'plan_created'
  plan_id: string
  title: string
  thought: string
  steps: { id: string; title: string; step_type: string; needs_search: boolean }[]
  iteration: number
}

export interface StepStartedEvent extends BaseStreamEvent {
  event_type: 'step_started'
  step_index: number
  step_id: string
  step_title: string
  step_type: string
}

export interface StepCompletedEvent extends BaseStreamEvent {
  event_type: 'step_completed'
  step_index: number
  step_id: string
  observation_summary: string
  sources_found: number
}

export interface ToolCallEvent extends BaseStreamEvent {
  event_type: 'tool_call'
  tool_name: string // 'web_search' | 'web_crawl'
  tool_args: Record<string, unknown>
  call_number: number
}

export interface ToolResultEvent extends BaseStreamEvent {
  event_type: 'tool_result'
  tool_name: string
  result_preview: string
  sources_crawled: number
}

export interface ReflectionDecisionEvent extends BaseStreamEvent {
  event_type: 'reflection_decision'
  decision: 'continue' | 'adjust' | 'complete'
  reasoning: string
  suggested_changes: string[] | null
}

export interface SynthesisStartedEvent extends BaseStreamEvent {
  event_type: 'synthesis_started'
  total_observations: number
  total_sources: number
}

export interface SynthesisProgressEvent extends BaseStreamEvent {
  event_type: 'synthesis_progress'
  content_chunk: string
}

export interface ResearchCompletedEvent extends BaseStreamEvent {
  event_type: 'research_completed'
  session_id: string
  total_steps_executed: number
  total_steps_skipped: number
  plan_iterations: number
  total_duration_ms: number
}

export interface StreamErrorEvent extends BaseStreamEvent {
  event_type: 'error'
  error_code: string
  error_message: string
  recoverable: boolean
}

export interface PersistenceCompletedEvent extends BaseStreamEvent {
  event_type: 'persistence_completed'
  chat_id: string
  message_id: string
  research_session_id: string
  chat_title: string
  was_draft: boolean
  counts: Record<string, number>
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
  // Persistence events
  | PersistenceCompletedEvent
