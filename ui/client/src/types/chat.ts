export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  metadata?: ResearchMetadata
  isStreaming?: boolean
}

export interface ResearchMetadata {
  searchQueries: string[]
  sources: Array<{
    url: string
    title: string
    relevanceScore?: number
    snippet?: string
  }>
  researchIterations: number
  confidenceScore?: number
  reasoningSteps: string[]

  // Enhanced progress tracking fields
  totalSourcesFound?: number
  phase?: string
  progressPercentage?: number
  elapsedTime?: number
  currentNode?: string
  vectorResultsCount?: number

  // Multi-agent fields
  currentAgent?: string
  planDetails?: PlanMetadata
  factualityScore?: number
  reportStyle?: string
  verificationLevel?: string
  grounding?: GroundingMetadata
}

export interface ResearchProgress {
  currentPhase: 'coordinator' | 'background_investigation' | 'planning' | 'research' | 'fact_checking' | 'reporting' | 'complete'
  queriesGenerated: number
  sourcesFound: number
  iterationsComplete: number
  
  // Enhanced real-time progress fields
  progressPercentage?: number
  elapsedTime?: number
  currentNode?: string
  vectorResultsCount?: number
  estimatedTimeRemaining?: number
  currentOperation?: string
  subProgress?: {
    current: number
    total: number
    description: string
  }

  // Multi-agent specific fields
  currentAgent?: string
  agentHandoffs?: AgentHandoff[]
  planIterations?: number
  factualityScore?: number
  researchQualityScore?: number
  coverageScore?: number
}

export interface PhaseConfiguration {
  phase: ResearchProgress['currentPhase']
  icon: string
  label: string
  description: string
  color: string
  bgColor: string
  expectedDuration?: number
}

export interface StreamEvent {
  type: 'stream_start' | 'content_delta' | 'research_update' | 'message_complete' | 'stream_end' | 'error' | 'intermediate_event' | 'event_batch'
  content?: string
  metadata?: ResearchMetadata
  error?: string
  // Intermediate events
  events?: IntermediateEvent[]
  event?: IntermediateEvent
}

export interface IntermediateEvent {
  id: string
  timestamp: number
  stage_id?: string
  correlation_id?: string
  sequence: number
  event_type: IntermediateEventType
  data: Record<string, any>
  meta?: Record<string, any>
}

export enum IntermediateEventType {
  // Action events
  ACTION_START = 'action_start',
  ACTION_PROGRESS = 'action_progress',
  ACTION_COMPLETE = 'action_complete',

  // Tool-specific events
  TOOL_CALL_START = 'tool_call_start',
  TOOL_CALL_PROGRESS = 'tool_call_progress',
  TOOL_CALL_COMPLETE = 'tool_call_complete',
  TOOL_CALL_ERROR = 'tool_call_error',

  // Reasoning/LLM events
  THOUGHT_SNAPSHOT = 'thought_snapshot',
  SYNTHESIS_PROGRESS = 'synthesis_progress',

  // Content/citation events
  CITATION_ADDED = 'citation_added',

  // Multi-agent specific events
  AGENT_HANDOFF = 'agent_handoff',
  PLAN_CREATED = 'plan_created',
  PLAN_UPDATED = 'plan_updated',
  PLAN_ITERATION = 'plan_iteration',
  BACKGROUND_INVESTIGATION = 'background_investigation',
  GROUNDING_START = 'grounding_start',
  GROUNDING_COMPLETE = 'grounding_complete',
  GROUNDING_CONTRADICTION = 'grounding_contradiction',
  REPORT_GENERATION = 'report_generation',
  QUALITY_ASSESSMENT = 'quality_assessment',

  // Stage transitions (existing, for compatibility)
  STAGE_TRANSITION = 'stage_transition'
}

export interface ChatRequest {
  messages: Array<{ role: string; content: string }>
  config?: {
    maxTokens?: number
    temperature?: number
    researchMode?: 'quick' | 'standard' | 'deep'
  }
}

export interface ChatResponse {
  message: { role: string; content: string }
  metadata?: ResearchMetadata
}

// Multi-agent specific interfaces
export interface AgentHandoff {
  fromAgent: string
  toAgent: string
  reason: string
  context?: Record<string, any>
  timestamp: number
}

export interface PlanMetadata {
  steps: PlanStep[]
  quality?: number
  iterations: number
  status: 'draft' | 'approved' | 'executing' | 'completed'
  hasEnoughContext?: boolean
}

export interface PlanStep {
  id: string
  description: string
  status: 'pending' | 'in_progress' | 'completed' | 'skipped'
  result?: string
  completedAt?: number
}

export interface GroundingMetadata {
  factualityScore: number
  contradictions: Contradiction[]
  verifications: FactVerification[]
  verificationLevel: 'basic' | 'moderate' | 'thorough'
}

export interface Contradiction {
  id: string
  claim: string
  evidence: string
  severity: 'low' | 'medium' | 'high'
  resolved?: boolean
  resolution?: string
}

export interface FactVerification {
  id: string
  fact: string
  verified: boolean
  confidence: number
  sources: string[]
}

export interface AgentConfig {
  reportStyle: 'professional' | 'casual' | 'academic' | 'technical'
  verificationLevel: 'basic' | 'moderate' | 'thorough'
  enableIterativePlanning: boolean
  enableBackgroundInvestigation: boolean
  autoAcceptPlan: boolean
  maxPlanIterations: number
}