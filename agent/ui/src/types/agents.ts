// Types for the multi-agent research system
export interface IntermediateEvent {
  id: string
  timestamp: number
  correlation_id?: string
  sequence?: number
  event_type: string
  data: {
    agent?: string
    current_agent?: string
    from_agent?: string
    action?: string
    content_preview?: string
    query?: string
    parameters?: {
      query?: string
    }
    results_count?: number
    result_count?: number
    result_summary?: string
    error_message?: string
    step_id?: string
    plan?: any
  }
  meta?: {
    title?: string
    description?: string
    category?: string
    icon?: string
    priority?: number
    confidence?: number
    reasoning?: string
  }
}

export enum IntermediateEventType {
  AGENT_START = 'agent_start',
  AGENT_COMPLETE = 'agent_complete',
  LLM_THINKING = 'llm_thinking',
  QUERY_GENERATED = 'query_generated',
  QUERY_EXECUTING = 'query_executing',
  TOOL_CALL_START = 'tool_call_start',
  TOOL_CALL_COMPLETE = 'tool_call_complete',
  TOOL_CALL_ERROR = 'tool_call_error',
  SEARCH_RESULTS_FOUND = 'search_results_found',
  SYNTHESIS_PROGRESS = 'synthesis_progress',
  ACTION_COMPLETE = 'action_complete',
  PLAN_CREATED = 'plan_created',
  PLAN_UPDATED = 'plan_updated',
  PLAN_STRUCTURE_VISUALIZE = 'plan_structure_visualize',
  STEP_ACTIVATED = 'step_activated',
  STEP_COMPLETED = 'step_completed',
  STEP_FAILED = 'step_failed',
}

export interface PlanStep {
  id: string
  description: string
  status: 'pending' | 'in_progress' | 'completed' | 'skipped'
  result?: string
  completedAt?: number
}

export interface PlanMetadata {
  steps: PlanStep[]
  status: 'draft' | 'executing' | 'completed'
  iterations: number
  quality?: number
  hasEnoughContext: boolean
}

export interface ResearchMetadata {
  planDetails?: PlanMetadata
  sources?: Array<{
    url: string
    title: string
    relevanceScore?: number
    snippet?: string
  }>
  searchQueries?: string[]
  researchIterations?: number
  totalSourcesFound?: number
  phase?: string
  progressPercentage?: number
  elapsedTime?: number
  currentNode?: string
  vectorResultsCount?: number
  currentAgent?: string
  factualityScore?: number
  reportStyle?: string
  confidenceScore?: number
}

export type AgentType = 'coordinator' | 'planner' | 'researcher' | 'checker' | 'reporter'

export interface AgentActivity {
  agent: string
  currentAction: string
  status: 'thinking' | 'working' | 'completed' | 'error'
  events: IntermediateEvent[]
  queries: string[]
  findings: string[]
}