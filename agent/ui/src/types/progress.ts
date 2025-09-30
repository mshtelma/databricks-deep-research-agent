// Progress tracking types for research workflow

export interface ProgressItem {
  id: string
  label: string
  status: 'pending' | 'active' | 'completed' | 'failed' | 'skipped'
  timestamp?: number
  result?: string
  stepNumber?: number
  isWorkflowPhase?: boolean
  duration?: number
  progress?: number
}

export interface WorkflowPhase {
  id: string
  name: string
  agent: string
  description: string
  order: number
}

export interface ResearchProgress {
  currentPhase?: string
  currentAgent?: string
  progressItems: ProgressItem[]
  overallProgress: number
  startTime?: number
  elapsedTime?: number
  estimatedCompletion?: number
}

export interface StructuredProgress {
  workflowPhases: ProgressItem[]
  planSteps: ProgressItem[]
  currentPhase?: string
  currentAgent?: string
  overallProgress: number
  startTime?: number
  elapsedTime?: number
  estimatedCompletion?: number
}

// Standard workflow phases for the multi-agent research system
export const WORKFLOW_PHASES: WorkflowPhase[] = [
  {
    id: 'initiate',
    name: 'Understanding Your Request',
    agent: 'coordinator',
    description: 'Analyzing your question to plan the best approach',
    order: 1
  },
  {
    id: 'planning',
    name: 'Creating Research Plan',
    agent: 'planner',
    description: 'Designing a structured approach to answer your question',
    order: 2
  },
  {
    id: 'research',
    name: 'Gathering Information',
    agent: 'researcher',
    description: 'Searching and collecting relevant information from multiple sources',
    order: 3
  },
  {
    id: 'fact_checking',
    name: 'Verifying Information',
    agent: 'fact_checker',
    description: 'Checking accuracy and reliability of collected information',
    order: 4
  },
  {
    id: 'synthesizing',
    name: 'Preparing Your Report',
    agent: 'reporter',
    description: 'Organizing findings into a comprehensive answer',
    order: 5
  }
]

// Agent to phase mapping
export const AGENT_PHASE_MAP: Record<string, string> = {
  'coordinator': 'initiate',
  'planner': 'planning',
  'researcher': 'research',
  'background_investigation': 'research',
  'fact_checker': 'fact_checking',
  'checker': 'fact_checking',
  'reporter': 'synthesizing'
}

// Phase to agent mapping (reverse)
export const PHASE_AGENT_MAP: Record<string, string> = {
  'initiate': 'coordinator',
  'planning': 'planner',
  'research': 'researcher',
  'fact_checking': 'fact_checker',
  'synthesizing': 'reporter'
}

export type PhaseStatus = ProgressItem['status']

export interface PhaseTransition {
  from?: string
  to: string
  timestamp: number
  agent: string
  event?: string
}