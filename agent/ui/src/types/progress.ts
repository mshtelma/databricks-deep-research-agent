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
    name: 'Initiating research',
    agent: 'coordinator',
    description: 'Analyzing request and routing to appropriate agents',
    order: 1
  },
  {
    id: 'planning',
    name: 'Planning research approach',
    agent: 'planner',
    description: 'Creating structured research plan with quality assessment',
    order: 2
  },
  {
    id: 'research',
    name: 'Executing research steps',
    agent: 'researcher',
    description: 'Conducting research according to the plan',
    order: 3
  },
  {
    id: 'fact_checking',
    name: 'Fact checking and verification',
    agent: 'fact_checker',
    description: 'Verifying claims and ensuring factual accuracy',
    order: 4
  },
  {
    id: 'synthesizing',
    name: 'Synthesizing final report',
    agent: 'reporter',
    description: 'Compiling findings into comprehensive report',
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