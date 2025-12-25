/**
 * Activity label formatting utilities for the Research Activity panel.
 * Transforms raw event types into human-readable labels with emojis and colors.
 */

import type {
  StreamEvent,
  AgentStartedEvent,
  AgentCompletedEvent,
  PlanCreatedEvent,
  StepStartedEvent,
  StepCompletedEvent,
  ReflectionDecisionEvent,
  SynthesisStartedEvent,
  ResearchCompletedEvent,
  StreamErrorEvent,
} from '../types'

/** Human-readable labels for each agent */
const AGENT_STARTED_LABELS: Record<string, string> = {
  coordinator: '🔍 Analyzing query...',
  background_investigator: '📚 Background search...',
  planner: '📋 Creating plan...',
  researcher: '🔬 Researching...',
  reflector: '🤔 Evaluating...',
  synthesizer: '✍️ Writing report...',
}

/** Human-readable labels for completed agents */
const AGENT_COMPLETED_LABELS: Record<string, string> = {
  coordinator: 'Analyzed',
  background_investigator: 'Background done',
  planner: 'Plan created',
  researcher: 'Research done',
  reflector: 'Evaluated',
  synthesizer: 'Report done',
}

/**
 * Format a stream event into a human-readable activity label.
 */
export function formatActivityLabel(event: StreamEvent): string {
  switch (event.event_type) {
    case 'agent_started':
      return formatAgentStarted(event)
    case 'agent_completed':
      return formatAgentCompleted(event)
    case 'clarification_needed':
      return '❓ Need more info...'
    case 'plan_created':
      return formatPlanCreated(event)
    case 'step_started':
      return formatStepStarted(event)
    case 'step_completed':
      return formatStepCompleted(event)
    case 'reflection_decision':
      return formatReflectionDecision(event)
    case 'synthesis_started':
      return formatSynthesisStarted(event)
    case 'synthesis_progress':
      return '✍️ Writing...'
    case 'research_completed':
      return formatResearchCompleted(event)
    case 'error':
      return formatError(event)
    default:
      return (event as StreamEvent).event_type
  }
}

function formatAgentStarted(event: AgentStartedEvent): string {
  return AGENT_STARTED_LABELS[event.agent] || `${event.agent} started...`
}

function formatAgentCompleted(event: AgentCompletedEvent): string {
  const label = AGENT_COMPLETED_LABELS[event.agent] || event.agent
  const duration = (event.duration_ms / 1000).toFixed(1)
  return `✓ ${label} (${duration}s)`
}

function formatPlanCreated(event: PlanCreatedEvent): string {
  const stepCount = event.steps.length
  return `📋 Plan: ${stepCount} step${stepCount !== 1 ? 's' : ''}`
}

function formatStepStarted(event: StepStartedEvent): string {
  const stepNum = event.step_index + 1
  const title = truncate(event.step_title, 25)
  return `▶ Step ${stepNum}: ${title}`
}

function formatStepCompleted(event: StepCompletedEvent): string {
  const sources = event.sources_found
  return `✓ Found ${sources} source${sources !== 1 ? 's' : ''}`
}

function formatReflectionDecision(event: ReflectionDecisionEvent): string {
  switch (event.decision) {
    case 'continue':
      return '→ Continue'
    case 'adjust':
      return '↻ Adjusting plan...'
    case 'complete':
      return '✓ Research sufficient'
    default:
      return `→ ${event.decision}`
  }
}

function formatSynthesisStarted(event: SynthesisStartedEvent): string {
  return `✍️ Writing (${event.total_sources} sources)`
}

function formatResearchCompleted(event: ResearchCompletedEvent): string {
  const duration = (event.total_duration_ms / 1000).toFixed(1)
  return `🎉 Done (${duration}s)`
}

function formatError(event: StreamErrorEvent): string {
  const message = truncate(event.error_message, 30)
  return `❌ ${message}`
}

/**
 * Get the Tailwind CSS color class for an event.
 */
export function getActivityColor(event: StreamEvent): string {
  switch (event.event_type) {
    case 'error':
      return 'text-red-500'
    case 'agent_completed':
    case 'step_completed':
    case 'plan_created':
    case 'research_completed':
      return 'text-green-600 dark:text-green-400'
    case 'reflection_decision':
    case 'clarification_needed':
      return 'text-blue-500 dark:text-blue-400'
    default:
      return 'text-amber-500 dark:text-amber-400'
  }
}

/**
 * Truncate a string to maxLength, adding ellipsis if needed.
 */
function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength - 1) + '…'
}
