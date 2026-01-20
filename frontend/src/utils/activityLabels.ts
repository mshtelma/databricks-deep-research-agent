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
  ToolCallEvent,
  ToolResultEvent,
} from '../types'

/** Human-readable labels for each agent (no emojis - EnhancedEventLabel adds icons) */
const AGENT_STARTED_LABELS: Record<string, string> = {
  coordinator: 'Analyzing query...',
  background_investigator: 'Background search...',
  planner: 'Creating plan...',
  researcher: 'Researching...',
  reflector: 'Evaluating...',
  synthesizer: 'Writing report...',
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
  switch (event.eventType) {
    case 'agent_started':
      return formatAgentStarted(event)
    case 'agent_completed':
      return formatAgentCompleted(event)
    case 'clarification_needed':
      return 'Need more info...'
    case 'plan_created':
      return formatPlanCreated(event)
    case 'step_started':
      return formatStepStarted(event)
    case 'step_completed':
      return formatStepCompleted(event)
    case 'tool_call':
      return formatToolCall(event)
    case 'tool_result':
      return formatToolResult(event)
    case 'reflection_decision':
      return formatReflectionDecision(event)
    case 'synthesis_started':
      return formatSynthesisStarted(event)
    case 'synthesis_progress':
      return 'Writing...'
    case 'research_completed':
      return formatResearchCompleted(event)
    case 'error':
      return formatError(event)
    case 'research_started':
      return 'Research started'
    case 'claim_generated':
      return 'Claim generated'
    case 'citation_corrected':
      return 'Citation corrected'
    case 'numeric_claim_detected':
      return 'Numeric claim detected'
    case 'content_revised':
      return 'Content revised'
    case 'persistence_completed':
      return 'Saved to database'
    case 'claim_verified':
      return 'Claim verified'
    case 'verification_summary':
      return 'Verification complete'
    default:
      return (event as StreamEvent).eventType
  }
}

function formatAgentStarted(event: AgentStartedEvent): string {
  return AGENT_STARTED_LABELS[event.agent] || `${event.agent} started...`
}

function formatAgentCompleted(event: AgentCompletedEvent): string {
  const label = AGENT_COMPLETED_LABELS[event.agent] || event.agent
  const durationMs = event.durationMs
  if (durationMs == null || isNaN(durationMs)) {
    console.warn('[Activity] Missing/invalid duration:', event)
  }
  const duration = durationMs != null && !isNaN(durationMs)
    ? (durationMs / 1000).toFixed(1)
    : '?'
  return `${label} (${duration}s)`
}

function formatPlanCreated(event: PlanCreatedEvent): string {
  const stepCount = event.steps.length
  return `Plan: ${stepCount} step${stepCount !== 1 ? 's' : ''}`
}

function formatStepStarted(event: StepStartedEvent): string {
  const stepNum = event.stepIndex + 1
  const title = truncate(event.stepTitle, 80)
  return `Step ${stepNum}: ${title}`
}

function formatStepCompleted(event: StepCompletedEvent): string {
  const sources = event.sourcesFound
  return `Found ${sources} source${sources !== 1 ? 's' : ''}`
}

function formatToolCall(event: ToolCallEvent): string {
  const toolName = event.toolName
  if (toolName === 'web_search') {
    const query = typeof event.toolArgs?.query === 'string' ? truncate(event.toolArgs.query, 80) : ''
    return `Searching: ${query}`
  } else if (toolName === 'web_crawl') {
    return 'Crawling page...'
  }
  return `${toolName}...`
}

function formatToolResult(event: ToolResultEvent): string {
  const sourcesCrawled = event.sourcesCrawled
  if (sourcesCrawled != null && sourcesCrawled > 0) {
    return `Crawled ${sourcesCrawled} page${sourcesCrawled !== 1 ? 's' : ''}`
  }
  // For web_search results or failed crawls, return empty to skip display
  return ''
}

function formatReflectionDecision(event: ReflectionDecisionEvent): string {
  switch (event.decision) {
    case 'continue':
      return 'Continue'
    case 'adjust':
      return 'Adjusting plan...'
    case 'complete':
      return 'Research sufficient'
    default:
      return event.decision
  }
}

function formatSynthesisStarted(event: SynthesisStartedEvent): string {
  return `Writing (${event.totalSources} sources)`
}

function formatResearchCompleted(event: ResearchCompletedEvent): string {
  const totalDurationMs = event.totalDurationMs
  const duration = totalDurationMs != null && !isNaN(totalDurationMs)
    ? (totalDurationMs / 1000).toFixed(1)
    : '?'
  return `Done (${duration}s)`
}

function formatError(event: StreamErrorEvent): string {
  const message = truncate(event.errorMessage, 30)
  return message
}

/**
 * Get the Tailwind CSS color class for an event.
 */
export function getActivityColor(event: StreamEvent): string {
  switch (event.eventType) {
    case 'error':
      return 'text-red-500'
    case 'agent_completed':
    case 'step_completed':
    case 'plan_created':
    case 'research_completed':
    case 'tool_result':
      return 'text-green-600 dark:text-green-400'
    case 'reflection_decision':
    case 'clarification_needed':
      return 'text-blue-500 dark:text-blue-400'
    case 'tool_call':
      return 'text-cyan-500 dark:text-cyan-400'
    case 'research_started':
      return 'text-blue-500 dark:text-blue-400'
    case 'claim_generated':
      return 'text-purple-500 dark:text-purple-400'
    case 'citation_corrected':
      return 'text-amber-500 dark:text-amber-400'
    case 'numeric_claim_detected':
      return 'text-cyan-500 dark:text-cyan-400'
    case 'content_revised':
      return 'text-orange-500 dark:text-orange-400'
    case 'persistence_completed':
      return 'text-green-500 dark:text-green-400'
    case 'claim_verified':
      return 'text-green-600 dark:text-green-400'
    case 'verification_summary':
      return 'text-purple-500 dark:text-purple-400'
    default:
      return 'text-amber-500 dark:text-amber-400'
  }
}

/**
 * Truncate a string to maxLength, adding ellipsis if needed.
 */
function truncate(str: string, maxLength: number): string {
  if (!str) return ''
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength - 1) + '…'
}
