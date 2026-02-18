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
import { inferSourceType } from './eventStats'

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

const ENTERPRISE_SOURCE_TYPES = new Set(['genie', 'vector_search', 'knowledge_assistant']);

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
  const total = event.sourcesFound
  const fileSources = event.fileSourcesFound ?? 0
  const webSources = total - fileSources

  if (fileSources > 0 && webSources > 0) {
    return `Found ${webSources} source${webSources !== 1 ? 's' : ''} + file evidence`
  }
  if (fileSources > 0) {
    return 'Found file evidence'
  }
  return `Found ${total} source${total !== 1 ? 's' : ''}`
}

function formatToolCall(event: ToolCallEvent): string {
  const sourceType = inferSourceType(event)
  const query = findQueryArg(event.toolArgs)

  switch (sourceType) {
    case 'web_search':
      return query ? `Searching: ${truncate(query, 80)}` : 'Searching...'
    case 'web_crawl':
      return 'Crawling page...'
    case 'genie':
      return query ? `Querying Genie: ${truncate(query, 60)}` : 'Querying enterprise database...'
    case 'vector_search':
      return query ? `Searching docs: ${truncate(query, 60)}` : 'Searching enterprise documents...'
    case 'knowledge_assistant':
      return query ? `Asking assistant: ${truncate(query, 60)}` : 'Asking knowledge assistant...'
    case 'file_search':
      return query ? `Searching files: ${truncate(query, 60)}` : 'Searching files...'
    default:
      return query ? `${event.toolName}: ${truncate(query, 60)}` : `${event.toolName}...`
  }
}

/** Extract the most likely query argument from tool args. */
function findQueryArg(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null
  for (const key of ['question', 'query', 'query_text', 'search_query']) {
    if (typeof args[key] === 'string' && args[key]) return args[key] as string
  }
  for (const val of Object.values(args)) {
    if (typeof val === 'string' && val.length > 0 && val.length < 200) return val
  }
  return null
}

function formatToolResult(event: ToolResultEvent): string {
  const sourceType = inferSourceType(event)
  const count = event.sourcesAdded ?? event.sourcesCrawled ?? 0

  switch (sourceType) {
    case 'genie':
      return count > 0 ? `Genie: ${count} result${count !== 1 ? 's' : ''}` : 'Genie query complete'
    case 'vector_search':
      return count > 0 ? `Found ${count} document${count !== 1 ? 's' : ''}` : 'Search complete'
    case 'knowledge_assistant':
      return 'Assistant answered'
    case 'file_search':
      return count > 0 ? `Searched files (${count} match${count !== 1 ? 'es' : ''})` : 'Searched files'
    case 'web_crawl': {
      if (count > 0) return `Crawled ${count} page${count !== 1 ? 's' : ''}`
      return ''
    }
    default: {
      // Web search or unknown: show preview
      const preview = event.resultPreview
      if (preview && preview.length > 0) {
        return truncate(preview, 60)
      }
      return ''
    }
  }
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
      return 'text-green-600 dark:text-green-400'
    case 'tool_result': {
      const trSt = inferSourceType(event)
      if (ENTERPRISE_SOURCE_TYPES.has(trSt))
        return 'text-indigo-600 dark:text-indigo-400'
      return 'text-green-600 dark:text-green-400'
    }
    case 'reflection_decision':
    case 'clarification_needed':
      return 'text-blue-500 dark:text-blue-400'
    case 'tool_call': {
      const tcSt = inferSourceType(event)
      if (ENTERPRISE_SOURCE_TYPES.has(tcSt))
        return 'text-indigo-500 dark:text-indigo-400'
      return 'text-cyan-500 dark:text-cyan-400'
    }
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
  return str.slice(0, maxLength - 1) + '\u2026'
}
