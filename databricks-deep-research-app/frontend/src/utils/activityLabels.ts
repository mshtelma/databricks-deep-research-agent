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
import type { MergedResultData } from './eventFilters'
import runStatusContract from '../../../contracts/run_status_contract.json'

/**
 * Typed terminal-status label map, sourced from the SINGLE shared contract
 * fixture (`databricks-deep-research-app/contracts/run_status_contract.json`).
 *
 * This is the frontend half of the backend<->frontend status parity: the same
 * JSON pins the backend `RunStatus` enum (via a framework pytest) and these TS
 * label-map keys (via a vitest). Either side drifting fails CI — killing the
 * recurring "frontend-gate-vs-backend-prose" status drift.
 */
export const RUN_STATUS_LABELS: Readonly<Record<string, string>> =
  runStatusContract.labels

/** The terminal-status enum values, from the shared contract. */
export const RUN_STATUSES: readonly string[] = runStatusContract.statuses

/**
 * Pure status -> human label. Falls back to a safe label for an unknown status
 * (never throws), so an unrecognized backend status degrades gracefully.
 */
export function formatStatusLabel(status: string | null | undefined): string {
  if (!status) return 'Unknown'
  return RUN_STATUS_LABELS[status] ?? status
}

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
 *
 * If the event carries a recognized terminal `status` field (the typed
 * status contract), render it from the shared status->label map. Events
 * without a status keep their existing per-eventType formatting below.
 */
export function formatActivityLabel(event: StreamEvent): string {
  const status = (event as { status?: unknown }).status
  if (typeof status === 'string' && status in RUN_STATUS_LABELS) {
    return formatStatusLabel(status)
  }
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
  const merged = (event as unknown as { _mergedResult?: MergedResultData })._mergedResult
  const suffix = buildResultSuffix(sourceType, merged)

  switch (sourceType) {
    case 'web_search':
      return query ? `Searching: ${truncate(query, 70)}${suffix}` : `Searching...${suffix}`
    case 'web_crawl':
      return `Crawling page...${suffix}`
    case 'genie':
      return query ? `Querying Genie: ${truncate(query, 50)}${suffix}` : `Querying enterprise database...${suffix}`
    case 'vector_search':
      return query ? `Searching docs: ${truncate(query, 50)}${suffix}` : `Searching enterprise documents...${suffix}`
    case 'knowledge_assistant':
      return query ? `Asking assistant: ${truncate(query, 50)}${suffix}` : `Asking knowledge assistant...${suffix}`
    case 'file_search':
      return query ? `Searching files: ${truncate(query, 50)}${suffix}` : `Searching files...${suffix}`
    default:
      return query ? `${event.toolName}: ${truncate(query, 50)}${suffix}` : `${event.toolName}...${suffix}`
  }
}

/** Build result count suffix, e.g. " → 5 sources" or " → 3 pages". */
function buildResultSuffix(sourceType: string, merged: MergedResultData | undefined): string {
  if (!merged) return ''
  const count = merged.sourcesAdded || merged.sourcesCrawled || 0
  if (count === 0) return ''
  const unit = sourceType === 'web_crawl' ? 'page' : 'source'
  return ` \u2192 ${count} ${unit}${count !== 1 ? 's' : ''}`
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
