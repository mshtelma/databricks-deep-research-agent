import { useCallback, useRef, useState } from 'react'
import { IntermediateEvent, PlanMetadata, PlanStep, ResearchMetadata } from '../types/agents'
import { ProgressItem, StructuredProgress, WORKFLOW_PHASES, AGENT_PHASE_MAP, createInitialStructuredProgress } from '../types/progress'
import { filterContent, extractPhaseInfo } from '../utils/contentFilter'
import { evaluatePhaseStatus } from '../config/phaseTransitionRules'

// ResponsesAgent types following MLflow specifications
export interface ResponsesAgentRequest {
  input: Array<{
    type: "message"
    role: "user" | "assistant"
    content: string
  }>
  stream?: boolean
  custom_inputs?: Record<string, any>
}

export interface ResponsesAgentResponse {
  output: Array<{
    type: "message"
    role: "assistant"
    content: Array<{
      type: "output_text"
      text: string
    }>
    id: string
  }>
  custom_outputs?: Record<string, any>
}

export interface ResponsesAgentStreamEvent {
  type: "response.output_text.delta" | "response.output_item.done" | "intermediate_event" | "response.metadata"
  item_id?: string
  delta?: string
  item?: {
    type: "message"
    role: "assistant"
    content: Array<{
      type: "output_text"
      text: string
    }>
    id: string
  }
  intermediate_event?: {
    event_type: string
    data: any
    timestamp?: number
    correlation_id?: string
    meta?: any
  }
  metadata?: Record<string, any>
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp?: number
  isStreaming?: boolean
  metadata?: ResearchMetadata
}

type StepStatus = PlanStep['status']

// Removed PLAN_GENERATION_STEP_ID since plan generation is a phase, not a step
const STEP_ID_ALIASES: Record<string, string> = {
  // Plan-related aliases removed - plan generation is handled as workflow phase
}

const STEP_STATUS_PRIORITY: Record<StepStatus, number> = {
  completed: 5,
  in_progress: 4,
  skipped: 2,
  pending: 1
}

const PLAN_DEBUG_ENABLED = true

function debugPlanSnapshot(reason: string, plan?: PlanMetadata) {
  if (!PLAN_DEBUG_ENABLED) return
  if (!plan) {
    console.debug(`[PlanDebug] ${reason}: (no plan)`)
    return
  }
  console.debug(
    `[PlanDebug] ${reason}: status=${plan.status}, steps=${plan.steps.length}`,
    plan.steps.map(step => ({ id: step.id, status: step.status, desc: step.description?.slice(0, 60) })),
  )
}

function debugStreamEvent(reason: string, payload: Record<string, unknown>) {
  if (!PLAN_DEBUG_ENABLED) return
  console.debug(`[StreamDebug] ${reason}`, payload)
}

const PLAN_EVENT_TYPES = new Set([
  'plan_created',
  'plan_updated',
  'plan_structure',
  'plan_structure_visualize',
  'plan_ready'
])

const PLAN_LIFECYCLE_EVENT_TYPES = new Map([
  ['plan_started', 'in_progress'],
  ['plan_completed', 'completed'],
  ['plan_failed', 'skipped']
])
const PLAN_LIFECYCLE_PROGRESS_AGENT: Record<string, string> = {
  plan_started: 'planner',
  plan_completed: 'researcher',
  plan_failed: 'researcher'
}

const STEP_EVENT_TYPES = new Set([
  'step_activated',
  'step_completed',
  'step_failed'
])

// Removed PLAN_LIFECYCLE_STEP_IDS since plan generation is a phase, not a step

function canonicalizeStepId(value: unknown, fallbackIndex?: number): string {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (trimmed.length > 0) {
      const numericMatch = trimmed.match(/(\d+)/)
      if (numericMatch) {
        const numericId = `step_${numericMatch[1].padStart(3, '0')}`
        return STEP_ID_ALIASES[numericId] ?? numericId
      }
      const normalized = trimmed.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
      if (normalized.length > 0) {
        const candidate = normalized.startsWith('step_') ? normalized : `step_${normalized}`
        return STEP_ID_ALIASES[candidate] ?? candidate
      }
    }
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    const numericId = `step_${Math.max(0, value).toString().padStart(3, '0')}`
    return STEP_ID_ALIASES[numericId] ?? numericId
  }

  if (fallbackIndex !== undefined) {
    const fallbackId = `step_${(fallbackIndex + 1).toString().padStart(3, '0')}`
    return STEP_ID_ALIASES[fallbackId] ?? fallbackId
  }

  return `step_001` // Default fallback
}

function normalizeStepStatus(status?: string | null): StepStatus {
  if (!status) return 'pending'

  const normalized = status
    .toString()
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')

  if (['completed', 'complete', 'done', 'finished'].includes(normalized)) {
    return 'completed'
  }

  if (['in_progress', 'inprogress', 'active', 'running', 'working'].includes(normalized)) {
    return 'in_progress'
  }

  if (['failed', 'error'].includes(normalized)) {
    return 'skipped'
  }

  if (['skipped', 'cancelled', 'canceled'].includes(normalized)) {
    return 'skipped'
  }

  return 'pending'
}

function mergeStepStatus(current?: StepStatus, incoming?: StepStatus): StepStatus {
  if (!current) return incoming ?? 'pending'
  if (!incoming) return current

  return STEP_STATUS_PRIORITY[incoming] >= STEP_STATUS_PRIORITY[current] ? incoming : current
}

// function createStaticStep(id: string, description: string, status: StepStatus = 'pending'): PlanStep {
//   return {
//     id,
//     description,
//     status
//   }
// }

// Removed SPECIAL_STEP_IDS since we no longer have special static steps

// Remove static plan generation step - this is a phase, not a step
const INITIAL_PLACEHOLDER_STEPS: PlanStep[] = []


// Removed STEP_DESCRIPTION_ALIASES since we no longer have static steps


function clonePlanStep(step: PlanStep): PlanStep {
  return {
    id: step.id,
    description: step.description,
    status: step.status,
    result: step.result,
    completedAt: step.completedAt
  }
}

function createInitialPlanDetails(): PlanMetadata | undefined {
  // Don't create empty plan with placeholder steps
  if (INITIAL_PLACEHOLDER_STEPS.length === 0) {
    return undefined
  }
  return {
    steps: [
      ...INITIAL_PLACEHOLDER_STEPS.map(clonePlanStep)
    ],
    status: 'draft',
    iterations: 1,
    hasEnoughContext: false
  }
}

function mergeDynamicSteps(existing: PlanStep[], incoming: PlanStep[]): PlanStep[] {
  const existingMap = new Map(existing.map(step => [step.id, step]))
  return incoming.map(step => {
    const existingStep = existingMap.get(step.id)
    if (!existingStep) {
      return clonePlanStep(step)
    }

    return {
      id: step.id,
      description: step.description || existingStep.description,
      status: mergeStepStatus(existingStep.status, step.status),
      result: step.result ?? existingStep.result,
      completedAt: step.completedAt ?? existingStep.completedAt
    }
  })
}

function recomputePlanStatus(steps: PlanStep[]): PlanMetadata['status'] {
  if (steps.length === 0) {
    return 'draft'
  }

  const allCompleted = steps.every(step => step.status === 'completed')
  if (allCompleted) {
    return 'completed'
  }

  const anyInProgress = steps.some(step => step.status === 'in_progress')
  if (anyInProgress) {
    return 'executing'
  }

  return 'draft'
}

function normalizePlanLifecycleStatus(status?: string): PlanMetadata['status'] | undefined {
  if (!status) return undefined
  const normalized = status.trim().toLowerCase()
  if (['completed', 'complete', 'done', 'finished'].includes(normalized)) {
    return 'completed'
  }
  if (['executing', 'in_progress', 'running', 'active'].includes(normalized)) {
    return 'executing'
  }
  if (['draft', 'pending', 'planned'].includes(normalized)) {
    return 'draft'
  }
  return undefined
}

function buildPlanDetailsFromPayload(planPayload: any, existing?: PlanMetadata): PlanMetadata | undefined {
  if (!planPayload) {
    return existing
  }

  const rawSteps: any[] = Array.isArray(planPayload.steps)
    ? planPayload.steps
    : typeof planPayload === 'object' && Array.isArray(planPayload?.plan?.steps)
      ? planPayload.plan.steps
      : []

  const existingSteps = existing?.steps ?? []
  const existingMap = new Map(existingSteps.map(step => [step.id, step]))

  const dynamicSteps: PlanStep[] = rawSteps
    .map((rawStep, index) => {
      const candidateId = rawStep.step_id ?? rawStep.stepId ?? rawStep.id ?? rawStep.step ?? rawStep.name ?? rawStep.title
      const canonicalId = canonicalizeStepId(candidateId, index)
      const description = rawStep.description || rawStep.summary || rawStep.title || `Step ${index + 1}`

      console.log('🎯 Plan step created:', {
        index,
        originalId: candidateId,
        canonicalId,
        description
      })

      const normalizedStatus = normalizeStepStatus(rawStep.status ?? rawStep.state ?? rawStep.step_status)
      const existingStep = existingMap.get(canonicalId)

      return {
        id: canonicalId,
        description,
        status: existingStep ? mergeStepStatus(existingStep.status, normalizedStatus) : normalizedStatus,
        result: rawStep.result ?? rawStep.summary ?? existingStep?.result,
        completedAt: rawStep.completedAt ?? rawStep.completed_at ?? existingStep?.completedAt
      }
    })

  const mergedDynamicSteps = mergeDynamicSteps(
    existingSteps, // No need to filter since we don't have special steps anymore
    dynamicSteps
  )

  const combined: PlanStep[] = [
    ...mergedDynamicSteps
  ]

  // Deduplicate by canonical id while preserving highest status priority
  const dedupedMap = new Map<string, PlanStep>()
  combined.forEach((step, index) => {
    const canonicalId = canonicalizeStepId(step.id, index)
    const existingStep = dedupedMap.get(canonicalId)
    if (!existingStep) {
      console.log('🆕 Adding unique step:', { originalId: step.id, canonicalId, description: step.description })
      dedupedMap.set(canonicalId, { ...step, id: canonicalId })
    } else {
      console.log('🔄 Merging duplicate step:', {
        originalId: step.id,
        canonicalId,
        existingDesc: existingStep.description,
        newDesc: step.description
      })
      dedupedMap.set(canonicalId, {
        id: canonicalId,
        description: step.description || existingStep.description,
        status: mergeStepStatus(existingStep.status, step.status),
        result: step.result ?? existingStep.result,
        completedAt: step.completedAt ?? existingStep.completedAt
      })
    }
  })

  const finalSteps = Array.from(dedupedMap.values())
  const status = normalizePlanLifecycleStatus(planPayload.status) ?? recomputePlanStatus(finalSteps)

  return {
    steps: finalSteps,
    status,
    iterations: planPayload.iterations ?? planPayload.iteration ?? existing?.iterations ?? 1,
    quality: planPayload.quality ?? planPayload.qualityScore ?? existing?.quality,
    hasEnoughContext: planPayload.hasEnoughContext ?? planPayload.has_enough_context ?? planPayload.context_ready ?? existing?.hasEnoughContext ?? true
  }
}

function mergeMetadataPlanDetails(
  current: PlanMetadata | undefined,
  planPayload: any
): PlanMetadata | undefined {
  const normalizedPlan = buildPlanDetailsFromPayload(planPayload, current)
  if (!normalizedPlan) {
    return current
  }
  return normalizedPlan
}

function updateStepStatusInPlan(
  planDetails: PlanMetadata | undefined,
  stepId: string,
  status: StepStatus
): PlanMetadata | undefined {
  if (!planDetails) return planDetails

  const canonicalId = canonicalizeStepId(stepId)
  let matched = false
  let updated = false

  const updatedSteps = planDetails.steps.map((step, index) => {
    const stepCanonicalId = canonicalizeStepId(step.id, index)

    if (stepCanonicalId === canonicalId) {
      matched = true
      const mergedStatus = mergeStepStatus(step.status, status)
      const completedAt = status === 'completed' ? Date.now() : step.completedAt
      if (
        mergedStatus !== step.status ||
        completedAt !== step.completedAt ||
        stepCanonicalId !== step.id
      ) {
        updated = true
        return {
          ...step,
          id: stepCanonicalId,
          status: mergedStatus,
          completedAt
        }
      }
      if (stepCanonicalId !== step.id) {
        updated = true
        return { ...step, id: stepCanonicalId }
      }
      return step
    }

    if (stepCanonicalId !== step.id) {
      updated = true
      return {
        ...step,
        id: stepCanonicalId
      }
    }

    return step
  })

  let finalSteps = updatedSteps

  if (!matched) {
    updated = true
    finalSteps = [
      ...updatedSteps,
      {
        id: canonicalId,
        description: `Step ${updatedSteps.length + 1}`,
        status,
        result: undefined,
        completedAt: status === 'completed' ? Date.now() : undefined
      }
    ]
  }

  if (!updated) {
    return planDetails
  }

  const nextStatus = recomputePlanStatus(finalSteps)
  debugPlanSnapshot(`updateStepStatus(${canonicalId}, ${status})`, {
    ...planDetails,
    steps: finalSteps,
    status: nextStatus
  })

  return {
    ...planDetails,
    steps: finalSteps,
    status: nextStatus
  }
}

// Removed applyAgentPhaseToPlan since plan generation is handled as a workflow phase

// Removed finalizeStaticSteps since we no longer have static steps

class AgentApiClient {
  private baseUrl: string

  constructor() {
    // Always use same origin for the unified architecture
    if (typeof window !== "undefined") {
      this.baseUrl = window.location.origin
    } else {
      this.baseUrl = "http://localhost:8000"
    }
  }

  async sendMessage(request: ResponsesAgentRequest): Promise<ResponsesAgentResponse> {
    const response = await fetch(`${this.baseUrl}/invocations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  async *streamMessage(
    request: ResponsesAgentRequest
  ): AsyncGenerator<ResponsesAgentStreamEvent, void, unknown> {
    const streamRequest = { ...request, stream: true }

    const response = await fetch(`${this.baseUrl}/invocations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(streamRequest),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    if (!response.body) {
      throw new Error("No response body for streaming")
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()

            if (data === '[DONE]') {
              return
            }

            try {
              const event = JSON.parse(data) as ResponsesAgentStreamEvent
              yield event
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', data, parseError)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      })
      return response.ok
    } catch {
      return false
    }
  }
}

// Utility functions for progress tracking
function buildStructuredProgress(
  events: IntermediateEvent[],
  planDetails?: PlanMetadata,
  currentAgent?: string
): StructuredProgress {
  // CRITICAL FIX: Start from baseline to ensure phases are ALWAYS present
  // This prevents React from ever seeing an empty array during rebuilds
  const baseline = createInitialStructuredProgress()
  
  // Update each baseline phase with current status from events
  const workflowPhases: ProgressItem[] = baseline.workflowPhases.map(phase => {
    const status = determinePhaseStatus(phase.id, events, currentAgent, planDetails)
    return {
      ...phase,
      status,
      timestamp: getPhaseTimestamp(phase.id, events)
    }
  })

  if (PLAN_DEBUG_ENABLED) {
    console.log(`[buildStructuredProgress] Updated ${workflowPhases.length} phases from baseline`, {
      eventCount: events.length,
      currentAgent,
      statuses: workflowPhases.map(p => `${p.id}:${p.status}`).join(', ')
    })
  }

  // Build plan steps if available
  const planSteps: ProgressItem[] = []
  if (planDetails?.steps) {
    planDetails.steps.forEach((step: PlanStep, index: number) => {
      planSteps.push({
        id: step.id || canonicalizeStepId(index + 1, index),
        label: step.description || `Step ${index + 1}`,
        status: mapPlanStepStatus(step.status),
        stepNumber: index + 1,
        result: step.result,
        timestamp: step.completedAt,
        isWorkflowPhase: false
      })
    })
  }

  const allItems = [...workflowPhases, ...planSteps]
  const completedCount = allItems.filter(item => item.status === 'completed').length
  const overallProgress = allItems.length > 0 ? (completedCount / allItems.length) * 100 : 0

  return {
    workflowPhases,
    planSteps,
    currentAgent,
    currentPhase: currentAgent ? AGENT_PHASE_MAP[currentAgent] : undefined,
    overallProgress
  }
}

/**
 * Marks all phases and steps as completed with 100% progress.
 * Used when global completion events arrive (research_completed, report_generated, etc.)
 */
function markAllPhasesCompleted(current: StructuredProgress): StructuredProgress {
  return {
    ...current,
    workflowPhases: current.workflowPhases.map(phase => ({
      ...phase,
      status: 'completed' as const
    })),
    planSteps: current.planSteps.map(step => ({
      ...step,
      status: 'completed' as const
    })),
    overallProgress: 100
  }
}

/**
 * Checks if any workflow_phase event indicates the phase has started.
 * Backend emits: { event_type: 'workflow_phase', data: { phase: 'initiate', status: 'started' } }
 */
function hasPhaseStartedEvent(phaseId: string, events: IntermediateEvent[]): boolean {
  return events.some(e =>
    e.event_type === 'workflow_phase' &&
    e.data?.phase === phaseId &&
    (e.data as any)?.status === 'started'
  )
}

/**
 * Checks if any workflow_phase event indicates the phase has completed.
 * Backend emits: { event_type: 'workflow_phase', data: { phase: 'initiate', status: 'completed' } }
 */
function hasPhaseCompletedEvent(phaseId: string, events: IntermediateEvent[]): boolean {
  return events.some(e =>
    e.event_type === 'workflow_phase' &&
    e.data?.phase === phaseId &&
    (e.data as any)?.status === 'completed'
  )
}

/**
 * Checks if any global completion event has been emitted.
 * These events signal the entire research workflow is done.
 */
function hasGlobalCompletionEvent(events: IntermediateEvent[]): boolean {
  return events.some(e =>
    ['research_completed', 'report_generated', 'synthesis_completed'].includes(e.event_type)
  )
}

function determinePhaseStatus(
  phaseId: string,
  events: IntermediateEvent[],
  currentAgent?: string,
  planDetails?: any
): ProgressItem['status'] {
  // PRIORITY 1: Global completion events mark ALL phases complete
  if (hasGlobalCompletionEvent(events)) {
    return 'completed'
  }

  // PRIORITY 2: Explicit workflow_phase completion events
  if (hasPhaseCompletedEvent(phaseId, events)) {
    return 'completed'
  }

  // PRIORITY 3: Legacy phase_completed events
  const phaseCompletionEvents = events.filter(e =>
    e.event_type === 'phase_completed' && e.data?.phase === phaseId
  )
  if (phaseCompletionEvents.length > 0) {
    return 'completed'
  }

  // PRIORITY 4: Current agent match indicates active
  if (currentAgent && AGENT_PHASE_MAP[currentAgent] === phaseId) {
    return 'active'
  }

  // PRIORITY 5: Explicit workflow_phase started events
  if (hasPhaseStartedEvent(phaseId, events)) {
    return 'active'
  }

  // PRIORITY 6: Check agent_handoff events for activation
  const phaseEvents = events.filter(e => {
    const agent = e.data?.agent || e.data?.current_agent || e.data?.from_agent
    return agent && AGENT_PHASE_MAP[agent] === phaseId
  })

  if (phaseEvents.length > 0) {
    // Has events for this phase, check if completed
    const completedEvents = phaseEvents.filter(e =>
      ['completed', 'done', 'finished'].includes((e.data as any)?.status?.toLowerCase() || '')
    )
    if (completedEvents.length > 0) {
      return 'completed'
    }

    // Has events but not completed - check if later phase is active
    const currentPhaseOrder = WORKFLOW_PHASES.find(p => p.id === phaseId)?.order || 0
    const hasLaterPhaseEvents = events.some(e => {
      const agent = e.data?.agent || e.data?.current_agent
      const laterPhase = agent && AGENT_PHASE_MAP[agent]
      if (!laterPhase) return false
      const laterPhaseOrder = WORKFLOW_PHASES.find(p => p.id === laterPhase)?.order || 0
      return laterPhaseOrder > currentPhaseOrder
    })

    return hasLaterPhaseEvents ? 'completed' : 'active'
  }

  // PRIORITY 7: Fallback to transition rules (from phaseTransitionRules.ts)
  const ruleBasedStatus = evaluatePhaseStatus(phaseId, events, currentAgent, planDetails)
  if (ruleBasedStatus !== 'pending') {
    return ruleBasedStatus
  }

  // DEFAULT: pending (waiting to start)
  return 'pending'
}

function mapPlanStepStatus(stepStatus?: string): ProgressItem['status'] {
  switch (stepStatus) {
    case 'completed':
      return 'completed'
    case 'in_progress':
      return 'active'
    case 'skipped':
      return 'skipped'
    default:
      return 'pending'
  }
}

function getPhaseTimestamp(phaseId: string, events: IntermediateEvent[]): number | undefined {
  // Find the earliest relevant event for this phase
  const relevantEvents = events.filter(e => {
    const agent = e.data.agent || e.data.current_agent || e.data.from_agent
    return agent && AGENT_PHASE_MAP[agent] === phaseId
  })

  if (relevantEvents.length > 0) {
    return Math.min(...relevantEvents.map(e => e.timestamp * 1000))
  }

  return undefined
}

export function useAgentClient() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [intermediateEvents, setIntermediateEvents] = useState<IntermediateEvent[]>([])
  const [currentStreamingId, setCurrentStreamingId] = useState<string | null>(null)
  const [researchProgress, setResearchProgress] = useState<StructuredProgress>(
    createInitialStructuredProgress()  // All phases start as 'pending'
  )
  const clientRef = useRef(new AgentApiClient())
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentMetadataRef = useRef<ResearchMetadata>({})
  const intermediateEventsRef = useRef<IntermediateEvent[]>([])

  const addMessage = useCallback((message: Omit<ChatMessage, 'id'>): string => {
    const id = crypto.randomUUID()
    const newMessage: ChatMessage = {
      ...message,
      id,
      timestamp: message.timestamp || Date.now()
    }
    setMessages(prev => [...prev, newMessage])
    return id
  }, [])

  const updateMessage = useCallback((id: string, updates: Partial<ChatMessage>) => {
    // CRITICAL DEBUGGING: Log when we update message metadata
    if (updates.metadata?.researchProgress && PLAN_DEBUG_ENABLED) {
      console.log('📝 [updateMessage] Updating message with researchProgress:', {
        messageId: id,
        phases: updates.metadata.researchProgress.workflowPhases?.length,
        steps: updates.metadata.researchProgress.planSteps?.length,
        overallProgress: updates.metadata.researchProgress.overallProgress
      })
    }
    setMessages(prev =>
      prev.map(msg => msg.id === id ? { ...msg, ...updates } : msg)
    )
  }, [])

  const updateResearchProgress = useCallback((currentAgent?: string, planDetails?: PlanMetadata) => {
    if (PLAN_DEBUG_ENABLED) {
      console.debug('[ProgressDebug] updateResearchProgress', {
        currentAgent,
        planStatus: planDetails?.status,
        steps: planDetails?.steps?.length,
        eventCount: intermediateEventsRef.current.length
      })
    }
    const structuredProgress = buildStructuredProgress(intermediateEventsRef.current, planDetails, currentAgent)

    // Persist current snapshot into metadata ref so messages keep progress
    currentMetadataRef.current = {
      ...currentMetadataRef.current,
      researchProgress: structuredProgress
    }

    setResearchProgress(prev => {
      const structuredProgressWithTiming = {
        ...structuredProgress,
        startTime: prev.startTime ?? Date.now(),
        elapsedTime: prev.startTime ? (Date.now() - prev.startTime) / 1000 : undefined
      }

      if (PLAN_DEBUG_ENABLED) {
        console.log('[Progress] Updated research progress:', {
          phases: structuredProgressWithTiming.workflowPhases.length,
          steps: structuredProgressWithTiming.planSteps.length,
          currentAgent,
          overallProgress: structuredProgressWithTiming.overallProgress
        })
      }

      return structuredProgressWithTiming
    })
  }, [])

  const sendStreamingMessage = useCallback(async (userMessage: string) => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    // Add user message
    addMessage({ role: 'user', content: userMessage })

    // Add streaming assistant message
    const assistantMessageId = addMessage({
      role: 'assistant',
      content: '',
      isStreaming: true
    })

    setCurrentStreamingId(assistantMessageId)
    setIsStreaming(true)
    setIntermediateEvents([])
    intermediateEventsRef.current = []  // Reset ref for new session
    setResearchProgress({
      ...createInitialStructuredProgress(),  // Baseline with all phases
      startTime: Date.now()
    })
    currentMetadataRef.current = {}  // Reset metadata for new session

    try {
      const request: ResponsesAgentRequest = {
        input: [
          {
            type: "message",
            role: "user",
            content: userMessage
          }
        ],
        stream: true
      }

      let fullContent = ''
      let currentMetadata: ChatMessage['metadata'] = {
        planDetails: createInitialPlanDetails()
      }

      debugPlanSnapshot('initial-setup', currentMetadata.planDetails)

      updateMessage(assistantMessageId, {
        content: fullContent,
        isStreaming: true,
        metadata: currentMetadata
      })

      // Start with coordinator agent for progress tracking
      updateResearchProgress('coordinator', currentMetadata.planDetails)

      const streamGenerator = clientRef.current.streamMessage(request)

      for await (const event of streamGenerator) {
        if (abortController.signal.aborted) {
          break
        }

        switch (event.type) {
          case 'response.output_text.delta':
            if (event.delta) {
              fullContent += event.delta

              // Apply content filtering to remove progress markers
              const filterResult = filterContent(fullContent)
              const displayContent = filterResult.cleanContent

              // Debug logging for streaming content
              if (filterResult.hasProgressMarkers) {
                console.log('🔧 [DEBUG] Streaming content filtering:')
                console.log('📥 Delta:', event.delta)
                console.log('📤 Full filtered:', displayContent.substring(Math.max(0, displayContent.length - 100)))
                console.log('🗑️ Markers found:', filterResult.removedMarkers.slice(-3))
              }

              // Extract phase info from raw content before filtering
              const phaseInfo = extractPhaseInfo(event.delta)

              // Update research progress if phase info is available
              if (phaseInfo.phase || phaseInfo.node) {
                const currentAgent = phaseInfo.node || phaseInfo.phase
                debugStreamEvent('phase-marker', {
                  phase: phaseInfo.phase,
                  node: phaseInfo.node,
                  agent: currentAgent,
                })
                // Update message with filtered content
                updateMessage(assistantMessageId, {
                  content: displayContent,
                  isStreaming: true,
                  metadata: currentMetadataRef.current  // Include persisted metadata
                })
                updateResearchProgress(currentAgent, currentMetadata.planDetails)
              } else {
                updateMessage(assistantMessageId, {
                  content: displayContent,
                  isStreaming: true,
                  metadata: currentMetadataRef.current  // Include persisted metadata
                })
              }
            }
            break

          case 'response.output_item.done':
            if (event.item?.content) {
              const finalText = event.item.content
                .map(c => c.type === 'output_text' ? c.text : '')
                .join('')

              // CRITICAL: Apply content filtering to final text to remove all progress markers
              const filterResult = filterContent(finalText)
              const cleanFinalContent = filterResult.cleanContent

              // Debug logging to verify filtering is working
              if (filterResult.hasProgressMarkers) {
                console.log('🔧 [DEBUG] Content filtering applied:')
                console.log('📥 Raw final text:', finalText.substring(0, 200) + '...')
                console.log('📤 Filtered text:', cleanFinalContent.substring(0, 200) + '...')
                console.log('🗑️ Removed markers:', filterResult.removedMarkers.length)
              }

              // Final research progress update with reporter agent
              updateResearchProgress('reporter', currentMetadata.planDetails)

              // CRITICAL FIX: Persist final research progress in metadata
              const finalProgress = buildStructuredProgress(intermediateEventsRef.current, currentMetadata.planDetails, 'reporter')
              currentMetadata.researchProgress = {
                ...finalProgress,
                overallProgress: finalProgress.overallProgress || 100  // Ensure 100% on completion
              }
              currentMetadataRef.current = currentMetadata  // Sync ref with local

              console.log('✅ [output_item.done] Persisting final progress to message:', {
                messageId: assistantMessageId,
                phases: finalProgress.workflowPhases.length,
                steps: finalProgress.planSteps.length,
                overallProgress: currentMetadata.researchProgress.overallProgress
              })

              updateMessage(assistantMessageId, {
                content: cleanFinalContent,
                isStreaming: false,
                metadata: currentMetadata
              })

              // End the streaming state
              setIsStreaming(false)
              setCurrentStreamingId(null)
            }
            break

          case 'intermediate_event':
            if (event.intermediate_event) {
              // Debug log all intermediate events
              console.group('📊 [EVENT DEBUG]', event.intermediate_event.event_type)
              console.log('Type:', event.intermediate_event.event_type)
              console.log('Data:', event.intermediate_event.data)
              console.log('Meta:', event.intermediate_event.meta)
              console.groupEnd()

              // Add to intermediate events for live tracking
              const intermediateEvent: IntermediateEvent = {
                id: crypto.randomUUID(),
                timestamp: event.intermediate_event.timestamp || Date.now() / 1000,
                correlation_id: event.intermediate_event.correlation_id || assistantMessageId,
                sequence: Date.now(),
                event_type: event.intermediate_event.event_type,
                data: event.intermediate_event.data || {},
                meta: event.intermediate_event.meta || {}
              }

              // CRITICAL: Update ref BEFORE setState (setState is async, ref is sync)
              const updatedEvents = [...intermediateEventsRef.current, intermediateEvent]
              intermediateEventsRef.current = updatedEvents
              setIntermediateEvents(updatedEvents)

              const eventAgent = event.intermediate_event.data?.agent ||
                                 event.intermediate_event.data?.current_agent ||
                                 event.intermediate_event.data?.from_agent

              const normalizedEventType = event.intermediate_event.event_type?.toLowerCase?.() || ''
              console.log('🔍 All intermediate events:', {
                eventType: normalizedEventType,
                originalType: event.intermediate_event.event_type,
                stepId: event.intermediate_event.data?.step_id,
                status: event.intermediate_event.data?.status,
                isStepEvent: STEP_EVENT_TYPES.has(normalizedEventType),
                isPlanEvent: PLAN_EVENT_TYPES.has(normalizedEventType),
                fullData: event.intermediate_event.data
              })
              debugStreamEvent('intermediate_event', {
                type: normalizedEventType,
                rawStep: event.intermediate_event.data?.step_id,
                status: event.intermediate_event.data?.status,
              })

              let planDetailsWorking: PlanMetadata | undefined = currentMetadata.planDetails
              let planChanged = false
              let progressAgent = eventAgent

              if (PLAN_EVENT_TYPES.has(normalizedEventType)) {
                const planPayload = event.intermediate_event.data?.plan || event.intermediate_event.data

                // Only rebuild plan if it's actually new or changed
                const shouldRebuildPlan = !planDetailsWorking ||
                  normalizedEventType === 'plan_created' ||
                  normalizedEventType === 'plan_updated' ||
                  normalizedEventType === 'plan_ready' ||
                  (planPayload.steps && planPayload.steps.length !== planDetailsWorking.steps.length)

                if (shouldRebuildPlan) {
                  const mergedPlan = mergeMetadataPlanDetails(planDetailsWorking, planPayload)
                  if (mergedPlan) {
                    planDetailsWorking = mergedPlan
                    planChanged = true

                    // IMMEDIATELY update UI with plan
                    currentMetadata.planDetails = planDetailsWorking
                    updateMessage(assistantMessageId, {
                      content: fullContent,
                      isStreaming: true,
                      metadata: currentMetadata
                    })

                    // For plan_ready events, the researcher should now be active
                    progressAgent = normalizedEventType === 'plan_ready' ? 'researcher' : 'planner'

                    // Force progress update to show plan immediately
                    updateResearchProgress(progressAgent, planDetailsWorking)

                    debugPlanSnapshot('plan-event merge (rebuilt with immediate display)', mergedPlan)
                  }
                } else {
                  debugPlanSnapshot('plan-event skipped (no rebuild needed)', planDetailsWorking)
                }
              } else {
                const lifecycleStatus = PLAN_LIFECYCLE_EVENT_TYPES.get(normalizedEventType)
                if (lifecycleStatus) {
                  progressAgent = PLAN_LIFECYCLE_PROGRESS_AGENT[normalizedEventType] || progressAgent
                  // Plan lifecycle events are now handled as workflow phases, not individual steps
                }
              }

              if (STEP_EVENT_TYPES.has(normalizedEventType)) {
                const rawStepId = event.intermediate_event.data?.step_id || event.intermediate_event.data?.id
                console.log('📍 Step event received:', {
                  eventType: normalizedEventType,
                  rawStepId,
                  canonicalizedId: canonicalizeStepId(rawStepId),
                  eventData: event.intermediate_event.data,
                  planExists: !!planDetailsWorking,
                  planSteps: planDetailsWorking?.steps.map((s: PlanStep) => ({id: s.id, canonical: canonicalizeStepId(s.id)}))
                })
                if (rawStepId) {
                  const status: StepStatus = normalizedEventType === 'step_completed'
                    ? 'completed'
                    : normalizedEventType === 'step_failed'
                      ? 'skipped'
                      : 'in_progress'

                  if (!progressAgent) {
                    progressAgent = 'researcher'
                  }

                  // Idempotent check: Only update if status actually changed
                  const currentStep = currentMetadataRef.current.researchProgress?.planSteps
                    ?.find(s => canonicalizeStepId(s.id) === canonicalizeStepId(rawStepId))

                  if (currentStep?.status !== status) {
                    const updatedPlan = updateStepStatusInPlan(planDetailsWorking, rawStepId, status)
                    if (updatedPlan) {
                      planDetailsWorking = updatedPlan
                      planChanged = true
                      console.log('✅ Step status updated successfully for step:', rawStepId, 'to status:', status)
                    } else {
                      console.log('❌ Failed to update step status for step:', rawStepId, 'to status:', status)
                    }
                  } else {
                    console.log('⏭️ Step status unchanged, skipping:', rawStepId, status)
                  }

                }
              }

              // CRITICAL FIX: Handle step_added events to show dynamic plan items immediately
              if (normalizedEventType === 'step_added') {
                const stepData = event.intermediate_event.data
                console.log('📍 Step added event received:', stepData)

                if (stepData?.step_id && stepData?.step_title && planDetailsWorking) {
                  const newStep: PlanStep = {
                    id: stepData.step_id,
                    description: stepData.step_title,
                    status: 'pending',
                    result: undefined,
                    completedAt: undefined
                  }

                  // Insert the new step at the specified index or at the end
                  const insertIndex = stepData.index !== undefined ? stepData.index : planDetailsWorking.steps.length
                  const updatedSteps: PlanStep[] = [...planDetailsWorking.steps]
                  updatedSteps.splice(insertIndex, 0, newStep)

                  planDetailsWorking = {
                    ...planDetailsWorking,
                    steps: updatedSteps
                  }
                  planChanged = true

                  console.log('✅ Dynamic step added to plan:', stepData.step_title, 'at index:', insertIndex)

                  // Force immediate progress update
                  if (!progressAgent) {
                    progressAgent = 'researcher'
                  }
                }
              }

              // CRITICAL FIX: Handle workflow_phase events with idempotent state checks
              if (normalizedEventType === 'workflow_phase') {
                const phaseStatus = event.intermediate_event.data?.status
                const phaseName = event.intermediate_event.data?.phase

                // Get current phase state to check if update is needed
                const currentPhase = currentMetadataRef.current.researchProgress?.workflowPhases
                  ?.find(p => p.id === phaseName)

                if (phaseStatus === 'started') {
                  // Only update if phase is NOT already active (idempotent check)
                  if (currentPhase?.status !== 'active') {
                    console.log(`🟢 Phase started: ${phaseName}`)
                    updateResearchProgress(progressAgent, planDetailsWorking)
                  } else {
                    console.log(`⏭️ Phase already active, skipping: ${phaseName}`)
                  }
                } else if (phaseStatus === 'completed') {
                  // Only update if phase is NOT already completed (idempotent check)
                  if (currentPhase?.status !== 'completed') {
                    console.log(`✅ Phase completed: ${phaseName}`)
                    updateResearchProgress(progressAgent, planDetailsWorking)
                  } else {
                    console.log(`⏭️ Phase already completed, skipping: ${phaseName}`)
                  }
                }
              }

              // CRITICAL FIX: Handle global research completion events
              // Handle global research completion events
              if (['research_completed', 'report_generated', 'synthesis_completed'].includes(normalizedEventType)) {
                console.log(`✅ Global completion event: ${normalizedEventType} - marking all phases complete`)

                // Mark everything complete and update both live state and metadata
                setResearchProgress(prev => {
                  const completed = markAllPhasesCompleted(prev)
                  // Update metadata ref
                  currentMetadataRef.current = {
                    ...currentMetadataRef.current,
                    researchProgress: completed
                  }
                  // Also update local metadata object
                  currentMetadata = {
                    ...currentMetadata,
                    researchProgress: completed
                  }
                  return completed
                })

                // Update message content if available
                if (fullContent) {
                  updateMessage(assistantMessageId, {
                    content: fullContent,
                    metadata: currentMetadataRef.current
                  })
                }
              }

              // Agent phase transitions are now handled by workflow phase status tracking

              if (planChanged && planDetailsWorking) {
                currentMetadata = {
                  ...currentMetadata,
                  planDetails: planDetailsWorking
                }
                debugPlanSnapshot(`plan-update ${normalizedEventType}`, planDetailsWorking)

                updateMessage(assistantMessageId, {
                  content: fullContent,
                  metadata: currentMetadata
                })
              }

              // NOTE: Removed unconditional updateResearchProgress call here
              // All updates are now handled in specific event handlers with idempotent checks
              // This prevents duplicate/unnecessary re-renders
            }
            break

          case 'response.metadata':
            if (event.metadata) {
              const metadataUpdate: Record<string, any> = { ...event.metadata }
              const planPayload = metadataUpdate.planDetails ?? metadataUpdate.plan_details ?? metadataUpdate.plan

              let planDetailsWorking: PlanMetadata | undefined = currentMetadata.planDetails
              const metadataAgent = metadataUpdate.currentAgent || metadataUpdate.current_agent

              if (planPayload) {
                const normalizedPlan = mergeMetadataPlanDetails(planDetailsWorking, planPayload)
                if (normalizedPlan) {
                  metadataUpdate.planDetails = normalizedPlan
                  delete metadataUpdate.plan
                  delete metadataUpdate.plan_details
                  planDetailsWorking = normalizedPlan
                }
              }

              currentMetadata = {
                ...currentMetadata,
                ...metadataUpdate,
                planDetails: planDetailsWorking
              }
              debugPlanSnapshot('metadata-update', planDetailsWorking)

              if (metadataAgent) {
                updateResearchProgress(metadataAgent, planDetailsWorking)
              } else {
                updateResearchProgress(undefined, planDetailsWorking)
              }

              debugStreamEvent('metadata', {
                agent: metadataAgent,
                planStatus: planDetailsWorking?.status,
                stepCount: planDetailsWorking?.steps.length,
              })

              updateMessage(assistantMessageId, {
                content: fullContent,
                metadata: currentMetadata
              })
            }
            break
        }
      }

    } catch (error) {
      console.error('Stream error:', error)
      if (!abortController.signal.aborted) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        updateMessage(assistantMessageId, {
          content: `Error processing request: ${errorMessage}`,
          isStreaming: false
        })
      }
      // Also end the streaming state on error
      setIsStreaming(false)
      setCurrentStreamingId(null)
    } finally {
      abortControllerRef.current = null
    }
  }, [addMessage, updateMessage])

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsStreaming(false)
      setCurrentStreamingId(null)
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    setIntermediateEvents([])
    intermediateEventsRef.current = []  // Reset ref too
    setCurrentStreamingId(null)
    setResearchProgress(createInitialStructuredProgress())  // Baseline state
    currentMetadataRef.current = {}  // Reset metadata
  }, [])

  return {
    messages,
    isStreaming,
    intermediateEvents,
    currentStreamingId,
    researchProgress,
    sendStreamingMessage,
    stopStreaming,
    clearMessages,
    client: clientRef.current
  }
}
