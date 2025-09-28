import { useCallback, useRef, useState } from 'react'
import { IntermediateEvent, PlanMetadata, PlanStep, ResearchMetadata } from '../types/agents'
import { ProgressItem, StructuredProgress, WORKFLOW_PHASES, AGENT_PHASE_MAP } from '../types/progress'
import { filterContent, extractPhaseInfo } from '../utils/contentFilter'

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

const PLAN_GENERATION_STEP_ID = 'step_plan_generation'

const STEP_ID_ALIASES: Record<string, string> = {
  step_plan: PLAN_GENERATION_STEP_ID,
  step_plan_created: PLAN_GENERATION_STEP_ID,
  step_plan_generation: PLAN_GENERATION_STEP_ID,
  step_planner: PLAN_GENERATION_STEP_ID,
  step_planning: PLAN_GENERATION_STEP_ID,
  step_background_investigation: PLAN_GENERATION_STEP_ID,
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
  'plan_structure_visualize'
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

const PLAN_LIFECYCLE_STEP_IDS: Record<string, string> = {
  plan_started: PLAN_GENERATION_STEP_ID,
  plan_completed: PLAN_GENERATION_STEP_ID,
  plan_failed: PLAN_GENERATION_STEP_ID
}

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

  return PLAN_GENERATION_STEP_ID
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

function createStaticStep(id: string, description: string, status: StepStatus = 'pending'): PlanStep {
  return {
    id,
    description,
    status
  }
}

const SPECIAL_STEP_IDS = new Set([PLAN_GENERATION_STEP_ID])

const INITIAL_PLACEHOLDER_STEPS: PlanStep[] = [
  createStaticStep(PLAN_GENERATION_STEP_ID, 'Generating detailed research plan', 'in_progress')
]


const STEP_DESCRIPTION_ALIASES: Record<string, string> = {
  [PLAN_GENERATION_STEP_ID]: 'Generating detailed research plan'
}


function clonePlanStep(step: PlanStep): PlanStep {
  return {
    id: step.id,
    description: step.description,
    status: step.status,
    result: step.result,
    completedAt: step.completedAt
  }
}

function createInitialPlanDetails(): PlanMetadata {
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

  const planGenerationExisting = existingMap.get(PLAN_GENERATION_STEP_ID) ?? INITIAL_PLACEHOLDER_STEPS[0]
  const planGenerationStep: PlanStep = {
    ...clonePlanStep(planGenerationExisting),
    status: 'completed',
    completedAt: planGenerationExisting.completedAt ?? Date.now()
  }

  const mergedDynamicSteps = mergeDynamicSteps(
    existingSteps.filter(step => !SPECIAL_STEP_IDS.has(step.id)),
    dynamicSteps
  )

  const combined: PlanStep[] = [
    planGenerationStep,
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
        description: STEP_DESCRIPTION_ALIASES[canonicalId] || `Step ${updatedSteps.length + 1}`,
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

function applyAgentPhaseToPlan(
  planDetails: PlanMetadata | undefined,
  agent?: string
): PlanMetadata | undefined {
  if (!planDetails || !agent) return planDetails

  const normalizedAgent = agent.toLowerCase()
  let updatedPlan = planDetails

  const applyStatus = (id: string, status: StepStatus) => {
    const nextPlan = updateStepStatusInPlan(updatedPlan, id, status)
    if (nextPlan) {
      updatedPlan = nextPlan
    }
  }

  if (normalizedAgent.includes('planner')) {
    applyStatus(PLAN_GENERATION_STEP_ID, 'in_progress')
  }

  if (normalizedAgent.includes('research')) {
    applyStatus(PLAN_GENERATION_STEP_ID, 'completed')
  }

  if (normalizedAgent.includes('fact')) {
    applyStatus(PLAN_GENERATION_STEP_ID, 'completed')
  }

  if (normalizedAgent.includes('report')) {
    applyStatus(PLAN_GENERATION_STEP_ID, 'completed')
  }

  debugPlanSnapshot(`applyAgentPhase(${agent})`, updatedPlan)

  return updatedPlan
}

function finalizeStaticSteps(planDetails: PlanMetadata | undefined): PlanMetadata | undefined {
  if (!planDetails) return planDetails

  let updated = planDetails
  const finalize = (id: string) => {
    const next = updateStepStatusInPlan(updated, id, 'completed')
    if (next) {
      updated = next
    }
  }

  finalize(PLAN_GENERATION_STEP_ID)

  debugPlanSnapshot('finalizeStaticSteps', updated)

  return updated
}

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
  const workflowPhases: ProgressItem[] = []
  const planSteps: ProgressItem[] = []

  // Build workflow phases (sorted by order)
  const sortedPhases = [...WORKFLOW_PHASES].sort((a, b) => a.order - b.order)
  sortedPhases.forEach(phase => {
    const status = determinePhaseStatus(phase.id, events, currentAgent, planDetails)
    workflowPhases.push({
      id: phase.id,
      label: phase.name,
      status,
      isWorkflowPhase: true,
      timestamp: getPhaseTimestamp(phase.id, events)
    })
  })

  // Build plan steps if available
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


function determinePhaseStatus(
  phaseId: string,
  events: IntermediateEvent[],
  currentAgent?: string,
  planDetails?: any
): ProgressItem['status'] {
  // Check if this phase is currently active
  if (currentAgent && AGENT_PHASE_MAP[currentAgent] === phaseId) {
    return 'active'
  }

  // Check events for completion/activation
  const phaseEvents = events.filter(e => {
    const agent = e.data.agent || e.data.current_agent || e.data.from_agent
    return agent && AGENT_PHASE_MAP[agent] === phaseId
  })

  if (phaseEvents.some(e => e.event_type.includes('complete') || e.event_type.includes('done'))) {
    return 'completed'
  }

  if (phaseEvents.some(e => e.event_type.includes('start') || e.event_type.includes('begin'))) {
    return 'active'
  }

  // Special logic for different phases
  switch (phaseId) {
    case 'initiate':
      // Look for coordinator handoff events or phase completion
      const coordinatorEvents = events.filter(e => {
        const agent = e.data.agent || e.data.current_agent || e.data.from_agent
        return agent === 'coordinator' || e.event_type.includes('handoff')
      })

      // If coordinator handed off to another agent, initiate is complete
      const handoffEvents = events.filter(e =>
        e.event_type.includes('handoff') &&
        e.data.from_agent === 'coordinator'
      )

      if (handoffEvents.length > 0) {
        return 'completed'
      }

      // If we have coordinator events but no handoff yet, it's active
      if (coordinatorEvents.length > 0) {
        return currentAgent === 'coordinator' ? 'active' : 'completed'
      }

      // If no events yet, it's pending
      return 'pending'

    case 'planning':
      return planDetails ? 'completed' : 'pending'

    case 'fact_checking':
      // Active if research is done but not synthesizing yet
      if (planDetails?.status === 'completed' && currentAgent !== 'reporter') {
        return 'active'
      }
      return planDetails?.status === 'completed' ? 'completed' : 'pending'

    case 'synthesizing':
      return currentAgent === 'reporter' ? 'active' : 'pending'

    default:
      return 'pending'
  }
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
  const [researchProgress, setResearchProgress] = useState<StructuredProgress>({
    workflowPhases: [],
    planSteps: [],
    overallProgress: 0
  })
  const clientRef = useRef(new AgentApiClient())
  const abortControllerRef = useRef<AbortController | null>(null)

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
    setMessages(prev =>
      prev.map(msg => msg.id === id ? { ...msg, ...updates } : msg)
    )
  }, [])

  const updateResearchProgress = useCallback((currentAgent?: string, planDetails?: PlanMetadata) => {
    if (PLAN_DEBUG_ENABLED) {
      console.debug('[ProgressDebug] updateResearchProgress', {
        currentAgent,
        planStatus: planDetails?.status,
        steps: planDetails?.steps?.length
      })
    }
    setResearchProgress(prev => {
      const structuredProgress = buildStructuredProgress(intermediateEvents, planDetails, currentAgent)

      return {
        ...structuredProgress,
        startTime: prev.startTime,
        elapsedTime: prev.startTime ? (Date.now() - prev.startTime) / 1000 : undefined
      }
    })
  }, [intermediateEvents])

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
    setResearchProgress({
      workflowPhases: [],
      planSteps: [],
      overallProgress: 0,
      startTime: Date.now()
    })

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

      debugPlanSnapshot('initial-placeholders', currentMetadata.planDetails)

      currentMetadata = {
        ...currentMetadata,
        planDetails: applyAgentPhaseToPlan(currentMetadata.planDetails, 'planner')
      }

      debugPlanSnapshot('after-initial-phase', currentMetadata.planDetails)

      updateMessage(assistantMessageId, {
        content: fullContent,
        isStreaming: true,
        metadata: currentMetadata
      })

      updateResearchProgress('planner', currentMetadata.planDetails)

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
                const adjustedPlan = applyAgentPhaseToPlan(currentMetadata.planDetails, currentAgent)
                if (adjustedPlan && adjustedPlan !== currentMetadata.planDetails) {
                  currentMetadata = {
                    ...currentMetadata,
                    planDetails: adjustedPlan
                  }
                  updateMessage(assistantMessageId, {
                    content: displayContent,
                    isStreaming: true,
                    metadata: currentMetadata
                  })
                } else {
                  updateMessage(assistantMessageId, {
                    content: displayContent,
                    isStreaming: true
                  })
                }
                updateResearchProgress(currentAgent, currentMetadata.planDetails)
              } else {
                updateMessage(assistantMessageId, {
                  content: displayContent,
                  isStreaming: true
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

              const finalizedPlan = finalizeStaticSteps(currentMetadata.planDetails)
              if (finalizedPlan) {
                currentMetadata = {
                  ...currentMetadata,
                  planDetails: finalizedPlan
                }
                updateResearchProgress('reporter', finalizedPlan)
              } else {
                updateResearchProgress('reporter', currentMetadata.planDetails)
              }

              updateMessage(assistantMessageId, {
                content: cleanFinalContent,
                isStreaming: false,
                metadata: currentMetadata
              })
            }
            break

          case 'intermediate_event':
            if (event.intermediate_event) {
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

              setIntermediateEvents(prev => [...prev, intermediateEvent])

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
                  (planPayload.steps && planPayload.steps.length !== planDetailsWorking.steps.length)

                if (shouldRebuildPlan) {
                  const mergedPlan = mergeMetadataPlanDetails(planDetailsWorking, planPayload)
                  if (mergedPlan) {
                    planDetailsWorking = mergedPlan
                    planChanged = true
                    progressAgent = 'planner'
                    debugPlanSnapshot('plan-event merge (rebuilt)', mergedPlan)
                  }
                } else {
                  debugPlanSnapshot('plan-event skipped (no rebuild needed)', planDetailsWorking)
                }
              } else {
                const lifecycleStatus = PLAN_LIFECYCLE_EVENT_TYPES.get(normalizedEventType)
                if (lifecycleStatus) {
                  const lifecycleStepId = PLAN_LIFECYCLE_STEP_IDS[normalizedEventType]
                  if (lifecycleStepId) {
                    const lifecycleStepStatus: StepStatus = lifecycleStatus as StepStatus
                    const adjustedPlan = updateStepStatusInPlan(planDetailsWorking, lifecycleStepId, lifecycleStepStatus)
                    if (adjustedPlan) {
                      planDetailsWorking = adjustedPlan
                      planChanged = true
                    }
                  }

                  progressAgent = PLAN_LIFECYCLE_PROGRESS_AGENT[normalizedEventType] || progressAgent

                  if (normalizedEventType === 'plan_started') {
                    const placeholdersUpdated = updateStepStatusInPlan(planDetailsWorking, PLAN_GENERATION_STEP_ID, 'in_progress')
                    if (placeholdersUpdated) {
                      planDetailsWorking = placeholdersUpdated
                      planChanged = true
                    }
                  }

                  if (normalizedEventType === 'plan_completed') {
                    const finalizedPlaceholders = finalizeStaticSteps(planDetailsWorking)
                    if (finalizedPlaceholders) {
                      planDetailsWorking = finalizedPlaceholders
                      planChanged = true
                    }
                  }
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

                  const updatedPlan = updateStepStatusInPlan(planDetailsWorking, rawStepId, status)
                  if (updatedPlan) {
                    planDetailsWorking = updatedPlan
                    planChanged = true
                    console.log('✅ Step status updated successfully for step:', rawStepId, 'to status:', status)
                  } else {
                    console.log('❌ Failed to update step status for step:', rawStepId, 'to status:', status)
                  }

                }
              }

              if (eventAgent) {
                const phaseAdjusted = applyAgentPhaseToPlan(planDetailsWorking, eventAgent)
                if (phaseAdjusted && phaseAdjusted !== planDetailsWorking) {
                  planDetailsWorking = phaseAdjusted
                  planChanged = true
                }
              }

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

              if (planDetailsWorking) {
                updateResearchProgress(progressAgent, planDetailsWorking)
              }
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
                const adjusted = applyAgentPhaseToPlan(currentMetadata.planDetails, metadataAgent)
                if (adjusted && adjusted !== currentMetadata.planDetails) {
                  currentMetadata = {
                    ...currentMetadata,
                    planDetails: adjusted
                  }
                  planDetailsWorking = adjusted
                }
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
    } finally {
      setIsStreaming(false)
      setCurrentStreamingId(null)
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
    setCurrentStreamingId(null)
    setResearchProgress({
      workflowPhases: [],
      planSteps: [],
      overallProgress: 0
    })
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
