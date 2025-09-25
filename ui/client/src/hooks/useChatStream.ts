import { useCallback, useRef } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { StreamEvent, ChatRequest, IntermediateEventType, PlanMetadata, PlanStep, ResearchMetadata } from '@/types/chat'
import {
  processStreamingWithTableReconstruction
} from '@/utils/tableStreamReconstructor'
import { filterContent } from '@/utils/contentFilter'

const isObjectRecord = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const toNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : undefined
  }
  return undefined
}

const normalizeStepStatus = (status?: string): PlanStep['status'] => {
  if (!status) return 'pending'
  const normalized = status
    .toString()
    .trim()
    .toLowerCase()
    .replace(/^step_/, '')
    .replace(/\s+/g, '_')
    .replace(/-+/g, '_')

  switch (normalized) {
    case 'completed':
    case 'complete':
    case 'done':
      return 'completed'
    case 'in_progress':
    case 'inprogress':
    case 'active':
    case 'running':
      return 'in_progress'
    case 'skipped':
    case 'cancelled':
      return 'skipped'
    case 'failed':
    case 'error':
      return 'skipped'
    default:
      return 'pending'
  }
}

const computePlanStatus = (steps: PlanStep[]): PlanMetadata['status'] => {
  if (!steps.length) {
    return 'draft'
  }

  const total = steps.length
  const completed = steps.filter(step => step.status === 'completed').length
  const inProgress = steps.some(step => step.status === 'in_progress')

  if (completed === total) {
    return 'completed'
  }

  if (completed > 0 || inProgress) {
    return 'executing'
  }

  return 'draft'
}

const normalizeStepIdentifier = (value: unknown): string | undefined => {
  if (value === undefined || value === null) {
    return undefined
  }

  // Defensive coding: ensure value has toString method
  if (typeof value === 'object' && !('toString' in value)) {
    return undefined
  }

  let raw: string
  try {
    raw = value.toString().trim().toLowerCase()
  } catch (e) {
    console.warn('[useChatStream] Error converting value to string:', value, e)
    return undefined
  }

  if (!raw) {
    return undefined
  }

  const stepMatch = raw.match(/step[^0-9]*(\d+)/)
  if (stepMatch) {
    const numeric = parseInt(stepMatch[1], 10)
    if (!Number.isNaN(numeric)) {
      return `step_${numeric}`
    }
  }

  return raw.replace(/[^a-z0-9]+/g, '_')
}

const canonicalizeStepId = (stepId: string | number | unknown, fallbackIndex?: number): string => {
  // MUST convert any format to "step_XXX" with zero padding per specification
  if (typeof stepId === 'string') {
    const match = stepId.match(/(\d+)/)
    if (match) {
      return `step_${match[1].padStart(3, '0')}`
    }
  }
  
  if (typeof stepId === 'number') {
    return `step_${stepId.toString().padStart(3, '0')}`
  }
  
  // Fallback
  const index = typeof fallbackIndex === 'number' ? fallbackIndex : 0
  return `step_${(index + 1).toString().padStart(3, '0')}`
}

const getHigherPriorityStatus = (status1: PlanStep['status'], status2: PlanStep['status']): PlanStep['status'] => {
  // Priority: completed > in_progress > skipped > pending
  const priority = { completed: 4, in_progress: 3, skipped: 2, pending: 1 }
  return priority[status1] >= priority[status2] ? status1 : status2
}

const computeOverallStatus = (steps: PlanStep[]): PlanMetadata['status'] => {
  if (!steps.length) return 'draft'
  
  const total = steps.length
  const completed = steps.filter(step => step.status === 'completed').length
  const inProgress = steps.some(step => step.status === 'in_progress')
  
  if (completed === total) return 'completed'
  if (completed > 0 || inProgress) return 'executing'
  return 'draft'
}

const mergePlanStepStatus = (current?: PlanStep['status'], incoming?: PlanStep['status']): PlanStep['status'] => {
  const normalizedCurrent = current ?? 'pending'
  const normalizedIncoming = incoming ?? 'pending'

  if (normalizedIncoming === 'completed' || normalizedCurrent === 'completed') {
    return 'completed'
  }

  if (normalizedIncoming === 'in_progress' || normalizedCurrent === 'in_progress') {
    return 'in_progress'
  }

  if (normalizedIncoming === 'skipped') {
    return normalizedCurrent === 'pending' ? 'skipped' : normalizedCurrent
  }

  if (normalizedCurrent === 'skipped') {
    return normalizedIncoming === 'pending' ? 'skipped' : normalizedIncoming
  }

  return normalizedIncoming
}

const buildPlanDetailsFromPayload = (plan: unknown): PlanMetadata | undefined => {
  if (!isObjectRecord(plan)) {
    return undefined
  }

  const planRecord = plan as Record<string, any>
  
  // Defensive coding: ensure steps exists and is an array
  let rawSteps: unknown[] = []
  try {
    if (Array.isArray(planRecord.steps)) {
      rawSteps = planRecord.steps
    } else if (planRecord.steps && typeof planRecord.steps === 'object') {
      // Try to convert object to array if it has numeric keys
      const keys = Object.keys(planRecord.steps)
      const numericKeys = keys.filter(k => /^\d+$/.test(k)).sort((a, b) => Number(a) - Number(b))
      if (numericKeys.length > 0) {
        rawSteps = numericKeys.map(k => planRecord.steps[k])
      }
    }
  } catch (e) {
    console.warn('[useChatStream] Error processing plan steps:', e)
    return undefined
  }

  if (!rawSteps.length) {
    return undefined
  }

  const seenCanonicalIds = new Set<string>()

  const steps: PlanStep[] = rawSteps.map((rawStep, index) => {
    const step = isObjectRecord(rawStep) ? rawStep : {}
    // CRITICAL: Handle step_id field correctly per specification
    const idCandidates = [step.step_id, step.id, step.stepId]
    let id = canonicalizeStepId(idCandidates.find(value => typeof value === 'string' && value.trim().length > 0), index)

    while (seenCanonicalIds.has(id)) {
      id = canonicalizeStepId(`${id}_${index}`, index)
    }
    seenCanonicalIds.add(id)

    const descriptionCandidates = [step.description, step.title, step.content, step.summary]
    const description = descriptionCandidates.find(value => typeof value === 'string' && value.trim().length > 0)
      || `Step ${index + 1}`

    const result = typeof step.result === 'string' ? step.result
      : typeof step.summary === 'string' ? step.summary
      : undefined

    const completedAt = typeof step.completedAt === 'number' ? step.completedAt
      : typeof step.completed_at === 'number' ? step.completed_at
      : undefined

    const statusSource = typeof step.status === 'string'
      ? step.status
      : typeof step.state === 'string'
        ? step.state
        : typeof step.step_status === 'string'
          ? step.step_status
          : undefined

    return {
      id,
      description,
      status: normalizeStepStatus(statusSource),
      result,
      completedAt
    }
  })

  const canonicalSteps = steps.map((step, index) => ({
    ...step,
    id: canonicalizeStepId(step.id, index)
  }))

  const iterationsRaw = planRecord.iterations ?? planRecord.iteration ?? 1
  const iterations = toNumber(iterationsRaw) ?? 1

  const qualityCandidates = [
    planRecord.quality_assessment?.overall_score,
    planRecord.qualityAssessment?.overallScore,
    planRecord.quality,
    planRecord.confidence,
    planRecord.score
  ]
  const quality = qualityCandidates.find(value => typeof value === 'number' && Number.isFinite(value))

  const hasEnoughContext = Boolean(
    planRecord.hasEnoughContext ??
    planRecord.has_enough_context ??
    planRecord.enoughContext ??
    planRecord.context_ready
  )

  return {
    steps: canonicalSteps,
    iterations: iterations > 0 ? iterations : 1,
    quality: typeof quality === 'number' ? quality : undefined,
    status: computePlanStatus(canonicalSteps),
    hasEnoughContext
  }
}

const mergePlanDetails = (existing?: PlanMetadata, incoming?: PlanMetadata): PlanMetadata => {
  // MUST preserve progress when merging plan updates per specification
  if (!existing) return incoming || { steps: [], status: 'draft', iterations: 1, hasEnoughContext: true }
  if (!incoming) return existing
  
  const existingSteps = new Map(existing.steps.map(step => [step.id, step]))
  
  const mergedSteps = incoming.steps.map(incomingStep => {
    const canonicalId = canonicalizeStepId(incomingStep.id)
    const existingStep = existingSteps.get(canonicalId)
    
    if (existingStep) {
      // Preserve higher-priority status (completed > in_progress > pending)
      const mergedStatus = getHigherPriorityStatus(existingStep.status, incomingStep.status)
      return {
        ...existingStep,
        ...incomingStep,
        id: canonicalId,
        status: mergedStatus,
        completedAt: incomingStep.completedAt || existingStep.completedAt
      }
    }
    
    return { ...incomingStep, id: canonicalId }
  })
  
  return {
    ...existing,
    ...incoming,
    steps: mergedSteps,
    status: computeOverallStatus(mergedSteps)
  }
}

const normalizeMetadataPayload = (metadata?: ResearchMetadata | Record<string, any>): Partial<ResearchMetadata> | undefined => {
  if (!isObjectRecord(metadata)) {
    return undefined
  }

  const normalized: Partial<ResearchMetadata> = {}

  const planPayload = metadata.planDetails ?? metadata.plan_details ?? metadata.plan
  const planDetails = buildPlanDetailsFromPayload(planPayload)
  if (planDetails) {
    normalized.planDetails = planDetails
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined
}

export function useChatStream() {
  const {
    addMessage,
    updateStreamingMessage,
    setLoading,
    setResearchProgress,
    updateProgressWithETA,
    addIntermediateEvent,
    addIntermediateEvents,
    clearIntermediateEvents
  } = useChatStore()
  const abortControllerRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (content: string, config?: ChatRequest['config']) => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    // Add user message
    addMessage({ role: 'user', content })

    // Clear previous intermediate events
    clearIntermediateEvents()

    // Start a new assistant message for streaming and mark it as current
    const assistantMessageId = addMessage({ role: 'assistant', content: '', isStreaming: true })
    useChatStore.setState({ currentStreamingId: assistantMessageId })

    setLoading(true)

    try {
      abortControllerRef.current = new AbortController()

      // Get current messages for context
      const currentMessages = useChatStore.getState().messages
        .filter(msg => !msg.isStreaming) // Exclude the streaming placeholder
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }))

      // Start streaming request
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: currentMessages,
          config
        } as ChatRequest),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      let buffer = ''
      let fullContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += new TextDecoder().decode(value)
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)) as StreamEvent
              fullContent = await handleStreamEvent(data, assistantMessageId, fullContent)
            } catch (e) {
              console.error('Failed to parse stream event:', e, line)
            }
          }
        }
      }

    } catch (error) {
      console.error('Stream error:', error)
      if (error instanceof Error && error.name === 'AbortError') {
        // When request is cancelled, properly update the message and clear streaming state
        updateStreamingMessage(assistantMessageId, 'Request cancelled.', {
          search_queries: [], 
          sources: [], 
          research_iterations: 0, 
          total_sources_found: 0, 
          phase: 'complete', 
          progress_percentage: 100, 
          elapsed_time: 0, 
          current_node: '', 
          vector_results_count: 0
        } as any) // This metadata will set isStreaming to false
      } else {
        let errorMessage = 'Sorry, I encountered an error while processing your request.'

        if (error instanceof Error) {
          // Check for specific error types
          if (error.message.includes('401') || error.message.includes('Authentication')) {
            errorMessage = '**Authentication Error**\n\nPlease check your Databricks credentials and try again.\n\n**Troubleshooting:**\n- Verify your personal access token\n- Check CLI profile configuration\n- Ensure workspace URL is correct'
          } else if (error.message.includes('404') || error.message.includes('not found')) {
            errorMessage = '**Agent Not Found**\n\nThe research agent endpoint could not be found.\n\n**Troubleshooting:**\n- Check the agent endpoint name\n- Verify the serving endpoint is deployed\n- Confirm the endpoint is running'
          } else if (error.message.includes('timeout') || error.message.includes('network')) {
            errorMessage = '**Connection Error**\n\nUnable to connect to the research agent.\n\n**Troubleshooting:**\n- Check your internet connection\n- Verify the workspace URL\n- Try again in a few moments'
          } else {
            errorMessage = `**Error:** ${error.message}\n\n**Troubleshooting:**\n- Check server logs for details\n- Restart the development server\n- Verify configuration at \`/api/debug/config\``
          }
        }

        updateStreamingMessage(assistantMessageId, errorMessage)
      }
    } finally {
      // Ensure streaming flag is cleared on completion or error
      updateStreamingMessage(assistantMessageId, useChatStore.getState().messages.find(m => m.id === assistantMessageId)?.content || '', {
        // minimal metadata presence flips isStreaming to false
        search_queries: [], sources: [], research_iterations: 0, total_sources_found: 0, phase: 'complete', progress_percentage: 100, elapsed_time: 0, current_node: '', vector_results_count: 0
      } as any)
      setLoading(false)
      abortControllerRef.current = null
    }
  }, [addMessage, updateStreamingMessage, setLoading, updateProgressWithETA])

  const handleStreamEvent = async (data: StreamEvent, messageId: string, currentContent: string): Promise<string> => {
    // === EXTENSIVE LOGGING FOR DEBUGGING ===
    console.log(`🔵 [EVENT RAW] Type: "${data.type}", Full event:`, JSON.stringify(data, null, 2))
    
    // Log specific fields we care about
    if (data.event) {
      console.log(`🔵 [EVENT NESTED] event.event_type: "${data.event.event_type}"`)
      console.log(`🔵 [EVENT NESTED] event data:`, data.event)
    }
    if (data.events) {
      console.log(`🔵 [EVENT BATCH] Found ${data.events.length} events`)
      data.events.forEach((evt: any, idx: number) => {
        console.log(`🔵 [EVENT BATCH ${idx}] Type: "${evt.event_type}", Data:`, evt)
      })
    }
    
    // Log events with consistent JSON payload to aid test parsing
    let serializedPayload = '{}'
    try {
      serializedPayload = JSON.stringify(data)
    } catch (error) {
      serializedPayload = JSON.stringify({ parse_error: true })
    }
    console.log(`🔵 [EVENT STREAM] Type: ${data.type} Payload: ${serializedPayload}`)
    
    switch (data.type) {
      case 'stream_start':
        setResearchProgress({ currentPhase: 'querying' })
        break

      case 'content_delta':
        if (data.content) {
          const filterResult = filterContent(data.content)
          const filteredContent = filterResult.cleanContent
          const newContent = currentContent + filteredContent
          const processed = processStreamingWithTableReconstruction(newContent)

          updateStreamingMessage(messageId, processed.display)
          return processed.raw
        }
        break

      case 'research_update':
        if (data.metadata) {
          const normalizedMetadata = normalizeMetadataPayload(data.metadata)
          const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
          if (currentMessage) {
          // MUST merge planDetails without losing progress per specification
          const mergedPlan = mergePlanDetails(currentMessage.metadata?.planDetails, normalizedMetadata?.planDetails)
          
          if (normalizedMetadata?.planDetails) {
            console.log('📊 research_update with planDetails:', normalizedMetadata.planDetails)
          }
          if (mergedPlan) {
            console.log('🔀 Merged plan in research_update:', mergedPlan)
          }

          updateStreamingMessage(messageId, currentMessage.content, {
            ...currentMessage.metadata,
            ...data.metadata,
            ...normalizedMetadata,
            planDetails: mergedPlan
          } as any)
          }
          console.log('Research update received:', {
            phase: data.metadata.phase,
            progress: data.metadata.progressPercentage,
            sources: data.metadata.totalSourcesFound,
            queries: data.metadata.searchQueries?.length,
            elapsed: data.metadata.elapsedTime,
            currentNode: data.metadata.currentNode,
            vectorResults: data.metadata.vectorResultsCount,
            raw: data.metadata
          }) // Enhanced debug logging

          // Validate and normalize the phase
          const validPhases = ['querying', 'searching', 'analyzing', 'synthesizing', 'complete'] as const
          const normalizedPhase = validPhases.includes(data.metadata.phase as any)
            ? data.metadata.phase as typeof validPhases[number]
            : 'searching'

          // Use the new updateProgressWithETA function for better progress tracking
          updateProgressWithETA({
            currentPhase: normalizedPhase,
            queriesGenerated: data.metadata.searchQueries?.length || 0,
            sourcesFound: data.metadata.sources?.length || data.metadata.totalSourcesFound || 0,
            iterationsComplete: data.metadata.researchIterations || 0,
            progressPercentage: data.metadata.progressPercentage || 0,
            elapsedTime: data.metadata.elapsedTime || 0,
            currentNode: data.metadata.currentNode || '',
            vectorResultsCount: data.metadata.vectorResultsCount || 0,
            currentOperation: data.content || ''
          })
        } else {
          console.warn('Research update received without metadata')
        }
        break

      case 'message_complete':
        {
          let finalContent = data.content && data.content.length > 0 ? data.content : currentContent
          const filterResult = filterContent(finalContent)
          finalContent = filterResult.cleanContent

          const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
          if (currentMessage) {
            const normalizedMetadata = normalizeMetadataPayload(data.metadata)
            // MUST merge planDetails without losing progress per specification
            const mergedPlan = mergePlanDetails(currentMessage.metadata?.planDetails, normalizedMetadata?.planDetails)

            updateStreamingMessage(messageId, finalContent, {
              ...currentMessage.metadata,
              ...normalizedMetadata,
              planDetails: mergedPlan
            } as any)
          } else {
            updateStreamingMessage(messageId, finalContent, data.metadata)
          }

          setResearchProgress({ currentPhase: 'complete' })
          return finalContent
        }

      case 'stream_end':
        // Final cleanup if needed
        break

      case 'error':
        // Parse detailed error information
        const errorMessage = data.error || 'Unknown error occurred'
        const errorType = (data as any).error_type || 'unknown'

        // Create user-friendly error messages with troubleshooting suggestions
        let displayMessage = `**Error:** ${errorMessage}`

        // Add troubleshooting suggestions based on error type
        if (errorType === 'Authentication error' || errorMessage.includes('Authentication') || errorMessage.includes('401')) {
          displayMessage += `\n\n**Troubleshooting:**\n- Check your Databricks credentials in \`.env.local\`\n- Verify your personal access token is valid\n- Ensure your CLI profile is configured correctly`
        } else if (errorType === 'Agent endpoint not found' || errorMessage.includes('404')) {
          displayMessage += `\n\n**Troubleshooting:**\n- Verify the agent endpoint name is correct\n- Check if the serving endpoint exists in your workspace\n- Ensure the endpoint is deployed and running`
        } else if (errorType === 'Network error' || errorMessage.includes('Connection') || errorMessage.includes('timeout')) {
          displayMessage += `\n\n**Troubleshooting:**\n- Check your internet connection\n- Verify the workspace URL is correct\n- Try again in a few moments`
        } else if (errorType === 'server_error') {
          displayMessage += `\n\n**Debug Info:** Check server logs for detailed error information.\n\n**Troubleshooting:**\n- Restart the development server\n- Check the configuration with \`/api/debug/config\``
        }

        updateStreamingMessage(messageId, displayMessage)
        setResearchProgress({ currentPhase: 'complete' })
        break

      case 'intermediate_event':
        // Handle single intermediate event with enhanced processing
        if (data.event) {
          const receivedAt = new Date().toISOString()
          const eventTypeRaw = data.event.event_type
          const payloadKeys = Object.keys(data.event)
          const dataKeys = Object.keys(data.event.data ?? {})
          const metaKeys = Object.keys(data.event.meta ?? {})
          console.log('🔵 [INTERMEDIATE_EVENT] Raw payload received:', data.event)
          console.log(
            '🔵 [INTERMEDIATE_EVENT] Received event @',
            receivedAt,
            {
              eventType: eventTypeRaw,
              payloadKeys,
              dataPreview: dataKeys.slice(0, 5),
              metaPreview: metaKeys.slice(0, 5),
              hasTimestamp: Boolean(data.event.timestamp),
              correlationId: data.event.correlation_id ?? data.event.correlationId ?? null
            },
            data.event
          )
          
          // MUST handle plan creation events per specification
          const eventType = eventTypeRaw
          console.log(`🔵 [INTERMEDIATE_EVENT] Event type check: "${eventType}" (exact match)`) 
          
          if (eventType === 'PLAN_CREATED' || eventType === 'plan_created') {
            console.log('🎯 [PLAN_CREATED] event detected:', data.event)
            const planData = data.event.data?.plan
            if (planData) {
              console.log('📋 [PLAN_CREATED] Plan data found:', planData)
              const planDetails = buildPlanDetailsFromPayload(planData)
              if (planDetails) {
                console.log('✅ [PLAN_CREATED] Plan details built summary:', {
                  keys: Object.keys(planDetails),
                  stepCount: Array.isArray(planDetails.steps) ? planDetails.steps.length : 0,
                  status: planDetails.status,
                  iterations: planDetails.iterations
                })
                const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
                if (currentMessage) {
                  const mergedPlan = mergePlanDetails(currentMessage.metadata?.planDetails, planDetails)
                  console.log('🔀 [PLAN_CREATED] Merged plan summary:', {
                    stepCount: Array.isArray(mergedPlan?.steps) ? mergedPlan.steps.length : 0,
                    status: mergedPlan?.status,
                    iterations: mergedPlan?.iterations,
                    hasEnoughContext: mergedPlan?.hasEnoughContext
                  })
                  updateStreamingMessage(messageId, currentMessage.content, {
                    ...currentMessage.metadata,
                    planDetails: mergedPlan
                  } as any)
                  console.log('💾 [PLAN_CREATED] Updated message metadata plan steps:', mergedPlan?.steps?.map((step: any) => ({ id: step.id, status: step.status })))
                  console.log('💾 [PLAN_CREATED] Updated message with plan details')
                }
              } else {
                console.log('❌ [PLAN_CREATED] Failed to build plan details from payload')
              }
            } else {
              console.log('❌ [PLAN_CREATED] No plan data in event')
            }
          } else {
            console.log(`🔵 [INTERMEDIATE_EVENT] Event "${eventType}" is not a plan creation event`)
          }
          
          // MUST handle step progress events per specification (lowercase)
          const lowerEventType = eventTypeRaw?.toLowerCase()
          console.log(`🔵 [STEP_EVENT] Checking step event: "${lowerEventType}"`)
          if (lowerEventType === 'step_activated' || lowerEventType === 'step_completed' || lowerEventType === 'step_failed') {
            console.log(`🎯 [STEP_EVENT] Step event detected: "${lowerEventType}"`)
            const stepIdRaw = data.event.data?.step_id
            const stepId = stepIdRaw ? canonicalizeStepId(stepIdRaw) : undefined
            console.log(`🔵 [STEP_EVENT] Step ID: ${stepId}, raw: ${stepIdRaw}`)
            const newStatus: PlanStep['status'] = lowerEventType === 'step_completed'
              ? 'completed'
              : lowerEventType === 'step_failed'
                ? 'skipped'
                : 'in_progress'
            console.log(`🔵 [STEP_EVENT] New status: ${newStatus}`)
            
            // Update the specific step in the plan
            const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
            if (stepId && currentMessage?.metadata?.planDetails?.steps) {
              const updatedSteps = currentMessage.metadata.planDetails.steps.map((step: any) => 
                canonicalizeStepId(step.id) === stepId ? { 
                  ...step, 
                  status: newStatus, 
                  completedAt: newStatus === 'completed' ? Date.now() : step.completedAt 
                } : step
              )
              
              updateStreamingMessage(messageId, currentMessage.content, {
                ...currentMessage.metadata,
                planDetails: {
                  ...currentMessage.metadata.planDetails,
                  steps: updatedSteps,
                  status: computeOverallStatus(updatedSteps)
                }
              } as any)
              console.log('🛠️ [STEP_EVENT] Updated steps summary:', updatedSteps.map((step: any) => ({ id: step.id, status: step.status })))
            }
          }
          
          // Ensure event has proper structure for UI
          const processedEvent = {
            ...data.event,
            timestamp: data.event.timestamp || Date.now() / 1000,
            meta: {
              ...data.event.meta,
              // Ensure we have display fields from the agent's event templates
              title: data.event.meta?.title || data.event.event_type?.replace(/_/g, ' '),
              description: data.event.meta?.description,
              category: data.event.meta?.category,
              icon: data.event.meta?.icon,
              priority: data.event.meta?.priority,
              confidence: data.event.meta?.confidence,
              reasoning: data.event.meta?.reasoning
            }
          }
          
          addIntermediateEvent(processedEvent)
          console.log('📥 [INTERMEDIATE_EVENT] Stored processed event summary:', {
            eventType: processedEvent.event_type,
            metaTitle: processedEvent.meta?.title,
            hasMeta: Boolean(processedEvent.meta),
            hasData: Boolean(processedEvent.data)
          })
        }
        break

      case 'event_batch':
        // Handle batch of intermediate events with enhanced processing
        if (data.events && Array.isArray(data.events)) {
          console.log('Adding batch of events:', data.events.length, data.events)
          const batchSummary = data.events.reduce(
            (acc, event, index) => {
              const eventType = event?.event_type ?? 'unknown'
              const normalizedType = eventType?.toLowerCase?.() ?? 'unknown'
              acc.types.push(eventType)
              if (normalizedType.includes('plan')) acc.planEvents += 1
              if (normalizedType.includes('step')) acc.stepEvents += 1
              if (!event || typeof event !== 'object') acc.invalidIndices.push(index)
              return acc
            },
            { types: [] as string[], planEvents: 0, stepEvents: 0, invalidIndices: [] as number[] }
          )
          console.log('🔵 [EVENT_BATCH] Summary:', batchSummary)
          
          // Check for plan events in the batch with defensive coding
          data.events.forEach((event, eventIndex) => {
            try {
              console.log('🔵 [EVENT_BATCH] Processing index', eventIndex, 'event_type=', event?.event_type)
              if (!event || typeof event !== 'object') {
                console.warn(`[useChatStream] Invalid event at index ${eventIndex}:`, event)
                return
              }

              if (event.event_type === IntermediateEventType.PLAN_CREATED || 
                  event.event_type === IntermediateEventType.PLAN_UPDATED ||
                  event.event_type === IntermediateEventType.PLAN_STRUCTURE_VISUALIZE ||
                  event.event_type === 'plan_structure') {

                const planData = event.data?.plan || event.data
                const planDetails = buildPlanDetailsFromPayload(planData)

                if (planDetails) {
                  console.log('🧾 [EVENT_BATCH] Plan event summary:', {
                    keys: Object.keys(planDetails),
                    stepCount: Array.isArray(planDetails.steps) ? planDetails.steps.length : 0,
                    status: planDetails.status,
                    iterations: planDetails.iterations
                  })
                  const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
                  if (currentMessage) {
                    const mergedPlan = mergePlanDetails(currentMessage.metadata?.planDetails, planDetails)
                    updateStreamingMessage(messageId, currentMessage.content, {
                      ...currentMessage.metadata,
                      planDetails: mergedPlan ?? planDetails
                    } as any)
                    console.log('🔀 [EVENT_BATCH] Plan merged total steps:', mergedPlan?.steps?.length ?? 0)
                  }
                }
              }

              const eventType = event.event_type?.toLowerCase()
              if (eventType === 'step_activated' ||
                  eventType === 'step_completed' ||
                  eventType === 'step_in_progress') {

                const stepIdRaw = event.data?.step_id ?? event.data?.id
                const stepId = stepIdRaw ? canonicalizeStepId(stepIdRaw) : undefined
                const newStatus = eventType === 'step_completed' ? 'completed' : 'in_progress'

                const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
                if (stepId && currentMessage?.metadata?.planDetails?.steps && Array.isArray(currentMessage.metadata.planDetails.steps)) {
                  try {
                    const updatedSteps = currentMessage.metadata.planDetails.steps.map((step: any, index: number) => {
                      if (!step || typeof step !== 'object') return step
                      return canonicalizeStepId(step.id, index) === stepId
                        ? { ...step, status: newStatus, completedAt: newStatus === 'completed' ? Date.now() : step.completedAt }
                        : step
                    })

                    const mergedPlan = {
                      ...currentMessage.metadata.planDetails,
                      steps: updatedSteps.map((step, index) => ({
                        ...step,
                        id: canonicalizeStepId(step.id, index),
                        status: normalizeStepStatus(step.status)
                      })),
                      status: computePlanStatus(updatedSteps)
                    }

                    updateStreamingMessage(messageId, currentMessage.content, {
                      ...currentMessage.metadata,
                      planDetails: mergedPlan
                    } as any)
                    console.log('🛠️ [EVENT_BATCH] Step update summary:', {
                      stepId,
                      newStatus,
                      statuses: mergedPlan.steps?.map((step: any) => ({ id: step.id, status: step.status }))
                    })
                  } catch (stepUpdateError) {
                    console.warn(`[useChatStream] Error updating step in batch at index ${eventIndex}:`, stepUpdateError)
                  }
                }
              }
            } catch (eventProcessingError) {
              console.warn(`[useChatStream] Error processing event at index ${eventIndex}:`, eventProcessingError, event)
            }
          })
          
          // Process each event in the batch
          const processedEvents = data.events.map(event => ({
            ...event,
            timestamp: event.timestamp || Date.now() / 1000,
            meta: {
              ...event.meta,
              title: event.meta?.title || event.event_type?.replace(/_/g, ' '),
              description: event.meta?.description,
              category: event.meta?.category,
              icon: event.meta?.icon,
              priority: event.meta?.priority,
              confidence: event.meta?.confidence,
              reasoning: event.meta?.reasoning
            }
          }))
          console.log('📥 [EVENT_BATCH] Prepared processed events count:', processedEvents.length)
          
          addIntermediateEvents(processedEvents)
          console.log('📥 [EVENT_BATCH] Stored processed events count:', processedEvents.length)
        }
        break
        
      // Handle enhanced agent streaming events
      case 'agent_start':
      case 'agent_complete':
      case 'tool_start':
      case 'tool_complete':
      case 'llm_streaming':
        // These come from the enhanced _process_stream_event method
        // Create intermediate events for the UI with enhanced metadata
        const eventType = data.metadata?.event_type || data.type
        const eventData = {
          agent: data.metadata?.agent,
          current_agent: data.metadata?.agent,
          action: data.content,
          tool_name: data.metadata?.tool,
          query: data.metadata?.query,
          result_count: data.metadata?.result_count,
          is_streaming: data.metadata?.is_streaming
        }
        
        // Map event types to categories for backward compatibility
        const getEventCategory = (type: string) => {
          if (type.includes('agent')) return 'coordination'
          if (type.includes('tool')) return 'search'
          if (type.includes('llm')) return 'reflection'
          return 'unknown'
        }
        
        // Create intermediate event with enhanced structure
        const intermediateEvent = {
          id: crypto.randomUUID(),
          timestamp: Date.now() / 1000,
          correlation_id: 'stream_' + messageId,
          sequence: Date.now(),
          event_type: eventType,
          data: eventData,
          meta: {
            title: eventType.replace(/_/g, ' '),
            description: data.content,
            category: getEventCategory(eventType),
            priority: eventType.includes('error') ? 8 : eventType.includes('start') ? 6 : 4,
            confidence: data.metadata?.confidence,
            reasoning: data.metadata?.reasoning
          }
        }
        
        console.log('Creating intermediate event from stream:', intermediateEvent)
        addIntermediateEvent(intermediateEvent)
        break
    }

    return currentContent
  }

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setLoading(false)
    updateProgressWithETA({ currentPhase: 'complete' })
  }, [setLoading, updateProgressWithETA])

  return { sendMessage, stopStream }
}

export const __testables__ = {
  canonicalizeStepId,
  mergePlanDetails,
  buildPlanDetailsFromPayload,
  handleIntermediateEventForTest: (
    currentMessage: any,
    event: { event_type: string; data: any }
  ) => {
    if (!currentMessage) return undefined

    if (event.event_type === 'PLAN_CREATED') {
      const planData = event.data?.plan
      const planDetails = buildPlanDetailsFromPayload(planData)
      if (!planDetails) return currentMessage

      return {
        ...currentMessage,
        metadata: {
          ...currentMessage.metadata,
          planDetails: mergePlanDetails(currentMessage.metadata?.planDetails, planDetails)
        }
      }
    }

    const eventType = event.event_type?.toLowerCase()
    if (eventType === 'step_activated' || eventType === 'step_completed' || eventType === 'step_failed') {
      const stepIdRaw = event.data?.step_id
      const stepId = stepIdRaw ? canonicalizeStepId(stepIdRaw) : undefined
      const newStatus: PlanStep['status'] = eventType === 'step_completed'
        ? 'completed'
        : eventType === 'step_failed'
          ? 'skipped'
          : 'in_progress'

      if (stepId && currentMessage?.metadata?.planDetails?.steps) {
        const updatedSteps = currentMessage.metadata.planDetails.steps.map((step: any, index: number) =>
          canonicalizeStepId(step.id, index) === stepId
            ? {
                ...step,
                status: newStatus,
                completedAt: newStatus === 'completed' ? Date.now() : step.completedAt
              }
            : step
        )

        return {
          ...currentMessage,
          metadata: {
            ...currentMessage.metadata,
            planDetails: {
              ...currentMessage.metadata.planDetails,
              steps: updatedSteps,
              status: computeOverallStatus(updatedSteps)
            }
          }
        }
      }
    }

    return currentMessage
  }
}