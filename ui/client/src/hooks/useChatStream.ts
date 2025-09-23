import { useCallback, useRef } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { StreamEvent, ChatRequest, IntermediateEvent, IntermediateEventType } from '@/types/chat'
import {
  processStreamingWithTableReconstruction
} from '@/utils/tableStreamReconstructor'
import { filterContent, ContentType } from '@/utils/contentFilter'

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
    console.log('Stream event:', data.type, data) // Debug logging
    switch (data.type) {
      case 'stream_start':
        setResearchProgress({ currentPhase: 'querying' })
        break

      case 'content_delta':
        if (data.content) {
          // Check if content contains plan creation markers
          if (data.content.includes('📋 Created research plan with') || 
              data.content.includes('Created research plan') ||
              data.content.includes('Research plan:')) {
            // Extract plan steps count from the content
            const stepsMatch = data.content.match(/with (\d+) steps/)
            const stepsCount = stepsMatch ? parseInt(stepsMatch[1]) : 5
            
            // Create placeholder plan metadata
            const planMetadata = {
              planDetails: {
                steps: Array.from({ length: stepsCount }, (_, i) => ({
                  id: `step-${i + 1}`,
                  description: `Step ${i + 1}`,
                  status: 'pending' as const
                })),
                quality: 0.8,
                iterations: 1,
                status: 'executing' as const,
                hasEnoughContext: true
              }
            }
            
            // Update the message metadata with plan details
            const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
            if (currentMessage) {
              const updatedMetadata = {
                ...currentMessage.metadata,
                ...planMetadata
              }
              updateStreamingMessage(messageId, currentMessage.content, updatedMetadata as any)
            }
          }
          
          // Apply content filtering to incoming content to prevent JSON contamination
          const filterResult = filterContent(data.content)
          
          // Log filtering activity for debugging
          if (filterResult.filteringApplied) {
            console.warn(`[ContentFilter] Applied filtering to content_delta:`, {
              originalLength: filterResult.originalLength,
              cleanLength: filterResult.cleanLength,
              contentType: filterResult.contentType,
              extractedData: filterResult.extractedData.length,
              warnings: filterResult.warnings
            })
            
            // Log any warnings
            filterResult.warnings.forEach(warning => {
              console.warn(`[ContentFilter] Warning: ${warning}`)
            })
          }
          
          // Use the filtered content for accumulation
          const filteredContent = filterResult.cleanContent
          const newContent = currentContent + filteredContent

          // Process streaming content - returns both display and raw versions
          const processed = processStreamingWithTableReconstruction(newContent)

          // Debug logging to verify table handling
          if (newContent.includes('|') && newContent.includes('---')) {
            console.log('[TABLE DEBUG] Streaming content with potential table:', {
              hasTable: newContent.includes('|') && newContent.includes('---'),
              rawLength: processed.raw.length,
              displayLength: processed.display.length,
              hasPlaceholders: processed.display.includes('📊')
            })
          }

          // Show display version with placeholders during streaming
          updateStreamingMessage(messageId, processed.display)

          // Return the raw content for tracking
          return processed.raw
        }
        break

      case 'research_update':
        if (data.metadata) {
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
          // Use the raw content that was being tracked
          let finalContent = data.content && data.content.length > 0 ? data.content : currentContent

          // Apply content filtering to final content to remove any JSON elements
          const filterResult = filterContent(finalContent)
          
          // Log filtering activity for final content
          if (filterResult.filteringApplied) {
            console.warn(`[ContentFilter] Applied filtering to final report:`, {
              originalLength: filterResult.originalLength,
              cleanLength: filterResult.cleanLength,
              contentType: filterResult.contentType,
              extractedData: filterResult.extractedData.length,
              warnings: filterResult.warnings
            })
            
            // Log any warnings
            filterResult.warnings.forEach(warning => {
              console.warn(`[ContentFilter] Warning: ${warning}`)
            })
          }
          
          // Use the filtered content
          finalContent = filterResult.cleanContent

          // Debug logging for final content
          if (finalContent.includes('|') && finalContent.includes('---')) {
            console.log('[TABLE DEBUG] Final content with tables:', {
              hasTable: true,
              contentLength: finalContent.length,
              hasPlaceholders: finalContent.includes('📊'),
              sample: finalContent.substring(0, 200) + '...'
            })
          }

          // The content should already be raw (no placeholders), but ensure it's clean
          // The MarkdownRenderer will handle table reconstruction for the final display

          updateStreamingMessage(messageId, finalContent, data.metadata)
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
          console.log('Adding intermediate event:', data.event.event_type, data.event)
          
          // Check if this is a plan-related event and update metadata accordingly
          if (data.event.event_type === IntermediateEventType.PLAN_CREATED || 
              data.event.event_type === IntermediateEventType.PLAN_UPDATED ||
              data.event.event_type === IntermediateEventType.PLAN_STRUCTURE_VISUALIZE) {
            
            // Extract plan details from the event
            const planData = data.event.data?.plan || data.event.data
            
            if (planData && planData.steps) {
              // Convert the plan to the expected format
              const planMetadata = {
                planDetails: {
                  steps: planData.steps.map((step: any, index: number) => ({
                    id: step.id || `step-${index + 1}`,
                    description: step.description || step.content || '',
                    status: step.status || 'pending',
                    result: step.result,
                    completedAt: step.completedAt
                  })),
                  quality: planData.quality || planData.confidence || 0,
                  iterations: planData.iterations || 1,
                  status: planData.status || 'executing',
                  hasEnoughContext: planData.hasEnoughContext || false
                }
              }
              
              // Update the message metadata with plan details
              const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
              if (currentMessage) {
                const updatedMetadata = {
                  ...currentMessage.metadata,
                  ...planMetadata
                }
                updateStreamingMessage(messageId, currentMessage.content, updatedMetadata as any)
              }
            }
          }
          
          // Check for step activation/completion events
          if (data.event.event_type === IntermediateEventType.STEP_ACTIVATED ||
              data.event.event_type === IntermediateEventType.STEP_COMPLETED) {
            
            const stepId = data.event.data?.step_id
            const newStatus = data.event.event_type === IntermediateEventType.STEP_COMPLETED ? 'completed' : 'in_progress'
            
            // Update the specific step in the plan
            const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
            if (currentMessage?.metadata?.planDetails?.steps) {
              const updatedSteps = currentMessage.metadata.planDetails.steps.map((step: any) => 
                step.id === stepId ? { ...step, status: newStatus, completedAt: newStatus === 'completed' ? Date.now() : step.completedAt } : step
              )
              
              const updatedMetadata = {
                ...currentMessage.metadata,
                planDetails: {
                  ...currentMessage.metadata.planDetails,
                  steps: updatedSteps
                }
              }
              
              updateStreamingMessage(messageId, currentMessage.content, updatedMetadata as any)
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
        }
        break

      case 'event_batch':
        // Handle batch of intermediate events with enhanced processing
        if (data.events && Array.isArray(data.events)) {
          console.log('Adding batch of events:', data.events.length, data.events)
          
          // Check for plan events in the batch
          data.events.forEach(event => {
            if (event.event_type === IntermediateEventType.PLAN_CREATED || 
                event.event_type === IntermediateEventType.PLAN_UPDATED ||
                event.event_type === IntermediateEventType.PLAN_STRUCTURE_VISUALIZE) {
              
              const planData = event.data?.plan || event.data
              
              if (planData && planData.steps) {
                const planMetadata = {
                  planDetails: {
                    steps: planData.steps.map((step: any, index: number) => ({
                      id: step.id || `step-${index + 1}`,
                      description: step.description || step.content || '',
                      status: step.status || 'pending',
                      result: step.result,
                      completedAt: step.completedAt
                    })),
                    quality: planData.quality || planData.confidence || 0,
                    iterations: planData.iterations || 1,
                    status: planData.status || 'executing',
                    hasEnoughContext: planData.hasEnoughContext || false
                  }
                }
                
                const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
                if (currentMessage) {
                  const updatedMetadata = {
                    ...currentMessage.metadata,
                    ...planMetadata
                  }
                  updateStreamingMessage(messageId, currentMessage.content, updatedMetadata as any)
                }
              }
            }
            
            // Check for step updates
            if (event.event_type === IntermediateEventType.STEP_ACTIVATED ||
                event.event_type === IntermediateEventType.STEP_COMPLETED) {
              
              const stepId = event.data?.step_id
              const newStatus = event.event_type === IntermediateEventType.STEP_COMPLETED ? 'completed' : 'in_progress'
              
              const currentMessage = useChatStore.getState().messages.find(m => m.id === messageId)
              if (currentMessage?.metadata?.planDetails?.steps) {
                const updatedSteps = currentMessage.metadata.planDetails.steps.map((step: any) => 
                  step.id === stepId ? { ...step, status: newStatus, completedAt: newStatus === 'completed' ? Date.now() : step.completedAt } : step
                )
                
                const updatedMetadata = {
                  ...currentMessage.metadata,
                  planDetails: {
                    ...currentMessage.metadata.planDetails,
                    steps: updatedSteps
                  }
                }
                
                updateStreamingMessage(messageId, currentMessage.content, updatedMetadata as any)
              }
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
          
          addIntermediateEvents(processedEvents)
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