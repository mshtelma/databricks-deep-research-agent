import { useCallback, useRef } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { StreamEvent, ChatRequest } from '@/types/chat'
import {
  processStreamingWithTableReconstruction
} from '@/utils/tableStreamReconstructor'

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
          const newContent = currentContent + data.content

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
        // Handle single intermediate event
        if (data.event) {
          addIntermediateEvent(data.event)
        }
        break

      case 'event_batch':
        // Handle batch of intermediate events
        if (data.events && Array.isArray(data.events)) {
          addIntermediateEvents(data.events)
        }
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