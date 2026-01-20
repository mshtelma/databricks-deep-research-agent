/**
 * Research Jobs Hook
 *
 * Provides hooks for managing background research jobs:
 * - useActiveJobs: Get list of currently running jobs
 * - useSubmitJob: Submit a new research job
 * - useCancelJob: Cancel a running job
 * - useJobEventStream: Connect to SSE stream for a job
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect, useCallback, useRef } from 'react'
import { jobsApi, type JobEvent } from '../api/client'

// Query keys for react-query
const JOBS_KEY = ['jobs']
const ACTIVE_JOBS_KEY = [...JOBS_KEY, 'in_progress']

/**
 * Hook to get all active (in_progress) jobs for the current user.
 * Polls every 5 seconds to keep status updated.
 */
export function useActiveJobs() {
  return useQuery({
    queryKey: ACTIVE_JOBS_KEY,
    queryFn: () => jobsApi.list({ status: 'in_progress' }),
    refetchInterval: 5000, // Poll every 5 seconds
    staleTime: 2000, // Consider data stale after 2 seconds
  })
}

/**
 * Hook to list jobs with optional filtering.
 */
export function useJobs(params?: { status?: string; limit?: number }) {
  return useQuery({
    queryKey: [...JOBS_KEY, params],
    queryFn: () => jobsApi.list(params),
    staleTime: 5000,
  })
}

/**
 * Hook to get a specific job by ID.
 */
export function useJob(sessionId: string | null) {
  return useQuery({
    queryKey: [...JOBS_KEY, sessionId],
    queryFn: () => (sessionId ? jobsApi.get(sessionId) : null),
    enabled: !!sessionId,
    staleTime: 1000,
  })
}

/**
 * Hook to submit a new research job.
 * Returns mutation that can be called with job parameters.
 */
export function useSubmitJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      chatId: string
      query: string
      queryMode?: string
      researchDepth?: string
      verifySources?: boolean
    }) => jobsApi.submit(data),
    onSuccess: () => {
      // Invalidate job lists to show new job
      queryClient.invalidateQueries({ queryKey: JOBS_KEY })
    },
    onError: (error: Error) => {
      // Handle 429 (rate limit) specially
      if (error.message.includes('429') || error.message.includes('concurrent')) {
        console.warn('Job limit reached:', error.message)
      }
    },
  })
}

/**
 * Hook to cancel a running job.
 */
export function useCancelJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (sessionId: string) => jobsApi.cancel(sessionId),
    onSuccess: (_, sessionId) => {
      // Invalidate specific job and all lists
      queryClient.invalidateQueries({ queryKey: JOBS_KEY })
      queryClient.invalidateQueries({ queryKey: [...JOBS_KEY, sessionId] })
    },
  })
}

/**
 * Hook to get active job for a specific chat.
 */
export function useChatActiveJob(chatId: string | null) {
  return useQuery({
    queryKey: [...JOBS_KEY, 'chat', chatId, 'active'],
    queryFn: () => (chatId ? jobsApi.getChatActiveJob(chatId) : null),
    enabled: !!chatId,
    staleTime: 2000,
    refetchInterval: 3000, // Poll to detect when job starts/stops
  })
}

/**
 * Event handler type for job events.
 */
export type JobEventHandler = (event: JobEvent | { eventType: 'job_completed'; status: string }) => void

/**
 * Hook to connect to a job's SSE event stream.
 *
 * Handles:
 * - Automatic reconnection with resume from last sequence
 * - Event parsing and delivery to handler
 * - Cleanup on unmount or job completion
 *
 * @param sessionId - Job session ID to stream events for (null to disable)
 * @param onEvent - Callback for each event received
 * @returns Connection state
 */
export function useJobEventStream(
  sessionId: string | null,
  onEvent: JobEventHandler
) {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const lastSequenceRef = useRef(0)
  const eventSourceRef = useRef<EventSource | null>(null)
  const onEventRef = useRef(onEvent)
  const queryClient = useQueryClient()

  // Keep onEvent ref updated
  useEffect(() => {
    onEventRef.current = onEvent
  }, [onEvent])

  // Connect to event stream
  useEffect(() => {
    if (!sessionId) {
      setIsConnected(false)
      return
    }

    const connect = () => {
      const url = jobsApi.streamUrl(sessionId, lastSequenceRef.current)
      const eventSource = new EventSource(url)
      eventSourceRef.current = eventSource

      eventSource.onopen = () => {
        setIsConnected(true)
        setError(null)
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          // Call event handler
          onEventRef.current(data)

          // Update sequence number for reconnection
          if (data.sequenceNumber) {
            lastSequenceRef.current = data.sequenceNumber
          }

          // Handle job completion
          if (data.eventType === 'job_completed') {
            eventSource.close()
            setIsConnected(false)
            // Invalidate jobs to update status
            queryClient.invalidateQueries({ queryKey: JOBS_KEY })
          }
        } catch (err) {
          console.error('Error parsing job event:', err)
        }
      }

      eventSource.onerror = (err) => {
        console.error('Job EventSource error:', err)
        setIsConnected(false)
        eventSource.close()

        // Attempt reconnection after delay
        setTimeout(() => {
          if (eventSourceRef.current === eventSource) {
            connect()
          }
        }, 2000)
      }
    }

    connect()

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
      setIsConnected(false)
    }
  }, [sessionId, queryClient])

  return {
    isConnected,
    error,
    lastSequence: lastSequenceRef.current,
  }
}

/**
 * Hook for polling job events (alternative to SSE for environments that don't support it).
 *
 * @param sessionId - Job session ID to poll events for
 * @param enabled - Whether polling is enabled
 * @param pollInterval - Polling interval in ms (default 1000)
 */
export function useJobEventPolling(
  sessionId: string | null,
  enabled: boolean = true,
  pollInterval: number = 1000
) {
  const [events, setEvents] = useState<JobEvent[]>([])
  const [lastSequence, setLastSequence] = useState(0)
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery({
    queryKey: [...JOBS_KEY, sessionId, 'events', lastSequence],
    queryFn: () => (sessionId ? jobsApi.getEvents(sessionId, lastSequence) : null),
    enabled: enabled && !!sessionId,
    refetchInterval: (query) => {
      // Stop polling when job is no longer in progress
      const data = query.state.data
      if (data && !data.hasMore) {
        return false
      }
      return pollInterval
    },
  })

  // Append new events
  useEffect(() => {
    if (data?.events && data.events.length > 0) {
      setEvents((prev) => [...prev, ...data.events])
      const lastEvent = data.events[data.events.length - 1]
      if (lastEvent && lastEvent.sequenceNumber) {
        setLastSequence(lastEvent.sequenceNumber)
      }
    }

    // Invalidate job lists when done
    if (data && !data.hasMore) {
      queryClient.invalidateQueries({ queryKey: JOBS_KEY })
    }
  }, [data, queryClient])

  return {
    events,
    sessionStatus: data?.sessionStatus,
    hasMore: data?.hasMore ?? true,
    isLoading,
    error,
  }
}

/**
 * Combined hook for managing job submission and event streaming.
 *
 * Provides a complete workflow:
 * 1. Submit job via submitJob()
 * 2. Automatically connect to event stream
 * 3. Receive events via onEvent callback
 * 4. Handle completion/cancellation
 */
export function useResearchJob(chatId: string) {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed' | 'cancelled'>('idle')

  const submitMutation = useSubmitJob()
  const cancelMutation = useCancelJob()
  const queryClient = useQueryClient()

  // Handle incoming events
  const handleEvent = useCallback((event: JobEvent | { eventType: 'job_completed'; status: string }) => {
    if ('sequenceNumber' in event) {
      setEvents((prev) => [...prev, event])
    }

    if (event.eventType === 'job_completed') {
      const apiStatus = 'status' in event ? event.status : 'completed'
      // Map API status to hook status (in_progress -> running for UI)
      let newStatus: typeof status = 'completed'
      if (apiStatus === 'failed') newStatus = 'failed'
      else if (apiStatus === 'cancelled') newStatus = 'cancelled'
      else if (apiStatus === 'in_progress') newStatus = 'running' // Shouldn't happen for completed event
      setStatus(newStatus)
      setActiveSessionId(null)
      queryClient.invalidateQueries({ queryKey: JOBS_KEY })
    }
  }, [queryClient])

  // Connect to event stream when job is active
  const { isConnected } = useJobEventStream(activeSessionId, handleEvent)

  // Submit new job
  const submitJob = useCallback(
    async (params: {
      query: string
      queryMode?: string
      researchDepth?: string
      verifySources?: boolean
    }) => {
      setEvents([])
      setStatus('running')

      try {
        const job = await submitMutation.mutateAsync({
          chatId,
          ...params,
        })
        setActiveSessionId(job.sessionId)
        return job
      } catch (error) {
        setStatus('failed')
        throw error
      }
    },
    [chatId, submitMutation]
  )

  // Cancel active job
  const cancel = useCallback(async () => {
    if (activeSessionId) {
      await cancelMutation.mutateAsync(activeSessionId)
      setStatus('cancelled')
      setActiveSessionId(null)
    }
  }, [activeSessionId, cancelMutation])

  // Reset state
  const reset = useCallback(() => {
    setActiveSessionId(null)
    setEvents([])
    setStatus('idle')
  }, [])

  return {
    // State
    sessionId: activeSessionId,
    events,
    status,
    isStreaming: isConnected,
    isSubmitting: submitMutation.isPending,

    // Actions
    submitJob,
    cancel,
    reset,

    // Error state
    submitError: submitMutation.error,
    cancelError: cancelMutation.error,
  }
}
