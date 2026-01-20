/**
 * useResearchReconnection - Hook for reconnecting to in-progress research after browser reload.
 *
 * This hook enables crash resilience by:
 * 1. Checking for active research session on chat load
 * 2. Fetching missed events via polling
 * 3. Restoring UI state from persisted events
 *
 * Uses polling-based approach (2-second intervals) instead of WebSockets for simplicity.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { StreamEvent } from '@/types';

interface ReconnectionState {
  isReconnecting: boolean;
  reconnectionError: string | null;
  sessionId: string | null;
}

interface ActiveResearchResponse {
  hasActiveResearch: boolean;
  sessionId?: string;
  status?: string;
  lastSequenceNumber?: number;
  query?: string;
  queryMode?: string;  // "simple", "web_search", "deep_research"
  startedAt?: string;  // ISO timestamp for timer display
}

interface ReconnectionResult {
  reconnected: boolean;
  queryMode?: string;
  startedAt?: string;
}

interface ResearchEventsResponse {
  events: Array<{
    id: string;
    eventType: string;
    timestamp: string;
    sequenceNumber: number | null;
    payload: Record<string, unknown>;
  }>;
  sessionStatus: string;
  hasMore: boolean;
}

interface ResearchStateResponse {
  sessionId: string;
  status: string;
  query?: string;
  plan?: Record<string, unknown>;
  observations?: Array<Record<string, unknown>>;
  currentStepIndex?: number;
  planIterations?: number;
  finalReport?: string;
  completedAt?: string;
}

interface UseResearchReconnectionOptions {
  chatId: string | null;
  onEvent: (event: StreamEvent) => void;
  onComplete: (finalReport: string, state: ResearchStateResponse) => void;
  onError: (error: string) => void;
  enabled?: boolean;
}

const POLL_INTERVAL_MS = 2000;

export function useResearchReconnection({
  chatId,
  onEvent,
  onComplete,
  onError,
  enabled = true,
}: UseResearchReconnectionOptions) {
  const [state, setState] = useState<ReconnectionState>({
    isReconnecting: false,
    reconnectionError: null,
    sessionId: null,
  });

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastSequenceRef = useRef<number>(0);
  const isPollingRef = useRef(false);

  // Stop polling
  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    isPollingRef.current = false;
  }, []);

  // Poll for events
  const pollEvents = useCallback(
    async (chatId: string, sessionId: string): Promise<boolean> => {
      try {
        const url = `/api/v1/chats/${chatId}/research/${sessionId}/events?sinceSequence=${lastSequenceRef.current}`;
        const res = await fetch(url);

        if (!res.ok) {
          throw new Error(`Failed to fetch events: ${res.status}`);
        }

        const data: ResearchEventsResponse = await res.json();

        // Process new events
        for (const event of data.events) {
          // Update sequence tracker
          if (event.sequenceNumber !== null && event.sequenceNumber > lastSequenceRef.current) {
            lastSequenceRef.current = event.sequenceNumber;
          }

          // Reconstruct StreamEvent from payload and dispatch
          // The payload contains the full event data in camelCase format
          const streamEvent = event.payload as unknown as StreamEvent;
          onEvent(streamEvent);
        }

        // Check if research completed
        if (!data.hasMore) {
          stopPolling();

          // Fetch final state
          const stateRes = await fetch(
            `/api/v1/chats/${chatId}/research/${sessionId}/state`
          );

          if (stateRes.ok) {
            const finalState: ResearchStateResponse = await stateRes.json();

            if (finalState.status === 'completed' && finalState.finalReport) {
              onComplete(finalState.finalReport, finalState);
            } else if (finalState.status === 'failed') {
              onError('Research failed');
            }
          }

          setState((prev) => ({
            ...prev,
            isReconnecting: false,
            reconnectionError: null,
          }));

          return true; // Completed
        }

        return false; // Still in progress
      } catch (err) {
        console.error('Polling error:', err);
        // Continue polling on transient errors
        return false;
      }
    },
    [onEvent, onComplete, onError, stopPolling]
  );

  // Start polling for events
  const startPolling = useCallback(
    async (chatId: string, sessionId: string) => {
      if (isPollingRef.current) return;
      isPollingRef.current = true;

      // Initial poll (get all events so far)
      const completed = await pollEvents(chatId, sessionId);
      if (completed) return;

      // Continue polling every 2 seconds if still in progress
      pollingRef.current = setInterval(async () => {
        const completed = await pollEvents(chatId, sessionId);
        if (completed) {
          stopPolling();
        }
      }, POLL_INTERVAL_MS);
    },
    [pollEvents, stopPolling]
  );

  // Check for active research and reconnect
  const checkAndReconnect = useCallback(async (): Promise<ReconnectionResult> => {
    if (!chatId || !enabled) return { reconnected: false };

    try {
      const res = await fetch(`/api/v1/chats/${chatId}/research/active`);

      if (!res.ok) {
        return { reconnected: false };
      }

      const data: ActiveResearchResponse = await res.json();

      if (!data.hasActiveResearch || !data.sessionId) {
        return { reconnected: false }; // No active research to reconnect to
      }

      setState({
        isReconnecting: true,
        reconnectionError: null,
        sessionId: data.sessionId,
      });

      // Reset sequence counter (start from 0 to get all events)
      lastSequenceRef.current = 0;

      // Start polling for events
      await startPolling(chatId, data.sessionId);

      return {
        reconnected: true,
        queryMode: data.queryMode,
        startedAt: data.startedAt,
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setState({
        isReconnecting: false,
        reconnectionError: errorMessage,
        sessionId: null,
      });
      return { reconnected: false };
    }
  }, [chatId, enabled, startPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  // Reset state when chatId changes
  useEffect(() => {
    setState({
      isReconnecting: false,
      reconnectionError: null,
      sessionId: null,
    });
    lastSequenceRef.current = 0;
    stopPolling();
  }, [chatId, stopPolling]);

  return {
    ...state,
    checkAndReconnect,
    stopPolling,
  };
}

export type { ResearchStateResponse };
