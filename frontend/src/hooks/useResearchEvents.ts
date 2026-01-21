/**
 * Hook for accumulating and managing research events during streaming.
 *
 * This hook:
 * 1. Accumulates SSE events during research streaming
 * 2. Tracks streaming state (isStreaming)
 * 3. Provides event array for CenteredActivityPanel and ActivityAccordion
 * 4. Supports clearing events for new research sessions
 */

import { useState, useCallback, useMemo } from 'react';
import type { StreamEvent } from '@/types';

/**
 * Categorized event for display purposes.
 */
export interface CategorizedEvent {
  /** Original event data */
  event: StreamEvent;
  /** Event category for grouping/filtering */
  category: 'agent' | 'step' | 'verification' | 'tool' | 'synthesis' | 'error' | 'other';
  /** Timestamp when event was received */
  receivedAt: Date;
  /** Unique key for React rendering */
  key: string;
}

/**
 * Event statistics for summary display.
 */
export interface EventStats {
  /** Total number of events */
  total: number;
  /** Events by category */
  byCategory: Record<string, number>;
  /** Number of verified claims */
  claimsVerified: number;
  /** Number of tool calls made */
  toolCalls: number;
  /** Number of steps completed */
  stepsCompleted: number;
}

export interface UseResearchEventsReturn {
  /** All accumulated events */
  events: CategorizedEvent[];
  /** Whether streaming is active */
  isStreaming: boolean;
  /** Event statistics */
  stats: EventStats;
  /** Add a new event */
  addEvent: (event: StreamEvent) => void;
  /** Mark streaming as started */
  startStreaming: () => void;
  /** Mark streaming as ended */
  endStreaming: () => void;
  /** Clear all events (for new session) */
  clearEvents: () => void;
  /** Get events by category */
  getEventsByCategory: (category: CategorizedEvent['category']) => CategorizedEvent[];
}

/**
 * Categorize an event based on its type.
 */
function categorizeEvent(event: StreamEvent): CategorizedEvent['category'] {
  // StreamEvent uses camelCase: eventType
  const eventType = event.eventType;

  if (!eventType) return 'other';

  // Agent lifecycle events
  if (eventType.includes('agent_')) return 'agent';

  // Step events
  if (eventType.includes('step_')) return 'step';

  // Verification events
  if (
    eventType === 'claim_verified' ||
    eventType === 'verification_summary' ||
    eventType === 'citation_corrected' ||
    eventType === 'numeric_claim_detected'
  ) {
    return 'verification';
  }

  // Tool events
  if (eventType === 'tool_call' || eventType === 'tool_result') return 'tool';

  // Synthesis events
  if (eventType.includes('synthesis')) return 'synthesis';

  // Error events
  if (eventType === 'error') return 'error';

  return 'other';
}

/**
 * Generate a unique key for an event.
 */
function generateEventKey(event: StreamEvent, index: number): string {
  const eventType = event.eventType || 'unknown';
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 6);
  return `${eventType}-${index}-${timestamp}-${random}`;
}

/** Maximum number of events to retain (sliding window for memory safety) */
const MAX_EVENTS = 1000;

/**
 * Hook for managing research events during streaming.
 */
export function useResearchEvents(): UseResearchEventsReturn {
  const [events, setEvents] = useState<CategorizedEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const addEvent = useCallback((event: StreamEvent) => {
    setEvents((prev) => {
      const newEvent: CategorizedEvent = {
        event,
        category: categorizeEvent(event),
        receivedAt: new Date(),
        key: generateEventKey(event, prev.length), // Use prev.length to avoid stale closure
      };

      // Keep only the most recent MAX_EVENTS (sliding window for memory safety)
      const updated = [...prev, newEvent];
      if (updated.length > MAX_EVENTS) {
        return updated.slice(-MAX_EVENTS);
      }
      return updated;
    });
  }, []); // No dependencies - all state accessed via functional update

  const startStreaming = useCallback(() => {
    setIsStreaming(true);
  }, []);

  const endStreaming = useCallback(() => {
    setIsStreaming(false);
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
    setIsStreaming(false);
  }, []);

  const getEventsByCategory = useCallback(
    (category: CategorizedEvent['category']) => {
      return events.filter((e) => e.category === category);
    },
    [events]
  );

  // Compute statistics
  const stats = useMemo<EventStats>(() => {
    const byCategory: Record<string, number> = {};
    let claimsVerified = 0;
    let toolCalls = 0;
    let stepsCompleted = 0;

    for (const categorizedEvent of events) {
      const { category, event } = categorizedEvent;
      byCategory[category] = (byCategory[category] || 0) + 1;

      const eventType = event.eventType;
      if (eventType === 'claim_verified') claimsVerified++;
      if (eventType === 'tool_call') toolCalls++;
      if (eventType === 'step_completed') stepsCompleted++;
    }

    return {
      total: events.length,
      byCategory,
      claimsVerified,
      toolCalls,
      stepsCompleted,
    };
  }, [events]);

  return {
    events,
    isStreaming,
    stats,
    addEvent,
    startStreaming,
    endStreaming,
    clearEvents,
    getEventsByCategory,
  };
}

export default useResearchEvents;
