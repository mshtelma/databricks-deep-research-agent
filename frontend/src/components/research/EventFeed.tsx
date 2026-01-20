import * as React from 'react';
import { cn } from '@/lib/utils';
import type { StreamEvent } from '@/types';
import { EnhancedEventLabel } from './EnhancedEventLabel';

interface EventFeedProps {
  events: StreamEvent[];
  isLive?: boolean;
  className?: string;
  /** Maximum height for the scrollable area */
  maxHeight?: string;
}

/**
 * Get stable event ID for React keys.
 * Uses _eventId if available, falls back to index-based key.
 */
function getEventId(event: StreamEvent, index: number): string {
  const eventWithId = event as unknown as { _eventId?: string };
  return eventWithId._eventId ?? `${event.eventType}-fallback-${index}`;
}

/**
 * Filter to keep only interesting events for display.
 */
function filterInterestingEvents(events: StreamEvent[]): StreamEvent[] {
  return events.filter((event) => {
    // Always show errors
    if (event.eventType === 'error') return true;

    // Skip synthesis_progress (too noisy during writing)
    if (event.eventType === 'synthesis_progress') return false;

    // Skip tool_result events with no useful info (web_search results or failed crawls)
    if (event.eventType === 'tool_result') {
      const result = event as unknown as { sourcesCrawled?: number; sources_crawled?: number };
      const sourcesCrawled = result.sourcesCrawled ?? result.sources_crawled;
      return sourcesCrawled != null && sourcesCrawled > 0;
    }

    // Keep meaningful milestone events
    return [
      'agent_started',
      'agent_completed',
      'plan_created',
      'step_started',
      'step_completed',
      'tool_call',
      'reflection_decision',
      'synthesis_started',
      'research_completed',
      'claim_verified',
      'verification_summary',
    ].includes(event.eventType);
  });
}

/**
 * EventFeed - Displays a scrollable list of research activity events.
 * Auto-scrolls to the latest event during live streaming.
 */
export function EventFeed({
  events,
  isLive = false,
  className,
  maxHeight,
}: EventFeedProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Track seen event IDs to only animate new events (prevents blinking)
  const seenEventIdsRef = React.useRef<Set<string>>(new Set());

  // Filter to interesting events only
  const filteredEvents = React.useMemo(
    () => filterInterestingEvents(events),
    [events]
  );

  // Auto-scroll to latest event
  React.useEffect(() => {
    if (containerRef.current && isLive) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [filteredEvents.length, isLive]);

  // Mark events as seen after render (prevents animation re-trigger)
  React.useEffect(() => {
    filteredEvents.forEach((event, index) => {
      const id = getEventId(event, index);
      seenEventIdsRef.current.add(id);
    });
  }, [filteredEvents]);

  if (filteredEvents.length === 0) {
    return (
      <div className={cn('p-4 text-center text-muted-foreground text-sm', className)}>
        {isLive ? (
          <div className="flex items-center justify-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
            </span>
            <span className="animate-pulse">Waiting for activity...</span>
          </div>
        ) : (
          'No activity recorded'
        )}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn('space-y-1 overflow-y-auto scroll-smooth', className)}
      style={maxHeight ? { maxHeight } : undefined}
    >
      {filteredEvents.map((event, index) => {
        const eventId = getEventId(event, index);
        const isLatest = index === filteredEvents.length - 1 && isLive;
        // Only animate events we haven't seen before
        const isNewEvent = !seenEventIdsRef.current.has(eventId);

        return (
          <div
            key={eventId}
            className={cn(
              'py-1 px-1.5 rounded transition-all duration-200',
              isLatest && 'bg-primary/10 border-l-2 border-primary',
              event.eventType === 'error' && 'bg-red-50 dark:bg-red-950/40 border-l-2 border-red-500',
              !isLatest && event.eventType !== 'error' && 'hover:bg-muted/50',
              // Only apply animation to new events (prevents blinking)
              isNewEvent && 'animate-in slide-in-from-bottom-1 fade-in duration-200'
            )}
          >
            <EnhancedEventLabel event={event} />
          </div>
        );
      })}
    </div>
  );
}

export default EventFeed;
