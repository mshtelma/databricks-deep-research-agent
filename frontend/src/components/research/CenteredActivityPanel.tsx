import * as React from 'react';
import { cn } from '@/lib/utils';
import type { StreamEvent } from '@/types';
import { EnhancedEventLabel } from './EnhancedEventLabel';
import { PlanProgress } from './PlanProgress';

interface PlanStep {
  index: number;
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  stepType?: 'research' | 'analysis';
  needsSearch?: boolean;
  observation?: string | null;
}

interface Plan {
  title?: string;
  reasoning?: string;
  thought?: string;
  steps: PlanStep[];
  iteration?: number;
}

interface CenteredActivityPanelProps {
  events: StreamEvent[];
  isLive?: boolean;
  className?: string;
  /** Maximum recent events to show (unused, now shows all with scroll) */
  maxEvents?: number;
  /** Research plan to display */
  plan?: Plan | null;
  /** Current step index in the plan */
  currentStepIndex?: number;
  /** Timestamp when research started (for elapsed time) */
  startTime?: number;
  /** Current active agent name */
  currentAgent?: string;
}

/**
 * Hook for live elapsed time that updates every second during streaming.
 */
function useElapsedTime(startTime: number | undefined, isLive: boolean): string {
  const [now, setNow] = React.useState(Date.now());

  React.useEffect(() => {
    if (!isLive || !startTime) return;

    const interval = setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => clearInterval(interval);
  }, [isLive, startTime]);

  if (!startTime) return '0:00';

  const elapsed = Math.floor((now - startTime) / 1000);
  const minutes = Math.floor(elapsed / 60);
  const seconds = elapsed % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Compute stats from events for display in stats bar.
 */
function useEventStats(events: StreamEvent[]) {
  return React.useMemo(() => {
    let searchQueries = 0;
    let sourcesFound = 0;
    let claimsVerified = 0;
    let claimsSupported = 0;

    for (const event of events) {
      if (event.eventType === 'tool_call') {
        const toolCall = event as unknown as { toolName?: string; tool_name?: string };
        const toolName = toolCall.toolName ?? toolCall.tool_name;
        if (toolName === 'web_search') searchQueries++;
      }
      if (event.eventType === 'tool_result') {
        const result = event as unknown as { urlsCrawled?: number; urls_crawled?: number };
        sourcesFound += result.urlsCrawled ?? result.urls_crawled ?? 0;
      }
      if (event.eventType === 'claim_verified') {
        claimsVerified++;
        const claim = event as unknown as { verdict?: string };
        if (claim.verdict === 'supported') claimsSupported++;
      }
    }

    return { searchQueries, sourcesFound, claimsVerified, claimsSupported };
  }, [events]);
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
 * Centered activity panel displayed during active research.
 * Shows all events with smooth animations for new entries only.
 *
 * This panel appears centered in the main content area during Deep Research
 * and transitions to the ActivityAccordion when research completes.
 */
export function CenteredActivityPanel({
  events,
  isLive = false,
  className,
  plan,
  currentStepIndex,
  startTime,
  currentAgent,
}: CenteredActivityPanelProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Track seen event IDs to only animate new events (prevents blinking)
  const seenEventIdsRef = React.useRef<Set<string>>(new Set());

  // Filter to interesting events only
  const filteredEvents = React.useMemo(
    () => filterInterestingEvents(events),
    [events]
  );

  // Compute stats for display
  const stats = useEventStats(events);

  // Live elapsed time
  const elapsedTime = useElapsedTime(startTime, isLive);

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

  // Show panel if we have plan OR events OR streaming (to prevent unmount flicker)
  const hasPlan = plan && plan.steps.length > 0;
  const hasEvents = filteredEvents.length > 0;

  // Only return null when NOT streaming AND nothing to show
  // During streaming, always render to prevent unmount/remount (keeps timer stable)
  if (!isLive && !hasPlan && !hasEvents) {
    return null;
  }

  return (
    <div
      className={cn(
        'mx-auto max-w-4xl w-full',
        'rounded-xl border bg-card shadow-md',
        'transition-all duration-300 ease-in-out',
        isLive && 'border-primary/40 ring-1 ring-primary/20',
        className
      )}
    >
      {/* Plan Progress Section */}
      {hasPlan && (
        <PlanProgress
          plan={plan}
          currentStepIndex={currentStepIndex}
          className="border-0 rounded-none rounded-t-lg"
        />
      )}

      {/* Stats Bar - shown during live streaming */}
      {isLive && (
        <div className="flex items-center gap-3 px-4 py-1.5 border-b text-xs text-muted-foreground bg-muted/20">
          {/* Live timer */}
          <div className="flex items-center gap-1">
            <span className="text-primary font-mono font-medium">{elapsedTime}</span>
          </div>

          {/* Separator */}
          <span className="text-muted-foreground/40">|</span>

          {/* Current agent */}
          {currentAgent && (
            <div className="flex items-center gap-1">
              <span className="animate-pulse text-primary">●</span>
              <span className="font-medium">{currentAgent}</span>
            </div>
          )}

          {/* Search queries */}
          {stats.searchQueries > 0 && (
            <span>{stats.searchQueries} searches</span>
          )}

          {/* Sources */}
          {stats.sourcesFound > 0 && (
            <span>{stats.sourcesFound} sources</span>
          )}

          {/* Verification stats */}
          {stats.claimsVerified > 0 && (
            <span className="text-green-600 dark:text-green-400">
              {stats.claimsSupported}/{stats.claimsVerified} verified
            </span>
          )}
        </div>
      )}

      {/* Activity Section */}
      {hasEvents ? (
        <>
          {/* Activity Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b bg-gradient-to-r from-muted/40 to-transparent">
            <div className="flex items-center gap-2">
              {isLive && (
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary" />
                </span>
              )}
              <span className="text-sm font-semibold">Research Activity</span>
            </div>
            <span className="text-xs text-muted-foreground font-medium">
              {filteredEvents.length} events
            </span>
          </div>

          {/* Events list - shows all events with scroll */}
          <div
            ref={containerRef}
            className="p-4 space-y-2 max-h-[350px] overflow-y-auto scroll-smooth"
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
                    'py-2 px-3 rounded-lg transition-all duration-200',
                    isLatest && 'bg-primary/10 border-l-[3px] border-primary shadow-sm',
                    event.eventType === 'error' && 'bg-red-50 dark:bg-red-950/40 border-l-[3px] border-red-500',
                    !isLatest && event.eventType !== 'error' && 'hover:bg-muted/50',
                    // Only apply animation to new events (prevents blinking)
                    isNewEvent && 'animate-in slide-in-from-bottom-2 fade-in duration-200'
                  )}
                >
                  <EnhancedEventLabel event={event} />
                </div>
              );
            })}
          </div>
        </>
      ) : isLive ? (
        /* Placeholder when streaming but no events yet */
        <div className="p-6 text-center text-muted-foreground">
          <div className="flex items-center justify-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary" />
            </span>
            <span className="animate-pulse">Starting research...</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Filter to keep only interesting events for live display.
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
      // 'tool_result' - handled above with conditional check
      'reflection_decision',
      'synthesis_started',
      'research_completed',
      'claim_verified',
      'verification_summary',
    ].includes(event.eventType);
  });
}
