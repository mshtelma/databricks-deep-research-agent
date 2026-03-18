import * as React from 'react';
import { cn } from '@/lib/utils';
import type { StreamEvent } from '@/types';
import { EnhancedEventLabel } from './EnhancedEventLabel';

interface ActivityAccordionProps {
  events: StreamEvent[];
  className?: string;
  /** Default expanded state */
  defaultExpanded?: boolean;
  /** Maximum events to show initially (pagination) */
  initialLimit?: number;
}

/**
 * Collapsible accordion for displaying research activity events.
 * Used after research completes to show all events in a compact format.
 */
export function ActivityAccordion({
  events,
  className,
  defaultExpanded = false,
  initialLimit = 50,
}: ActivityAccordionProps) {
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);
  const [displayLimit, setDisplayLimit] = React.useState(initialLimit);

  if (events.length === 0) {
    return null;
  }

  // Filter out duplicate events and noise
  const filteredEvents = filterEvents(events);
  const displayEvents = filteredEvents.slice(0, displayLimit);
  const hasMore = filteredEvents.length > displayLimit;
  const remaining = filteredEvents.length - displayLimit;

  const handleLoadMore = () => {
    setDisplayLimit((prev) => prev + 50);
  };

  return (
    <div className={cn('rounded-lg border bg-card overflow-hidden', className)}>
      {/* Header - always visible */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          'w-full flex items-center justify-between p-3 text-left',
          'hover:bg-muted/50 transition-colors',
          isExpanded && 'border-b'
        )}
      >
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-sm">
            {isExpanded ? '\u25BC' : '\u25B6'}
          </span>
          <span className="text-sm font-medium">
            Research Activity
          </span>
          <span className="text-xs text-muted-foreground">
            ({filteredEvents.length} events)
          </span>
        </div>
        <ChevronIcon
          className={cn(
            'w-4 h-4 text-muted-foreground transition-transform duration-200',
            isExpanded && 'rotate-180'
          )}
        />
      </button>

      {/* Content - collapsible */}
      <div
        className={cn(
          'overflow-hidden transition-[max-height] duration-300 ease-in-out',
          isExpanded ? 'max-h-[500px]' : 'max-h-0'
        )}
      >
        <div className="p-3 space-y-1 overflow-y-auto max-h-[450px]">
          {displayEvents.map((event, index) => (
            <div
              key={`${event.eventType}-${index}`}
              className={cn(
                'py-1 animate-in fade-in-50 duration-200',
                event.eventType === 'error' && 'bg-red-50 dark:bg-red-950/30 rounded px-2'
              )}
              style={{ animationDelay: `${Math.min(index * 20, 300)}ms` }}
            >
              <EnhancedEventLabel event={event} />
            </div>
          ))}

          {/* Load more button */}
          {hasMore && (
            <button
              type="button"
              onClick={handleLoadMore}
              className="w-full py-2 text-xs text-primary hover:text-primary/80 transition-colors"
            >
              Load {Math.min(remaining, 50)} more ({remaining} remaining)
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Filter events to remove noise and duplicates.
 * Keeps the most informative events for the activity log.
 */
function filterEvents(events: StreamEvent[]): StreamEvent[] {
  return events.filter((event) => {
    // Always keep errors
    if (event.eventType === 'error') return true;

    // Filter out duplicate synthesis_progress events (keep one)
    if (event.eventType === 'synthesis_progress') return false;

    // Keep all other meaningful events
    return [
      'agent_started',
      'agent_completed',
      'plan_created',
      'step_started',
      'step_completed',
      'tool_call',
      'tool_result',
      'reflection_decision',
      'synthesis_started',
      'research_completed',
      'claim_verified',
      'verification_summary',
      'numeric_claim_detected',
      'citation_corrected',
    ].includes(event.eventType);
  });
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
