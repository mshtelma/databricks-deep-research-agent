import * as React from 'react';
import { cn } from '@/lib/utils';
import type { StreamEvent } from '@/types';
import { PlanProgress } from './PlanProgress';
import { EventFeed } from './EventFeed';

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

interface ActivityTabContentProps {
  events: StreamEvent[];
  plan: Plan | null;
  currentStepIndex: number;
  isLive?: boolean;
  startTime?: number;
  currentAgent?: string;
  className?: string;
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
 * ActivityTabContent - Two-column layout for Plan + Event Feed.
 * Responsive: stacks vertically on mobile, side-by-side on desktop.
 */
export function ActivityTabContent({
  events,
  plan,
  currentStepIndex,
  isLive = false,
  startTime,
  currentAgent,
  className,
}: ActivityTabContentProps) {
  const stats = useEventStats(events);
  const elapsedTime = useElapsedTime(startTime, isLive);

  const hasPlan = plan && plan.steps.length > 0;
  const hasEvents = events.length > 0;

  // Show placeholder if nothing to display
  if (!hasPlan && !hasEvents && !isLive) {
    return (
      <div className={cn('p-4 text-center text-muted-foreground text-sm', className)}>
        No activity to display
      </div>
    );
  }

  return (
    <div className={cn('space-y-2', className)}>
      {/* Stats Bar - shown during live streaming */}
      {isLive && (
        <div className="flex items-center gap-2 px-2 py-1 text-[11px] text-muted-foreground bg-muted/30 rounded">
          {/* Live timer */}
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
            </span>
            <span className="text-primary font-mono font-medium">{elapsedTime}</span>
          </div>

          {/* Separator */}
          <span className="text-muted-foreground/40">|</span>

          {/* Current agent */}
          {currentAgent && (
            <span className="font-medium">{currentAgent}</span>
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

      {/* Two-column layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {/* Left column: Research Plan */}
        <div className="order-2 md:order-1">
          {hasPlan ? (
            <PlanProgress
              plan={plan}
              currentStepIndex={currentStepIndex}
              showDetails
              className="h-full"
            />
          ) : isLive ? (
            <div className="rounded border bg-card p-3 h-full flex items-center justify-center">
              <div className="text-center text-xs text-muted-foreground">
                <div className="flex items-center justify-center gap-1.5 mb-1">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                  </span>
                  <span className="animate-pulse">Creating plan...</span>
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Right column: Event Feed - structure matches PlanProgress */}
        <div className="order-1 md:order-2">
          <div className="rounded-lg border bg-card p-3 h-full flex flex-col">
            {/* Header matches PlanProgress style */}
            <div className="flex items-center justify-between mb-2 flex-shrink-0">
              <h4 className="font-semibold text-xs flex items-center gap-1.5">
                <ActivityIcon className="w-3.5 h-3.5" />
                Activity Feed
              </h4>
              <span className="text-[10px] text-muted-foreground">
                ({events.length})
              </span>
            </div>
            <EventFeed
              events={events}
              isLive={isLive}
              maxHeight="240px"
              className="flex-1 min-h-0 pr-0.5"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function ActivityIcon({ className }: { className?: string }) {
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
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}

export default ActivityTabContent;
