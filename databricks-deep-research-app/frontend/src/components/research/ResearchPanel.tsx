import { cn } from '@/lib/utils';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { useResearchPanel, type ResearchPanelTab } from '@/hooks/useResearchPanel';
import { ActivityTabContent } from './ActivityTabContent';
import { SourceGroupedCitations } from '@/components/citations/SourceGroupedCitations';
import { VisitedSourcesPanel } from './VisitedSourcesPanel';
import type { StreamEvent } from '@/types';
import type { Claim, VerificationSummary } from '@/types/citation';

interface PlanStep {
  index: number;
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  step_type?: 'research' | 'analysis';
  needs_search?: boolean;
  observation?: string | null;
}

interface Plan {
  title?: string;
  reasoning?: string;
  thought?: string;
  steps: PlanStep[];
  iteration?: number;
}

interface SourceInfo {
  url: string;
  title?: string | null;
  snippet?: string | null;
  is_cited?: boolean;
  step_index?: number;
  step_title?: string;
  crawl_status?: 'success' | 'failed' | 'timeout' | 'blocked';
  error_reason?: string | null;
}

interface ResearchPanelProps {
  /** Whether research is currently streaming */
  isStreaming: boolean;

  /** Activity data from useStreamingQuery */
  events: StreamEvent[];
  plan: Plan | null;
  currentStepIndex: number;
  startTime?: number;
  currentAgent?: string;

  /** Sources data from useCitations/API */
  claims: Claim[];
  allSources: SourceInfo[];
  verificationSummary?: VerificationSummary | null;

  className?: string;
}

/**
 * ResearchPanel - Tabbed panel for research activity, cited sources, and all sources.
 *
 * Behavior:
 * - During streaming: expanded with Activity tab active
 * - After streaming: auto-collapses, switches to Cited Sources tab
 * - Manual expand/collapse via toggle button
 */
export function ResearchPanel({
  isStreaming,
  events,
  plan,
  currentStepIndex,
  startTime,
  currentAgent,
  claims,
  allSources,
  verificationSummary,
  className,
}: ResearchPanelProps) {
  const hasContent = events.length > 0 || claims.length > 0 || allSources.length > 0;

  const {
    activeTab,
    setActiveTab,
    isExpanded,
    toggleExpanded,
    showLiveIndicator,
  } = useResearchPanel(isStreaming, hasContent);

  // Count badges for tabs
  const citedCount = claims.length;
  const allSourcesCount = allSources.length;

  return (
    <div
      className={cn(
        'mx-auto max-w-4xl w-full',
        'rounded-xl border bg-card shadow-md',
        'transition-all duration-300 ease-in-out',
        isStreaming && 'border-primary/40 ring-1 ring-primary/20',
        className
      )}
    >
      <Tabs
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as ResearchPanelTab)}
        className="w-full"
      >
        {/* Tab Header */}
        <div className="flex items-center justify-between px-2 py-1.5 border-b bg-muted/50">
          <TabsList className="bg-transparent p-0 h-auto gap-0.5">
            <TabsTrigger
              value="activity"
              className={cn(
                'text-[11px] px-2 py-1 rounded',
                'data-[state=active]:bg-background data-[state=active]:shadow-sm'
              )}
            >
              <span className="flex items-center gap-1">
                {showLiveIndicator && (
                  <span className="relative flex h-1.5 w-1.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary" />
                  </span>
                )}
                Activity
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="cited"
              className={cn(
                'text-[11px] px-2 py-1 rounded',
                'data-[state=active]:bg-background data-[state=active]:shadow-sm'
              )}
            >
              <span className="flex items-center gap-1">
                Cited Sources
                {citedCount > 0 && (
                  <span className="text-[9px] px-1 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
                    {citedCount}
                  </span>
                )}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="all"
              className={cn(
                'text-[11px] px-2 py-1 rounded',
                'data-[state=active]:bg-background data-[state=active]:shadow-sm'
              )}
            >
              <span className="flex items-center gap-1">
                All Sources
                {allSourcesCount > 0 && (
                  <span className="text-[9px] px-1 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                    {allSourcesCount}
                  </span>
                )}
              </span>
            </TabsTrigger>
          </TabsList>

          {/* Expand/Collapse Button */}
          <button
            type="button"
            onClick={toggleExpanded}
            className={cn(
              'flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[11px] text-muted-foreground',
              'hover:bg-muted/50 transition-colors'
            )}
          >
            <span>{isExpanded ? 'Collapse' : 'Expand'}</span>
            <ChevronIcon
              className={cn(
                'w-3.5 h-3.5 transition-transform duration-200',
                !isExpanded && 'rotate-180'
              )}
            />
          </button>
        </div>

        {/* Tab Content - Collapsible */}
        <div
          className={cn(
            'overflow-hidden transition-all duration-300 ease-in-out',
            isExpanded ? 'max-h-[420px] opacity-100' : 'max-h-0 opacity-0'
          )}
        >
          <TabsContent value="activity" className="mt-0 p-2">
            <ActivityTabContent
              events={events}
              plan={plan}
              currentStepIndex={currentStepIndex}
              isLive={isStreaming}
              startTime={startTime}
              currentAgent={currentAgent}
            />
          </TabsContent>

          <TabsContent value="cited" className="mt-0 p-2 max-h-[375px] overflow-y-auto">
            {claims.length > 0 ? (
              <SourceGroupedCitations
                claims={claims}
                verificationSummary={verificationSummary}
              />
            ) : (
              <div className="text-center py-8 text-muted-foreground text-sm">
                {isStreaming ? (
                  <div className="flex items-center justify-center gap-2">
                    <span className="animate-pulse">Citations will appear here...</span>
                  </div>
                ) : (
                  'No cited sources available'
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="all" className="mt-0 p-2 max-h-[375px] overflow-y-auto">
            {allSources.length > 0 ? (
              <VisitedSourcesPanel
                sources={allSources}
                showStepGrouping
                className="border-0 rounded-none"
              />
            ) : (
              <div className="text-center py-8 text-muted-foreground text-sm">
                {isStreaming ? (
                  <div className="flex items-center justify-center gap-2">
                    <span className="animate-pulse">Sources will appear here...</span>
                  </div>
                ) : (
                  'No sources visited'
                )}
              </div>
            )}
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
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

export default ResearchPanel;
