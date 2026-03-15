import * as React from 'react';
import { cn } from '@/lib/utils';

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

interface VisitedSourcesPanelProps {
  sources: SourceInfo[];
  className?: string;
  /** Show step grouping for sources */
  showStepGrouping?: boolean;
  /** Show only cited sources */
  citedOnly?: boolean;
}

/**
 * Comprehensive panel displaying all sources visited during research.
 * Separates cited sources from all visited sources, with step grouping.
 */
export function VisitedSourcesPanel({
  sources,
  className,
  showStepGrouping = true,
  citedOnly = false,
}: VisitedSourcesPanelProps) {
  const [expandedSection, setExpandedSection] = React.useState<'cited' | 'all' | null>('cited');

  if (sources.length === 0) {
    return null;
  }

  // Separate cited and uncited sources
  const citedSources = sources.filter((s) => s.is_cited);
  const allSources = sources;

  // Group sources by step if enabled
  const groupedSources = showStepGrouping
    ? groupSourcesByStep(allSources)
    : { ungrouped: allSources };

  // If citedOnly mode, just show cited sources
  if (citedOnly) {
    return (
      <div className={cn('rounded-lg border bg-card', className)}>
        <div className="p-3 border-b bg-muted/30">
          <h4 className="text-sm font-medium">
            Sources ({citedSources.length})
          </h4>
        </div>
        <div className="p-3 space-y-2 max-h-[300px] overflow-y-auto">
          {citedSources.map((source, i) => (
            <SourceItem key={source.url + i} source={source} index={i} showCitedBadge={false} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('rounded-lg border bg-card overflow-hidden', className)}>
      {/* Cited Sources Section */}
      {citedSources.length > 0 && (
        <CollapsibleSection
          title="Cited Sources"
          count={citedSources.length}
          isExpanded={expandedSection === 'cited'}
          onToggle={() => setExpandedSection(expandedSection === 'cited' ? null : 'cited')}
          variant="primary"
        >
          <div className="space-y-2">
            {citedSources.map((source, i) => (
              <SourceItem key={source.url + i} source={source} index={i + 1} showCitedBadge={false} />
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* All Visited Sources Section */}
      <CollapsibleSection
        title="All Visited Sources"
        count={allSources.length}
        isExpanded={expandedSection === 'all'}
        onToggle={() => setExpandedSection(expandedSection === 'all' ? null : 'all')}
        variant="secondary"
      >
        {showStepGrouping ? (
          <div className="space-y-4">
            {Object.entries(groupedSources).map(([stepKey, stepSources]) => (
              <StepSourceGroup
                key={stepKey}
                stepTitle={stepKey === 'ungrouped' ? 'Background Research' : stepKey}
                sources={stepSources}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {allSources.map((source, i) => (
              <SourceItem key={source.url + i} source={source} showCitedBadge />
            ))}
          </div>
        )}
      </CollapsibleSection>
    </div>
  );
}

interface CollapsibleSectionProps {
  title: string;
  count: number;
  isExpanded: boolean;
  onToggle: () => void;
  variant: 'primary' | 'secondary';
  children: React.ReactNode;
}

function CollapsibleSection({
  title,
  count,
  isExpanded,
  onToggle,
  variant,
  children,
}: CollapsibleSectionProps) {
  return (
    <div className={cn(variant === 'secondary' && 'border-t')}>
      <button
        type="button"
        onClick={onToggle}
        className={cn(
          'w-full flex items-center justify-between p-3 text-left',
          'hover:bg-muted/50 transition-colors',
          variant === 'primary' && 'bg-muted/30',
          isExpanded && 'border-b'
        )}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{title}</span>
          <span className="text-xs text-muted-foreground">({count})</span>
        </div>
        <ChevronIcon
          className={cn(
            'w-4 h-4 text-muted-foreground transition-transform duration-200',
            isExpanded && 'rotate-180'
          )}
        />
      </button>
      <div
        className={cn(
          'overflow-hidden transition-[max-height] duration-300 ease-in-out',
          isExpanded ? 'max-h-[500px]' : 'max-h-0'
        )}
      >
        <div className="p-3 overflow-y-auto max-h-[450px]">{children}</div>
      </div>
    </div>
  );
}

interface StepSourceGroupProps {
  stepTitle: string;
  sources: SourceInfo[];
}

function StepSourceGroup({ stepTitle, sources }: StepSourceGroupProps) {
  return (
    <div>
      <h5 className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
        <span aria-hidden="true">\u25B8</span>
        {stepTitle}
      </h5>
      <div className="space-y-2 pl-3 border-l-2 border-muted">
        {sources.map((source, i) => (
          <SourceItem key={source.url + i} source={source} showCitedBadge />
        ))}
      </div>
    </div>
  );
}

interface SourceItemProps {
  source: SourceInfo;
  index?: number;
  showCitedBadge?: boolean;
}

function SourceItem({ source, index, showCitedBadge = true }: SourceItemProps) {
  const domain = getDomain(source.url);
  const hasFailed = source.crawl_status && source.crawl_status !== 'success';

  return (
    <div
      className={cn(
        'group rounded-md p-2 transition-colors',
        'hover:bg-muted/50',
        hasFailed && 'opacity-60'
      )}
    >
      <div className="flex items-start gap-2">
        {/* Index badge if provided */}
        {index !== undefined && (
          <span className="flex-shrink-0 w-5 h-5 rounded bg-primary/10 text-primary text-xs flex items-center justify-center font-medium">
            {index}
          </span>
        )}

        <div className="flex-1 min-w-0">
          {/* Title and badges */}
          <div className="flex items-center gap-2 flex-wrap">
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-foreground hover:text-primary truncate max-w-[300px]"
              title={source.title || source.url}
            >
              {source.title || domain}
            </a>

            {showCitedBadge && source.is_cited && (
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                Cited
              </span>
            )}

            {hasFailed && (
              <span
                className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                title={source.error_reason || `Crawl ${source.crawl_status}`}
              >
                {source.crawl_status === 'timeout' ? 'Timeout' : 'Failed'}
              </span>
            )}
          </div>

          {/* Domain */}
          <p className="text-xs text-muted-foreground truncate">{domain}</p>

          {/* Snippet preview */}
          {source.snippet && (
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {source.snippet.slice(0, 150)}
              {source.snippet.length > 150 && '\u2026'}
            </p>
          )}
        </div>

        {/* External link icon */}
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
          aria-label="Open source"
        >
          <ExternalLinkIcon className="w-4 h-4 text-muted-foreground hover:text-foreground" />
        </a>
      </div>
    </div>
  );
}

/** Extract domain from URL */
function getDomain(url: string): string {
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return url;
  }
}

/** Group sources by their step */
function groupSourcesByStep(
  sources: SourceInfo[]
): Record<string, SourceInfo[]> {
  const groups: Record<string, SourceInfo[]> = {};

  for (const source of sources) {
    const key = source.step_title || `Step ${source.step_index ?? 0}`;
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(source);
  }

  return groups;
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

function ExternalLinkIcon({ className }: { className?: string }) {
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
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}
