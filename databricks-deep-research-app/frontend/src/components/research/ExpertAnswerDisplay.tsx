/**
 * ExpertAnswerDisplay - Component to display Knowledge Assistant responses.
 *
 * Features (T049):
 * - Display Knowledge Assistant responses
 * - Show confidence level indicator
 * - Citation list with source references
 * - Context indicator (whether research context was included)
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { MarkdownRenderer } from '@/components/common/MarkdownRenderer';
import type {
  KnowledgeAssistantResult,
  AssistantConfidenceLevel,
  AssistantSourceReference,
} from '@/types/dataSources';

interface ExpertAnswerDisplayProps {
  result: KnowledgeAssistantResult;
  sourceName?: string;
  className?: string;
}

export function ExpertAnswerDisplay({
  result,
  sourceName,
  className,
}: ExpertAnswerDisplayProps) {
  const [showSources, setShowSources] = React.useState(false);

  return (
    <div className={cn('rounded-lg border bg-card', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-emerald-50/50 dark:bg-emerald-950/20">
        <div className="flex items-center gap-2">
          <BrainIcon className="h-4 w-4 text-emerald-600" />
          <span className="font-medium text-sm">
            {sourceName ? `Expert: ${sourceName}` : 'Knowledge Assistant'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Context indicator */}
          {result.includedContext && (
            <span
              className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
              title="Research context was provided to the assistant"
            >
              <ContextIcon className="h-3 w-3" />
              Context Used
            </span>
          )}

          {/* Confidence indicator */}
          <ConfidenceIndicator level={result.confidenceLevel} />
        </div>
      </div>

      {/* Answer content */}
      <div className="px-4 py-3">
        <MarkdownRenderer content={result.answer} className="text-sm" />
      </div>

      {/* Source references */}
      {result.sources.length > 0 && (
        <div className="border-t">
          <button
            type="button"
            onClick={() => setShowSources(!showSources)}
            className="w-full flex items-center justify-between px-4 py-2 text-sm text-muted-foreground hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <BookmarkIcon className="h-4 w-4" />
              <span>
                {result.sources.length} Source Reference{result.sources.length !== 1 ? 's' : ''}
              </span>
            </div>
            <ChevronIcon
              className={cn('h-4 w-4 transition-transform', showSources && 'rotate-180')}
            />
          </button>

          {showSources && (
            <div className="px-4 py-3 bg-muted/30 border-t space-y-2">
              {result.sources.map((source, index) => (
                <SourceReferenceCard key={index} source={source} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ConfidenceIndicatorProps {
  level: AssistantConfidenceLevel;
}

function ConfidenceIndicator({ level }: ConfidenceIndicatorProps) {
  const config = CONFIDENCE_CONFIG[level];

  return (
    <div
      className={cn(
        'flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium',
        config.bgClass
      )}
      title={`Confidence Level: ${config.label}`}
    >
      <div className="flex items-center gap-0.5">
        {[1, 2, 3].map((bar) => (
          <div
            key={bar}
            className={cn(
              'w-1 rounded-full transition-all',
              bar === 1 ? 'h-2' : bar === 2 ? 'h-3' : 'h-4',
              bar <= config.bars ? config.barClass : 'bg-gray-300 dark:bg-gray-600'
            )}
          />
        ))}
      </div>
      <span className={config.textClass}>{config.label}</span>
    </div>
  );
}

const CONFIDENCE_CONFIG: Record<
  AssistantConfidenceLevel,
  {
    label: string;
    bars: number;
    bgClass: string;
    textClass: string;
    barClass: string;
  }
> = {
  high: {
    label: 'High Confidence',
    bars: 3,
    bgClass: 'bg-green-100 dark:bg-green-900/50',
    textClass: 'text-green-800 dark:text-green-200',
    barClass: 'bg-green-600 dark:bg-green-400',
  },
  medium: {
    label: 'Medium Confidence',
    bars: 2,
    bgClass: 'bg-amber-100 dark:bg-amber-900/50',
    textClass: 'text-amber-800 dark:text-amber-200',
    barClass: 'bg-amber-600 dark:bg-amber-400',
  },
  low: {
    label: 'Low Confidence',
    bars: 1,
    bgClass: 'bg-red-100 dark:bg-red-900/50',
    textClass: 'text-red-800 dark:text-red-200',
    barClass: 'bg-red-600 dark:bg-red-400',
  },
};

interface SourceReferenceCardProps {
  source: AssistantSourceReference;
}

function SourceReferenceCard({ source }: SourceReferenceCardProps) {
  return (
    <div className="p-3 rounded-md border bg-background">
      <div className="flex items-start gap-2">
        <DocumentIcon className="h-4 w-4 text-muted-foreground mt-0.5" />
        <div className="flex-1 min-w-0">
          {source.url ? (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-primary hover:underline truncate block"
            >
              {source.title}
            </a>
          ) : (
            <span className="text-sm font-medium truncate block">{source.title}</span>
          )}
          {source.snippet && (
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {source.snippet}
            </p>
          )}
        </div>
        {source.url && (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground transition-colors"
            title="Open in new tab"
          >
            <ExternalLinkIcon className="h-4 w-4" />
          </a>
        )}
      </div>
    </div>
  );
}

// Icons
function BrainIcon({ className }: { className?: string }) {
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
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
    </svg>
  );
}

function ContextIcon({ className }: { className?: string }) {
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
      <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  );
}

function BookmarkIcon({ className }: { className?: string }) {
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
      <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z" />
    </svg>
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

function DocumentIcon({ className }: { className?: string }) {
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
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" x2="8" y1="13" y2="13" />
      <line x1="16" x2="8" y1="17" y2="17" />
      <line x1="10" x2="8" y1="9" y2="9" />
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
      <line x1="10" x2="21" y1="14" y2="3" />
    </svg>
  );
}

export default ExpertAnswerDisplay;
